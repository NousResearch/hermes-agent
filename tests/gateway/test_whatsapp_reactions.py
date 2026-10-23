"""Tests for WhatsApp processing lifecycle reactions."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from plugins.platforms.whatsapp.adapter import WhatsAppAdapter


@pytest.fixture(autouse=True)
def _isolate_auth_env(monkeypatch):
    for var in ("WHATSAPP_ALLOWED_USERS", "WHATSAPP_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(var, raising=False)


def _authorize_all(*_args, **_kwargs):
    return True


class _AsyncCM:
    """Minimal async context manager returning a fixed value."""

    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


def _make_adapter(*, config_extra=None, running=True, http_session=None, authorization_check=_authorize_all,
                  dm_policy="pairing", allow_from=()):
    """Create a WhatsAppAdapter with test attributes (bypass __init__). By default the runner's auth check
    admits every sender; ``authorization_check=None`` exercises the bare-adapter access-policy fallback."""
    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    adapter._authorization_check = authorization_check
    adapter._dm_policy = dm_policy
    adapter._allow_from = set(allow_from)
    adapter.platform = Platform.WHATSAPP
    adapter.config = MagicMock()
    adapter.config.extra = config_extra or {}
    adapter._bridge_port = 3000
    adapter._running = running
    adapter._http_session = http_session if http_session is not None else MagicMock()
    adapter._bridge_process = None
    adapter._fatal_error_code = None
    adapter._fatal_error_message = None
    adapter._fatal_error_retryable = True
    adapter._fatal_error_handler = None
    adapter._background_tasks = set()
    adapter._reaction_tail = None
    return adapter


async def _drain(adapter):
    """Wait for the queued reaction sends (each waits for the one before it)."""
    if adapter._reaction_tail is not None:
        await asyncio.wait({adapter._reaction_tail})


def _event(*, chat_id="chat@s.whatsapp.net", message_id="msg-123", sender_id="sender@s.whatsapp.net", from_me=False,
           chat_type="dm"):
    from gateway.platforms.base import SessionSource

    source = SessionSource(
        platform=Platform.WHATSAPP,
        chat_id=chat_id,
        chat_name="Chat",
        chat_type=chat_type,
        user_id=sender_id,
        user_name="Sender",
    )
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        raw_message={"senderId": sender_id, "fromMe": from_me},
        message_id=message_id,
    )


class TestWhatsAppReactionConfig:
    def test_reactions_enabled_by_default(self):
        adapter = _make_adapter()

        assert adapter._reactions_enabled() is True

    @pytest.mark.parametrize("value", [False, "false", "0", "no", "off"])
    def test_reactions_can_be_disabled_by_config(self, value):
        adapter = _make_adapter(config_extra={"reactions": value})

        assert adapter._reactions_enabled() is False


class TestWhatsAppLifecycleReactions:
    @pytest.mark.asyncio
    async def test_processing_start_reacts_with_eyes(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"success": True})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.on_processing_start(_event())
        await _drain(adapter)

        adapter._http_session.post.assert_called_once()
        url = adapter._http_session.post.call_args.args[0]
        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert url == "http://127.0.0.1:3000/react"
        assert payload == {
            "chatId": "chat@s.whatsapp.net",
            "messageId": "msg-123",
            "emoji": "👀",
            "senderId": "sender@s.whatsapp.net",
            "fromMe": False,
        }

    @pytest.mark.asyncio
    async def test_processing_complete_reacts_with_success(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"success": True})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
        await _drain(adapter)

        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["emoji"] == "✅"

    @pytest.mark.asyncio
    async def test_processing_complete_reacts_with_failure(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"success": True})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.on_processing_complete(_event(), ProcessingOutcome.FAILURE)
        await _drain(adapter)

        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["emoji"] == "❌"

    @pytest.mark.asyncio
    async def test_cancelled_completion_clears_processing_reaction(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        resp.json = AsyncMock(return_value={"success": True})
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.on_processing_complete(_event(), ProcessingOutcome.CANCELLED)
        await _drain(adapter)

        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["emoji"] == ""

    @pytest.mark.asyncio
    async def test_disabled_reactions_are_noop(self):
        adapter = _make_adapter(config_extra={"reactions": False})
        adapter._http_session.post = MagicMock()

        await adapter.on_processing_start(_event())
        await adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS)
        await _drain(adapter)

        adapter._http_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_message_id_is_noop(self):
        adapter = _make_adapter()
        adapter._http_session.post = MagicMock()

        await adapter.on_processing_start(_event(message_id=None))
        await _drain(adapter)

        adapter._http_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_bare_phone_chat_id_is_normalized_for_bridge(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))

        await adapter.on_processing_start(_event(chat_id="1234567890", sender_id="1234567890"))
        await _drain(adapter)

        payload = adapter._http_session.post.call_args.kwargs["json"]
        assert payload["chatId"] == "1234567890@s.whatsapp.net"
        assert payload["senderId"] == "1234567890@s.whatsapp.net"


def _recording_session(adapter):
    resp = MagicMock(status=200)
    resp.json = AsyncMock(return_value={"success": True})
    adapter._http_session.post = MagicMock(return_value=_AsyncCM(resp))
    return adapter._http_session.post


def _sent_emojis(post):
    return [call.kwargs["json"]["emoji"] for call in post.call_args_list]


async def _run_lifecycle(adapter, outcome=ProcessingOutcome.SUCCESS, **event_kwargs):
    await adapter.on_processing_start(_event(**event_kwargs))
    await adapter.on_processing_complete(_event(**event_kwargs), outcome)
    await _drain(adapter)


class TestWhatsAppReactionSenderGate:
    """The hooks run before run.py's auth gate: an unauthorized sender must not see a reaction."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("outcome", [ProcessingOutcome.SUCCESS, ProcessingOutcome.FAILURE])
    async def test_unauthorized_dm_sender_gets_no_reaction(self, outcome):
        adapter = _make_adapter(authorization_check=lambda *_a, **_k: False)
        post = _recording_session(adapter)

        await _run_lifecycle(adapter, outcome)

        post.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("outcome, final", [(ProcessingOutcome.SUCCESS, "✅"), (ProcessingOutcome.FAILURE, "❌")])
    async def test_authorized_dm_sender_gets_start_and_outcome(self, outcome, final):
        adapter = _make_adapter()
        post = _recording_session(adapter)

        await _run_lifecycle(adapter, outcome)

        assert _sent_emojis(post) == ["👀", final]

    @pytest.mark.asyncio
    async def test_check_sees_the_event_sender_and_chat(self):
        seen = []
        adapter = _make_adapter(authorization_check=lambda *args, **_k: seen.append(args) or True)
        _recording_session(adapter)

        await adapter.on_processing_start(_event(chat_id="120363@g.us", sender_id="member@s.whatsapp.net", chat_type="group"))
        await _drain(adapter)

        assert seen == [("member@s.whatsapp.net", "group", "120363@g.us")]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("paired", [False, True])
    async def test_runner_auth_chain_decides_under_default_pairing(self, paired):
        """Real ``GatewayRunner._is_user_authorized``: an unpaired DM under ``dm_policy: pairing`` gets no 👀,
        a paired one does."""
        from gateway.run import GatewayRunner
        from gateway.session import SessionSource

        runner = object.__new__(GatewayRunner)
        runner.pairing_store = SimpleNamespace(is_approved=lambda *_a, **_k: paired)

        def check(user_id, chat_type=None, chat_id=None, **_kwargs):
            return runner._is_user_authorized(SessionSource(
                platform=Platform.WHATSAPP, chat_id=chat_id or "", chat_type=chat_type or "dm", user_id=user_id))

        adapter = _make_adapter(authorization_check=check, dm_policy="pairing")
        runner.adapters = {Platform.WHATSAPP: adapter}
        post = _recording_session(adapter)

        await _run_lifecycle(adapter)

        assert _sent_emojis(post) == (["👀", "✅"] if paired else [])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("dm_policy, allow_from, reacts", [
        ("pairing", (), False),
        ("allowlist", ("other@s.whatsapp.net",), False),
        ("allowlist", ("sender@s.whatsapp.net",), True),
        ("allowlist", ("*",), True),
    ])
    async def test_without_runner_check_falls_back_to_access_policy(self, dm_policy, allow_from, reacts):
        adapter = _make_adapter(authorization_check=None, dm_policy=dm_policy, allow_from=allow_from)
        post = _recording_session(adapter)

        await _run_lifecycle(adapter)

        assert _sent_emojis(post) == (["👀", "✅"] if reacts else [])

    @pytest.mark.asyncio
    async def test_open_policy_with_opt_in_keeps_reacting_without_runner_check(self, monkeypatch):
        monkeypatch.setenv("WHATSAPP_ALLOW_ALL_USERS", "true")
        adapter = _make_adapter(authorization_check=None, dm_policy="open")
        post = _recording_session(adapter)

        await _run_lifecycle(adapter)

        assert _sent_emojis(post) == ["👀", "✅"]

    @pytest.mark.asyncio
    async def test_event_without_sender_gets_no_reaction(self):
        adapter = _make_adapter()
        post = _recording_session(adapter)

        await _run_lifecycle(adapter, sender_id=None)

        post.assert_not_called()


class _BlockedCM:
    """Async context manager whose entry waits until ``release`` is set (a stalled bridge request)."""

    def __init__(self, release, value):
        self.release = release
        self.value = value

    async def __aenter__(self):
        await self.release.wait()
        return self.value

    async def __aexit__(self, *exc):
        return False


class TestWhatsAppReactionsInBackground:
    """A slow or failing reaction send must not hold up the turn, and 👀 still lands before ✅/❌."""

    @pytest.mark.asyncio
    async def test_hooks_return_while_the_bridge_is_stalled_and_keep_order(self):
        adapter = _make_adapter()
        release = asyncio.Event()
        resp = MagicMock(status=200)
        adapter._http_session.post = MagicMock(
            side_effect=lambda *_a, **_k: _BlockedCM(release, resp) if not release.is_set() else _AsyncCM(resp))

        await asyncio.wait_for(adapter.on_processing_start(_event()), timeout=1)
        await asyncio.wait_for(adapter.on_processing_complete(_event(), ProcessingOutcome.SUCCESS), timeout=1)
        for _ in range(5):
            await asyncio.sleep(0)

        # 👀 is in flight and stalled; ✅ waits behind it instead of overtaking it.
        assert _sent_emojis(adapter._http_session.post) == ["👀"]
        release.set()
        await _drain(adapter)
        assert _sent_emojis(adapter._http_session.post) == ["👀", "✅"]

    @pytest.mark.asyncio
    async def test_failed_reaction_send_is_contained_and_the_next_one_still_goes(self):
        adapter = _make_adapter()
        resp = MagicMock(status=200)
        adapter._http_session.post = MagicMock(side_effect=[OSError("bridge down"), _AsyncCM(resp)])

        await _run_lifecycle(adapter, ProcessingOutcome.FAILURE)

        assert _sent_emojis(adapter._http_session.post) == ["👀", "❌"]
        assert adapter._background_tasks == set()


class TestWhatsAppInboundReactionEvents:
    @pytest.mark.asyncio
    async def test_inbound_reaction_event_gets_no_reaction(self):
        """bridge_helpers.js forwards a reaction as a TEXT "[Reaction: …]" event with nativeType reactionMessage."""
        adapter = _make_adapter()
        post = _recording_session(adapter)
        event = _event(message_id="reaction-msg-1")
        event.raw_message.update({"nativeType": "reactionMessage", "mediaType": "reaction"})

        await adapter.on_processing_start(event)
        await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
        await _drain(adapter)

        post.assert_not_called()
