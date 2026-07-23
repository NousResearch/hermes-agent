"""Tests for Slack Block Kit interactive clarify buttons.

Mirrors test_slack_approval_buttons.py (harness) and
test_telegram_clarify_buttons.py (semantics) for the ``send_clarify`` override
and the indexed ``hermes_clarify_choice_<idx>`` /
``hermes_clarify_other`` action dispatch.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)


# ---------------------------------------------------------------------------
# Minimal Slack SDK mock so SlackAdapter can be imported (mirrors
# test_slack_approval_buttons.py)
# ---------------------------------------------------------------------------
def _ensure_slack_mock():
    if "slack_bolt" in sys.modules:
        return
    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    sys.modules["slack_bolt"] = slack_bolt
    sys.modules["slack_bolt.async_app"] = slack_bolt.async_app
    handler_mod = MagicMock()
    handler_mod.AsyncSocketModeHandler = MagicMock
    sys.modules["slack_bolt.adapter"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode"] = MagicMock()
    sys.modules["slack_bolt.adapter.socket_mode.async_handler"] = handler_mod
    sdk_mod = MagicMock()
    sdk_mod.web = MagicMock()
    sdk_mod.web.async_client = MagicMock()
    sdk_mod.web.async_client.AsyncWebClient = MagicMock
    sys.modules["slack_sdk"] = sdk_mod
    sys.modules["slack_sdk.web"] = sdk_mod.web
    sys.modules["slack_sdk.web.async_client"] = sdk_mod.web.async_client


_ensure_slack_mock()

from plugins.platforms.slack.adapter import SlackAdapter
from gateway.config import PlatformConfig


def _make_adapter():
    config = PlatformConfig(enabled=True, token="xoxb-test-token")
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._bot_user_id = "U_BOT"
    adapter._team_clients = {"T1": AsyncMock()}
    adapter._team_bot_user_ids = {"T1": "U_BOT"}
    adapter._channel_team = {"C1": "T1"}
    return adapter


class _AuthRunner:
    def __init__(self, auth_fn=None):
        self._auth_fn = auth_fn or (lambda _source: True)

    async def handle(self, event):
        return None

    def _is_user_authorized(self, source):
        return self._auth_fn(source)


def _attach_auth_runner(adapter, auth_fn=None):
    adapter.set_message_handler(_AuthRunner(auth_fn=auth_fn).handle)


def _clear_clarify_state():
    from tools import clarify_gateway as cm
    with cm._lock:
        cm._entries.clear()
        cm._session_index.clear()
        cm._notify_cbs.clear()


# ===========================================================================
# send_clarify — Block Kit render (a)
# ===========================================================================

class TestSlackSendClarify:
    def setup_method(self):
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_multi_choice_renders_buttons_and_other(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "1234.5678"})

        result = await adapter.send_clarify(
            chat_id="C1",
            question="Which environment?",
            choices=["staging", "production"],
            clarify_id="cid1",
            session_key="sk1",
        )

        assert result.success is True
        assert result.message_id == "1234.5678"
        # ts recorded for the double-click guard
        assert adapter._clarify_resolved.get("1234.5678") is False

        kwargs = mock_client.chat_postMessage.call_args[1]
        blocks = kwargs["blocks"]
        assert blocks[0]["type"] == "section"
        assert "Which environment?" in blocks[0]["text"]["text"]
        assert blocks[1]["type"] == "actions"
        elements = blocks[1]["elements"]
        # 2 choices + Other
        assert len(elements) == 3
        assert elements[0]["action_id"] == "hermes_clarify_choice_0"
        assert elements[0]["value"] == "cid1|0"
        assert elements[1]["action_id"] == "hermes_clarify_choice_1"
        assert elements[1]["value"] == "cid1|1"
        assert elements[0]["text"]["text"] == "staging"
        # Final button is the free-text "Other"
        assert elements[2]["action_id"] == "hermes_clarify_other"
        assert elements[2]["value"] == "cid1|other"
        for block in blocks:
            if block["type"] == "actions":
                action_ids = [element["action_id"] for element in block["elements"]]
                assert len(action_ids) == len(set(action_ids))

    @pytest.mark.asyncio
    async def test_open_ended_no_buttons(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "9.9"})

        result = await adapter.send_clarify(
            chat_id="C1",
            question="What should I name the branch?",
            choices=None,
            clarify_id="cid-open",
            session_key="sk-open",
        )

        assert result.success is True
        kwargs = mock_client.chat_postMessage.call_args[1]
        # Open-ended delegates to the base plain-text path — no action blocks.
        assert "blocks" not in kwargs or all(
            b.get("type") != "actions" for b in (kwargs.get("blocks") or [])
        )
        assert "What should I name the branch?" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_mrkdwn_escapes_question(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "1.1"})

        await adapter.send_clarify(
            chat_id="C1",
            question="Use <A> & <B>?",
            choices=["yes"],
            clarify_id="cid2",
            session_key="sk2",
        )
        section_text = mock_client.chat_postMessage.call_args[1]["blocks"][0]["text"]["text"]
        assert "<A>" not in section_text
        assert "&lt;A&gt;" in section_text
        assert "&amp;" in section_text

    @pytest.mark.asyncio
    async def test_sends_in_thread(self):
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "1.2"})

        await adapter.send_clarify(
            chat_id="C1",
            question="?",
            choices=["a"],
            clarify_id="cid3",
            session_key="sk3",
            metadata={"thread_id": "8888.0000"},
        )
        assert mock_client.chat_postMessage.call_args[1].get("thread_ts") == "8888.0000"

    @pytest.mark.asyncio
    async def test_not_connected(self):
        adapter = _make_adapter()
        adapter._app = None
        result = await adapter.send_clarify(
            chat_id="C1", question="?", choices=["a"], clarify_id="c", session_key="s"
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_five_choices_chunk_across_actions_blocks(self):
        """Slack caps 5 elements per actions block; 5 choices + Other = 6
        buttons must spill into a second block instead of 400ing."""
        adapter = _make_adapter()
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_postMessage = AsyncMock(return_value={"ts": "1.3"})

        await adapter.send_clarify(
            chat_id="C1",
            question="?",
            choices=["a", "b", "c", "d", "e"],
            clarify_id="cid5",
            session_key="sk5",
        )
        blocks = mock_client.chat_postMessage.call_args[1]["blocks"]
        action_blocks = [b for b in blocks if b["type"] == "actions"]
        assert len(action_blocks) == 2
        for b in action_blocks:
            assert len(b["elements"]) <= 5


# ===========================================================================
# _handle_clarify_action — choice click resolves (b)
# ===========================================================================

class TestSlackClarifyChoiceAction:
    def setup_method(self):
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_choice_resolves_with_choice_text(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        cm.register("cidA", "sk-cb", "Pick", ["red", "green", "blue"])
        adapter._clarify_resolved["1234.5678"] = False

        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()

        ack = AsyncMock()
        body = {
            "message": {
                "ts": "1234.5678",
                "blocks": [
                    {"type": "section", "text": {"type": "mrkdwn", "text": "❓ Pick"}},
                    {"type": "actions", "elements": []},
                ],
            },
            "channel": {"id": "C1"},
            "user": {"name": "norbert", "id": "U_NORBERT"},
        }
        action = {"action_id": "hermes_clarify_choice_1", "value": "cidA|1"}

        await adapter._handle_clarify_action(ack, body, action)

        ack.assert_called_once()
        with cm._lock:
            entry = cm._entries.get("cidA")
        assert entry is not None
        assert entry.response == "green"
        assert entry.event.is_set()
        # Message updated with the answer, buttons dropped.
        update_kwargs = mock_client.chat_update.call_args[1]
        assert "green" in update_kwargs["text"]
        assert all(b["type"] != "actions" for b in update_kwargs["blocks"])

    @pytest.mark.asyncio
    async def test_prevents_double_click(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        cm.register("cidDup", "sk-dup", "Pick", ["x"])
        adapter._clarify_resolved["1.1"] = True  # already resolved

        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()

        ack = AsyncMock()
        body = {
            "message": {"ts": "1.1", "blocks": []},
            "channel": {"id": "C1"},
            "user": {"name": "n", "id": "U1"},
        }
        action = {"action_id": "hermes_clarify_choice", "value": "cidDup|0"}

        await adapter._handle_clarify_action(ack, body, action)

        ack.assert_called_once()
        with cm._lock:
            entry = cm._entries.get("cidDup")
        assert entry is not None
        assert not entry.event.is_set()
        mock_client.chat_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_unauthorized_click_ignored(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        _attach_auth_runner(adapter, auth_fn=lambda _s: False)
        cm.register("cidAuth", "sk-auth", "Pick", ["a", "b"])
        adapter._clarify_resolved["2.2"] = False

        ack = AsyncMock()
        body = {
            "message": {"ts": "2.2", "blocks": []},
            "channel": {"id": "C1"},
            "user": {"name": "mallory", "id": "U_BAD"},
        }
        action = {"action_id": "hermes_clarify_choice", "value": "cidAuth|0"}

        await adapter._handle_clarify_action(ack, body, action)

        with cm._lock:
            entry = cm._entries.get("cidAuth")
        assert entry is not None
        assert not entry.event.is_set()

    @pytest.mark.asyncio
    async def test_expired_choice_shows_notice(self):
        """Late tap after the entry was evicted must surface expiry, not a ✓."""
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        # No entry registered → resolve returns False.
        adapter._clarify_resolved["3.3"] = False

        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()

        ack = AsyncMock()
        body = {
            "message": {"ts": "3.3", "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": "❓ Pick"}},
            ]},
            "channel": {"id": "C1"},
            "user": {"name": "t", "id": "U_T"},
        }
        action = {"action_id": "hermes_clarify_choice", "value": "cidGone|0"}

        await adapter._handle_clarify_action(ack, body, action)

        assert "expired" in mock_client.chat_update.call_args[1]["text"].lower()


# ===========================================================================
# _handle_clarify_action — "Other" → text-capture → typed reply (c)
# ===========================================================================

class TestSlackClarifyOtherFlow:
    def setup_method(self):
        _clear_clarify_state()

    @pytest.mark.asyncio
    async def test_other_flips_to_text_mode_then_typed_reply_resolves(self):
        from tools import clarify_gateway as cm

        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        cm.register("cidO", "sk-other", "Pick", ["x", "y"])
        adapter._clarify_resolved["4.4"] = False

        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()

        ack = AsyncMock()
        body = {
            "message": {"ts": "4.4", "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": "❓ Pick"}},
                {"type": "actions", "elements": []},
            ]},
            "channel": {"id": "C1"},
            "user": {"name": "norbert", "id": "U_N"},
        }
        action = {"action_id": "hermes_clarify_other", "value": "cidO|other"}

        await adapter._handle_clarify_action(ack, body, action)

        # Entry flipped to text-capture; NOT yet resolved.
        pending = cm.get_pending_for_session("sk-other")
        assert pending is not None and pending.clarify_id == "cidO"
        assert pending.awaiting_text is True
        with cm._lock:
            entry = cm._entries.get("cidO")
        assert not entry.event.is_set()
        assert "awaiting" in mock_client.chat_update.call_args[1]["text"].lower()

        # Now the gateway text-intercept (platform-agnostic) resolves from the
        # user's next typed message. We exercise that leveraged path directly.
        assert cm.resolve_text_response_for_session("sk-other", "my custom answer") is True
        with cm._lock:
            entry = cm._entries.get("cidO")
        assert entry.response == "my custom answer"
        assert entry.event.is_set()

    @pytest.mark.asyncio
    async def test_other_expired_shows_notice(self):
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        # No entry → mark_awaiting_text returns False.
        adapter._clarify_resolved["5.5"] = False

        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()

        ack = AsyncMock()
        body = {
            "message": {"ts": "5.5", "blocks": [
                {"type": "section", "text": {"type": "mrkdwn", "text": "❓ Pick"}},
            ]},
            "channel": {"id": "C1"},
            "user": {"name": "t", "id": "U_T"},
        }
        action = {"action_id": "hermes_clarify_other", "value": "cidOtherGone|other"}

        await adapter._handle_clarify_action(ack, body, action)
        assert "expired" in mock_client.chat_update.call_args[1]["text"].lower()

    @pytest.mark.asyncio
    async def test_malformed_value_ignored(self):
        adapter = _make_adapter()
        _attach_auth_runner(adapter)
        adapter._clarify_resolved["6.6"] = False
        mock_client = adapter._team_clients["T1"]
        mock_client.chat_update = AsyncMock()

        ack = AsyncMock()
        body = {
            "message": {"ts": "6.6", "blocks": []},
            "channel": {"id": "C1"},
            "user": {"name": "t", "id": "U_T"},
        }
        action = {"action_id": "hermes_clarify_choice", "value": "no-delimiter"}

        await adapter._handle_clarify_action(ack, body, action)
        mock_client.chat_update.assert_not_called()


# ===========================================================================
# Base text-fallback unchanged for platforms without an override (e)
# ===========================================================================

class TestBaseAdapterClarifyFallbackUnchanged:
    @pytest.mark.asyncio
    async def test_base_numbered_text_fallback(self):
        from gateway.platforms.base import BasePlatformAdapter, SendResult

        class _Stub(BasePlatformAdapter):
            name = "stub"

            def __init__(self):
                self.sent: list = []

            async def connect(self, *, is_reconnect: bool = False): pass
            async def disconnect(self): pass
            async def send(self, chat_id, content, **kw):
                self.sent.append(content)
                return SendResult(success=True, message_id="1")
            async def edit(self, *a, **k): return SendResult(success=False)
            async def get_history(self, *a, **k): return []
            async def get_chat_info(self, *a, **k): return {}

        adapter = _Stub()
        result = await adapter.send_clarify(
            chat_id="c", question="Pick a fruit",
            choices=["apple", "banana"], clarify_id="x", session_key="s",
        )
        assert result.success is True
        text = adapter.sent[0]
        assert "Pick a fruit" in text
        assert "1." in text and "apple" in text
        assert "2." in text and "banana" in text
