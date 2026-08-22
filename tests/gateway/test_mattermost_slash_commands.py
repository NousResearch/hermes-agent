"""Native Mattermost slash-command HTTP endpoint.

Mattermost's client swallows any message starting with ``/`` as an
unregistered slash command, so gateway commands typed in Mattermost never
reach the WebSocket. The adapter's slash-command listener receives the
HTTP callbacks of *registered* server-side slash commands and injects them
into the gateway message pipeline as native COMMAND events — landing in
the SAME session a regular channel message would.

Covered here:
1. Valid token + payload → 200 ephemeral ack + a COMMAND MessageEvent with
   the right platform/chat_id/user_id/text, keyed to the same session the
   WebSocket path would produce.
2. Bad token → 401, no event injected.
3. No MATTERMOST_SLASH_TOKENS → listener never binds (fail closed).
4. allowed_channels configured + disallowed channel → 403, no event.
5. Wrong method / path / content-type → 404.
6. connect → disconnect → connect rebinds the same port (no leak).
7. The HTTP ack returns without waiting for the agent pipeline.
8. Confirmed-missing thread roots (HTTP 400/404 on GET posts/{id}) resolve
   to "" so replies post flat on the first try — no invalid root_id POST
   and no "⚠️ thread delivery failed" banner (the trigger_id path).
9. send_typing() includes parent_id (the thread root) only for thread
   sessions, so the typing indicator renders in thread views too;
   channel sessions keep the exact {"channel_id": ...} payload.
"""

import asyncio
import json
import socket
import time
from urllib.parse import urlencode

import aiohttp
import pytest
from unittest.mock import AsyncMock

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageType
from gateway.session import build_session_key

BOT_USER_ID = "bot11111111111111111111111111"
BOT_USERNAME = "hermesbot"
CHANNEL_ID = "ch22222222222222222222222222"
USER_ID = "u3333333333333333333333333333"
USER_NAME = "alice"
POST_ID = "post55555555555555555555555555"
ROOT_ID = "root66666666666666666666666666"
TRIGGER_ID = "trig7777777777777777777777777"
NEW_POST_ID = "newpost8888888888888888888888888"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _slash_payload(command="/sethome", text="", token="tokA", channel_id=CHANNEL_ID):
    return {
        "token": token,
        "team_id": "team9999999999999999999999999",
        "team_domain": "example",
        "channel_id": channel_id,
        "channel_name": "town-square",
        "user_id": USER_ID,
        "user_name": USER_NAME,
        "command": command,
        "text": text,
        "response_url": "https://mm.example.com/hooks/resp",
        "trigger_id": "trig7777777777777777777777777",
    }


class SlashAdapterHarness:
    """A MattermostAdapter with the Mattermost REST/WS side stubbed out.

    ``connect()`` runs for real (real HTTP listener, real session
    lifecycle); ``_api_get`` is stubbed for ``users/me`` + channel lookups
    and ``_ws_loop`` is a no-op so no outbound traffic happens. The message
    handler records every MessageEvent that reaches the gateway pipeline.
    """

    def __init__(self, monkeypatch, tokens="tokA,tokB", extra=None):
        from plugins.platforms.mattermost.adapter import MattermostAdapter

        self.port = _free_port()
        cfg_extra = {
            "url": "https://mm.example.com",
            "slash_command_host": "127.0.0.1",
            "slash_command_port": self.port,
        }
        cfg_extra.update(extra or {})
        config = PlatformConfig(enabled=True, token="test-token", extra=cfg_extra)
        # Deterministic reply mode: ambient .env may leak MATTERMOST_REPLY_MODE
        # via `import gateway.run` in other test modules; the adapter reads it
        # at __init__. Thread-mode behavior has its own dedicated test below.
        monkeypatch.setenv("MATTERMOST_REPLY_MODE", "off")
        self.adapter = MattermostAdapter(config)

        if tokens is None:
            monkeypatch.delenv("MATTERMOST_SLASH_TOKENS", raising=False)
        else:
            monkeypatch.setenv("MATTERMOST_SLASH_TOKENS", tokens)

        async def fake_api_get(path):
            if path == "users/me":
                return {"id": BOT_USER_ID, "username": BOT_USERNAME}
            if path.startswith("channels/"):
                return {
                    "id": path.split("/", 1)[1],
                    "type": "O",
                    "display_name": "Town Square",
                    "name": "town-square",
                }
            return {}

        self.adapter._api_get = fake_api_get
        self.adapter._ws_loop = AsyncMock()
        self.adapter.send_typing = AsyncMock()

        self.events = []
        self.event_arrived = asyncio.Event()

        async def handler(event):
            self.events.append(event)
            self.event_arrived.set()
            return None

        self.adapter.set_message_handler(handler)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def connect(self) -> bool:
        return await self.adapter.connect()

    async def wait_for_event(self, timeout=5.0):
        await asyncio.wait_for(self.event_arrived.wait(), timeout)
        return self.events[-1]


@pytest.fixture
def harness(monkeypatch):
    return SlashAdapterHarness(monkeypatch)


async def _post(harness, payload=None, *, path="/", headers=None, skip_ct=False, method="POST"):
    body = urlencode(payload if payload is not None else _slash_payload())
    kwargs = {"data": body.encode()}
    if skip_ct:
        kwargs["skip_auto_headers"] = ("Content-Type",)
    else:
        kwargs["headers"] = {"Content-Type": "application/x-www-form-urlencoded"}
    if headers:
        kwargs["headers"] = {**kwargs.get("headers", {}), **headers}
    async with aiohttp.ClientSession() as session:
        requester = getattr(session, method.lower())
        async with requester(f"{harness.url}{path}", **kwargs) as resp:
            return resp.status, await resp.json()


class TestSlashCommandEndpoint:
    @pytest.mark.asyncio
    async def test_valid_token_injects_command_event(self, harness):
        """Valid token → 200 ephemeral ack + COMMAND event in the pipeline."""
        assert await harness.connect() is True

        status, body = await _post(harness, _slash_payload(command="/sethome"))
        assert status == 200
        assert body["response_type"] == "ephemeral"
        assert "/sethome" in body["text"]

        event = await harness.wait_for_event()
        assert event.message_type is MessageType.COMMAND
        assert event.text == "/sethome"
        assert event.source.platform is Platform.MATTERMOST
        assert event.source.chat_id == CHANNEL_ID
        assert event.source.user_id == USER_ID

        await harness.adapter.disconnect()

    @pytest.mark.asyncio
    async def test_command_with_args_reconstructs_text(self, harness):
        """``command`` + ``text`` → "/status mattermost"."""
        assert await harness.connect() is True
        status, _ = await _post(harness, _slash_payload(command="/status", text="mattermost"))
        assert status == 200
        event = await harness.wait_for_event()
        assert event.text == "/status mattermost"
        await harness.adapter.disconnect()

    @pytest.mark.asyncio
    async def test_session_key_matches_ws_path(self, harness, monkeypatch):
        """The injected event keys into the SAME session as the WS path.

        Drives the real ``_handle_ws_event`` with an equivalent channel
        message (leading-space workaround: "␣/sethome") and compares the
        session keys both paths produce. require_mention is disabled so the
        WS message passes channel gating (the slash path bypasses it by
        design — explicit intent).

        Parity is asserted under the default ``reply_mode=off``; thread mode
        keys slash commands to the stable per-user channel session instead
        (see test_thread_mode_slash_keys_to_stable_channel_session).
        """
        monkeypatch.setenv("MATTERMOST_REQUIRE_MENTION", "false")
        monkeypatch.setenv("MATTERMOST_REPLY_MODE", "off")  # parity holds in default mode
        assert await harness.connect() is True

        captured = []
        captured_evt = asyncio.Event()

        async def capture_handle_message(event):
            captured.append(event)
            captured_evt.set()

        monkeypatch.setattr(harness.adapter, "handle_message", capture_handle_message)

        ws_event = {
            "event": "posted",
            "data": {
                "channel_type": "O",
                "sender_name": f"@{USER_NAME}",
                "post": json.dumps({
                    "id": "post55555555555555555555555555",
                    "user_id": USER_ID,
                    "channel_id": CHANNEL_ID,
                    "message": " /sethome",
                    "root_id": "",
                    "file_ids": [],
                }),
            },
        }
        await harness.adapter._handle_ws_event(ws_event)
        assert len(captured) == 1  # WS path captured (handle_message patched)

        status, _ = await _post(harness, _slash_payload(command="/sethome"))
        assert status == 200
        await asyncio.wait_for(captured_evt.wait(), 5.0)
        assert len(captured) == 2  # slash path captured too

        ws_path_event = captured[0]
        slash_event = captured[1]

        ws_key = build_session_key(ws_path_event.source)
        slash_key = build_session_key(slash_event.source)
        assert slash_key == ws_key
        assert slash_event.source.chat_type == ws_path_event.source.chat_type
        assert slash_event.source.user_name == ws_path_event.source.user_name

        await harness.adapter.disconnect()

    @pytest.mark.asyncio
    async def test_thread_mode_slash_keys_to_stable_channel_session(self, monkeypatch):
        """In reply_mode=thread a slash command keys to the stable per-user
        channel session, NOT a per-invocation thread.

        The WS path in thread mode treats each top-level post as its own
        thread (thread_id = post_id) so replies can thread onto it. A slash
        invocation creates no post, and trigger_id is not a post id — using
        it as a thread root would give the bot's reply an invalid root_id.
        So the slash path intentionally keys to ``…:chat:user_id`` in every
        mode (and to the DM session for DMs).
        """
        monkeypatch.setenv("MATTERMOST_SLASH_TOKENS", "tokA,tokB")
        harness = SlashAdapterHarness(monkeypatch, extra={"reply_mode": "thread"})
        assert await harness.connect() is True

        status, _ = await _post(harness, _slash_payload(command="/sethome"))
        assert status == 200
        event = await harness.wait_for_event()
        assert event.source.thread_id is None
        key = build_session_key(event.source)
        assert key == (
            f"agent:main:mattermost:channel:{CHANNEL_ID}:{USER_ID}"
        )
        await harness.adapter.disconnect()

    @pytest.mark.asyncio
    async def test_bad_token_returns_401_no_event(self, harness):
        assert await harness.connect() is True
        status, body = await _post(harness, _slash_payload(token="wrong-token"))
        assert status == 401
        assert body == {"error": "unauthorized"}
        await asyncio.sleep(0.15)  # no task is ever scheduled on 401
        assert harness.events == []
        await harness.adapter.disconnect()

    @pytest.mark.asyncio
    async def test_missing_token_returns_401_no_event(self, harness):
        assert await harness.connect() is True
        payload = _slash_payload()
        payload.pop("token")
        status, _ = await _post(harness, payload)
        assert status == 401
        await asyncio.sleep(0.15)
        assert harness.events == []
        await harness.adapter.disconnect()

    @pytest.mark.asyncio
    async def test_listener_disabled_without_tokens(self, monkeypatch):
        """No MATTERMOST_SLASH_TOKENS → connect still works, nothing binds."""
        h = SlashAdapterHarness(monkeypatch, tokens=None)
        assert await h.connect() is True
        assert h.adapter._slash_runner is None

        # Nothing is listening on the configured port.
        with pytest.raises(aiohttp.ClientConnectorError):
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    h.url, data=b"token=tokA", timeout=aiohttp.ClientTimeout(total=2)
                ):
                    pass
        await h.adapter.disconnect()

    @pytest.mark.asyncio
    async def test_allowed_channels_rejects_disallowed(self, harness, monkeypatch):
        monkeypatch.setenv("MATTERMOST_ALLOWED_CHANNELS", "ch_allowed99999999999999999")
        assert await harness.connect() is True

        status, body = await _post(harness, _slash_payload(channel_id="ch_other8888888888888888"))
        assert status == 403
        assert body == {"error": "forbidden"}
        await asyncio.sleep(0.15)
        assert harness.events == []

        # An allowed channel still goes through.
        status, _ = await _post(harness, _slash_payload(channel_id="ch_allowed99999999999999999"))
        assert status == 200
        event = await harness.wait_for_event()
        assert event.source.chat_id == "ch_allowed99999999999999999"
        await harness.adapter.disconnect()

    @pytest.mark.asyncio
    async def test_wrong_method_path_or_content_type_404(self, harness):
        assert await harness.connect() is True

        async with aiohttp.ClientSession() as session:
            async with session.get(harness.url) as resp:
                assert resp.status == 404
            async with session.post(
                f"{harness.url}/nope", data=b"token=tokA"
            ) as resp:
                assert resp.status == 404
            async with session.post(
                f"{harness.url}/",
                data=json.dumps(_slash_payload()),
                headers={"Content-Type": "application/json"},
            ) as resp:
                assert resp.status == 404

        await asyncio.sleep(0.15)
        assert harness.events == []
        await harness.adapter.disconnect()

    @pytest.mark.asyncio
    async def test_missing_content_type_tolerated(self, harness):
        """Mattermost always sends form content-type, but we tolerate none."""
        assert await harness.connect() is True
        status, _ = await _post(harness, skip_ct=True)
        assert status == 200
        event = await harness.wait_for_event()
        assert event.text == "/sethome"
        await harness.adapter.disconnect()

    @pytest.mark.asyncio
    async def test_connect_disconnect_connect_no_port_leak(self, harness):
        """Repeated lifecycle on the same port must not leak the bind."""
        assert await harness.connect() is True
        status, _ = await _post(harness, _slash_payload(command="/first"))
        assert status == 200
        await harness.wait_for_event()

        await harness.adapter.disconnect()
        assert harness.adapter._slash_runner is None

        # Reconnect on the SAME port (reconnect cycles happen in practice).
        assert await harness.connect() is True
        assert harness.adapter._slash_runner is not None
        status, _ = await _post(harness, _slash_payload(command="/second"))
        assert status == 200
        event = await harness.wait_for_event()
        assert event.text == "/second"
        await harness.adapter.disconnect()

    @pytest.mark.asyncio
    async def test_ack_does_not_wait_for_agent_pipeline(self, monkeypatch):
        """Mattermost enforces ~5s — the ack must not block on the agent."""
        h = SlashAdapterHarness(monkeypatch)
        pipeline_done = asyncio.Event()

        async def slow_handler(event):
            await asyncio.sleep(2.0)  # simulated agent turn
            pipeline_done.set()
            return None

        h.adapter.set_message_handler(slow_handler)
        assert await h.connect() is True

        started = time.monotonic()
        status, body = await _post(h, _slash_payload(command="/slowcmd"))
        elapsed = time.monotonic() - started

        assert status == 200
        assert body["response_type"] == "ephemeral"
        # The pipeline sleeps 2s; the ack must return far earlier.
        assert elapsed < 1.0

        # ...but the injection does eventually complete.
        await asyncio.wait_for(pipeline_done.wait(), 5.0)
        await h.adapter.disconnect()


class TestThreadRootConfirmedMissing:
    """_resolve_root_id / send() when the thread root is CONFIRMED missing.

    In reply_mode=thread, slash-command replies carry reply_to=trigger_id
    (not a post id), and WS replies can target since-deleted roots — the
    GET posts/{id} lookup comes back 400/404. A confirmed-missing root must
    resolve to "" so send() posts flat on the FIRST attempt: no invalid
    root_id POST, no "⚠️ Mattermost thread delivery failed" banner. A
    transient/unknown GET failure (e.g. 500) keeps the legacy post_id
    passthrough so _post_preserving_thread remains the safety net for
    genuine races.
    """

    def _make_adapter(self, monkeypatch, extra=None):
        from plugins.platforms.mattermost.adapter import MattermostAdapter

        cfg_extra = {"url": "https://mm.example.com"}
        cfg_extra.update(extra or {})
        # Deterministic reply mode even here: ambient .env may leak
        # MATTERMOST_REPLY_MODE; extra={"reply_mode": ...} wins over env.
        monkeypatch.setenv("MATTERMOST_REPLY_MODE", "off")
        return MattermostAdapter(
            PlatformConfig(enabled=True, token="test-token", extra=cfg_extra)
        )

    def _stub_get(self, adapter, status, data):
        """Stub _api_get the way the real one behaves: record the HTTP
        status on adapter._last_get_status, return {} on failure."""

        async def fake_api_get(path):
            adapter._last_get_status = status
            return data

        adapter._api_get = fake_api_get

    @pytest.mark.asyncio
    async def test_resolve_root_id_404_returns_empty(self, monkeypatch):
        """GET posts/{id} → 404 → "" (post flat; trigger_id path)."""
        adapter = self._make_adapter(monkeypatch)
        self._stub_get(adapter, 404, {})
        assert await adapter._resolve_root_id(TRIGGER_ID) == ""

    @pytest.mark.asyncio
    async def test_resolve_root_id_400_returns_empty(self, monkeypatch):
        adapter = self._make_adapter(monkeypatch)
        self._stub_get(adapter, 400, {})
        assert await adapter._resolve_root_id(TRIGGER_ID) == ""

    @pytest.mark.asyncio
    async def test_resolve_root_id_existing_reply_returns_root(self, monkeypatch):
        """Post exists and is a thread reply → its root_id."""
        adapter = self._make_adapter(monkeypatch)
        self._stub_get(adapter, 200, {"id": POST_ID, "root_id": ROOT_ID})
        assert await adapter._resolve_root_id(POST_ID) == ROOT_ID

    @pytest.mark.asyncio
    async def test_resolve_root_id_top_level_post_returns_itself(self, monkeypatch):
        """Post exists and is a top-level post → the post_id itself."""
        adapter = self._make_adapter(monkeypatch)
        self._stub_get(adapter, 200, {"id": POST_ID, "root_id": ""})
        assert await adapter._resolve_root_id(POST_ID) == POST_ID

    @pytest.mark.asyncio
    async def test_resolve_root_id_500_keeps_legacy_passthrough(self, monkeypatch):
        """Unknown/transient failure → post_id unchanged (no flat guess)."""
        adapter = self._make_adapter(monkeypatch)
        self._stub_get(adapter, 500, {})
        assert await adapter._resolve_root_id(POST_ID) == POST_ID

    @pytest.mark.asyncio
    async def test_resolve_root_id_network_error_keeps_legacy_passthrough(self, monkeypatch):
        """Network error never sets a status (None) → legacy passthrough."""
        adapter = self._make_adapter(monkeypatch)
        self._stub_get(adapter, None, {})
        assert await adapter._resolve_root_id(POST_ID) == POST_ID

    @pytest.mark.asyncio
    async def test_send_with_confirmed_missing_root_posts_flat_once(self, monkeypatch):
        """Regression guard for the trigger_id path: reply_to that is not a
        post id → ONE POST call, no root_id key, no ⚠️ banner."""
        adapter = self._make_adapter(monkeypatch, extra={"reply_mode": "thread"})
        self._stub_get(adapter, 404, {})

        adapter._api_post = AsyncMock(return_value={"id": NEW_POST_ID})

        result = await adapter.send(CHANNEL_ID, "final reply", reply_to=TRIGGER_ID)
        assert result.success is True
        assert result.message_id == NEW_POST_ID

        adapter._api_post.assert_called_once()
        path, payload = adapter._api_post.call_args.args
        assert path == "posts"
        assert "root_id" not in payload
        assert "final reply" in payload["message"]
        assert "⚠️" not in payload["message"]


class TestSendTypingThreadScoped:
    """send_typing() must scope the typing indicator to the active thread.

    Mattermost's PublishUserTyping copies the request's ``parent_id`` into
    the typing WebSocket event, and the webapp only renders the indicator
    where ``rootId === parent_id`` — without it, thread views never show
    "hermes is typing". Thread sessions carry the root post id in
    ``metadata["thread_id"]``; channel/slash sessions have none, and their
    wire payload must stay exactly ``{"channel_id": ...}``.
    """

    def _make_adapter(self, monkeypatch):
        from plugins.platforms.mattermost.adapter import MattermostAdapter

        monkeypatch.setenv("MATTERMOST_REPLY_MODE", "off")
        adapter = MattermostAdapter(
            PlatformConfig(enabled=True, token="test-token", extra={"url": "https://mm.example.com"})
        )
        adapter._bot_user_id = BOT_USER_ID
        return adapter

    @pytest.mark.asyncio
    async def test_thread_metadata_sends_parent_id(self, monkeypatch):
        """metadata["thread_id"] → POST body carries parent_id=root post id."""
        adapter = self._make_adapter(monkeypatch)
        adapter._api_post = AsyncMock(return_value={})

        await adapter.send_typing(CHANNEL_ID, metadata={"thread_id": ROOT_ID})

        path, payload = adapter._api_post.call_args.args
        assert path == f"users/{BOT_USER_ID}/typing"
        assert payload == {"channel_id": CHANNEL_ID, "parent_id": ROOT_ID}

    @pytest.mark.asyncio
    async def test_no_metadata_keeps_channel_scoped_payload(self, monkeypatch):
        """metadata=None → channel typing unchanged (regression guard)."""
        adapter = self._make_adapter(monkeypatch)
        adapter._api_post = AsyncMock(return_value={})

        await adapter.send_typing(CHANNEL_ID, metadata=None)

        path, payload = adapter._api_post.call_args.args
        assert path == f"users/{BOT_USER_ID}/typing"
        assert payload == {"channel_id": CHANNEL_ID}
        assert "parent_id" not in payload

    @pytest.mark.asyncio
    async def test_empty_thread_id_keeps_channel_scoped_payload(self, monkeypatch):
        """metadata={"thread_id": ""} → no parent_id key."""
        adapter = self._make_adapter(monkeypatch)
        adapter._api_post = AsyncMock(return_value={})

        await adapter.send_typing(CHANNEL_ID, metadata={"thread_id": ""})

        path, payload = adapter._api_post.call_args.args
        assert path == f"users/{BOT_USER_ID}/typing"
        assert payload == {"channel_id": CHANNEL_ID}
        assert "parent_id" not in payload
