"""Tests for Nextcloud Talk processing-indicator lifecycle.

Covers:
  1. Fast handler: indicator task is cancelled before the delay fires — no send, no delete.
  2. Slow handler: send returns a message_id → delete called with that id.
  3. Slow handler: send returns data=null (message_id=None) → reference_id fallback
     lookup resolves the id → delete called with the resolved id.
  4. Slow handler: indicator send throws → no delete called (no phantom cleanup).
  5. Indicator disabled via settings → no send.
  6. Constructor default keeps the chat-message indicator disabled.
  7. Missing control credentials → indicator suppressed entirely.
  8. _resolve_message_id_by_reference_id unit tests:
       a. reference found on first attempt
       b. not found after all attempts → None
       c. empty body on first attempt, found on second (retry path)
       d. HTTP 401 → None immediately (no retry)
       e. HTTP 304 → retries → found on second attempt
       f. no control credentials → None (guard check)
"""
from __future__ import annotations

import asyncio
import json
import urllib.error
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_mod = load_plugin_adapter("nextcloud_talk")
NextcloudTalkAdapter = _mod.NextcloudTalkAdapter
NextcloudTalkSettings = _mod.NextcloudTalkSettings


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_adapter(
    *,
    indicator_delay: float = 0.0,
    indicator_enabled: bool = True,
    indicator_text: str = "thinking…",
    control_user: str = "admin",
    control_password: str = "secret",
    native_typing_enabled: bool = False,
) -> NextcloudTalkAdapter:
    from gateway.config import PlatformConfig

    adapter = NextcloudTalkAdapter(PlatformConfig(enabled=True, extra={}))
    adapter.settings = NextcloudTalkSettings(
        base_url="https://nc.example.com",
        bot_secret="s3cr3t",
        processing_indicator_enabled=indicator_enabled,
        processing_indicator_delay_seconds=indicator_delay,
        processing_indicator_text=indicator_text,
        control_user=control_user,
        control_password=control_password,
        native_typing_enabled=native_typing_enabled,
    )
    return adapter


def _make_event(adapter: NextcloudTalkAdapter, chat_id: str = "room123", message_id: str = "42") -> Any:
    from gateway.platforms.base import MessageEvent, MessageType

    source = adapter.build_source(
        chat_id=chat_id,
        chat_name="Test Room",
        chat_type="group",
        user_id="users/alice",
        user_name="Alice",
    )
    return MessageEvent(
        text="hello",
        message_type=MessageType.TEXT,
        source=source,
        message_id=message_id,
    )


async def _fake_to_thread(func, *args, **kwargs):
    """Drop-in replacement for asyncio.to_thread that runs the callable inline.

    Used in indicator tests so the indicator task completes predictably
    within a single asyncio.sleep(0) yield rather than racing against a
    real thread-pool submission.
    """
    return func(*args, **kwargs)


class _FakeHTTPResp:
    """Minimal stand-in for a urllib response used as a context manager."""

    def __init__(self, body: bytes, status: int = 200):
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def _ocs_chat_response(messages: list) -> bytes:
    return json.dumps({"ocs": {"meta": {"status": "ok"}, "data": messages}}).encode()


def _ocs_message(ref_id: str, msg_id: int) -> Dict[str, Any]:
    return {"id": msg_id, "referenceId": ref_id, "message": "hello"}


# ---------------------------------------------------------------------------
# Indicator lifecycle tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_fast_handler_cancels_indicator_before_send():
    """Handler completes before the delay fires → indicator never sent or deleted."""
    adapter = _make_adapter(indicator_delay=100.0)
    event = _make_event(adapter)

    send_one = MagicMock(return_value={"message_id": "99", "reference_id": "ref99"})
    delete_msg = MagicMock()
    adapter._send_one = send_one
    adapter._delete_message = delete_msg
    adapter.handle_message = AsyncMock()  # returns immediately

    with patch("asyncio.to_thread", _fake_to_thread):
        await adapter._handle_message_with_processing_indicator(event)

    send_one.assert_not_called()
    delete_msg.assert_not_called()


@pytest.mark.asyncio
async def test_slow_handler_sends_and_deletes_indicator_with_message_id():
    """Bot send returns message_id → delete is called with that id."""
    adapter = _make_adapter(indicator_delay=0.0)
    event = _make_event(adapter)

    send_one = MagicMock(return_value={"message_id": "55", "reference_id": "ref55"})
    delete_msg = MagicMock(return_value={"status": 200})
    adapter._send_one = send_one
    adapter._delete_message = delete_msg

    # Two yields needed:
    # yield 1 → indicator task advances past asyncio.sleep(0.0) to its own yield
    # yield 2 → indicator task's sleep resolves, _fake_to_thread runs inline, task done
    async def _slow_handle(ev):
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    adapter.handle_message = _slow_handle

    with patch("asyncio.to_thread", _fake_to_thread):
        await adapter._handle_message_with_processing_indicator(event)

    assert send_one.called
    # _send_one is called with _resolve_reference_id=True (4th positional arg).
    call_args = send_one.call_args[0]
    assert call_args[-1] is True, "indicator send must pass _resolve_reference_id=True"
    delete_msg.assert_called_once_with(event.source.chat_id, "55")


@pytest.mark.asyncio
async def test_slow_handler_data_null_resolves_via_reference_id():
    """Send returns data=null (message_id=None) → fallback lookup finds id → delete called."""
    adapter = _make_adapter(indicator_delay=0.0)
    event = _make_event(adapter)

    # Simulates the Talk bot API returning data=null: message_id is None.
    send_one = MagicMock(return_value={"message_id": None, "reference_id": "refABC"})
    delete_msg = MagicMock(return_value={"status": 200})
    resolve_mock = MagicMock(return_value="77")

    adapter._send_one = send_one
    adapter._delete_message = delete_msg
    adapter._resolve_message_id_by_reference_id = resolve_mock

    async def _slow_handle(ev):
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    adapter.handle_message = _slow_handle

    with patch("asyncio.to_thread", _fake_to_thread):
        await adapter._handle_message_with_processing_indicator(event)

    # Fallback resolve must be called.
    assert resolve_mock.called
    # First arg is chat_id, second is reference_id from send.
    resolve_call_args = resolve_mock.call_args[0]
    assert resolve_call_args[0] == event.source.chat_id
    assert resolve_call_args[1] == "refABC"
    # Delete must use the resolved id.
    delete_msg.assert_called_once_with(event.source.chat_id, "77")


@pytest.mark.asyncio
async def test_slow_handler_indicator_send_fails_no_delete():
    """Indicator send throws → no delete called (avoids phantom cleanup)."""
    adapter = _make_adapter(indicator_delay=0.0)
    event = _make_event(adapter)

    send_one = MagicMock(side_effect=OSError("connection refused"))
    delete_msg = MagicMock()
    adapter._send_one = send_one
    adapter._delete_message = delete_msg

    async def _slow_handle(ev):
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    adapter.handle_message = _slow_handle

    with patch("asyncio.to_thread", _fake_to_thread):
        await adapter._handle_message_with_processing_indicator(event)

    delete_msg.assert_not_called()


@pytest.mark.asyncio
async def test_indicator_disabled_no_send():
    """processing_indicator_enabled=False → indicator task is never created."""
    adapter = _make_adapter(indicator_delay=0.0, indicator_enabled=False)
    event = _make_event(adapter)

    send_one = MagicMock()
    delete_msg = MagicMock()
    adapter._send_one = send_one
    adapter._delete_message = delete_msg
    adapter.handle_message = AsyncMock()

    with patch("asyncio.to_thread", _fake_to_thread):
        await adapter._handle_message_with_processing_indicator(event)

    send_one.assert_not_called()
    delete_msg.assert_not_called()


@pytest.mark.asyncio
async def test_constructor_default_disables_processing_message_fallback(monkeypatch):
    """Default configuration relies on native typing, not a temporary chat message."""
    from gateway.config import PlatformConfig

    monkeypatch.delenv("NEXTCLOUD_TALK_PROCESSING_INDICATOR_ENABLED", raising=False)
    adapter = NextcloudTalkAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "base_url": "https://nc.example.com",
                "bot_secret": "s3cr3t",
            },
        )
    )
    event = _make_event(adapter)
    adapter.settings.control_user = "tron"
    adapter.settings.control_password = "secret"

    send_one = MagicMock()
    delete_msg = MagicMock()
    adapter._send_one = send_one
    adapter._delete_message = delete_msg
    adapter.handle_message = AsyncMock()

    with patch("asyncio.to_thread", _fake_to_thread):
        await adapter._handle_message_with_processing_indicator(event)

    assert adapter.settings.processing_indicator_enabled is False
    send_one.assert_not_called()
    delete_msg.assert_not_called()


@pytest.mark.asyncio
async def test_no_control_credentials_suppresses_indicator():
    """Indicator is gated on control credentials; without them nothing is sent."""
    adapter = _make_adapter(indicator_delay=0.0, control_user="", control_password="")
    event = _make_event(adapter)

    send_one = MagicMock()
    delete_msg = MagicMock()
    adapter._send_one = send_one
    adapter._delete_message = delete_msg
    adapter.handle_message = AsyncMock()

    with patch("asyncio.to_thread", _fake_to_thread):
        await adapter._handle_message_with_processing_indicator(event)

    send_one.assert_not_called()
    delete_msg.assert_not_called()


@pytest.mark.asyncio
async def test_native_typing_does_not_suppress_processing_message_fallback(monkeypatch):
    """Native HPB typing is best-effort; keep the removable chat indicator visible."""
    adapter = _make_adapter(indicator_delay=0.0, control_user="tron", control_password="secret")
    event = _make_event(adapter)

    send_one = MagicMock(return_value={"message_id": "88", "reference_id": "ref88"})
    delete_msg = MagicMock(return_value={"status": 200})
    adapter._send_one = send_one
    adapter._delete_message = delete_msg

    async def _slow_handle(ev):
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    adapter.handle_message = _slow_handle
    monkeypatch.setattr(adapter, "_can_native_type", lambda: True)

    with patch("asyncio.to_thread", _fake_to_thread):
        await adapter._handle_message_with_processing_indicator(event)

    send_one.assert_called()
    delete_msg.assert_called_once_with(event.source.chat_id, "88")


@pytest.mark.asyncio
async def test_real_handle_message_path_keeps_indicator_until_background_finishes():
    """Regression: BasePlatformAdapter.handle_message returns after scheduling
    background processing; the indicator must stay alive until that task exits."""
    adapter = _make_adapter(indicator_delay=0.0)
    event = _make_event(adapter)

    send_one = MagicMock(return_value={"message_id": "88", "reference_id": "ref88"})
    delete_msg = MagicMock(return_value={"status": 200})
    adapter._send_one = send_one
    adapter._delete_message = delete_msg
    adapter._send_with_retry = AsyncMock(return_value=MagicMock(success=True, message_id="final"))

    background_started = asyncio.Event()
    release_background = asyncio.Event()

    async def _slow_message_handler(ev):
        background_started.set()
        await release_background.wait()
        return "final response"

    adapter._message_handler = _slow_message_handler

    with patch("asyncio.to_thread", _fake_to_thread):
        task = asyncio.create_task(adapter._handle_message_with_processing_indicator(event))
        await asyncio.wait_for(background_started.wait(), timeout=1.0)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert send_one.called, "indicator should be visible while background processing is running"
        delete_msg.assert_not_called()
        release_background.set()
        await asyncio.wait_for(task, timeout=1.0)

    delete_msg.assert_called_once_with(event.source.chat_id, "88")


@pytest.mark.asyncio
async def test_real_handle_message_path_waits_with_profile_namespaced_session_key(tmp_path):
    """Multiplexed profile keys must be resolved through SessionStore, not the legacy fallback."""
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    adapter = _make_adapter(indicator_delay=0.0)
    store = SessionStore(sessions_dir=tmp_path, config=GatewayConfig(multiplex_profiles=True))
    adapter.set_session_store(store)
    event = _make_event(adapter)
    event.source.profile = "coder"

    send_one = MagicMock(return_value={"message_id": "89", "reference_id": "ref89"})
    delete_msg = MagicMock(return_value={"status": 200})
    adapter._send_one = send_one
    adapter._delete_message = delete_msg
    adapter._send_with_retry = AsyncMock(return_value=MagicMock(success=True, message_id="final"))

    background_started = asyncio.Event()
    release_background = asyncio.Event()

    async def _slow_message_handler(ev):
        background_started.set()
        await release_background.wait()
        return "final response"

    adapter._message_handler = _slow_message_handler

    with patch("asyncio.to_thread", _fake_to_thread):
        task = asyncio.create_task(adapter._handle_message_with_processing_indicator(event))
        await asyncio.wait_for(background_started.wait(), timeout=1.0)
        for _ in range(20):
            if send_one.called:
                break
            await asyncio.sleep(0)
        assert send_one.called
        delete_msg.assert_not_called()
        release_background.set()
        await asyncio.wait_for(task, timeout=1.0)

    delete_msg.assert_called_once_with(event.source.chat_id, "89")


def test_native_typing_room_response_collects_existing_participant_sessions():
    """Initial HPB room responses may include already-present sessions without a join event."""
    adapter = _make_adapter()
    client = _mod._NextcloudTalkTypingClient(adapter, "room123")
    client._own_signaling_session_id = "own-session"

    client._handle_signaling_payload({
        "type": "room",
        "room": {
            "sessions": [
                {"sessionid": "own-session"},
                {"sessionid": "user-session-1"},
                {"sessionId": "user-session-2"},
            ]
        },
    })

    assert client._participants == {"user-session-1", "user-session-2"}


# ---------------------------------------------------------------------------
# _resolve_message_id_by_reference_id unit tests
# ---------------------------------------------------------------------------

def _make_ctrl_adapter() -> NextcloudTalkAdapter:
    return _make_adapter(control_user="admin", control_password="secret")


def test_resolve_reference_found_on_first_attempt():
    adapter = _make_ctrl_adapter()
    ref_id = "abc123"
    body = _ocs_chat_response([_ocs_message(ref_id, 42)])

    with patch("urllib.request.urlopen", return_value=_FakeHTTPResp(body)):
        result = adapter._resolve_message_id_by_reference_id("tok1", ref_id)

    assert result == "42"


def test_resolve_reference_not_in_history_returns_none():
    adapter = _make_ctrl_adapter()
    body = _ocs_chat_response([_ocs_message("other-ref", 99)])

    with patch("urllib.request.urlopen", return_value=_FakeHTTPResp(body)):
        result = adapter._resolve_message_id_by_reference_id("tok1", "missing", attempts=1)

    assert result is None


def test_resolve_empty_body_retries_then_finds():
    """An empty response body is treated as transient; retry finds the message."""
    adapter = _make_ctrl_adapter()
    ref_id = "xyz789"
    good_body = _ocs_chat_response([_ocs_message(ref_id, 55)])

    call_count = [0]

    def _urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return _FakeHTTPResp(b"")
        return _FakeHTTPResp(good_body)

    with patch("urllib.request.urlopen", side_effect=_urlopen):
        with patch("time.sleep"):  # skip real delays
            result = adapter._resolve_message_id_by_reference_id("tok1", ref_id, attempts=3)

    assert result == "55"
    assert call_count[0] == 2


def test_resolve_http_401_returns_none_immediately():
    """HTTP 401 is not retried — credentials are wrong, not transient."""
    adapter = _make_ctrl_adapter()

    err = urllib.error.HTTPError("url", 401, "Unauthorized", {}, None)

    with patch("urllib.request.urlopen", side_effect=err):
        result = adapter._resolve_message_id_by_reference_id("tok1", "ref", attempts=5)

    assert result is None


def test_resolve_http_304_retries_then_finds():
    """HTTP 304 from chat history endpoint is retried; second attempt returns the message."""
    adapter = _make_ctrl_adapter()
    ref_id = "ref304"
    good_body = _ocs_chat_response([_ocs_message(ref_id, 77)])

    call_count = [0]

    def _urlopen(req, timeout=None):
        call_count[0] += 1
        if call_count[0] == 1:
            raise urllib.error.HTTPError("url", 304, "Not Modified", {}, None)
        return _FakeHTTPResp(good_body)

    with patch("urllib.request.urlopen", side_effect=_urlopen):
        with patch("time.sleep"):
            result = adapter._resolve_message_id_by_reference_id("tok1", ref_id, attempts=3)

    assert result == "77"
    assert call_count[0] == 2


def test_resolve_no_control_credentials_returns_none():
    """Guard check: if control creds are not configured, skip lookup entirely."""
    adapter = _make_adapter(control_user="", control_password="")
    result = adapter._resolve_message_id_by_reference_id("tok1", "someref")
    assert result is None


def test_nextcloud_actor_ids_are_sanitized_for_session_keys():
    """Nextcloud actor IDs use slashes; Hermes session keys must not."""
    from gateway.session import build_session_key

    adapter = _make_adapter()
    source = adapter.build_source(
        chat_id="room123",
        chat_name="Test Room",
        chat_type="group",
        user_id=_mod._session_safe_actor_id("users/alice"),
        user_name="Alice",
    )

    session_key = build_session_key(source)

    assert source.user_id == "users_alice"
    assert "/" not in session_key
    assert "\\" not in session_key
    assert "users_alice" in session_key


# ---------------------------------------------------------------------------
# Native HPB typing indicator tests
# ---------------------------------------------------------------------------

def test_websocket_url_from_signaling_server_normalizes_spreed_path():
    build_url = _mod._websocket_url_from_signaling_server

    assert build_url("https://nc.example.com/standalone-signaling/") == "wss://nc.example.com/standalone-signaling/spreed"
    assert build_url("http://nc.example.com/standalone-signaling") == "ws://nc.example.com/standalone-signaling/spreed"
    assert build_url("https://nc.example.com/standalone-signaling/spreed") == "wss://nc.example.com/standalone-signaling/spreed"
    assert build_url("") == ""


@pytest.mark.asyncio
async def test_send_typing_starts_native_client_and_stop_typing_stops(monkeypatch):
    adapter = _make_adapter(control_user="tron", control_password="secret", native_typing_enabled=True)
    created = []

    class FakeTypingClient:
        def __init__(self, adapter_arg, token):
            self.adapter = adapter_arg
            self.token = token
            self.started = False
            self.stopped = False
            created.append(self)

        async def ensure_started(self):
            self.started = True

        async def stop(self):
            self.stopped = True

    monkeypatch.setattr(_mod, "_NextcloudTalkTypingClient", FakeTypingClient)

    await adapter.send_typing("room123")
    await adapter.send_typing("room123")

    assert len(created) == 1
    assert created[0].started is True
    assert created[0].token == "room123"
    assert adapter._typing_clients["room123"] is created[0]

    await adapter.stop_typing("room123")

    assert created[0].stopped is True
    assert "room123" not in adapter._typing_clients


@pytest.mark.asyncio
async def test_send_typing_noops_without_control_credentials(monkeypatch):
    adapter = _make_adapter(control_user="", control_password="")

    class ExplodingTypingClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("native typing client should not be created")

    monkeypatch.setattr(_mod, "_NextcloudTalkTypingClient", ExplodingTypingClient)
    await adapter.send_typing("room123")

    assert adapter._typing_clients == {}


@pytest.mark.asyncio
async def test_native_typing_client_sends_started_and_stopped_payloads():
    adapter = _make_adapter(control_user="tron", control_password="secret")
    client = _mod._NextcloudTalkTypingClient(adapter, "room123")

    class FakeWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(json.loads(payload))

    websocket = FakeWebSocket()
    client._websocket = websocket
    client._own_signaling_session_id = "own-session"
    client._handle_signaling_payload({
        "type": "event",
        "event": {
            "target": "room",
            "type": "join",
            "join": [
                {"sessionid": "own-session"},
                {"sessionid": "ronny-session"},
            ],
        },
    })

    assert client._participants == {"ronny-session"}

    await client._broadcast_typing(True)
    await client._broadcast_typing(False)

    assert [item["message"]["data"]["type"] for item in websocket.sent] == ["startedTyping", "stoppedTyping"]
    assert websocket.sent[0]["message"]["recipient"] == {"type": "room"}
    assert "to" not in websocket.sent[0]["message"]["data"]
    assert websocket.sent[1]["message"]["recipient"] == {"type": "room"}

    client._handle_signaling_payload({
        "type": "event",
        "event": {"target": "room", "type": "leave", "leave": [{"sessionid": "ronny-session"}]},
    })
    assert client._participants == set()


@pytest.mark.asyncio
async def test_native_typing_room_broadcast_reaches_empty_observed_participants():
    """Room-recipient broadcast covers users already present before relay join."""
    adapter = _make_adapter(control_user="tron", control_password="secret")
    client = _mod._NextcloudTalkTypingClient(adapter, "room123")

    class FakeWebSocket:
        def __init__(self):
            self.sent = []

        async def send(self, payload):
            self.sent.append(json.loads(payload))

    websocket = FakeWebSocket()
    client._websocket = websocket
    assert client._participants == set()

    await client._broadcast_typing(True)
    await client._broadcast_typing(False)

    assert [item["message"]["recipient"] for item in websocket.sent] == [{"type": "room"}, {"type": "room"}]
    assert [item["message"]["data"]["type"] for item in websocket.sent] == ["startedTyping", "stoppedTyping"]


@pytest.mark.asyncio
async def test_native_typing_room_broadcast_noops_without_websocket():
    adapter = _make_adapter(control_user="tron", control_password="secret")
    client = _mod._NextcloudTalkTypingClient(adapter, "room123")
    client._websocket = None

    await client._broadcast_typing(True)
    await client._broadcast_typing(False)
