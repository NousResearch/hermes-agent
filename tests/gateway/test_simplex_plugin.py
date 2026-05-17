"""Tests for the SimpleX Chat platform-plugin adapter.

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_simplex`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests on the same xdist worker.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_simplex = load_plugin_adapter("simplex")

SimplexAdapter = _simplex.SimplexAdapter
check_requirements = _simplex.check_requirements
validate_config = _simplex.validate_config
is_connected = _simplex.is_connected
register = _simplex.register
_env_enablement = _simplex._env_enablement
_standalone_send = _simplex._standalone_send
_guess_extension = _simplex._guess_extension
_is_image_ext = _simplex._is_image_ext
_is_audio_ext = _simplex._is_audio_ext
_CORR_PREFIX = _simplex._CORR_PREFIX
POLL_COMMAND_TIMEOUT = _simplex.POLL_COMMAND_TIMEOUT
POLL_CONNECT_TIMEOUT = _simplex.POLL_CONNECT_TIMEOUT
POLL_WALL_TIMEOUT = _simplex.POLL_WALL_TIMEOUT
POLL_STALL_WARN_SECONDS = _simplex.POLL_STALL_WARN_SECONDS
SIMPLEX_ACTIVE_SESSION_MAX_SECONDS = _simplex.SIMPLEX_ACTIVE_SESSION_MAX_SECONDS
SIMPLEX_PROCESSING_NOTICE_DELAY = _simplex.SIMPLEX_PROCESSING_NOTICE_DELAY
_simplex_quote_name = _simplex._simplex_quote_name


# ---------------------------------------------------------------------------
# 1. Platform enum (plugin-discovered, not bundled)
# ---------------------------------------------------------------------------

def test_platform_enum_resolves_via_plugin_scan():
    """The plugin filesystem scan should expose Platform("simplex")."""
    from gateway.config import Platform
    p = Platform("simplex")
    assert p.value == "simplex"
    # Identity stability — repeated lookups return the same pseudo-member
    assert Platform("simplex") is p


# ---------------------------------------------------------------------------
# 2. check_requirements / validate_config / is_connected
# ---------------------------------------------------------------------------

def test_check_requirements_needs_url(monkeypatch):
    monkeypatch.delenv("SIMPLEX_WS_URL", raising=False)
    assert check_requirements() is False


def test_check_requirements_true_when_configured(monkeypatch):
    monkeypatch.setenv("SIMPLEX_WS_URL", "ws://127.0.0.1:5225")
    # websockets is a dev dep in this repo via the test plugins; the
    # check_requirements() gate also asserts the package imports.
    websockets_present = True
    try:
        import websockets  # noqa: F401
    except ImportError:
        websockets_present = False
    assert check_requirements() is websockets_present


def test_validate_config_uses_env_or_extra(monkeypatch):
    from gateway.config import PlatformConfig
    monkeypatch.delenv("SIMPLEX_WS_URL", raising=False)
    # Empty extra + no env → invalid
    cfg = PlatformConfig(enabled=True)
    assert validate_config(cfg) is False
    # extra-only path → valid
    cfg2 = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    assert validate_config(cfg2) is True


def test_is_connected_mirrors_validate(monkeypatch):
    from gateway.config import PlatformConfig
    monkeypatch.delenv("SIMPLEX_WS_URL", raising=False)
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://x"})
    assert is_connected(cfg) is True
    assert is_connected(PlatformConfig(enabled=True)) is False


# ---------------------------------------------------------------------------
# 3. _env_enablement seeds PlatformConfig.extra
# ---------------------------------------------------------------------------

def test_env_enablement_none_when_unset(monkeypatch):
    monkeypatch.delenv("SIMPLEX_WS_URL", raising=False)
    assert _env_enablement() is None


def test_env_enablement_seeds_ws_url(monkeypatch):
    monkeypatch.setenv("SIMPLEX_WS_URL", "ws://127.0.0.1:5225")
    monkeypatch.delenv("SIMPLEX_HOME_CHANNEL", raising=False)
    seed = _env_enablement()
    assert seed == {"ws_url": "ws://127.0.0.1:5225"}


def test_env_enablement_seeds_home_channel(monkeypatch):
    monkeypatch.setenv("SIMPLEX_WS_URL", "ws://127.0.0.1:5225")
    monkeypatch.setenv("SIMPLEX_HOME_CHANNEL", "42")
    monkeypatch.setenv("SIMPLEX_HOME_CHANNEL_NAME", "Personal")
    seed = _env_enablement()
    assert seed["home_channel"] == {"chat_id": "42", "name": "Personal"}


def test_env_enablement_home_channel_defaults_name_to_id(monkeypatch):
    monkeypatch.setenv("SIMPLEX_WS_URL", "ws://127.0.0.1:5225")
    monkeypatch.setenv("SIMPLEX_HOME_CHANNEL", "42")
    monkeypatch.delenv("SIMPLEX_HOME_CHANNEL_NAME", raising=False)
    seed = _env_enablement()
    assert seed["home_channel"] == {"chat_id": "42", "name": "42"}


# ---------------------------------------------------------------------------
# 4. Adapter init
# ---------------------------------------------------------------------------

def test_adapter_init_custom_url():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    assert adapter.ws_url == "ws://localhost:5225"
    assert adapter._running is False
    assert adapter._ws is None


def test_adapter_init_default_url():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True)
    adapter = SimplexAdapter(cfg)
    assert adapter.ws_url == "ws://127.0.0.1:5225"


def test_adapter_platform_identity():
    """Adapter should expose Platform("simplex") identity."""
    from gateway.config import Platform, PlatformConfig
    cfg = PlatformConfig(enabled=True)
    adapter = SimplexAdapter(cfg)
    assert adapter.platform is Platform("simplex")


# ---------------------------------------------------------------------------
# 5. Helper functions (magic-byte detection)
# ---------------------------------------------------------------------------

def test_guess_extension_png():
    assert _guess_extension(b"\x89PNG\r\n\x1a\n") == ".png"


def test_guess_extension_jpg():
    assert _guess_extension(b"\xff\xd8\xff\xe0") == ".jpg"


def test_guess_extension_ogg():
    assert _guess_extension(b"OggS\x00\x02") == ".ogg"


def test_guess_extension_unknown():
    assert _guess_extension(b"\x00\x01\x02\x03") == ".bin"


def test_is_image_ext():
    assert _is_image_ext(".png") is True
    assert _is_image_ext(".webp") is True
    assert _is_image_ext(".ogg") is False


def test_is_audio_ext():
    assert _is_audio_ext(".ogg") is True
    assert _is_audio_ext(".mp3") is True
    assert _is_audio_ext(".pdf") is False


# ---------------------------------------------------------------------------
# 6. Correlation IDs
# ---------------------------------------------------------------------------

def test_corr_id_starts_with_prefix_and_tracks_pending():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    corr_id = adapter._make_corr_id()
    assert corr_id.startswith(_CORR_PREFIX)
    assert corr_id in adapter._pending_corr_ids


def test_corr_id_pending_set_self_trims():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._max_pending_corr = 4
    for _ in range(10):
        adapter._make_corr_id()
    # After many additions, the pending set should be bounded by the trim
    # logic — at most one trim window above the cap.
    assert len(adapter._pending_corr_ids) <= adapter._max_pending_corr + 1


# ---------------------------------------------------------------------------
# 7. Outbound send (mocked WS)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_dm():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    mock_ws = AsyncMock()
    adapter._ws = mock_ws

    result = await adapter.send("contact-42", "Hello, SimpleX!")
    mock_ws.send.assert_called_once()
    payload = json.loads(mock_ws.send.call_args[0][0])
    assert payload["cmd"] == "@[contact-42] Hello, SimpleX!"
    assert payload["corrId"].startswith(_CORR_PREFIX)
    assert result.success is True


@pytest.mark.asyncio
async def test_send_group():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    mock_ws = AsyncMock()
    adapter._ws = mock_ws

    result = await adapter.send("group:grp-99", "Hello, group!")
    payload = json.loads(mock_ws.send.call_args[0][0])
    assert payload["cmd"] == "#[grp-99] Hello, group!"
    assert result.success is True


@pytest.mark.asyncio
async def test_send_when_ws_not_connected_does_not_crash():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    # No _ws assigned — _send_ws should drop quietly
    result = await adapter.send("contact-42", "hi")
    assert result.success is True  # send() always returns success — fire-and-forget


# ---------------------------------------------------------------------------
# 8. Inbound: seen item bookkeeping
# ---------------------------------------------------------------------------

def test_item_key_is_per_chat_not_global_item_id():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    direct = {
        "chatInfo": {"type": "direct", "contact": {"contactId": 4}},
        "chatItem": {"meta": {"itemId": 7}},
    }
    group = {
        "chatInfo": {"type": "group", "groupInfo": {"groupId": 1}},
        "chatItem": {"meta": {"itemId": 7}},
    }

    assert adapter._item_key(direct) == "direct:4:7"
    assert adapter._item_key(group) == "group:1:7"
    assert adapter._chat_ref(direct) == "@4"
    assert adapter._chat_ref(group) == "#1"


@pytest.mark.asyncio
async def test_handle_new_chat_item_marks_inbound_item_read_without_blocking():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    calls = []
    handled = []

    async def fake_command_once(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return {"resp": {"type": "itemsReadForChat"}}

    async def handler(event):
        handled.append(event.text)
        return ""

    adapter._command_once = fake_command_once  # type: ignore[method-assign]
    adapter.set_message_handler(handler)
    wrapper = {
        "chatInfo": {"type": "direct", "contact": {"contactId": 4, "localDisplayName": "Elkim"}},
        "chatItem": {
            "meta": {"itemId": 95, "itemStatus": {"type": "rcvNew"}, "itemTs": "2026-05-17T10:24:35Z"},
            "content": {"type": "rcvMsgContent", "msgContent": {"type": "text", "text": "ping"}},
        },
    }

    await adapter._handle_new_chat_item(wrapper)
    await _simplex.asyncio.sleep(0)

    assert handled == ["ping"]
    assert calls == [(
        "/_read chat items @4 95",
        {"timeout": 1.0, "open_timeout": 1.0, "wall_timeout": 2.0},
    )]


def test_seen_items_persist_across_adapter_restart(tmp_path):
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})

    first = SimplexAdapter(cfg)
    first._seen_items_path = tmp_path / "simplex_seen_items.json"
    first._seen_item_ids.update({"direct:4:41", "group:1:39"})
    first._save_seen_items()

    second = SimplexAdapter(cfg)
    second._seen_items_path = first._seen_items_path
    second._load_seen_items()

    assert "direct:4:41" in second._seen_item_ids
    assert "group:1:39" in second._seen_item_ids


def test_item_key_requires_chat_id():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    assert adapter._item_key({"chatInfo": {"type": "direct"}, "chatItem": {"meta": {"itemId": 7}}}) is None


def test_simplex_quote_name_escapes_single_quotes():
    assert _simplex_quote_name("Bob's Room") == "Bob\\'s Room"


@pytest.mark.asyncio
async def test_resolve_chat_target_uses_display_name():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    async def fake_command_once(cmd, **kwargs):
        assert cmd == "/chats all"
        return {
            "resp": {
                "chats": [
                    {"chatInfo": {"contact": {"contactId": 4, "localDisplayName": "Elkim"}}},
                    {"chatInfo": {"groupInfo": {"groupId": 1, "groupProfile": {"displayName": "Žofka_1"}}}},
                ]
            }
        }

    adapter._command_once = fake_command_once  # type: ignore[method-assign]

    assert await adapter._resolve_chat_target("4") == "@'Elkim'"
    assert await adapter._resolve_chat_target("group:1") == "#'Žofka_1'"


@pytest.mark.asyncio
async def test_resolve_chat_target_keeps_legacy_non_numeric_fallback():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    async def fake_command_once(cmd, **kwargs):
        return {"resp": {"chats": []}}

    adapter._command_once = fake_command_once  # type: ignore[method-assign]

    assert await adapter._resolve_chat_target("contact-42") == "@[contact-42]"
    assert await adapter._resolve_chat_target("group:grp-99") == "#[grp-99]"


@pytest.mark.asyncio
async def test_seed_seen_items_does_not_consume_unread_inbound_text(tmp_path):
    """Startup recovery must not silently eat missed SimpleX DMs."""
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._seen_items_path = tmp_path / "simplex_seen_items.json"

    unread = {
        "chatInfo": {"type": "direct", "contact": {"contactId": 4}},
        "chatItem": {
            "meta": {"itemId": 100, "itemStatus": {"type": "rcvNew"}, "itemTs": "2026-05-17T10:43:07Z"},
            "content": {"type": "rcvMsgContent", "msgContent": {"type": "text", "text": "latency test 3"}},
        },
    }
    sent = {
        "chatInfo": {"type": "direct", "contact": {"contactId": 4}},
        "chatItem": {
            "meta": {"itemId": 99, "itemStatus": {"type": "sndRcvd"}, "itemTs": "2026-05-17T10:51:47Z"},
            "content": {"type": "sndMsgContent", "msgContent": {"type": "text", "text": "shutdown"}},
        },
    }

    async def fake_command_once(_cmd, **_kwargs):
        return {"resp": {"chatItems": [unread, sent]}}

    adapter._command_once = fake_command_once  # type: ignore[method-assign]
    await adapter._seed_seen_items()

    assert "direct:4:100" not in adapter._seen_item_ids
    assert "direct:4:99" in adapter._seen_item_ids


@pytest.mark.asyncio
async def test_connect_runs_polling_outside_gateway_event_loop(monkeypatch):
    """SimpleX polling must not starve behind long gateway/agent turns."""
    import sys
    import types

    from gateway.config import PlatformConfig

    class FakeConnect:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *_exc):
            return False

    fake_websockets = types.SimpleNamespace(connect=lambda *_a, **_kw: FakeConnect())
    monkeypatch.setitem(sys.modules, "websockets", fake_websockets)

    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    async def noop_seed():
        return None

    async def idle_task():
        await _simplex.asyncio.sleep(60)

    adapter._seed_seen_items = noop_seed  # type: ignore[method-assign]
    adapter._ws_listener = idle_task  # type: ignore[method-assign]
    adapter._health_monitor = idle_task  # type: ignore[method-assign]

    assert await adapter.connect() is True
    try:
        assert adapter._poll_task is None
        assert adapter._poll_thread is not None
        assert adapter._poll_thread.is_alive()
    finally:
        await adapter.disconnect()


@pytest.mark.asyncio
async def test_poll_dispatches_preconnect_unread_inbound_text(monkeypatch, tmp_path):
    """Unread user text remains actionable even if its timestamp predates reconnect."""
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._seen_items_path = tmp_path / "simplex_seen_items.json"
    adapter._running = True
    adapter._connected_at = 1_779_012_000.0  # after the item timestamp
    handled = []

    unread = {
        "chatInfo": {"type": "direct", "contact": {"contactId": 4}},
        "chatItem": {
            "meta": {"itemId": 100, "itemStatus": {"type": "rcvNew"}, "itemTs": "2026-05-17T10:43:07Z"},
            "content": {"type": "rcvMsgContent", "msgContent": {"type": "text", "text": "latency test 3"}},
        },
    }

    async def fake_sleep(_seconds):
        adapter._running = False

    async def fake_command_once(_cmd, **_kwargs):
        return {"resp": {"chatItems": [unread]}}

    async def fake_handle(wrapper):
        handled.append(wrapper["chatItem"]["meta"]["itemId"])

    monkeypatch.setattr(_simplex.asyncio, "sleep", fake_sleep)
    adapter._command_once = fake_command_once  # type: ignore[method-assign]
    adapter._handle_new_chat_item = fake_handle  # type: ignore[method-assign]

    await adapter._poll_unread_items()
    for task in list(adapter._poll_dispatch_tasks):
        await task

    assert handled == [100]
    assert "direct:4:100" in adapter._seen_item_ids


@pytest.mark.asyncio
async def test_poll_unread_uses_short_command_timeouts(monkeypatch):
    """Polling is the latency fallback; it must not wait on 10s WS stalls."""
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._running = True

    calls = []

    async def fake_sleep(_seconds):
        adapter._running = False

    async def fake_command_once(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return {"resp": {"chatItems": []}}

    monkeypatch.setattr(_simplex.asyncio, "sleep", fake_sleep)
    adapter._command_once = fake_command_once  # type: ignore[method-assign]

    await adapter._poll_unread_items()

    assert calls == [
        (
            "/tail 50",
            {
                "timeout": POLL_COMMAND_TIMEOUT,
                "open_timeout": POLL_CONNECT_TIMEOUT,
                "wall_timeout": POLL_WALL_TIMEOUT,
            },
        )
    ]
    assert POLL_COMMAND_TIMEOUT <= 2.0
    assert POLL_CONNECT_TIMEOUT <= 2.0
    assert POLL_WALL_TIMEOUT <= 3.5


@pytest.mark.asyncio
async def test_command_once_has_hard_wall_timeout(monkeypatch, caplog):
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    async def stuck_impl(*_args, **_kwargs):
        await _simplex.asyncio.sleep(60)

    adapter._command_once_impl = stuck_impl  # type: ignore[method-assign]

    with caplog.at_level("WARNING"):
        result = await adapter._command_once(
            "/tail 50",
            timeout=10,
            open_timeout=10,
            wall_timeout=0.01,
        )

    assert result is None
    assert "exceeded wall timeout" in caplog.text


@pytest.mark.asyncio
async def test_command_once_wall_timeout_does_not_wait_for_slow_cancellation(caplog):
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    cleanup_started = _simplex.asyncio.Event()

    async def stubborn_impl(*_args, **_kwargs):
        try:
            await _simplex.asyncio.sleep(60)
        except _simplex.asyncio.CancelledError:
            cleanup_started.set()
            await _simplex.asyncio.sleep(0.2)
            raise

    adapter._command_once_impl = stubborn_impl  # type: ignore[method-assign]

    started = _simplex.time.time()
    with caplog.at_level("WARNING"):
        result = await adapter._command_once(
            "/tail 50",
            timeout=10,
            open_timeout=10,
            wall_timeout=0.01,
        )
    elapsed = _simplex.time.time() - started

    assert result is None
    assert elapsed < 0.1
    assert "exceeded wall timeout" in caplog.text
    await _simplex.asyncio.wait_for(cleanup_started.wait(), timeout=1)
    await _simplex.asyncio.sleep(0.25)


@pytest.mark.asyncio
async def test_poll_loop_logs_stall_gap(monkeypatch, caplog):
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._running = True
    adapter._last_poll_started_at = 100.0

    async def fake_sleep(_seconds):
        adapter._running = False

    async def fake_command_once(_cmd, **_kwargs):
        return {"resp": {"chatItems": []}}

    times = iter([100.0 + POLL_STALL_WARN_SECONDS + 1.0, 100.1])
    monkeypatch.setattr(_simplex.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(_simplex.time, "time", lambda: next(times, 100.1))
    adapter._command_once = fake_command_once  # type: ignore[method-assign]

    with caplog.at_level("WARNING"):
        await adapter._poll_unread_items()

    assert "loop stalled" in caplog.text


@pytest.mark.asyncio
async def test_processing_notice_is_delayed_and_cancelled_before_send(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.base import MessageType, ProcessingOutcome

    cfg = PlatformConfig(
        enabled=True,
        extra={"ws_url": "ws://localhost:5225", "processing_notice_delay": 10},
    )
    adapter = SimplexAdapter(cfg)
    sent = []

    async def fake_send(chat_id, content, **_kwargs):
        sent.append((chat_id, content))
        from gateway.platforms.base import SendResult
        return SendResult(success=True)

    adapter.send = fake_send  # type: ignore[method-assign]
    source = adapter.build_source(
        chat_id="4",
        chat_name="Elkim",
        chat_type="dm",
        user_id="4",
        user_name="Elkim",
    )
    event = _simplex.MessageEvent(source=source, text="slow", message_type=MessageType.TEXT)

    await adapter.on_processing_start(event)
    assert adapter._processing_notice_tasks
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)
    await _simplex.asyncio.sleep(0)

    assert sent == []
    assert adapter._processing_notice_tasks == {}


@pytest.mark.asyncio
async def test_processing_notice_sends_for_slow_simplex_turn(monkeypatch):
    from gateway.config import PlatformConfig
    from gateway.platforms.base import MessageType, ProcessingOutcome

    cfg = PlatformConfig(
        enabled=True,
        extra={"ws_url": "ws://localhost:5225", "processing_notice_delay": 0.01},
    )
    adapter = SimplexAdapter(cfg)
    sent = []

    async def fake_send(chat_id, content, **_kwargs):
        sent.append((chat_id, content))
        from gateway.platforms.base import SendResult
        return SendResult(success=True)

    adapter.send = fake_send  # type: ignore[method-assign]
    source = adapter.build_source(
        chat_id="4",
        chat_name="Elkim",
        chat_type="dm",
        user_id="4",
        user_name="Elkim",
    )
    event = _simplex.MessageEvent(source=source, text="slow", message_type=MessageType.TEXT)

    await adapter.on_processing_start(event)
    await _simplex.asyncio.sleep(0.03)

    assert sent == [("4", "Still here — SimpleX has no typing indicator, but I’m working on it.")]
    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)


def test_simplex_adapter_opts_into_active_session_hard_timeout():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    assert adapter._max_active_session_seconds == SIMPLEX_ACTIVE_SESSION_MAX_SECONDS
    assert adapter._active_session_hard_timeout_seconds() == SIMPLEX_ACTIVE_SESSION_MAX_SECONDS


def test_simplex_adapter_respects_configured_active_session_hard_timeout():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(
        enabled=True,
        extra={"ws_url": "ws://localhost:5225", "max_active_session_seconds": 12},
    )
    adapter = SimplexAdapter(cfg)

    assert adapter._active_session_hard_timeout_seconds() == 12.0


@pytest.mark.asyncio
async def test_dispatch_polled_item_tracks_task_and_logs_dispatch():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    gate = _simplex.asyncio.Event()
    handled = []

    async def fake_handle(wrapper):
        handled.append(wrapper)
        await gate.wait()

    adapter._handle_new_chat_item = fake_handle  # type: ignore[method-assign]
    task = adapter._dispatch_polled_item({"x": 1}, "direct:4:99")

    await _simplex.asyncio.sleep(0)
    assert task in adapter._poll_dispatch_tasks
    assert handled == [{"x": 1}]

    gate.set()
    await task
    assert task not in adapter._poll_dispatch_tasks


@pytest.mark.asyncio
async def test_dispatch_polled_items_preserves_tail_order_for_bursts():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    handled = []

    async def fake_handle(wrapper):
        # If the burst were dispatched as independent tasks, the second item
        # would finish first. Ordered batch dispatch must preserve daemon order.
        if wrapper["item"] == 1:
            await _simplex.asyncio.sleep(0.02)
        handled.append(wrapper["item"])

    adapter._handle_new_chat_item = fake_handle  # type: ignore[method-assign]
    task = adapter._dispatch_polled_items([
        ({"item": 1}, "direct:4:101"),
        ({"item": 2}, "direct:4:102"),
        ({"item": 3}, "direct:4:103"),
    ])

    await task
    assert handled == [1, 2, 3]
    assert task not in adapter._poll_dispatch_tasks


@pytest.mark.asyncio
async def test_stale_active_simplex_session_is_cancelled_for_fresh_message():
    from gateway.config import PlatformConfig
    from gateway.platforms.base import MessageType

    cfg = PlatformConfig(
        enabled=True,
        extra={"ws_url": "ws://localhost:5225", "max_active_session_seconds": 0.01},
    )
    adapter = SimplexAdapter(cfg)
    adapter._max_active_session_seconds = 0.01

    started = _simplex.asyncio.Event()
    cancelled = _simplex.asyncio.Event()
    handled_texts = []

    async def handler(event):
        handled_texts.append(event.text)
        if event.text == "old":
            started.set()
            try:
                await _simplex.asyncio.sleep(60)
            except _simplex.asyncio.CancelledError:
                cancelled.set()
                raise
        return ""

    adapter.set_message_handler(handler)
    source = adapter.build_source(
        chat_id="4",
        chat_name="Elkim",
        chat_type="dm",
        user_id="4",
        user_name="Elkim",
    )
    old = _simplex.MessageEvent(source=source, text="old", message_type=MessageType.TEXT)
    fresh = _simplex.MessageEvent(source=source, text="fresh", message_type=MessageType.TEXT)

    await adapter.handle_message(old)
    await _simplex.asyncio.wait_for(started.wait(), timeout=1)
    await _simplex.asyncio.sleep(0.02)
    await adapter.handle_message(fresh)
    await _simplex.asyncio.wait_for(cancelled.wait(), timeout=1)
    await _simplex.asyncio.sleep(0)

    assert handled_texts[:2] == ["old", "fresh"]


# ---------------------------------------------------------------------------
# 9. Inbound: filter own-echo by corrId prefix
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_event_filters_own_corr_id():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    # Pretend we sent a command with this corrId
    own = adapter._make_corr_id()
    handler_mock = AsyncMock()
    adapter._handle_new_chat_item = handler_mock  # type: ignore

    await adapter._handle_event({"corrId": own, "type": "newChatItem"})
    handler_mock.assert_not_called()
    assert own not in adapter._pending_corr_ids  # discarded


# ---------------------------------------------------------------------------
# 9. Standalone (out-of-process) send for cron
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_standalone_send_missing_websockets(monkeypatch):
    """When websockets is unimportable, return a clean error dict.

    Implementation detail: the standalone path does ``import websockets``
    inside the function body. We simulate the package being absent by
    pulling it out of ``sys.modules`` and pointing the finder at None.
    """
    import sys
    saved_websockets = sys.modules.pop("websockets", None)
    saved_meta = list(sys.meta_path)

    class _Blocker:
        @staticmethod
        def find_spec(name, path=None, target=None):
            if name == "websockets" or name.startswith("websockets."):
                raise ImportError("websockets blocked for test")
            return None

    sys.meta_path.insert(0, _Blocker())
    try:
        pconfig = MagicMock()
        pconfig.extra = {"ws_url": "ws://localhost:5225"}
        result = await _standalone_send(pconfig, "contact-42", "hi")
        assert isinstance(result, dict)
        assert "error" in result
        assert "websockets" in result["error"]
    finally:
        sys.meta_path[:] = saved_meta
        if saved_websockets is not None:
            sys.modules["websockets"] = saved_websockets


@pytest.mark.asyncio
async def test_standalone_send_missing_url(monkeypatch):
    monkeypatch.delenv("SIMPLEX_WS_URL", raising=False)
    pconfig = MagicMock()
    pconfig.extra = {}
    # We expect the URL fallback (extra+env both empty) to be empty string,
    # producing an error. We also need websockets to be importable for the
    # url-check branch to be reached, so skip when it's not.
    try:
        import websockets.client  # noqa: F401
    except ImportError:
        pytest.skip("websockets not installed")

    result = await _standalone_send(pconfig, "contact-42", "hi")
    assert isinstance(result, dict)
    # Either error about URL or a connection attempt failure — both are valid
    # signals that the standalone path requires configuration.
    assert "error" in result


# ---------------------------------------------------------------------------
# 10. register() — plugin-side metadata
# ---------------------------------------------------------------------------

def test_register_calls_register_platform():
    ctx = MagicMock()
    register(ctx)
    ctx.register_platform.assert_called_once()
    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "simplex"
    assert kwargs["label"] == "SimpleX Chat"
    assert kwargs["required_env"] == ["SIMPLEX_WS_URL"]
    assert kwargs["allowed_users_env"] == "SIMPLEX_ALLOWED_USERS"
    assert kwargs["allow_all_env"] == "SIMPLEX_ALLOW_ALL_USERS"
    assert kwargs["cron_deliver_env_var"] == "SIMPLEX_HOME_CHANNEL"
    assert callable(kwargs["check_fn"])
    assert callable(kwargs["validate_config"])
    assert callable(kwargs["is_connected"])
    assert callable(kwargs["env_enablement_fn"])
    assert callable(kwargs["standalone_sender_fn"])
    assert callable(kwargs["adapter_factory"])
    assert callable(kwargs["setup_fn"])
    # SimpleX uses opaque IDs only — no PII to redact.
    assert kwargs["pii_safe"] is True
