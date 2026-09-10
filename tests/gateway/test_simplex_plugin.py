"""Tests for the SimpleX Chat platform-plugin adapter.

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_simplex`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests on the same xdist worker.
"""

from __future__ import annotations

import asyncio
import json
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
_is_image_ext = _simplex._is_image_ext
_is_audio_ext = _simplex._is_audio_ext
_CORR_PREFIX = _simplex._CORR_PREFIX


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


def test_validate_config_uses_env_or_extra():
    from gateway.config import PlatformConfig
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


def test_env_enablement_seeds_home_channel(monkeypatch):
    monkeypatch.setenv("SIMPLEX_WS_URL", "ws://127.0.0.1:5225")
    monkeypatch.setenv("SIMPLEX_HOME_CHANNEL", "42")
    monkeypatch.setenv("SIMPLEX_HOME_CHANNEL_NAME", "Personal")
    seed = _env_enablement()
    assert seed["home_channel"] == {"chat_id": "42", "name": "Personal"}


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


# ---------------------------------------------------------------------------
# 6. Correlation IDs
# ---------------------------------------------------------------------------


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
    """DMs use the structured ``/_send @<id> json [...]`` form.

    The bare ``@<id> text`` chat-command form is unreliable — the
    daemon silently drops messages when it cannot resolve the display
    name.  The structured ``/_send`` form addresses by ID and
    survives newlines/quoting through ``json.dumps``, matching what
    ``send_image`` and ``send_document`` already do.
    """
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    mock_ws = AsyncMock()
    adapter._ws = mock_ws

    result = await adapter.send("contact-42", "Hello, SimpleX!")
    mock_ws.send.assert_called_once()
    payload = json.loads(mock_ws.send.call_args[0][0])
    assert payload["cmd"].startswith("/_send @contact-42 json ")
    msg_content = json.loads(payload["cmd"].split(" json ", 1)[1])[0][
        "msgContent"
    ]
    assert msg_content == {"type": "text", "text": "Hello, SimpleX!"}
    assert payload["corrId"].startswith(_CORR_PREFIX)
    assert result.success is True



@pytest.mark.asyncio
async def test_send_group():
    """Groups use the structured ``/_send #<id> json [...]`` form.

    The bracket chat-command form ``#[<id>] text`` *looks* like an exact
    ID match in the daemon docs but is parsed as a display-name lookup
    — so messages to groups whose display name isn't literally the ID
    silently drop. The structured ``/_send`` form addresses by numeric
    ID and survives newlines/quoting through ``json.dumps``.
    """
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    mock_ws = AsyncMock()
    adapter._ws = mock_ws

    result = await adapter.send("group:grp-99", "Hello, group!")
    payload = json.loads(mock_ws.send.call_args[0][0])
    assert payload["cmd"].startswith("/_send #grp-99 json ")
    msg_content = json.loads(payload["cmd"].split(" json ", 1)[1])[0][
        "msgContent"
    ]
    assert msg_content == {"type": "text", "text": "Hello, group!"}
    assert result.success is True


# ---------------------------------------------------------------------------
# 7b. Channel directory enumeration (list_channels)
# ---------------------------------------------------------------------------


def _adapter_with_ws():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._ws = AsyncMock()
    return adapter


@pytest.mark.asyncio
async def test_list_channels_contacts_and_groups():
    adapter = _adapter_with_ws()

    async def fake_send_command(command, timeout=30.0):
        if command == "/contacts":
            return {
                "contacts": [
                    {"contactId": 1, "localDisplayName": "alice"},
                    {"contactId": 2, "profile": {"displayName": "bob"}},
                    "garbage",
                ]
            }
        if command == "/groups":
            return {
                "groups": [
                    {"groupId": 7, "localDisplayName": "friends"},
                    # [groupInfo, groupSummary] pair form
                    [{"groupId": 9, "groupProfile": {"displayName": "work"}}, {}],
                ]
            }
        return None

    adapter._send_command = fake_send_command
    channels = await adapter.list_channels()

    assert {"id": "alice", "name": "alice", "type": "dm"} in channels
    assert {"id": "bob", "name": "bob", "type": "dm"} in channels
    assert {"id": "group:7", "name": "friends", "type": "group"} in channels
    assert {"id": "group:9", "name": "work", "type": "group"} in channels


@pytest.mark.asyncio
async def test_list_channels_returns_none_when_disconnected():
    """None (not []) so the directory falls back to session discovery."""
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    assert adapter._ws is None
    assert await adapter.list_channels() is None


@pytest.mark.asyncio
async def test_list_channels_returns_none_on_contacts_timeout():
    adapter = _adapter_with_ws()

    async def fake_send_command(command, timeout=30.0):
        return None  # daemon unresponsive

    adapter._send_command = fake_send_command
    assert await adapter.list_channels() is None


# ---------------------------------------------------------------------------
# 8. Inbound: filter own-echo by corrId prefix
# ---------------------------------------------------------------------------


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
async def test_standalone_send_defaults_to_local_daemon(monkeypatch):
    monkeypatch.delenv("SIMPLEX_WS_URL", raising=False)
    pconfig = MagicMock()
    pconfig.extra = {}

    sent_payloads = []

    class DummyWs:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def send(self, payload):
            sent_payloads.append(json.loads(payload))

    def fake_connect(url, **kwargs):
        assert url == "ws://127.0.0.1:5225"
        assert kwargs["open_timeout"] == 10
        assert kwargs["close_timeout"] == 5
        return DummyWs()

    import websockets
    monkeypatch.setattr(websockets, "connect", fake_connect)

    result = await _standalone_send(pconfig, "contact-42", "hi")
    assert result == {"success": True, "platform": "simplex", "chat_id": "contact-42"}
    assert sent_payloads[0]["cmd"].startswith("/_send @contact-42 json ")
    msg_content = json.loads(
        sent_payloads[0]["cmd"].split(" json ", 1)[1]
    )[0]["msgContent"]
    assert msg_content == {"type": "text", "text": "hi"}


@pytest.mark.asyncio
async def test_contact_accept_does_not_block_event_listener_for_response():
    adapter = _adapter_with_ws()
    adapter.auto_accept = True
    adapter._send_fire_and_forget = AsyncMock()

    await adapter._on_contact_request({"contactRequest": {"contactRequestId": 7}})

    adapter._send_fire_and_forget.assert_awaited_once_with("/accept 7")


@pytest.mark.asyncio
async def test_health_monitor_does_not_reconnect_quiet_healthy_ws(monkeypatch):
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._running = True
    adapter._last_ws_activity = 0
    adapter._ws = AsyncMock()
    adapter._run_health_check = AsyncMock(return_value=True)

    monkeypatch.setattr(_simplex, "HEALTH_CHECK_INTERVAL", 0.01)
    monkeypatch.setattr(_simplex, "HEALTH_CHECK_STALE_THRESHOLD", 0.01)

    task = asyncio.create_task(adapter._health_monitor())
    await asyncio.sleep(0.03)
    adapter._running = False
    await asyncio.wait_for(task, timeout=1)

    adapter._run_health_check.assert_awaited()
    adapter._ws.close.assert_not_called()


@pytest.mark.asyncio
async def test_health_monitor_reconnects_when_daemon_probe_fails(monkeypatch):
    adapter = _adapter_with_ws()
    adapter._running = True
    adapter._run_health_check = AsyncMock(return_value=False)

    monkeypatch.setattr(_simplex, "HEALTH_CHECK_INTERVAL", 0.01)

    task = asyncio.create_task(adapter._health_monitor())
    await asyncio.sleep(0.03)
    adapter._running = False
    await asyncio.wait_for(task, timeout=1)

    adapter._run_health_check.assert_awaited()
    adapter._ws.close.assert_awaited()


@pytest.mark.asyncio
async def test_stale_health_probe_does_not_close_reconnected_socket(monkeypatch):
    adapter = _adapter_with_ws()
    adapter._running = True
    old_ws = adapter._ws
    new_ws = AsyncMock()

    async def stale_probe():
        adapter._ws = new_ws
        adapter._running = False
        return False

    adapter._run_health_check = stale_probe
    monkeypatch.setattr(_simplex, "HEALTH_CHECK_INTERVAL", 0.01)

    await adapter._health_monitor()

    old_ws.close.assert_not_awaited()
    new_ws.close.assert_not_awaited()


@pytest.mark.asyncio
async def test_health_check_replays_inbound_item_missing_from_live_ws():
    adapter = _adapter_with_ws()
    replayed_item = {
        "chatInfo": {
            "type": "direct",
            "contact": {"contactId": 4, "localDisplayName": "tester"},
        },
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "meta": {"itemId": 11},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {"type": "text", "text": "missed"},
            },
        },
    }
    commands = []

    async def fake_send_command(command, timeout=30.0):
        commands.append(command)
        return {"type": "chatItems", "chatItems": [replayed_item]}

    adapter._send_command = fake_send_command
    adapter._health_item_highwater = 10
    adapter._handle_chat_item = AsyncMock()

    healthy = await adapter._run_health_check()

    assert healthy is True
    adapter._handle_chat_item.assert_awaited_once_with(replayed_item, replay=True)
    assert adapter._health_item_highwater == 11
    assert commands == ["/_get items after=10 count=100"]


@pytest.mark.asyncio
async def test_health_cursor_is_primed_before_live_listener_starts():
    adapter = _adapter_with_ws()
    adapter._send_command = AsyncMock(
        return_value={
            "type": "chatItems",
            "chatItems": [{"chatItem": {"meta": {"itemId": 41}}}],
        }
    )

    assert await adapter._prime_health_cursor() is True
    assert adapter._health_item_highwater == 41
    adapter._send_command.assert_awaited_once_with(
        "/_get items count=1", timeout=10.0
    )


@pytest.mark.asyncio
async def test_health_cursor_rejects_malformed_chat_items():
    adapter = _adapter_with_ws()
    adapter._send_command = AsyncMock(
        return_value={"type": "chatItems", "chatItems": None}
    )

    assert await adapter._prime_health_cursor() is False
    assert adapter._health_item_highwater is None
    adapter._ws.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_health_cursor_stays_before_item_that_failed_during_priming():
    adapter = _adapter_with_ws()
    adapter._retry_chat_items.add(41)
    adapter._send_command = AsyncMock(
        return_value={
            "type": "chatItems",
            "chatItems": [{"chatItem": {"meta": {"itemId": 41}}}],
        }
    )

    assert await adapter._prime_health_cursor() is True
    assert adapter._health_item_highwater == 40


@pytest.mark.parametrize(
    "malformed",
    [
        {"chatItem": {"meta": {"itemId": True}}},
        {"chatItem": []},
        {"chatItem": {"meta": []}},
    ],
)
@pytest.mark.asyncio
async def test_health_page_rejects_malformed_item_before_advancing_cursor(malformed):
    adapter = _adapter_with_ws()
    adapter._health_item_highwater = 10
    adapter._handle_chat_item = AsyncMock()
    adapter._send_command = AsyncMock(
        return_value={
            "type": "chatItems",
            "chatItems": [
                {"chatItem": {"meta": {"itemId": 11}}},
                malformed,
                {"chatItem": {"meta": {"itemId": 12}}},
            ],
        }
    )

    assert await adapter._run_health_check() is False
    assert adapter._health_item_highwater == 10
    adapter._handle_chat_item.assert_not_awaited()


@pytest.mark.asyncio
async def test_health_page_does_not_advance_past_missing_retry_item():
    adapter = _adapter_with_ws()
    adapter._health_item_highwater = 9
    adapter._retry_chat_items.add(10)
    adapter._handle_chat_item = AsyncMock()
    adapter._send_command = AsyncMock(
        return_value={
            "type": "chatItems",
            "chatItems": [{"chatItem": {"meta": {"itemId": 11}}}],
        }
    )

    assert await adapter._run_health_check() is False
    assert adapter._health_item_highwater == 9
    adapter._handle_chat_item.assert_not_awaited()


@pytest.mark.asyncio
async def test_concurrent_retry_rewind_is_not_overwritten_by_replay_success():
    adapter = _adapter_with_ws()
    adapter._health_item_highwater = 10
    item = {"chatItem": {"meta": {"itemId": 11}}}
    adapter._send_command = AsyncMock(
        return_value={"type": "chatItems", "chatItems": [item]}
    )

    async def deliver_while_older_item_fails(chat_item, replay=False):
        adapter._retry_chat_items.add(10)
        adapter._rewind_health_cursor(10)
        return True

    adapter._handle_chat_item = AsyncMock(side_effect=deliver_while_older_item_fails)

    assert await adapter._run_health_check() is True
    assert adapter._health_item_highwater == 9
    assert adapter._retry_chat_items == {10}


def test_seen_items_above_stalled_cursor_are_not_evicted():
    adapter = _adapter_with_ws()
    adapter._health_item_highwater = 0

    for item_id in range(1, 2003):
        adapter._mark_chat_item_delivered(item_id)

    assert 1 in adapter._seen_chat_items
    assert 2002 in adapter._seen_chat_items

    adapter._max_recent_reconciled_seen_items = 2
    adapter._health_item_highwater = 1000
    adapter._prune_reconciled_seen_items()
    assert 1 not in adapter._seen_chat_items
    assert 998 not in adapter._seen_chat_items
    assert 999 in adapter._seen_chat_items
    assert 1000 in adapter._seen_chat_items
    assert 1001 in adapter._seen_chat_items


@pytest.mark.asyncio
async def test_live_delivery_is_deferred_when_unreconciled_state_is_full():
    adapter = _adapter_with_ws()
    adapter._health_item_highwater = None
    adapter._max_unreconciled_seen_items = 2
    adapter._seen_chat_items.update({1, 2})
    adapter.handle_message = AsyncMock()
    item = {
        "chatInfo": {
            "type": "direct",
            "contact": {"contactId": 4, "localDisplayName": "tester"},
        },
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "meta": {"itemId": 3},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {"type": "text", "text": "later"},
            },
        },
    }

    assert await adapter._handle_chat_item(item) is False
    adapter.handle_message.assert_not_awaited()
    assert adapter._seen_chat_items == {1, 2}
    assert adapter._preprime_deferred_item == 3

    adapter._send_command = AsyncMock(
        return_value={"type": "chatItems", "chatItems": [item]}
    )
    assert await adapter._prime_health_cursor() is True
    assert adapter._health_item_highwater == 2
    assert adapter._preprime_deferred_item is None


@pytest.mark.parametrize("invalid_id", [None, True, 0, -1])
@pytest.mark.asyncio
async def test_live_delivery_rejects_invalid_item_id(invalid_id):
    adapter = _adapter_with_ws()
    adapter.handle_message = AsyncMock()
    item = {
        "chatInfo": {
            "type": "direct",
            "contact": {"contactId": 4, "localDisplayName": "tester"},
        },
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "meta": {"itemId": invalid_id},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {"type": "text", "text": "invalid"},
            },
        },
    }

    assert await adapter._handle_chat_item(item) is False
    adapter.handle_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_health_check_paginates_without_skipping_large_gap():
    adapter = _adapter_with_ws()
    adapter._health_item_highwater = 10
    adapter._handle_chat_item = AsyncMock()

    def item(item_id):
        return {"chatItem": {"meta": {"itemId": item_id}}}

    responses = iter(
        [
            {"type": "chatItems", "chatItems": [item(i) for i in range(11, 111)]},
            {"type": "chatItems", "chatItems": [item(111)]},
        ]
    )
    commands = []

    async def fake_send_command(command, timeout=30.0):
        commands.append(command)
        return next(responses)

    adapter._send_command = fake_send_command

    assert await adapter._run_health_check() is True
    assert adapter._handle_chat_item.await_count == 101
    assert adapter._health_item_highwater == 111
    assert commands == [
        "/_get items after=10 count=100",
        "/_get items after=110 count=100",
    ]


@pytest.mark.asyncio
async def test_health_replay_does_not_duplicate_item_seen_live():
    adapter = _adapter_with_ws()
    adapter._text_batch_delay = 0.01
    adapter.handle_message = AsyncMock()
    item = {
        "chatInfo": {
            "type": "direct",
            "contact": {"contactId": 4, "localDisplayName": "tester"},
        },
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "meta": {"itemId": 11, "itemTs": "2026-09-10T08:00:00Z"},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {"type": "text", "text": "missed"},
            },
        },
    }

    await adapter._handle_chat_item(item)
    await adapter._handle_chat_item(item)
    await asyncio.sleep(0.03)

    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args is not None
    assert adapter.handle_message.await_args.args[0].text == "missed"


@pytest.mark.asyncio
async def test_delayed_live_event_does_not_duplicate_replayed_item():
    adapter = _adapter_with_ws()
    adapter._health_item_highwater = 10
    adapter.handle_message = AsyncMock()
    item = {
        "chatInfo": {
            "type": "direct",
            "contact": {"contactId": 4, "localDisplayName": "tester"},
        },
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "meta": {"itemId": 11},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {"type": "text", "text": "replayed"},
            },
        },
    }
    adapter._send_command = AsyncMock(
        return_value={"type": "chatItems", "chatItems": [item]}
    )

    assert await adapter._run_health_check() is True
    assert adapter._health_item_highwater == 11
    assert 11 in adapter._seen_chat_items

    assert await adapter._handle_chat_item(item) is True
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancelled_text_flush_does_not_mark_undelivered_item_seen():
    adapter = _adapter_with_ws()
    adapter._text_batch_delay = 0
    first_started = asyncio.Event()

    async def deliver(event):
        if event.text == "first":
            first_started.set()
            await asyncio.Event().wait()

    adapter.handle_message = AsyncMock(side_effect=deliver)

    def text_item(item_id, text):
        return {
            "chatInfo": {
                "type": "direct",
                "contact": {"contactId": 4, "localDisplayName": "tester"},
            },
            "chatItem": {
                "chatDir": {"type": "directRcv"},
                "meta": {"itemId": item_id},
                "content": {
                    "type": "rcvMsgContent",
                    "msgContent": {"type": "text", "text": text},
                },
            },
        }

    await adapter._handle_chat_item(text_item(11, "first"))
    await asyncio.wait_for(first_started.wait(), timeout=1)
    await adapter._handle_chat_item(text_item(12, "second"))
    await asyncio.sleep(0.03)

    assert 11 not in adapter._seen_chat_items
    assert 11 in adapter._retry_chat_items
    assert 12 in adapter._seen_chat_items


@pytest.mark.asyncio
async def test_health_cursor_waits_for_pending_live_text_batch():
    adapter = _adapter_with_ws()
    adapter._text_batch_delay = 0.01
    adapter.handle_message = AsyncMock()
    item = {
        "chatInfo": {
            "type": "direct",
            "contact": {"contactId": 4, "localDisplayName": "tester"},
        },
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "meta": {"itemId": 11, "itemTs": "2026-09-10T08:00:00Z"},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {"type": "text", "text": "pending"},
            },
        },
    }
    adapter._health_item_highwater = 10
    adapter._send_command = AsyncMock(
        return_value={"type": "chatItems", "chatItems": [item]}
    )

    assert await adapter._handle_chat_item(item) is False
    assert await adapter._run_health_check() is True
    assert adapter._health_item_highwater == 10

    await asyncio.sleep(0.03)
    assert 11 in adapter._seen_chat_items
    assert await adapter._run_health_check() is True
    assert adapter._health_item_highwater == 11
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_deferred_voice_item_is_not_deduplicated_before_delivery(tmp_path):
    adapter = _adapter_with_ws()
    adapter.handle_message = AsyncMock()
    adapter._send_fire_and_forget = AsyncMock()
    pending = {
        "chatInfo": {
            "type": "direct",
            "contact": {"contactId": 4, "localDisplayName": "tester"},
        },
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "meta": {"itemId": 11, "itemTs": "2026-09-10T08:00:00Z"},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {"type": "voice", "text": ""},
            },
            "file": {"fileId": 7, "fileName": "voice.ogg"},
        },
    }

    await adapter._handle_chat_item(pending)
    assert 11 not in adapter._seen_chat_items
    adapter._send_fire_and_forget.reset_mock()
    assert await adapter._handle_chat_item(pending, replay=True) is False
    adapter._send_fire_and_forget.assert_awaited_once_with("/freceive 7")

    voice_path = tmp_path / "voice.ogg"
    voice_path.write_bytes(b"voice")
    completed = {
        "chatItem": {
            "chatItem": {
                "file": {
                    "fileId": 7,
                    "fileSource": {"filePath": str(voice_path)},
                }
            }
        }
    }
    await adapter._on_rcv_file_complete(completed)

    adapter.handle_message.assert_awaited_once()
    assert 11 in adapter._seen_chat_items


@pytest.mark.asyncio
async def test_health_replay_finishes_voice_when_completion_event_was_missed(tmp_path):
    adapter = _adapter_with_ws()
    adapter._health_item_highwater = 10
    delivery_started = asyncio.Event()
    release_delivery = asyncio.Event()

    async def deliver(_event):
        delivery_started.set()
        await release_delivery.wait()

    adapter.handle_message = AsyncMock(side_effect=deliver)
    pending = {
        "chatInfo": {
            "type": "direct",
            "contact": {"contactId": 4, "localDisplayName": "tester"},
        },
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "meta": {"itemId": 11, "itemTs": "2026-09-10T08:00:00Z"},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {"type": "voice", "text": ""},
            },
            "file": {"fileId": 7, "fileName": "voice.ogg"},
        },
    }
    assert await adapter._handle_chat_item(pending) is False

    voice_path = tmp_path / "voice.ogg"
    voice_path.write_bytes(b"voice")
    ready = {
        **pending,
        "chatItem": {
            **pending["chatItem"],
            "file": {
                "fileId": 7,
                "fileName": "voice.ogg",
                "fileSource": {"filePath": str(voice_path)},
            },
        },
    }
    adapter._send_command = AsyncMock(
        return_value={"type": "chatItems", "chatItems": [ready]}
    )

    health_task = asyncio.create_task(adapter._run_health_check())
    await asyncio.wait_for(delivery_started.wait(), timeout=1)
    assert 7 not in adapter._pending_file_transfers

    completed = {
        "chatItem": {
            "chatItem": {
                "file": {
                    "fileId": 7,
                    "fileSource": {"filePath": str(voice_path)},
                }
            }
        }
    }
    await adapter._on_rcv_file_complete(completed)
    adapter.handle_message.assert_awaited_once()

    release_delivery.set()
    assert await asyncio.wait_for(health_task, timeout=1) is True
    assert adapter._health_item_highwater == 11
    adapter.handle_message.assert_awaited_once()
    assert 11 not in adapter._pending_chat_items

    await adapter._on_rcv_file_complete(completed)
    adapter.handle_message.assert_awaited_once()
    assert 11 not in adapter._retry_chat_items


@pytest.mark.asyncio
async def test_voice_completion_owns_delivery_before_ready_health_replay(tmp_path):
    adapter = _adapter_with_ws()
    delivery_started = asyncio.Event()
    release_delivery = asyncio.Event()

    async def deliver(_event):
        delivery_started.set()
        await release_delivery.wait()

    adapter.handle_message = AsyncMock(side_effect=deliver)
    pending = {
        "chatInfo": {
            "type": "direct",
            "contact": {"contactId": 4, "localDisplayName": "tester"},
        },
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "meta": {"itemId": 11},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {"type": "voice", "text": ""},
            },
            "file": {"fileId": 7, "fileName": "voice.ogg"},
        },
    }
    assert await adapter._handle_chat_item(pending) is False

    voice_path = tmp_path / "voice.ogg"
    voice_path.write_bytes(b"voice")
    completed = {
        "chatItem": {
            "chatItem": {
                "file": {
                    "fileId": 7,
                    "fileSource": {"filePath": str(voice_path)},
                }
            }
        }
    }
    completion_task = asyncio.create_task(adapter._on_rcv_file_complete(completed))
    await asyncio.wait_for(delivery_started.wait(), timeout=1)

    ready = {
        **pending,
        "chatItem": {
            **pending["chatItem"],
            "file": {
                "fileId": 7,
                "fileName": "voice.ogg",
                "fileSource": {"filePath": str(voice_path)},
            },
        },
    }
    assert await adapter._handle_chat_item(ready, replay=True) is False
    adapter.handle_message.assert_awaited_once()

    release_delivery.set()
    await asyncio.wait_for(completion_task, timeout=1)
    adapter.handle_message.assert_awaited_once()


@pytest.mark.asyncio
async def test_failed_nontext_delivery_is_retryable(tmp_path):
    adapter = _adapter_with_ws()
    adapter._health_item_highwater = 12
    adapter.handle_message = AsyncMock(side_effect=[RuntimeError("temporary"), None])
    voice_path = tmp_path / "voice.ogg"
    voice_path.write_bytes(b"voice")
    item = {
        "chatInfo": {
            "type": "direct",
            "contact": {"contactId": 4, "localDisplayName": "tester"},
        },
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "meta": {"itemId": 12, "itemTs": "2026-09-10T08:00:01Z"},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {"type": "voice", "text": ""},
            },
            "file": {
                "fileId": 8,
                "fileName": "voice.ogg",
                "fileSource": {"filePath": str(voice_path)},
            },
        },
    }

    with pytest.raises(RuntimeError, match="temporary"):
        await adapter._handle_chat_item(item)
    assert 12 not in adapter._seen_chat_items
    assert 12 in adapter._retry_chat_items
    assert adapter._health_item_highwater == 11

    await adapter._handle_chat_item(item)
    assert adapter.handle_message.await_count == 2
    assert 12 in adapter._seen_chat_items




# ---------------------------------------------------------------------------
# 10. register() — plugin-side metadata
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Inbound attachment message type classification
# ---------------------------------------------------------------------------

def _make_file_chat_item(file_path: str, file_name: str) -> dict:
    """Minimal direct-chat rcvMsgContent item carrying a completed file."""
    return {
        "chatInfo": {
            "type": "direct",
            "contact": {"contactId": 42, "localDisplayName": "tester"},
        },
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "meta": {"itemTs": "2026-01-01T00:00:00Z"},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {"type": "file", "text": "here you go"},
            },
            "file": {
                "fileId": 7,
                "fileName": file_name,
                "fileSource": {"filePath": file_path},
            },
        },
    }




# ---------------------------------------------------------------------------
# Multiplex secondary-profile scope
# ---------------------------------------------------------------------------
#
# Every SIMPLEX_* read (auto_accept / group_allowed in __init__, ws_url in the
# registry gates, everything in _env_enablement) went through raw os.getenv,
# which under multiplexing holds the DEFAULT profile's YAML-to-env bridge
# output -- a secondary profile silently borrowed the default's daemon URL,
# group allowlist and auto-accept setting. Reads now go through the module's
# ``_get_scoped_secret`` (profile .env AND extra both honored; scoped miss
# fails closed; unscoped default profile keeps env precedence).


@pytest.fixture
def multiplex_scope():
    """Install multiplex + a secondary-profile secret scope; restore after."""
    from agent.secret_scope import (
        reset_secret_scope,
        set_multiplex_active,
        set_secret_scope,
    )

    tokens = []

    def install(scope=None):
        set_multiplex_active(True)
        tokens.append(set_secret_scope(scope or {}))

    yield install
    for token in reversed(tokens):
        reset_secret_scope(token)
    set_multiplex_active(False)


@pytest.fixture
def default_profile_env(monkeypatch):
    """The default profile's YAML-to-env bridge output in os.environ."""
    monkeypatch.setenv("SIMPLEX_WS_URL", "ws://default:5225")
    monkeypatch.setenv("SIMPLEX_GROUP_ALLOWED", "*")
    monkeypatch.setenv("SIMPLEX_AUTO_ACCEPT", "true")


def test_multiplex_scoped_miss_does_not_borrow_default_profile_env(
    multiplex_scope, default_profile_env
):
    """A secondary profile with no SimpleX config of its own must not be
    auto-enabled off the default's daemon URL, nor inherit its wide-open
    group allowlist."""
    from gateway.config import PlatformConfig

    multiplex_scope({"SOMETHING_ELSE": "x"})
    assert _env_enablement() is None
    assert check_requirements() is False
    assert is_connected(PlatformConfig(enabled=True, extra={})) is False
    adapter = SimplexAdapter(PlatformConfig(enabled=True, extra={"auto_accept": False}))
    assert adapter.group_allow_from == set()
    assert adapter.auto_accept is False


def test_multiplex_scope_reads_profile_own_env_not_default(
    multiplex_scope, default_profile_env
):
    """A secondary profile's own .env (installed as the scope) is honored --
    the extra-only shape would have ignored it."""
    multiplex_scope({"SIMPLEX_WS_URL": "ws://profile:5225", "SIMPLEX_GROUP_ALLOWED": "g1"})
    seeded = _env_enablement()
    assert seeded == {"ws_url": "ws://profile:5225", "group_allowed": "g1"}
    assert check_requirements() is True
