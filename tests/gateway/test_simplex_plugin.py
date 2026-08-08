"""Tests for the SimpleX Chat platform-plugin adapter.

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_simplex`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests on the same xdist worker.
"""

from __future__ import annotations

import asyncio
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
# 5. Helper functions (magic-byte detection)
# ---------------------------------------------------------------------------

def test_guess_extension_png():
    assert _guess_extension(b"\x89PNG\r\n\x1a\n") == ".png"


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
async def test_health_monitor_does_not_reconnect_quiet_healthy_ws(monkeypatch):
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._running = True
    adapter._last_ws_activity = 0
    adapter._ws = AsyncMock()

    monkeypatch.setattr(_simplex, "HEALTH_CHECK_INTERVAL", 0.01)
    monkeypatch.setattr(_simplex, "HEALTH_CHECK_STALE_THRESHOLD", 0.01)

    task = asyncio.create_task(adapter._health_monitor())
    await asyncio.sleep(0.03)
    adapter._running = False
    await asyncio.wait_for(task, timeout=1)

    adapter._ws.close.assert_not_called()




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
# Inbound media: fresh-path /freceive, defer-without-accept, completion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rcv_file_descr_ready_sends_freceive_with_fresh_path():
    """rcvFileDescrReady must trigger /freceive with approved_relays=on AND
    a fresh non-existing path (uuid-prefixed, original filename preserved).

    Without a path the daemon parks the transfer at rcvTransfer and
    rcvFileComplete never fires; with an existing path it answers
    CEFileAlreadyExists. The uuid prefix guarantees the path does not exist
    (regression for the v7 media stall)."""
    from gateway.config import PlatformConfig

    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    sent_cmds = []

    async def _fake_send_fire_and_forget(cmd):
        sent_cmds.append(cmd)

    adapter._send_fire_and_forget = _fake_send_fire_and_forget

    resp = {
        "type": "rcvFileDescrReady",
        "rcvFileTransfer": {"fileId": 777, "fileName": "holiday.jpg"},
    }
    await adapter._handle_event(resp)

    assert len(sent_cmds) == 1
    cmd = sent_cmds[0]
    assert cmd.startswith("/freceive 777 approved_relays=on /tmp/simplex-rcv-")
    assert cmd.endswith("-holiday.jpg")
    path = cmd.split()[-1]
    # The path must not exist at accept time (fresh-path requirement).
    assert not os.path.exists(path)
    # Sanitization: weird file names must keep only safe chars.
    resp["rcvFileTransfer"]["fileName"] = "../../evil name!.png"
    await adapter._handle_event(resp)
    path2 = sent_cmds[1].split()[-1]
    assert path2.endswith("-evil_name_.png")
    assert "/../" not in path2
    assert path2.startswith("/tmp/simplex-rcv-")

    # Voice: the daemon carries fileName in chatItem.chatItem.file, not in
    # rcvFileTransfer — the fresh path must still end with the .m4a name or
    # voice transcription fails on the extension.
    resp = {
        "type": "rcvFileDescrReady",
        "rcvFileTransfer": {"fileId": 888},
        "chatItem": {
            "chatItem": {
                "file": {"fileId": 888, "fileName": "voice_note.m4a"},
            }
        },
    }
    await adapter._handle_event(resp)
    path3 = sent_cmds[2].split()[-1]
    assert path3.endswith("-voice_note.m4a"), path3


@pytest.mark.asyncio
async def test_deferred_file_is_parked_no_freceive():
    """A deferred inbound file must be parked in _pending_file_transfers
    WITHOUT sending /freceive.

    Accepting before the file description is ready (rcvFileDescrReady) parks
    the transfer as rcvAccepted and the subsequent descrReady accept then
    fails with CEFileAlreadyReceiving — the download never starts. Accept
    exactly once, from the rcvFileDescrReady handler, like the official
    clients do."""
    from gateway.config import PlatformConfig

    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    dispatched = []

    async def _capture(event):
        dispatched.append(event)

    adapter.handle_message = _capture

    sent_cmds = []

    async def _fake_send_fire_and_forget(cmd):
        sent_cmds.append(cmd)

    adapter._send_fire_and_forget = _fake_send_fire_and_forget

    item = {
        "chatInfo": {
            "type": "direct",
            "contact": {"contactId": 42, "localDisplayName": "tester"},
        },
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "meta": {"itemTs": "2026-01-01T00:00:00Z"},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {"type": "image", "text": ""},
            },
            "file": {
                "fileId": 99,
                "fileName": "pic.jpg",
                "fileSource": {},
            },
        },
    }
    await adapter._handle_chat_item(item)

    assert dispatched == []
    assert 99 in adapter._pending_file_transfers
    assert sent_cmds == [], "deferred file must not send /freceive"


@pytest.mark.asyncio
async def test_rcv_file_complete_marks_status_complete_and_dispatches(tmp_path):
    """rcvFileComplete must mark the parked item's fileStatus as rcvComplete
    before re-dispatching — otherwise _handle_chat_item re-defers it forever
    and the media never reaches the agent."""
    from gateway.config import PlatformConfig
    from gateway.platforms.base import MessageType

    p = tmp_path / "simplex-rcv-deadbeef-holiday.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)

    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    dispatched = []

    async def _capture(event):
        dispatched.append(event)

    adapter.handle_message = _capture

    # Parked item first (fileStatus rcvInvitation, no path).
    parked = {
        "chatInfo": {
            "type": "direct",
            "contact": {"contactId": 42, "localDisplayName": "tester"},
        },
        "chatItem": {
            "chatDir": {"type": "directRcv"},
            "meta": {"itemTs": "2026-01-01T00:00:00Z"},
            "content": {
                "type": "rcvMsgContent",
                "msgContent": {"type": "image", "text": ""},
            },
            "file": {
                "fileId": 777,
                "fileName": "holiday.jpg",
                "fileStatus": {"type": "rcvInvitation"},
                "fileSource": {},
            },
        },
    }
    adapter._pending_file_transfers[777] = parked

    resp = {
        "type": "rcvFileComplete",
        "chatItem": {
            "chatItem": {
                "file": {
                    "fileId": 777,
                    "fileName": "holiday.jpg",
                    "fileSource": {"filePath": str(p)},
                }
            }
        },
    }
    await adapter._handle_event(resp)

    assert dispatched, "rcvFileComplete must dispatch the deferred media"
    assert dispatched[0].message_type == MessageType.PHOTO


@pytest.mark.asyncio
async def test_classify_temp_xftp_as_image(tmp_path):
    """An inbound file stored under the daemon's temp name (.xftp) whose
    content is a JPEG must classify as image/jpeg — the extension of the
    temp path is not meaningful, content sniffing decides (regression for
    simplex-rcv-*.xftp attachments)."""
    from gateway.config import PlatformConfig
    from gateway.platforms.base import MessageType

    p = tmp_path / "simplex-rcv-deadbeef.xftp"
    p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)

    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    dispatched = []

    async def _capture(event):
        dispatched.append(event)

    adapter.handle_message = _capture

    item = _make_file_chat_item(str(p), "IMG_1234.jpg")
    item["chatItem"]["file"]["fileStatus"] = {"type": "rcvComplete"}
    await adapter._handle_chat_item(item)

    assert dispatched, "_handle_chat_item did not dispatch any event"
    assert dispatched[0].message_type == MessageType.PHOTO
    assert dispatched[0].media_types == ["image/jpg"]
