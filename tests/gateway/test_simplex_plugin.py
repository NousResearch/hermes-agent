"""Tests for the SimpleX Chat platform-plugin adapter.

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_simplex`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests in the same process.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path
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
        import websockets
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


@pytest.mark.parametrize(("mode", "color"), [
    ("RGBA", (255, 0, 0, 128)),
    ("LA", (128, 128)),
    ("P", 0),
])
def test_prepare_image_flattens_jpeg_incompatible_modes(tmp_path, mode, color):
    from PIL import Image

    image_path = tmp_path / f"{mode}.png"
    image = Image.new(mode, (256, 128), color)
    if mode == "P":
        image.putpalette([255, 0, 0] + [0, 0, 0] * 255)
        image.info["transparency"] = 0
    image.save(image_path)

    prepared_path, thumb_uri = SimplexAdapter._prepare_image(str(image_path))

    assert prepared_path == str(image_path)
    encoded = thumb_uri.removeprefix(_simplex._THUMB_URI_PREFIX)
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as thumbnail:
        assert thumbnail.mode == "RGB"
        assert thumbnail.size == (128, 64)


@pytest.mark.asyncio
async def test_send_image_reports_thumbnail_preparation_failure(tmp_path, monkeypatch):
    adapter = _adapter_with_ws()
    image_path = tmp_path / "broken.png"
    image_path.write_bytes(b"not an image")
    monkeypatch.setattr(adapter, "_prepare_image", MagicMock(side_effect=OSError("bad image")))
    adapter._send_items = AsyncMock()

    result = await adapter.send_image_file("contact-42", str(image_path))

    assert result.success is False
    assert result.error == "Failed to prepare image: bad image"
    adapter._send_items.assert_not_awaited()


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


@pytest.mark.asyncio
async def test_name_allowlist_warning_once_scoped_even_if_first_connect_fails(monkeypatch, caplog):
    """Name entries in SIMPLEX_ALLOWED_USERS are ignored by authz, so connect()
    warns -- exactly once per process for this profile/value, read the way authz
    reads it (the profile scope, not the default profile's os.environ), and even
    when the first connect fails. Reconnects build a FRESH adapter each attempt
    (gateway/run_adapters.py), so the dedup cannot live on the instance."""
    import logging

    import websockets
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope
    from gateway.config import PlatformConfig

    monkeypatch.setenv("SIMPLEX_ALLOWED_USERS", "bob")  # default profile's bridge output
    monkeypatch.setattr(_simplex, "_NAME_ALLOWLIST_WARNED", set())

    class DummyWs:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    attempts = []

    def fake_connect(url, **kwargs):
        attempts.append(url)
        if len(attempts) == 1:
            raise OSError("daemon down")
        return DummyWs()

    async def _idle():
        return None

    def _fresh_adapter():
        adapter = SimplexAdapter(PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"}))
        monkeypatch.setattr(adapter, "_ws_listener", _idle)
        monkeypatch.setattr(adapter, "_health_monitor", _idle)
        return adapter

    monkeypatch.setattr(websockets, "connect", fake_connect)

    # Scope set inside the test: a sync fixture's ContextVar token can't be reset from here.
    set_multiplex_active(True)
    token = set_secret_scope({"SIMPLEX_ALLOWED_USERS": "4, alice"})
    try:
        with caplog.at_level(logging.WARNING):
            assert await _fresh_adapter().connect() is False  # cold boot, daemon down
            retry = _fresh_adapter()  # the reconnect watcher builds a new adapter
            assert await retry.connect(is_reconnect=True) is True
        await retry.disconnect()
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)

    warnings = [r.getMessage() for r in caplog.records if "not numeric contactIds" in r.getMessage()]
    assert len(warnings) == 1
    assert "['alice']" in warnings[0]
    assert "bob" not in warnings[0]


# ---------------------------------------------------------------------------
# 11. Inline images: mobile clients inline the bytes in msgContent.image and
#     send NO chatItem.file transfer, so the file-path block finds nothing.
# ---------------------------------------------------------------------------

_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00\x01\x02\x03" * 16
_JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00\x01\x02\x03" * 16
_GIF_BYTES = b"GIF89a" + b"\x00\x01\x02\x03" * 16
_WEBP_BYTES = b"RIFF" + b"\x00\x00\x00\x00" + b"WEBP" + b"\x00\x01\x02\x03" * 16


def _data_uri(raw: bytes, declared: str = "image/png") -> str:
    return f"data:{declared};base64," + base64.b64encode(raw).decode()


def _inline_image_item(data_uri: str, *, item_id: int = 77, file_path: str | None = None) -> dict:
    """Inbound image whose bytes are inline in ``msgContent.image`` (no file transfer)."""
    chat_item = {
        "chatDir": {"type": "directRcv"},
        "meta": {"itemId": item_id, "itemTs": "2026-11-02T07:00:00Z"},
        "content": {"type": "rcvMsgContent", "msgContent": {"type": "image", "image": data_uri}},
    }
    if file_path is not None:
        chat_item["file"] = {"fileId": 9, "fileName": "photo.png",
                             "fileSource": {"filePath": file_path}}
    return {
        "chatInfo": {"type": "direct", "contact": {"contactId": 4, "localDisplayName": "tester"}},
        "chatItem": chat_item,
    }


def _capture_inbound(monkeypatch):
    """Adapter whose inbound events are captured instead of processed."""
    adapter = _adapter_with_ws()
    captured: list = []
    enqueued: list = []

    async def _handle(event):
        captured.append(event)

    monkeypatch.setattr(adapter, "handle_message", _handle)
    monkeypatch.setattr(adapter, "_enqueue_text_event", lambda e: enqueued.append(e))
    return adapter, captured, enqueued


@pytest.mark.asyncio
async def test_inline_image_is_decoded_and_reaches_the_event(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter, captured, _ = _capture_inbound(monkeypatch)

    await adapter._handle_chat_item(_inline_image_item(_data_uri(_PNG_BYTES)))

    assert len(captured) == 1, "an inline image must not produce an empty message"
    event = captured[0]
    from gateway.platforms.event import MessageType
    assert event.message_type is MessageType.PHOTO
    assert event.media_types == ["image/png"]
    written = Path(event.media_urls[0])
    assert written.parent == tmp_path / "cache" / "simplex-inline"
    assert written.name == "inline-77.png"
    assert written.read_bytes() == _PNG_BYTES


@pytest.mark.parametrize(
    "raw,ext,mime",
    [
        (_PNG_BYTES, ".png", "image/png"),
        (_JPEG_BYTES, ".jpg", "image/jpeg"),
        (_GIF_BYTES, ".gif", "image/gif"),
        (_WEBP_BYTES, ".webp", "image/webp"),
    ],
)
def test_format_comes_from_the_magic_bytes(monkeypatch, tmp_path, raw, ext, mime):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    decoded = _simplex._decode_inline_image(_data_uri(raw), 5)
    assert decoded is not None
    path, sniffed = decoded
    assert sniffed == mime
    assert path.endswith(f"inline-5{ext}")


def test_declared_mime_never_reaches_the_filename(monkeypatch, tmp_path):
    """``data:image/x.exe`` is attacker-controlled: the extension is sniffed, never trusted."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    decoded = _simplex._decode_inline_image(_data_uri(_PNG_BYTES, "image/x.exe"), 6)
    assert decoded is not None
    path, mime = decoded
    assert mime == "image/png"
    assert path.endswith(".png")
    assert ".exe" not in path


@pytest.mark.parametrize(
    "declared",
    ["image/svg+xml", "image/x-executable", "text/html", ""],
)
def test_declared_image_with_non_image_bytes_is_dropped(monkeypatch, tmp_path, declared):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    payload = _data_uri(b"#!/bin/sh\nrm -rf /\n" * 4, declared) if declared else _data_uri(b"#!/bin/sh\nrm -rf /\n" * 4)
    assert _simplex._decode_inline_image(payload, 7) is None
    assert not (tmp_path / "cache" / "simplex-inline").exists()


def test_oversized_payload_is_rejected_before_decoding(monkeypatch, tmp_path):
    """The cap must bite before base64 decode, or untrusted input sizes our memory."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(_simplex, "_INLINE_IMAGE_MAX_B64", 64)

    class _NeverDecode:
        @staticmethod
        def b64decode(*_a, **_k):
            raise AssertionError("decoded a payload that should have been rejected by size")

    monkeypatch.setattr(_simplex, "base64", _NeverDecode)
    assert _simplex._decode_inline_image(_data_uri(_PNG_BYTES), 8) is None


def test_decoded_size_cap_is_enforced_before_writing(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(_simplex, "_INLINE_IMAGE_MAX_BYTES", 64)
    assert len(_PNG_BYTES) > 64
    assert _simplex._decode_inline_image(_data_uri(_PNG_BYTES), 9) is None
    assert not (tmp_path / "cache" / "simplex-inline").exists()


@pytest.mark.parametrize("file_id", ["77/../../evil", "a77b", "../../etc/passwd"])
def test_filename_component_is_sanitised_to_digits(monkeypatch, tmp_path, file_id):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    decoded = _simplex._decode_inline_image(_data_uri(_PNG_BYTES), file_id)
    assert decoded is not None
    written = Path(decoded[0])
    assert written.parent == tmp_path / "cache" / "simplex-inline"
    assert written.name in {"inline-77.png", "inline-x.png"}
    assert ".." not in written.name and "/" not in written.name


def test_missing_item_id_falls_back_to_a_timestamp(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    decoded = _simplex._decode_inline_image(_data_uri(_PNG_BYTES))
    assert decoded is not None
    stem = Path(decoded[0]).stem
    assert stem.startswith("inline-") and stem[7:].isdigit()


@pytest.mark.parametrize("payload", ["", "   ", None, 42, "data:image/png;base64,%%%%", "data:image/png;base64,AAAA"])
def test_unusable_payloads_return_none(monkeypatch, tmp_path, payload):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert _simplex._decode_inline_image(payload, 10) is None


@pytest.mark.asyncio
async def test_decodable_file_transfer_wins_over_inline_bytes(monkeypatch, tmp_path):
    """A real file path is already downloaded; do not decode the inline copy over it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    on_disk = tmp_path / "photo.png"
    on_disk.write_bytes(_PNG_BYTES)
    adapter, captured, _ = _capture_inbound(monkeypatch)

    await adapter._handle_chat_item(
        _inline_image_item(_data_uri(_PNG_BYTES), file_path=str(on_disk)))

    assert captured[0].media_urls == [str(on_disk)]
    assert not (tmp_path / "cache" / "simplex-inline").exists()
