"""Tests for the SimpleX Chat platform-plugin adapter.

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_simplex`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests on the same xdist worker.
"""

from __future__ import annotations

import json
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.platforms.base import MessageType
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


class _QueueWS:
    def __init__(self, responses=None):
        self.sent = []
        self._responses = list(responses or [])

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    async def recv(self):
        if not self._responses:
            raise TimeoutError("no queued websocket response")
        return json.dumps(self._responses.pop(0))


class _RespondingWS:
    def __init__(self, adapter, response):
        self.adapter = adapter
        self.response = response
        self.sent = []

    async def send(self, payload):
        decoded = json.loads(payload)
        self.sent.append(decoded)
        await self.adapter._handle_event(
            {"corrId": decoded["corrId"], "resp": self.response}
        )


def _install_fake_websockets(monkeypatch, *, connect=None):
    class _ConnectionClosed(Exception):
        pass

    fake = types.SimpleNamespace(ConnectionClosed=_ConnectionClosed)
    if connect is not None:
        fake.connect = connect
    monkeypatch.setitem(sys.modules, "websockets", fake)
    return fake


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

def test_check_requirements_does_not_gate_config_url(monkeypatch):
    monkeypatch.delenv("SIMPLEX_WS_URL", raising=False)
    assert check_requirements() is True


def test_check_requirements_true_when_configured(monkeypatch):
    monkeypatch.setenv("SIMPLEX_WS_URL", "ws://127.0.0.1:5225")
    assert check_requirements() is True


def test_check_requirements_keeps_configured_platform_visible_without_websockets(monkeypatch):
    monkeypatch.setenv("SIMPLEX_WS_URL", "ws://127.0.0.1:5225")

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "websockets" or name.startswith("websockets."):
            raise ImportError("websockets blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert check_requirements() is True


def test_registry_can_create_adapter_with_configured_url(monkeypatch):
    """Config-yaml ws_url must not be rejected by the env-only check_fn."""
    from gateway.config import PlatformConfig
    from gateway.platform_registry import PlatformEntry, PlatformRegistry

    monkeypatch.delenv("SIMPLEX_WS_URL", raising=False)
    registry = PlatformRegistry()
    registry.register(
        PlatformEntry(
            name="simplex",
            label="SimpleX Chat",
            adapter_factory=lambda cfg: SimplexAdapter(cfg),
            check_fn=check_requirements,
            validate_config=validate_config,
        )
    )

    cfg = PlatformConfig(
        enabled=True,
        extra={"ws_url": "ws://127.0.0.1:5225"},
    )

    assert isinstance(registry.create_adapter("simplex", cfg), SimplexAdapter)


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
async def test_send_dm_uses_api_send_command(monkeypatch):
    _install_fake_websockets(monkeypatch)
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._ws = _RespondingWS(
        adapter,
        {
            "type": "newChatItems",
            "chatItems": [{"chatItem": {"meta": {"itemId": 41}}}],
        },
    )

    result = await adapter.send("42", "Hello, SimpleX!")
    payload = adapter._ws.sent[0]
    assert payload["cmd"] == (
        '/_send @42 json [{"msgContent":{"type":"text","text":"Hello, SimpleX!"},"mentions":{}}]'
    )
    assert payload["corrId"].startswith(_CORR_PREFIX)
    assert result.success is True
    assert result.message_id == "41"


@pytest.mark.asyncio
async def test_send_group_uses_api_send_command(monkeypatch):
    _install_fake_websockets(monkeypatch)
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._ws = _RespondingWS(
        adapter,
        {
            "type": "newChatItems",
            "chatItems": [{"chatItem": {"meta": {"itemId": 42}}}],
        },
    )

    result = await adapter.send("group:99", "Hello, group!")
    payload = adapter._ws.sent[0]
    assert payload["cmd"] == (
        '/_send #99 json [{"msgContent":{"type":"text","text":"Hello, group!"},"mentions":{}}]'
    )
    assert result.success is True
    assert result.message_id == "42"


@pytest.mark.asyncio
async def test_send_when_ws_not_connected_fails_loudly(monkeypatch, caplog):
    _install_fake_websockets(monkeypatch)
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    caplog.set_level("ERROR")

    result = await adapter.send("contact-42", "hi")

    assert result.success is False
    assert result.retryable is True
    assert "WebSocket not connected" in (result.error or "")
    assert "WebSocket not connected" in caplog.text


@pytest.mark.asyncio
async def test_send_streaming_uses_live_flag_and_returns_item_id(monkeypatch):
    _install_fake_websockets(monkeypatch)
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._ws = _RespondingWS(
        adapter,
        {
            "type": "newChatItems",
            "chatItems": [{"chatItem": {"meta": {"itemId": 77}}}],
        },
    )

    result = await adapter.send(
        "42",
        "**Hello** from `Hermes`",
        metadata={"_hermes_live_stream": True},
    )

    assert result.success is True
    assert result.message_id == "77"
    assert adapter._ws.sent[0]["cmd"] == (
        '/_send @42 live=on json [{"msgContent":{"type":"text","text":"*Hello* from `Hermes`"},"mentions":{}}]'
    )


@pytest.mark.asyncio
async def test_edit_message_live_update_uses_live_on(monkeypatch):
    _install_fake_websockets(monkeypatch)
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._ws = _RespondingWS(adapter, {"type": "chatItemUpdated"})

    result = await adapter.edit_message("42", "77", "**partial**", finalize=False)

    assert result.success is True
    assert adapter._ws.sent[0]["cmd"] == (
        '/_update item @42 77 live=on json {"msgContent":{"type":"text","text":"*partial*"},"mentions":{}}'
    )


@pytest.mark.asyncio
async def test_edit_message_finalize_omits_live_on(monkeypatch):
    _install_fake_websockets(monkeypatch)
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._ws = _RespondingWS(adapter, {"type": "chatItemNotChanged"})

    result = await adapter.edit_message("42", "77", "**done**", finalize=True)

    assert result.success is True
    assert adapter._ws.sent[0]["cmd"] == (
        '/_update item @42 77 json {"msgContent":{"type":"text","text":"*done*"},"mentions":{}}'
    )


def test_format_message_converts_markdown_to_simplex():
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    assert adapter.format_message("## Summary\nUse **bold** and ~~strike~~.") == (
        "*Summary*\nUse *bold* and ~strike~."
    )
    assert adapter.format_message("See [docs](https://example.com)") == (
        "See docs (https://example.com)"
    )


# ---------------------------------------------------------------------------
# 8. Inbound: filter own-echo by corrId prefix
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


@pytest.mark.asyncio
async def test_handle_new_chat_items_accepts_flat_simplex_items():
    """SimpleX v6.5 emits chatItems as flat item objects in newChatItems."""
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter.handle_message = AsyncMock()  # type: ignore[method-assign]

    await adapter._handle_event(
        {
            "type": "newChatItems",
            "chatInfo": {
                "type": "direct",
                "contact": {"contactId": 4, "localDisplayName": "Bryan"},
            },
            "chatItems": [
                {
                    "chatDir": {"type": "directRcv"},
                    "meta": {
                        "itemTs": "2026-05-26T01:00:00Z",
                        "itemStatus": {"type": "rcvNew"},
                    },
                    "content": {
                        "type": "rcvMsgContent",
                        "msgContent": {"type": "text", "text": "hello"},
                    },
                }
            ],
        }
    )

    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.text == "hello"
    assert event.source.chat_id == "4"
    assert event.source.user_id == "4"
    assert event.source.chat_type == "dm"


@pytest.mark.asyncio
async def test_handle_event_accepts_resp_envelope():
    """Live SimpleX events arrive under a top-level resp envelope."""
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter.handle_message = AsyncMock()  # type: ignore[method-assign]

    await adapter._handle_event(
        {
            "corrId": "",
            "resp": {
                "type": "newChatItems",
                "chatItems": [
                    {
                        "chatInfo": {
                            "type": "direct",
                            "contact": {"contactId": 4, "localDisplayName": "Bryan"},
                        },
                        "chatItem": {
                            "chatDir": {"type": "directRcv"},
                            "meta": {
                                "itemId": 12,
                                "itemTs": "2026-05-26T01:12:09Z",
                                "itemStatus": {"type": "rcvNew"},
                            },
                            "content": {
                                "type": "rcvMsgContent",
                                "msgContent": {"type": "text", "text": "Hey!!"},
                            },
                        },
                    }
                ],
            },
        }
    )

    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].text == "Hey!!"


@pytest.mark.asyncio
async def test_replay_unread_fetches_chat_and_dispatches_new_items(monkeypatch):
    """Reconnect replay should process unread items missed while WS was down."""
    _install_fake_websockets(monkeypatch)
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter.handle_message = AsyncMock()  # type: ignore[method-assign]
    corr_ids = iter(["corr-list", "corr-chat", "corr-read"])
    adapter._make_corr_id = lambda: next(corr_ids)  # type: ignore[method-assign]

    ws = _QueueWS(
        [
            {
                "corrId": "corr-list",
                "resp": {
                    "type": "apiChats",
                    "chats": [
                        {
                            "chatInfo": {
                                "type": "direct",
                                "contact": {"contactId": 4, "localDisplayName": "Bryan"},
                            },
                            "chatItems": [],
                            "chatStats": {"unreadCount": 1},
                        }
                    ],
                },
            },
            {
                "corrId": "corr-chat",
                "resp": {
                    "type": "apiChat",
                    "chat": {
                        "chatInfo": {
                            "type": "direct",
                            "contact": {"contactId": 4, "localDisplayName": "Bryan"},
                        },
                        "chatItems": [
                            {
                                "chatDir": {"type": "directSnd"},
                                "meta": {"itemId": 9, "itemStatus": {"type": "sndSent"}},
                                "content": {
                                    "type": "sndMsgContent",
                                    "msgContent": {"type": "text", "text": "old outbound"},
                                },
                            },
                            {
                                "chatDir": {"type": "directRcv"},
                                "meta": {
                                    "itemId": 10,
                                    "itemTs": "2026-05-26T01:12:09Z",
                                    "itemStatus": {"type": "rcvNew"},
                                },
                                "content": {
                                    "type": "rcvMsgContent",
                                    "msgContent": {"type": "text", "text": "Hey!!"},
                                },
                            },
                        ],
                        "chatStats": {"unreadCount": 1},
                    },
                },
            },
        ]
    )
    adapter._ws = ws

    await adapter._replay_unread_chats(ws)

    adapter.handle_message.assert_awaited_once()
    assert adapter.handle_message.await_args.args[0].text == "Hey!!"
    assert [payload["cmd"] for payload in ws.sent] == [
        '/_get chats 1 pcc=on count=50 {"type":"filters","favorite":false,"unread":true}',
        "/_get chat @4 count=6",
        "/_read chat items @4 10",
    ]


@pytest.mark.asyncio
async def test_handle_new_chat_item_accepts_dict_file_status():
    """SimpleX voice notes use dict-shaped fileStatus values."""
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter.handle_message = AsyncMock()  # type: ignore[method-assign]
    adapter._fetch_file = AsyncMock(return_value="/tmp/hermes-cache/voice.m4a")  # type: ignore[method-assign]

    await adapter._handle_new_chat_item(
        {
            "chatInfo": {
                "type": "direct",
                "contact": {"contactId": 4, "localDisplayName": "Bryan"},
            },
            "chatItem": {
                "chatDir": {"type": "directRcv"},
                "meta": {
                    "itemId": 11,
                    "itemTs": "2026-05-26T01:12:23Z",
                    "itemStatus": {"type": "rcvNew"},
                },
                "content": {
                    "type": "rcvMsgContent",
                    "msgContent": {"type": "voice", "text": ""},
                },
                "file": {
                    "fileId": 1,
                    "fileName": "voice_20260526_011223.m4a",
                    "fileStatus": {"type": "rcvInvitation"},
                },
            },
        }
    )

    adapter._fetch_file.assert_awaited_once_with(1, "voice_20260526_011223.m4a")
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.media_urls == ["/tmp/hermes-cache/voice.m4a"]
    assert event.media_types == ["audio/m4a"]
    assert event.message_type == MessageType.VOICE


@pytest.mark.asyncio
async def test_handle_new_chat_item_keeps_plain_audio_as_audio():
    """Audio file attachments should not be treated as voice notes unless msgContent says voice."""
    from gateway.config import PlatformConfig
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter.handle_message = AsyncMock()  # type: ignore[method-assign]
    adapter._fetch_file = AsyncMock(return_value="/tmp/hermes-cache/clip.m4a")  # type: ignore[method-assign]

    await adapter._handle_new_chat_item(
        {
            "chatInfo": {
                "type": "direct",
                "contact": {"contactId": 4, "localDisplayName": "Bryan"},
            },
            "chatItem": {
                "chatDir": {"type": "directRcv"},
                "meta": {"itemId": 12, "itemStatus": {"type": "rcvNew"}},
                "content": {
                    "type": "rcvMsgContent",
                    "msgContent": {"type": "file", "text": "audio attachment"},
                },
                "file": {
                    "fileId": 2,
                    "fileName": "clip.m4a",
                    "fileStatus": {"type": "rcvInvitation"},
                },
            },
        }
    )

    event = adapter.handle_message.await_args.args[0]
    assert event.message_type == MessageType.AUDIO


@pytest.mark.asyncio
async def test_fetch_file_searches_configured_simplex_files_folder(monkeypatch, tmp_path):
    """Voice downloads land in the daemon's configured files folder."""
    _install_fake_websockets(monkeypatch)
    from gateway.config import PlatformConfig
    files_dir = tmp_path / "simplex-files"
    files_dir.mkdir()
    audio_path = files_dir / "voice.m4a"
    audio_path.write_bytes(b"not-real-audio")
    cfg = PlatformConfig(
        enabled=True,
        extra={"ws_url": "ws://localhost:5225", "files_folder": str(files_dir)},
    )
    adapter = SimplexAdapter(cfg)
    adapter._send_command = AsyncMock(return_value={"type": "rcvFileAccepted"})  # type: ignore[attr-defined]
    monkeypatch.setattr(_simplex.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(_simplex, "cache_audio_from_bytes", lambda data, ext: f"cached:{ext}")

    cached = await adapter._fetch_file(1, "voice.m4a")

    assert cached == "cached:.m4a"


@pytest.mark.asyncio
async def test_fetch_file_uses_safe_explicit_receive_path(monkeypatch, tmp_path):
    """File receive should stay inside the configured folder and not scan broad dirs."""
    _install_fake_websockets(monkeypatch)
    from gateway.config import PlatformConfig
    files_dir = tmp_path / "simplex-files"
    files_dir.mkdir()
    audio_path = files_dir / "voice.m4a"
    audio_path.write_bytes(b"not-real-audio")
    cfg = PlatformConfig(
        enabled=True,
        extra={"ws_url": "ws://localhost:5225", "files_folder": str(files_dir)},
    )
    adapter = SimplexAdapter(cfg)
    adapter._send_command = AsyncMock(return_value={"type": "rcvFileAccepted"})  # type: ignore[attr-defined]
    monkeypatch.setattr(_simplex.asyncio, "sleep", AsyncMock())
    monkeypatch.setattr(_simplex, "cache_audio_from_bytes", lambda data, ext: f"cached:{ext}")

    cached = await adapter._fetch_file(5, "../voice.m4a")

    assert cached == "cached:.m4a"
    cmd = adapter._send_command.await_args.args[0]
    assert cmd.startswith("/freceive 5 approved_relays=on inline=on ")
    assert str(audio_path) in cmd
    assert ".." not in cmd


@pytest.mark.asyncio
async def test_send_voice_uses_simplex_voice_file_message(monkeypatch, tmp_path):
    _install_fake_websockets(monkeypatch)
    from gateway.config import PlatformConfig
    audio_path = tmp_path / "reply.mp3"
    audio_path.write_bytes(b"\xff\xfb" + b"0" * 16000)
    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    adapter._ws = _RespondingWS(
        adapter,
        {
            "type": "newChatItems",
            "chatItems": [{"chatItem": {"meta": {"itemId": 88}}}],
        },
    )

    result = await adapter.send_voice("42", str(audio_path))

    assert result.success is True
    assert result.message_id == "88"
    cmd = adapter._ws.sent[0]["cmd"]
    assert cmd.startswith("/_send @42 json ")
    payload = json.loads(cmd.split(" json ", 1)[1])
    assert payload[0]["fileSource"] == {"filePath": str(audio_path)}
    assert payload[0]["msgContent"]["type"] == "voice"
    assert isinstance(payload[0]["msgContent"]["duration"], int)
    assert payload[0]["msgContent"]["duration"] >= 1


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


@pytest.mark.asyncio
async def test_standalone_send_uses_api_send_command(monkeypatch):
    """Standalone SimpleX delivery should use the daemon API command syntax."""
    sent_payloads = []

    class _FakeWS:
        async def send(self, payload):
            sent_payloads.append(json.loads(payload))

        async def recv(self):
            return json.dumps(
                {
                    "corrId": sent_payloads[-1]["corrId"],
                    "resp": {
                        "type": "newChatItems",
                        "chatItems": [{"chatItem": {"meta": {"itemId": 91}}}],
                    },
                }
            )

    class _FakeConnect:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return _FakeWS()

        async def __aexit__(self, *_exc):
            return False

    async def _no_sleep(_delay):
        return None

    _install_fake_websockets(monkeypatch, connect=_FakeConnect)
    monkeypatch.setattr(_simplex.asyncio, "sleep", _no_sleep)
    monkeypatch.delenv("SIMPLEX_WS_URL", raising=False)

    pconfig = MagicMock()
    pconfig.extra = {"ws_url": "ws://localhost:5225"}
    result = await _standalone_send(pconfig, "42", "hello from cron")

    assert result == {
        "success": True,
        "platform": "simplex",
        "chat_id": "42",
        "message_id": "91",
    }
    assert len(sent_payloads) == 1
    assert sent_payloads[0]["corrId"].startswith("hermes-snd-")
    assert sent_payloads[0]["cmd"] == (
        '/_send @42 json [{"msgContent":{"type":"text","text":"hello from cron"},"mentions":{}}]'
    )


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
