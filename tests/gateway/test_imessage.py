"""Tests for the iMessage gateway platform adapter."""

import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Platform enum
# ---------------------------------------------------------------------------

def test_platform_enum_has_imessage():
    assert Platform.IMESSAGE.value == "imessage"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def test_config_env_override_imessage_enabled():
    """IMESSAGE_ENABLED=true should create an enabled platform config."""
    with patch.dict(os.environ, {"IMESSAGE_ENABLED": "true"}, clear=False):
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.IMESSAGE)
        assert pconfig is not None
        assert pconfig.enabled is True


def test_config_env_override_imessage_home_channel():
    """IMESSAGE_HOME_CHANNEL should set the home channel."""
    with patch.dict(os.environ, {
        "IMESSAGE_ENABLED": "true",
        "IMESSAGE_HOME_CHANNEL": "+15551234567",
    }, clear=False):
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.IMESSAGE)
        assert pconfig is not None
        assert pconfig.home_channel is not None
        assert pconfig.home_channel.chat_id == "+15551234567"


def test_config_env_override_imessage_watch_chat_ids():
    """IMESSAGE_WATCH_CHAT_IDS should populate extra config."""
    with patch.dict(os.environ, {
        "IMESSAGE_ENABLED": "true",
        "IMESSAGE_WATCH_CHAT_IDS": "123,456,789",
    }, clear=False):
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.IMESSAGE)
        assert pconfig is not None
        assert pconfig.extra.get("watch_chat_ids") == ["123", "456", "789"]


def test_config_env_override_imessage_watch_mode():
    """IMESSAGE_WATCH_MODE should populate extra config."""
    with patch.dict(os.environ, {
        "IMESSAGE_ENABLED": "true",
        "IMESSAGE_WATCH_MODE": "poll",
    }, clear=False):
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.IMESSAGE)
        assert pconfig is not None
        assert pconfig.extra.get("watch_mode") == "poll"


def test_config_env_override_imessage_poll_interval():
    """IMESSAGE_POLL_INTERVAL should set poll interval."""
    with patch.dict(os.environ, {
        "IMESSAGE_ENABLED": "true",
        "IMESSAGE_POLL_INTERVAL": "5.0",
    }, clear=False):
        from gateway.config import load_gateway_config
        config = load_gateway_config()
        pconfig = config.platforms.get(Platform.IMESSAGE)
        assert pconfig is not None
        assert pconfig.extra.get("poll_interval") == 5.0


def test_adapter_default_watch_mode():
    """Default watch mode should be auto."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    assert adapter._watch_mode == "auto"


def test_adapter_poll_mode_from_config():
    """watch_mode=poll should set poll mode."""
    from gateway.platforms.imessage import IMessageAdapter
    config = PlatformConfig(enabled=True, extra={"watch_mode": "poll", "poll_interval": 5.0})
    adapter = IMessageAdapter(config)
    assert adapter._watch_mode == "poll"
    assert adapter._poll_interval == 5.0


def test_connected_platforms_includes_imessage():
    """iMessage should appear in connected platforms when enabled."""
    from gateway.config import GatewayConfig
    config = GatewayConfig(platforms={
        Platform.IMESSAGE: PlatformConfig(enabled=True),
    })
    connected = config.get_connected_platforms()
    assert Platform.IMESSAGE in connected


def test_connected_platforms_excludes_disabled_imessage():
    """iMessage should not appear when disabled."""
    from gateway.config import GatewayConfig
    config = GatewayConfig(platforms={
        Platform.IMESSAGE: PlatformConfig(enabled=False),
    })
    connected = config.get_connected_platforms()
    assert Platform.IMESSAGE not in connected


# ---------------------------------------------------------------------------
# Requirements check
# ---------------------------------------------------------------------------

def test_check_requirements_not_macos():
    """check_imessage_requirements should return False on non-macOS."""
    from gateway.platforms.imessage import check_imessage_requirements
    with patch("gateway.platforms.imessage.sys") as mock_sys:
        mock_sys.platform = "linux"
        assert check_imessage_requirements() is False


def test_check_requirements_no_imsg():
    """check_imessage_requirements should return False when imsg is not in PATH."""
    from gateway.platforms.imessage import check_imessage_requirements
    with patch("gateway.platforms.imessage.sys") as mock_sys, \
         patch("gateway.platforms.imessage.shutil") as mock_shutil:
        mock_sys.platform = "darwin"
        mock_shutil.which.return_value = None
        assert check_imessage_requirements() is False


def test_check_requirements_ok():
    """check_imessage_requirements should return True on macOS with imsg."""
    from gateway.platforms.imessage import check_imessage_requirements
    with patch("gateway.platforms.imessage.sys") as mock_sys, \
         patch("gateway.platforms.imessage.shutil") as mock_shutil:
        mock_sys.platform = "darwin"
        mock_shutil.which.return_value = "/usr/local/bin/imsg"
        assert check_imessage_requirements() is True


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

def test_redact_phone_number():
    from gateway.platforms.imessage import _redact_imessage_id
    assert _redact_imessage_id("+15551234567") == "+155****4567"


def test_redact_short_phone():
    from gateway.platforms.imessage import _redact_imessage_id
    # Short phone numbers still get redacted
    result = _redact_imessage_id("+1234")
    assert "****" in result


def test_redact_apple_id():
    from gateway.platforms.imessage import _redact_imessage_id
    assert _redact_imessage_id("user@example.com") == "us****@example.com"


def test_redact_empty():
    from gateway.platforms.imessage import _redact_imessage_id
    assert _redact_imessage_id("") == "<none>"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def test_is_duplicate_basic():
    """First time should not be duplicate, second time should be."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    assert adapter._is_duplicate("guid-1") is False
    assert adapter._is_duplicate("guid-1") is True


def test_is_duplicate_different_guids():
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    assert adapter._is_duplicate("guid-1") is False
    assert adapter._is_duplicate("guid-2") is False


def test_is_duplicate_max_size():
    """Should evict oldest when hitting max size."""
    from gateway.platforms.imessage import IMessageAdapter, DEDUP_MAX_SIZE
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    # Fill up to max
    for i in range(DEDUP_MAX_SIZE):
        adapter._is_duplicate(f"guid-{i}")
    # One more should still work (evicts oldest)
    assert adapter._is_duplicate(f"guid-new") is False
    assert len(adapter._seen_messages) <= DEDUP_MAX_SIZE


# ---------------------------------------------------------------------------
# Recipient resolution
# ---------------------------------------------------------------------------

def test_resolve_recipient_phone():
    """Phone numbers should pass through directly."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    assert adapter._resolve_recipient("+15551234567") == "+15551234567"


def test_resolve_recipient_email():
    """Apple IDs (emails) should pass through directly."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    assert adapter._resolve_recipient("user@icloud.com") == "user@icloud.com"


def test_resolve_recipient_cache_lookup():
    """Numeric chat_id should resolve via cache."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    adapter._chat_cache = {
        42: {"identifier": "+15559876543", "name": "Alice", "service": "iMessage"},
    }
    assert adapter._resolve_recipient("42") == "+15559876543"


def test_resolve_recipient_cache_miss():
    """Unknown numeric chat_id should fall back to the raw value."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    assert adapter._resolve_recipient("999") == "999"


def test_resolve_recipient_empty():
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    assert adapter._resolve_recipient("") == ""


# ---------------------------------------------------------------------------
# Message handling
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_message_filters_from_me():
    """Messages with is_from_me should be silently dropped."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    adapter.handle_message = AsyncMock()

    await adapter._handle_message_data({"is_from_me": True, "text": "hello", "guid": "g1"})
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_dedup():
    """Duplicate guids should be dropped."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    adapter.handle_message = AsyncMock()
    adapter._refresh_chat_cache = AsyncMock()

    data = {"is_from_me": False, "text": "hello", "guid": "g1", "chat_id": 1, "sender": "+1234"}
    await adapter._handle_message_data(data)
    assert adapter.handle_message.call_count == 1

    await adapter._handle_message_data(data)
    assert adapter.handle_message.call_count == 1  # Not called again


@pytest.mark.asyncio
async def test_handle_message_builds_event():
    """Valid message should produce a MessageEvent with correct fields."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    adapter._refresh_chat_cache = AsyncMock()

    events = []
    async def capture_event(event):
        events.append(event)
    adapter.handle_message = capture_event

    data = {
        "is_from_me": False,
        "text": "Hey there",
        "guid": "unique-guid",
        "chat_id": 5,
        "sender": "+15551234567",
        "service": "iMessage",
    }
    await adapter._handle_message_data(data)

    assert len(events) == 1
    event = events[0]
    assert event.text == "Hey there"
    assert event.source.user_id == "+15551234567"


@pytest.mark.asyncio
async def test_handle_message_filters_watch_chat_ids():
    """Messages from chats not in watch_chat_ids should be dropped."""
    from gateway.platforms.imessage import IMessageAdapter
    config = PlatformConfig(enabled=True, extra={"watch_chat_ids": ["10", "20"]})
    adapter = IMessageAdapter(config)
    adapter.handle_message = AsyncMock()

    data = {"is_from_me": False, "text": "hello", "guid": "g1", "chat_id": 99, "sender": "+1"}
    await adapter._handle_message_data(data)
    adapter.handle_message.assert_not_called()


# ---------------------------------------------------------------------------
# Chat cache parsing
# ---------------------------------------------------------------------------

def test_parse_chat_list():
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))

    json_output = '[{"id": 1, "identifier": "+15551234567", "name": "Alice", "service": "iMessage"}, {"id": 2, "identifier": "bob@icloud.com", "display_name": "Bob"}]'
    adapter._parse_chat_list(json_output)

    assert 1 in adapter._chat_cache
    assert adapter._chat_cache[1]["identifier"] == "+15551234567"
    assert adapter._chat_cache[1]["name"] == "Alice"
    assert 2 in adapter._chat_cache
    assert adapter._chat_cache[2]["identifier"] == "bob@icloud.com"
    assert adapter._chat_cache[2]["name"] == "Bob"


def test_parse_chat_list_ndjson():
    """imsg chats --json outputs NDJSON (one JSON object per line)."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))

    ndjson_output = '{"id": 1, "identifier": "+15551234567", "name": "Alice", "service": "iMessage"}\n{"id": 2, "identifier": "bob@icloud.com", "display_name": "Bob"}\n'
    adapter._parse_chat_list(ndjson_output)

    assert 1 in adapter._chat_cache
    assert adapter._chat_cache[1]["identifier"] == "+15551234567"
    assert 2 in adapter._chat_cache
    assert adapter._chat_cache[2]["identifier"] == "bob@icloud.com"
    assert adapter._chat_cache[2]["name"] == "Bob"


def test_parse_chat_list_invalid_json():
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    adapter._parse_chat_list("not json")
    assert len(adapter._chat_cache) == 0


# ---------------------------------------------------------------------------
# Authorization maps (integration test)
# ---------------------------------------------------------------------------

def test_auth_maps_include_imessage():
    """Verify IMESSAGE entries exist in the run.py authorization maps."""
    # Read the source and check for the entries
    run_py = os.path.join(os.path.dirname(__file__), "..", "..", "gateway", "run.py")
    with open(run_py, encoding="utf-8") as f:
        content = f.read()
    assert "IMESSAGE_ALLOWED_USERS" in content
    assert "IMESSAGE_ALLOW_ALL_USERS" in content


# ---------------------------------------------------------------------------
# Toolset integration
# ---------------------------------------------------------------------------

def test_toolset_includes_imessage():
    """hermes-gateway should include hermes-imessage."""
    from toolsets import TOOLSETS
    gateway = TOOLSETS.get("hermes-gateway", {})
    assert "hermes-imessage" in gateway.get("includes", [])


def test_imessage_toolset_exists():
    from toolsets import TOOLSETS
    assert "hermes-imessage" in TOOLSETS


# ---------------------------------------------------------------------------
# Platform hints
# ---------------------------------------------------------------------------

def test_platform_hint_exists():
    from agent.prompt_builder import PLATFORM_HINTS
    assert "imessage" in PLATFORM_HINTS
    assert "markdown" in PLATFORM_HINTS["imessage"].lower()


# ---------------------------------------------------------------------------
# Send message tool integration
# ---------------------------------------------------------------------------

def test_send_message_platform_map_has_imessage():
    """The send_message tool should route 'imessage' to Platform.IMESSAGE."""
    # This tests by importing and checking the code path
    source = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "send_message_tool.py")
    with open(source, encoding="utf-8") as f:
        content = f.read()
    assert '"imessage": Platform.IMESSAGE' in content


# ---------------------------------------------------------------------------
# Cron integration
# ---------------------------------------------------------------------------

def test_cron_known_delivery_platforms():
    from cron.scheduler import _KNOWN_DELIVERY_PLATFORMS
    assert "imessage" in _KNOWN_DELIVERY_PLATFORMS


def test_cron_platform_map_has_imessage():
    source = os.path.join(os.path.dirname(__file__), "..", "..", "cron", "scheduler.py")
    with open(source, encoding="utf-8") as f:
        content = f.read()
    assert '"imessage": Platform.IMESSAGE' in content


# ---------------------------------------------------------------------------
# Channel directory
# ---------------------------------------------------------------------------

def test_channel_directory_includes_imessage():
    source = os.path.join(os.path.dirname(__file__), "..", "..", "gateway", "channel_directory.py")
    with open(source, encoding="utf-8") as f:
        content = f.read()
    assert '"imessage"' in content


# ---------------------------------------------------------------------------
# Send methods
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_text_success():
    """send() should call imsg send and return success."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"OK", b""))
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
        result = await adapter.send("+15551234567", "Hello!")
        assert result.success is True
        mock_exec.assert_called_once()
        cmd_args = mock_exec.call_args[0]
        assert "imsg" in cmd_args
        assert "send" in cmd_args
        assert "--to" in cmd_args
        assert "+15551234567" in cmd_args
        assert "--text" in cmd_args
        assert "Hello!" in cmd_args


@pytest.mark.asyncio
async def test_send_text_failure():
    """send() should return failure on non-zero exit."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"", b"error occurred"))
    mock_proc.returncode = 1

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        result = await adapter.send("+15551234567", "Hello!")
        assert result.success is False
        assert "error" in result.error.lower()


@pytest.mark.asyncio
async def test_send_text_timeout():
    """send() should return failure on timeout."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))

    with patch("asyncio.create_subprocess_exec", side_effect=asyncio.TimeoutError()):
        result = await adapter.send("+15551234567", "Hello!")
        assert result.success is False
        assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_send_empty_recipient():
    """send() should fail when recipient cannot be resolved."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    result = await adapter.send("", "Hello!")
    assert result.success is False


@pytest.mark.asyncio
async def test_send_file_success():
    """_send_file should call imsg send --file and return success."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))

    mock_proc = AsyncMock()
    mock_proc.communicate = AsyncMock(return_value=(b"OK", b""))
    mock_proc.returncode = 0

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc), \
         patch("os.path.exists", return_value=True):
        result = await adapter._send_file("+15551234567", "/tmp/test.png", caption="Look!")
        assert result.success is True


@pytest.mark.asyncio
async def test_send_file_not_found():
    """_send_file should fail if file doesn't exist."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))

    with patch("os.path.exists", return_value=False):
        result = await adapter._send_file("+15551234567", "/nonexistent.png")
        assert result.success is False
        assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_send_image_local_file():
    """send_image should delegate to send_image_file for local paths."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    adapter.send_image_file = AsyncMock(return_value=MagicMock(success=True))

    with patch("os.path.exists", return_value=True):
        result = await adapter.send_image("+15551234567", "/tmp/photo.jpg", caption="pic")
        adapter.send_image_file.assert_called_once_with("+15551234567", "/tmp/photo.jpg", caption="pic")


@pytest.mark.asyncio
async def test_send_image_url_unsupported():
    """send_image should fail for non-local URLs."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))

    with patch("os.path.exists", return_value=False):
        result = await adapter.send_image("+15551234567", "https://example.com/img.jpg")
        assert result.success is False


# ---------------------------------------------------------------------------
# Poll mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_seed_last_rowid_uses_max():
    """_seed_last_rowid should find the max rowid across sampled chats."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    adapter._chat_cache = {
        1: {"identifier": "+1", "name": "A", "service": "iMessage"},
        2: {"identifier": "+2", "name": "B", "service": "iMessage"},
    }

    call_count = 0
    async def fake_exec(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_proc = AsyncMock()
        # Chat 1 has rowid 100, chat 2 has rowid 200
        rowid = 100 if call_count == 1 else 200
        mock_proc.communicate = AsyncMock(
            return_value=(f'{{"id": {rowid}}}\n'.encode(), b"")
        )
        mock_proc.returncode = 0
        return mock_proc

    with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
        await adapter._seed_last_rowid()
        assert adapter._last_rowid == 200


@pytest.mark.asyncio
async def test_seed_last_rowid_empty_cache():
    """_seed_last_rowid should do nothing with empty cache."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    adapter._chat_cache = {}
    await adapter._seed_last_rowid()
    assert adapter._last_rowid == 0


@pytest.mark.asyncio
async def test_poll_once_processes_messages():
    """_poll_once should process new messages and update last_rowid."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    adapter._last_rowid = 50
    adapter.handle_message = AsyncMock()
    adapter._refresh_chat_cache = AsyncMock()

    lines = [
        b'{"id": 51, "guid": "g1", "is_from_me": false, "text": "hi", "chat_id": 1, "sender": "+1"}\n',
        b'{"id": 52, "guid": "g2", "is_from_me": false, "text": "hey", "chat_id": 1, "sender": "+1"}\n',
    ]

    mock_proc = AsyncMock()
    line_iter = iter(lines + [b""])  # Empty bytes signals EOF
    async def fake_readline():
        return next(line_iter)
    mock_proc.stdout = AsyncMock()
    mock_proc.stdout.readline = fake_readline
    mock_proc.kill = MagicMock()
    mock_proc.wait = AsyncMock()

    with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
        count = await adapter._poll_once()
        assert count == 2
        assert adapter._last_rowid == 52


@pytest.mark.asyncio
async def test_send_typing_is_noop():
    """send_typing should be a no-op for iMessage."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    # Should not raise
    await adapter.send_typing("+15551234567")


# ---------------------------------------------------------------------------
# Chat info
# ---------------------------------------------------------------------------

def test_get_chat_info_from_cache():
    """get_chat_info should return metadata from cache."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    adapter._chat_cache = {
        42: {"identifier": "+15559876543", "name": "Alice", "service": "iMessage"},
    }
    import asyncio
    info = asyncio.get_event_loop().run_until_complete(adapter.get_chat_info("42"))
    assert info["name"] == "Alice"
    assert info["identifier"] == "+15559876543"


def test_get_chat_info_unknown():
    """get_chat_info should return basic info for unknown chat_id."""
    from gateway.platforms.imessage import IMessageAdapter
    adapter = IMessageAdapter(PlatformConfig(enabled=True))
    import asyncio
    info = asyncio.get_event_loop().run_until_complete(adapter.get_chat_info("unknown"))
    assert info["name"] == "unknown"
    assert info["chat_id"] == "unknown"
