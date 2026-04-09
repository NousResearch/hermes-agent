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
