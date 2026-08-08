"""Tests for the gateway ``/platform reload`` config-diff hot-reload."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.run import GatewayRunner


class StubAdapter(BasePlatformAdapter):
    """Minimal adapter whose connect() result can be controlled."""

    def __init__(self, *, platform=Platform.TELEGRAM, config=None, succeed=True):
        super().__init__(config or PlatformConfig(enabled=True, token="test"), platform)
        self._succeed = succeed
        self.disconnected = False

    async def connect(self):
        return self._succeed

    async def disconnect(self):
        self.disconnected = True
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


def _make_runner(*, adapters=None, config=None):
    """Build a GatewayRunner with just enough wiring for reload tests."""
    runner = object.__new__(GatewayRunner)
    runner.config = config or GatewayConfig(platforms={})
    runner.adapters = adapters or {}
    runner._failed_platforms = {}
    runner.delivery_router = MagicMock()
    runner.session_store = MagicMock()
    runner._busy_text_mode = "default"
    runner._handle_message = MagicMock()
    runner._handle_adapter_fatal_error = MagicMock()
    runner._handle_active_session_busy_message = MagicMock()
    runner._recover_telegram_topic_thread_id = MagicMock()
    runner._update_platform_runtime_status = MagicMock()
    runner._sync_voice_mode_state_to_adapter = MagicMock()
    async def _disconnect(adapter, platform):
        await adapter.disconnect()

    runner._safe_adapter_disconnect = AsyncMock(side_effect=_disconnect)
    return runner


def _config_with(platforms):
    return GatewayConfig(platforms=platforms)


@pytest.mark.asyncio
async def test_reload_connects_newly_enabled_platform():
    runner = _make_runner()
    new_adapter = StubAdapter(platform=Platform.TELEGRAM, succeed=True)
    runner._create_adapter = MagicMock(return_value=new_adapter)
    runner._connect_adapter_with_timeout = AsyncMock(return_value=True)

    new_cfg = _config_with({Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")})
    with patch("gateway.config.load_gateway_config", return_value=new_cfg):
        with patch("gateway.channel_directory.build_channel_directory", new=AsyncMock()):
            report = await runner._reload_platforms_from_config()

    assert Platform.TELEGRAM in runner.adapters
    assert report["connected"] == ["telegram"]
    assert runner.delivery_router.adapters is runner.adapters


@pytest.mark.asyncio
async def test_reload_disconnects_disabled_platform():
    existing = StubAdapter(platform=Platform.TELEGRAM)
    runner = _make_runner(adapters={Platform.TELEGRAM: existing})
    runner._create_adapter = MagicMock()

    # Telegram now disabled.
    new_cfg = _config_with({Platform.TELEGRAM: PlatformConfig(enabled=False, token="t")})
    with patch("gateway.config.load_gateway_config", return_value=new_cfg):
        with patch("gateway.channel_directory.build_channel_directory", new=AsyncMock()):
            report = await runner._reload_platforms_from_config()

    assert Platform.TELEGRAM not in runner.adapters
    assert existing.disconnected is True
    assert report["disconnected"] == ["telegram"]
    runner._create_adapter.assert_not_called()


@pytest.mark.asyncio
async def test_reload_reconnects_changed_config():
    old_cfg = PlatformConfig(enabled=True, token="old")
    existing = StubAdapter(platform=Platform.TELEGRAM, config=old_cfg)
    runner = _make_runner(adapters={Platform.TELEGRAM: existing})
    fresh = StubAdapter(
        platform=Platform.TELEGRAM,
        config=PlatformConfig(enabled=True, token="new"),
        succeed=True,
    )
    runner._create_adapter = MagicMock(return_value=fresh)
    runner._connect_adapter_with_timeout = AsyncMock(return_value=True)

    new_cfg = _config_with({Platform.TELEGRAM: PlatformConfig(enabled=True, token="new")})
    with patch("gateway.config.load_gateway_config", return_value=new_cfg):
        with patch("gateway.channel_directory.build_channel_directory", new=AsyncMock()):
            report = await runner._reload_platforms_from_config()

    assert existing.disconnected is True
    assert runner.adapters[Platform.TELEGRAM] is fresh
    assert report["reconnected"] == ["telegram"]


@pytest.mark.asyncio
async def test_reload_leaves_unchanged_platform_untouched():
    cfg = PlatformConfig(enabled=True, token="same")
    existing = StubAdapter(platform=Platform.TELEGRAM, config=cfg)
    runner = _make_runner(adapters={Platform.TELEGRAM: existing})
    runner._create_adapter = MagicMock()
    runner._connect_adapter_with_timeout = AsyncMock()

    new_cfg = _config_with({Platform.TELEGRAM: PlatformConfig(enabled=True, token="same")})
    with patch("gateway.config.load_gateway_config", return_value=new_cfg):
        with patch("gateway.channel_directory.build_channel_directory", new=AsyncMock()):
            report = await runner._reload_platforms_from_config()

    # Same adapter object stays connected; nothing recreated or disconnected.
    assert runner.adapters[Platform.TELEGRAM] is existing
    assert existing.disconnected is False
    assert report["unchanged"] == ["telegram"]
    runner._create_adapter.assert_not_called()


@pytest.mark.asyncio
async def test_reload_queues_failed_connect_for_retry():
    runner = _make_runner()
    failing = StubAdapter(platform=Platform.TELEGRAM, succeed=False)
    runner._create_adapter = MagicMock(return_value=failing)
    runner._connect_adapter_with_timeout = AsyncMock(return_value=False)

    new_cfg = _config_with({Platform.TELEGRAM: PlatformConfig(enabled=True, token="t")})
    with patch("gateway.config.load_gateway_config", return_value=new_cfg):
        with patch("gateway.channel_directory.build_channel_directory", new=AsyncMock()):
            report = await runner._reload_platforms_from_config()

    assert Platform.TELEGRAM not in runner.adapters
    assert Platform.TELEGRAM in runner._failed_platforms
    assert report["failed"] == ["telegram"]


@pytest.mark.asyncio
async def test_reload_drops_disabled_platform_from_retry_queue():
    runner = _make_runner()
    runner._failed_platforms = {
        Platform.TELEGRAM: {"config": PlatformConfig(enabled=True), "attempts": 2}
    }
    runner._create_adapter = MagicMock()

    new_cfg = _config_with({Platform.TELEGRAM: PlatformConfig(enabled=False)})
    with patch("gateway.config.load_gateway_config", return_value=new_cfg):
        with patch("gateway.channel_directory.build_channel_directory", new=AsyncMock()):
            report = await runner._reload_platforms_from_config()

    assert Platform.TELEGRAM not in runner._failed_platforms
    assert report["disconnected"] == ["telegram"]


@pytest.mark.asyncio
async def test_platform_reload_command_formats_report():
    runner = _make_runner()
    runner._reload_platforms_from_config = AsyncMock(
        return_value={
            "connected": ["telegram"],
            "reconnected": [],
            "disconnected": ["slack"],
            "unchanged": ["discord"],
            "failed": [],
        }
    )
    event = SimpleNamespace(content="/platform reload")
    out = await runner._handle_platform_command(event)

    assert "Connected: telegram" in out
    assert "Disconnected: slack" in out
    assert "Unchanged: discord" in out
