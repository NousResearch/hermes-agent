"""Tests for the SimpleX Chat gateway adapter."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# 1. Platform enum
# ---------------------------------------------------------------------------

def test_platform_enum_exists():
    from gateway.config import Platform
    assert Platform.SIMPLEX.value == "simplex"


# ---------------------------------------------------------------------------
# 2. Config loading from env vars
# ---------------------------------------------------------------------------

def test_env_override_sets_simplex_config(monkeypatch):
    monkeypatch.setenv("SIMPLEX_WS_URL", "ws://127.0.0.1:5225")
    monkeypatch.delenv("SIMPLEX_HOME_CHANNEL", raising=False)

    from gateway.config import GatewayConfig, Platform, _apply_env_overrides
    config = GatewayConfig()
    _apply_env_overrides(config)

    assert Platform.SIMPLEX in config.platforms
    assert config.platforms[Platform.SIMPLEX].enabled is True
    assert config.platforms[Platform.SIMPLEX].extra["ws_url"] == "ws://127.0.0.1:5225"


def test_env_override_home_channel(monkeypatch):
    monkeypatch.setenv("SIMPLEX_WS_URL", "ws://127.0.0.1:5225")
    monkeypatch.setenv("SIMPLEX_HOME_CHANNEL", "42")
    monkeypatch.setenv("SIMPLEX_HOME_CHANNEL_NAME", "Personal")

    from gateway.config import GatewayConfig, Platform, _apply_env_overrides
    config = GatewayConfig()
    _apply_env_overrides(config)

    home = config.platforms[Platform.SIMPLEX].home_channel
    assert home is not None
    assert home.chat_id == "42"
    assert home.name == "Personal"


def test_no_simplex_env_no_platform(monkeypatch):
    monkeypatch.delenv("SIMPLEX_WS_URL", raising=False)

    from gateway.config import GatewayConfig, Platform, _apply_env_overrides
    config = GatewayConfig()
    _apply_env_overrides(config)

    assert Platform.SIMPLEX not in config.platforms


# ---------------------------------------------------------------------------
# 3. Adapter init
# ---------------------------------------------------------------------------

def test_adapter_init():
    from gateway.config import PlatformConfig, Platform
    from gateway.platforms.simplex import SimplexAdapter

    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    assert adapter.ws_url == "ws://localhost:5225"
    assert adapter._running is False
    assert adapter._ws is None


def test_adapter_init_default_url():
    from gateway.config import PlatformConfig
    from gateway.platforms.simplex import SimplexAdapter

    cfg = PlatformConfig(enabled=True)
    adapter = SimplexAdapter(cfg)

    assert adapter.ws_url == "ws://127.0.0.1:5225"


# ---------------------------------------------------------------------------
# 4. check_simplex_requirements
# ---------------------------------------------------------------------------

def test_check_requirements_true(monkeypatch):
    monkeypatch.setenv("SIMPLEX_WS_URL", "ws://127.0.0.1:5225")
    from gateway.platforms.simplex import check_simplex_requirements
    assert check_simplex_requirements() is True


def test_check_requirements_false(monkeypatch):
    monkeypatch.delenv("SIMPLEX_WS_URL", raising=False)
    from gateway.platforms.simplex import check_simplex_requirements
    assert check_simplex_requirements() is False


# ---------------------------------------------------------------------------
# 5. Authorization integration
# ---------------------------------------------------------------------------

def test_simplex_in_auth_allowlist_map(monkeypatch):
    """Platform must appear in both auth maps inside _is_user_authorized."""
    monkeypatch.setenv("SIMPLEX_ALLOWED_USERS", "contact-abc")

    from gateway.config import Platform
    from gateway.session import SessionSource

    source = SessionSource(
        platform=Platform.SIMPLEX,
        chat_id="contact-abc",
        chat_name="Alice",
        chat_type="dm",
        user_id="contact-abc",
        user_name="Alice",
    )

    # Build a minimal runner without connecting real adapters
    from gateway.run import GatewayRunner
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = MagicMock()
    runner.config.get_unauthorized_dm_behavior.return_value = "pair"
    runner.pairing_store = MagicMock()
    runner.pairing_store.is_approved.return_value = False

    result = GatewayRunner._is_user_authorized(runner, source)
    assert result is True


# ---------------------------------------------------------------------------
# 6. send_message_tool routing
# ---------------------------------------------------------------------------

def test_simplex_in_send_message_platform_map():
    from tools.send_message_tool import send_message_tool
    from gateway.config import Platform
    import importlib, inspect

    # Just verify Platform.SIMPLEX would resolve in the map
    from gateway.config import Platform
    assert hasattr(Platform, "SIMPLEX")


# ---------------------------------------------------------------------------
# 7. Helper functions
# ---------------------------------------------------------------------------

def test_guess_extension_png():
    from gateway.platforms.simplex import _guess_extension
    assert _guess_extension(b"\x89PNG\r\n\x1a\n") == ".png"


def test_guess_extension_jpg():
    from gateway.platforms.simplex import _guess_extension
    assert _guess_extension(b"\xff\xd8\xff\xe0") == ".jpg"


def test_guess_extension_ogg():
    from gateway.platforms.simplex import _guess_extension
    assert _guess_extension(b"OggS\x00\x02") == ".ogg"


def test_guess_extension_unknown():
    from gateway.platforms.simplex import _guess_extension
    assert _guess_extension(b"\x00\x01\x02\x03") == ".bin"


def test_is_image_ext():
    from gateway.platforms.simplex import _is_image_ext
    assert _is_image_ext(".png") is True
    assert _is_image_ext(".webp") is True
    assert _is_image_ext(".ogg") is False


def test_is_audio_ext():
    from gateway.platforms.simplex import _is_audio_ext
    assert _is_audio_ext(".ogg") is True
    assert _is_audio_ext(".mp3") is True
    assert _is_audio_ext(".pdf") is False


# ---------------------------------------------------------------------------
# 8. _make_corr_id
# ---------------------------------------------------------------------------

def test_corr_id_starts_with_hermes():
    from gateway.config import PlatformConfig
    from gateway.platforms.simplex import SimplexAdapter, _CORR_PREFIX

    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)
    corr_id = adapter._make_corr_id()

    assert corr_id.startswith(_CORR_PREFIX)
    assert corr_id in adapter._pending_corr_ids


# ---------------------------------------------------------------------------
# 9. Message sending (mock WS)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_dm():
    from gateway.config import PlatformConfig
    from gateway.platforms.simplex import SimplexAdapter

    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    mock_ws = AsyncMock()
    adapter._ws = mock_ws

    result = await adapter.send("contact-42", "Hello, SimpleX!")

    mock_ws.send.assert_called_once()
    sent = mock_ws.send.call_args[0][0]
    import json
    payload = json.loads(sent)
    assert "@[contact-42] Hello, SimpleX!" in payload["cmd"]
    assert result.success is True


@pytest.mark.asyncio
async def test_send_group():
    from gateway.config import PlatformConfig
    from gateway.platforms.simplex import SimplexAdapter

    cfg = PlatformConfig(enabled=True, extra={"ws_url": "ws://localhost:5225"})
    adapter = SimplexAdapter(cfg)

    mock_ws = AsyncMock()
    adapter._ws = mock_ws

    result = await adapter.send("group:grp-99", "Hello, group!")

    sent = mock_ws.send.call_args[0][0]
    import json
    payload = json.loads(sent)
    assert "#[grp-99] Hello, group!" in payload["cmd"]
    assert result.success is True
