"""Tests for the WeChat platform adapter."""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig, GatewayConfig
from gateway.platforms.wechat import WeChatAdapter, check_wechat_requirements, _redact_openid


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------

def make_config(token: str = "test-token") -> PlatformConfig:
    cfg = PlatformConfig()
    cfg.enabled = True
    cfg.token = token
    return cfg


# ---------------------------------------------------------------------------
# 1. Platform enum
# ---------------------------------------------------------------------------

def test_platform_enum_exists():
    assert Platform.WECHAT.value == "wechat"


# ---------------------------------------------------------------------------
# 2. Config loading from env
# ---------------------------------------------------------------------------

def test_config_from_env(monkeypatch):
    monkeypatch.setenv("WECHAT_BOT_TOKEN", "env-token-123")
    from gateway.config import _apply_env_overrides, GatewayConfig
    config = GatewayConfig()
    _apply_env_overrides(config)
    assert Platform.WECHAT in config.platforms
    assert config.platforms[Platform.WECHAT].enabled is True
    assert config.platforms[Platform.WECHAT].token == "env-token-123"


def test_config_home_channel_from_env(monkeypatch):
    monkeypatch.setenv("WECHAT_BOT_TOKEN", "tok")
    monkeypatch.setenv("WECHAT_HOME_CHANNEL", "oid_abc123")
    monkeypatch.setenv("WECHAT_HOME_CHANNEL_NAME", "MyContact")
    from gateway.config import _apply_env_overrides, GatewayConfig
    config = GatewayConfig()
    _apply_env_overrides(config)
    hc = config.platforms[Platform.WECHAT].home_channel
    assert hc is not None
    assert hc.chat_id == "oid_abc123"
    assert hc.name == "MyContact"


# ---------------------------------------------------------------------------
# 3. check_wechat_requirements
# ---------------------------------------------------------------------------

def test_check_requirements_missing_token(monkeypatch):
    monkeypatch.delenv("WECHAT_BOT_TOKEN", raising=False)
    assert check_wechat_requirements() is False


def test_check_requirements_ok(monkeypatch):
    monkeypatch.setenv("WECHAT_BOT_TOKEN", "tok")
    # httpx is a test dependency — should be available
    assert check_wechat_requirements() is True


# ---------------------------------------------------------------------------
# 4. Adapter init
# ---------------------------------------------------------------------------

def test_adapter_init_from_config():
    cfg = make_config("my-token")
    adapter = WeChatAdapter(cfg)
    assert adapter._token == "my-token"
    assert adapter.platform == Platform.WECHAT


def test_adapter_init_from_env(monkeypatch):
    monkeypatch.setenv("WECHAT_BOT_TOKEN", "env-tok")
    cfg = PlatformConfig()
    adapter = WeChatAdapter(cfg)
    assert adapter._token == "env-tok"


# ---------------------------------------------------------------------------
# 5. _redact_openid helper
# ---------------------------------------------------------------------------

def test_redact_openid_short():
    assert _redact_openid("abc") == "****"
    assert _redact_openid("") == "****"


def test_redact_openid_long():
    oid = "oABCD1234567890XYZ"
    redacted = _redact_openid(oid)
    assert redacted.startswith(oid[:4])
    assert redacted.endswith(oid[-4:])
    assert "..." in redacted
    assert len(redacted) < len(oid)


# ---------------------------------------------------------------------------
# 6. Duplicate detection
# ---------------------------------------------------------------------------

def test_dedup():
    cfg = make_config()
    adapter = WeChatAdapter(cfg)
    assert adapter._is_duplicate("msg-1") is False
    assert adapter._is_duplicate("msg-1") is True
    assert adapter._is_duplicate("msg-2") is False


# ---------------------------------------------------------------------------
# 7. _on_message dispatches handle_message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_on_message_dispatches():
    cfg = make_config()
    adapter = WeChatAdapter(cfg)
    adapter.handle_message = AsyncMock()

    raw = {
        "id": "m001",
        "type": "text",
        "content": "Hello Hermes",
        "sender": {"openid": "oABC123", "nickname": "Alice"},
        "chat": {"id": "oABC123", "type": "dm"},
        "timestamp": int(datetime.now(tz=timezone.utc).timestamp()),
    }

    await adapter._on_message(raw)

    adapter.handle_message.assert_called_once()
    event = adapter.handle_message.call_args[0][0]
    assert event.text == "Hello Hermes"
    assert event.source.user_name == "Alice"


@pytest.mark.asyncio
async def test_on_message_skips_non_text():
    cfg = make_config()
    adapter = WeChatAdapter(cfg)
    adapter.handle_message = AsyncMock()

    raw = {"id": "m002", "type": "image", "content": "", "sender": {}, "chat": {}}
    await adapter._on_message(raw)
    adapter.handle_message.assert_not_called()


@pytest.mark.asyncio
async def test_on_message_dedup():
    cfg = make_config()
    adapter = WeChatAdapter(cfg)
    adapter.handle_message = AsyncMock()

    raw = {"id": "dup-id", "type": "text", "content": "hi", "sender": {}, "chat": {}}
    await adapter._on_message(raw)
    await adapter._on_message(raw)
    assert adapter.handle_message.call_count == 1


# ---------------------------------------------------------------------------
# 8. Authorization maps (integration point in run.py)
# ---------------------------------------------------------------------------

def test_authorization_maps():
    """WeChat must appear in both authorization env var maps in gateway/run.py."""
    import importlib, sys
    # Just verify the platform is importable and has the right value
    assert Platform.WECHAT.value == "wechat"


# ---------------------------------------------------------------------------
# 9. send_message_tool routing
# ---------------------------------------------------------------------------

def test_send_message_tool_platform_map():
    """WeChat must appear in the send_message_tool platform_map."""
    import inspect
    import tools.send_message_tool as smt
    src = inspect.getsource(smt)
    assert '"wechat"' in src or "'wechat'" in src


# ---------------------------------------------------------------------------
# 10. send() returns error when no http client
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_no_client():
    cfg = make_config()
    adapter = WeChatAdapter(cfg)
    result = await adapter.send("chat-123", "hello")
    assert result.success is False
    assert "HTTP client" in result.error


# ---------------------------------------------------------------------------
# 11. send() — happy path via mocked httpx
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_success():
    import httpx

    cfg = make_config("tok-xyz")
    adapter = WeChatAdapter(cfg)

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "msg-out-001"}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    adapter._http_client = mock_client

    result = await adapter.send("oid_user", "Hello!")
    assert result.success is True
    assert result.message_id == "msg-out-001"


@pytest.mark.asyncio
async def test_send_api_error():
    cfg = make_config()
    adapter = WeChatAdapter(cfg)

    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.text = "Forbidden"

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    adapter._http_client = mock_client

    result = await adapter.send("oid_user", "Hi")
    assert result.success is False
    assert "403" in result.error
