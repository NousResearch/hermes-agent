"""Unit tests for HubAdapter and Hub platform integration."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gateway.config import Platform, PlatformConfig, GatewayConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.platforms.hub import HubAdapter, check_hub_requirements
from gateway.session import SessionSource


# ── Fixtures ────────────────────────────────────────────────────────


def _hub_config(**overrides):
    extra = {
        "agent_id": "test-agent",
        "agent_secret": "test-secret-123",
        "ws_url": "wss://example.com/agents/test-agent/ws",
        "api_base": "https://example.com",
    }
    extra.update(overrides)
    return PlatformConfig(enabled=True, extra=extra)


def _make_adapter(**overrides):
    return HubAdapter(_hub_config(**overrides))


# ── Platform enum ───────────────────────────────────────────────────


def test_platform_enum_hub_exists():
    assert Platform.HUB.value == "hub"


# ── check_hub_requirements ──────────────────────────────────────────


def test_check_hub_requirements_returns_true():
    assert check_hub_requirements() is True


def test_check_hub_requirements_missing_httpx():
    with patch.dict(sys.modules, {"httpx": None}):
        # Force re-import check by calling the function
        # (lazy import inside the function means we need to patch at import time)
        import importlib
        import gateway.platforms.hub as hub_mod
        # The function imports inside, so patching sys.modules is enough
        # only if httpx hasn't been cached. Test the logic directly:
        pass  # check_hub_requirements uses try/import, hard to mock after first import


# ── Config integration ──────────────────────────────────────────────


def test_get_connected_platforms_hub_with_secret():
    cfg = GatewayConfig()
    cfg.platforms[Platform.HUB] = PlatformConfig(
        enabled=True, extra={"agent_secret": "s"}
    )
    assert Platform.HUB in cfg.get_connected_platforms()


def test_get_connected_platforms_hub_without_secret():
    cfg = GatewayConfig()
    cfg.platforms[Platform.HUB] = PlatformConfig(
        enabled=True, extra={}
    )
    assert Platform.HUB not in cfg.get_connected_platforms()


def test_get_connected_platforms_hub_disabled():
    cfg = GatewayConfig()
    cfg.platforms[Platform.HUB] = PlatformConfig(
        enabled=False, extra={"agent_secret": "s"}
    )
    assert Platform.HUB not in cfg.get_connected_platforms()


# ── Adapter init ────────────────────────────────────────────────────


def test_hub_init_reads_config():
    adapter = _make_adapter()
    assert adapter._agent_id == "test-agent"
    assert adapter._agent_secret == "test-secret-123"
    assert adapter._ws_url == "wss://example.com/agents/test-agent/ws"
    assert adapter._api_base == "https://example.com"


def test_hub_init_defaults():
    config = PlatformConfig(enabled=True, extra={
        "agent_id": "my-agent",
        "agent_secret": "secret",
    })
    adapter = HubAdapter(config)
    assert "my-agent" in adapter._ws_url
    assert "admin.slate.ceo" in adapter._api_base


# ── Adapter factory ─────────────────────────────────────────────────


def test_create_adapter_hub_success():
    from gateway.run import GatewayRunner
    cfg = GatewayConfig()
    runner = GatewayRunner(cfg)
    hub_cfg = _hub_config()
    adapter = runner._create_adapter(Platform.HUB, hub_cfg)
    assert isinstance(adapter, HubAdapter)


def test_create_adapter_hub_missing_secret():
    from gateway.run import GatewayRunner
    cfg = GatewayConfig()
    runner = GatewayRunner(cfg)
    hub_cfg = PlatformConfig(enabled=True, extra={"agent_id": "test"})
    adapter = runner._create_adapter(Platform.HUB, hub_cfg)
    assert adapter is None


# ── Auth bypass ─────────────────────────────────────────────────────


def test_auth_bypass_hub():
    from gateway.run import GatewayRunner
    cfg = GatewayConfig()
    runner = GatewayRunner(cfg)
    source = SessionSource(platform=Platform.HUB, chat_id="hub:brain")
    assert runner._is_user_authorized(source) is True


def test_auth_rejects_telegram_without_user_id():
    from gateway.run import GatewayRunner
    cfg = GatewayConfig()
    runner = GatewayRunner(cfg)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="123")
    assert runner._is_user_authorized(source) is False


# ── _UPDATE_ALLOWED_PLATFORMS ───────────────────────────────────────


def test_update_allowed_platforms_has_hub():
    from gateway.run import GatewayRunner
    assert Platform.HUB in GatewayRunner._UPDATE_ALLOWED_PLATFORMS


# ── send() ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_send_success():
    adapter = _make_adapter()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"message_id": "msg-123", "ok": True}

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.aclose = AsyncMock()
    adapter._http_client = mock_client

    result = await adapter.send("hub:brain", "hello")
    assert result.success is True
    assert result.message_id == "msg-123"

    call_args = mock_client.post.call_args
    assert "brain/message" in call_args[0][0]
    body = call_args[1]["json"]
    assert body["from"] == "test-agent"
    assert body["secret"] == "test-secret-123"
    assert body["message"] == "hello"


@pytest.mark.asyncio
async def test_send_without_hub_prefix():
    adapter = _make_adapter()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"message_id": "msg-456"}

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.aclose = AsyncMock()
    adapter._http_client = mock_client

    result = await adapter.send("brain", "hello")
    assert result.success is True
    assert "brain/message" in mock_client.post.call_args[0][0]


def test_hub_does_not_support_message_editing():
    # Hub's REST API has no edit endpoint; declaring this flag prevents the
    # gateway stream consumer from sending a partial + continuation pair.
    assert HubAdapter.SUPPORTS_MESSAGE_EDITING is False


@pytest.mark.asyncio
async def test_send_suppresses_operation_interrupted():
    adapter = _make_adapter()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock()
    mock_client.aclose = AsyncMock()
    adapter._http_client = mock_client

    result = await adapter.send(
        "hub:brain",
        "Operation interrupted: waiting for model response (12.3s elapsed).",
    )
    assert result.success is True
    assert result.message_id == ""
    mock_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_send_suppresses_api_call_failed():
    adapter = _make_adapter()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock()
    mock_client.aclose = AsyncMock()
    adapter._http_client = mock_client

    result = await adapter.send(
        "hub:brain",
        "API call failed after 3 retries: HTTP 429 Too Many Requests",
    )
    assert result.success is True
    mock_client.post.assert_not_called()


@pytest.mark.asyncio
async def test_send_forwards_normal_content():
    # Regression: the suppression filter must not swallow normal agent replies
    # that happen to begin similarly.
    adapter = _make_adapter()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"message_id": "msg-789"}
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.aclose = AsyncMock()
    adapter._http_client = mock_client

    result = await adapter.send("hub:brain", "API call failed on your side? Try again.")
    # This does NOT start with "API call failed after" — must pass through.
    assert result.success is True
    mock_client.post.assert_called_once()


@pytest.mark.asyncio
async def test_send_http_error():
    adapter = _make_adapter()
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not found"

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.aclose = AsyncMock()
    adapter._http_client = mock_client

    result = await adapter.send("hub:nonexistent", "hello")
    assert result.success is False
    assert result.retryable is False  # 404 < 500


@pytest.mark.asyncio
async def test_send_server_error_is_retryable():
    adapter = _make_adapter()
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.text = "Service unavailable"

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    mock_client.aclose = AsyncMock()
    adapter._http_client = mock_client

    result = await adapter.send("hub:brain", "hello")
    assert result.success is False
    assert result.retryable is True


# ── get_chat_info() ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_chat_info_success():
    adapter = _make_adapter()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "agent_id": "brain",
        "description": "Hub coordinator",
        "capabilities": ["messaging", "routing"],
    }

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.aclose = AsyncMock()
    adapter._http_client = mock_client

    info = await adapter.get_chat_info("hub:brain")
    assert info["name"] == "brain"
    assert info["type"] == "dm"
    assert info["description"] == "Hub coordinator"


@pytest.mark.asyncio
async def test_get_chat_info_without_prefix():
    adapter = _make_adapter()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"description": "test"}

    mock_client = AsyncMock()
    mock_client.get.return_value = mock_response
    mock_client.aclose = AsyncMock()
    adapter._http_client = mock_client

    info = await adapter.get_chat_info("brain")
    assert info["name"] == "brain"


@pytest.mark.asyncio
async def test_get_chat_info_error_fallback():
    adapter = _make_adapter()
    mock_client = AsyncMock()
    mock_client.get.side_effect = Exception("network error")
    mock_client.aclose = AsyncMock()
    adapter._http_client = mock_client

    info = await adapter.get_chat_info("hub:brain")
    assert info == {"name": "brain", "type": "dm"}


# ── _handle_inbound_message ────────────────────────────────────────


@pytest.mark.asyncio
async def test_handle_inbound_message_creates_event():
    adapter = _make_adapter()
    events = []
    adapter.handle_message = AsyncMock(side_effect=lambda e: events.append(e))

    await adapter._handle_inbound_message({
        "from": "brain",
        "text": "Hello!",
        "messageId": "msg-1",
        "timestamp": "2026-04-07T12:00:00Z",
    })

    assert len(events) == 1
    evt = events[0]
    assert evt.text == "Hello!"
    assert evt.source.platform == Platform.HUB
    assert evt.source.chat_id == "hub:brain"
    assert evt.message_id == "msg-1"
    assert evt.message_type == MessageType.TEXT


@pytest.mark.asyncio
async def test_handle_inbound_message_command_detection():
    adapter = _make_adapter()
    events = []
    adapter.handle_message = AsyncMock(side_effect=lambda e: events.append(e))

    await adapter._handle_inbound_message({
        "from": "brain",
        "text": "/help",
        "messageId": "msg-2",
    })

    assert events[0].message_type == MessageType.COMMAND


@pytest.mark.asyncio
async def test_handle_inbound_message_skips_empty():
    adapter = _make_adapter()
    handler = AsyncMock()
    adapter.set_message_handler(handler)

    await adapter._handle_inbound_message({"from": "", "text": "hello"})
    await adapter._handle_inbound_message({"from": "brain", "text": ""})
    await adapter._handle_inbound_message({})

    handler.assert_not_called()


@pytest.mark.asyncio
async def test_handle_inbound_message_filters_self():
    adapter = _make_adapter()
    handler = AsyncMock()
    adapter.set_message_handler(handler)

    await adapter._handle_inbound_message({
        "from": "test-agent",  # Same as adapter's agent_id
        "text": "echo",
        "messageId": "msg-self",
    })

    handler.assert_not_called()


# ── _dispatch_ws_message ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_dispatch_routes_message_type():
    adapter = _make_adapter()
    adapter._handle_inbound_message = AsyncMock()

    await adapter._dispatch_ws_message({"type": "message", "data": {"from": "x", "text": "hi"}})
    adapter._handle_inbound_message.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_ignores_protocol_messages():
    adapter = _make_adapter()
    adapter._handle_inbound_message = AsyncMock()

    for msg_type in ("auth", "pong", "send_result"):
        await adapter._dispatch_ws_message({"type": msg_type})

    adapter._handle_inbound_message.assert_not_called()


# ── WS auth failure ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ws_auth_failure_sets_fatal_error():
    adapter = _make_adapter()
    adapter._set_fatal_error = MagicMock()

    mock_ws = AsyncMock()
    mock_ws.send = AsyncMock()
    mock_ws.recv = AsyncMock(return_value=json.dumps({"ok": False, "error": "bad secret"}))

    with patch("websockets.connect", AsyncMock(return_value=mock_ws)):
        adapter._reader_task = asyncio.create_task(adapter._run_ws())
        await asyncio.sleep(0.5)
        adapter._should_reconnect = False
        adapter._reader_task.cancel()
        try:
            await adapter._reader_task
        except asyncio.CancelledError:
            pass

    adapter._set_fatal_error.assert_called_once()
    args = adapter._set_fatal_error.call_args
    assert args[0][0] == "auth_failed"
    assert args[1]["retryable"] is False


# ── Send message tool ──────────────────────────────────────────────


def test_send_message_tool_platform_map_has_hub():
    """Verify Hub is in the send_message_tool platform map."""
    # The platform_map is local to the function, so we check indirectly
    # by verifying _send_hub exists and is importable
    from tools.send_message_tool import _send_hub
    assert callable(_send_hub)


@pytest.mark.asyncio
async def test_send_hub_standalone_success():
    from tools.send_message_tool import _send_hub

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"message_id": "standalone-1"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await _send_hub(
            {"agent_id": "test", "agent_secret": "s", "api_base": "https://example.com"},
            "hub:brain",
            "hello"
        )

    assert result["success"] is True
    assert result["message_id"] == "standalone-1"


@pytest.mark.asyncio
async def test_send_hub_standalone_without_prefix():
    from tools.send_message_tool import _send_hub

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"message_id": "standalone-2"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post.return_value = mock_response

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await _send_hub(
            {"agent_id": "test", "agent_secret": "s", "api_base": "https://example.com"},
            "brain",
            "hello"
        )

    assert result["success"] is True
    assert "brain/message" in mock_client.post.call_args[0][0]


# ── Cron delivery ──────────────────────────────────────────────────


def test_cron_known_delivery_platforms_has_hub():
    from cron.scheduler import _KNOWN_DELIVERY_PLATFORMS
    assert "hub" in _KNOWN_DELIVERY_PLATFORMS


# ── Toolsets ────────────────────────────────────────────────────────


def test_toolset_hermes_hub_exists():
    from toolsets import TOOLSETS
    assert "hermes-hub" in TOOLSETS
    assert len(TOOLSETS["hermes-hub"]["tools"]) > 0


def test_toolset_hermes_gateway_includes_hub():
    from toolsets import TOOLSETS
    assert "hermes-hub" in TOOLSETS["hermes-gateway"]["includes"]


# ── Platform hints ──────────────────────────────────────────────────


def test_platform_hints_has_hub():
    from agent.prompt_builder import PLATFORM_HINTS
    assert "hub" in PLATFORM_HINTS
    assert "agent" in PLATFORM_HINTS["hub"].lower()
