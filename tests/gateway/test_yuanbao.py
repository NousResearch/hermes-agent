"""Tests for Yuanbao platform adapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.platforms.yuanbao import YuanbaoAdapter
from gateway.platforms.base import SendResult, MessageEvent, MessageType
from gateway.config import PlatformConfig


def _make_config():
    """Create a test PlatformConfig for Yuanbao."""
    return PlatformConfig(
        enabled=True,
        token="test-token",
        extra={
            "app_id": "test-app-id",
            "app_secret": "test-app-secret",
            "bot_id": "test-bot-id",
        },
    )


def _make_adapter():
    """Create a YuanbaoAdapter for testing."""
    config = _make_config()
    adapter = YuanbaoAdapter(config)
    # Mock the outbound manager to avoid WS connection
    adapter._outbound = MagicMock()
    adapter._outbound.send_text = AsyncMock(return_value=SendResult(success=True, message_id="msg-123"))
    return adapter


class TestYuanbaoModelPicker:
    """Tests for the text-based model picker on Yuanbao."""

    @pytest.mark.asyncio
    async def test_send_model_picker_delegates_to_text_model_picker(self):
        """Test send_model_picker delegates to TextModelPicker."""
        adapter = _make_adapter()
        
        providers = [
            {"name": "Alibaba", "slug": "alibaba", "models": ["qwen-v3"], "is_current": True},
        ]
        
        async def on_model_selected(chat_id, model, provider_slug):
            return f"Switched to {model}"
        
        result = await adapter.send_model_picker(
            chat_id="chat-123",
            providers=providers,
            current_model="qwen-v3",
            current_provider="alibaba",
            session_key="sess-abc",
            on_model_selected=on_model_selected,
        )
        
        assert result.success is True
        assert "chat-123" in adapter._model_picker._state
        state = adapter._model_picker._state["chat-123"]
        assert state["stage"] == "provider"
        assert state["on_model_selected"] is on_model_selected

    @pytest.mark.asyncio
    async def test_handle_picker_response_delegates_to_text_model_picker(self):
        """Test _handle_picker_response delegates to TextModelPicker."""
        adapter = _make_adapter()
        
        # Setup state directly in the picker
        adapter._model_picker._state["chat-123"] = {
            "stage": "provider",
            "providers": [
                {"name": "Alibaba", "slug": "alibaba", "models": ["qwen-v3"], "is_current": True},
            ],
            "current_model": "old-model",
            "on_model_selected": AsyncMock(return_value="Switched"),
        }
        
        result = await adapter._handle_picker_response("chat-123", "1")
        assert result == "picker_consumed"

    @pytest.mark.asyncio
    async def test_handle_picker_response_no_state_returns_none(self):
        """Test _handle_picker_response returns None when no active picker."""
        adapter = _make_adapter()
        
        result = await adapter._handle_picker_response("chat-123", "1")
        assert result is None

    @pytest.mark.asyncio
    async def test_model_picker_initialized_in_init(self):
        """Test TextModelPicker is initialized in __init__."""
        adapter = _make_adapter()
        
        from gateway.platforms.helpers import TextModelPicker
        assert isinstance(adapter._model_picker, TextModelPicker)
        assert adapter._model_picker._adapter is adapter


class TestYuanbaoAdapterBasics:
    """Basic tests for Yuanbao adapter."""

    def test_adapter_creation(self):
        """Test YuanbaoAdapter can be created."""
        config = _make_config()
        adapter = YuanbaoAdapter(config)
        assert adapter is not None
        assert adapter.PLATFORM.value == "yuanbao"

    def test_max_text_chunk(self):
        """Test MAX_TEXT_CHUNK is set."""
        config = _make_config()
        adapter = YuanbaoAdapter(config)
        assert adapter.MAX_TEXT_CHUNK == 4000

    def test_splits_long_messages(self):
        """Test splits_long_messages flag is set."""
        config = _make_config()
        adapter = YuanbaoAdapter(config)
        assert adapter.splits_long_messages is True

    def test_format_message_exists(self):
        """Test format_message method exists."""
        config = _make_config()
        adapter = YuanbaoAdapter(config)
        # Yuanbao uses MarkdownProcessor.markdown_hint_system_prompt
        # but format_message is inherited from base
        assert hasattr(adapter, 'format_message')


class TestYuanbaoSendMethods:
    """Tests for Yuanbao send methods."""

    @pytest.mark.asyncio
    async def test_send_delegates_to_outbound(self):
        """Test send delegates to OutboundManager."""
        adapter = _make_adapter()
        
        result = await adapter.send("chat-123", "Hello")
        
        assert result.success is True
        adapter._outbound.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_model_picker_exists(self):
        """Test send_model_picker method exists and works."""
        adapter = _make_adapter()
        
        providers = [
            {"name": "Test", "slug": "test", "models": ["model-1"], "is_current": True},
        ]
        
        async def callback(chat_id, model, provider):
            return "Done"
        
        result = await adapter.send_model_picker(
            chat_id="chat-123",
            providers=providers,
            current_model="model-1",
            current_provider="test",
            session_key="sess",
            on_model_selected=callback,
        )
        
        assert result.success is True
