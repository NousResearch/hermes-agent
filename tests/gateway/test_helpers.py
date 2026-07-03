"""Tests for gateway platform helpers."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from gateway.platforms.helpers import TextModelPicker
from gateway.platforms.base import SendResult


class TestTextModelPicker:
    """Tests for the TextModelPicker helper class."""

    @pytest.fixture
    def mock_adapter(self):
        """Create a mock adapter for testing."""
        adapter = MagicMock()
        adapter.name = "test"
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="msg-123"))
        adapter.format_message = lambda x: x
        return adapter

    @pytest.fixture
    def picker(self, mock_adapter):
        """Create a TextModelPicker instance."""
        return TextModelPicker(mock_adapter)

    def test_build_provider_text_format(self, picker):
        """Test provider list text formatting."""
        providers = [
            {"name": "Alibaba", "slug": "alibaba", "models": ["qwen-v3"], "is_current": True},
            {"name": "DeepSeek", "slug": "deepseek", "models": ["ds-v4"], "is_current": False},
        ]
        text = TextModelPicker._build_provider_text(providers, "qwen-v3", "alibaba")
        
        assert "Current: qwen-v3" in text
        assert "1. Alibaba (1 models) [current]" in text
        assert "2. DeepSeek (1 models)" in text
        assert "or 0 to cancel" in text

    def test_build_provider_text_empty_current(self, picker):
        """Test provider text with empty current model."""
        providers = [
            {"name": "Alibaba", "slug": "alibaba", "models": ["qwen-v3"], "is_current": False},
        ]
        text = TextModelPicker._build_provider_text(providers, "", "alibaba")
        assert "Current: unknown" in text

    def test_build_model_text_format(self, picker):
        """Test model list text formatting."""
        models = ["model-a", "model-b", "model-c"]
        text = TextModelPicker._build_model_text(models, "TestProvider")
        
        assert "Models available on TestProvider:" in text
        assert "1. model-a" in text
        assert "2. model-b" in text
        assert "3. model-c" in text
        assert "or 0 to cancel" in text

    @pytest.mark.asyncio
    async def test_send_stores_state(self, picker, mock_adapter):
        """Test send_model_picker stores state correctly."""
        providers = [
            {"name": "Alibaba", "slug": "alibaba", "models": ["qwen-v3"], "is_current": True},
        ]
        
        async def on_model_selected(chat_id, model, provider_slug):
            return f"Switched to {model}"
        
        result = await picker.send(
            chat_id="chat-123",
            providers=providers,
            current_model="qwen-v3",
            current_provider="alibaba",
            session_key="sess-abc",
            on_model_selected=on_model_selected,
        )
        
        assert result.success is True
        assert "chat-123" in picker._state
        state = picker._state["chat-123"]
        assert state["stage"] == "provider"
        assert state["current_model"] == "qwen-v3"
        assert state["on_model_selected"] is on_model_selected
        mock_adapter.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_response_no_state_returns_none(self, picker):
        """Test handle_response returns None when no active state."""
        result = await picker.handle_response("chat-123", "1")
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_response_provider_selection_advances_to_model(self, picker, mock_adapter):
        """Test selecting a provider with multiple models advances to model stage."""
        providers = [
            {"name": "Alibaba", "slug": "alibaba", "models": ["qwen-v3", "qwen-v2"], "is_current": True},
            {"name": "DeepSeek", "slug": "deepseek", "models": ["ds-v4"], "is_current": False},
        ]
        
        async def on_model_selected(chat_id, model, provider_slug):
            return f"Switched to {model}"
        
        # Setup initial state
        picker._state["chat-123"] = {
            "stage": "provider",
            "providers": providers,
            "current_model": "qwen-v3",
            "on_model_selected": on_model_selected,
        }
        
        result = await picker.handle_response("chat-123", "1")
        assert result == "picker_consumed"
        assert picker._state["chat-123"]["stage"] == "model"
        assert picker._state["chat-123"]["selected_provider_slug"] == "alibaba"

    @pytest.mark.asyncio
    async def test_handle_response_single_model_auto_switches(self, picker, mock_adapter):
        """Test selecting a provider with single model auto-switches."""
        async def on_model_selected(chat_id, model, provider_slug):
            return f"Switched to {model} via {provider_slug}"
        
        picker._state["chat-123"] = {
            "stage": "provider",
            "providers": [
                {"name": "DeepSeek", "slug": "deepseek", "models": ["ds-v4"], "is_current": False},
            ],
            "current_model": "old-model",
            "on_model_selected": on_model_selected,
        }
        
        result = await picker.handle_response("chat-123", "1")
        assert result == "picker_consumed"
        assert "chat-123" not in picker._state  # State cleared after switch
        mock_adapter.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_response_model_selection_completes(self, picker, mock_adapter):
        """Test selecting a model completes the flow."""
        async def on_model_selected(chat_id, model, provider_slug):
            return f"Switched to {model} via {provider_slug}"
        
        picker._state["chat-123"] = {
            "stage": "model",
            "selected_provider_slug": "alibaba",
            "selected_provider_name": "Alibaba",
            "selected_provider_models": ["qwen-v3", "qwen-v2"],
            "on_model_selected": on_model_selected,
        }
        
        result = await picker.handle_response("chat-123", "2")
        assert result == "picker_consumed"
        assert "chat-123" not in picker._state  # State cleared
        mock_adapter.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_response_cancel_with_zero(self, picker, mock_adapter):
        """Test typing 0 cancels the picker."""
        picker._state["chat-123"] = {
            "stage": "provider",
            "providers": [],
            "current_model": "old",
            "on_model_selected": AsyncMock(),
        }
        
        result = await picker.handle_response("chat-123", "0")
        assert result == "picker_cancelled"
        assert "chat-123" not in picker._state
        mock_adapter.send.assert_called_once()
        assert "cancelled" in mock_adapter.send.call_args[0][1].lower()

    @pytest.mark.asyncio
    async def test_handle_response_cancel_with_quit(self, picker, mock_adapter):
        """Test typing quit cancels the picker."""
        picker._state["chat-123"] = {
            "stage": "provider",
            "providers": [],
            "current_model": "old",
            "on_model_selected": AsyncMock(),
        }
        
        result = await picker.handle_response("chat-123", "quit")
        assert result == "picker_cancelled"
        assert "chat-123" not in picker._state

    @pytest.mark.asyncio
    async def test_handle_response_cancel_with_empty(self, picker, mock_adapter):
        """Test empty message cancels the picker."""
        picker._state["chat-123"] = {
            "stage": "provider",
            "providers": [],
            "current_model": "old",
            "on_model_selected": AsyncMock(),
        }
        
        result = await picker.handle_response("chat-123", "")
        assert result == "picker_cancelled"
        assert "chat-123" not in picker._state

    @pytest.mark.asyncio
    async def test_handle_response_invalid_selection(self, picker, mock_adapter):
        """Test invalid selection sends error but keeps state."""
        picker._state["chat-123"] = {
            "stage": "provider",
            "providers": [
                {"name": "Alibaba", "slug": "alibaba", "models": ["qwen"], "is_current": True},
            ],
            "current_model": "old",
            "on_model_selected": AsyncMock(),
        }
        
        result = await picker.handle_response("chat-123", "99")
        assert result == "picker_consumed"
        assert picker._state["chat-123"]["stage"] == "provider"  # State preserved
        mock_adapter.send.assert_called_once()
        assert "Invalid selection" in mock_adapter.send.call_args[0][1]

    @pytest.mark.asyncio
    async def test_handle_response_slug_selection(self, picker, mock_adapter):
        """Test selecting by provider slug works."""
        async def on_model_selected(chat_id, model, provider_slug):
            return f"Switched"
        
        picker._state["chat-123"] = {
            "stage": "provider",
            "providers": [
                {"name": "Alibaba", "slug": "alibaba", "models": ["qwen"], "is_current": True},
            ],
            "current_model": "old",
            "on_model_selected": on_model_selected,
        }
        
        result = await picker.handle_response("chat-123", "alibaba")
        assert result == "picker_consumed"

    @pytest.mark.asyncio
    async def test_handle_response_model_by_name(self, picker, mock_adapter):
        """Test selecting model by exact name works."""
        async def on_model_selected(chat_id, model, provider_slug):
            return f"Switched"
        
        picker._state["chat-123"] = {
            "stage": "model",
            "selected_provider_slug": "alibaba",
            "selected_provider_name": "Alibaba",
            "selected_provider_models": ["qwen-v3", "qwen-v2"],
            "on_model_selected": on_model_selected,
        }
        
        result = await picker.handle_response("chat-123", "qwen-v3")
        assert result == "picker_consumed"
        assert "chat-123" not in picker._state

    @pytest.mark.asyncio
    async def test_handle_response_model_case_insensitive(self, picker, mock_adapter):
        """Test model selection is case insensitive."""
        async def on_model_selected(chat_id, model, provider_slug):
            return f"Switched"
        
        picker._state["chat-123"] = {
            "stage": "model",
            "selected_provider_slug": "alibaba",
            "selected_provider_name": "Alibaba",
            "selected_provider_models": ["Qwen-V3", "qwen-v2"],
            "on_model_selected": on_model_selected,
        }
        
        result = await picker.handle_response("chat-123", "QWEN-V3")
        assert result == "picker_consumed"

    def test_is_active(self, picker):
        """Test is_active method."""
        assert picker.is_active("chat-123") is False
        picker._state["chat-123"] = {"stage": "provider"}
        assert picker.is_active("chat-123") is True

    def test_clear_state(self, picker):
        """Test clear_state method."""
        picker._state["chat-123"] = {"stage": "provider"}
        picker.clear_state("chat-123")
        assert "chat-123" not in picker._state

    def test_exit_keywords_coverage(self):
        """Test all exit keywords are recognized."""
        keywords = ["q", "quit", "exit", "cancel", "done", "0"]
        for kw in keywords:
            assert kw in TextModelPicker.EXIT_KEYWORDS
