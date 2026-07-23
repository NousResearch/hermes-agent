"""Tests for gateway platform helpers."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.base import SendResult
from gateway.platforms.helpers import TextModelPicker


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

    # ── Text formatting ──────────────────────────────────────────────

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

    # ── send() ──────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_send_stores_state(self, picker, mock_adapter):
        """Test send_model_picker stores state correctly, keyed by session_key."""
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
            requester_user_id="user-A",
        )

        assert result.success is True
        # State is keyed by session_key, NOT chat_id
        assert "sess-abc" in picker._state
        assert "chat-123" not in picker._state
        state = picker._state["sess-abc"]
        assert state["stage"] == "provider"
        assert state["chat_id"] == "chat-123"
        assert state["current_model"] == "qwen-v3"
        assert state["on_model_selected"] is on_model_selected
        assert state["requester_user_id"] == "user-A"
        assert "expires_at" in state
        mock_adapter.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_failure_does_not_register_state(self, picker, mock_adapter):
        """send() must not register state when adapter.send() fails.

        Without this guard, the gateway falls back to its static text list
        but the stale state would intercept the next ordinary message.
        (hermes-sweeper review #48199)
        """
        mock_adapter.send = AsyncMock(return_value=SendResult(success=False, error="network"))

        providers = [
            {"name": "Alibaba", "slug": "alibaba", "models": ["qwen-v3"], "is_current": True},
        ]

        async def on_model_selected(chat_id, model, provider_slug):
            return "ok"

        result = await picker.send(
            chat_id="chat-123",
            providers=providers,
            current_model="qwen-v3",
            current_provider="alibaba",
            session_key="sess-abc",
            on_model_selected=on_model_selected,
        )

        assert result.success is False
        # Critical: no state registered on failure
        assert "sess-abc" not in picker._state

    # ── handle_response() — no state / fall-through ──────────────────

    @pytest.mark.asyncio
    async def test_handle_response_no_state_returns_none(self, picker):
        """Test handle_response returns None when no active state."""
        result = await picker.handle_response("sess-abc", "1")
        assert result is None

    # ── handle_response() — expiry ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_handle_response_expired_state_returns_none(self, picker, mock_adapter):
        """Expired state must be discarded and fall through to the agent.

        (hermes-sweeper review #48199)
        """
        picker._state["sess-abc"] = {
            "stage": "provider",
            "providers": [],
            "chat_id": "chat-123",
            "current_model": "old",
            "on_model_selected": AsyncMock(),
            "requester_user_id": "user-A",
            "expires_at": time.monotonic() - 1,  # already expired
        }

        result = await picker.handle_response("sess-abc", "1", requester_user_id="user-A")
        assert result is None
        assert "sess-abc" not in picker._state  # expired state cleaned up
        mock_adapter.send.assert_not_called()  # no message sent

    # ── handle_response() — requester validation ─────────────────────

    @pytest.mark.asyncio
    async def test_handle_response_requester_mismatch_returns_none(self, picker, mock_adapter):
        """A reply from a different user in the same group session must fall through.

        (hermes-sweeper review #48199 — group isolation)
        """
        picker._state["sess-abc"] = {
            "stage": "provider",
            "providers": [{"name": "Alibaba", "slug": "alibaba", "models": ["qwen"], "is_current": True}],
            "chat_id": "chat-123",
            "current_model": "old",
            "on_model_selected": AsyncMock(),
            "requester_user_id": "user-A",
            "expires_at": time.monotonic() + 300,
        }

        # User-B tries to interact with user-A's picker
        result = await picker.handle_response("sess-abc", "1", requester_user_id="user-B")
        assert result is None
        # State preserved — user-A can still interact
        assert "sess-abc" in picker._state
        mock_adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_response_no_requester_stored_allows_anonymous(self, picker, mock_adapter):
        """When no requester_user_id was stored, any user may interact (DM case)."""
        picker._state["sess-abc"] = {
            "stage": "provider",
            "providers": [{"name": "Alibaba", "slug": "alibaba", "models": ["qwen"], "is_current": True}],
            "chat_id": "chat-123",
            "current_model": "old",
            "on_model_selected": AsyncMock(),
            "requester_user_id": "",  # not set — DM or legacy
            "expires_at": time.monotonic() + 300,
        }

        result = await picker.handle_response("sess-abc", "1", requester_user_id="anyone")
        # Should proceed normally (no requester mismatch check when stored is empty)
        assert result is not None

    # ── handle_response() — provider stage ──────────────────────────

    @pytest.mark.asyncio
    async def test_handle_response_provider_selection_advances_to_model(self, picker, mock_adapter):
        """Test selecting a provider with multiple models advances to model stage."""
        providers = [
            {"name": "Alibaba", "slug": "alibaba", "models": ["qwen-v3", "qwen-v2"], "is_current": True},
            {"name": "DeepSeek", "slug": "deepseek", "models": ["ds-v4"], "is_current": False},
        ]

        async def on_model_selected(chat_id, model, provider_slug):
            return f"Switched to {model}"

        picker._state["sess-abc"] = {
            "stage": "provider",
            "providers": providers,
            "chat_id": "chat-123",
            "current_model": "qwen-v3",
            "on_model_selected": on_model_selected,
            "requester_user_id": "user-A",
            "expires_at": time.monotonic() + 300,
        }

        result = await picker.handle_response("sess-abc", "1", requester_user_id="user-A")
        assert result == "picker_consumed"
        assert picker._state["sess-abc"]["stage"] == "model"
        assert picker._state["sess-abc"]["selected_provider_slug"] == "alibaba"

    @pytest.mark.asyncio
    async def test_handle_response_invalid_provider_selection(self, picker, mock_adapter):
        """Test invalid selection sends error but keeps state."""
        picker._state["sess-abc"] = {
            "stage": "provider",
            "providers": [
                {"name": "Alibaba", "slug": "alibaba", "models": ["qwen"], "is_current": True},
            ],
            "chat_id": "chat-123",
            "current_model": "old",
            "on_model_selected": AsyncMock(),
            "requester_user_id": "",
            "expires_at": time.monotonic() + 300,
        }

        result = await picker.handle_response("sess-abc", "99")
        assert result == "picker_consumed"
        assert picker._state["sess-abc"]["stage"] == "provider"  # State preserved
        mock_adapter.send.assert_called_once()
        assert "Invalid selection" in mock_adapter.send.call_args[0][1]

    @pytest.mark.asyncio
    async def test_handle_response_slug_selection(self, picker, mock_adapter):
        """Test selecting by provider slug works."""
        async def on_model_selected(chat_id, model, provider_slug):
            return "Switched"

        picker._state["sess-abc"] = {
            "stage": "provider",
            "providers": [
                {"name": "Alibaba", "slug": "alibaba", "models": ["qwen"], "is_current": True},
            ],
            "chat_id": "chat-123",
            "current_model": "old",
            "on_model_selected": on_model_selected,
            "requester_user_id": "",
            "expires_at": time.monotonic() + 300,
        }

        result = await picker.handle_response("sess-abc", "alibaba")
        assert result == "picker_consumed"

    # ── handle_response() — model stage ─────────────────────────────

    @pytest.mark.asyncio
    async def test_handle_response_model_selection_completes(self, picker, mock_adapter):
        """Test selecting a model completes the flow (non-expensive model)."""
        async def on_model_selected(chat_id, model, provider_slug):
            return f"Switched to {model} via {provider_slug}"

        picker._state["sess-abc"] = {
            "stage": "model",
            "chat_id": "chat-123",
            "selected_provider_slug": "alibaba",
            "selected_provider_name": "Alibaba",
            "selected_provider_models": ["qwen-v3", "qwen-v2"],
            "on_model_selected": on_model_selected,
            "current_model": "old",
            "requester_user_id": "",
            "expires_at": time.monotonic() + 300,
        }

        # Patch expensive_model_warning to return None (not expensive)
        with patch("hermes_cli.model_cost_guard.expensive_model_warning", return_value=None):
            result = await picker.handle_response("sess-abc", "2")
        assert result == "picker_consumed"
        assert "sess-abc" not in picker._state  # State cleared
        mock_adapter.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_response_model_by_name(self, picker, mock_adapter):
        """Test selecting model by exact name works (non-expensive)."""
        async def on_model_selected(chat_id, model, provider_slug):
            return "Switched"

        picker._state["sess-abc"] = {
            "stage": "model",
            "chat_id": "chat-123",
            "selected_provider_slug": "alibaba",
            "selected_provider_name": "Alibaba",
            "selected_provider_models": ["qwen-v3", "qwen-v2"],
            "on_model_selected": on_model_selected,
            "current_model": "old",
            "requester_user_id": "",
            "expires_at": time.monotonic() + 300,
        }

        with patch("hermes_cli.model_cost_guard.expensive_model_warning", return_value=None):
            result = await picker.handle_response("sess-abc", "qwen-v3")
        assert result == "picker_consumed"
        assert "sess-abc" not in picker._state

    @pytest.mark.asyncio
    async def test_handle_response_model_case_insensitive(self, picker, mock_adapter):
        """Test model selection is case insensitive (non-expensive)."""
        async def on_model_selected(chat_id, model, provider_slug):
            return "Switched"

        picker._state["sess-abc"] = {
            "stage": "model",
            "chat_id": "chat-123",
            "selected_provider_slug": "alibaba",
            "selected_provider_name": "Alibaba",
            "selected_provider_models": ["Qwen-V3", "qwen-v2"],
            "on_model_selected": on_model_selected,
            "current_model": "old",
            "requester_user_id": "",
            "expires_at": time.monotonic() + 300,
        }

        with patch("hermes_cli.model_cost_guard.expensive_model_warning", return_value=None):
            result = await picker.handle_response("sess-abc", "QWEN-V3")
        assert result == "picker_consumed"

    # ── handle_response() — single-model auto-switch ─────────────────

    @pytest.mark.asyncio
    async def test_handle_response_single_model_auto_switches(self, picker, mock_adapter):
        """Test selecting a provider with single model auto-switches (non-expensive)."""
        async def on_model_selected(chat_id, model, provider_slug):
            return f"Switched to {model} via {provider_slug}"

        picker._state["sess-abc"] = {
            "stage": "provider",
            "providers": [
                {"name": "DeepSeek", "slug": "deepseek", "models": ["ds-v4"], "is_current": False},
            ],
            "chat_id": "chat-123",
            "current_model": "old-model",
            "on_model_selected": on_model_selected,
            "requester_user_id": "",
            "expires_at": time.monotonic() + 300,
        }

        with patch("hermes_cli.model_cost_guard.expensive_model_warning", return_value=None):
            result = await picker.handle_response("sess-abc", "1")
        assert result == "picker_consumed"
        assert "sess-abc" not in picker._state  # State cleared after switch
        mock_adapter.send.assert_called_once()

    # ── handle_response() — expensive-model confirmation ────────────

    @pytest.mark.asyncio
    async def test_expensive_model_enters_confirm_stage(self, picker, mock_adapter):
        """Selecting an expensive model must enter a confirmation stage.

        (hermes-sweeper review #48199)
        """
        from decimal import Decimal
        from hermes_cli.model_cost_guard import ExpensiveModelWarning

        fake_warning = ExpensiveModelWarning(
            model="gpt-5.5-pro",
            provider="openai",
            input_cost_per_million=Decimal("50"),
            output_cost_per_million=Decimal("200"),
            source="models.dev",
            message="!!! EXPENSIVE MODEL WARNING !!!",
        )

        picker._state["sess-abc"] = {
            "stage": "model",
            "chat_id": "chat-123",
            "selected_provider_slug": "openai",
            "selected_provider_name": "OpenAI",
            "selected_provider_models": ["gpt-5.5-pro", "gpt-4o"],
            "on_model_selected": AsyncMock(return_value="Switched!"),
            "current_model": "old",
            "requester_user_id": "",
            "expires_at": time.monotonic() + 300,
        }

        with patch(
            "hermes_cli.model_cost_guard.expensive_model_warning",
            return_value=fake_warning,
        ):
            result = await picker.handle_response("sess-abc", "1")

        assert result == "picker_consumed"
        # State should be in confirm stage
        assert picker._state["sess-abc"]["stage"] == "confirm"
        assert picker._state["sess-abc"]["pending_model"] == "gpt-5.5-pro"
        # on_model_selected should NOT have been called yet
        picker._state["sess-abc"]["on_model_selected"].assert_not_called()

    @pytest.mark.asyncio
    async def test_expensive_model_confirm_proceeds(self, picker, mock_adapter):
        """User confirming in the confirm stage completes the switch."""
        on_selected = AsyncMock(return_value="Switched to gpt-5.5-pro")

        picker._state["sess-abc"] = {
            "stage": "confirm",
            "chat_id": "chat-123",
            "pending_model": "gpt-5.5-pro",
            "pending_provider_slug": "openai",
            "pending_provider_name": "OpenAI",
            "on_model_selected": on_selected,
            "current_model": "old",
            "requester_user_id": "",
            "expires_at": time.monotonic() + 300,
        }

        result = await picker.handle_response("sess-abc", "y")
        assert result == "picker_consumed"
        assert "sess-abc" not in picker._state  # State cleared
        on_selected.assert_called_once()
        # Check it was called with the pending model
        assert on_selected.call_args[0][1] == "gpt-5.5-pro"

    @pytest.mark.asyncio
    async def test_expensive_model_confirm_cancel(self, picker, mock_adapter):
        """User declining in the confirm stage cancels the switch."""
        on_selected = AsyncMock(return_value="should not be called")

        picker._state["sess-abc"] = {
            "stage": "confirm",
            "chat_id": "chat-123",
            "pending_model": "gpt-5.5-pro",
            "pending_provider_slug": "openai",
            "pending_provider_name": "OpenAI",
            "on_model_selected": on_selected,
            "current_model": "old",
            "requester_user_id": "",
            "expires_at": time.monotonic() + 300,
        }

        result = await picker.handle_response("sess-abc", "n")
        assert result == "picker_cancelled"
        assert "sess-abc" not in picker._state
        on_selected.assert_not_called()

    # ── handle_response() — cancel conditions ───────────────────────

    @pytest.mark.asyncio
    async def test_handle_response_cancel_with_zero(self, picker, mock_adapter):
        """Test typing 0 cancels the picker."""
        picker._state["sess-abc"] = {
            "stage": "provider",
            "providers": [],
            "chat_id": "chat-123",
            "current_model": "old",
            "on_model_selected": AsyncMock(),
            "requester_user_id": "",
            "expires_at": time.monotonic() + 300,
        }

        result = await picker.handle_response("sess-abc", "0")
        assert result == "picker_cancelled"
        assert "sess-abc" not in picker._state
        mock_adapter.send.assert_called_once()
        assert "cancelled" in mock_adapter.send.call_args[0][1].lower()

    @pytest.mark.asyncio
    async def test_handle_response_cancel_with_quit(self, picker, mock_adapter):
        """Test typing quit cancels the picker."""
        picker._state["sess-abc"] = {
            "stage": "provider",
            "providers": [],
            "chat_id": "chat-123",
            "current_model": "old",
            "on_model_selected": AsyncMock(),
            "requester_user_id": "",
            "expires_at": time.monotonic() + 300,
        }

        result = await picker.handle_response("sess-abc", "quit")
        assert result == "picker_cancelled"
        assert "sess-abc" not in picker._state

    @pytest.mark.asyncio
    async def test_handle_response_cancel_with_empty(self, picker, mock_adapter):
        """Test empty message cancels the picker."""
        picker._state["sess-abc"] = {
            "stage": "provider",
            "providers": [],
            "chat_id": "chat-123",
            "current_model": "old",
            "on_model_selected": AsyncMock(),
            "requester_user_id": "",
            "expires_at": time.monotonic() + 300,
        }

        result = await picker.handle_response("sess-abc", "")
        assert result == "picker_cancelled"
        assert "sess-abc" not in picker._state

    # ── Utility methods ──────────────────────────────────────────────

    def test_is_active(self, picker):
        """Test is_active method."""
        assert picker.is_active("sess-abc") is False
        picker._state["sess-abc"] = {"stage": "provider"}
        assert picker.is_active("sess-abc") is True

    def test_clear_state(self, picker):
        """Test clear_state method."""
        picker._state["sess-abc"] = {"stage": "provider"}
        picker.clear_state("sess-abc")
        assert "sess-abc" not in picker._state

    def test_exit_keywords_coverage(self):
        """Test all exit keywords are recognized."""
        keywords = ["q", "quit", "exit", "cancel", "done", "0"]
        for kw in keywords:
            assert kw in TextModelPicker.EXIT_KEYWORDS

    def test_confirm_keywords_coverage(self):
        """Test all confirm keywords are recognized."""
        keywords = ["y", "yes", "confirm", "ok", "1"]
        for kw in keywords:
            assert kw in TextModelPicker.CONFIRM_KEYWORDS

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_stale_entries(self, picker):
        """Test that _cleanup_expired removes expired states."""
        picker._state["sess-old"] = {
            "stage": "provider",
            "expires_at": time.monotonic() - 100,  # expired
        }
        picker._state["sess-fresh"] = {
            "stage": "provider",
            "expires_at": time.monotonic() + 300,  # fresh
        }

        picker._cleanup_expired()
        assert "sess-old" not in picker._state
        assert "sess-fresh" in picker._state

    # ── DM-only guard ─────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_send_group_chat_returns_failure_without_registering_state(self, picker, mock_adapter):
        """send() in a group chat must fail fast and never register state.

        The gateway reads ``result.success`` and falls back to its static
        text list when the picker is unavailable — so we must NOT leave a
        stale state that would hijack the next ordinary message.
        """
        providers = [
            {"name": "Alibaba", "slug": "alibaba", "models": ["qwen-v3"], "is_current": True},
        ]

        async def on_model_selected(chat_id, model, provider_slug):
            return "ok"

        result = await picker.send(
            chat_id="chat-123",
            providers=providers,
            current_model="qwen-v3",
            current_provider="alibaba",
            session_key="sess-abc",
            on_model_selected=on_model_selected,
            chat_type="group",
        )

        assert result.success is False
        assert "DM-only" in (result.error or "")
        # Adapter.send must NOT have been called — fail before doing real work.
        mock_adapter.send.assert_not_called()
        # And no state registered.
        assert "sess-abc" not in picker._state

    @pytest.mark.asyncio
    async def test_send_dm_chat_succeeds(self, picker, mock_adapter):
        """Explicit chat_type='dm' produces a normal picker send."""
        providers = [
            {"name": "Alibaba", "slug": "alibaba", "models": ["qwen-v3"], "is_current": True},
        ]

        async def on_model_selected(chat_id, model, provider_slug):
            return "ok"

        result = await picker.send(
            chat_id="chat-123",
            providers=providers,
            current_model="qwen-v3",
            current_provider="alibaba",
            session_key="sess-abc",
            on_model_selected=on_model_selected,
            chat_type="dm",
        )

        assert result.success is True
        assert "sess-abc" in picker._state

    @pytest.mark.asyncio
    async def test_send_direct_chat_type_accepted_as_dm(self, picker, mock_adapter):
        """chat_type='direct' is in the DM set alongside 'dm'/'private'."""
        providers = [
            {"name": "Alibaba", "slug": "alibaba", "models": ["qwen-v3"], "is_current": True},
        ]

        async def on_model_selected(chat_id, model, provider_slug):
            return "ok"

        result = await picker.send(
            chat_id="chat-123",
            providers=providers,
            current_model="qwen-v3",
            current_provider="alibaba",
            session_key="sess-abc",
            on_model_selected=on_model_selected,
            chat_type="direct",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_empty_chat_type_succeeds_for_backward_compat(self, picker, mock_adapter):
        """When chat_type is empty (legacy callers), picker still works.

        Empty string is in _DM_CHAT_TYPES — this preserves backward
        compatibility for adapters that don't pass chat_type.
        """
        providers = [
            {"name": "Alibaba", "slug": "alibaba", "models": ["qwen-v3"], "is_current": True},
        ]

        async def on_model_selected(chat_id, model, provider_slug):
            return "ok"

        result = await picker.send(
            chat_id="chat-123",
            providers=providers,
            current_model="qwen-v3",
            current_provider="alibaba",
            session_key="sess-abc",
            on_model_selected=on_model_selected,
            chat_type="",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_handle_response_group_chat_returns_none_without_touching_state(self, picker, mock_adapter):
        """handle_response in a group chat falls through before any state lookup.

        Even if a stale state somehow exists (e.g. from a future code path
        or a manual test setup), a non-DM scope must not consume messages.
        """
        picker._state["sess-abc"] = {
            "stage": "provider",
            "providers": [{"name": "Alibaba", "slug": "alibaba", "models": ["qwen"], "is_current": True}],
            "chat_id": "chat-123",
            "current_model": "old",
            "on_model_selected": AsyncMock(),
            "requester_user_id": "",
            "expires_at": time.monotonic() + 300,
        }

        result = await picker.handle_response(
            session_key="sess-abc", text="1", requester_user_id="user-A", chat_type="group",
        )
        assert result is None
        # State untouched.
        assert "sess-abc" in picker._state
        mock_adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_response_dm_chat_consumes_state_normally(self, picker, mock_adapter):
        """chat_type='dm' does not block — picker behaves as before."""
        picker._state["sess-abc"] = {
            "stage": "provider",
            "providers": [{"name": "Alibaba", "slug": "alibaba", "models": ["qwen"], "is_current": True}],
            "chat_id": "chat-123",
            "current_model": "old",
            "on_model_selected": AsyncMock(return_value="Switched"),
            "requester_user_id": "",
            "expires_at": time.monotonic() + 300,
        }

        with patch("hermes_cli.model_cost_guard.expensive_model_warning", return_value=None):
            result = await picker.handle_response(
                session_key="sess-abc", text="1", chat_type="dm",
            )
        assert result == "picker_consumed"

    @pytest.mark.asyncio
    async def test_handle_response_empty_chat_type_consumes_state_normally(self, picker, mock_adapter):
        """Empty chat_type (legacy callers) — picker behaves as before."""
        picker._state["sess-abc"] = {
            "stage": "provider",
            "providers": [{"name": "Alibaba", "slug": "alibaba", "models": ["qwen"], "is_current": True}],
            "chat_id": "chat-123",
            "current_model": "old",
            "on_model_selected": AsyncMock(return_value="Switched"),
            "requester_user_id": "",
            "expires_at": time.monotonic() + 300,
        }

        with patch("hermes_cli.model_cost_guard.expensive_model_warning", return_value=None):
            result = await picker.handle_response(
                session_key="sess-abc", text="1", chat_type="",
            )
        assert result == "picker_consumed"
