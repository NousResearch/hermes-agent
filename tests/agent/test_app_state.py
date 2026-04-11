# tests/agent/test_app_state.py
import pytest
from agent.app_state import AppState


class TestAppState:
    """Tests for the AppState dataclass."""

    def test_can_be_instantiated_with_keyword_args(self):
        """AppState can be created with keyword arguments."""
        state = AppState(
            model="gpt-4",
            provider="openai",
            session_id="test-session-123",
            agent_name="test-agent",
            max_iterations=50,
            verbose_logging=True,
        )
        assert state.model == "gpt-4"
        assert state.provider == "openai"
        assert state.session_id == "test-session-123"
        assert state.agent_name == "test-agent"
        assert state.max_iterations == 50
        assert state.verbose_logging is True

    def test_fields_are_accessible(self):
        """All major fields are accessible after instantiation."""
        state = AppState()

        # Identity fields
        assert state.model == ""
        assert state.provider == ""
        assert state.session_id == ""
        assert state.agent_name == ""

        # Runtime fields
        assert state.max_iterations == 90
        assert state.tool_delay == 1.0

        # Feature flags
        assert state.compression_enabled is True
        assert state.checkpoints_enabled is False

        # Token tracking
        assert state.total_cost == 0.0

    def test_default_values(self):
        """AppState has correct default values."""
        state = AppState()

        assert state.api_mode == "chat_completions"
        assert state.max_iterations == 90
        assert state.tool_delay == 1.0
        assert state.save_trajectories is False
        assert state.verbose_logging is False
        assert state.quiet_mode is False
        assert state.persist_session is True
        assert state.compression_enabled is True
        assert state.max_tool_call_iterations == 90
        assert state.tool_delay_type == "fixed"
        assert state.return_context is False
        assert state.no_prompt_override is False
        assert state.enable_mention_suggestions is True
        assert state.use_progressive_summarization is False
        assert state.enable_flask_agent is False

    def test_list_fields_default_to_empty_list(self):
        """List fields are initialized as empty lists via field(default_factory=list)."""
        state = AppState()

        assert state.acp_args == []
        assert state.extra_kwargs == {}
        assert state._session_messages == []
        assert state._fallback_chain == []

    def test_set_fields_default_to_none(self):
        """Set fields are initialized as None."""
        state = AppState()

        assert state.providers_allowed is None
        assert state.providers_ignored is None
        assert state.providers_order is None
        assert state.provider_sort is None
        assert state.enabled_toolsets is None
        assert state.disabled_toolsets is None

    def test_callback_fields_default_to_none(self):
        """Callback fields default to None."""
        state = AppState()

        assert state.stream_delta_callback is None
        assert state.tool_progress_callback is None
        assert state.tool_start_callback is None
        assert state.tool_complete_callback is None
        assert state.clarify_callback is None
        assert state.reasoning_callback is None
        assert state.thinking_callback is None
        assert state.step_callback is None
        assert state.status_callback is None
        assert state.tool_gen_callback is None
        assert state.background_review_callback is None
        assert state.message_callback is None

    def test_retry_counters_default_to_zero(self):
        """Retry counter fields default to 0."""
        state = AppState()

        assert state._invalid_tool_retries == 0
        assert state._invalid_json_retries == 0
        assert state._empty_content_retries == 0
        assert state._incomplete_scratchpad_retries == 0
        assert state._codex_incomplete_retries == 0

    def test_budget_thresholds_default_values(self):
        """Budget threshold fields have correct defaults."""
        state = AppState()

        assert state._budget_caution_threshold == 0.7
        assert state._budget_warning_threshold == 0.9
        assert state._budget_pressure_enabled is True
        assert state._context_pressure_warned is False

    def test_token_tracking_defaults(self):
        """Token and cost tracking fields default to zero."""
        state = AppState()

        assert state.total_cost == 0.0
        assert state._total_tokens == 0
        assert state._prompt_tokens == 0
        assert state._completion_tokens == 0
        assert state.session_total_tokens == 0
        assert state.session_input_tokens == 0
        assert state.session_output_tokens == 0
        assert state.session_prompt_tokens == 0
        assert state.session_completion_tokens == 0
        assert state.session_cache_read_tokens == 0
        assert state.session_cache_write_tokens == 0
        assert state.session_reasoning_tokens == 0
        assert state.session_api_calls == 0
        assert state.session_estimated_cost_usd == 0.0

    def test_interrupt_flags_default_to_false(self):
        """Interrupt-related flags default to False."""
        state = AppState()

        assert state._interrupt_requested is False
        assert state._waiting_for_user_input is False
        assert state._waiting_for_approval is False
        assert state._user_confirmed is False

    def test_memory_fields_default(self):
        """Memory-related fields have correct defaults."""
        state = AppState()

        assert state._memory_enabled is False
        assert state._user_profile_enabled is False
        assert state._memory_nudge_interval == 10
        assert state._memory_flush_min_turns == 6
        assert state._turns_since_memory == 0
        assert state._iters_since_skill == 0
        assert state._skill_nudge_interval == 10

    def test_context_compressor_default(self):
        """Context compressor fields default correctly."""
        state = AppState()

        assert state.context_compressor is None
        assert state._context_compressor is None
        assert state.compression_enabled is True
        assert state._user_turn_count == 0

    def test_has_80_plus_fields(self):
        """AppState has a large number of fields (80+ as specified)."""
        state = AppState()
        # Count all public and private attributes that are actual fields
        field_count = len([k for k in dir(state) if not k.startswith('__')])
        # The dataclass should have well over 80 fields
        assert field_count > 80, f"Expected >80 fields, got {field_count}"

    def test_multiple_instances_independent(self):
        """Multiple AppState instances are independent."""
        state1 = AppState(model="model-1", max_iterations=10)
        state2 = AppState(model="model-2", max_iterations=20)

        assert state1.model == "model-1"
        assert state1.max_iterations == 10
        assert state2.model == "model-2"
        assert state2.max_iterations == 20

    def test_field_assignment_mutable_default_issue(self):
        """Mutable default values are handled correctly via field(default_factory=...)."""
        state = AppState()

        # These use field(default_factory=...) so they should be independent lists/dicts
        state.acp_args.append("--test")
        state.extra_kwargs["key"] = "value"

        # Creating a new instance should not be affected
        state2 = AppState()
        assert state2.acp_args == []
        assert state2.extra_kwargs == {}

    def test_thinking_budget_tokens_default(self):
        """Thinking budget tokens defaults to 0."""
        state = AppState()
        assert state.thinking_budget_tokens == 0

    def test_reasoning_settings_default(self):
        """Reasoning settings defaults to None."""
        state = AppState()
        assert state.reasoning_settings is None

    def test_max_tokens_default(self):
        """max_tokens defaults to None."""
        state = AppState()
        assert state.max_tokens is None

    def test_mcp_servers_default(self):
        """mcpServers defaults to None."""
        state = AppState()
        assert state.mcpServers is None
