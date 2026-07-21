"""Unit tests for agent/domain_state.py — all 20 domain state dataclasses."""

import unittest
from agent.domain_state import (
    IdentityState,
    RuntimeState,
    FeatureState,
    CallbackState,
    OutputState,
    APIClientState,
    ProviderState,
    ToolState,
    MemoryState,
    SessionState,
    DatabaseState,
    TaskState,
    TokenState,
    InterruptState,
    ContextState,
    FallbackState,
    StreamState,
    CredentialState,
    ActivityState,
    ExtraState,
)


class TestIdentityState(unittest.TestCase):
    def test_default_values(self):
        s = IdentityState()
        self.assertEqual(s.model, "")
        self.assertEqual(s.provider, "")
        self.assertEqual(s.base_url, "")
        self.assertEqual(s.api_key, "")
        self.assertEqual(s.platform, "")
        self.assertEqual(s.session_id, "")
        self.assertEqual(s.agent_name, "")
        self.assertEqual(s.acp_command, "")
        self.assertEqual(s.acp_args, [])
        self.assertEqual(s.api_mode, "chat_completions")

    def test_custom_values(self):
        s = IdentityState(
            model="claude-3-opus",
            provider="anthropic",
            base_url="https://api.anthropic.com",
            platform="cli",
            session_id="sess-123",
            agent_name="hermes",
            acp_command="claude",
            acp_args=["--acp", "--stdio"],
            api_mode="anthropic_messages",
        )
        self.assertEqual(s.model, "claude-3-opus")
        self.assertEqual(s.api_mode, "anthropic_messages")
        self.assertEqual(s.acp_args, ["--acp", "--stdio"])

    def test_acp_args_is_list(self):
        s = IdentityState(acp_args=["--flag"])
        self.assertIsInstance(s.acp_args, list)


class TestRuntimeState(unittest.TestCase):
    def test_default_values(self):
        s = RuntimeState()
        self.assertEqual(s.max_iterations, 90)
        self.assertIsNone(s.iteration_budget)
        self.assertEqual(s.tool_delay, 1.0)
        self.assertEqual(s.tool_delay_type, "fixed")
        self.assertEqual(s.max_tool_call_iterations, 90)
        self.assertEqual(s.return_context, False)
        self.assertEqual(s.extra_kwargs, {})

    def test_retry_counters_initially_zero(self):
        s = RuntimeState()
        self.assertEqual(s._invalid_tool_retries, 0)
        self.assertEqual(s._invalid_json_retries, 0)
        self.assertEqual(s._empty_content_retries, 0)
        self.assertEqual(s._incomplete_scratchpad_retries, 0)
        self.assertEqual(s._codex_incomplete_retries, 0)

    def test_extra_kwargs_is_dict(self):
        s = RuntimeState(extra_kwargs={"timeout": 30})
        self.assertEqual(s.extra_kwargs["timeout"], 30)


class TestFeatureState(unittest.TestCase):
    def test_default_values(self):
        s = FeatureState()
        self.assertFalse(s.save_trajectories)
        self.assertFalse(s.verbose_logging)
        self.assertFalse(s.quiet_mode)
        self.assertEqual(s.ephemeral_system_prompt, "")
        self.assertFalse(s.skip_context_files)
        self.assertTrue(s.persist_session)
        self.assertTrue(s.compression_enabled)
        self.assertFalse(s.checkpoints_enabled)

    def test_budget_thresholds(self):
        s = FeatureState()
        self.assertEqual(s._budget_caution_threshold, 0.7)
        self.assertEqual(s._budget_warning_threshold, 0.9)
        self.assertTrue(s._budget_pressure_enabled)
        self.assertFalse(s._context_pressure_warned)

    def test_feature_flags(self):
        s = FeatureState(
            save_trajectories=True,
            verbose_logging=True,
            quiet_mode=True,
            enable_flask_agent=True,
        )
        self.assertTrue(s.save_trajectories)
        self.assertTrue(s.verbose_logging)
        self.assertTrue(s.enable_flask_agent)


class TestCallbackState(unittest.TestCase):
    def test_all_callbacks_default_to_none(self):
        s = CallbackState()
        self.assertIsNone(s.stream_delta_callback)
        self.assertIsNone(s.tool_progress_callback)
        self.assertIsNone(s.tool_start_callback)
        self.assertIsNone(s.tool_complete_callback)
        self.assertIsNone(s.clarify_callback)
        self.assertIsNone(s.reasoning_callback)
        self.assertIsNone(s.thinking_callback)
        self.assertIsNone(s.step_callback)
        self.assertIsNone(s.status_callback)
        self.assertIsNone(s.tool_gen_callback)
        self.assertIsNone(s.background_review_callback)
        self.assertIsNone(s.message_callback)

    def test_reasoning_deltas_initially_false(self):
        s = CallbackState()
        self.assertFalse(s._reasoning_deltas_fired)


class TestOutputState(unittest.TestCase):
    def test_default_values(self):
        s = OutputState()
        self.assertIsNone(s._print_fn)
        self.assertEqual(s.log_prefix, "")
        self.assertEqual(s.log_prefix_chars, 100)
        self.assertIsNone(s.print_callback)
        self.assertTrue(s.print_color)
        self.assertEqual(s.user_message_color, "yellow")
        self.assertEqual(s.agent_message_color, "green")


class TestAPIClientState(unittest.TestCase):
    def test_default_values(self):
        s = APIClientState()
        self.assertIsNone(s.client)
        self.assertEqual(s._client_kwargs, {})
        self.assertIsNone(s._anthropic_client)
        self.assertEqual(s._anthropic_api_key, "")
        self.assertEqual(s._anthropic_base_url, "")
        self.assertFalse(s._is_anthropic_oauth)
        self.assertFalse(s._use_prompt_caching)
        self.assertEqual(s._cache_ttl, "5m")
        self.assertIsNone(s.max_tokens)
        self.assertIsNone(s.reasoning_config)
        self.assertEqual(s.prefill_messages, [])


class TestProviderState(unittest.TestCase):
    def test_default_values(self):
        s = ProviderState()
        self.assertIsNone(s.providers_allowed)
        self.assertIsNone(s.providers_ignored)
        self.assertIsNone(s.providers_order)
        self.assertIsNone(s.provider_sort)
        self.assertFalse(s.provider_require_parameters)
        self.assertIsNone(s.provider_data_collection)

    def test_thinking_budget_default_zero(self):
        s = ProviderState()
        self.assertEqual(s.thinking_budget_tokens, 0)


class TestToolState(unittest.TestCase):
    def test_default_values(self):
        s = ToolState()
        self.assertEqual(s.tools, [])
        self.assertEqual(s.valid_tool_names, set())
        self.assertIsNone(s.enabled_toolsets)
        self.assertIsNone(s.disabled_toolsets)
        self.assertEqual(s._tool_use_enforcement, "auto")
        self.assertEqual(s._last_reported_tool, "")
        self.assertFalse(s._executing_tools)

    def test_valid_tool_names_is_set(self):
        s = ToolState(valid_tool_names={"read", "write", "terminal"})
        self.assertIn("read", s.valid_tool_names)
        self.assertEqual(len(s.valid_tool_names), 3)


class TestMemoryState(unittest.TestCase):
    def test_default_values(self):
        s = MemoryState()
        self.assertIsNone(s._memory_store)
        self.assertFalse(s._memory_enabled)
        self.assertFalse(s._user_profile_enabled)
        self.assertEqual(s._memory_nudge_interval, 10)
        self.assertEqual(s._memory_flush_min_turns, 6)
        self.assertEqual(s._turns_since_memory, 0)
        self.assertEqual(s._iters_since_skill, 0)
        self.assertEqual(s._skill_nudge_interval, 10)


class TestSessionState(unittest.TestCase):
    def test_default_values(self):
        s = SessionState()
        self.assertIsNone(s.session_start)
        self.assertIsNone(s.logs_dir)
        self.assertIsNone(s.session_log_file)
        self.assertEqual(s._session_messages, [])
        self.assertIsNone(s._persist_user_message_idx)
        self.assertIsNone(s._persist_user_message_override)
        self.assertIsNone(s._checkpoint_mgr)
        self.assertFalse(s._checkpoint_enabled)
        self.assertIsNone(s.trajectory_logger)
        self.assertEqual(s._session_start_time, 0.0)


class TestDatabaseState(unittest.TestCase):
    def test_default_values(self):
        s = DatabaseState()
        self.assertIsNone(s._session_db)
        self.assertEqual(s._parent_session_id, "")
        self.assertEqual(s._last_flushed_db_idx, 0)


class TestTaskState(unittest.TestCase):
    def test_default_values(self):
        s = TaskState()
        self.assertIsNone(s._todo_store)
        self.assertIsNone(s._task_output_queue)


class TestTokenState(unittest.TestCase):
    def test_default_values(self):
        s = TokenState()
        self.assertEqual(s.total_cost, 0.0)
        self.assertEqual(s._total_tokens, 0)
        self.assertEqual(s._prompt_tokens, 0)
        self.assertEqual(s._completion_tokens, 0)
        self.assertEqual(s._estimated_usage, {})
        self.assertEqual(s.session_total_tokens, 0)
        self.assertEqual(s.session_input_tokens, 0)
        self.assertEqual(s.session_output_tokens, 0)
        self.assertEqual(s.session_prompt_tokens, 0)
        self.assertEqual(s.session_completion_tokens, 0)
        self.assertEqual(s.session_cache_read_tokens, 0)
        self.assertEqual(s.session_cache_write_tokens, 0)
        self.assertEqual(s.session_reasoning_tokens, 0)
        self.assertEqual(s.session_api_calls, 0)
        self.assertEqual(s.session_estimated_cost_usd, 0.0)
        self.assertEqual(s.session_cost_status, "unknown")
        self.assertEqual(s.session_cost_source, "none")

    def test_cost_budget_defaults(self):
        s = TokenState()
        self.assertTrue(s.cost_budget_enabled)
        self.assertEqual(s.cost_max_usd, 0.0)
        self.assertEqual(s.cost_alert_thresholds, (0.5, 0.8, 1.0))

    def test_cost_budget_custom(self):
        s = TokenState(
            cost_budget_enabled=True,
            cost_max_usd=5.0,
            cost_alert_thresholds=(0.25, 0.5, 0.75, 1.0),
        )
        self.assertEqual(s.cost_max_usd, 5.0)
        self.assertEqual(s.cost_alert_thresholds, (0.25, 0.5, 0.75, 1.0))


class TestInterruptState(unittest.TestCase):
    def test_default_values(self):
        s = InterruptState()
        self.assertFalse(s._interrupt_requested)
        self.assertEqual(s._interrupt_message, "")
        self.assertFalse(s._waiting_for_user_input)
        self.assertFalse(s._waiting_for_approval)
        self.assertFalse(s._user_confirmed)
        self.assertIsNone(s._last_approval_response)
        self.assertIsNone(s._client_lock)
        self.assertEqual(s._delegate_depth, 0)
        self.assertEqual(s._active_children, [])
        self.assertIsNone(s._active_children_lock)


class TestContextState(unittest.TestCase):
    def test_default_values(self):
        s = ContextState()
        self.assertIsNone(s.context_compressor)
        self.assertTrue(s.compression_enabled)
        self.assertIsNone(s._context_compressor)
        self.assertIsNone(s._subdirectory_hints)
        self.assertEqual(s._user_turn_count, 0)
        self.assertEqual(s._primary_runtime, {})


class TestFallbackState(unittest.TestCase):
    def test_default_values(self):
        s = FallbackState()
        self.assertEqual(s._fallback_chain, [])
        self.assertEqual(s._fallback_index, 0)
        self.assertFalse(s._fallback_activated)
        self.assertIsNone(s._fallback_model)


class TestStreamState(unittest.TestCase):
    def test_default_values(self):
        s = StreamState()
        self.assertIsNone(s._stream_callback)
        self.assertFalse(s._stream_needs_break)


class TestCredentialState(unittest.TestCase):
    def test_default_values(self):
        s = CredentialState()
        self.assertIsNone(s._credential_pool)
        self.assertIsNone(s._credential_file_handler)


class TestActivityState(unittest.TestCase):
    def test_default_values(self):
        s = ActivityState()
        self.assertEqual(s._last_activity_time, 0.0)
        self.assertEqual(s._last_activity_ts, 0.0)
        self.assertEqual(s._last_activity_desc, "initializing")


class TestExtraState(unittest.TestCase):
    def test_default_values(self):
        s = ExtraState()
        self.assertEqual(s._private_state, {})
        self.assertEqual(s._anthropic_image_fallback_cache, {})
        self.assertIsNone(s._current_tool)
        self.assertEqual(s._api_call_count, 0)

    def test_private_state_mutable(self):
        s = ExtraState()
        s._private_state["key"] = "value"
        self.assertEqual(s._private_state["key"], "value")


class TestAllStatesUseSlots(unittest.TestCase):
    """Verify all 20 state classes use __slots__ for memory efficiency."""

    def test_all_classes_have_slots(self):
        classes = [
            IdentityState,
            RuntimeState,
            FeatureState,
            CallbackState,
            OutputState,
            APIClientState,
            ProviderState,
            ToolState,
            MemoryState,
            SessionState,
            DatabaseState,
            TaskState,
            TokenState,
            InterruptState,
            ContextState,
            FallbackState,
            StreamState,
            CredentialState,
            ActivityState,
            ExtraState,
        ]
        for cls in classes:
            self.assertTrue(
                hasattr(cls, "__slots__"),
                f"{cls.__name__} missing __slots__",
            )


if __name__ == "__main__":
    unittest.main()
