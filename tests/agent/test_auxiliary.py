"""Tests for AuxiliaryMixin -- auxiliary helper methods extracted from AIAgent."""

import pytest


class TestAuxiliaryMixinImport:
    def test_mixin_importable(self):
        from agent.auxiliary import AuxiliaryMixin
        assert AuxiliaryMixin is not None

    def test_has_build_memory_write_metadata(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_build_memory_write_metadata')

    def test_has_sync_external_memory_for_turn(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_sync_external_memory_for_turn')

    def test_has_hydrate_todo_store(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_hydrate_todo_store')

    def test_has_shutdown_memory_provider(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, 'shutdown_memory_provider')

    def test_has_append_guardrail_observation(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_append_guardrail_observation')

    def test_has_set_tool_guardrail_halt(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_set_tool_guardrail_halt')

    def test_has_toolguard_controlled_halt_response(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_toolguard_controlled_halt_response')

    def test_has_guardrail_block_result(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_guardrail_block_result')

    def test_has_content_has_image_parts(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_content_has_image_parts')

    def test_has_model_supports_vision(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_model_supports_vision')

    def test_has_materialize_data_url_for_vision(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_materialize_data_url_for_vision')

    def test_has_describe_image_for_anthropic_fallback(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_describe_image_for_anthropic_fallback')

    def test_has_try_shrink_image_parts_in_messages(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_try_shrink_image_parts_in_messages')

    def test_has_clean_error_message(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_clean_error_message')

    def test_has_extract_api_error_context(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_extract_api_error_context')

    def test_has_max_tokens_param(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_max_tokens_param')

    def test_has_has_natural_response_ending(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_has_natural_response_ending')

    def test_has_should_sanitize_tool_calls(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_should_sanitize_tool_calls')

    def test_has_deterministic_call_id(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_deterministic_call_id')

    def test_has_derive_responses_function_call_id(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_derive_responses_function_call_id')

    def test_has_split_responses_tool_id(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_split_responses_tool_id')

    def test_has_get_tool_call_name_static(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_get_tool_call_name_static')

    def test_has_get_tool_call_id_static(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_get_tool_call_id_static')

    def test_has_thread_identity(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_thread_identity')

    def test_has_get_activity_summary(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, 'get_activity_summary')

    def test_has_get_rate_limit_state(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, 'get_rate_limit_state')

    def test_has_current_main_runtime(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_current_main_runtime')

    def test_has_clean_session_content(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_clean_session_content')

    def test_has_client_log_context(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_client_log_context')

    def test_has_copilot_headers_for_request(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_copilot_headers_for_request')

    def test_has_anthropic_messages_create(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_anthropic_messages_create')

    def test_has_persist_session(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_persist_session')

    def test_has_execute_tool_calls(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_execute_tool_calls')

    def test_has_dispatch_delegate_task(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_dispatch_delegate_task')

    def test_has_cap_delegate_task_calls(self):
        from agent.auxiliary import AuxiliaryMixin
        assert hasattr(AuxiliaryMixin, '_cap_delegate_task_calls')
