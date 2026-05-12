"""Tests for MessagePrepMixin -- API message prep methods extracted from AIAgent."""

import pytest


class TestMessagePrepMixinImport:
    def test_mixin_importable(self):
        from agent.message_prep import MessagePrepMixin
        assert MessagePrepMixin is not None

    def test_has_sanitize_tool_call_arguments(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_sanitize_tool_call_arguments')
    def test_has_sanitize_api_messages(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_sanitize_api_messages')
    def test_has_sanitize_tool_calls_for_strict_api(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_sanitize_tool_calls_for_strict_api')
    def test_has_repair_message_sequence(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_repair_message_sequence')
    def test_has_repair_tool_call(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_repair_tool_call')
    def test_has_deduplicate_tool_calls(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_deduplicate_tool_calls')
    def test_has_prepare_messages_for_non_vision_model(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_prepare_messages_for_non_vision_model')
    def test_has_prepare_anthropic_messages_for_api(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_prepare_anthropic_messages_for_api')
    def test_has_preprocess_anthropic_content(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_preprocess_anthropic_content')
    def test_has_anthropic_prompt_cache_policy(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_anthropic_prompt_cache_policy')
    def test_has_anthropic_preserve_dots(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_anthropic_preserve_dots')
    def test_has_drop_thinking_only_and_merge_users(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_drop_thinking_only_and_merge_users')
    def test_has_strip_think_blocks(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_strip_think_blocks')
    def test_has_has_content_after_think_block(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_has_content_after_think_block')
    def test_has_copy_reasoning_content_for_api(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_copy_reasoning_content_for_api')
    def test_has_extract_reasoning(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_extract_reasoning')
    def test_has_is_thinking_only_assistant(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_is_thinking_only_assistant')
    def test_has_drop_trailing_empty_response_scaffolding(self):
        from agent.message_prep import MessagePrepMixin
        assert hasattr(MessagePrepMixin, '_drop_trailing_empty_response_scaffolding')
