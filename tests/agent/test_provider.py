"""Tests for ProviderMixin -- provider-specific methods extracted from AIAgent."""

import pytest


class TestProviderMixinImport:
    def test_mixin_importable(self):
        from agent.provider import ProviderMixin
        assert ProviderMixin is not None

    def test_has_qwen_prepare_chat_messages(self):
        from agent.provider import ProviderMixin
        assert hasattr(ProviderMixin, '_qwen_prepare_chat_messages')

    def test_has_qwen_prepare_chat_messages_inplace(self):
        from agent.provider import ProviderMixin
        assert hasattr(ProviderMixin, '_qwen_prepare_chat_messages_inplace')

    def test_has_needs_deepseek_tool_reasoning(self):
        from agent.provider import ProviderMixin
        assert hasattr(ProviderMixin, '_needs_deepseek_tool_reasoning')

    def test_has_needs_kimi_tool_reasoning(self):
        from agent.provider import ProviderMixin
        assert hasattr(ProviderMixin, '_needs_kimi_tool_reasoning')

    def test_has_needs_thinking_reasoning_pad(self):
        from agent.provider import ProviderMixin
        assert hasattr(ProviderMixin, '_needs_thinking_reasoning_pad')

    def test_has_lmstudio_reasoning_options_cached(self):
        from agent.provider import ProviderMixin
        assert hasattr(ProviderMixin, '_lmstudio_reasoning_options_cached')

    def test_has_resolve_lmstudio_summary_reasoning_effort(self):
        from agent.provider import ProviderMixin
        assert hasattr(ProviderMixin, '_resolve_lmstudio_summary_reasoning_effort')

    def test_has_github_models_reasoning_extra_body(self):
        from agent.provider import ProviderMixin
        assert hasattr(ProviderMixin, '_github_models_reasoning_extra_body')

    def test_has_supports_reasoning_extra_body(self):
        from agent.provider import ProviderMixin
        assert hasattr(ProviderMixin, '_supports_reasoning_extra_body')

    def test_has_provider_model_requires_responses_api(self):
        from agent.provider import ProviderMixin
        assert hasattr(ProviderMixin, '_provider_model_requires_responses_api')

    def test_has_model_requires_responses_api(self):
        from agent.provider import ProviderMixin
        assert hasattr(ProviderMixin, '_model_requires_responses_api')

    def test_has_should_treat_stop_as_truncated(self):
        from agent.provider import ProviderMixin
        assert hasattr(ProviderMixin, '_should_treat_stop_as_truncated')

    def test_has_is_ollama_glm_backend(self):
        from agent.provider import ProviderMixin
        assert hasattr(ProviderMixin, '_is_ollama_glm_backend')

    def test_has_is_qwen_portal(self):
        from agent.provider import ProviderMixin
        assert hasattr(ProviderMixin, '_is_qwen_portal')
