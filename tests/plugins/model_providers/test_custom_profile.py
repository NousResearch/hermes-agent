"""Unit tests for the Custom provider profile's wire-shape contracts.

Covers the ``build_api_kwargs_extras`` return for Ollama num_ctx, reasoning
disable, and the ``x-session-id`` header forwarding for self-hosted backends.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def custom_profile():
    """Resolve the registered Custom profile via the provider registry."""
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("custom")
    assert profile is not None, "custom provider profile must be registered"
    return profile


class TestCustomSessionIdHeader:
    """session_id is forwarded as x-session-id HTTP header."""

    def test_session_id_emits_header(self, custom_profile):
        _, top_level = custom_profile.build_api_kwargs_extras(
            session_id="abc-123"
        )
        assert top_level == {"extra_headers": {"x-session-id": "abc-123"}}

    def test_no_session_id_omits_header(self, custom_profile):
        _, top_level = custom_profile.build_api_kwargs_extras()
        assert top_level == {}

    def test_empty_session_id_omits_header(self, custom_profile):
        _, top_level = custom_profile.build_api_kwargs_extras(
            session_id=""
        )
        assert top_level == {}

    def test_none_session_id_omits_header(self, custom_profile):
        _, top_level = custom_profile.build_api_kwargs_extras(
            session_id=None
        )
        assert top_level == {}


class TestCustomOllamaNumCtx:
    """Ollama num_ctx is forwarded via extra_body.options.num_ctx."""

    def test_num_ctx_emitted(self, custom_profile):
        extra_body, _ = custom_profile.build_api_kwargs_extras(
            ollama_num_ctx=8192
        )
        assert extra_body == {"options": {"num_ctx": 8192}}

    def test_no_num_ctx_omits_options(self, custom_profile):
        extra_body, _ = custom_profile.build_api_kwargs_extras()
        assert extra_body == {}


class TestCustomReasoningDisable:
    """reasoning_config disabled → extra_body.think = False."""

    def test_disabled_emits_think_false(self, custom_profile):
        extra_body, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False}
        )
        assert extra_body == {"think": False}

    def test_effort_none_emits_think_false(self, custom_profile):
        extra_body, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "none"}
        )
        assert extra_body == {"think": False}

    def test_enabled_emits_no_think(self, custom_profile):
        extra_body, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}
        )
        assert extra_body == {}


class TestCustomCombinedOutput:
    """session_id header + reasoning + num_ctx all work together."""

    def test_all_three_present(self, custom_profile):
        extra_body, top_level = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            ollama_num_ctx=4096,
            session_id="sess-xyz",
        )
        assert extra_body == {"options": {"num_ctx": 4096}, "think": False}
        assert top_level == {"extra_headers": {"x-session-id": "sess-xyz"}}


class TestCustomFullKwargsIntegration:
    """End-to-end: the transport produces x-session-id in api_kwargs."""

    def test_full_kwargs_include_session_header(self, custom_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="my-model",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=custom_profile,
            session_id="conv-42",
        )
        assert kwargs["extra_headers"] == {"x-session-id": "conv-42"}

    def test_full_kwargs_omit_header_without_session(self, custom_profile):
        from agent.transports.chat_completions import ChatCompletionsTransport

        kwargs = ChatCompletionsTransport().build_kwargs(
            model="my-model",
            messages=[{"role": "user", "content": "ping"}],
            tools=None,
            provider_profile=custom_profile,
        )
        assert "extra_headers" not in kwargs
