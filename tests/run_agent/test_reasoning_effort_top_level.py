"""Tests for config-driven top-level reasoning_effort on custom providers.

Verifies that ``reasoning_effort_top_level: true`` in a custom provider
config flows through to the transport layer, causing ``reasoning_effort``
to be emitted as a top-level string instead of a nested ``reasoning``
object.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.agent_init import _apply_custom_provider_flags


def _make_minimal_agent(provider="custom", base_url="http://127.0.0.1:8317/v1"):
    """Create a minimal agent-like object for flag application tests."""
    return SimpleNamespace(
        provider=provider,
        base_url=base_url,
        model="glm-4.6",
    )


# ── _apply_custom_provider_flags ──────────────────────────────────────────


class TestApplyCustomProviderFlags:
    def test_sets_flag_when_config_has_it(self):
        agent = _make_minimal_agent()
        providers = [
            {
                "base_url": "http://127.0.0.1:8317/v1",
                "reasoning_effort_top_level": True,
            }
        ]
        _apply_custom_provider_flags(agent, providers)
        assert getattr(agent, "_provider_reasoning_effort_top_level", False) is True

    def test_does_not_set_flag_when_absent(self):
        agent = _make_minimal_agent()
        providers = [{"base_url": "http://127.0.0.1:8317/v1"}]
        _apply_custom_provider_flags(agent, providers)
        assert getattr(agent, "_provider_reasoning_effort_top_level", False) is False

    def test_does_not_set_flag_when_false(self):
        agent = _make_minimal_agent()
        providers = [
            {
                "base_url": "http://127.0.0.1:8317/v1",
                "reasoning_effort_top_level": False,
            }
        ]
        _apply_custom_provider_flags(agent, providers)
        assert getattr(agent, "_provider_reasoning_effort_top_level", False) is False

    def test_skips_non_custom_provider(self):
        agent = _make_minimal_agent(provider="openrouter")
        providers = [
            {
                "base_url": "http://127.0.0.1:8317/v1",
                "reasoning_effort_top_level": True,
            }
        ]
        _apply_custom_provider_flags(agent, providers)
        assert getattr(agent, "_provider_reasoning_effort_top_level", False) is False

    def test_matches_by_base_url(self):
        agent = _make_minimal_agent(base_url="http://localhost:9999/v1")
        providers = [
            {
                "base_url": "http://127.0.0.1:8317/v1",  # different URL
                "reasoning_effort_top_level": True,
            },
            {
                "base_url": "http://localhost:9999/v1",  # matching URL
                "reasoning_effort_top_level": True,
            },
        ]
        _apply_custom_provider_flags(agent, providers)
        assert getattr(agent, "_provider_reasoning_effort_top_level", False) is True

    def test_empty_providers_list(self):
        agent = _make_minimal_agent()
        _apply_custom_provider_flags(agent, [])
        assert getattr(agent, "_provider_reasoning_effort_top_level", False) is False

    def test_no_base_url_on_agent(self):
        agent = _make_minimal_agent(base_url="")
        providers = [{"base_url": "http://127.0.0.1:8317/v1", "reasoning_effort_top_level": True}]
        _apply_custom_provider_flags(agent, providers)
        assert getattr(agent, "_provider_reasoning_effort_top_level", False) is False


# ── Transport layer: reasoning_effort emitted as top-level string ─────────


class TestTransportReasoningEffortTopLevel:
    """Verify the transport emits reasoning_effort as a top-level string
    when reasoning_effort_top_level=True is in the params."""

    def _call_build_kwargs(self, **overrides):
        from agent.transports.chat_completions import ChatCompletionsTransport

        transport = ChatCompletionsTransport()
        messages = [{"role": "user", "content": "test"}]
        defaults = {
            "reasoning_config": {"enabled": True, "effort": "high"},
            "supports_reasoning": True,
            "is_lmstudio": False,
            "is_kimi": False,
            "is_tokenhub": False,
            "is_openrouter": False,
            "is_nous": False,
            "is_github_models": False,
            "is_nvidia_nim": False,
            "is_qwen_portal": False,
            "is_custom_provider": True,
            "reasoning_effort_top_level": True,
            "provider_name": "custom",
            "base_url": "http://127.0.0.1:8317/v1",
        }
        defaults.update(overrides)
        return transport.build_kwargs("glm-4.6", messages, **defaults)

    def test_emits_top_level_reasoning_effort(self):
        kwargs = self._call_build_kwargs()
        assert "reasoning_effort" in kwargs
        assert kwargs["reasoning_effort"] == "high"

    def test_uses_medium_default_when_no_effort_config(self):
        kwargs = self._call_build_kwargs(reasoning_config={"enabled": True})
        assert kwargs["reasoning_effort"] == "medium"

    def test_does_not_emit_when_flag_false(self):
        kwargs = self._call_build_kwargs(reasoning_effort_top_level=False)
        assert "reasoning_effort" not in kwargs

    def test_does_not_emit_when_supports_reasoning_false(self):
        kwargs = self._call_build_kwargs(supports_reasoning=False)
        assert "reasoning_effort" not in kwargs

    def test_does_not_clobber_lmstudio(self):
        """When both is_lmstudio and reasoning_effort_top_level are set,
        LM Studio's handler takes precedence."""
        kwargs = self._call_build_kwargs(
            is_lmstudio=True,
            lmstudio_reasoning_options=["medium", "high"],
        )
        # LM Studio handler runs first; reasoning_effort_top_level doesn't clobber
        if "reasoning_effort" in kwargs:
            assert kwargs["reasoning_effort"] in ("medium", "high", None)
