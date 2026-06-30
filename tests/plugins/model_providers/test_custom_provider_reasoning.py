"""Regression tests for CustomProfile reasoning_config passthrough.

Covers the fix for #55276 — reasoning_effort was silently dropped for
custom/zai providers because CustomProfile.build_api_kwargs_extras() only
handled the disable case (think=False) and emitted nothing on enable.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def custom_profile():
    """Resolve the registered Custom profile via the provider registry."""
    import model_tools  # noqa: F401 — registers tool providers
    import providers

    profile = providers.get_provider_profile("custom")
    assert profile is not None, "custom provider profile must be registered"
    return profile


class TestCustomProfileReasoningPassthrough:
    """CustomProfile should forward reasoning_config to extra_body when enabled."""

    def test_reasoning_enabled_emits_extra_body_reasoning(self, custom_profile):
        """When reasoning is enabled with an effort level, emit extra_body.reasoning."""
        extra, top = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert extra["reasoning"] == {"enabled": True, "effort": "high"}
        assert top == {}

    def test_reasoning_xhigh_effort(self, custom_profile):
        """xhigh effort should be forwarded as-is."""
        extra, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"effort": "xhigh"},
        )
        assert extra["reasoning"]["effort"] == "xhigh"

    def test_reasoning_minimal_effort(self, custom_profile):
        """minimal effort should be forwarded as-is."""
        extra, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"effort": "minimal"},
        )
        assert extra["reasoning"]["effort"] == "minimal"

    def test_unknown_effort_falls_back_to_medium(self, custom_profile):
        """An unrecognized effort level should fall back to 'medium'."""
        extra, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"effort": "ultra"},
        )
        assert extra["reasoning"]["effort"] == "medium"

    def test_reasoning_disabled_emits_think_false(self, custom_profile):
        """When reasoning is explicitly disabled, emit think=False (Ollama convention)."""
        extra, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
        )
        assert extra["think"] is False
        assert "reasoning" not in extra

    def test_effort_none_emits_think_false(self, custom_profile):
        """effort='none' should disable thinking, not emit reasoning."""
        extra, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"effort": "none"},
        )
        assert extra["think"] is False
        assert "reasoning" not in extra

    def test_no_reasoning_config_emits_empty(self, custom_profile):
        """When no reasoning_config is provided, extra_body should be empty."""
        extra, _ = custom_profile.build_api_kwargs_extras()
        assert extra == {}

    def test_empty_effort_string_emits_nothing(self, custom_profile):
        """When effort is empty string and enabled, no reasoning should be emitted."""
        extra, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": ""},
        )
        # Empty effort with enabled=True — no reasoning field, no think=False
        assert "reasoning" not in extra
        assert "think" not in extra

    def test_ollama_num_ctx_coexists_with_reasoning(self, custom_profile):
        """Ollama num_ctx and reasoning config should coexist in extra_body."""
        extra, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"effort": "high"},
            ollama_num_ctx=8192,
        )
        assert extra["options"]["num_ctx"] == 8192
        assert extra["reasoning"] == {"enabled": True, "effort": "high"}

    def test_reasoning_disabled_with_num_ctx(self, custom_profile):
        """When reasoning is disabled but num_ctx is set, both should work."""
        extra, _ = custom_profile.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            ollama_num_ctx=4096,
        )
        assert extra["think"] is False
        assert extra["options"]["num_ctx"] == 4096
        assert "reasoning" not in extra
