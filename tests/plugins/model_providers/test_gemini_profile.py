"""Contract tests for the native Google Gemini provider profile."""

from __future__ import annotations

import pytest


@pytest.fixture
def gemini_profile():
    import model_tools  # noqa: F401
    import providers

    profile = providers.get_provider_profile("gemini")
    assert profile is not None, "gemini provider profile must be registered"
    return profile


def test_native_gemini_auxiliary_default_is_in_curated_catalog(gemini_profile):
    """The profile's default_aux_model must stay in lockstep with the curated
    model picker catalog — whatever model the default points at has to be
    one the picker can actually offer. Deliberately durable against future
    model-generation bumps: it does not pin either side to a frozen
    model-name string, only to the invariant that they never drift apart.
    """
    from hermes_cli.models import _PROVIDER_MODELS

    assert gemini_profile.default_aux_model in _PROVIDER_MODELS["gemini"]


def test_gemini_build_extra_body_disabled_reasoning_sets_zero_thinking_budget(gemini_profile):
    """When reasoning is disabled (enabled=False or effort=none), GeminiProfile
    must emit thinkingBudget: 0 to ensure thinking tokens do not consume
    the output token budget (e.g. for concise auxiliary title generation)."""
    extra_disabled = gemini_profile.build_extra_body(
        model="gemini-3.6-flash",
        reasoning_config={"enabled": False},
    )
    assert extra_disabled == {
        "thinking_config": {"includeThoughts": False, "thinkingBudget": 0}
    }

    extra_none = gemini_profile.build_extra_body(
        model="gemini-3.6-flash",
        reasoning_config={"effort": "none"},
    )
    assert extra_none == {
        "thinking_config": {"includeThoughts": False, "thinkingBudget": 0}
    }


def test_gemini_openai_compat_disabled_reasoning_sets_zero_thinking_budget(gemini_profile):
    """On OpenAI-compatible Gemini endpoints, disabled reasoning translates to
    google.thinking_config with include_thoughts=False and thinking_budget=0."""
    extra = gemini_profile.build_extra_body(
        model="gemini-3.6-flash",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        reasoning_config={"enabled": False},
    )
    assert extra == {
        "extra_body": {
            "google": {
                "thinking_config": {
                    "include_thoughts": False,
                    "thinking_budget": 0,
                }
            }
        }
    }


def test_auxiliary_build_call_kwargs_propagates_task_reasoning_effort_to_profile():
    """_build_call_kwargs must extract extra_body['reasoning'] into
    effective_reasoning_config so provider profiles (Gemini, Vertex, etc.)
    receive task reasoning config and translate it."""
    from agent.auxiliary_client import _build_call_kwargs

    kwargs = _build_call_kwargs(
        provider="gemini",
        model="gemini-3.6-flash",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"reasoning": {"enabled": False}},
        task="title_generation",
    )
    assert kwargs.get("extra_body", {}).get("thinking_config") == {
        "includeThoughts": False,
        "thinkingBudget": 0,
    }
    # Raw 'reasoning' dict is handled by profile and stripped from extra_body
    assert "reasoning" not in kwargs.get("extra_body", {})
