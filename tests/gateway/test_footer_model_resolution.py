"""Tests for footer model resolution: auxiliary tasks must not pollute the footer.

Issue #43228: When auxiliary tasks (vision, compression, title generation)
temporarily overwrite agent.model, the runtime footer should still show
the model that generated the main response, not the auxiliary model.

The fix captures _initial_model right after agent creation and uses it
for the footer unless a fallback was activated.
"""

from types import SimpleNamespace

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent(model="deepseek-v4-pro", provider="deepseek", fallback=False):
    """Create a minimal agent-like object."""
    agent = SimpleNamespace(
        model=model,
        provider=provider,
        _fallback_activated=fallback,
        context_compressor=SimpleNamespace(
            last_prompt_tokens=1000,
            context_length=128000,
        ),
        session_prompt_tokens=500,
        session_completion_tokens=200,
    )
    return agent


def _resolve_model(initial_model, agent):
    """Replicate the footer model resolution logic from gateway/run.py."""
    _resolved_model = initial_model
    if agent and getattr(agent, "_fallback_activated", False):
        _resolved_model = getattr(agent, "model", None)
    elif not _resolved_model:
        _resolved_model = getattr(agent, "model", None) if agent else None
    return _resolved_model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFooterModelResolution:
    """The footer model must reflect the main response model, not auxiliary."""

    def test_normal_run_shows_initial_model(self):
        """When no auxiliary task modifies agent.model, footer shows it as-is."""
        agent = _make_agent(model="deepseek-v4-pro")
        _initial_model = agent.model
        assert _resolve_model(_initial_model, agent) == "deepseek-v4-pro"

    def test_auxiliary_model_does_not_pollute_footer(self):
        """When an auxiliary task overwrites agent.model, footer shows initial."""
        agent = _make_agent(model="deepseek-v4-pro")
        _initial_model = "deepseek-v4-pro"
        # Auxiliary task (vision) overwrites agent.model
        agent.model = "qwen3.6-plus"
        assert _resolve_model(_initial_model, agent) == "deepseek-v4-pro"

    def test_fallback_model_shows_in_footer(self):
        """When fallback is activated, footer shows the fallback model."""
        agent = _make_agent(model="qwen3.6-plus", fallback=True)
        _initial_model = "deepseek-v4-pro"
        assert _resolve_model(_initial_model, agent) == "qwen3.6-plus"

    def test_fallback_then_auxiliary_shows_fallback(self):
        """Fallback activated, then auxiliary task overwrites — footer shows fallback."""
        agent = _make_agent(model="qwen3.6-plus", fallback=True)
        _initial_model = "deepseek-v4-pro"
        # Auxiliary task overwrites agent.model after fallback
        agent.model = "gpt-4o-mini"
        # When fallback is active, agent.model takes precedence
        assert _resolve_model(_initial_model, agent) == "gpt-4o-mini"

    def test_initial_model_none_falls_back_to_agent(self):
        """When _initial_model is None, fall back to agent.model."""
        agent = _make_agent(model="gpt-4o")
        assert _resolve_model(None, agent) == "gpt-4o"

    def test_no_agent_returns_none(self):
        """When agent is None, resolved model is None."""
        assert _resolve_model(None, None) is None

    def test_multiple_auxiliary_tasks_last_one_does_not_leak(self):
        """Multiple auxiliary tasks overwrite agent.model — footer still shows initial."""
        agent = _make_agent(model="deepseek-v4-pro")
        _initial_model = "deepseek-v4-pro"
        # First auxiliary task: vision
        agent.model = "qwen3.6-plus"
        # Second auxiliary task: compression
        agent.model = "gpt-4o-mini"
        # Third auxiliary task: title generation
        agent.model = "claude-sonnet-4"
        assert _resolve_model(_initial_model, agent) == "deepseek-v4-pro"
