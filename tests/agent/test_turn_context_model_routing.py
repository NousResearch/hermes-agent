"""Tests for per-turn answerer-model routing from pre_llm_call hooks.

Covers the plugin-facing contract of
:func:`agent.turn_context.apply_plugin_routed_model` — a pre_llm_call
callback may return ``{"model": ..., "provider": ...}`` to swap the
answerer model for a single turn.  These tests guard the surface the
reviewer flagged: first-match-wins, the provider-change resolver path,
the same-provider direct path, and the fail-open fallback.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.turn_context import apply_plugin_routed_model


# ── Fixtures & helpers ───────────────────────────────────────────────────


class _FakeAgent(SimpleNamespace):
    """Minimal stand-in for the runtime agent with a recording switch_model."""

    def __init__(self, model="deepseek-v4-flash", provider="opencode-go",
                 base_url="https://opencode-go.example/v1",
                 api_key="current-key", api_mode="openai_chat"):
        self.switch_calls = []
        super().__init__(
            model=model, provider=provider, base_url=base_url,
            api_key=api_key, api_mode=api_mode,
        )

    def switch_model(self, *args, **kwargs):
        self.switch_calls.append((args, kwargs))


def _make_agent(**kw) -> _FakeAgent:
    return _FakeAgent(**kw)


def _resolved_ok(new_model="claude-sonnet-4.6", target_provider="anthropic",
                 api_key="resolved-key", base_url="https://anthropic.example/v1",
                 api_mode="anthropic_messages"):
    """A success result shaped like ModelSwitchResult (no real import needed)."""
    return SimpleNamespace(
        success=True,
        new_model=new_model,
        target_provider=target_provider,
        api_key=api_key,
        base_url=base_url,
        api_mode=api_mode,
        error_message="",
    )


# ── No-op paths ──────────────────────────────────────────────────────────


def test_no_model_key_is_noop():
    """A hook that contributes only context must not trigger a model swap."""
    agent = _make_agent()
    apply_plugin_routed_model(agent, [{"context": "just context"}])
    assert agent.switch_calls == []


def test_no_routing_key_at_all_is_noop():
    """Non-dict returns (strings) must not be interpreted as routing."""
    agent = _make_agent()
    apply_plugin_routed_model(agent, ["plain context string"])
    assert agent.switch_calls == []


def test_same_model_case_insensitive_is_noop():
    """Routing to the current model (any casing) must not rebuild the client."""
    agent = _make_agent(model="deepseek-v4-flash")
    apply_plugin_routed_model(agent, [{"model": "DEEPSEEK-V4-FLASH"}])
    assert agent.switch_calls == []


def test_empty_model_key_is_ignored():
    """A '' or whitespace model key is treated as 'no route requested'."""
    agent = _make_agent()
    apply_plugin_routed_model(agent, [{"model": "   "}])
    assert agent.switch_calls == []


# ── First-match-wins ─────────────────────────────────────────────────────


def test_first_routing_hook_wins():
    """The first callback carrying a model key wins; later ones are ignored."""
    agent = _make_agent(provider="opencode-go")
    with patch("hermes_cli.model_switch.switch_model",
               return_value=_resolved_ok()), \
         patch("hermes_cli.config.load_config",
               return_value={"providers": {"anthropic": {}}}), \
         patch("hermes_cli.config.get_compatible_custom_providers", return_value=[]):
        apply_plugin_routed_model(
            agent,
            [
                {"model": "claude-sonnet-4.6", "provider": "anthropic"},
                {"model": "gpt-5", "provider": "openai"},  # must be ignored
            ],
        )
    assert len(agent.switch_calls) == 1
    args, _kwargs = agent.switch_calls[0]
    assert args[0] == "claude-sonnet-4.6"
    assert args[1] == "anthropic"


# ── Same-provider direct path ────────────────────────────────────────────


def test_same_provider_direct_path_forwards_current_creds_by_keyword():
    """Same-provider swaps apply current creds directly and use keyword args."""
    agent = _make_agent(provider="opencode-go")
    apply_plugin_routed_model(
        agent,
        [{"model": "deepseek-v4-pro", "provider": "opencode-go"}],
    )
    assert len(agent.switch_calls) == 1
    args, kwargs = agent.switch_calls[0]
    # Positional: (new_model, new_provider)
    assert args[0] == "deepseek-v4-pro"
    assert args[1] == "opencode-go"
    # Credentials forwarded as keywords from the current agent (review item #2).
    assert kwargs == {
        "api_key": "current-key",
        "base_url": "https://opencode-go.example/v1",
        "api_mode": "openai_chat",
    }


def test_omitted_provider_defaults_to_current():
    """A hook that only names a model keeps the current provider."""
    agent = _make_agent(provider="opencode-go")
    apply_plugin_routed_model(agent, [{"model": "deepseek-v4-pro"}])
    assert len(agent.switch_calls) == 1
    args, _kwargs = agent.switch_calls[0]
    assert args[0] == "deepseek-v4-pro"
    assert args[1] == "opencode-go"


# ── Provider-change resolver path ────────────────────────────────────────


def test_provider_change_resolves_and_forwards_target_creds():
    """Cross-provider routes resolve the TARGET provider's creds (not the old
    host's) and forward them — the #47828 400-safety guarantee."""
    agent = _make_agent(provider="opencode-go")
    with patch("hermes_cli.model_switch.switch_model",
               return_value=_resolved_ok()) as mock_resolve, \
         patch("hermes_cli.config.load_config",
               return_value={"providers": {"anthropic": {}}}) as mock_cfg, \
         patch("hermes_cli.config.get_compatible_custom_providers",
               return_value=[]):
        apply_plugin_routed_model(
            agent,
            [{"model": "claude-sonnet-4.6", "provider": "anthropic"}],
        )

    # Resolver was invoked with the TARGET provider explicit.
    assert mock_resolve.called
    _, resolve_kwargs = mock_resolve.call_args
    assert resolve_kwargs["raw_input"] == "claude-sonnet-4.6"
    assert resolve_kwargs["explicit_provider"] == "anthropic"
    assert resolve_kwargs["current_provider"] == "opencode-go"

    # Resolved target creds are forwarded (not the current provider's).
    assert len(agent.switch_calls) == 1
    args, kwargs = agent.switch_calls[0]
    assert args[0] == "claude-sonnet-4.6"
    assert args[1] == "anthropic"
    assert kwargs == {
        "api_key": "resolved-key",
        "base_url": "https://anthropic.example/v1",
        "api_mode": "anthropic_messages",
    }


def test_provider_change_resolve_failure_fails_open():
    """A failed resolver must NOT swap the model; the turn keeps the current one."""
    agent = _make_agent(provider="opencode-go")
    failed = SimpleNamespace(
        success=False, new_model="", target_provider="",
        api_key="", base_url="", api_mode="",
        error_message="provider not configured",
    )
    with patch("hermes_cli.model_switch.switch_model", return_value=failed), \
         patch("hermes_cli.config.load_config", return_value={}), \
         patch("hermes_cli.config.get_compatible_custom_providers", return_value=[]):
        apply_plugin_routed_model(
            agent,
            [{"model": "claude-sonnet-4.6", "provider": "anthropic"}],
        )
    assert agent.switch_calls == []
    # Current model untouched.
    assert agent.model == "deepseek-v4-flash"


def test_provider_change_resolve_exception_fails_open():
    """A resolver that RAISES must still not break the turn."""
    agent = _make_agent(provider="opencode-go")
    with patch("hermes_cli.model_switch.switch_model",
               side_effect=RuntimeError("boom")), \
         patch("hermes_cli.config.load_config", return_value={}), \
         patch("hermes_cli.config.get_compatible_custom_providers", return_value=[]):
        # Must not raise.
        apply_plugin_routed_model(
            agent,
            [{"model": "claude-sonnet-4.6", "provider": "anthropic"}],
        )
    assert agent.switch_calls == []


def test_same_provider_does_not_call_resolver():
    """Same-provider swaps must skip the resolver round-trip entirely."""
    agent = _make_agent(provider="opencode-go")
    with patch("hermes_cli.model_switch.switch_model") as mock_resolve:
        apply_plugin_routed_model(
            agent,
            [{"model": "deepseek-v4-pro", "provider": "opencode-go"}],
        )
    mock_resolve.assert_not_called()