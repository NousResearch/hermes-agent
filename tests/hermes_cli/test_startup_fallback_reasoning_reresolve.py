"""Startup provider fallback re-resolves reasoning config for the fallback model.

Regression for #113492: ``_resolve_fallback_runtime`` in
``hermes_cli/cli_agent_setup_mixin.py`` switched ``requested_provider``/``model``
after a primary-provider AuthError without re-resolving ``reasoning_config``,
so the stale primary-model value flowed into the agent build. The mid-session
path already re-resolves via ``_reresolve_fallback_reasoning_config``
(``agent/chat_completion_helpers.py``); this covers the startup switch.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def _make_cli(monkeypatch):
    """Minimal stand-in driving the mixin unbound with real resolution logic."""
    import cli
    from hermes_cli.auth import AuthError

    monkeypatch.setattr(cli, "_cprint", lambda text: None)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda **kwargs: {"provider": "anthropic", "api_key": "fb-tok",
                          "base_url": "https://api.anthropic.com",
                          "requested_provider": "anthropic"},
    )
    monkeypatch.setattr(
        "hermes_cli.fallback_config.resolve_entry_api_key",
        lambda entry: "sk-fb",
    )
    state = SimpleNamespace(
        requested_provider="openrouter",
        model="primary-model",
        reasoning_config={"enabled": True, "effort": "medium"},
        _fallback_model=[{"provider": "anthropic", "model": "claude-fallback"}],
    )
    return state, AuthError("primary key rejected")


_CFG = {
    "agent": {
        "reasoning_effort": "medium",
        "reasoning_overrides": {"claude-fallback": "xhigh"},
    }
}


def test_startup_fallback_reresolves_reasoning_config_for_fallback_model(monkeypatch):
    """After the fallback switch, reasoning_config targets the fallback model.

    The primary model's resolved value (medium) must not leak into the agent
    built for ``claude-fallback``, which carries an ``xhigh`` per-model override.
    """
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _CFG)

    state, exc = _make_cli(monkeypatch)
    runtime = CLIAgentSetupMixin._resolve_fallback_runtime(state, exc)

    assert runtime["provider"] == "anthropic"
    assert state.requested_provider == "anthropic"
    assert state.model == "claude-fallback"

    from hermes_constants import resolve_reasoning_config
    expected = resolve_reasoning_config(_CFG, "claude-fallback")
    assert state.reasoning_config == expected
    # The override must differ from the stale primary-model value.
    assert state.reasoning_config != resolve_reasoning_config(_CFG, "primary-model")
    assert state.reasoning_config["effort"] == "xhigh"


def test_startup_fallback_keeps_current_reasoning_on_config_failure(monkeypatch):
    """A config load failure must not kill the fallback switch.

    Mirrors the mid-session helper: the stale value is kept and the switch
    still completes (resolution failure is logged, never raised).
    """
    from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin

    def _boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr("hermes_cli.config.load_config", _boom)

    state, exc = _make_cli(monkeypatch)
    stale = state.reasoning_config
    runtime = CLIAgentSetupMixin._resolve_fallback_runtime(state, exc)

    assert runtime is not None
    assert state.model == "claude-fallback"
    assert state.reasoning_config is stale
