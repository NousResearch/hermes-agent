"""Tests for agent.transport_fallback_threshold (config-driven eager fallback).

The threshold gates eager fallback on transport-layer failures (timeout /
overloaded).  Before this change the threshold was hardcoded to 2 in
``run_conversation``; it is now read from ``agent.transport_fallback_threshold``
in config.yaml (default 2, floor-clamped at 1).
"""

import pytest

from hermes_cli.config_defaults import DEFAULT_CONFIG


def _build_agent(monkeypatch, config_value):
    """Spin up an AIAgent with a patched agent-section config value.

    ``init_agent`` reads ``_load_agent_config()`` for the ``agent`` section;
    we patch just that section so every other default (api_max_retries etc.)
    stays at the shipped default.
    """
    from run_agent import AIAgent
    import hermes_cli.config as hc

    real_loader = hc.load_config_readonly

    def _patched():
        cfg = real_loader()
        agent_sec = dict(cfg.get("agent", {}))
        if config_value is not None:
            agent_sec["transport_fallback_threshold"] = config_value
        cfg["agent"] = agent_sec
        return cfg

    monkeypatch.setattr(hc, "load_config_readonly", _patched)
    return AIAgent(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-test",
        model="test-model",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
    )


def test_config_defaults_registers_transport_fallback_threshold():
    """DEFAULT_CONFIG registers the key with an int >= 1 (behaviour contract,
    not a snapshot of the current literal value).
    """
    agent_cfg = DEFAULT_CONFIG.get("agent", {})
    assert "transport_fallback_threshold" in agent_cfg
    assert isinstance(agent_cfg["transport_fallback_threshold"], int)
    assert agent_cfg["transport_fallback_threshold"] >= 1


def test_init_agent_applies_configured_threshold(monkeypatch):
    agent = _build_agent(monkeypatch, 5)
    assert agent._transport_fallback_threshold == 5


def test_init_agent_clamps_below_1(monkeypatch):
    """0 would mean 'never fall back on transport failure' — clamped to 1."""
    agent = _build_agent(monkeypatch, 0)
    assert agent._transport_fallback_threshold == 1


def test_init_agent_defaults_to_2_when_unset(monkeypatch):
    """The shipped default preserves the historical hardcoded behaviour."""
    agent = _build_agent(monkeypatch, None)
    assert agent._transport_fallback_threshold == 2


def test_init_agent_tolerates_non_int(monkeypatch):
    """A malformed config value falls back to the default, no crash."""
    agent = _build_agent(monkeypatch, "not-a-number")
    assert agent._transport_fallback_threshold == 2


def test_fallback_gate_references_configured_threshold():
    """run_conversation's eager-fallback gate reads the agent attribute
    (invariant: raising the config raises the transport-failure count
    required before fallback fires)."""
    import inspect

    from agent.conversation_loop import run_conversation

    src = inspect.getsource(run_conversation)
    assert "_transport_fallback_threshold" in src
