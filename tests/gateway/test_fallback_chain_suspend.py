"""Fallback configuration refresh and primary recovery share one expiry clock (#126516)."""

import time
from types import SimpleNamespace
from typing import Any

import pytest


def _refresh_chain(agent, config_path, monkeypatch):
    # The CLI and TUI refreshes call this same GatewayRunner path.
    from gateway.run import GatewayRunner

    monkeypatch.setattr("gateway.run._hermes_home", config_path.parent)
    runner: Any = SimpleNamespace(_fallback_model=None)
    GatewayRunner._apply_fallback_chain_to_agent(agent, GatewayRunner._refresh_fallback_model(runner))


@pytest.mark.platforms("linux", "macos")
def test_fallback_config_refresh_respects_cooldown_until_its_real_deadline(monkeypatch):
    from agent.error_classifier import FailoverReason
    from hermes_cli.config import atomic_config_write, get_config_path
    from run_agent import AIAgent

    config_path = get_config_path()
    live = [
        {
            "provider": "custom",
            "model": "gpt-4o-mini",
            "base_url": "http://127.0.0.1:9/fallback/v1",
            "api_key": "fixture-fallback-key",
        }
    ]
    config = {"model": {"context_length": 128000}, "fallback_providers": live}
    atomic_config_write(config_path, config)

    # Model a suspend gap without replacing Python's execution clock: drive only
    # this host's native expiry-clock input.
    clock_value = [time.monotonic() + 3600]
    monkeypatch.setattr(
        time,
        "clock_gettime",
        lambda _clock_id: clock_value[0],
    )
    agent: Any = AIAgent(
        provider="custom",
        model="gpt-4o",
        api_key="fixture-primary-key",
        base_url="http://127.0.0.1:9/primary/v1",
        api_mode="chat_completions",
        fallback_model=live[0],
        enabled_toolsets=[],
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        skip_background_review=True,
    )
    try:
        assert agent._try_activate_fallback(reason=FailoverReason.rate_limit)
        assert agent._fallback_activated is True
        assert agent._fallback_index == 1
        deadline = agent._rate_limited_until
        unavailable = (
            "custom",
            "unavailable-model",
            "http://127.0.0.1:9/unavailable/v1",
        )
        agent._unavailable_fallback_keys.add(unavailable)

        edited = [{**live[0], "model": "gpt-4o"}]
        config["fallback_providers"] = edited
        atomic_config_write(config_path, config)

        clock_value[0] += 30
        _refresh_chain(agent, config_path, monkeypatch)
        assert agent._fallback_chain == live
        assert agent._fallback_model == live[0]
        assert agent._fallback_index == 1
        assert agent._unavailable_fallback_keys == {unavailable}
        assert agent._restore_primary_runtime() is False

        clock_value[0] += 8 * 3600 - 30  # an eight-hour suspend
        assert clock_value[0] >= deadline
        _refresh_chain(agent, config_path, monkeypatch)
        assert agent._fallback_chain == edited
        assert agent._fallback_model == edited[0]
        assert agent._unavailable_fallback_keys == set()
        # Refresh owns the configuration, while restoration owns the active runtime.
        assert agent._fallback_activated is True
        assert agent._fallback_index == 1
        assert agent.model == live[0]["model"]
        assert agent._restore_primary_runtime() is True
        assert agent.model == agent._primary_runtime["model"]
        assert agent._fallback_activated is False
        assert agent._fallback_index == 0
        assert agent._fallback_chain == edited
    finally:
        agent.close()
