"""Regression tests: the gateway must preserve a custom provider's
``request_overrides`` on per-turn agent config.

A ``custom_providers`` entry can carry an ``extra_body`` (e.g.
``chat_template_kwargs`` to toggle a local model's thinking).
``resolve_runtime_provider`` surfaces it as ``request_overrides`` on the
resolved runtime dict, but the gateway used to rebuild the runtime from a
fixed key whitelist that omitted it -- so the provider's configured
``extra_body`` never reached the model on the gateway path, and only
``/fast`` service-tier overrides survived.
"""

import pytest

from gateway.run import GatewayRunner


PROVIDER_OVERRIDES = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}


def _runtime_kwargs(**extra):
    base = {
        "api_key": "no-key-required",
        "base_url": "http://10.0.0.1:8000/v1",
        "provider": "custom",
        "api_mode": "chat_completions",
        "command": None,
        "args": [],
        "credential_pool": None,
        "max_tokens": None,
    }
    base.update(extra)
    return base


def _runner(service_tier=None):
    runner = object.__new__(GatewayRunner)
    runner._service_tier = service_tier
    return runner


def test_provider_request_overrides_preserved_without_service_tier():
    """No /fast: the provider's extra_body must pass straight through."""
    runner = _runner(service_tier=None)
    rk = _runtime_kwargs(request_overrides=PROVIDER_OVERRIDES)
    route = runner._resolve_turn_agent_config("hi", "main", rk)
    assert route["request_overrides"] == PROVIDER_OVERRIDES
    # A copy, not an alias into runtime_kwargs.
    assert route["request_overrides"] is not rk["request_overrides"]


def test_provider_request_overrides_merged_under_fast_mode(monkeypatch):
    """/fast active: provider extra_body AND the service-tier marker both survive."""
    monkeypatch.setattr(
        "hermes_cli.models.resolve_fast_mode_overrides",
        lambda model_id, **_route: {"service_tier": "priority"},
    )
    runner = _runner(service_tier="priority")
    rk = _runtime_kwargs(request_overrides=PROVIDER_OVERRIDES)
    route = runner._resolve_turn_agent_config("hi", "main", rk)
    assert route["request_overrides"]["extra_body"] == PROVIDER_OVERRIDES["extra_body"]
    assert route["request_overrides"]["service_tier"] == "priority"


def test_no_provider_overrides_yields_empty():
    """Regression: absent provider overrides, behaviour is unchanged ({})."""
    runner = _runner(service_tier=None)
    route = runner._resolve_turn_agent_config("hi", "main", _runtime_kwargs())
    assert route["request_overrides"] == {}


def test_resolve_runtime_agent_kwargs_carries_request_overrides(monkeypatch):
    """The module-level runtime resolver must not drop request_overrides."""
    import gateway.run as gateway_run

    fake_runtime = {
        "api_key": "k",
        "base_url": "http://10.0.0.1:8000/v1",
        "provider": "custom",
        "api_mode": "chat_completions",
        "request_overrides": PROVIDER_OVERRIDES,
    }
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda *a, **k: dict(fake_runtime),
    )
    monkeypatch.setattr(
        "hermes_cli.runtime_provider._get_model_config", lambda: {}
    )
    rk = gateway_run._resolve_runtime_agent_kwargs()
    assert rk["request_overrides"] == PROVIDER_OVERRIDES


# --- /model session-override follow-up: request_overrides must survive a switch ---

def test_session_override_applies_request_overrides():
    """A /model switch to a custom provider carries its extra_body into runtime."""
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess1": {
            "model": "thinkmodel",
            "provider": "custom",
            "api_key": "k",
            "base_url": "http://10.0.0.1:8000/v1",
            "api_mode": "chat_completions",
            "request_overrides": PROVIDER_OVERRIDES,
        }
    }
    rk = _runtime_kwargs()  # default resolution carried no overrides
    model, out = runner._apply_session_model_override("sess1", "oldmodel", rk)
    assert model == "thinkmodel"
    assert out["request_overrides"] == PROVIDER_OVERRIDES


def test_session_override_clears_stale_request_overrides():
    """Switching to a provider with no overrides clears a stale value."""
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess1": {
            "model": "plain",
            "provider": "openrouter",
            "api_key": "k",
            "base_url": "https://openrouter.ai/api/v1",
            "api_mode": "chat_completions",
            "request_overrides": None,
        }
    }
    rk = _runtime_kwargs(request_overrides=PROVIDER_OVERRIDES)  # stale, from default
    _, out = runner._apply_session_model_override("sess1", "old", rk)
    assert out.get("request_overrides") is None


def test_session_override_absent_is_noop():
    """No override for the session leaves runtime_kwargs untouched."""
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {}
    rk = _runtime_kwargs(request_overrides=PROVIDER_OVERRIDES)
    model, out = runner._apply_session_model_override("nope", "keepme", rk)
    assert model == "keepme"
    assert out["request_overrides"] == PROVIDER_OVERRIDES


def _empty_agent_tier_cfg(monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {"agent": {"service_tier": "", "service_tier_overrides": {}}},
    )


def _openrouter_agent(**extra):
    from run_agent import AIAgent

    kwargs = dict(
        api_key="k",
        base_url="https://openrouter.ai/api/v1",
        provider="openrouter",
        api_mode="chat_completions",
        model="openai/gpt-5",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        save_trajectories=False,
        enabled_toolsets=["file"],
    )
    kwargs.update(extra)
    return AIAgent(**kwargs)


def test_user_raw_tier_survives_gateway_chain_without_framework(monkeypatch):
    """Route → merge → api kwargs: unmarked user flex survives when /fast is off."""
    from gateway.run_turn_runner import TurnRunner

    _empty_agent_tier_cfg(monkeypatch)
    runner = _runner(service_tier=None)
    rk = _runtime_kwargs(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        request_overrides={"service_tier": "flex"},
    )
    route = runner._resolve_turn_agent_config("hi", "openai/gpt-5", rk)
    assert route.get("framework_baked_tier_keys") in (None, {})
    agent = _openrouter_agent()
    try:
        TurnRunner._merge_turn_request_overrides(agent, route)
        agent.service_tier = runner._service_tier
        agent._service_tier_session_pinned = False
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["service_tier"] == "flex"
    finally:
        agent.close()


def test_gateway_pin_replaces_user_raw_tier_on_wire(monkeypatch):
    """Route → merge → api kwargs: session /fast priority replaces user flex."""
    from gateway.run_turn_runner import TurnRunner

    _empty_agent_tier_cfg(monkeypatch)
    runner = _runner(service_tier="priority")
    rk = _runtime_kwargs(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        request_overrides={"service_tier": "flex"},
    )
    route = runner._resolve_turn_agent_config("hi", "openai/gpt-5", rk)
    assert route["request_overrides"]["service_tier"] == "priority"
    assert "service_tier" in (route.get("framework_baked_tier_keys") or {})
    agent = _openrouter_agent()
    try:
        TurnRunner._merge_turn_request_overrides(agent, route)
        agent.service_tier = runner._service_tier
        agent._service_tier_session_pinned = True
        kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}])
        assert kwargs["service_tier"] == "priority"
    finally:
        agent.close()
