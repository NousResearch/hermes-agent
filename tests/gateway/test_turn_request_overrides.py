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
        "hermes_cli.runtime_provider._get_model_config", dict
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


def test_reused_agent_held_on_fallback_does_not_regain_primary_fast_speed():
    """#122010: the turn route pins /fast for the PRIMARY route. A cached agent held on a local
    fallback (primary still in 429 cooldown) must not get that ``speed`` back on the next turn:
    the chat-completions transport passes it as a kwarg the OpenAI SDK rejects."""
    from types import SimpleNamespace

    from gateway.run_turn_runner import TurnRunner

    turn_route = {"request_overrides": {"speed": "fast"}}
    agent = SimpleNamespace(
        service_tier="priority", model="Hermes-3-Llama-3.1-8B-4bit", provider="custom",
        base_url="http://127.0.0.1:8080/v1", api_mode="chat_completions", request_overrides={},
        _gateway_turn_request_overrides={"speed": "fast"}, _fallback_activated=True,
        _primary_runtime={"request_overrides": {"speed": "fast"}},
    )
    TurnRunner._merge_turn_request_overrides(agent, turn_route)
    assert "speed" not in agent.request_overrides

    # On the primary route the turn's /fast value still applies.
    agent.model, agent.provider, agent.base_url = "claude-opus-4-8", "anthropic", "https://api.anthropic.com"
    agent.api_mode, agent._fallback_activated = "anthropic_messages", False
    TurnRunner._merge_turn_request_overrides(agent, turn_route)
    assert agent.request_overrides == {"speed": "fast"}


def test_reused_agent_on_an_init_time_fallback_keeps_dropping_the_primary_tier():
    """An init-time fallback stops reporting ``_fallback_activated`` once the first
    ``restore_primary_runtime`` clears it, yet the agent stays on the fallback route. Every
    later turn must still re-gate the primary's /fast tier away from it."""
    from types import SimpleNamespace

    from gateway.run_turn_runner import TurnRunner

    turn_route = {"request_overrides": {"service_tier": "priority"}}
    agent = SimpleNamespace(
        service_tier="priority", model="deepseek/deepseek-v4-pro", provider="openrouter",
        base_url="https://openrouter.ai/api/v1", api_mode="chat_completions", request_overrides={},
        _gateway_turn_request_overrides={}, _fallback_activated=False, _primary_runtime={"request_overrides": {}},
    )
    for _turn in range(3):
        TurnRunner._merge_turn_request_overrides(agent, turn_route)
        assert "service_tier" not in agent.request_overrides


def test_init_time_fallback_request_stays_free_of_the_primary_tier_every_turn():
    """The same init-time fallback through a real agent: merge, restore and the built request
    kwargs on turns 1-3 (turn 2 used to send the primary's ``service_tier`` to the fallback)."""
    from unittest.mock import MagicMock, patch

    from gateway.run_turn_runner import TurnRunner
    from run_agent import AIAgent

    fallback_client = MagicMock(base_url="https://openrouter.ai/api/v1", api_key="fallback-key-1234")
    route = {"request_overrides": {"service_tier": "priority"}}
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
        patch("agent.model_metadata.get_model_context_length", return_value=200_000),
        patch("agent.auxiliary_client.resolve_provider_client",
              side_effect=lambda provider, **_: (None, None) if provider == "openai"
              else (fallback_client, "deepseek/deepseek-v4-pro")),
    ):
        agent = AIAgent(
            model="gpt-6-sol", provider="openai", service_tier="priority", quiet_mode=True,
            skip_context_files=True, skip_memory=True, request_overrides=dict(route["request_overrides"]),
            fallback_model={"provider": "openrouter", "model": "deepseek/deepseek-v4-pro"},
        )
        assert agent.model == "deepseek/deepseek-v4-pro"
        for turn in (1, 2, 3):
            TurnRunner._merge_turn_request_overrides(agent, route)
            agent._restore_primary_runtime()
            kwargs = agent._build_api_kwargs([{"role": "user", "content": "hi"}], tools_for_api=[])
            assert "service_tier" not in kwargs, f"turn {turn}"
