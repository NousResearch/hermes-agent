"""Regression contracts for route ownership under ephemeral model overrides (#99053).

Fake only the model wire and intentionally measured policy/credential inputs.
The real AIAgent run_conversation() drives response and fallback paths.
"""
from __future__ import annotations

from types import SimpleNamespace

from tests.e2e.test_runtime_override_e2e import (
    _build_agent, _WireRecorder, _register_pre_llm_call_hook,
    _register_plugin_callback, _FakeCompletions, _run_turn,
)


def _hook(manager):
    return _register_pre_llm_call_hook(
        manager, lambda **kw: {"runtime_override": {"model": "override-model"}}
    )


def test_response_route_stays_bound_to_producing_model(monkeypatch):
    from hermes_cli.plugins import get_plugin_manager
    manager = get_plugin_manager()
    rec = _WireRecorder()
    observed = []
    handles = [
        _hook(manager),
        _register_plugin_callback(
            manager, "hook", "post_api_request",
            lambda **kw: observed.append(kw.get("model")),
        ),
    ]
    real_create = _FakeCompletions.create

    def _reasoning_only(self, **kwargs):
        value = real_create(self, **kwargs)
        value.choices[0].message.content = ""
        value.choices[0].message.reasoning = "PRIVATE CHAIN"
        return value

    monkeypatch.setattr(_FakeCompletions, "create", _reasoning_only)
    seen_policy_models = []

    def promotion_policy(agent):
        seen_policy_models.append(agent.model)
        return agent.model == "base-model"

    monkeypatch.setattr("agent.turn_final_response.answer_in_reasoning_capability", promotion_policy)
    try:
        agent = _build_agent(monkeypatch, rec)
        result = _run_turn(agent)
        assert rec.openai_requests[0]["model"] == "override-model"
        assert observed and all(model == "override-model" for model in observed)
        assert seen_policy_models and all(model == "override-model" for model in seen_policy_models)
        assert "PRIVATE CHAIN" not in (result.get("final_response") or "")
        assert agent.model == "base-model"
    finally:
        for h in handles:
            h.dispose()


def test_model_override_rebuilds_in_flight_system_prompt(monkeypatch):
    from hermes_cli.plugins import get_plugin_manager
    manager = get_plugin_manager()
    rec = _WireRecorder()
    handle = _hook(manager)
    try:
        agent = _build_agent(monkeypatch, rec)
        monkeypatch.setattr(
            agent, "_build_system_prompt",
            lambda system_message=None: f"Model: {agent.model}\nProvider: {agent.provider}\nWindow: {agent.context_compressor.context_length}",
        )
        agent._cached_system_prompt = "Model: base-model\nProvider: openai\nWindow: 200000"
        _run_turn(agent)
        sent = rec.openai_requests[0]["messages"]
        system = next(x["content"] for x in sent if x["role"] == "system")
        assert "Model: override-model" in system
        assert "Model: base-model" not in system
        assert agent.model == "base-model"
        assert agent._cached_system_prompt.startswith("Model: base-model")
    finally:
        handle.dispose()


def test_explicit_extra_body_survives_different_model(monkeypatch):
    from hermes_cli.plugins import get_plugin_manager
    manager = get_plugin_manager()
    rec = _WireRecorder()
    handle = _hook(manager)
    try:
        agent = _build_agent(monkeypatch, rec)
        policy = {"provider": {"sort": "throughput"}}
        agent.request_overrides = {"extra_body": policy, "service_tier": "flex"}
        _run_turn(agent)
        sent = rec.openai_requests[0]
        assert sent.get("model") == "override-model"
        assert sent.get("extra_body") == policy
        assert sent.get("service_tier") == "flex"
    finally:
        handle.dispose()


def test_fallback_rebuilds_in_flight_system_prompt(monkeypatch):
    from hermes_cli.plugins import get_plugin_manager
    manager = get_plugin_manager()
    rec = _WireRecorder(fail_models={"override-model"})
    handle = _hook(manager)
    try:
        agent = _build_agent(monkeypatch, rec)
        monkeypatch.setattr(agent, "_build_system_prompt",
                            lambda system_message=None: f"Model: {agent.model}\nProvider: {agent.provider}")
        agent._cached_system_prompt = "Model: base-model\nProvider: openai"
        agent._fallback_chain = [{"provider": "openai", "model": "fallback-model", "api_mode": "chat_completions"}]
        agent._fallback_index = 0
        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client",
            lambda provider, model=None, **kw: (
                SimpleNamespace(api_key="test-key", base_url="https://api.openai.com/v1"), model,
            ),
        )
        _run_turn(agent)
        models = [q["model"] for q in rec.openai_requests]
        assert models[:2] == ["override-model", "fallback-model"]
        systems = [next(m["content"] for m in r["messages"] if m["role"] == "system") for r in rec.openai_requests]
        assert "Model: override-model" in systems[0]
        assert "Model: fallback-model" in systems[1]
        assert "Model: base-model" not in systems[1]
        assert agent._cached_system_prompt.startswith("Model: fallback-model")
    finally:
        handle.dispose()
