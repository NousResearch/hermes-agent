"""The ``pre_llm_call`` runtime override through the real conversation loop.

Drives ``AIAgent.run_conversation`` against an in-process fake wire client, with
the ``pre_llm_call`` hook stubbed at the lifecycle boundary, so request
assembly, preflight and the wire all observe the overridden model. The second
turn returns to the base model — the override is turn-scoped. A wire failure on
the overridden route hands off to the fallback chain, which then owns the turn.
"""

from __future__ import annotations

from types import SimpleNamespace


class _SimulatedRateLimit(Exception):
    status_code = 429


def _response(content: str = "ok"):
    message = SimpleNamespace(content=content, tool_calls=[], reasoning=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None
    )


class _Completions:
    def __init__(self, calls, fail_model=None):
        self._calls = calls
        self._fail_model = fail_model

    def create(self, **kwargs):
        self._calls.append(kwargs)
        if kwargs.get("model") == self._fail_model:
            raise _SimulatedRateLimit(f"simulated rate limit for {kwargs.get('model')}")
        return _response()


class _Client:
    def __init__(self, calls, fail_model=None):
        self.chat = SimpleNamespace(completions=_Completions(calls, fail_model))

    def close(self):
        pass


def _build_agent(monkeypatch, calls, hook, *, fail_model=None):
    from run_agent import AIAgent

    monkeypatch.setattr(
        "agent.process_bootstrap.OpenAI", lambda **_kw: _Client(calls, fail_model)
    )
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda *a, **k: [])
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", hook)

    agent = AIAgent(
        model="base-model", api_key="test-key", base_url="http://localhost:8080/v1",
        platform="cli", max_iterations=3, quiet_mode=True, skip_memory=True,
    )
    agent._disable_streaming = True
    return agent


def _override_once_hook():
    turns = []

    def hook(name, **_kw):
        if name == "pre_llm_call":
            turns.append(1)
            if len(turns) == 1:
                return [{"runtime_override": {"model": "override-model"}}]
        return []

    return hook


def test_override_reaches_the_wire_and_does_not_survive_the_turn(monkeypatch):
    calls = []
    agent = _build_agent(monkeypatch, calls, _override_once_hook())

    first = agent.run_conversation("hello")
    assert "ok" in (first.get("final_response") or "")
    assert [c.get("model") for c in calls] == ["override-model"]
    # Turn-scoped: the base model is back once the turn ends.
    assert agent.model == "base-model"

    second = agent.run_conversation("again")
    assert "ok" in (second.get("final_response") or "")
    assert [c.get("model") for c in calls] == ["override-model", "base-model"]


def test_failed_override_hands_the_route_to_the_fallback(monkeypatch):
    calls = []

    def hook(name, **_kw):
        if name == "pre_llm_call":
            return [{"runtime_override": {"model": "override-model"}}]
        return []

    # The fallback client is resolved through this boundary; its wire requests
    # still land on the fake recorder because the per-request client is rebuilt
    # from ``process_bootstrap.OpenAI``.
    monkeypatch.setattr(
        "agent.auxiliary_client.resolve_provider_client",
        lambda provider, model=None, **kwargs: (
            SimpleNamespace(api_key="test-key", base_url="http://localhost:8080/v1"), model,
        ),
    )
    agent = _build_agent(monkeypatch, calls, hook, fail_model="override-model")
    agent._fallback_chain = [
        {"provider": "openai", "model": "fallback-model", "api_mode": "chat_completions"},
    ]
    agent._fallback_index = 0

    result = agent.run_conversation("hello")

    assert "ok" in (result.get("final_response") or "")
    models = [c.get("model") for c in calls]
    assert models[0] == "override-model"
    assert models[1] == "fallback-model", f"retry not on the fallback route: {models}"
    assert "override-model" not in models[1:]
