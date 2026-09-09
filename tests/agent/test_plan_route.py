"""Behavioral invariants for the dedicated /plan turn route."""

from agent.plan_route import PLAN_PROMPT_MARKER, planning_route_for_message


def test_planning_route_is_exactly_turn_scoped():
    config = {"planning": {"provider": "openai-codex", "model": "gpt-6-astra", "reasoning_effort": "high"}}
    prompt = f"{PLAN_PROMPT_MARKER}\nplan this"

    assert planning_route_for_message(prompt, config) == {
        "provider": "openai-codex", "model": "gpt-6-astra", "reasoning_effort": "high",
    }
    for sender in ("Any User", "Alice", "Voice Assistant"):
        assert planning_route_for_message(f"[{sender}] {prompt}", config) == {
            "provider": "openai-codex", "model": "gpt-6-astra", "reasoning_effort": "high",
        }
    assert planning_route_for_message("plan this", config) is None
    assert planning_route_for_message(prompt, {}) is None


def test_gateway_uses_planning_route_without_session_override(monkeypatch):
    from gateway import run as gateway_run
    from gateway.run_turn import GatewayTurnMixin

    class Runner(GatewayTurnMixin):
        def _resolve_session_key_or_none(self, source, session_key):
            return "session-1"

    monkeypatch.setattr(gateway_run, "_resolve_gateway_model", lambda config: "default-model")
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs_for_provider",
        lambda provider: {"provider": provider, "model": "provider-model", "api_key": "secret"},
    )
    config = {"planning": {"provider": "planning-provider", "model": "planning-model", "reasoning_effort": "high"}}

    model, runtime = Runner()._resolve_session_agent_runtime(
        user_config=config, user_message=f"{PLAN_PROMPT_MARKER}\nplan"
    )

    assert (model, runtime["provider"], runtime["api_key"]) == (
        "planning-model", "planning-provider", "secret"
    )
    assert "model_override" not in runtime
