"""A cancelled advisor phase must not dispatch a new acting-model request."""
from types import SimpleNamespace

import pytest


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=[]), finish_reason="stop")], usage=None, model="fixture")


def test_cancel_during_advisor_does_not_send_aggregator(monkeypatch, tmp_path):
    from run_agent import AIAgent

    (tmp_path / "config.yaml").write_text("""
moa:
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openrouter
          model: fixture-advisor
      aggregator:
        provider: openrouter
        model: fixture-actor
""", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []
    def send(**kwargs):
        calls.append(kwargs["task"])
        if kwargs["task"] == "moa_reference":
            agent._interrupt_requested = True
            return _response("advisor finished while cancellation arrived")
        return _response("must not act after cancellation")
    monkeypatch.setattr("agent.moa_loop.call_llm", send)
    agent = AIAgent(api_key="fixture", base_url="moa://local", model="review", provider="moa", quiet_mode=True, skip_context_files=True, skip_memory=True, enabled_toolsets=["file"], max_iterations=1)
    result = agent.run_conversation("review this")
    assert calls == ["moa_reference"]
    assert "must not act" not in result["final_response"]


@pytest.mark.parametrize("cancel_at", ["advisor", "planning"])
def test_one_shot_cancellation_does_not_start_synthesis(monkeypatch, cancel_at):
    from agent import moa_loop

    owner = SimpleNamespace(_interrupt_requested=False)
    calls = []
    def send(**kwargs):
        calls.append(kwargs["task"])
        if kwargs["task"] == "moa_reference" and cancel_at == "advisor":
            owner._interrupt_requested = True
        return _response("completed advice")
    monkeypatch.setattr(moa_loop, "call_llm", send)
    def runtime(slot):
        if slot["model"] == "actor" and cancel_at == "planning":
            owner._interrupt_requested = True
        return {"provider": "custom", "model": slot["model"], "base_url": "http://fixture.local/v1"}
    monkeypatch.setattr(moa_loop, "_slot_runtime", runtime)
    with pytest.raises(InterruptedError, match="before MoA aggregator"):
        moa_loop.aggregate_moa_context(
            user_prompt="review", api_messages=[{"role": "user", "content": "review"}],
            reference_models=[{"provider": "custom", "model": "advisor"}],
            aggregator={"provider": "custom", "model": "actor"}, agent=owner,
        )
    assert calls == ["moa_reference"]


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("cancel_at", ["before", "planning", "retry"])
def test_prepared_dispatch_honors_cancellation(monkeypatch, stream, cancel_at):
    from agent import moa_loop

    owner = SimpleNamespace(_interrupt_requested=cancel_at == "before")
    facade = moa_loop.MoAChatCompletions("default", agent=owner)
    prepared = {
        "messages": [{"role": "user", "content": "task"}, {"role": "user", "content": "advice"}],
        "guidance": None,
        "aggregator": {"provider": "custom", "model": "fixture"},
        "aggregator_temperature": 0.2,
    }
    monkeypatch.setattr(moa_loop, "_slot_runtime", lambda slot: {
        "provider": "custom", "model": "fixture", "base_url": "http://fixture.local/v1",
    })
    def plan(messages, tools, guidance, runtime):
        if cancel_at == "planning":
            owner._interrupt_requested = True
        return messages, tools
    monkeypatch.setattr(facade, "_plan_aggregator_cache", plan)
    calls = []
    failure = RuntimeError("fixture rejected alternation")
    def send(**kwargs):
        calls.append(kwargs)
        owner._interrupt_requested = True
        raise failure
    monkeypatch.setattr(moa_loop, "call_llm", send)
    monkeypatch.setattr(moa_loop, "is_role_alternation_rejection", lambda exc, runtime: exc is failure)
    with pytest.raises(InterruptedError, match="before MoA aggregator"):
        facade.create(_moa_prepared_request=prepared, stream=stream)
    assert len(calls) == (1 if cancel_at == "retry" else 0)


@pytest.mark.parametrize("owner", [None, SimpleNamespace(_interrupt_requested=False)])
@pytest.mark.parametrize("outcome", ["normal", "empty", "error"])
def test_uncancelled_dispatch_preserves_result_and_configuration(monkeypatch, owner, outcome):
    from agent import moa_loop

    facade = moa_loop.MoAChatCompletions("default", agent=owner)
    prepared = {
        "messages": [{"role": "user", "content": "task"}], "guidance": None,
        "aggregator": {"provider": "custom", "model": "fixture"}, "aggregator_temperature": 0.3,
    }
    monkeypatch.setattr(moa_loop, "_slot_runtime", lambda slot: {"provider": "custom", "model": "fixture", "base_url": "http://fixture.local/v1"})
    monkeypatch.setattr(facade, "_plan_aggregator_cache", lambda messages, tools, *args: (messages, tools))
    calls = []
    response = _response("done" if outcome == "normal" else "")
    failure = ValueError("fixture failure")
    def send(**kwargs):
        calls.append(kwargs)
        if outcome == "error":
            raise failure
        return response
    monkeypatch.setattr(moa_loop, "call_llm", send)
    if outcome == "error":
        with pytest.raises(ValueError) as raised:
            facade.create(_moa_prepared_request=prepared, max_tokens=73)
        assert raised.value is failure
    else:
        assert facade.create(_moa_prepared_request=prepared, max_tokens=73) is response
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 73
    assert calls[0]["temperature"] == 0.3
