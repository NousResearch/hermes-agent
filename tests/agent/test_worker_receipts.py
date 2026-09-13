"""Route evidence observes dispatched fields, not requested model inference."""

from types import SimpleNamespace

import pytest

from agent.worker_receipts import observe_worker_request, observe_worker_response


def test_request_records_wire_override_without_credentials_or_prompt():
    agent = SimpleNamespace(provider="provider-a", api_mode="chat_completions", _worker_route_receipt={"resolved_model": "requested"})
    payload = {"model": "requested", "reasoning_effort": "high", "messages": [{"content": "private"}],
               "extra_headers": {"authorization": "synthetic-secret"},
               "extra_body": {"model": "routed", "reasoning_effort": "low", "api_key": "synthetic-secret"}}
    observe_worker_request(agent, payload)
    observe_worker_response(agent, SimpleNamespace(model="provider-reported"))
    receipt = agent._worker_route_receipt
    assert receipt["resolved_model"] == "requested"
    assert receipt["transmitted_model"] == "routed"
    assert receipt["transmitted_reasoning_effort"] == "low"
    assert receipt["provider_reported_model"] == "provider-reported"
    assert "private" not in str(receipt) and "synthetic-secret" not in str(receipt)
    observe_worker_request(agent, {"model": "fallback"})
    assert receipt["provider_reported_model"] is None
    assert receipt["execution_attempts"][0]["provider_reported_model"] == "provider-reported"
    assert receipt["execution_attempts"][1]["provider_reported_model"] is None


@pytest.mark.parametrize("reported", [None, "upstream-model"])
def test_codex_assembler_separates_requested_and_provider_reported_model(reported):
    from agent.codex_runtime import _consume_codex_event_stream

    response_data = {"id": "fixture", "status": "completed"}
    if reported:
        response_data["model"] = reported
    response = _consume_codex_event_stream([
        {"type": "response.output_text.delta", "delta": "done"},
        {"type": "response.completed", "response": response_data},
    ], model="requested-model")
    assert response.model == "requested-model"  # preserve external compatibility
    agent = SimpleNamespace(provider="openai-codex", api_mode="codex_responses", _worker_route_receipt={})
    observe_worker_request(agent, {"model": "requested-model", "reasoning": {"effort": "high"}})
    observe_worker_response(agent, response)
    assert agent._worker_route_receipt["provider_reported_model"] == reported
    assert agent._worker_route_receipt["transmitted_reasoning_effort"] == "high"


def test_observation_runs_after_execution_middleware_at_real_dispatch(monkeypatch):
    from agent.turn_api_call import perform_api_call
    from hermes_cli import middleware
    from agent import relay_llm

    wire = []
    response = SimpleNamespace(model="actual")
    agent = SimpleNamespace(
        provider="custom", api_mode="chat_completions", session_id="fixture", platform="cli", base_url=None,
        model="requested", _worker_route_receipt={}, _disable_streaming=True, _has_pending_redirect=lambda: False,
        _interruptible_api_call=lambda kwargs: wire.append(kwargs) or response,
    )
    monkeypatch.setattr(middleware, "run_llm_execution_middleware", lambda kwargs, send, **kw: send({**kwargs, "model": "middleware-selected"}))
    monkeypatch.setattr(relay_llm, "execute", lambda kwargs, send, **kw: send(kwargs))
    result = perform_api_call(agent, api_kwargs={"model": "requested", "reasoning_effort": "medium"},
        _original_api_kwargs={}, _llm_middleware_trace=[], _moa_prepared_request=None, _retry=SimpleNamespace(),
        thinking_spinner=None, retry_count=0, api_call_count=0, api_request_id="request", effective_task_id="fixture",
        turn_id="turn", interrupted=False)
    assert result.response is response
    assert wire[0]["model"] == agent._worker_route_receipt["transmitted_model"] == "middleware-selected"


def test_partial_and_synthesized_responses_do_not_invent_actual_model():
    from agent.chat_completion_helpers import PARTIAL_STREAM_STUB_ID

    agent = SimpleNamespace(provider="custom", api_mode="chat_completions", _worker_route_receipt={})
    observe_worker_response(agent, SimpleNamespace(id=PARTIAL_STREAM_STUB_ID, model="requested"))
    assert agent._worker_route_receipt["provider_reported_model"] is None
    agent.api_mode = "bedrock_converse"
    observe_worker_response(agent, SimpleNamespace(model="requested"))
    assert agent._worker_route_receipt["provider_reported_model"] is None
