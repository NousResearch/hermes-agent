"""LM Studio's unloaded-model errors stop without retries or fallback (PR #62765)."""

import json

import httpx
import pytest
from openai import APIError, APIStatusError

from agent.error_classifier import FailoverReason, classify_api_error


@pytest.mark.parametrize("status", [None, 400, 404])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "message, should_fallback",
    [
        ("Model unloaded", False),
        ("Model not loaded", False),
        ("No model loaded", False),
        ("No models loaded", False),
        ("No chat-capable models are loaded", False),
        ("model not found", True),
        ("unknown model", True),
    ],
)
def test_unloaded_and_missing_models_have_distinct_recovery(status, nested, message, should_fallback):
    """The SDK body can carry the only useful wording; ordinary missing IDs still fall back."""
    request = httpx.Request("POST", "http://127.0.0.1:1234/v1/chat/completions")
    payload = {"message": message}
    body = {"error": payload} if nested else payload
    error = (
        APIStatusError("Request failed", response=httpx.Response(status, request=request), body=body)
        if status else APIError(message, request=request, body=body)
    )

    classified = classify_api_error(error, provider="lmstudio")

    assert classified.reason is FailoverReason.model_not_found
    assert classified.retryable is False
    assert classified.should_fallback is should_fallback


@pytest.mark.parametrize(
    "status, message, should_fallback",
    [(400, "No models loaded", False), (404, "Model unloaded", False), (404, "model not found", True)],
)
def test_real_turn_stops_unloaded_model_but_recovers_missing_id(model_peer, monkeypatch, status, message,
                                                               should_fallback):
    """Real HTTP + SDK + agent loop; both primary and fallback are disposable local endpoints."""
    from run_agent import AIAgent

    handler = model_peer.RequestHandlerClass
    respond = handler.do_POST

    def primary_error(self):
        if self.path != "/primary/v1/chat/completions":
            return respond(self)
        self.server.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": {"message": message}}).encode())

    monkeypatch.setattr(handler, "do_POST", primary_error)
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kwargs: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda **kwargs: {})
    base = f"http://127.0.0.1:{model_peer.server_port}"
    agent = AIAgent(
        api_key="fixture", provider="custom", model="primary-fixture", base_url=f"{base}/primary/v1",
        api_mode="chat_completions", quiet_mode=True, skip_context_files=True, skip_memory=True,
        skip_background_review=True, save_trajectories=False, max_iterations=3, enabled_toolsets=[],
        fallback_model=[{"provider": "custom", "model": "fallback-fixture", "base_url": f"{base}/fallback/v1",
                         "api_key": "fixture", "api_mode": "chat_completions"}],
    )
    try:
        assert agent._has_pending_fallback()
        result = agent.run_conversation("Reply with the fixture acknowledgement.")
        called_models = [request["model"] for request in model_peer.requests]
        if should_fallback:
            assert called_models == ["primary-fixture", "fallback-fixture"]
            assert "LOCAL_ACK" in result["final_response"]
            assert agent._fallback_index == 1
        else:
            assert called_models == ["primary-fixture"]
            assert result["failure_reason"] == FailoverReason.model_not_found.value
            assert result["failure_retryable"] is False
            assert agent._fallback_index == 0
            assert agent._has_pending_fallback()
    finally:
        agent.close()
