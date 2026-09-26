"""Native API Skill delivery using real Agents/SDKs and synthetic HTTP only."""
import json
from contextlib import suppress
from copy import deepcopy

import httpx
import pytest

from tests.agent.test_skill_reload_delivery import hermes_home, _write_skill


@pytest.mark.parametrize("api_mode", ["anthropic_messages", "bedrock_converse"])
@pytest.mark.parametrize("case", ["valid", "suffix", "wrong_id", "wrong_role", "partial", "quote", "http_error"])
def test_native_skill_delivery(hermes_home, monkeypatch, api_mode, case):
    from run_agent import AIAgent
    from hermes_cli.plugins import get_plugin_manager
    from agent import bedrock_adapter

    skill_name = _write_skill(hermes_home)
    requests, pending_at_http = [], []
    agent = None

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        first = len(requests) == 1
        if not first:
            pending_at_http.append([cid for cid, _ in agent.context_compressor.pending_skill_view_results()])
            if case == "http_error":
                return httpx.Response(400, json={"type": "error", "error": {"type": "invalid_request_error", "message": "Synthetic failure"}})
        if api_mode == "anthropic_messages":
            assert request.url.path.endswith("/messages")
            blocks = ([{"type": "tool_use", "id": "delivery_call", "name": "skill_view", "input": {"name": skill_name}}]
                      if first else [{"type": "text", "text": "Done."}])
            response = {"id": "msg_fixture", "type": "message", "role": "assistant", "model": "claude-sonnet-4-20250514",
                        "content": blocks, "stop_reason": "tool_use" if first else "end_turn", "stop_sequence": None,
                        "usage": {"input_tokens": 128, "output_tokens": 8}}
        else:
            assert request.url.path.endswith("/converse")
            blocks = ([{"toolUse": {"toolUseId": "delivery_call", "name": "skill_view", "input": {"name": skill_name}}}]
                      if first else [{"text": "Done."}])
            response = {"output": {"message": {"role": "assistant", "content": blocks}},
                        "stopReason": "tool_use" if first else "end_turn",
                        "usage": {"inputTokens": 128, "outputTokens": 8, "totalTokens": 136}, "metrics": {"latencyMs": 1}}
        return httpx.Response(200, json=response)

    mock_http = httpx.MockTransport(respond)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda _self, request: mock_http.handle_request(request))
    if api_mode == "bedrock_converse":
        from botocore.awsrequest import AWSResponse
        from botocore.httpsession import URLLib3Session

        class RawResponse:
            def __init__(self, content):
                self.content = content

            def stream(self, amt=None, decode_content=False):
                yield self.content

        def send(_self, request):
            response = mock_http.handle_request(httpx.Request(request.method, request.url, content=request.body))
            return AWSResponse(request.url, response.status_code, dict(response.headers), RawResponse(response.content))

        monkeypatch.setattr(URLLib3Session, "send", send)
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "synthetic-access")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "synthetic-secret")
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
        bedrock_adapter.reset_client_cache()

    def mutate(*, request, next_call=None, **_context):
        payload = deepcopy(request)
        for message in payload.get("messages", []):
            for block in message.get("content", []):
                if not isinstance(block, dict):
                    continue
                result = block if api_mode == "anthropic_messages" and block.get("type") == "tool_result" else block.get("toolResult")
                if not isinstance(result, dict):
                    continue
                identity = "tool_use_id" if api_mode == "anthropic_messages" else "toolUseId"
                if result.get(identity) != "delivery_call":
                    continue
                if case == "wrong_id":
                    result[identity] = "other_call"
                elif case == "wrong_role":
                    message["role"] = "assistant"
                elif case in ("partial", "quote"):
                    exact = result["content"]
                    result["content"] = "Removed." if api_mode == "anthropic_messages" else [{"text": "Removed."}]
                    payload["metadata"] = {"quoted": exact}
                    if case == "quote":
                        message["content"] = [{"type": "text", "text": json.dumps(exact)}]
                elif case == "suffix":
                    if isinstance(result["content"], str):
                        result["content"] += "\nAdditional explanation."
                    else:
                        result["content"][0]["text"] += "\nAdditional explanation."
        return next_call(payload) if next_call else {"request": payload}

    agent = AIAgent(model="claude-sonnet-4-20250514" if api_mode == "anthropic_messages" else "amazon.nova-pro-v1:0",
                    provider="custom" if api_mode == "anthropic_messages" else "bedrock", api_mode=api_mode,
                    base_url="https://mock.anthropic.test" if api_mode == "anthropic_messages" else "https://bedrock-runtime.us-east-1.amazonaws.com",
                    api_key="synthetic-test-key", enabled_toolsets=["skills"], max_iterations=2,
                    quiet_mode=True, skip_context_files=True, skip_memory=True)
    requests.clear()  # Initialization may make a synthetic context-window probe.
    agent._persist_session = lambda *a, **k: None
    agent._save_trajectory = lambda *a, **k: None
    agent._cleanup_task_resources = lambda *a, **k: None
    agent._disable_streaming = True
    monkeypatch.setitem(get_plugin_manager()._middleware, "llm_execution", [mutate])
    try:
        result = agent.run_conversation("Read the synthetic skill.")
        pending = [cid for cid, _ in agent.context_compressor.pending_skill_view_results()]
    finally:
        with suppress(Exception):
            agent.close()
        bedrock_adapter.reset_client_cache()
    if case in ("valid", "suffix", "http_error"):
        assert len(requests) >= 2 if case == "http_error" else len(requests) == 2, result
        assert pending_at_http and all(ids == ["delivery_call"] for ids in pending_at_http), result
        assert pending == (["delivery_call"] if case == "http_error" else [])
        assert bool(result.get("completed")) == (case != "http_error")
    else:
        assert len(requests) == 1, result
        assert pending == ["delivery_call"]
        assert result.get("turn_exit_reason") == "skill_reload_delivery_blocked", result
