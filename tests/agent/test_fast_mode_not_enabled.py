"""An organization-disabled Anthropic fast request retries at standard speed (#141768)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import anthropic
import httpx
import pytest

from agent.fast_mode import fast_mode_unprovisioned

_DISABLED = (
    "Fast mode is not enabled for your organization. "
    "An organization admin must enable this feature."
)


@pytest.mark.parametrize("status,message,fast,expected", [
    (400, _DISABLED, True, True),
    (400, _DISABLED, False, False),
    (400, "Feature is not enabled for your organization.", True, False),
    (400, "Invalid messages: expected an assistant message", True, False),
    (429, "Rate limit exceeded", True, False),
    (500, _DISABLED, True, False),
])
def test_only_disabled_fast_requests_are_recoverable(status, message, fast, expected):
    response = httpx.Response(status, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    error_type = anthropic.BadRequestError if status == 400 else anthropic.APIStatusError
    error = error_type(message, response=response, body={"type": "invalid_request_error", "message": message})
    kwargs = {"extra_body": {"speed": "fast"}} if fast else {}
    assert fast_mode_unprovisioned(error, kwargs) is expected


@pytest.mark.parametrize("service_tier", ["auto", "priority"])
def test_disabled_fast_mode_retries_same_model_and_stays_session_scoped(monkeypatch, service_tier):
    import run_agent

    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            requests.append(body)
            if body.get("speed") == "fast":
                payload = json.dumps({"type": "error", "error": {
                    "type": "invalid_request_error", "message": _DISABLED,
                }}).encode()
                self.send_response(400)
                self.send_header("content-type", "application/json")
            else:
                message = {"id": "msg_test", "type": "message", "role": "assistant",
                           "model": body["model"], "content": [], "stop_reason": None,
                           "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 0}}
                events = [
                    ("message_start", {"message": message}),
                    ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
                    ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "standard reply"}}),
                    ("content_block_stop", {"index": 0}),
                    ("message_delta", {"delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 2}}),
                    ("message_stop", {}),
                ]
                payload = "".join(f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n" for kind, data in events).encode()
                self.send_response(200)
                self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    # Keep first-party request shaping; only the physical SDK endpoint is local.
    monkeypatch.setattr("agent.anthropic_adapter.build_anthropic_client", lambda *_a, **_k:
                        anthropic.Anthropic(api_key="test-key", base_url=endpoint, max_retries=0))

    agents = []

    def make_agent():
        agent = run_agent.AIAgent(
            api_key="test-key", base_url="https://api.anthropic.com", provider="anthropic",
            api_mode="anthropic_messages", model="claude-opus-5", service_tier=service_tier,
            request_overrides={"speed": "fast"} if service_tier == "priority" else {},
            quiet_mode=True, skip_context_files=True,
            skip_memory=True, enabled_toolsets=[], max_iterations=2,
        )
        agents.append(agent)
        return agent

    try:
        agent = make_agent()
        first = agent.run_conversation("first", conversation_history=[], task_id="fast-first")
        assert first["final_response"] == "standard reply"
        assert [r.get("speed") for r in requests] == ["fast", None]
        assert {r["model"] for r in requests} == {"claude-opus-5"}
        second = agent.run_conversation("second", conversation_history=first["messages"], task_id="fast-second")
        assert second["final_response"] == "standard reply"
        assert [r.get("speed") for r in requests] == ["fast", None, None]
        agent.model = "claude-opus-5-5"
        assert agent._build_api_kwargs([{"role": "user", "content": "other model"}])["extra_body"]["speed"] == "fast"
        other = make_agent()
        third = other.run_conversation("new session", conversation_history=[], task_id="fast-new-session")
        assert third["final_response"] == "standard reply"
        assert [r.get("speed") for r in requests] == ["fast", None, None, "fast", None]
    finally:
        for agent in agents:
            agent.close()
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
