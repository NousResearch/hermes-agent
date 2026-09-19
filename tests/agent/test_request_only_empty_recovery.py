"""Recovery changes the next HTTP request, never the durable conversation."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


@pytest.mark.parametrize("after_tool", [False, True], ids=["user", "tool"])
@pytest.mark.parametrize("parts", [False, True], ids=["text", "parts"])
@pytest.mark.parametrize("stream", [False, True], ids=["json", "sse"])
def test_recovery_is_request_only(tmp_path, monkeypatch, after_tool, parts, stream):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("model:\n  supports_vision: true\n")
    from agent.conversation_loop import _CODEX_INCOMPLETE_NUDGE, _EMPTY_TOOL_RESPONSE_NUDGE
    from hermes_state import SessionDB
    from run_agent import AIAgent

    reasoning = "I should look this up.\n<tool_call>\nlookup({})\n</tool_call>"
    stages = (["tool"] if after_tool else []) + ["empty", "tool", "answer", "answer"]
    requests = []

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            requests.append(request["messages"])
            stage = stages.pop(0)
            message = {"role": "assistant", "content": None}
            finish = "stop"
            if stage == "tool":
                message["tool_calls"] = [{
                    "id": f"lookup-{len(requests)}", "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                }]
                finish = "tool_calls"
            elif stage == "empty":
                # Clean-stop promotion is a separate policy. Exercise the existing
                # empty ladder on main without relying on any promotion-gate PR.
                message["reasoning_content"] = reasoning
                finish = "tool_calls"
            else:
                message["content"] = "The answer is 42."
            response = {
                "id": "recovery", "model": "test-model", "created": 0,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            }
            self.send_response(200)
            if request.get("stream"):
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for call in message.get("tool_calls", []):
                    call["index"] = 0
                chunk = {**response, "choices": [{"index": 0, "delta": message, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                chunk["choices"] = [{"index": 0, "delta": {}, "finish_reason": finish}]
                self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode())
            else:
                body = json.dumps(response).encode()
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Provider) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        agent = AIAgent(
            model="test-model", provider="openai-compat", api_key="test-key",
            base_url=f"http://127.0.0.1:{server.server_port}/v1", enabled_toolsets=[],
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, max_iterations=10,
        )
        agent._disable_streaming = not stream
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        agent.tool_delay = 0
        agent.session_id = "request-only-recovery"
        agent._session_db = SessionDB(tmp_path / "recovery.db")
        agent.tools = [{"type": "function", "function": {
            "name": "lookup", "description": "Look up a number",
            "parameters": {"type": "object", "properties": {}},
        }}]
        agent.valid_tool_names = {"lookup"}
        # Only the leaf tool is substituted; Hermes owns request assembly,
        # the real SDK/HTTP transport, execution bookkeeping and SQLite writes.
        monkeypatch.setattr("model_tools.handle_function_call", lambda *a, **kw: "42")
        prompt = "Look up the answer."
        if parts:
            prompt = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": (
                    "data:image/png;base64,"
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a2eoAAAAASUVORK5CYII="
                )}},
            ]
        try:
            result = agent.run_conversation(prompt)
            assert result["final_response"] == "The answer is 42."
            retry_index = 2 if after_tool else 1
            original, retry = requests[retry_index - 1:retry_index + 1]
            if parts and not after_tool:
                assert isinstance(original[-1]["content"], list)
            hint = _EMPTY_TOOL_RESPONSE_NUDGE if after_tool else _CODEX_INCOMPLETE_NUDGE
            assert [m["role"] for m in retry] == [m["role"] for m in original]
            assert retry[:-1] == original[:-1]
            assert retry[-1] != original[-1], "Recovery must change the outgoing request"
            assert hint in json.dumps(retry[-1])
            assert requests[retry_index + 1][:len(original)] == original

            stored = agent._session_db.get_messages(agent.session_id)
            for history in (result["messages"], stored):
                serialized = json.dumps(history, default=str)
                assert hint not in serialized
                assert json.dumps(reasoning)[1:-1] not in serialized
                assert not any(m.get("_thinking_prefill") or m.get("_empty_recovery_synthetic") for m in history)
                assert sum(m["role"] == "user" for m in history) == 1
            assert agent._thinking_prefill_retries == 0
            agent.run_conversation("Next question", conversation_history=result["messages"])
            assert hint not in json.dumps(requests[-1])
            assert not stages
        finally:
            agent.client.close()
            agent._session_db.close()
            server.shutdown()
            thread.join(timeout=5)
