"""A batch consumer can reject partial streams without disabling length recovery.

Exercise the real SDK, stream reader, config loader and conversation loop against
a local SSE server. No model credentials or external services are needed.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import yaml

from run_agent import AIAgent


@pytest.mark.parametrize(
    "field,ending,continue_partial",
    [
        (field, ending, False)
        for field in ("content", "reasoning_content", "tool_calls")
        for ending in ("disconnect", "eof")
        if (field, ending) != ("reasoning_content", "disconnect")
    ] + [("content", "eof", None), ("content", "length", False),
         ("content", "length_then_eof", False)],
)
def test_partial_stream_continuation_policy(tmp_path, monkeypatch, field, ending, continue_partial):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_STREAM_RETRIES", "0")
    settings = {"api_max_retries": 1, "environment_probe": False}
    if continue_partial is not None:
        settings["partial_stream_continuation"] = continue_partial
    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"agent": settings}), encoding="utf-8")
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not self.path.endswith("/chat/completions"):
                self.send_response(404)
                self.end_headers()
                return
            requests.append(body)
            first = len(requests) == 1
            prior_length = ending == "length_then_eof"
            partial = first or (prior_length and len(requests) == 2)
            delta = {field: "Unfinished answer"} if partial else {"content": "Finished answer."}
            if first and prior_length:
                delta = {"content": "First section. "}
            if first and field == "tool_calls":
                delta = {"tool_calls": [{"index": 0, "id": "call_test", "type": "function",
                                         "function": {"name": "terminal", "arguments": '{"command":'}}]}
            chunk = {"id": "fixture", "object": "chat.completion.chunk", "created": 1,
                     "model": "fixture", "choices": [{"index": 0, "delta": delta,
                                                       "finish_reason": None}]}
            data = f"data: {json.dumps(chunk)}\n\n".encode()
            if first and field == "tool_calls" and ending == "disconnect":
                # Without visible text, a socket error is raised to the API retry
                # layer instead of returning a recoverable partial-stream stub.
                prefix = dict(chunk, choices=[{"index": 0, "delta": {"content": "Running command."},
                                               "finish_reason": None}])
                data = f"data: {json.dumps(prefix)}\n\n".encode() + data
            if not partial or ending == "length" or (first and prior_length):
                chunk["choices"] = [{"index": 0, "delta": {},
                                     "finish_reason": "length" if first else "stop"}]
                data += f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            if first and ending == "disconnect":
                self.send_header("Transfer-Encoding", "chunked")
                data = f"{len(data):x}\r\n".encode() + data + b"\r\n"
            self.end_headers()
            self.wfile.write(data)
            self.wfile.flush()
            # A disconnect deliberately omits the terminal HTTP chunk. EOF is
            # a clean HTTP close with no terminal model event instead.
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        agent = AIAgent(
            model="fixture", provider="custom", api_mode="chat_completions",
            base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="fixture-only",
            enabled_toolsets=[], max_iterations=3, max_tokens=128, quiet_mode=True,
            skip_context_files=True, skip_memory=True, skip_background_review=True,
            stream_delta_callback=lambda text: None,
        )
        result = agent.run_conversation("Answer the fixture question.", system_message="Fixture system.")
    finally:
        server.shutdown()
        worker.join(timeout=5)
        server.server_close()

    if continue_partial is False and ending != "length":
        assert len(requests) == (2 if ending == "length_then_eof" else 1)
        assert result["completed"] is False
        assert result["partial"] is True
        assert result["failure_reason"] == "invalid_response"
        assert not any(m.get("tool_calls") for m in result["messages"])
        if field == "content":
            assert "Unfinished answer" in result["final_response"]
            assert any("Unfinished answer" in (m.get("content") or "")
                       for m in result["messages"] if m.get("role") == "assistant")
        if ending == "length_then_eof":
            assert "First section." in result["final_response"]
            assert not any(m.get("_length_continuation_nudge") or m.get("_length_continuation_fragment")
                           for m in result["messages"])
    else:
        assert len(requests) == 2
        assert result["completed"] is True
        assert "Finished answer." in result["final_response"]
