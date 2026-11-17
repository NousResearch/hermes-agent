"""A real turn against a local provider reports each model call and each rate-limit wait on the
agent's structured progress feed (``usage.recorded`` / ``model.rate_limited``)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def test_rate_limited_then_streamed_call_emits_wait_and_usage(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "model:\n  context_length: 128000\nauxiliary:\n  title_generation:\n    enabled: false\n"
    )
    from run_agent import AIAgent

    replies = ["429", "200"]

    class Provider(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if replies.pop(0) == "429":
                body = json.dumps({"error": {"message": "slow down", "type": "rate_limit_error"}}).encode()
                self.send_response(429)
                self.send_header("Retry-After", "1")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            base = {"id": "c1", "object": "chat.completion.chunk", "model": "test-model", "created": 0}
            chunks = [
                {**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": None}]},
                {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {**base, "choices": [], "usage": {
                    "prompt_tokens": 120, "completion_tokens": 7, "total_tokens": 127,
                    "prompt_tokens_details": {"cached_tokens": 100}}},
            ]
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    events = []
    with ThreadingHTTPServer(("127.0.0.1", 0), Provider) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        agent = AIAgent(
            model="test-model", provider="custom", api_key="test-key",
            base_url=f"http://127.0.0.1:{server.server_port}/v1", enabled_toolsets=[],
            quiet_mode=True, skip_context_files=True, skip_memory=True, save_trajectories=False,
            max_iterations=3,
            stream_delta_callback=lambda _text: None,
            tool_progress_callback=lambda event, *_a, **kw: events.append((event, kw)),
        )
        agent._use_prompt_caching = False
        agent.compression_enabled = False
        try:
            result = agent.run_conversation("Say hi.")
        finally:
            agent.client.close()
            server.shutdown()
            thread.join(timeout=5)

    assert result["final_response"] == "Hi"
    model_calls = [(name, kw) for name, kw in events if name in {"usage.recorded", "model.rate_limited"}]
    assert [name for name, _ in model_calls] == ["model.rate_limited", "usage.recorded"]

    waited = model_calls[0][1]
    assert waited["retry_after_ms"] == 1000
    assert waited["attempt"] == 1 and waited["max_attempts"] >= 1
    assert waited["model"] == "test-model" and waited["provider"] == "custom"

    usage = model_calls[1][1]
    assert usage["call_index"] == 1
    assert (usage["input_tokens"], usage["cache_read_tokens"], usage["output_tokens"]) == (20, 100, 7)
    assert usage["used_tokens"] == 120 and usage["window_tokens"] == 128000
    # The streamed attempt's own timing, not the turn-long span that included the 1 s wait.
    assert isinstance(usage["first_token_ms"], int) and 0 <= usage["first_token_ms"] <= usage["duration_ms"]
    assert usage["duration_ms"] < 1000
