"""Token-dense summaries must fit the summariser, including rolling handoffs."""

from unittest.mock import patch
import re

import pytest

from agent.context_compressor import ContextCompressor
from agent.model_metadata import estimate_tokens_rough


@pytest.mark.parametrize("mode", ["lean", "legacy"])
@pytest.mark.parametrize("rolling", [False, True])
def test_dense_summary_uses_one_shared_input_budget(mode, rolling):
    compressor = ContextCompressor(
        model="local/test", provider="custom", base_url="http://127.0.0.1:8080/v1",
        config_context_length=131072, quiet_mode=True, tail_mode=mode,
    )
    compressor._previous_summary = "既存の情報" * 30000 if rolling else None
    compressor._active_compression_telemetry = {}
    turns = [{"role": "user" if i % 2 == 0 else "assistant",
              "content": f"record-{i} " + "日本語の重要な記録" * 1000} for i in range(100)]
    prompts = []

    def complete(prompt, _started):
        prompts.append(prompt)
        return "## Goal\nContinue the user's work."

    with patch.object(compressor, "_call_summary_llm", side_effect=complete):
        assert compressor._generate_summary(turns)
    assert len(prompts) == 1
    # 25% input plus the scaffold and output reservation, rather than 160K CJK characters.
    assert estimate_tokens_rough(prompts[0]) <= 40000
    assert "record-0" in prompts[0] and "record-99" in prompts[0]
    if rolling:
        assert "既存の情報" in prompts[0]
    assert "omitted" in prompts[0] or "elided" in prompts[0]
    if mode == "lean":
        assert "record-50" in prompts[0]
        assert compressor._active_compression_telemetry["summary_input_sampled_record_count"] == len(
            re.findall(r"record-\d+ ", prompts[0]))


def test_small_auxiliary_window_bounds_the_complete_prompt():
    compressor = ContextCompressor(model="main/test", config_context_length=262144, quiet_mode=True)
    compressor.summary_model = "aux/test"
    compressor._summary_model_context_length = 65536
    compressor._previous_summary = "旧" * 160000
    prompts = []
    with patch.object(compressor, "_call_summary_llm", side_effect=lambda p, _: prompts.append(p) or "Summary"):
        compressor._generate_summary([{"role": "user", "content": "新" * 6000}] * 80, focus_topic="焦点" * 5000)
    assert len(prompts) == 1
    assert estimate_tokens_rough(prompts[0]) + compressor.max_summary_tokens < 65536


@pytest.mark.parametrize("pinned", [False, True])
def test_supported_local_fallback_keeps_the_complete_prompt_under_64k(pinned):
    from agent.auxiliary_task_config import session_auxiliary_scope
    from agent.context_compressor import pin_summary_route

    compressor = ContextCompressor(model="local/main", config_context_length=262144, quiet_mode=True)
    route = {"provider": "custom", "model": "local/fallback", "base_url": "http://127.0.0.1:8081/v1"}
    prompts = []
    task = {"compression": {"fallback_chain": [] if pinned else [route]}}
    with (session_auxiliary_scope(task), pin_summary_route(route if pinned else None),
          patch.object(compressor, "_call_summary_llm", side_effect=lambda p, _: prompts.append(p) or "Summary")):
        compressor._generate_summary([{"role": "user", "content": "新" * 6000}] * 80, focus_topic="焦点" * 9000)
    assert len(prompts) == 1
    assert estimate_tokens_rough(prompts[0]) + compressor.max_summary_tokens < 64000


def test_local_http_summary_sends_a_bounded_request_without_cloud_fallback():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from agent.auxiliary_task_config import session_auxiliary_scope

    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            body = json.dumps({"id": "local-test", "object": "chat.completion", "created": 0, "model": "local/test",
                               "choices": [{"index": 0, "message": {"role": "assistant", "content": "## Goal\nContinue."},
                                            "finish_reason": "stop"}]}).encode()
            content_type = "application/json"
            if received[-1].get("stream"):
                content_type = "text/event-stream"
                chunk = {"id": "local-test", "object": "chat.completion.chunk", "created": 0, "model": "local/test",
                         "choices": [{"index": 0, "delta": {"content": "## Goal\nContinue."}, "finish_reason": "stop"}]}
                body = ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    route = {"provider": "custom", "model": "local/test", "base_url": endpoint, "api_key": "local-test"}
    compressor = ContextCompressor(**{k: v for k, v in route.items() if k != "model"}, model="local/test",
                                   config_context_length=131072, quiet_mode=True)
    try:
        with session_auxiliary_scope({"compression": {**route, "fallback_chain": [], "max_retries": 0}}):
            assert compressor._generate_summary([{"role": "user", "content": "重要" * 3000}] * 80)
        assert len(received) == 1
        assert estimate_tokens_rough(received[0]["messages"][0]["content"]) < 40000
        assert "max_tokens" not in received[0] and "max_completion_tokens" not in received[0]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
