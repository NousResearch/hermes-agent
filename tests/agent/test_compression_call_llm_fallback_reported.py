"""A compression summary that ``call_llm`` recovered on a fallback route is reported as a fallback.

When ``auxiliary.compression`` points at a dead endpoint, ``call_llm``'s recovery ladder answers the summary
with the main model. The compressor saw a plain success: no user warning, and telemetry
``method=llm_summary fallback_used=false``. Real HTTP path (closed port + loopback main model, temp ``HERMES_HOME``).
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import agent.auxiliary_client as aux
from agent.context_compressor import ContextCompressor

SUMMARY = "## Goal\nAnswer the numbered questions.\n## Progress\nQuestions answered.\n## Next Steps\nContinue.\n"


def _chunk(text, finish=None):
    payload = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m",
               "choices": [{"index": 0, "delta": {"content": text} if text else {}, "finish_reason": finish}]}
    return f"data: {json.dumps(payload)}\n\n".encode()


def _serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    return server, thread


def _stop(server, thread):
    aux.shutdown_cached_clients()
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def _write_config(home, config):
    (home / "config.yaml").write_text(json.dumps(config), encoding="utf-8")


@pytest.fixture
def main_model_endpoint():
    """The main model: answers every chat completion (streamed or not) with a summary."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            seen.append(body.get("model"))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(_chunk(SUMMARY) + _chunk("", "stop") + b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True

    aux.shutdown_cached_clients()
    aux._reset_aux_unhealthy_cache()
    server, thread = _serve(Handler)
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", seen
    finally:
        _stop(server, thread)


def _closed_port_url():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


def _history():
    messages = []
    for i in range(30):
        messages.append({"role": "user", "content": f"Question {i}: explain topic {i}. " + "lorem ipsum " * 200})
        messages.append({"role": "assistant", "content": f"Answer {i}: " + "dolor sit amet " * 200})
    return messages


def test_summary_recovered_on_the_main_model_by_call_llm_is_reported(tmp_path, monkeypatch, main_model_endpoint):
    main_url, seen = main_model_endpoint
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    _write_config(tmp_path, {
        "model": {"provider": "custom", "base_url": main_url, "api_key": "dummy", "default": "main-fake",
                  "context_length": 64000},
        "auxiliary": {"compression": {"provider": "custom", "base_url": _closed_port_url(), "api_key": "dummy",
                                      "model": "dead-summarizer", "timeout": 20}},
    })
    compressor = ContextCompressor(model="main-fake", provider="custom", base_url=main_url, api_key="dummy",
                                   config_context_length=64000, quiet_mode=True)

    compressed = compressor.compress(_history(), force=True)

    assert len(compressed) < 60 and seen == ["main-fake"]  # the main model wrote the summary
    telemetry = compressor._last_compression_telemetry
    assert telemetry["fallback_used"] is True
    assert telemetry["method"] == "aux_fallback_main"
    # The user warning (``_warn_summary_or_aux_fallback``) keys on these.
    assert compressor._last_aux_model_failure_model == "dead-summarizer"
    assert "connection error" in compressor._last_aux_model_failure_error


@pytest.mark.parametrize("label, warns", [("main-agent(custom)", True), ("fallback_chain[0](openrouter)", False)])
def test_only_a_main_model_recovery_names_the_broken_compression_model(label, warns):
    compressor = ContextCompressor(model="main-fake", config_context_length=64000, quiet_mode=True)
    compressor._active_compression_telemetry = telemetry = {}

    # Same model name on both routes: only the route that answered decides the "main model" warning.
    compressor._note_aux_route_fallback(
        {"provider": "custom", "model": "main-fake", "failed_model": "dead", "failure": "connection error: x",
         "fallback_label": label})

    assert telemetry["fallback_used"] is True
    assert (compressor._last_aux_model_failure_model == "dead") is warns
