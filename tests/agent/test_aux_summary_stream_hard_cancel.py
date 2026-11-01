"""A hard cancel (/stop, session boundary) of a compression summary closes the provider stream.

The owner unwinds at once, but the streamed summary runs in an isolated daemon whose chunk consumer only
checked the total ceiling and the host deadline, so the provider stream kept generating (and billing) until
that deadline, up to ten minutes later. It must close at the next chunk. Real HTTP path against a loopback
server that trickles a chunk every 50 ms, with a temp ``HERMES_HOME``.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import agent.auxiliary_client as aux


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
def trickle_endpoint():
    """Streams one content chunk every 50 ms for up to 30 s; records when the client hangs up."""
    sent_some, client_closed = threading.Event(), threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                for n in range(600):
                    self.wfile.write(_chunk("word "))
                    self.wfile.flush()
                    if n == 3:
                        sent_some.set()
                    time.sleep(0.05)  # health: allow HX013 -- the trickle pace is the slow provider under test
            except OSError:
                client_closed.set()
            self.close_connection = True

    aux.shutdown_cached_clients()
    aux._reset_aux_unhealthy_cache()
    server, thread = _serve(Handler)
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", sent_some, client_closed
    finally:
        _stop(server, thread)


def test_hard_cancel_closes_the_streamed_summary_at_its_next_chunk(tmp_path, monkeypatch, trickle_endpoint):
    url, sent_some, client_closed = trickle_endpoint
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    _write_config(tmp_path, {"auxiliary": {"compression": {
        "provider": "custom", "base_url": url, "api_key": "dummy", "model": "slow-summarizer"}}})
    stop = threading.Event()
    threading.Thread(target=lambda: sent_some.wait(10) and stop.set(), daemon=True).start()

    # The compression attempt's seams: a progress hook (streams the call) and the hard-cancel source.
    with aux.aux_progress_hook(lambda: None), aux.aux_interrupt_protection(cancel_check=stop.is_set):
        with pytest.raises(aux.AuxiliaryExplicitCancellation):
            aux.call_llm(task="compression", messages=[{"role": "user", "content": "Summarize."}], timeout=20)

    assert stop.is_set()
    assert client_closed.wait(5), "the cancelled summary stream kept streaming after the hard stop"
