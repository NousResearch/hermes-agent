"""Native Zen requests remain interruptible through the owner-safe request lifecycle."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread

import httpx
import pytest

from agent.gemini_native_adapter import GeminiAPIError, GeminiNativeClient
from run_agent import AIAgent


@pytest.mark.parametrize("stream", [False, True])
def test_native_request_abort_unblocks_the_owner_thread(monkeypatch, stream):
    entered, peer_closed, release, finished = Event(), Event(), Event(), Event()
    errors = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            entered.set()
            # A provider observes the interrupted connection before finishing its response.
            self.connection.settimeout(3)
            try:
                if self.connection.recv(1) == b"":
                    peer_closed.set()
                    return
            except TimeoutError:
                pass
            release.wait(10)
            payload = {"candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}]}
            body = json.dumps(payload).encode()
            if stream:
                body = b"data: " + body + b"\n\n"
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # The tested cancellation shuts down this exact connection.

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    class LoopbackTransport(httpx.HTTPTransport):
        def handle_request(self, request):
            request.url = request.url.copy_with(scheme="http", host="127.0.0.1", port=server.server_port)
            return super().handle_request(request)

    monkeypatch.setattr(AIAgent, "_build_keepalive_http_client", lambda *_a, **_kw:
                        httpx.Client(transport=LoopbackTransport(), timeout=10))
    agent = AIAgent(provider="opencode-zen", model="gemini-4-argon", api_key="dummy-test-key",
                    base_url="https://opencode.ai/zen/v1", quiet_mode=True,
                    skip_context_files=True, skip_memory=True)
    client = agent._create_request_openai_client(reason="native_abort_contract")
    assert isinstance(client, GeminiNativeClient)

    def request():
        try:
            result = client.chat.completions.create(model=agent.model, stream=stream,
                                                   messages=[{"role": "user", "content": "OK"}])
            if stream:
                list(result)
        except (httpx.HTTPError, GeminiAPIError) as exc:
            errors.append(exc)
        finally:
            agent._close_request_openai_client(client, reason="aborted")
            finished.set()

    worker = Thread(target=request, daemon=True)
    try:
        worker.start()
        assert entered.wait(3), "Request never reached the loopback server"
        agent._abort_request_openai_client(client, reason="native_abort_contract")
        assert peer_closed.wait(3), "The interrupted request must disconnect from the provider"
        assert finished.wait(3), "Native provider request remained blocked after cancellation"
        assert errors, "The aborted operation must not report a successful completion"
        assert client.is_closed
    finally:
        release.set()
        worker.join(5)
        agent.close()
        server.shutdown()
        server.server_close()
        server_thread.join(5)
