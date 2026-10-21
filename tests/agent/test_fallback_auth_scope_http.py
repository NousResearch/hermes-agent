"""Loopback HTTP evidence: twenty sibling models do not repeat a rejected credential."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import openai
import pytest

from agent.error_classifier import FailoverReason
from tests.agent.test_provider_fallback import _make_agent


def test_rejected_key_is_sent_once_before_the_explicit_alternative():
    attempts = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def reply(self, status, value):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200, {"data": [{"id": "healthy", "context_length": 131072}]})

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not self.path.endswith("/chat/completions"):
                self.reply(200, {})
                return
            attempts.append((self.headers.get("Authorization"), payload["model"]))
            if self.headers.get("Authorization") == "Bearer rejected-test-key":
                self.reply(401, {"error": {"message": "Invalid API key", "type": "authentication_error"}})
                return
            self.reply(200, {"id": "loopback", "object": "chat.completion", "created": 0, "model": payload["model"],
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": "Recovered"},
                                          "finish_reason": "stop"}]})

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    chain = [{"provider": "custom", "model": f"sibling-{i}", "base_url": endpoint,
              "api_key": "rejected-test-key"} for i in range(20)]
    chain.append({"provider": "custom", "model": "healthy", "base_url": endpoint, "api_key": "healthy-test-key"})
    agent = _make_agent(fallback_model=chain)
    agent.provider, agent.model, agent.base_url, agent.api_key = "custom", "primary", endpoint, "rejected-test-key"
    agent._credential_pool = None
    primary = openai.OpenAI(base_url=endpoint, api_key=agent.api_key, max_retries=0)
    agent.client = primary
    try:
        with pytest.raises(openai.AuthenticationError):
            primary.chat.completions.create(model=agent.model, messages=[{"role": "user", "content": "local test"}])
        assert agent._try_activate_fallback(FailoverReason.auth)
        response = agent.client.chat.completions.create(model=agent.model, messages=[{"role": "user", "content": "local test"}])
        assert response.choices[0].message.content == "Recovered"
        assert attempts == [("Bearer rejected-test-key", "primary"), ("Bearer healthy-test-key", "healthy")]
    finally:
        primary.close()
        agent.client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
