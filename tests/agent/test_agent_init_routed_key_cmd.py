"""A ``key_cmd`` provider reached through the provider router keeps its credential.

``AIAgent`` built without an explicit ``api_key``/``base_url`` resolves its client through
``resolve_provider_client`` and rebuilds it from ``_client_kwargs_from_routed``. A ``key_cmd``
client holds its token provider in the SDK's ``_api_key_provider`` slot with ``.api_key == ""``,
so mirroring ``.api_key`` alone rebuilt a client with no credential and agent init failed with
the SDK's "Missing credentials" error.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from openai import OpenAI

from agent.agent_init import _client_kwargs_from_routed


def test_routed_kwargs_pass_the_token_provider_through_uncalled():
    calls = []

    def provider():
        calls.append(1)
        return "minted-token"

    kwargs = _client_kwargs_from_routed(OpenAI(api_key=provider, base_url="http://127.0.0.1:9/v1"), None)

    assert kwargs["api_key"] is provider
    assert calls == [], "a key_cmd token is minted per request, never frozen at init"


def test_agent_on_a_key_cmd_provider_sends_the_minted_token(monkeypatch):
    from hermes_constants import get_hermes_home

    captured = {}

    class _Handler(BaseHTTPRequestHandler):
        def _reply(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._reply({"object": "list", "data": []})

        def do_POST(self):
            captured["authorization"] = self.headers.get("authorization")
            self.rfile.read(int(self.headers.get("content-length", 0)))
            self._reply({
                "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "m1",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        (get_hermes_home() / "config.yaml").write_text(
            "model:\n  provider: dbx\n  default: m1\n"
            "providers:\n  dbx:\n"
            f"    base_url: http://127.0.0.1:{server.server_port}/v1\n"
            "    api_mode: chat_completions\n"
            "    key_cmd: printf minted-token\n",
            encoding="utf-8",
        )
        from run_agent import AIAgent

        agent = AIAgent(model="m1", provider="dbx", quiet_mode=True,
                        skip_context_files=True, skip_memory=True)
        client = agent.client
        assert isinstance(client, OpenAI)
        client.chat.completions.create(model="m1", messages=[{"role": "user", "content": "hi"}])
    finally:
        server.shutdown()

    assert captured["authorization"] == "Bearer minted-token"
