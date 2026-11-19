"""An upstream account ban relayed as HTTP 200 + an SSE ``error`` event is permanent.

OpenRouter relays OpenAI's "this user has been blocked for a previous policy violation"
inside a 200 stream; the SDK raises a status-less ``APIError``. Classified ``unknown`` it
was retried ``api_max_retries`` times and the user was told the provider "looks temporarily
unavailable" — advice that can never work for a banned account.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import openai
import pytest

import run_agent
from agent.error_classifier import FailoverReason, classify_api_error

BAN = (
    "Policy Violation: this user has been blocked for a previous policy violation. "
    "Learn more: https://platform.openai.com/docs/guides/safety-best-practices"
)


@pytest.mark.parametrize("status", [None, 403])
def test_account_ban_is_a_permanent_policy_block_not_transient_or_auth(status):
    request = httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    if status is None:  # what openai/_streaming.py raises for a 200 stream carrying {"error": ...}
        err = openai.APIError(BAN, request, body={"message": BAN, "code": 403})
    else:
        err = openai.PermissionDeniedError(
            BAN, response=httpx.Response(status, request=request), body={"error": {"message": BAN}}
        )
    result = classify_api_error(err, provider="openrouter", model="openai/gpt-4.1-nano")
    assert result.reason == FailoverReason.provider_policy_blocked
    assert result.retryable is False
    assert result.should_fallback is True
    # Every key on a banned account is banned: rotating the pool only burns credentials.
    assert result.should_rotate_credential is False


def test_streamed_account_ban_fails_once_without_transient_advice():
    streamed = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or b"{}")
            if not self.path.endswith("/chat/completions"):
                self.send_response(404)
                self.end_headers()
                return
            if body.get("stream"):
                streamed.append(body)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.write(f"data: {json.dumps({'error': {'code': 403, 'message': BAN}})}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        agent = run_agent.AIAgent(
            api_key="test-key", base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
            model="m", provider="custom", quiet_mode=True, skip_context_files=True,
            skip_memory=True, enabled_toolsets=[], max_iterations=1,
        )
        result = agent.run_conversation("ping", conversation_history=[], task_id="t")
    finally:
        server.shutdown()
        server.server_close()

    assert len(streamed) == 1, f"a permanent ban was re-sent {len(streamed)} times"
    assert result["failed"] is True
    assert result["failure_retryable"] is False
    assert "temporarily unavailable" not in result["final_response"]
    assert BAN in result["final_response"]
