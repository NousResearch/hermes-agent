"""A transient OAuth token-endpoint failure keeps the sign-in and is never reported as a logout.

The refresh POST of an expired pooled Anthropic login hit a token endpoint answering 503. The pool
benched the only credential like an unknown failure, and resolution then said "No Anthropic
credentials found. Run 'hermes auth add anthropic'", which also made cron's preflight skip the
job as "provider credential missing". The endpoint here is a real loopback HTTP server.
"""

from __future__ import annotations

import http.server
import json
import threading
import time

import pytest

from hermes_cli.auth import AuthError


@pytest.fixture
def token_endpoint(monkeypatch):
    """Loopback OAuth token endpoint: 503 until ``state["ok"]`` is set, then a fresh token pair."""
    state = {"ok": False, "hits": 0}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            state["hits"] += 1
            if state["ok"]:
                body = json.dumps({"access_token": "sk-ant-oat01-fresh", "refresh_token": "rt-2",
                                   "expires_in": 3600}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(503)
            self.end_headers()
            self.wfile.write(b'{"error":"overloaded"}')

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr("agent.anthropic_credentials._OAUTH_TOKEN_URLS",
                        [f"http://127.0.0.1:{server.server_port}/v1/oauth/token"])
    yield state
    server.shutdown()


@pytest.fixture
def expired_anthropic_login(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("ANTHROPIC_TOKEN", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    (home / "config.yaml").write_text("auth:\n  adopt_external_logins: false\n", encoding="utf-8")
    monkeypatch.setattr("agent.anthropic_credentials.read_claude_code_credentials", lambda *a, **k: None)
    from agent.credential_pool import write_credential_pool
    write_credential_pool("anthropic", [dict(
        id="abc12345", label="work", auth_type="oauth", priority=0, source="manual:hermes_pkce",
        access_token="sk-ant-oat01-old", refresh_token="rt-1", expires_at_ms=int(time.time() * 1000) - 1000)])
    return home


def test_token_endpoint_503_is_a_retryable_outage_not_a_logout(token_endpoint, expired_anthropic_login):
    from cron.scheduler_preflight import _preflight_check_provider_key
    from hermes_cli.runtime_provider import resolve_runtime_provider

    with pytest.raises(AuthError) as info:
        resolve_runtime_provider(requested="anthropic")

    assert token_endpoint["hits"] == 1
    assert info.value.retryable is True and not info.value.relogin_required
    assert "temporarily unavailable (HTTP 503)" in str(info.value)
    assert "hermes auth add" not in str(info.value)
    # An overnight cron job is not refused as "provider credential missing" on the blip.
    assert _preflight_check_provider_key({"id": "j1", "provider": "anthropic", "model": "claude-sonnet-4-5"}, {}) is None


def test_gateway_reply_for_a_token_endpoint_outage_does_not_ask_for_login():
    """Every provider's retryable, no-re-login AuthError (here Nous's 5xx refresh error) is a
    retryable server failure in chat, not "Use /login to sign in again"."""
    from gateway.config import Platform
    from gateway.run_turn_runner import _resolution_failure_reply

    outage = AuthError("Nous Portal is temporarily unavailable (HTTP 503).", provider="nous",
                       code="temporarily_unavailable", retryable=True)
    wrapped = RuntimeError("provider resolution failed")
    wrapped.__cause__ = outage

    text, reason, retryable = _resolution_failure_reply(wrapped, Platform.TELEGRAM)

    assert retryable is True and reason == "server_error"
    assert "temporarily unavailable" in text and "/login" not in text
