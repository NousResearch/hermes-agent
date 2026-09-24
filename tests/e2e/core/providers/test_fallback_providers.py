"""Provider fallback (``fallback_providers``) through REAL ``hermes -z`` processes.

Two loopback fakes stand in for two vendors: the primary and the fallback. Everything
between the CLI and those sockets is real Hermes: config loading, credential resolution,
the retry ladder, fallback activation and the fallback client.

Proven here:

* a primary that answers every request with 503 is tried, then the turn is answered by
  the configured fallback, which receives the user's conversation and is reported as the
  model that served the turn;
* a primary whose credentials cannot be resolved walks ``fallback_providers`` at
  resolution time: a Nous Portal OAuth refresh answered with 5xx (an ``AuthError``) does,
  and one that cannot connect at all (an outage: connection refused) must too (#120608).
"""

from __future__ import annotations

import base64
import json
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

import pytest

from tests.e2e.core.providers._openai_helpers import (
    Home,
    chat_messages,
    custom_chat_config,
    db_messages,
    oneshot,
)
from tests.fakes.fake_llm_provider import FakeLLMServer
from tests.fakes.providers.chat_variants import CError, FakeChatVariantServer

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="subprocess harness is Linux-gated")

# Scenario -> "#issue one-line symptom" for scenarios red on origin/main. Strict: the test
# FAILS the moment the bug is fixed, forcing the entry out.
KNOWN: dict[str, str] = {
    "unreachable_portal_falls_back": "#120608 transport error during credential resolution skips fallback_providers",
}

FALLBACK_MODEL = "fallback-model"
PROMPT = "CANARY-PROMPT say hello"
TURN_BUDGET = 45.0


def known(name: str) -> list:
    """Marks for a scenario: a strict xfail while it is in KNOWN, nothing once fixed."""
    return [pytest.mark.xfail(strict=True, raises=AssertionError, reason=KNOWN[name])] if name in KNOWN else []


def _fallback_entry(fallback: FakeLLMServer) -> list[dict]:
    return [{"provider": "custom", "model": FALLBACK_MODEL, "base_url": fallback.base_url}]


def _bounded_turn(h: Home, prompt: str, **kw):
    """One turn that must finish well inside the harness budget: a primary that is never
    abandoned for the fallback keeps backing off for minutes, which is itself the failure."""
    try:
        return oneshot(h, prompt, timeout=TURN_BUDGET, **kw)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(f"turn still running after {TURN_BUDGET}s: the fallback never took over") from exc


def _user_texts(body: dict) -> list[str]:
    return [str(m.get("content")) for m in chat_messages(body, "user")]


def test_persistent_primary_503_is_answered_by_fallback(tmp_path) -> None:
    def always_503(_record: dict) -> CError:
        return CError(503, "Service temporarily unavailable", code="service_unavailable")

    with FakeChatVariantServer(always_503) as primary, FakeLLMServer(default_text="FROM-FALLBACK") as fallback:
        cfg = custom_chat_config(primary.base_url)
        cfg["fallback_providers"] = _fallback_entry(fallback)
        h = Home(tmp_path).write(cfg, {"OPENAI_API_KEY": "sk-fake"})
        run = _bounded_turn(h, PROMPT)
        primary_mains = primary.main_requests()
        fallback_mains = fallback.main_requests()

    assert run.proc.returncode == 0 and run.stdout.strip() == "FROM-FALLBACK", run.describe()
    assert primary_mains, "the primary was never tried"
    assert fallback_mains, f"fallback never received a request: {run.describe()}"
    assert any(PROMPT in t for t in _user_texts(fallback_mains[0])), fallback_mains[0].get("messages")
    assert fallback_mains[0].get("model") == FALLBACK_MODEL, fallback_mains[0].get("model")
    assert run.usage.get("model") == FALLBACK_MODEL, run.describe()
    answers = [r for r in db_messages(h, run.session_id)
               if r["role"] == "assistant" and "FROM-FALLBACK" in (r.get("content") or "")]
    assert len(answers) == 1, db_messages(h, run.session_id)


# Nous Portal credential-resolution outage (#120608) ------------------------------------


def _jwt(claims: dict) -> str:
    def seg(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()
    return f"{seg({'alg': 'none', 'typ': 'JWT'})}.{seg(claims)}.sig"


def _expired_nous_auth() -> dict:
    """An auth.json whose Nous invoke JWT expired an hour ago but whose refresh token is fine:
    the next turn must redeem it against the Portal before any inference call."""
    past = int(time.time()) - 3600
    iso = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(past))
    return {"version": 1, "active_provider": "nous", "providers": {"nous": {
        "portal_base_url": "https://portal.nousresearch.com",
        "inference_base_url": "https://inference-api.nousresearch.com/v1",
        "client_id": "hermes-cli", "token_type": "Bearer", "scope": "inference:invoke",
        "access_token": _jwt({"sub": "e2e-user", "scope": "inference:invoke", "exp": past}),
        "refresh_token": "refresh-e2e", "obtained_at": iso, "expires_in": 3600, "expires_at": iso,
        "agent_key": None, "agent_key_id": None, "agent_key_expires_at": None,
        "agent_key_expires_in": None, "agent_key_reused": None, "agent_key_obtained_at": None,
    }}}


def _closed_port() -> int:
    """A loopback port nothing listens on: connecting is refused immediately."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Portal5xx(BaseHTTPRequestHandler):
    """A Portal that is up but failing: every token refresh gets a 503 OAuth error body."""

    hits: list[str] = []

    def log_message(self, *_a: object) -> None:
        pass

    def do_POST(self) -> None:  # noqa: N802
        self.hits.append(self.path)
        body = json.dumps({"error": "server_error", "error_description": "portal overloaded"}).encode()
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@contextmanager
def _portal(kind: str) -> Iterator[tuple[str, list[str]]]:
    """``(portal_base_url, refresh_paths_hit)``: a 5xx-answering Portal or a refused port."""
    if kind == "refused":
        yield f"http://127.0.0.1:{_closed_port()}", []
        return
    handler = type("Portal", (_Portal5xx,), {"hits": []})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", handler.hits
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("portal_kind", [
    pytest.param("5xx", id="portal_5xx"),
    pytest.param("refused", id="portal_refused", marks=known("unreachable_portal_falls_back")),
])
def test_primary_credential_resolution_failure_falls_back(tmp_path, portal_kind: str) -> None:
    dead = f"http://127.0.0.1:{_closed_port()}"
    with _portal(portal_kind) as (portal_url, portal_hits), FakeLLMServer(default_text="FROM-FALLBACK") as fallback:
        env = {
            "HERMES_PORTAL_BASE_URL": portal_url,
            # Nothing may reach a real vendor host; the loopback fakes stay direct.
            "HTTPS_PROXY": dead, "HTTP_PROXY": dead, "NO_PROXY": "127.0.0.1,localhost",
            "HERMES_NOUS_TIMEOUT_SECONDS": "5",
        }
        cfg = {"model": {"provider": "nous", "default": "Hermes-4-70B", "context_length": 128000},
               "fallback_providers": _fallback_entry(fallback)}
        h = Home(tmp_path).write(cfg, {"OPENAI_API_KEY": "sk-fake"}, auth=_expired_nous_auth())
        run = _bounded_turn(h, PROMPT, env=env)
        fallback_mains = fallback.main_requests()

    if portal_kind == "5xx":
        assert "/api/oauth/token" in portal_hits, f"precondition: the expired token was never refreshed: {run.describe()}"
    assert fallback_mains, f"fallback_providers never consulted: {run.describe()}"
    assert any(PROMPT in t for t in _user_texts(fallback_mains[0])), fallback_mains[0].get("messages")
    assert run.proc.returncode == 0 and run.stdout.strip() == "FROM-FALLBACK", run.describe()
    assert run.usage.get("model") == FALLBACK_MODEL, run.describe()
