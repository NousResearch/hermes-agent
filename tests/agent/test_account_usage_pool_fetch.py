"""Real-path source parsing and routing for per-credential usage fetches.

Loopback HTTP + actual auth stores: the Codex read-only fetcher hits the entry's own route,
parses real payloads, marks Opus/Sonnet windows model-scoped, and a 401 raises (never a
rotation/refresh) — while the live fetcher's repair path stays intact.
"""

from __future__ import annotations

import base64
import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from agent import account_usage
from agent.account_usage import fetch_account_usage


def _b64(part: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(part).encode()).decode().rstrip("=")


def _codex_jwt(account_id: str, sub: str) -> str:
    return (f"{_b64({'alg': 'none'})}."
            f"{_b64({'sub': sub, 'https://api.openai.com/auth': {'chatgpt_account_id': account_id}})}.sig")


class _LoopbackUsageServer:
    """Loopback Codex usage API: one handler per token, keyed by Authorization bearer."""

    def __init__(self):
        self.requests: list[dict] = []
        self.responses: dict[str, tuple[int, dict]] = {}
        self.lock = threading.Lock()
        handler = self

        class _Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                with handler.lock:
                    handler.requests.append({"path": self.path, "token": self.headers.get("Authorization", "")})
                token = (self.headers.get("Authorization") or "").removeprefix("Bearer ").strip()
                status, payload = handler.responses.get(token, (200, {}))
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/backend-api/codex"

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        return False


def _usage_payload(session_used: float = 20.0, weekly_used: float = 5.0) -> dict:
    now = datetime.now(timezone.utc)
    return {
        "plan_type": "pro",
        "rate_limit": {
            "primary_window": {"used_percent": session_used, "reset_at": int((now + timedelta(hours=1)).timestamp()),
                               "limit_window_seconds": 18000},
            "secondary_window": {"used_percent": weekly_used, "reset_at": int((now + timedelta(days=3)).timestamp()),
                                 "limit_window_seconds": 604800},
        },
        "credits": {"has_credits": False},
    }


def test_read_only_fetch_routes_to_entries_own_host_and_parses(monkeypatch, tmp_path):
    """The per-credential read-only fetch hits the entry's own route host with its own token and
    parses the payload: duration-labeled windows, identity from the JWT, no refresh attempted."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from hermes_cli import auth as auth_mod

    rotations: list = []
    monkeypatch.setattr(auth_mod, "resolve_codex_runtime_credentials",
                        lambda **kw: rotations.append(kw) or (_ for _ in ()).throw(AssertionError(
                            "read-only per-credential fetch must never consult the singleton resolver")))
    token = _codex_jwt("acct-9", "user-9")
    with _LoopbackUsageServer() as server:
        server.responses[token] = (200, _usage_payload(session_used=44.0))
        snapshot = fetch_account_usage(
            "openai-codex", base_url=server.url, api_key=token, read_only=True, identity_id="codex:acct-9:user-9")

    assert snapshot is not None
    assert [(w.label, w.used_percent, w.scope) for w in snapshot.windows] == [
        ("Session", 44.0, "account"), ("Weekly", 5.0, "account")]
    assert snapshot.identity == "codex:acct-9:user-9"
    assert server.requests[0]["path"].endswith("/wham/usage"), \
        "a /backend-api base routes to the ChatGPT wham usage path"
    assert server.requests[0]["token"] == f"Bearer {token}"


def test_read_only_401_raises_and_never_retries_with_rotation(monkeypatch, tmp_path):
    """A 401 on the read-only path is reported (snapshot None → unknown), never repaired via a
    forced refresh that could rotate a pooled credential; the live (non-read-only) fetch keeps its
    retry path for the session's own credential."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from hermes_cli import auth as auth_mod

    def _forbidden_refresh(**kw):
        raise AssertionError("read-only path must not force-refresh (would rotate a pool credential)")

    token = _codex_jwt("acct-1", "user-1")
    with _LoopbackUsageServer() as server:
        server.responses[token] = (401, {"error": "unauthorized"})
        monkeypatch.setattr(account_usage, "_resolve_codex_usage_credentials",
                            lambda base_url, api_key, **kw: (api_key, base_url, None))
        monkeypatch.setattr(auth_mod, "resolve_codex_runtime_credentials", _forbidden_refresh)
        # The fetcher under test imports _resolve_codex_usage_credentials at module level, so the
        # 401 path must NOT re-enter it with force_refresh: patch it to blow up on force_refresh.
        original = account_usage._resolve_codex_usage_credentials

        def no_force_refresh(base_url, api_key, *, force_refresh=False):
            if force_refresh:
                raise AssertionError("read-only path must not force-refresh")
            return original(base_url, api_key)

        monkeypatch.setattr(account_usage, "_resolve_codex_usage_credentials", no_force_refresh)
        snapshot = fetch_account_usage(
            "openai-codex", base_url=server.url, api_key=token, read_only=True)

    assert snapshot is None, "401 → no snapshot (account renders unknown), no repair attempted"
    assert len(server.requests) == 1, "exactly one request — no retry on the read-only path"


def test_anthropic_fetch_honors_explicit_api_key_and_marks_model_scoped_windows(monkeypatch):
    """An explicit Anthropic api_key must not be shadowed by the ambient OAuth singleton, and the
    Opus/Sonnet weekly windows parse as ``model`` scope (they can never imply the account is out)."""
    called: dict = {}

    def fake_get_json(url, headers, *, timeout):
        called["url"], called["headers"] = url, headers
        return {"five_hour": {"utilization": 0.3, "resets_at": "2026-10-05T12:00:00Z"},
                "seven_day": {"utilization": 0.2, "resets_at": "2026-10-08T12:00:00Z"},
                "seven_day_opus": {"utilization": 1.0, "resets_at": "2026-10-09T12:00:00Z"},
                "seven_day_sonnet": {"utilization": 0.4, "resets_at": "2026-10-09T12:00:00Z"}}

    monkeypatch.setattr(account_usage, "_get_json", fake_get_json)
    monkeypatch.setattr(account_usage, "resolve_anthropic_token",
                        lambda **kw: (_ for _ in ()).throw(AssertionError("ambient token must not be consulted")))

    snapshot = fetch_account_usage("anthropic", api_key="sk-ant-oat-explicit-token")

    assert snapshot is not None
    assert called["headers"]["Authorization"] == "Bearer sk-ant-oat-explicit-token"
    scopes = {w.label: w.scope for w in snapshot.windows}
    assert scopes == {"Current session": "account", "Current week": "account",
                      "Opus week": "model", "Sonnet week": "model"}
    percents = {w.label: w.used_percent for w in snapshot.windows}
    assert percents["Opus week"] == 100.0 and percents["Current session"] == 30.0


def test_read_only_fetch_remembered_under_identity_slot(monkeypatch, tmp_path):
    """A per-credential fetch's result lands in THAT account's cache slot (readable by
    ``cached_account_usage(identity_id=...)``) and never in the provider-wide legacy slot."""
    from agent.account_usage_cache import _snapshots, cached_account_usage

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    _snapshots.clear()
    token = _codex_jwt("acct-7", "user-7")
    with _LoopbackUsageServer() as server:
        server.responses[token] = (200, _usage_payload(session_used=66.0))
        fetch_account_usage("openai-codex", base_url=server.url, api_key=token,
                            read_only=True, identity_id="codex:acct-7:user-7")

    snapshot = cached_account_usage("openai-codex", identity_id="codex:acct-7:user-7")
    assert snapshot is not None and snapshot.windows[0].used_percent == 66.0
    assert cached_account_usage("openai-codex") is None, \
        "a per-credential snapshot must never pose as the provider-wide gauge"
