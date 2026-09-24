"""Codex credential pool: a dead refresh grant retires only the rows tied to that grant.

Tier (stated honestly): REAL ``hermes -z`` processes with ``model.provider: openai-codex``
and a seeded two-login pool in ``auth.json``. Two vendor boundaries are faked:

* the ChatGPT Codex Responses backend — ``HERMES_CODEX_BASE_URL`` points at
  ``FakeResponsesServer``, which accepts ONLY login A's bearer, so a reply proves which login
  served the turn;
* the OpenAI OAuth token endpoint. Its URL is a hard-coded ``https://`` constant with no
  override, and the proxy-impersonation route cannot carry https, so a ``sitecustomize`` shim
  on the child's ``PYTHONPATH`` redirects exactly that URL, at the ``httpx.Client.send``
  layer, to a loopback token fake. No Hermes function is patched; everything from the pool
  through the refresh POST, the terminal classification and the auth.json write-back runs as
  shipped.

The pool is the #120741 layout: login A is the singleton plus its seeded ``device_code`` row
(fresh), login B is an independently authorized ``manual:device_code`` row (``hermes auth
add``) whose access token is about to expire, so every selection defers a refresh of B.
"""

from __future__ import annotations

import base64
import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import pytest

from tests.e2e.core.providers._openai_helpers import (
    REPO_ROOT,
    Home,
    bug_assertions,
    oneshot,
    write_sitecustomize_shim,
)
from tests.fakes.providers.openai_responses import FakeResponsesServer, Message, Turn

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="subprocess harness is Linux-gated")

KNOWN: dict[str, tuple[str, str]] = {
    "independent_login_survives": (
        r"login A was quarantined by B's invalid_grant: live pool is ",
        "#120741 terminal refresh failure on one pooled login quarantines an independent login"),
}


TOKEN_URL = "https://auth.openai.com/oauth/token"
MODEL = "gpt-5.3-codex"

_SHIM = '''\
"""E2E vendor-boundary shim: route the hard-coded OAuth token URL to a loopback fake."""
import os

_pairs = [p.split("=", 1) for p in os.environ.get("HERMES_E2E_URL_REDIRECT", "").split(";") if "=" in p]
if _pairs:
    import httpx

    _redirect = dict(_pairs)
    _send = httpx.Client.send

    def send(self, request, *args, **kwargs):
        target = _redirect.get(str(request.url))
        if target:
            request.url = httpx.URL(target)
            request.headers["Host"] = request.url.netloc.decode()
        return _send(self, request, *args, **kwargs)

    httpx.Client.send = send
'''


class FakeTokenServer:
    """OAuth token endpoint: ``invalid_grant`` for the configured refresh tokens, else a rotation."""

    def __init__(self, rejected: set[str], minted: dict[str, tuple[str, str]] | None = None,
                 throttled: set[str] | None = None) -> None:
        self.rejected = rejected
        self.throttled = throttled or set()
        self.minted = minted or {}
        self.posts: list[dict[str, str]] = []
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> "FakeTokenServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/oauth/token"

    def refresh_tokens_posted(self) -> list[str]:
        return [p.get("refresh_token", "") for p in self.posts]

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a: object) -> None:  # noqa: D401 - silence per-request logging
                pass

            def do_POST(self) -> None:  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0)).decode()
                form = {k: v[0] for k, v in parse_qs(raw).items()}
                server.posts.append(form)
                rt = form.get("refresh_token", "")
                extra = {}
                if rt in server.throttled:
                    status, body, extra = 429, {"error": "rate_limited"}, {"Retry-After": "600"}
                elif form.get("grant_type") != "refresh_token" or rt in server.rejected or rt not in server.minted:
                    status, body = 400, {"error": "invalid_grant",
                                         "error_description": "The refresh token is invalid or has been revoked."}
                else:
                    at, next_rt = server.minted[rt]
                    status, body = 200, {"access_token": at, "refresh_token": next_rt, "id_token": at,
                                         "token_type": "Bearer", "expires_in": 28800}
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                for k, v in extra.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        return Handler


def _jwt(account: str, sub: str, exp: float) -> str:
    def seg(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    payload = {"exp": int(exp), "sub": sub, "https://api.openai.com/auth": {"chatgpt_account_id": account}}
    return f"{seg({'alg': 'none'})}.{seg(payload)}.sig"


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _pool_store(a: dict[str, str], b: dict[str, str], now: float) -> dict[str, Any]:
    a_refresh = _iso(now - 3600)
    return {
        "version": 1,
        "active_provider": "openai-codex",
        "providers": {"openai-codex": {"tokens": dict(a), "last_refresh": a_refresh, "auth_mode": "chatgpt"}},
        "credential_pool": {"openai-codex": [
            {"id": "login-a", "label": "device_code", "auth_type": "oauth", "priority": 0,
             "source": "device_code", **a, "last_refresh": a_refresh},
            {"id": "login-b", "label": "second-account", "auth_type": "oauth", "priority": 1,
             "source": "manual:device_code", **b, "last_refresh": _iso(now - 7200)},
        ]},
    }


def _home(tmp_path: Path, store: dict[str, Any]) -> Home:
    h = Home(tmp_path).write({"model": {"provider": "openai-codex", "default": MODEL, "context_length": 128000}},
                             auth=store)
    write_sitecustomize_shim(tmp_path / "shim", _SHIM)
    return h


def _env(tmp_path: Path, codex_url: str, token: FakeTokenServer, dead_port: int) -> dict[str, str]:
    """Child env: the shim, the Codex fake, and every other egress pinned to a refusing proxy."""
    proxy = f"http://127.0.0.1:{dead_port}"
    return {"HERMES_CODEX_BASE_URL": codex_url, "PYTHONPATH": f"{tmp_path / 'shim'}:{REPO_ROOT}",
            "HERMES_E2E_URL_REDIRECT": f"{TOKEN_URL}={token.url}",
            **{k: proxy for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy")},
            "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}


def _auth(h: Home) -> dict[str, Any]:
    return json.loads((h.hermes_home / "auth.json").read_text(encoding="utf-8"))


def _answer(run) -> str:
    return run.stdout.strip()


def _pair(account: str, rt: str, ttl: float) -> dict[str, str]:
    return {"access_token": _jwt(account, f"user-{account}", time.time() + ttl), "refresh_token": rt}


def _two_logins(tmp_path: Path, a: dict[str, str], b: dict[str, str], token: FakeTokenServer,
                replies: list[str], bearer: str) -> tuple[list, list[dict[str, Any]], list[dict[str, Any]]]:
    """Two sequential real ``hermes -z`` turns against the seeded pool; auth.json after each."""
    runs, stores = [], []
    with FakeResponsesServer([Turn([Message(r)]) for r in replies], api_key=bearer) as codex, \
            socket.socket() as dead:
        dead.bind(("127.0.0.1", 0))  # bound, never listening: a proxy that refuses at once
        h = _home(tmp_path, _pool_store(a, b, time.time()))
        env = _env(tmp_path, codex.base_url, token, dead.getsockname()[1])
        for n in range(len(replies)):
            runs.append(oneshot(h, f"hello {n}", env=env))
            stores.append(_auth(h))
    return runs, stores, [s.get("credential_pool", {}).get("openai-codex", []) for s in stores]


def _live_order(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """``(id, source)`` of rows still in rotation, in fill_first (priority) order."""
    live = [r for r in rows if r.get("last_status") != "dead"]
    return [(r["id"], r["source"]) for r in sorted(live, key=lambda r: r.get("priority", 0))]


def test_dead_grant_on_independent_login_keeps_login_a(tmp_path) -> None:
    """B's refresh token is terminally rejected (``invalid_grant``). B leaves rotation; A — a
    different grant, a different account — keeps its pool row (id, source, first place in the
    fill_first order), its singleton tokens, and keeps serving turns in later processes."""
    a, b = _pair("acct-A", "rt-A", 8 * 3600), _pair("acct-B", "rt-B", 60)
    with FakeTokenServer(rejected={"rt-B"}) as token:
        runs, stores, pools = _two_logins(tmp_path, a, b, token, ["SERVED-BY-A-1", "SERVED-BY-A-2"],
                                          a["access_token"])
        posted = token.refresh_tokens_posted()

    for run, want in zip(runs, ["SERVED-BY-A-1", "SERVED-BY-A-2"]):
        assert run.proc.returncode == 0 and _answer(run) == want, run.describe()
    # The terminal failure happened on the wire, for B's grant only, and B is retired.
    assert posted and set(posted) == {"rt-B"}, posted
    b_row = next((r for r in pools[0] if r["id"] == "login-b"), None)
    assert b_row is None or b_row.get("last_status") == "dead", b_row
    # A is a different grant: nothing about it may change.
    with bug_assertions(KNOWN, "independent_login_survives"):
        assert stores[0]["providers"]["openai-codex"]["tokens"] == a, stores[0]["providers"]["openai-codex"]
        for rows in pools:
            assert _live_order(rows) == [("login-a", "device_code")], (
                f"login A was quarantined by B's invalid_grant: live pool is {_live_order(rows)}")


def test_throttled_refresh_on_independent_login_leaves_login_a_untouched(tmp_path) -> None:
    """Control (green on main): a non-terminal token-endpoint failure for B (HTTP 429 quota)
    benches B only. A keeps its row, first place, singleton tokens, and serves every turn;
    no re-login is demanded."""
    a, b = _pair("acct-A", "rt-A", 8 * 3600), _pair("acct-B", "rt-B", 60)
    with FakeTokenServer(rejected=set(), throttled={"rt-B"}) as token:
        runs, stores, pools = _two_logins(tmp_path, a, b, token, ["SERVED-BY-A-1", "SERVED-BY-A-2"],
                                          a["access_token"])
        posted = token.refresh_tokens_posted()

    for run, want in zip(runs, ["SERVED-BY-A-1", "SERVED-BY-A-2"]):
        assert run.proc.returncode == 0 and _answer(run) == want, run.describe()
    assert posted and set(posted) == {"rt-B"}, posted
    singleton = stores[-1]["providers"]["openai-codex"]
    assert singleton["tokens"] == a and "last_auth_error" not in singleton, singleton
    assert _live_order(pools[-1])[0] == ("login-a", "device_code"), _live_order(pools[-1])
    rows = {r["id"]: r for r in pools[-1]}
    assert rows["login-b"].get("last_status") == "exhausted", rows["login-b"]


def test_dead_shared_grant_retires_every_row_of_it(tmp_path) -> None:
    """Control (green on main): when the singleton, its seeded row and a ``manual:`` alias all
    hold the SAME rejected refresh token, none stays selectable. The user gets the re-login
    instruction, no answer is fabricated, and auth.json records the dead grant (tokens gone,
    ``relogin_required``) so the next process does not re-seed it."""
    shared = _pair("acct-A", "rt-S", -60)
    with FakeTokenServer(rejected={"rt-S"}) as token:
        runs, stores, pools = _two_logins(tmp_path, shared, dict(shared), token, ["NEVER-1", "NEVER-2"],
                                          "no-login-holds-this-bearer")
        posted = token.refresh_tokens_posted()

    assert posted and set(posted) == {"rt-S"}, posted
    for run, store, rows in zip(runs, stores, pools):
        assert run.proc.returncode != 0 and "NEVER" not in run.stdout, run.describe()
        assert "hermes auth add openai-codex" in run.stdout, run.describe()
        singleton = store["providers"]["openai-codex"]
        assert not singleton.get("tokens", {}).get("refresh_token"), singleton
        assert (singleton.get("last_auth_error") or {}).get("relogin_required") is True, singleton
        assert _live_order(rows) == [], rows
