"""Dashboard auth: every mounted ``/api/*`` route and every WebSocket route refuses a caller without
the session token (issue class: unauthenticated dashboard access, GHSA-ppp5-vxwm-4cf7 family).

A relationship test, not a list: the route inventory is read at runtime from the SAME app the live
server runs (``app.routes`` in a sandboxed interpreter, unioned with the live ``/openapi.json``), so a
route added tomorrow is covered the day it lands. The only exemptions are the ones the server itself
declares public (``PUBLIC_API_PATHS`` and the MCP OAuth redirect target, which a browser reaches from
the identity provider without our header).

Controls keep each negative honest: the scraped token (read out of ``index.html`` like the SPA does)
opens every GET route and the chat WebSockets, and a cross-origin page holding the token is still
refused (CSWSH / DNS rebinding).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from websockets.exceptions import ConnectionClosed, InvalidStatus
from websockets.sync.client import connect

from . import _helpers as H

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="POSIX PTY dashboard")

_INVENTORY = r"""
import json
from fastapi.routing import APIRoute, APIWebSocketRoute
from hermes_cli.web_server import app
from hermes_cli.dashboard_auth.public_paths import PUBLIC_API_PATHS
http = sorted({(m, r.path) for r in app.routes if isinstance(r, APIRoute) for m in r.methods if m != "HEAD"})
ws = sorted({r.path for r in app.routes if isinstance(r, APIWebSocketRoute)})
print(json.dumps({"http": http, "ws": ws, "public": sorted(PUBLIC_API_PATHS)}))
"""
# The OAuth redirect lands from the identity provider's page: no header can ride on it by design.
_BROWSER_REDIRECT_PREFIXES = ("/api/mcp/oauth/callback/",)
_PARAM = re.compile(r"\{([^}:]+)(?::[^}]+)?\}")
_CHAT_WS = ("/api/ws", "/api/events", "/api/console")  # accept a token-bearing loopback client


def _concrete(path: str) -> str:
    return _PARAM.sub(lambda m: f"e2e-{m.group(1)}", path)


@pytest.fixture(scope="module")
def dash(tmp_path_factory: pytest.TempPathFactory):
    sb = H.make_sandbox(tmp_path_factory.mktemp("dash-auth"))
    d = H.Dashboard(sb, sb.root / "dashboard.log")
    try:
        yield d
    finally:
        d.close()
        sb.finish()


@pytest.fixture(scope="module")
def inventory(dash: H.Dashboard) -> dict:
    r = H.run_py(dash.sb, _INVENTORY)
    assert r.returncode == 0, r.stderr[-3000:]
    inv = json.loads(r.stdout.strip().splitlines()[-1])
    live = dash.http.get("/openapi.json").json().get("paths", {})
    http = {tuple(x) for x in inv["http"]}
    http |= {(m.upper(), p) for p, ops in live.items() for m in ops if m.upper() in {"GET", "POST", "PUT", "PATCH", "DELETE"}}
    inv["http"] = sorted(http)
    return inv


def _gated_http(inv: dict) -> list[tuple[str, str]]:
    public = set(inv["public"])
    return [(m, p) for m, p in inv["http"]
            if p.startswith("/api/") and p not in public and not p.startswith(_BROWSER_REDIRECT_PREFIXES)]


def test_every_api_route_refuses_missing_and_wrong_token(dash: H.Dashboard, inventory: dict) -> None:
    gated = _gated_http(inventory)
    assert len(gated) > 150, f"route inventory implausibly small ({len(gated)}): enumeration broke"
    leaks: list[str] = []
    for method, path in gated:
        url = _concrete(path)
        for label, headers in (("no token", {}), ("wrong token", {H.TOKEN_HEADER: "not-the-token"}),
                               ("wrong bearer", {"Authorization": "Bearer not-the-token"})):
            r = dash.http.request(method, url, headers=headers, json={} if method != "GET" else None)
            if r.status_code != 401:
                leaks.append(f"{method} {path} ({label}) -> {r.status_code} {r.text[:120]!r}")
    assert not leaks, f"{len(leaks)} gated route(s) served a caller without the session token:\n  " + "\n  ".join(leaks[:40])
    assert dash.proc.poll() is None, "dashboard died during the unauthenticated sweep"


def test_scraped_token_opens_every_get_route(dash: H.Dashboard, inventory: dict) -> None:
    """Control: the 401s above are auth, not a dead router — the SPA's token passes every GET gate."""
    refused = []
    for method, path in _gated_http(inventory):
        # /api/auth/* returns the verified cookie-session identity; the loopback token carries none.
        if method != "GET" or path.startswith("/api/auth/"):
            continue
        r = dash.request("GET", _concrete(path))
        if r.status_code == 401:
            refused.append(f"GET {path}")
    assert not refused, f"routes refuse the dashboard's own token: {refused}"


def _ws_outcome(url: str, origin: str | None = None) -> str:
    """``accepted`` if the upgrade completes and the socket is not immediately closed with a 44xx
    policy code, else a refusal description."""
    headers = {"Origin": origin} if origin else None
    try:
        with connect(url, open_timeout=30, additional_headers=headers, max_size=None) as ws:
            try:
                ws.recv(timeout=1.5)
            except TimeoutError:
                return "accepted"
            except ConnectionClosed as exc:
                code = exc.rcvd.code if exc.rcvd else None
                return f"closed {code}" if code and 4400 <= code < 4500 else "accepted"
            return "accepted"
    except InvalidStatus as exc:
        return f"http {exc.response.status_code}"
    except ConnectionClosed as exc:
        return f"closed {exc.rcvd.code if exc.rcvd else '?'}"


def test_every_websocket_refuses_missing_wrong_and_cross_origin_credentials(dash: H.Dashboard, inventory: dict) -> None:
    ws_paths = inventory["ws"]
    assert set(_CHAT_WS) <= set(ws_paths) and "/api/pty" in ws_paths, ws_paths
    q = {"channel": "e2e-auth", "profile": None}
    leaks = []
    for path in ws_paths:
        for label, token in (("no token", None), ("wrong token", "not-the-token")):
            outcome = _ws_outcome(dash.ws_url(path, token=token, **q))
            if outcome == "accepted":
                leaks.append(f"{path} ({label}) accepted")
    for path in ("/api/pty", "/api/ws", "/api/console"):
        outcome = _ws_outcome(dash.ws_url(path, token=dash.token, **q), origin="http://evil.example")
        if outcome == "accepted":
            leaks.append(f"{path} (valid token, cross-origin page) accepted")
    assert not leaks, "websocket upgrade(s) accepted without a valid same-origin credential:\n  " + "\n  ".join(leaks)
    # Control: the same URLs with the scraped token open.
    for path in _CHAT_WS:
        assert _ws_outcome(dash.ws_url(path, token=dash.token, **q)) == "accepted", f"{path} refused the real token"
    log = Path(dash.log_path).read_text(encoding="utf-8", errors="replace")
    assert dash.token not in log, "the session token was written to the dashboard log"
