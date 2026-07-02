"""End-to-end test for the zero-CLI Honcho OAuth flow against a fake AS.

Stands up a real local authorization server (no network, no browser) and drives
the full path: begin → /authorize 302 → loopback :8765 callback → token
exchange → install_grant → forced-expiry refresh with rotation. This is the
deterministic "real smoke test" for the consumer flow.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from plugins.memory.honcho import oauth, oauth_flow


class _FakeAS(BaseHTTPRequestHandler):
    """Minimal OAuth 2.1 AS: /authorize 302s to the callback; /oauth/token mints."""

    # Rotation counter shared across requests so refresh returns a new token.
    issued = {"n": 0}

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/authorize":
            self.send_response(404)
            self.end_headers()
            return
        q = parse_qs(parsed.query)
        redirect = q["redirect_uri"][0]
        # The redirect must be the IP literal matching the bound host — a
        # `localhost` redirect can resolve to ::1 and miss the IPv4 listener.
        # Host must be the IP literal (port may fall back off :8765).
        assert redirect.startswith("http://127.0.0.1:") and "/callback" in redirect, redirect
        # Consent shows a home-relative display path — never an absolute path
        # that would leak the username / home layout off the machine.
        cp = q["config_path"][0]
        assert cp.endswith("honcho.json"), q.get("config_path")
        assert not cp.startswith("/"), cp
        state = q["state"][0]
        location = f"{redirect}?code=test-auth-code&state={state}"
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def do_POST(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/oauth/token":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        form = parse_qs(self.rfile.read(length).decode())
        grant_type = form["grant_type"][0]
        self.issued["n"] += 1
        n = self.issued["n"]
        body = {
            "access_token": f"hch-at-{n}",
            "refresh_token": f"hch-rt-{n}",
            "token_type": "Bearer",
            "expires_in": 3600,
            "scope": "write",
        }
        if grant_type == "authorization_code":
            body["config"] = {
                "peerName": "lyra",
                "environment": "production",
                "hosts": {"hermes": {"saveMessages": True, "recallMode": "hybrid"}},
            }
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        return


@pytest.fixture
def fake_as(monkeypatch):
    _FakeAS.issued["n"] = 0
    server = HTTPServer(("127.0.0.1", 0), _FakeAS)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    monkeypatch.setenv("HONCHO_OAUTH_AUTHORIZE_URL", f"{base}/authorize")
    monkeypatch.setenv("HONCHO_OAUTH_TOKEN_URL", f"{base}/oauth/token")
    monkeypatch.setenv("HONCHO_OAUTH_CLIENT_ID", "hermes-desktop")
    try:
        yield base
    finally:
        server.shutdown()
        server.server_close()


def _browser_driver(authorize_url: str) -> None:
    """Stand in for the user's browser: follow /authorize's 302 into the callback.

    Retries the callback GET so it can't lose the race to the loopback bind.
    """
    resp = httpx.get(authorize_url, follow_redirects=False)
    location = resp.headers["Location"]
    for _ in range(50):
        try:
            httpx.get(location, timeout=2)
            return
        except httpx.ConnectError:
            time.sleep(0.05)
    raise RuntimeError("loopback callback never came up")


def test_full_loopback_flow_then_refresh(tmp_path, fake_as):
    config_path = tmp_path / "honcho.json"
    config_path.write_text(json.dumps({"hosts": {"obsidian": {"workspace": "obsidian"}}}))

    cred = oauth_flow.authorize_via_loopback(
        config_path=config_path,
        host="hermes",
        open_url=lambda url: _browser_driver(url),
        timeout=10,
    )

    # Grant installed: token stored, config deep-merged, other host preserved.
    assert cred.access_token == "hch-at-1"
    saved = json.loads(config_path.read_text())
    assert saved["hosts"]["hermes"]["apiKey"] == "hch-at-1"
    assert saved["hosts"]["hermes"]["oauth"]["refreshToken"] == "hch-rt-1"
    assert saved["hosts"]["hermes"]["recallMode"] == "hybrid"
    assert saved["environment"] == "production"
    assert saved["hosts"]["obsidian"] == {"workspace": "obsidian"}

    # Force expiry; ensure_fresh_token refreshes against the same AS and rotates.
    token, refreshed = oauth.ensure_fresh_token(
        config_path, "hermes", now=saved["hosts"]["hermes"]["oauth"]["expiresAt"] + 10
    )
    assert refreshed is True
    assert token == "hch-at-2"
    rotated = json.loads(config_path.read_text())["hosts"]["hermes"]["oauth"]
    assert rotated["refreshToken"] == "hch-rt-2"


def test_state_mismatch_is_rejected(fake_as, tmp_path):
    endpoints = oauth_flow.resolve_endpoints()
    _, state = oauth_flow.begin_authorization(endpoints)
    with pytest.raises(ValueError, match="unknown or expired"):
        oauth_flow.complete_authorization(
            endpoints, "code", "not-the-real-state",
            config_path=tmp_path / "honcho.json", host="hermes",
        )


def test_source_tags_the_authorize_link(fake_as):
    endpoints = oauth_flow.resolve_endpoints()
    url, _ = oauth_flow.begin_authorization(endpoints, source="hermes-cli")
    assert "source=hermes-cli" in url
    untagged, _ = oauth_flow.begin_authorization(endpoints)
    assert "source=" not in untagged


def test_client_id_defaults_to_hermes_agent(monkeypatch):
    # One client for every surface; the env var overrides for unusual deployments.
    monkeypatch.delenv("HONCHO_OAUTH_CLIENT_ID", raising=False)
    common = {"environment": "production", "base_url": "https://api.honcho.dev"}
    assert oauth_flow.resolve_endpoints(**common).client_id == "hermes-agent"
    monkeypatch.setenv("HONCHO_OAUTH_CLIENT_ID", "custom-id")
    assert oauth_flow.resolve_endpoints(**common).client_id == "custom-id"


def test_grant_persists_default_client_id(tmp_path, fake_as, monkeypatch):
    # Drop the fixture's override so the default takes effect; the grant must
    # store client_id=hermes-agent so refresh reuses the right client.
    monkeypatch.delenv("HONCHO_OAUTH_CLIENT_ID", raising=False)
    config_path = tmp_path / "honcho.json"
    config_path.write_text(json.dumps({"hosts": {}}))

    oauth_flow.authorize_via_loopback(
        config_path=config_path,
        host="hermes",
        source="hermes-cli",
        apply_config=False,
        open_url=lambda url: _browser_driver(url),
        timeout=10,
    )
    saved = json.loads(config_path.read_text())
    assert saved["hosts"]["hermes"]["oauth"]["clientId"] == "hermes-agent"


def test_config_path_rides_the_authorize_link(fake_as):
    endpoints = oauth_flow.resolve_endpoints()
    url, _ = oauth_flow.begin_authorization(endpoints, config_path="~/.hermes/honcho.json")
    q = parse_qs(urlparse(url).query)
    assert q["config_path"][0] == "~/.hermes/honcho.json"
    bare, _ = oauth_flow.begin_authorization(endpoints)
    assert "config_path=" not in bare


def test_display_config_path_never_leaks_absolute_path():
    from pathlib import Path

    # Under home → collapsed to ~/…; outside home → bare filename only.
    under_home = Path.home() / ".hermes" / "profiles" / "work" / "honcho.json"
    assert oauth_flow._display_config_path(under_home) == "~/.hermes/profiles/work/honcho.json"
    assert oauth_flow._display_config_path("/var/folders/tmp/honcho.json") == "honcho.json"


def test_cli_flow_stores_tokens_without_applying_config(tmp_path, fake_as):
    # apply_config=False (the CLI path): grant config must NOT touch settings.
    config_path = tmp_path / "honcho.json"
    config_path.write_text(json.dumps({"hosts": {"hermes": {"saveMessages": False}}}))

    cred = oauth_flow.authorize_via_loopback(
        config_path=config_path,
        host="hermes",
        source="hermes-cli",
        apply_config=False,
        open_url=lambda url: _browser_driver(url),
        timeout=10,
    )

    saved = json.loads(config_path.read_text())
    host = saved["hosts"]["hermes"]
    assert host["apiKey"] == cred.access_token
    assert host["oauth"]["refreshToken"] == cred.refresh_token
    # Wizard-owned setting untouched; grant config keys absent.
    assert host["saveMessages"] is False
    assert "recallMode" not in host
    assert "environment" not in saved
    # consent peer name still surfaced (seeds the CLI wizard prompt) despite no merge
    assert cred.consent_peer_name == "lyra"


# ── Desktop "Connect" button path: background launcher, status, dispatch ──


@pytest.fixture
def reset_flow():
    oauth_flow._status = oauth_flow.FlowStatus()
    oauth_flow._flow_thread = None
    yield
    oauth_flow._status = oauth_flow.FlowStatus()
    oauth_flow._flow_thread = None


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_launcher_runs_flow_in_background_and_reports_connected(monkeypatch, reset_flow):
    seen = {}
    gate = threading.Event()

    def fake(**kwargs):
        seen.update(kwargs)  # captures source default + eagerly-resolved path/host
        gate.wait(2)  # hold the flow open so the launcher returns while pending

    monkeypatch.setattr(oauth_flow, "authorize_via_loopback", fake)
    monkeypatch.setattr(oauth_flow, "_detect_connection", lambda: (True, "oauth"))

    st = oauth_flow.start_loopback_flow_background(config_path=Path("/t/honcho.json"), host="hermes")
    assert st["state"] == "pending"  # returns immediately, before the flow finishes
    assert _wait_until(lambda: seen.get("source") == "hermes-desktop")  # default source tag
    assert seen["host"] == "hermes"
    gate.set()
    assert _wait_until(lambda: oauth_flow.get_flow_status()["state"] == "connected")


def test_launcher_reports_error_on_flow_failure(monkeypatch, reset_flow):
    def boom(**kwargs):
        raise RuntimeError("loopback bind failed")

    monkeypatch.setattr(oauth_flow, "authorize_via_loopback", boom)
    monkeypatch.setattr(oauth_flow, "_detect_connection", lambda: (False, None))

    oauth_flow.start_loopback_flow_background(config_path=Path("/t/honcho.json"), host="hermes")
    assert _wait_until(lambda: oauth_flow.get_flow_status()["state"] == "error")
    assert "loopback bind failed" in oauth_flow.get_flow_status()["detail"]


def test_launcher_is_idempotent_while_pending(monkeypatch, reset_flow):
    block = threading.Event()
    calls = []

    def fake(**kwargs):
        calls.append(1)
        block.wait(2)

    monkeypatch.setattr(oauth_flow, "authorize_via_loopback", fake)
    monkeypatch.setattr(oauth_flow, "_detect_connection", lambda: (False, None))

    s1 = oauth_flow.start_loopback_flow_background(config_path=Path("/t/h.json"), host="hermes")
    assert _wait_until(lambda: len(calls) == 1)  # first flow is running
    s2 = oauth_flow.start_loopback_flow_background(config_path=Path("/t/h.json"), host="hermes")
    block.set()
    assert s1["state"] == "pending" and s2["state"] == "pending"
    assert _wait_until(lambda: oauth_flow.get_flow_status()["state"] == "connected")
    assert calls == [1]  # the second call did not spawn a second flow


def test_get_flow_status_reports_stored_connection(tmp_path, monkeypatch, reset_flow):
    from plugins.memory.honcho import client as honcho_client

    cfgfile = tmp_path / "honcho.json"
    monkeypatch.setattr(honcho_client, "resolve_config_path", lambda: cfgfile)
    monkeypatch.setattr(honcho_client, "resolve_active_host", lambda: "hermes")
    monkeypatch.delenv("HONCHO_API_KEY", raising=False)

    cfgfile.write_text(json.dumps({"hosts": {"hermes": {}}}))
    assert oauth_flow.get_flow_status()["connected"] is False

    cfgfile.write_text(json.dumps({"hosts": {"hermes": {"apiKey": "hch-v3-static"}}}))
    s = oauth_flow.get_flow_status()
    assert s["connected"] is True and s["auth"] == "apikey"

    cfgfile.write_text(json.dumps({"hosts": {"hermes": {
        "apiKey": "hch-at-tok",
        "oauth": {"refreshToken": "hch-rt-x", "expiresAt": 9_999_999_999,
                  "clientId": "hermes-desktop", "tokenEndpoint": "http://x/oauth/token"},
    }}}))
    s = oauth_flow.get_flow_status()
    assert s["connected"] is True and s["auth"] == "oauth"


def test_memory_oauth_router_dispatches_by_provider_convention():
    # The generic seam behind the two routes: provider → plugins.memory.<p>.oauth_flow.
    from fastapi import HTTPException

    from hermes_cli.memory_oauth import _resolve_flow

    mod = _resolve_flow("honcho")
    assert hasattr(mod, "start_loopback_flow_background") and hasattr(mod, "get_flow_status")

    for bad in ("builtin", "no-such-provider", "../etc"):
        with pytest.raises(HTTPException) as exc:
            _resolve_flow(bad)
        assert exc.value.status_code == 404
