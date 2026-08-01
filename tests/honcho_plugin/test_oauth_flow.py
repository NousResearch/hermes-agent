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

# Opt-in: эти тесты поднимают собственный loopback OAuth-сервер внутри теста
# и коннектятся к нему же - внешней сети нет (guard B8 это разрешает явно).
pytestmark = pytest.mark.allow_network


class _FakeAS(BaseHTTPRequestHandler):
    """Minimal OAuth 2.1 AS: /authorize 302s to the callback; /oauth/token mints."""

    # Rotation counter shared across requests so refresh returns a new token.
    issued = {"n": 0}
    # Scripted outcomes for device-grant token polls; "ok" mints, anything else
    # is returned as a 400 OAuth error code.
    device_responses: list[str] = []
    # Last form posted to /oauth/device_authorization, for assertions.
    last_device_form: dict = {}
    # Whether AS metadata advertises the device grant.
    advertise_device = True

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/.well-known/oauth-authorization-server":
            if not _FakeAS.advertise_device:
                self.send_response(404)
                self.end_headers()
                return
            self._send_json(200, {
                "grant_types_supported": [
                    "authorization_code", "refresh_token", oauth_flow.DEVICE_GRANT_TYPE,
                ],
            })
            return
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
        length = int(self.headers.get("Content-Length", 0))
        form = parse_qs(self.rfile.read(length).decode())
        if parsed.path == "/oauth/device_authorization":
            _FakeAS.last_device_form = {k: v[0] for k, v in form.items()}
            base = f"http://{self.server.server_address[0]}:{self.server.server_address[1]}"
            self._send_json(200, {
                "device_code": "dev-code-1",
                "user_code": "ABCD-EFGH",
                "verification_uri": f"{base}/device",
                "verification_uri_complete": f"{base}/device?user_code=ABCD-EFGH",
                "expires_in": 600,
                "interval": 0,
            })
            return
        if parsed.path != "/oauth/token":
            self.send_response(404)
            self.end_headers()
            return
        grant_type = form["grant_type"][0]
        if grant_type == oauth_flow.DEVICE_GRANT_TYPE:
            outcome = _FakeAS.device_responses.pop(0) if _FakeAS.device_responses else "ok"
            if outcome != "ok":
                self._send_json(400, {"error": outcome, "error_description": f"scripted {outcome}"})
                return
            self.issued["n"] += 1
            n = self.issued["n"]
            self._send_json(200, {
                "access_token": f"hch-at-{n}",
                "refresh_token": f"hch-rt-{n}",
                "token_type": "Bearer",
                "expires_in": 3600,
                "scope": "write",
                "config": {"peerName": "lyra"},
            })
            return
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
    _FakeAS.device_responses = []
    _FakeAS.last_device_form = {}
    _FakeAS.advertise_device = True
    server = HTTPServer(("127.0.0.1", 0), _FakeAS)
    port = server.server_address[1]
    thread = threading.Thread(
        # poll_interval=0.05: shutdown() waits for serve_forever's next poll,
        # so the default 0.5s added half a second of teardown per test.
        target=lambda: server.serve_forever(poll_interval=0.05),
        daemon=True,
    )
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


def test_display_config_path_uses_forward_slashes_on_windows():
    """The ~/-relative form must use forward slashes even when the underlying
    Path type separates with backslashes.

    Platform-independent by construction: _display_config_path does a
    call-time ``from pathlib import Path``, so patching ``pathlib.Path`` with
    a PureWindowsPath-based shim exercises real Windows path semantics on any
    host. The old ``"~/" + str(p.relative_to(home))`` implementation returns
    a backslash-separated ``~/AppData/Local/...`` (with backslashes) here —
    mixed separators on the consent screen — so unlike the native-Path test
    above, this one fails on the pre-fix code on POSIX CI too.

    The patch is scoped and released BEFORE the asserts: pytest's own failure
    reporting uses pathlib.Path, so asserting inside the patch would turn a
    clean failure into an INTERNALERROR.
    """
    import pathlib
    from unittest.mock import patch as _patch

    class _WinPath(pathlib.PureWindowsPath):
        @classmethod
        def home(cls):
            return cls("C:/Users/test")

    with _patch.object(pathlib, "Path", _WinPath):
        shown = oauth_flow._display_config_path(
            "C:\\Users\\test\\AppData\\Local\\hermes\\honcho.json"
        )
        outside = oauth_flow._display_config_path("D:\\scratch\\honcho.json")

    assert shown == "~/AppData/Local/hermes/honcho.json"
    assert "\\" not in shown
    # Outside home still degrades to the bare filename.
    assert outside == "honcho.json"
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


# ── Device authorization grant (RFC 8628): headless / remote-VM path ──


class _FakeClock:
    """Injectable sleep + monotonic pair so poll loops run instantly."""

    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds

    def monotonic(self) -> float:
        return self.t


def test_device_endpoint_derived_from_token_url(monkeypatch):
    monkeypatch.delenv("HONCHO_OAUTH_DEVICE_AUTH_URL", raising=False)
    monkeypatch.delenv("HONCHO_OAUTH_TOKEN_URL", raising=False)
    cloud = oauth_flow.resolve_endpoints(environment="production", base_url="https://api.honcho.dev")
    assert cloud.device_authorization_url == "https://api.honcho.dev/oauth/device_authorization"
    local = oauth_flow.resolve_endpoints(environment="local", base_url=None)
    assert local.device_authorization_url == "http://localhost:8000/oauth/device_authorization"


def test_supports_device_login_from_metadata(fake_as):
    endpoints = oauth_flow.resolve_endpoints()
    assert oauth_flow.supports_device_login(endpoints) is True
    _FakeAS.advertise_device = False
    assert oauth_flow.supports_device_login(endpoints) is False
    # Fail closed on an unreachable host.
    dead = oauth_flow.OAuthEndpoints(
        authorize_url="http://127.0.0.1:1/authorize",
        token_url="http://127.0.0.1:1/oauth/token",
        client_id="hermes-agent",
        scope="write",
    )
    assert oauth_flow.supports_device_login(dead, timeout=0.2) is False


def test_request_device_code_parses_response_and_sends_identity(fake_as):
    endpoints = oauth_flow.resolve_endpoints()
    device = oauth_flow.request_device_code(endpoints, source="hermes-cli")
    assert device.device_code == "dev-code-1"
    assert device.user_code == "ABCD-EFGH"
    assert device.verification_uri.endswith("/device")
    assert device.verification_uri_complete.endswith("?user_code=ABCD-EFGH")
    assert (device.expires_in, device.interval) == (600, 0)
    assert _FakeAS.last_device_form["client_id"] == "hermes-desktop"
    assert _FakeAS.last_device_form["scope"] == "write"
    assert _FakeAS.last_device_form["source"] == "hermes-cli"


def test_poll_backs_off_on_slow_down(fake_as):
    _FakeAS.device_responses = ["slow_down", "slow_down", "ok"]
    endpoints = oauth_flow.resolve_endpoints()
    device = oauth_flow.DeviceCode(
        device_code="dev-code-1", user_code="X", verification_uri="u",
        verification_uri_complete="u?c", expires_in=600, interval=5,
    )
    clock = _FakeClock()
    grant = oauth_flow.poll_for_token(endpoints, device, sleep=clock.sleep, monotonic=clock.monotonic)
    assert grant["access_token"] == "hch-at-1"
    assert clock.sleeps == [5, 10, 15]


def test_slow_down_interval_caps_at_60(fake_as):
    _FakeAS.device_responses = ["slow_down", "ok"]
    endpoints = oauth_flow.resolve_endpoints()
    device = oauth_flow.DeviceCode(
        device_code="dev-code-1", user_code="X", verification_uri="u",
        verification_uri_complete="u?c", expires_in=600, interval=58,
    )
    clock = _FakeClock()
    oauth_flow.poll_for_token(endpoints, device, sleep=clock.sleep, monotonic=clock.monotonic)
    assert clock.sleeps == [58, 60]  # 58 + 5 clamps to the 60s cap


def test_callback_page_shows_error_on_denied_consent():
    server, captured = oauth_flow._bind_loopback_server()
    port = server.server_address[1]
    result: dict = {}

    def _deny():
        for _ in range(50):
            try:
                result["resp"] = httpx.get(
                    f"http://127.0.0.1:{port}/callback"
                    "?error=access_denied&error_description=user+denied&state=x",
                    timeout=2,
                )
                return
            except httpx.ConnectError:
                time.sleep(0.05)

    thread = threading.Thread(target=_deny, daemon=True)
    thread.start()
    with pytest.raises(ValueError, match="access_denied.*user denied"):
        oauth_flow.capture_loopback_code(server, captured, timeout=5)
    thread.join(timeout=5)
    page = result["resp"].text
    assert "Connected" not in page
    assert "not completed" in page and "access_denied" in page


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
