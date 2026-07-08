"""Confirm the DESKTOP side of the Hindsight browser-connect flow, end to end,
against a fake UI — no real Hindsight Cloud, no Zitadel, no browser, no Cloud
changes.

A local server stands in for the Hindsight UI's ``/connect/desktop`` route: it
"mints" an API key and 302-redirects it to the desktop's loopback callback,
exactly as the counter-proposal describes. We drive the whole desktop path:
bind loopback → open (fake) browser → UI 302 with ?key= → capture → validate the
CSRF state → store the key as the provider's ``apiKey``.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from plugins.memory.hindsight import oauth_flow

MINTED_KEY = "hsk_testconnect_0123456789abcdef"


def _make_ui(*, state_override: str | None = None):
    """A fake Hindsight UI: /connect/desktop 302s a minted key to the loopback."""

    class _FakeUI(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/connect/desktop":
                self.send_response(404)
                self.end_headers()
                return
            q = parse_qs(parsed.query)
            port = q["port"][0]
            state = state_override if state_override is not None else q["state"][0]
            # The desktop identifies its surface so the consent side can attribute it.
            assert q["source"][0] == "hermes-desktop"
            # The UI (user already signed in) creates a key via its existing
            # key-creation path, then hands it back over the loopback redirect.
            location = (
                f"http://127.0.0.1:{port}/callback?state={state}&key={MINTED_KEY}"
            )
            self.send_response(302)
            self.send_header("Location", location)
            self.end_headers()

        def log_message(self, *args):
            return

    return _FakeUI


def _serve(handler_cls, monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv(
        "HINDSIGHT_CONNECT_URL", f"http://127.0.0.1:{port}/connect/desktop"
    )
    return server


def _browser_driver(url: str) -> None:
    """Stand in for the user's browser: follow the UI's 302 into the loopback."""
    resp = httpx.get(url, follow_redirects=False, timeout=5.0)
    location = resp.headers["Location"]
    for _ in range(100):
        try:
            httpx.get(location, timeout=1.0)
            return
        except Exception:
            time.sleep(0.02)


@pytest.fixture(autouse=True)
def _reset_status():
    oauth_flow._set_status("idle", "")
    yield
    oauth_flow._set_status("idle", "")


def test_connect_stores_minted_key(monkeypatch, tmp_path):
    server = _serve(_make_ui(), monkeypatch)
    try:
        cfg = tmp_path / "hindsight" / "config.json"
        key = oauth_flow.connect_via_loopback(
            config_path=cfg, open_url=_browser_driver, timeout=10.0
        )
        assert key == MINTED_KEY
        # Stored where the Hindsight provider actually reads it.
        data = json.loads(cfg.read_text())
        assert data["apiKey"] == MINTED_KEY
        assert (cfg.stat().st_mode & 0o777) == 0o600
    finally:
        server.shutdown()
        server.server_close()


def test_connect_rejects_state_mismatch(monkeypatch, tmp_path):
    # UI returns the WRONG state → CSRF guard fires, nothing is stored.
    server = _serve(_make_ui(state_override="not-the-real-state"), monkeypatch)
    try:
        cfg = tmp_path / "hindsight" / "config.json"
        with pytest.raises(ValueError, match="state mismatch"):
            oauth_flow.connect_via_loopback(
                config_path=cfg, open_url=_browser_driver, timeout=10.0
            )
        assert not cfg.exists()
    finally:
        server.shutdown()
        server.server_close()


def test_memory_oauth_router_dispatches_to_hindsight():
    """The generic connect framework resolves the hindsight flow by convention,
    so /api/memory/providers/hindsight/oauth/{start,status} work with no new
    route code — the same wiring Honcho uses."""
    from hermes_cli.memory_oauth import _resolve_flow

    flow = _resolve_flow("hindsight")
    assert callable(getattr(flow, "start_loopback_flow_background", None))
    assert callable(getattr(flow, "get_flow_status", None))


def test_background_flow_reports_connected(monkeypatch, tmp_path):
    server = _serve(_make_ui(), monkeypatch)
    try:
        cfg = tmp_path / "hindsight" / "config.json"
        # Point BOTH the flow and the connection detector at the temp config.
        monkeypatch.setattr(oauth_flow, "resolve_config_path", lambda: cfg)
        # The background flow uses the real webbrowser.open — swap in our driver.
        import webbrowser

        monkeypatch.setattr(webbrowser, "open", _browser_driver)

        initial = oauth_flow.start_loopback_flow_background(timeout=10.0)
        assert initial["state"] in ("pending", "connected")

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if oauth_flow.get_flow_status()["state"] == "connected":
                break
            time.sleep(0.05)

        status = oauth_flow.get_flow_status()
        assert status["state"] == "connected"
        assert status["connected"] is True
        assert status["auth"] == "apikey"
        assert json.loads(cfg.read_text())["apiKey"] == MINTED_KEY
    finally:
        server.shutdown()
        server.server_close()
