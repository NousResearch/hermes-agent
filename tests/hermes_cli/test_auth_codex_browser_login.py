"""``hermes auth add openai-codex --browser``: opt-in loopback authorization-code + PKCE (#95743).

Exercises the real loopback listener and a real token endpoint (both stdlib servers on ephemeral
ports); only the system browser is replaced by an HTTP client following the authorize redirect.
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlparse

import pytest


def _jwt(email: str) -> str:
    b64 = lambda raw: base64.urlsafe_b64encode(raw).rstrip(b"=").decode()  # noqa: E731
    return f"{b64(b'{}')}.{b64(json.dumps({'email': email}).encode())}.sig"


def _args(**overrides):
    base = dict(provider="openai-codex", auth_type="oauth", api_key=None, label=None, priority=None,
                browser=False, no_browser=True, timeout=10.0, scope=None)
    return SimpleNamespace(**{**base, **overrides})


class _FakeOpenAI(HTTPServer):
    """Authorize → 302 with a code; token → verifies PKCE and returns tokens; records both."""

    def __init__(self):
        self.seen: dict = {}
        super().__init__(("127.0.0.1", 0), _FakeOpenAIHandler)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def do_GET(self):
        parsed = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        self.server.seen["authorize"] = q
        self.send_response(302)
        self.send_header("Location", f"{q['redirect_uri']}?{urlencode({'code': 'AC-1', 'state': q['state']})}")
        self.end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"])).decode()
        form = {k: v[0] for k, v in parse_qs(body).items()}
        self.server.seen["token"] = form
        challenge = self.server.seen["authorize"]["code_challenge"]
        derived = base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"].encode()).digest()).rstrip(b"=").decode()
        ok = form.get("grant_type") == "authorization_code" and form.get("code") == "AC-1" and derived == challenge
        payload = json.dumps(
            {"access_token": _jwt("pkce@example.com"), "refresh_token": "rt-pkce"} if ok else {"error": "invalid_grant"}
        ).encode()
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture
def fake_openai():
    server = _FakeOpenAI()
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


def test_browser_flag_runs_loopback_pkce_and_stores_loopback_source(tmp_path, monkeypatch, fake_openai):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir()
    (tmp_path / "hermes" / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    from hermes_cli import auth_codex_browser as browser_mod
    from hermes_cli.auth_commands import auth_add_command

    monkeypatch.setattr(browser_mod, "CODEX_OAUTH_AUTHORIZE_URL", f"{fake_openai.base}/oauth/authorize")
    monkeypatch.setattr(browser_mod, "CODEX_OAUTH_TOKEN_URL", f"{fake_openai.base}/oauth/token")
    monkeypatch.setattr(browser_mod, "CODEX_BROWSER_CALLBACK_PORT", 0)  # ephemeral; production is 1455
    monkeypatch.setattr(browser_mod, "_can_open_graphical_browser", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.auth._codex_device_code_login",
        lambda: pytest.fail("--browser must not run the device-code flow"))

    def _browser(url):  # the "browser": follow the authorize redirect back to the loopback listener
        threading.Thread(target=lambda: urllib.request.urlopen(url, timeout=5).read(), daemon=True).start()
        return True
    monkeypatch.setattr(browser_mod.webbrowser, "open", _browser)

    auth_add_command(_args(browser=True, no_browser=False))

    payload = json.loads((tmp_path / "hermes" / "auth.json").read_text())
    [entry] = payload["credential_pool"]["openai-codex"]
    assert entry["source"] == "manual:loopback_pkce"
    assert entry["access_token"] == _jwt("pkce@example.com") and entry["refresh_token"] == "rt-pkce"
    assert payload["active_provider"] == "openai-codex"
    authorize, token = fake_openai.seen["authorize"], fake_openai.seen["token"]
    assert authorize["code_challenge_method"] == "S256" and authorize["response_type"] == "code"
    assert authorize["redirect_uri"].startswith("http://localhost:") and authorize["redirect_uri"].endswith("/auth/callback")
    assert token["redirect_uri"] == authorize["redirect_uri"] and token["client_id"] == authorize["client_id"]


def test_default_is_device_code_and_busy_callback_port_falls_back(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    (tmp_path / "hermes").mkdir()
    (tmp_path / "hermes" / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
    from hermes_cli import auth_codex_browser as browser_mod
    from hermes_cli.auth_commands import auth_add_command

    device_logins = []

    def _device():
        device_logins.append(1)
        return {"tokens": {"access_token": _jwt("device@example.com"), "refresh_token": "rt-dev"},
                "base_url": "https://chatgpt.com/backend-api/codex", "last_refresh": "2026-01-01T00:00:00Z"}
    monkeypatch.setattr("hermes_cli.auth._codex_device_code_login", _device)
    bind_attempts = []
    real_bind = browser_mod._bind_loopback_callback_server
    monkeypatch.setattr(
        browser_mod, "_bind_loopback_callback_server",
        lambda *a, **kw: bind_attempts.append(1) or real_bind(*a, **kw))

    # Default (no flag, default config): device code, and the loopback listener is never even bound.
    auth_add_command(_args())
    assert device_logins == [1] and bind_attempts == []

    # --browser while the registered port is taken (a Codex CLI login in progress): clear notice, device code.
    with socket.socket() as occupant:
        occupant.bind(("127.0.0.1", 0))
        occupant.listen(1)
        monkeypatch.setattr(browser_mod, "CODEX_BROWSER_CALLBACK_PORT", occupant.getsockname()[1])
        auth_add_command(_args(browser=True, label="second"))
    assert device_logins == [1, 1] and bind_attempts == [1]
    out = capsys.readouterr().out
    assert "already in use" in out and "falling back to the device-code login" in out
    sources = [e["source"] for e in json.loads((tmp_path / "hermes" / "auth.json").read_text())["credential_pool"]["openai-codex"]]
    assert sources == ["manual:device_code", "manual:device_code"]
