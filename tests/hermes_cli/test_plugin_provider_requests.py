"""Core makes Codex-signed requests for plugins; the token never reaches plugin code.

Real loopback HTTP servers stand in for the provider; a real plugin directory under the test
HERMES_HOME makes the call, and a fake Codex sign-in is seeded in that home's auth store.
"""

import base64
import importlib.util
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from hermes_cli import plugin_provider_requests as ppr


def _seg(d: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()


def _jwt() -> str:
    seg = _seg
    return f"{seg({'alg': 'none'})}.{seg({'exp': int(time.time()) + 86400, 'sub': 'fake'})}.sig"


TOKEN = _jwt()


class _Stub:
    def __init__(self, status=200, location=""):
        self.seen: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                stub.seen.append({"path": self.path, "headers": dict(self.headers),
                                  "body": self.rfile.read(length)})
                self.send_response(status)
                if location:
                    self.send_header("Location", location)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"value": "ek_ephemeral", "expires_at": 1}')

            do_GET = do_POST = _answer

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def stubs(monkeypatch):
    allowed, other = _Stub(), _Stub()
    origins, resolve = ppr._PROVIDERS["openai-codex"]
    # Test-only: the loopback stub joins the real allowlist; ``other`` stays outside it.
    monkeypatch.setitem(ppr._PROVIDERS, "openai-codex",
                        (origins | {("http", "127.0.0.1", allowed.server.server_address[1])}, resolve))
    yield allowed, other
    allowed.close()
    other.close()


def _home() -> Path:
    return Path(os.environ["HERMES_HOME"])


def _sign_in():
    (_home() / "auth.json").write_text(json.dumps({"version": 1, "providers": {"openai-codex": {
        "auth_mode": "chatgpt", "tokens": {"access_token": TOKEN, "refresh_token": "fake-rt"}}}}))


def _plugin(declares: bool):
    """A dashboard-style plugin module (``dashboard/plugin_api.py``) loaded the way the dashboard does."""
    root = _home() / "plugins" / ("voice" if declares else "undeclared")
    (root / "dashboard").mkdir(parents=True)
    (root / "plugin.yaml").write_text("name: voice\n" + ("requires_auth: [openai-codex]\n" if declares else ""))
    (root / "dashboard" / "plugin_api.py").write_text(
        "from hermes_cli.plugin_provider_requests import credentialed_provider_request\n"
        "def mint(url):\n"
        "    return credentialed_provider_request('openai-codex', 'POST', url, json={'session': {}})\n")
    spec = importlib.util.spec_from_file_location(f"hermes_dashboard_plugin_{root.name}", root / "dashboard" / "plugin_api.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_token_attached_for_allowlisted_origin_and_never_returned(stubs):
    allowed, _ = stubs
    _sign_in()
    result = _plugin(declares=True).mint(allowed.url + "/v1/realtime/client_secrets")

    assert result.status == 200 and result.json()["value"] == "ek_ephemeral"
    assert allowed.seen[0]["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert json.loads(allowed.seen[0]["body"]) == {"session": {}}
    assert TOKEN not in repr(result) and TOKEN not in json.dumps(result.headers)


def test_disallowed_origin_and_redirect_never_carry_the_token(stubs, monkeypatch):
    _, other = stubs
    _sign_in()
    plugin = _plugin(declares=True)
    with pytest.raises(PermissionError, match="Refusing"):
        plugin.mint(other.url + "/steal")
    assert other.seen == []

    redirecting = _Stub(status=302, location=other.url + "/steal")
    origins, resolve = ppr._PROVIDERS["openai-codex"]
    monkeypatch.setitem(ppr._PROVIDERS, "openai-codex",
                        (origins | {("http", "127.0.0.1", redirecting.server.server_address[1])}, resolve))
    try:
        result = plugin.mint(redirecting.url + "/v1/realtime/client_secrets")
    finally:
        redirecting.close()
    assert result.status == 302 and other.seen == []


def test_undeclared_plugin_and_non_plugin_callers_are_refused(stubs):
    allowed, _ = stubs
    _sign_in()
    with pytest.raises(PermissionError, match="requires_auth"):
        _plugin(declares=False).mint(allowed.url)
    with pytest.raises(PermissionError, match="installed plugin"):
        ppr.credentialed_provider_request("openai-codex", "GET", allowed.url)
    assert allowed.seen == []


def test_not_signed_in_names_the_sign_in_command(stubs):
    allowed, _ = stubs
    with pytest.raises(ppr.ProviderNotSignedIn, match="hermes auth add openai-codex"):
        _plugin(declares=True).mint(allowed.url)
    assert allowed.seen == []



def _xai_sign_in():
    (_home() / "auth.json").write_text(json.dumps({"version": 1, "providers": {"xai-oauth": {
        "auth_mode": "oauth_device_code", "tokens": {"access_token": TOKEN, "refresh_token": "fake-rt"}}}}))


def _xai_plugin(name: str, requires_auth: str):
    """A dashboard plugin minting an xAI Realtime client secret with the user's Grok sign-in."""
    root = _home() / "plugins" / name
    (root / "dashboard").mkdir(parents=True)
    (root / "plugin.yaml").write_text(f"name: {name}\nrequires_auth: [{requires_auth}]\n")
    (root / "dashboard" / "plugin_api.py").write_text(
        "from hermes_cli.plugin_provider_requests import credentialed_provider_request\n"
        "def mint(url):\n"
        "    return credentialed_provider_request('xai-oauth', 'POST', url, json={'expires_after': {'seconds': 300}})\n")
    spec = importlib.util.spec_from_file_location(f"hermes_dashboard_plugin_{name}", root / "dashboard" / "plugin_api.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def xai_stub(monkeypatch):
    stub = _Stub()
    origins, resolve = ppr._PROVIDERS["xai-oauth"]
    # Test-only: the loopback stub joins the real xAI allowlist.
    monkeypatch.setitem(ppr._PROVIDERS, "xai-oauth",
                        (origins | {("http", "127.0.0.1", stub.server.server_address[1])}, resolve))
    yield stub
    stub.close()


def test_xai_sign_in_mints_for_declared_plugin_and_stays_on_xai(xai_stub):
    _xai_sign_in()
    plugin = _xai_plugin("grok_voice", "xai-oauth")
    result = plugin.mint(xai_stub.url + "/v1/realtime/client_secrets")

    assert result.status == 200 and result.json()["value"] == "ek_ephemeral"
    assert xai_stub.seen[0]["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert json.loads(xai_stub.seen[0]["body"]) == {"expires_after": {"seconds": 300}}
    assert TOKEN not in repr(result)
    # The Grok token never goes to another provider's origin, even one allowlisted for Codex.
    with pytest.raises(PermissionError, match="Refusing"):
        plugin.mint("https://api.openai.com/v1/realtime/client_secrets")


def test_xai_needs_its_own_declaration_and_names_its_sign_in(xai_stub):
    with pytest.raises(ppr.ProviderNotSignedIn, match="hermes auth add xai-oauth"):
        _xai_plugin("grok_voice", "xai-oauth").mint(xai_stub.url)
    _xai_sign_in()
    with pytest.raises(PermissionError, match=r"requires_auth: \[xai-oauth\]"):
        _xai_plugin("codex_only", "openai-codex").mint(xai_stub.url)
    assert xai_stub.seen == []
