"""Core makes Codex-signed requests for plugins; the token never reaches plugin code.

Real loopback HTTP and TLS websocket servers stand in for the provider; a real plugin directory under
the test HERMES_HOME makes the call, and a fake Codex sign-in is seeded in that home's auth store.
"""

import base64
import importlib.util
import json
import os
import ssl
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from websockets.exceptions import InvalidStatus
from websockets.sync.server import serve

from hermes_cli import plugin_provider_requests as ppr
from tests.fakes.providers.oauth_token_server import make_test_ca


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
        "from hermes_cli.plugin_provider_requests import (\n"
        "    credentialed_provider_request, credentialed_provider_websocket)\n"
        "def mint(url):\n"
        "    return credentialed_provider_request('openai-codex', 'POST', url, json={'session': {}})\n"
        "def open_socket(url, headers=None):\n"
        "    return credentialed_provider_websocket('openai-codex', url, headers=headers, timeout=10)\n")
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


class _WssStub:
    """A TLS websocket echo server; with *redirect_to* it answers the handshake with a 302 instead."""

    def __init__(self, ca, redirect_to=""):
        self.seen: list = []
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(ca.leaf_pem, ca.leaf_key)

        def process_request(connection, request):
            self.seen.append(request.headers)
            if redirect_to:
                response = connection.respond(HTTPStatus.FOUND, "")
                response.headers["Location"] = redirect_to
                return response
            return None

        def echo(websocket):
            for message in websocket:
                websocket.send(message)

        self.server = serve(echo, "127.0.0.1", 0, ssl=context, process_request=process_request)
        self.port = self.server.socket.getsockname()[1]
        self.url = f"wss://127.0.0.1:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def _allow_https(monkeypatch, port):
    origins, resolve = ppr._PROVIDERS["openai-codex"]
    monkeypatch.setitem(ppr._PROVIDERS, "openai-codex", (origins | {("https", "127.0.0.1", port)}, resolve))


@pytest.fixture
def wss_stubs(tmp_path, monkeypatch):
    ca = make_test_ca(tmp_path / "ca", ["127.0.0.1"])
    monkeypatch.setenv("SSL_CERT_FILE", str(ca.ca_pem))
    allowed, other = _WssStub(ca), _WssStub(ca)
    # Test-only: the loopback stub's https origin joins the real allowlist; ``other`` stays outside it.
    _allow_https(monkeypatch, allowed.port)
    yield ca, allowed, other
    allowed.close()
    other.close()


def _reachable_text(exc: BaseException) -> str:
    """Everything plugin code can read off *exc*: message, args, chain and every traceback frame's locals."""
    parts = []
    while exc is not None:
        parts += [str(exc), repr(exc.args), repr(vars(exc))]
        tb = exc.__traceback__
        while tb is not None:
            parts.append(repr(tb.tb_frame.f_locals))
            tb = tb.tb_next
        exc = exc.__cause__ or exc.__context__
    return "\n".join(parts)


def test_websocket_carries_the_token_only_in_the_handshake(wss_stubs):
    _, allowed, _ = wss_stubs
    _sign_in()
    with _plugin(declares=True).open_socket(allowed.url + "/v1/realtime",
                                            headers={"authorization": "Bearer plugin-guess"}) as ws:
        ws.send("ping")
        assert ws.recv(timeout=10) == "ping"
        assert ws.request is None and TOKEN not in repr(vars(ws))
    assert allowed.seen[0].get_all("Authorization") == [f"Bearer {TOKEN}"]


def test_websocket_refuses_ws_scheme_and_other_hosts(wss_stubs):
    _, allowed, other = wss_stubs
    _sign_in()
    plugin = _plugin(declares=True)
    with pytest.raises(PermissionError, match="wss://"):
        plugin.open_socket(f"ws://127.0.0.1:{allowed.port}/v1/realtime")
    with pytest.raises(PermissionError, match="Refusing"):
        plugin.open_socket(other.url + "/steal")
    assert allowed.seen == [] and other.seen == []


def test_websocket_refuses_undeclared_plugins_and_non_plugin_callers(wss_stubs):
    _, allowed, _ = wss_stubs
    _sign_in()
    with pytest.raises(PermissionError, match="requires_auth"):
        _plugin(declares=False).open_socket(allowed.url)
    with pytest.raises(PermissionError, match="installed plugin"):
        ppr.credentialed_provider_websocket("openai-codex", allowed.url)
    assert allowed.seen == []


def test_websocket_redirect_is_not_followed_and_its_error_holds_no_token(wss_stubs, monkeypatch):
    ca, _, other = wss_stubs
    _sign_in()
    redirecting = _WssStub(ca, redirect_to=other.url + "/steal")
    _allow_https(monkeypatch, redirecting.port)
    try:
        with pytest.raises(InvalidStatus) as caught:
            _plugin(declares=True).open_socket(redirecting.url + "/v1/realtime")
    finally:
        redirecting.close()
    assert redirecting.seen[0]["Authorization"] == f"Bearer {TOKEN}"
    assert other.seen == []
    assert TOKEN not in _reachable_text(caught.value)
