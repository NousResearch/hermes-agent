"""Core makes Codex- and xAI-signed requests for plugins; the token never reaches plugin code.

Real loopback HTTP servers stand in for the provider; a real plugin directory under the test
HERMES_HOME makes the call, and a fake sign-in is seeded in that home's auth store.
"""

import base64
import importlib.util
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import hermes_cli.auth as hermes_auth
from hermes_cli import auth_xai
from hermes_cli import plugin_provider_requests as ppr
from hermes_cli.auth_constants import AuthError


def _seg(d: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()


def _jwt() -> str:
    seg = _seg
    return f"{seg({'alg': 'none'})}.{seg({'exp': int(time.time()) + 86400, 'sub': 'fake'})}.sig"


TOKEN = _jwt()


class _Stub:
    def __init__(self, status=200, location="", body=b'{"value": "ek_ephemeral", "expires_at": 1}'):
        self.seen: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                stub.seen.append({"path": self.path, "headers": dict(self.headers),
                                  "auth": self.headers.get_all("Authorization") or [],
                                  "body": self.rfile.read(length)})
                self.send_response(status)
                if location:
                    self.send_header("Location", location)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

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


# ── xAI Grok OAuth: the same helper, a different sign-in ─────────────────────────────────────────

XAI_TOKEN = _jwt()
XAI_CANARY = "rt-canary-8f3d2a"


def _sign_in_xai(access: str = XAI_TOKEN, refresh: str = "fake-xai-rt") -> Path:
    """Seed an xAI OAuth grant in the test home's auth store; returns the file for byte comparisons."""
    path = _home() / "auth.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"version": 1, "providers": {"xai-oauth": {
        "auth_mode": "oauth_device_code",
        "tokens": {"access_token": access, "refresh_token": refresh, "token_type": "Bearer"},
        "last_refresh": "2026-11-01T00:00:00Z"}}}), encoding="utf-8")
    return path


def _xai_plugin(requires: bool = True, name: str = "grok-usage"):
    """A dashboard-style plugin that asks core for Grok credits; *requires* False declares another provider."""
    root = _home() / "plugins" / name
    (root / "dashboard").mkdir(parents=True)
    (root / "plugin.yaml").write_text(
        f"name: {name}\nrequires_auth: [{'xai-oauth' if requires else 'openai-codex'}]\n", encoding="utf-8")
    (root / "dashboard" / "plugin_api.py").write_text(
        "from hermes_cli.plugin_provider_requests import credentialed_provider_request\n"
        "def credits(url, **kwargs):\n"
        "    return credentialed_provider_request('xai-oauth', 'GET', url, **kwargs)\n"
        "def post(url, **kwargs):\n"
        "    return credentialed_provider_request('xai-oauth', 'POST', url, **kwargs)\n", encoding="utf-8")
    target = root / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location(f"hermes_dashboard_plugin_{name}", target)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def xai_env(monkeypatch):
    """A custom inference base URL from the host shell would change the credential-base guard."""
    monkeypatch.delenv("HERMES_XAI_BASE_URL", raising=False)
    monkeypatch.delenv("XAI_BASE_URL", raising=False)


@pytest.fixture
def xai_stub(monkeypatch):
    allowed, other = _Stub(), _Stub()
    origins, resolve = ppr._PROVIDERS["xai-oauth"]
    # Test-only: the loopback stub joins the real allowlist; ``other`` stays outside it.
    monkeypatch.setitem(ppr._PROVIDERS, "xai-oauth",
                        (origins | {("http", "127.0.0.1", allowed.server.server_address[1])}, resolve))
    yield allowed, other
    allowed.close()
    other.close()


@pytest.fixture
def sent(monkeypatch):
    """Every URL httpx actually transmitted, so a refusal can be proven to precede the wire."""
    import httpx

    calls: list[str] = []
    real_send = httpx.Client.send

    def spy(self, request, *args, **kwargs):
        calls.append(str(request.url))
        return real_send(self, request, *args, **kwargs)

    monkeypatch.setattr(httpx.Client, "send", spy)
    return calls


def test_xai_declaration_and_plugin_location_are_required(xai_env, xai_stub, sent):
    allowed, _ = xai_stub
    _sign_in_xai()
    assert _xai_plugin().credits(allowed.url + "/v1/billing?format=credits").status == 200

    with pytest.raises(PermissionError, match="requires_auth"):
        _xai_plugin(requires=False, name="other-usage").credits(allowed.url)
    with pytest.raises(PermissionError, match="installed plugin"):
        ppr.credentialed_provider_request("xai-oauth", "GET", allowed.url)
    assert len(sent) == 1


@pytest.mark.parametrize("url,message", [
    ("https://grok.com.evil.test/v1/billing", "Refusing"),
    ("https://evil.test/cli-chat-proxy.grok.com/v1/billing", "Refusing"),
    ("https://evil.grok.com/v1/billing", "Refusing"),
    ("http://cli-chat-proxy.grok.com/v1/billing", "Refusing"),
    ("https://cli-chat-proxy.grok.com:8443/v1/billing", "Refusing"),
    ("https://grok.com:80/v1/billing", "Refusing"),
    ("https://cli-chat-proxy.grok.com:not-a-port/v1/billing", "malformed"),
    ("https://cli-chat-proxy.grok.com@evil.test/v1/billing", "Refusing"),
    ("https://user:pass@cli-chat-proxy.grok.com/v1/billing", "embedded credentials"),
])
def test_xai_refuses_non_allowlisted_urls(url, message, xai_env, xai_stub, sent):
    _sign_in_xai()
    with pytest.raises(PermissionError, match=message):
        _xai_plugin().credits(url)
    assert sent == []


def test_xai_uses_the_active_profiles_bearer_and_never_rotates(xai_env, xai_stub, sent, monkeypatch):
    allowed, _ = xai_stub
    auth_path = _sign_in_xai()
    before = auth_path.read_bytes()
    sibling = _home().parent / "other-profile" / "auth.json"
    sibling.parent.mkdir(parents=True, exist_ok=True)
    sibling.write_text(json.dumps({"version": 1, "providers": {"xai-oauth": {"tokens": {
        "access_token": "other-profile-token", "refresh_token": "other-rt"}}}}), encoding="utf-8")

    resolutions: list[dict] = []
    real_resolve = auth_xai.resolve_xai_oauth_runtime_credentials

    def spy(**kwargs):
        resolutions.append(kwargs)
        return real_resolve(**kwargs)

    monkeypatch.setattr(auth_xai, "resolve_xai_oauth_runtime_credentials", spy)
    result = _xai_plugin().credits(
        allowed.url + "/v1/billing?format=credits",
        headers={"x-xai-token-auth": "xai-grok-cli", "Accept": "application/json",
                 "User-Agent": "hermes-provider-usage/0.9"})

    seen = allowed.seen[0]
    assert result.status == 200 and result.json()["value"] == "ek_ephemeral"
    assert seen["auth"] == [f"Bearer {XAI_TOKEN}"]
    assert seen["path"] == "/v1/billing?format=credits"
    assert seen["headers"]["x-xai-token-auth"] == "xai-grok-cli"
    assert seen["headers"]["User-Agent"] == "hermes-provider-usage/0.9"
    assert "other-profile-token" not in repr(seen)
    assert XAI_TOKEN not in repr(result) and XAI_TOKEN not in json.dumps(result.headers)
    # Read-only: the token was never forced through a refresh and the store was not rewritten.
    assert resolutions == [{"force_refresh": False, "refresh_if_expiring": True}]
    assert auth_path.read_bytes() == before


def test_xai_billing_then_plan_settings_two_calls_one_grant(xai_env, xai_stub, sent):
    """The Grok fetch is two GETs (billing, then /v1/settings); both carry the same core-attached bearer."""
    allowed, _ = xai_stub
    _sign_in_xai()
    plugin = _xai_plugin()
    billing = plugin.credits(allowed.url + "/v1/billing?format=credits",
                             headers={"Accept": "application/json", "x-xai-token-auth": "xai-grok-cli"}, timeout=15)
    settings = plugin.credits(allowed.url + "/v1/settings",
                              headers={"Accept": "application/json", "x-xai-token-auth": "xai-grok-cli"}, timeout=2.0)

    assert billing.status == settings.status == 200
    assert [seen["path"] for seen in allowed.seen] == ["/v1/billing?format=credits", "/v1/settings"]
    assert all(seen["auth"] == [f"Bearer {XAI_TOKEN}"] for seen in allowed.seen)


def test_xai_caller_headers_cannot_override_authorization_host_or_length(xai_env, xai_stub, sent):
    allowed, _ = xai_stub
    _sign_in_xai()
    _xai_plugin().post(allowed.url + "/v1/billing", json={"probe": True}, headers={
        "authorization": "Bearer canary-lower", "Authorization": "Bearer canary-upper",
        "Host": "evil.test", "Content-Length": "999", "X-Request-Id": "abc"})

    seen = allowed.seen[0]
    assert seen["auth"] == [f"Bearer {XAI_TOKEN}"]
    assert seen["headers"]["Host"].startswith("127.0.0.1:")
    assert seen["headers"]["X-Request-Id"] == "abc"
    assert int(seen["headers"]["Content-Length"]) == len(seen["body"])
    assert b"canary" not in seen["body"]


def test_xai_grpc_web_bytes_body_and_response_are_preserved(xai_env, monkeypatch, sent):
    frame, reply = b"\x00\x00\x00\x00\x00", b"\x00\x00\x00\x00\x04\x00\xff\x10\x02"
    allowed = _Stub(body=reply)
    origins, resolve = ppr._PROVIDERS["xai-oauth"]
    monkeypatch.setitem(ppr._PROVIDERS, "xai-oauth",
                        (origins | {("http", "127.0.0.1", allowed.server.server_address[1])}, resolve))
    _sign_in_xai()
    try:
        result = _xai_plugin().post(
            allowed.url + "/grok_api_v2.GrokBuildBilling/GetGrokCreditsConfig", content=frame,
            headers={"Content-Type": "application/grpc-web+proto", "Accept": "application/grpc-web+proto",
                     "X-Grpc-Web": "1", "Origin": "https://grok.com", "Referer": "https://grok.com/"})
    finally:
        allowed.close()

    seen = allowed.seen[0]
    assert seen["body"] == frame
    assert seen["headers"]["Content-Type"] == "application/grpc-web+proto"
    assert seen["headers"]["X-Grpc-Web"] == "1"
    assert seen["headers"]["Referer"] == "https://grok.com/"
    assert result.body == reply and result.status == 200


def test_xai_bytes_and_json_are_mutually_exclusive(xai_env, xai_stub, sent):
    _sign_in_xai()
    with pytest.raises(ValueError, match="not both"):
        _xai_plugin().post("https://cli-chat-proxy.grok.com/v1/billing", json={}, content=b"\x00")
    assert sent == []


def test_xai_redirect_is_returned_not_followed(xai_env, monkeypatch, sent):
    other = _Stub()
    redirecting = _Stub(status=302, location=other.url + "/steal")
    origins, resolve = ppr._PROVIDERS["xai-oauth"]
    monkeypatch.setitem(ppr._PROVIDERS, "xai-oauth",
                        (origins | {("http", "127.0.0.1", redirecting.server.server_address[1])}, resolve))
    _sign_in_xai()
    try:
        result = _xai_plugin().credits(redirecting.url + "/v1/billing?format=credits")
    finally:
        redirecting.close()
        other.close()

    assert result.status == 302 and other.seen == []
    assert result.headers["location"].endswith("/steal")
    assert redirecting.seen[0]["auth"] == [f"Bearer {XAI_TOKEN}"]


def test_xai_custom_credential_endpoint_is_refused(xai_env, monkeypatch, sent):
    monkeypatch.setattr(auth_xai, "resolve_xai_oauth_runtime_credentials",
                        lambda **kwargs: {"api_key": XAI_TOKEN,
                                          "base_url": "https://gateway.example.test/v1"})
    with pytest.raises(PermissionError, match="custom endpoint"):
        _xai_plugin().credits("https://cli-chat-proxy.grok.com/v1/billing?format=credits")
    assert sent == []


def test_xai_refresh_error_is_redacted(xai_env, monkeypatch, sent):
    leaky = f'xAI token refresh failed. Response: {{"refresh_token": "{XAI_CANARY}"}}'

    def refuse(**kwargs):
        raise AuthError(leaky, provider="xai-oauth", code="xai_refresh_failed", relogin_required=True)

    monkeypatch.setattr(auth_xai, "resolve_xai_oauth_runtime_credentials", refuse)
    with pytest.raises(ppr.ProviderNotSignedIn) as info:
        _xai_plugin().credits("https://cli-chat-proxy.grok.com/v1/billing?format=credits")

    exc = info.value
    assert XAI_CANARY not in "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    assert exc.__cause__ is None and exc.__context__ is None
    assert "auth add xai-oauth" in str(exc)
    assert sent == []


def test_xai_not_signed_in_names_the_command(xai_env, monkeypatch, sent):
    monkeypatch.setattr(hermes_auth, "_load_global_auth_store", dict)
    with pytest.raises(ppr.ProviderNotSignedIn, match="auth add xai-oauth"):
        _xai_plugin().credits("https://cli-chat-proxy.grok.com/v1/billing?format=credits")
    assert sent == []
