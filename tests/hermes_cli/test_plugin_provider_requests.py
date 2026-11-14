"""Core makes signed-in provider requests for plugins; the token never reaches plugin code.

Real loopback HTTP servers stand in for the provider; a real plugin directory under the test
HERMES_HOME makes the call, and a fake sign-in is seeded in that home's auth store.
"""

import base64
import importlib.util
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from hermes_cli import auth as auth_module
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


def _allow_loopback(monkeypatch, stub: _Stub) -> None:
    """Test-only: the loopback stub joins every allowlist; real entries are kept, so refusals stay real."""
    real = ppr._origins
    loopback = ("http", "127.0.0.1", stub.server.server_address[1])
    monkeypatch.setattr(ppr, "_origins", lambda provider: real(provider) | {loopback})


@pytest.fixture
def stubs(monkeypatch):
    allowed, other = _Stub(), _Stub()
    _allow_loopback(monkeypatch, allowed)
    yield allowed, other
    allowed.close()
    other.close()


def _home() -> Path:
    return Path(os.environ["HERMES_HOME"])


def _sign_in(provider: str = "openai-codex", *, access_token: str = TOKEN) -> None:
    path = _home() / "auth.json"
    store = json.loads(path.read_text()) if path.exists() else {"version": 1}
    store.setdefault("providers", {})[provider] = {
        "auth_mode": "oauth_device_code",
        "tokens": {"access_token": access_token, "refresh_token": "fake-rt"},
    }
    path.write_text(json.dumps(store))


def _plugin(provider: str) -> ModuleType:
    """A dashboard-style plugin that declares *provider*, loaded the way the dashboard loads it."""
    return _load_plugin(directory=provider, declares=[provider])


def _load_plugin(*, directory: str, declares: list[str], calls: str | None = None) -> ModuleType:
    root = _home() / "plugins" / directory
    (root / "dashboard").mkdir(parents=True)
    declaration = "requires_auth: [" + ", ".join(declares) + "]\n" if declares else ""
    (root / "plugin.yaml").write_text(f"name: {directory}\n{declaration}")
    provider = calls or (declares[0] if declares else "openai-codex")
    (root / "dashboard" / "plugin_api.py").write_text(
        "from hermes_cli.plugin_provider_requests import credentialed_provider_request\n"
        "def mint(url, headers=None):\n"
        f"    return credentialed_provider_request({provider!r}, 'POST', url, json={{'session': {{}}}},"
        " headers=headers)\n"
        "def send_bytes(url, body, content_type):\n"
        f"    return credentialed_provider_request({provider!r}, 'POST', url, content=body,"
        " headers={'Content-Type': content_type})\n"
        "def both(url):\n"
        f"    return credentialed_provider_request({provider!r}, 'POST', url, json={{}}, content=b'x')\n"
        "def get(url):\n"
        f"    return credentialed_provider_request({provider!r}, 'GET', url)\n")
    spec = importlib.util.spec_from_file_location(f"hermes_dashboard_plugin_{root.name}",
                                                  root / "dashboard" / "plugin_api.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_token_attached_for_allowlisted_origin_and_never_returned(stubs):
    allowed, _ = stubs
    _sign_in()
    result = _plugin("openai-codex").mint(allowed.url + "/v1/realtime/client_secrets")

    assert result.status == 200 and result.json()["value"] == "ek_ephemeral"
    assert allowed.seen[0]["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert json.loads(allowed.seen[0]["body"]) == {"session": {}}
    assert TOKEN not in repr(result) and TOKEN not in json.dumps(result.headers)


def test_disallowed_origin_and_redirect_never_carry_the_token(stubs, monkeypatch):
    _, other = stubs
    _sign_in()
    plugin = _plugin("openai-codex")
    with pytest.raises(PermissionError, match="Refusing"):
        plugin.mint(other.url + "/steal")
    assert other.seen == []

    redirecting = _Stub(status=302, location=other.url + "/steal")
    _allow_loopback(monkeypatch, redirecting)
    try:
        result = plugin.mint(redirecting.url + "/v1/realtime/client_secrets")
    finally:
        redirecting.close()
    assert result.status == 302 and other.seen == []


def test_undeclared_plugin_and_non_plugin_callers_are_refused(stubs):
    allowed, _ = stubs
    _sign_in()
    with pytest.raises(PermissionError, match="requires_auth"):
        _load_plugin(directory="undeclared", declares=[]).mint(allowed.url)
    with pytest.raises(PermissionError, match="installed plugin"):
        ppr.credentialed_provider_request("openai-codex", "GET", allowed.url)
    assert allowed.seen == []


def test_not_signed_in_names_the_sign_in_command(stubs):
    allowed, _ = stubs
    with pytest.raises(ppr.ProviderNotSignedIn, match="hermes auth add openai-codex"):
        _plugin("openai-codex").mint(allowed.url)
    assert allowed.seen == []


def test_registered_providers_are_https_on_443():
    assert set(ppr.credentialed_providers()) >= {
        "openai-codex", "xai-oauth", "qwen-oauth", "minimax-oauth", "nous", "openai-api"}
    for name in ppr.credentialed_providers():
        origins = ppr._origins(name)
        assert origins, name
        assert all(scheme == "https" and port == 443 for scheme, _host, port in origins), name


def test_only_documented_providers_reach_more_than_their_own_api_host():
    """The whole per-provider list, in one assertion: same-host reads are automatic, extras are written down."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    extras = {"openai-codex": {"api.openai.com"}, "xai-oauth": {"cli-chat-proxy.grok.com", "grok.com"}}
    for name in ppr.credentialed_providers():
        api_host = PROVIDER_REGISTRY[name].inference_base_url.split("/")[2]
        hosts = {host for _scheme, host, _port in ppr._origins(name)}
        assert hosts == {api_host} | extras.get(name, set()), name


def test_an_api_key_provider_works_from_its_api_host_with_no_declaration_at_all():
    """The plug-and-play claim: an official provider nobody configured for this feature still works."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    access = ppr._access("deepseek")
    host = PROVIDER_REGISTRY["deepseek"].inference_base_url.split("/")[2]
    assert access is not None and access.read_only
    assert access.origins == frozenset({("https", host, 443)})


def test_a_provider_with_no_plugin_specific_data_works_from_its_api_host_alone():
    from hermes_cli.auth import PROVIDER_REGISTRY

    for name in ("qwen-oauth", "minimax-oauth", "nous"):
        host = PROVIDER_REGISTRY[name].inference_base_url.split("/")[2]
        assert ("https", host, 443) in ppr._origins(name), name


def test_a_new_sign_in_is_a_flow_row_not_broker_code(stubs, monkeypatch):
    """The module docstring's claim, tested: a registry row with a base URL is the whole integration."""
    allowed, _ = stubs
    rows = dict(ppr._provider_registry())
    rows["future-oauth"] = SimpleNamespace(auth_type="oauth_external",
                                          inference_base_url="https://api.future.test/v1")
    monkeypatch.setattr(ppr, "_provider_registry", lambda: rows)
    monkeypatch.setitem(auth_module.OAUTH_PROVIDER_FLOWS, "future-oauth", SimpleNamespace(
        resolve=lambda: {"api_key": TOKEN, "base_url": "https://api.future.test/v1"}))

    result = _plugin("future-oauth").mint(allowed.url + "/usage")

    assert result.status == 200
    assert allowed.seen[0]["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert "future-oauth" in ppr.credentialed_providers()


def test_xai_sign_in_is_used_for_its_own_hosts_and_never_returned(stubs):
    allowed, other = stubs
    _sign_in("xai-oauth")
    plugin = _plugin("xai-oauth")

    result = plugin.mint(allowed.url + "/v1/billing?format=credits")

    assert result.status == 200
    assert allowed.seen[0]["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert TOKEN not in repr(result) and TOKEN not in json.dumps(result.headers)
    with pytest.raises(PermissionError, match="Refusing"):
        plugin.mint(other.url + "/billing")
    assert other.seen == []


def test_xai_plugin_must_declare_the_sign_in(stubs):
    allowed, _ = stubs
    _sign_in("xai-oauth")
    with pytest.raises(PermissionError, match=r"requires_auth: \[xai-oauth\]"):
        _load_plugin(directory="grok-undeclared", declares=["openai-codex"], calls="xai-oauth").mint(allowed.url)
    assert allowed.seen == []


def test_xai_not_signed_in_names_the_sign_in_command(stubs):
    allowed, _ = stubs
    with pytest.raises(ppr.ProviderNotSignedIn, match="hermes auth add xai-oauth"):
        _plugin("xai-oauth").mint(allowed.url)
    assert allowed.seen == []


def test_caller_cannot_replace_reserved_headers(stubs):
    allowed, _ = stubs
    _sign_in()
    _plugin("openai-codex").mint(
        allowed.url + "/v1/realtime/client_secrets",
        headers={"Authorization": "Bearer attacker", "Host": "evil.example",
                 "Content-Length": "9999", "X-Trace": "keep"})

    seen = allowed.seen[0]["headers"]
    assert seen["Authorization"] == f"Bearer {TOKEN}"
    assert seen["Host"] != "evil.example"
    assert seen["Content-Length"] == str(len(allowed.seen[0]["body"]))
    assert seen["X-Trace"] == "keep"


def test_raw_body_and_caller_content_type_are_forwarded(stubs):
    allowed, _ = stubs
    _sign_in("xai-oauth")
    body = b"\x00\x00\x00\x00\x00"

    _plugin("xai-oauth").send_bytes(allowed.url + "/billing", body, "application/grpc-web+proto")

    assert allowed.seen[0]["body"] == body
    assert allowed.seen[0]["headers"]["Content-Type"] == "application/grpc-web+proto"
    assert allowed.seen[0]["headers"]["Authorization"] == f"Bearer {TOKEN}"


def test_json_and_content_are_mutually_exclusive(stubs):
    allowed, _ = stubs
    _sign_in()
    with pytest.raises(ValueError, match="json= or content="):
        _plugin("openai-codex").both(allowed.url)
    assert allowed.seen == []


def test_url_with_embedded_credentials_is_refused(stubs):
    allowed, _ = stubs
    _sign_in()
    credential_bearing = allowed.url.replace("http://", "http://user:password@")
    with pytest.raises(PermissionError, match="embedded credentials"):
        _plugin("openai-codex").mint(credential_bearing + "/v1/realtime/client_secrets")
    assert allowed.seen == []


def test_api_key_provider_is_read_only_and_uses_its_own_key(stubs, monkeypatch):
    """No declaration beyond requires_auth, and a plugin cannot spend an API key."""
    allowed, _ = stubs
    monkeypatch.setenv("DEEPSEEK_API_KEY", TOKEN)
    plugin = _plugin("deepseek")

    result = plugin.get(allowed.url + "/user/balance")

    assert result.status == 200
    assert allowed.seen[0]["headers"]["Authorization"] == f"Bearer {TOKEN}"
    with pytest.raises(PermissionError, match="only GET and HEAD"):
        plugin.mint(allowed.url)
    assert len(allowed.seen) == 1


def test_api_key_not_configured_names_the_env_var(stubs, monkeypatch):
    allowed, _ = stubs
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ppr.ProviderNotSignedIn, match="DEEPSEEK_API_KEY"):
        _plugin("deepseek").get(allowed.url)
    assert allowed.seen == []


def test_a_failed_refresh_leaves_no_chain_to_the_token(stubs, monkeypatch):
    """httpx errors carry their request; a rebuild must not leave the refresh token reachable."""
    allowed, _ = stubs
    import httpx

    secret = "rt_secret_refresh_value"

    def failing(_url):
        request = httpx.Request("POST", "https://api.xai.test/oauth2/token",
                                headers={"Authorization": f"Bearer {secret}"})
        raise httpx.HTTPStatusError("401 Unauthorized", request=request,
                                    response=httpx.Response(401, request=request))

    monkeypatch.setattr(ppr, "_access", lambda _name: ppr._Access(
        frozenset({("http", "127.0.0.1", allowed.server.server_address[1])}), failing, read_only=True))

    with pytest.raises(RuntimeError, match="401") as caught:
        _plugin("deepseek").get(allowed.url + "/user/balance")

    exc = caught.value
    assert exc.__cause__ is None and exc.__context__ is None
    broker_frames = []
    tb = exc.__traceback__
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == ppr.__file__:
            broker_frames.extend(repr(value) for value in tb.tb_frame.f_locals.values())
        tb = tb.tb_next
    assert broker_frames, "expected the broker frame in the traceback"
    assert all(secret not in value for value in broker_frames), "a broker frame still holds the refresh token"
    assert secret not in repr(exc) and secret not in str(exc)
    assert allowed.seen == []
