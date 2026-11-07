"""Core makes xAI-signed (and multipart) requests for plugins; the token never reaches plugin code.

A real plugin directory under the test HERMES_HOME makes the call, a fake xAI Grok sign-in is seeded
in that home's auth store and resolved by core's own xAI resolver, and loopback HTTP servers stand in
for api.x.ai.
"""

import base64
import importlib.util
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from hermes_cli import plugin_provider_requests as ppr


def _seg(d: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()


def _token(sub: str) -> str:
    return f"{_seg({'alg': 'none'})}.{_seg({'exp': int(time.time()) + 86400, 'sub': sub})}.sig"


class _Stub:
    def __init__(self, status=200, location="", hang_up=False):
        self.seen: list[dict] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self):
                length = int(self.headers.get("Content-Length") or 0)
                stub.seen.append({"headers": dict(self.headers), "body": self.rfile.read(length)})
                if hang_up:  # the upload arrived, then the connection died before any response
                    self.close_connection = True
                    return
                self.send_response(status)
                if location:
                    self.send_header("Location", location)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"text": "synthetic"}')

            do_GET = do_POST = _answer

            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.origin = ("http", "127.0.0.1", self.server.server_address[1])
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def _allow(monkeypatch, provider, *stubs):
    """Test-only: loopback stubs join the provider's real allowlist."""
    origins, resolve = ppr._PROVIDERS[provider]
    monkeypatch.setitem(ppr._PROVIDERS, provider, (origins | {s.origin for s in stubs}, resolve))


@pytest.fixture
def xai(monkeypatch):
    allowed, other = _Stub(), _Stub()
    _allow(monkeypatch, "xai-oauth", allowed)
    yield allowed, other
    allowed.close()
    other.close()


def _home() -> Path:
    return Path(os.environ["HERMES_HOME"])


def _sign_in(token: str):
    (_home() / "auth.json").write_text(json.dumps({"version": 1, "providers": {"xai-oauth": {"tokens": {
        "access_token": token, "refresh_token": "fake-rt",
        "expires_at": int(time.time() * 1000) + 86_400_000}}}, "credential_pool": {}}))


def _plugin(requires_auth: str = "xai-oauth"):
    """A plugin module under the active home's plugins directory, loaded from its own file."""
    root = _home() / "plugins" / "meeting"
    root.mkdir(parents=True, exist_ok=True)
    (root / "plugin.yaml").write_text("name: meeting\n" + (f"requires_auth: [{requires_auth}]\n" if requires_auth else ""))
    (root / "stt.py").write_text(
        "from hermes_cli.plugin_provider_requests import credentialed_provider_request\n"
        "def send(url, provider='xai-oauth', **kwargs):\n"
        "    return credentialed_provider_request(provider, 'POST', url, **kwargs)\n")
    spec = importlib.util.spec_from_file_location(f"plugin_stt_{abs(hash(str(root)))}", root / "stt.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _count_resolutions(monkeypatch) -> list[int]:
    from tools import xai_http

    calls: list[int] = []
    real = xai_http.resolve_xai_http_credentials
    monkeypatch.setattr(xai_http, "resolve_xai_http_credentials", lambda **kw: calls.append(1) or real(**kw))
    return calls


def _form(**files):
    return {"data": {"model": "stt-model", "diarize": "true"},
            "files": files or {"file": ("meeting.wav", b"RIFFsynthetic", "audio/wav")}}


def test_capabilities_need_no_sign_in(monkeypatch):
    from tools import xai_http

    def refuse(**_):
        raise AssertionError("capabilities must not resolve credentials")
    monkeypatch.setattr(xai_http, "resolve_xai_http_credentials", refuse)

    caps = ppr.provider_request_capabilities()
    assert caps["multipart"] is True
    assert caps["providers"] == sorted(ppr._PROVIDERS) and "xai-oauth" in caps["providers"]


def test_multipart_upload_carries_each_profiles_own_sign_in(xai, tmp_path, monkeypatch):
    allowed, _ = xai
    for profile in ("work", "home"):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / profile))
        _home().mkdir()
        _sign_in(_token(profile))
        result = _plugin().send(allowed.url + "/v1/stt", **_form())
        assert result.status == 200 and result.json() == {"text": "synthetic"}
        assert _token(profile).split(".")[1] not in repr(result)

    for seen, profile in zip(allowed.seen, ("work", "home")):
        assert seen["headers"]["Authorization"].split(".")[1] == _token(profile).split(".")[1]
        assert seen["headers"]["Content-Type"].startswith("multipart/form-data; boundary=")
        body = seen["body"]
        assert body.index(b'name="model"') < body.index(b'name="diarize"') < body.index(b'name="file"')
        assert b'filename="meeting.wav"' in body and b"RIFFsynthetic" in body


def test_undeclared_and_non_plugin_callers_are_refused_before_sign_in_lookup(xai, monkeypatch):
    allowed, _ = xai
    _sign_in(_token("user"))
    calls = _count_resolutions(monkeypatch)
    with pytest.raises(PermissionError, match=r"requires_auth: \[xai-oauth\]"):
        _plugin(requires_auth="openai-codex").send(allowed.url, **_form())
    with pytest.raises(PermissionError, match="installed plugin"):
        ppr.credentialed_provider_request("xai-oauth", "POST", allowed.url, **_form())
    assert calls == [] and allowed.seen == []


@pytest.mark.parametrize("url", [
    "http://api.x.ai/v1/stt", "https://api.x.ai:444/v1/stt", "https://api.x.ai.example.test/v1/stt",
    "https://example.test/v1/stt"])
def test_only_the_xai_origin_gets_the_token(xai, monkeypatch, url):
    _sign_in(_token("user"))
    calls = _count_resolutions(monkeypatch)
    with pytest.raises(PermissionError, match="Refusing"):
        _plugin().send(url, **_form())
    assert calls == []


@pytest.mark.parametrize("headers", [
    {"Authorization": "Bearer plugin-key"}, {"proxy-authorization": "x"}, {"Cookie": "s=1"}, {"Host": "example.test"}])
def test_plugin_cannot_bring_credentials_or_routing(xai, monkeypatch, headers):
    allowed, _ = xai
    _sign_in(_token("user"))
    calls = _count_resolutions(monkeypatch)
    with pytest.raises(PermissionError, match="Hermes sets"):
        _plugin().send(allowed.url, headers=headers, **_form())
    with pytest.raises(PermissionError, match="carries its own credentials"):
        _plugin().send(allowed.url.replace("http://", "http://user:pw@"), **_form())
    assert calls == [] and allowed.seen == []


def test_json_and_multipart_together_are_refused():
    with pytest.raises(ValueError, match="not both"):
        _plugin().send("https://api.x.ai/v1/stt", json={"a": 1}, data={"b": "2"})


def test_api_key_is_never_a_fallback_for_the_sign_in(xai, monkeypatch):
    from hermes_cli.config import invalidate_env_cache

    allowed, _ = xai
    monkeypatch.setenv("XAI_API_KEY", "paid-api-key-sentinel")
    invalidate_env_cache()
    from tools.xai_http import resolve_xai_http_credentials

    assert resolve_xai_http_credentials()["provider"] == "xai"  # core's own STT would use the API key here
    with pytest.raises(ppr.ProviderNotSignedIn, match="auth add xai-oauth") as raised:
        _plugin().send(allowed.url, **_form())
    assert "paid-api-key-sentinel" not in str(raised.value) and allowed.seen == []
    assert raised.value.__context__ is None and raised.value.__cause__ is None


def test_redirect_is_returned_not_followed(xai, monkeypatch):
    _, other = xai
    redirecting = _Stub(status=307, location=other.url + "/steal")
    _allow(monkeypatch, "xai-oauth", redirecting)
    _sign_in(_token("user"))
    try:
        result = _plugin().send(redirecting.url + "/v1/stt", **_form())
    finally:
        redirecting.close()
    assert result.status == 307 and len(redirecting.seen) == 1 and other.seen == []


@pytest.mark.parametrize("error", [httpx.ConnectError("refresh-secret-sentinel"), AttributeError("refresh-secret-sentinel")])
def test_sign_in_failure_is_typed_not_sent_and_carries_nothing(xai, monkeypatch, error):
    from tools import xai_http

    allowed, _ = xai

    def broken(**_):
        raise error
    monkeypatch.setattr(xai_http, "resolve_xai_http_credentials", broken)
    with pytest.raises(ppr.ProviderRequestNotSent, match="was not sent") as raised:
        _plugin().send(allowed.url, **_form())
    assert "refresh-secret-sentinel" not in str(raised.value) and allowed.seen == []
    assert raised.value.__context__ is None and raised.value.__cause__ is None


def test_unreachable_provider_is_typed_not_sent(monkeypatch):
    closed = _Stub()
    _allow(monkeypatch, "xai-oauth", closed)
    closed.close()
    _sign_in(_token("user"))
    with pytest.raises(ppr.ProviderRequestNotSent, match="Could not connect"):
        _plugin().send(closed.url + "/v1/stt", **_form())


def test_failure_after_upload_is_not_retried_and_not_called_unsent(monkeypatch):
    hanging = _Stub(hang_up=True)
    _allow(monkeypatch, "xai-oauth", hanging)
    _sign_in(_token("user"))
    try:
        with pytest.raises(RuntimeError, match="did not retry") as raised:
            _plugin().send(hanging.url + "/v1/stt", **_form())
    finally:
        hanging.close()
    assert not isinstance(raised.value, ppr.ProviderRequestNotSent) and len(hanging.seen) == 1
    assert _token("user").split(".")[1] not in str(raised.value)
    assert raised.value.__context__ is None and not hasattr(raised.value, "request")


def test_codex_keeps_its_original_error_and_header_contract(monkeypatch):
    allowed = _Stub()
    _allow(monkeypatch, "openai-codex", allowed)
    origins, _ = ppr._PROVIDERS["openai-codex"]
    plugin = _plugin(requires_auth="openai-codex")
    try:
        monkeypatch.setitem(ppr._PROVIDERS, "openai-codex",
                            (origins, lambda url: ({"Authorization": "Bearer core-token"}, "https://chatgpt.com")))
        plugin.send(allowed.url, provider="openai-codex", headers={"Authorization": "plugin-key"}, json={})
        assert allowed.seen[0]["headers"]["Authorization"] == "Bearer core-token"

        def failing(url):
            raise ValueError("codex detail")
        monkeypatch.setitem(ppr._PROVIDERS, "openai-codex", (origins, failing))
        with pytest.raises(ValueError, match="^codex detail$"):
            plugin.send(allowed.url, provider="openai-codex", json={})
    finally:
        allowed.close()
    monkeypatch.setitem(ppr._PROVIDERS, "openai-codex",
                        (origins, lambda url: ({"Authorization": "Bearer core-token"}, "https://chatgpt.com")))
    with pytest.raises(httpx.ConnectError):
        plugin.send(allowed.url, provider="openai-codex", json={})
