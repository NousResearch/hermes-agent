"""Real local HTTP fixtures for doctor's per-target proxy policy (#118159)."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest


@pytest.fixture(autouse=True)
def proxy_environment(monkeypatch):
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(key, raising=False)
    for key in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(key, "127.0.0.1,localhost,::1,[::1]")
    monkeypatch.setenv("DOCTOR_FIXTURE_KEY", "fixture-not-a-secret")


@contextmanager
def endpoint(status=200):
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append((self.command, self.path, self.headers.get("Authorization")))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b'{"data": []}')

        do_POST = do_GET

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("status", [200, 401, 503])
def test_apikey_probe_reaches_local_origin_with_bracketed_no_proxy(status):
    from hermes_cli.doctor_connectivity import _probe_apikey_provider

    with endpoint(status) as (url, seen):
        result = _probe_apikey_provider("Fixture", ("DOCTOR_FIXTURE_KEY",), url + "/models", None, True)
    assert seen == [("GET", "/models", "Bearer fixture-not-a-secret")], result
    if status == 401:
        assert result.issues == ["Check DOCTOR_FIXTURE_KEY in .env"]
    else:
        assert not result.issues
    assert "Invalid port" not in str(result)


@pytest.mark.parametrize("method", ["get", "post"])
def test_unbypassed_target_uses_proxy_without_resolving_origin(monkeypatch, method):
    from hermes_cli.doctor_connectivity import _http_probe

    with endpoint() as (proxy, seen):
        monkeypatch.setenv("HTTP_PROXY", proxy)
        response = _http_probe(method, "http://doctor-fixture.invalid/models", timeout=2)
    assert response.status_code == 200
    assert seen == [(method.upper(), "http://doctor-fixture.invalid/models", None)]


def test_bypass_uses_runtime_matcher_and_preserves_environment(monkeypatch):
    import os
    from hermes_cli.doctor_connectivity import _http_probe
    from agent.process_bootstrap import _get_proxy_for_base_url

    with endpoint() as (origin, seen), endpoint(502) as (proxy, proxied):
        monkeypatch.setenv("HTTP_PROXY", proxy)
        original = os.environ["NO_PROXY"]
        assert _get_proxy_for_base_url(origin) is None
        assert _http_probe("get", origin + "/models", timeout=2).status_code == 200
        assert os.environ["NO_PROXY"] == original
        assert os.environ["no_proxy"] == original
    assert seen == [("GET", "/models", None)]
    assert proxied == []


def test_explicit_caller_proxy_and_verify_are_preserved(monkeypatch):
    from hermes_cli.doctor_connectivity import _http_probe

    monkeypatch.setenv("SSL_CERT_FILE", "nonexistent-fixture-ca.pem")
    with endpoint() as (origin, seen), endpoint(502) as (proxy, proxied):
        monkeypatch.setenv("HTTP_PROXY", proxy)
        monkeypatch.setenv("NO_PROXY", "[::1]")
        monkeypatch.setenv("no_proxy", "[::1]")
        assert _http_probe("get", origin + "/models", proxy=None, verify=False, timeout=2).status_code == 200
    assert seen
    assert proxied == []


def test_environment_ca_override_is_still_loaded(monkeypatch):
    from hermes_cli.doctor_connectivity import _http_probe

    monkeypatch.setenv("SSL_CERT_FILE", "nonexistent-fixture-ca.pem")
    with endpoint() as (origin, seen):
        with pytest.raises(FileNotFoundError):
            _http_probe("get", origin, timeout=2)
    assert seen == []


def test_probe_exception_is_reported_without_sending(monkeypatch):
    from hermes_cli.doctor_connectivity import _probe_apikey_provider

    monkeypatch.setenv("SSL_CERT_FILE", "nonexistent-fixture-ca.pem")
    with endpoint() as (origin, seen):
        result = _probe_apikey_provider("Fixture", ("DOCTOR_FIXTURE_KEY",), origin, None, True)
    assert seen == []
    assert result.lines and "No such file" in str(result)


def test_empty_selection_sends_nothing(monkeypatch):
    from hermes_cli.doctor_connectivity import _probe_apikey_provider

    monkeypatch.delenv("DOCTOR_FIXTURE_KEY")
    with endpoint() as (origin, seen):
        result = _probe_apikey_provider("Fixture", ("DOCTOR_FIXTURE_KEY",), origin, None, True)
    assert seen == []
    assert result.lines == []


def test_alibaba_fallback_resolves_policy_for_each_url(monkeypatch):
    import httpx
    from hermes_cli.doctor_connectivity import _probe_apikey_provider

    seen = []
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("NO_PROXY", "[::1],dashscope.aliyuncs.com")
    monkeypatch.setenv("no_proxy", "[::1],dashscope.aliyuncs.com")

    def get(url, **kwargs):
        seen.append((url, kwargs))
        return httpx.Response(401 if len(seen) == 1 else 200)

    monkeypatch.setattr(httpx, "get", get)
    result = _probe_apikey_provider("Alibaba/DashScope", ("DOCTOR_FIXTURE_KEY",),
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models", None, True)
    assert len(seen) == 2
    assert seen[0][1]["proxy"] == "http://127.0.0.1:7890"
    assert seen[1][1]["proxy"] is None
    assert all(not kwargs["trust_env"] for _, kwargs in seen)
    assert not result.issues
