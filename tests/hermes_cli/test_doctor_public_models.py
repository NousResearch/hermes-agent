"""An HTTP 200 from a public catalog does not validate an API key."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

import pytest

from hermes_cli import doctor_connectivity as dc


@pytest.fixture
def catalog():
    calls = []
    state = {"anonymous_status": 200, "authenticated_status": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            auth = self.headers.get("Authorization")
            calls.append((self.path, auth))
            self.send_response(state["authenticated_status"] if auth else state["anonymous_status"])
            self.end_headers()
            self.wfile.write(b'{"data": []}')

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", state, calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def probe(monkeypatch, base):
    monkeypatch.setenv("SCOUT_TEST_API_KEY", "expired-fixture-key")
    return dc._probe_apikey_provider("Fixture", ("SCOUT_TEST_API_KEY",), base + "/models", None, True)


def test_public_catalog_does_not_validate_key(monkeypatch, catalog):
    base, _, calls = catalog
    result = probe(monkeypatch, base)
    assert "⚠" in result.lines[0][0]
    assert "not verified" in result.lines[0][2]
    assert calls == [("/v1/models", "Bearer expired-fixture-key"), ("/v1/models", None)]


@pytest.mark.parametrize("anonymous_status", [401, 403])
def test_authenticated_catalog_still_passes(monkeypatch, catalog, anonymous_status):
    base, state, _ = catalog
    state["anonymous_status"] = anonymous_status
    assert "✓" in probe(monkeypatch, base).lines[0][0]


def test_rejected_key_still_fails_without_baseline(monkeypatch, catalog):
    base, state, calls = catalog
    state["authenticated_status"] = 401
    result = probe(monkeypatch, base)
    assert "✗" in result.lines[0][0]
    assert result.issues
    assert len(calls) == 1


@pytest.mark.parametrize("anonymous_status", [204, 429, 500])
def test_inconclusive_baseline_never_passes(monkeypatch, catalog, anonymous_status):
    base, state, _ = catalog
    state["anonymous_status"] = anonymous_status
    assert "⚠" in probe(monkeypatch, base).lines[0][0]


def test_caller_base_override_is_used_for_both_requests(monkeypatch, catalog):
    base, _, calls = catalog
    monkeypatch.setenv("SCOUT_TEST_API_KEY", "fixture")
    monkeypatch.setenv("SCOUT_TEST_BASE_URL", base)
    result = dc._probe_apikey_provider("Fixture", ("SCOUT_TEST_API_KEY",),
                                      "http://unused.invalid/models", "SCOUT_TEST_BASE_URL", True)
    assert "⚠" in result.lines[0][0]
    assert calls == [("/v1/models", "Bearer fixture"), ("/v1/models", None)]


def test_baseline_exception_preserves_reachability_not_success(monkeypatch):
    import httpx

    def get(url, headers, timeout):
        if "Authorization" not in headers:
            raise httpx.ConnectError("fixture failure")
        return httpx.Response(200)

    monkeypatch.setattr(httpx, "get", get)
    result = probe(monkeypatch, "https://fixture.invalid/v1")
    assert "⚠" in result.lines[0][0]
    assert "reachable" in result.lines[0][2]
    assert "not verified" in result.lines[0][2]


def test_dashscope_fallback_baseline_uses_successful_url(monkeypatch):
    import httpx
    calls = []

    def get(url, headers, timeout):
        calls.append((url, headers.copy()))
        return httpx.Response(401 if len(calls) == 1 or "Authorization" not in headers else 200)

    monkeypatch.setattr(httpx, "get", get)
    monkeypatch.setenv("SCOUT_TEST_API_KEY", "fixture")
    result = dc._probe_apikey_provider("Alibaba/DashScope", ("SCOUT_TEST_API_KEY",),
                                      "https://fixture.invalid/models", None, True)
    assert "✓" in result.lines[0][0]
    assert calls[1][0] == calls[2][0]
    assert "Authorization" not in calls[2][1]


def test_google_header_is_removed_from_baseline(monkeypatch):
    import httpx
    calls = []

    def get(url, headers, timeout):
        calls.append(headers.copy())
        return httpx.Response(200 if "x-goog-api-key" in headers else 401)

    monkeypatch.setattr(httpx, "get", get)
    result = probe(monkeypatch, "https://generativelanguage.googleapis.com/v1beta")
    assert "✓" in result.lines[0][0]
    assert calls[0]["x-goog-api-key"] == "expired-fixture-key"
    assert "x-goog-api-key" not in calls[1]
    assert "Authorization" not in calls[1]


def test_missing_key_skips_all_requests(monkeypatch):
    monkeypatch.delenv("SCOUT_TEST_API_KEY", raising=False)
    result = dc._probe_apikey_provider("Fixture", ("SCOUT_TEST_API_KEY",), None, None, True)
    assert result.lines == []
