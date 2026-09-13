import gzip
import http.server
import json
import threading
from types import SimpleNamespace

import pytest

from agent import account_usage


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_raw(self):
        yield json.dumps(self._payload).encode()

    def close(self):
        pass


class _FakeClient:
    def __init__(self, calls, payload):
        self.calls = calls
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def stream(self, method, url, headers=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": headers})
        return _FakeResponse(self.payload)


@pytest.fixture
def codex_usage_payload():
    return {
        "plan_type": "plus",
        "rate_limit": {
            "primary_window": {
                "used_percent": 21,
                "reset_at": 1779846359,
            },
            "secondary_window": {
                "used_percent": 4,
                "reset_at": 1780230796,
            },
        },
        "credits": {"has_credits": False},
    }


def test_codex_usage_prefers_explicit_live_agent_credentials(monkeypatch, codex_usage_payload):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )
    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy auth should not be used")),
    )

    snapshot = account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )

    assert snapshot is not None
    assert snapshot.provider == "openai-codex"
    assert snapshot.plan == "Plus"
    assert [w.label for w in snapshot.windows] == ["Session", "Weekly"]
    assert snapshot.windows[0].used_percent == 21
    assert calls[0]["url"] == "https://chatgpt.com/backend-api/wham/usage"
    assert calls[0]["headers"]["Authorization"] == "Bearer live-agent-token"
    assert calls[0]["headers"]["Accept-Encoding"] == "identity"


def test_codex_usage_falls_back_to_native_credential_pool(monkeypatch, codex_usage_payload):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )
    # Pool fallback fires only on AuthError (the documented "no creds" mode of
    # the resolver), NOT on arbitrary exceptions — see the transient-error guard
    # test below.
    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: (_ for _ in ()).throw(
            account_usage.AuthError("no singleton auth", provider="openai-codex", code="codex_auth_missing")
        ),
    )

    pool_entry = SimpleNamespace(
        runtime_api_key="pooled-token",
        runtime_base_url="https://chatgpt.com/backend-api/codex",
    )
    pool = SimpleNamespace(select=lambda: pool_entry)

    import agent.credential_pool as credential_pool

    monkeypatch.setattr(credential_pool, "load_pool", lambda provider: pool)

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert snapshot.windows[0].label == "Session"
    assert snapshot.windows[1].label == "Weekly"
    assert calls[0]["url"] == "https://chatgpt.com/backend-api/wham/usage"
    assert calls[0]["headers"]["Authorization"] == "Bearer pooled-token"
    # Pool creds have no account_id concept — the ChatGPT-Account-Id header must
    # be omitted rather than sent stale/wrong.
    assert "ChatGPT-Account-Id" not in calls[0]["headers"]




def test_codex_usage_account_id_read_failure_keeps_singleton_token(monkeypatch, codex_usage_payload):
    """When the resolver succeeds but the separate account_id read raises, the
    working singleton token must still be used (best-effort account_id), NOT
    abandoned in favor of a header-less pool credential."""
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )
    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: {
            "api_key": "singleton-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
        },
    )
    monkeypatch.setattr(
        account_usage,
        "_read_codex_tokens",
        lambda *a, **k: (_ for _ in ()).throw(
            account_usage.AuthError("partial store", provider="openai-codex", code="codex_auth_invalid_shape")
        ),
    )

    import agent.credential_pool as credential_pool

    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        lambda provider: (_ for _ in ()).throw(AssertionError("pool must not be consulted")),
    )

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert calls[0]["headers"]["Authorization"] == "Bearer singleton-token"
    # account_id read failed → header omitted, but the singleton token is kept.
    assert "ChatGPT-Account-Id" not in calls[0]["headers"]




# ── Banked rate-limit reset credits (`/usage reset`) ─────────────────────────


class _FakeResetClient:
    """GET returns the usage payload; POST returns the consume payload."""

    def __init__(self, calls, usage_payload, consume_payload=None):
        self.calls = calls
        self.usage_payload = usage_payload
        self.consume_payload = consume_payload or {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers):
        self.calls.append({"method": "GET", "url": url, "headers": headers})
        return _FakeResponse(self.usage_payload)

    def post(self, url, headers=None, json=None):
        self.calls.append({"method": "POST", "url": url, "headers": headers, "json": json})
        return _FakeResponse(self.consume_payload)


def _usage_payload_with_resets(primary_used, secondary_used, banked):
    return {
        "plan_type": "plus",
        "rate_limit": {
            "primary_window": {"used_percent": primary_used, "reset_at": 1779846359},
            "secondary_window": {"used_percent": secondary_used, "reset_at": 1780230796},
        },
        "rate_limit_reset_credits": {"available_count": banked},
        "credits": {"has_credits": False},
    }


@pytest.mark.parametrize("body_kind", ["valid", "oversize", "gzip"])
@pytest.mark.parametrize("stage", [
    "codex", "anthropic", "openrouter-credits", "openrouter-key", "reset-preflight", "reset-consume",
])
def test_usage_json_bounds_preserve_provider_and_reset_fail_open(monkeypatch, stage, body_kind):
    """Every account-usage response is bounded on the real HTTPX/socket path (#54949)."""
    payloads = {
        "/api/codex/usage": _usage_payload_with_resets(100, 25, 1),
        "/api/codex/rate-limit-reset-credits/consume": {"code": "reset", "windows_reset": 2},
        "/anthropic": {"five_hour": {"utilization": 0.25}},
        "/openrouter/credits": {"data": {"total_credits": 100, "total_usage": 25}},
        "/openrouter/key": {"data": {"limit": 100, "limit_remaining": 75}},
    }
    target = {
        "codex": "/api/codex/usage", "anthropic": "/anthropic",
        "openrouter-credits": "/openrouter/credits", "openrouter-key": "/openrouter/key",
        "reset-preflight": "/api/codex/usage",
        "reset-consume": "/api/codex/rate-limit-reset-credits/consume",
    }[stage]
    calls, responses = [], []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):  # noqa: N802
            calls.append((self.command, self.path, self.headers.get("Accept-Encoding")))
            payload = dict(payloads[self.path])
            if self.path == target and body_kind != "valid":
                payload["padding"] = "x" * 1_048_576
            body = json.dumps(payload).encode()
            encoded = self.path == target and body_kind == "gzip"
            if encoded:
                body = gzip.compress(body)
            self.send_response(200)
            if encoded:
                self.send_header("Content-Encoding", "gzip")
            # No Content-Length on oversize: the iterator, not a header, must enforce the cap.
            if body_kind != "oversize":
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self.do_GET()

    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{server.server_port}"
        real_client = account_usage.httpx.Client

        def route_anthropic(request):
            if request.url.host == "api.anthropic.com":
                request.url = account_usage.httpx.URL(base_url + "/anthropic")

        monkeypatch.setattr(account_usage.httpx, "Client", lambda **kwargs: real_client(
            **kwargs, trust_env=False,
            event_hooks={"request": [route_anthropic], "response": [responses.append]},
        ))
        monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda: "oauth-test")
        monkeypatch.setattr(account_usage, "_is_oauth_token", lambda token: True)
        monkeypatch.setattr(account_usage, "resolve_runtime_provider", lambda **kwargs: {
            "base_url": base_url + "/openrouter", "api_key": "test-token",
        })
        try:
            if stage.startswith("reset"):
                result = account_usage.redeem_codex_reset_credit(base_url=base_url, api_key="test-token")
                assert result.status == ("reset" if body_kind == "valid" else "unavailable")
                expected_methods = ["GET"] if stage == "reset-preflight" and body_kind != "valid" else ["GET", "POST"]
                assert [method for method, _, _ in calls] == expected_methods
            else:
                provider = "openrouter" if stage.startswith("openrouter") else {"codex": "openai-codex", "anthropic": "anthropic"}[stage]
                snapshot = account_usage.fetch_account_usage(provider, base_url=base_url, api_key="test-token")
                if body_kind == "valid":
                    assert snapshot is not None and snapshot.windows
                elif stage == "openrouter-key":
                    assert snapshot is not None and not snapshot.windows
                    assert "Credits balance: $75.00" in snapshot.details
                else:
                    assert snapshot is None
            assert calls and all(encoding == "identity" for _, _, encoding in calls)
            assert responses and all(response.is_closed for response in responses)
        finally:
            server.shutdown()
            thread.join(timeout=2)
















def test_redeem_missing_credentials_reports_unavailable(monkeypatch):
    monkeypatch.setattr(
        account_usage,
        "_resolve_codex_usage_credentials",
        lambda base_url, api_key: (_ for _ in ()).throw(RuntimeError("no creds")),
    )

    result = account_usage.redeem_codex_reset_credit()

    assert result.status == "unavailable"
    assert "hermes auth" in result.message
