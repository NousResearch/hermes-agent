import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from agent import account_usage


def _codex_jwt(account_id, marker):
    def part(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    claims = {
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
        "marker": marker,
    }
    return f"{part({'alg': 'none'})}.{part(claims)}.sig"


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://chatgpt.com/backend-api/wham/usage")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("request failed", request=request, response=response)

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, calls, payload):
        self.calls = calls
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers):
        self.calls.append({"url": url, "headers": headers})
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




def test_codex_usage_token_without_account_claim_keeps_singleton_token(monkeypatch, codex_usage_payload):
    """A token without an account claim stays usable without an account header."""
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
    import agent.credential_pool as credential_pool

    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        lambda provider: (_ for _ in ()).throw(AssertionError("pool must not be consulted")),
    )

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert calls[0]["headers"]["Authorization"] == "Bearer singleton-token"
    # No JWT account claim → header omitted, but the singleton token is kept.
    assert "ChatGPT-Account-Id" not in calls[0]["headers"]


def test_codex_usage_retries_401_with_forced_refresh(monkeypatch, codex_usage_payload):
    request_calls = []
    responses = [_FakeResponse({}, status_code=401), _FakeResponse(codex_usage_payload)]
    stale_token = _codex_jwt("account-a", "stale")
    fresh_token = _codex_jwt("account-a", "fresh")
    refreshed = SimpleNamespace(
        runtime_api_key=fresh_token,
        runtime_base_url="https://account-a.example/backend-api/codex",
    )
    refresh_calls = []
    pool = SimpleNamespace(
        refresh_matching_api_key=lambda token: refresh_calls.append(token) or refreshed,
    )

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, headers):
            request_calls.append((headers["Authorization"], headers.get("ChatGPT-Account-Id")))
            return responses.pop(0)

    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("global account B must not be selected")),
    )
    import agent.credential_pool as credential_pool
    monkeypatch.setattr(credential_pool, "load_pool", lambda provider: pool)
    monkeypatch.setattr(account_usage.httpx, "Client", lambda timeout: Client())

    snapshot = account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://account-a.example/backend-api/codex",
        api_key=stale_token,
    )

    assert snapshot is not None
    assert snapshot.windows[0].label == "Session"
    assert refresh_calls == [stale_token]
    assert request_calls == [
        (f"Bearer {stale_token}", "account-a"),
        (f"Bearer {fresh_token}", "account-a"),
    ]


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
















def test_redeem_retries_401_with_forced_refresh(monkeypatch):
    request_calls = []
    client_count = 0
    payload = _usage_payload_with_resets(100, 40, 1)
    stale_token = _codex_jwt("account-a", "stale")
    fresh_token = _codex_jwt("account-a", "fresh")

    refreshed = SimpleNamespace(
        runtime_api_key=fresh_token,
        runtime_base_url="https://account-a.example/backend-api/codex",
    )
    refresh_calls = []
    pool = SimpleNamespace(
        refresh_matching_api_key=lambda token: refresh_calls.append(token) or refreshed,
    )

    class Client(_FakeResetClient):
        def get(self, url, headers):
            request_calls.append(("GET", headers["Authorization"], headers.get("ChatGPT-Account-Id")))
            if headers["Authorization"] == f"Bearer {stale_token}":
                return _FakeResponse({}, status_code=401)
            return _FakeResponse(payload)

        def post(self, url, headers=None, json=None):
            request_calls.append(("POST", headers["Authorization"], headers.get("ChatGPT-Account-Id")))
            return _FakeResponse({"code": "reset", "windows_reset": 2})

    def client_factory(timeout):
        nonlocal client_count
        client_count += 1
        return Client([], payload)

    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("global account B must not be selected")),
    )
    import agent.credential_pool as credential_pool
    monkeypatch.setattr(credential_pool, "load_pool", lambda provider: pool)
    monkeypatch.setattr(account_usage.httpx, "Client", client_factory)

    result = account_usage.redeem_codex_reset_credit(
        base_url="https://account-a.example/backend-api/codex",
        api_key=stale_token,
    )

    assert result.status == "reset"
    assert refresh_calls == [stale_token]
    assert request_calls == [
        ("GET", f"Bearer {stale_token}", "account-a"),
        ("GET", f"Bearer {fresh_token}", "account-a"),
        ("POST", f"Bearer {fresh_token}", "account-a"),
    ]
    assert client_count == 2


def test_redeem_missing_credentials_reports_unavailable(monkeypatch):
    monkeypatch.setattr(
        account_usage,
        "_resolve_codex_usage_credentials",
        lambda base_url, api_key, **kwargs: (_ for _ in ()).throw(RuntimeError("no creds")),
    )

    result = account_usage.redeem_codex_reset_credit()

    assert result.status == "unavailable"
    assert "hermes auth" in result.message
