from types import SimpleNamespace

import httpx
import pytest

from agent import account_usage


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

    from agent import credential_pool

    monkeypatch.setattr(credential_pool, "load_pool", lambda provider: pool)

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert snapshot.windows[0].label == "Session"
    assert snapshot.windows[1].label == "Weekly"
    assert calls[0]["url"] == "https://chatgpt.com/backend-api/wham/usage"
    assert calls[0]["headers"]["Authorization"] == "Bearer pooled-token"
    # Pool creds have no account_id concept — the ChatGPT-Account-ID header must
    # be omitted rather than sent stale/wrong.
    assert "ChatGPT-Account-ID" not in calls[0]["headers"]




def _explicit_creds_snapshot(monkeypatch, payload):
    calls = []
    monkeypatch.setattr(account_usage.httpx, "Client", lambda timeout: _FakeClient(calls, payload))
    snapshot = account_usage.fetch_account_usage(
        "openai-codex", base_url="https://chatgpt.com/backend-api/codex", api_key="live-agent-token",
    )
    return snapshot, calls


def test_codex_weekly_only_primary_window_is_labeled_weekly(monkeypatch):
    """#65387: a lone 604800s primary_window is the weekly limit, not the session one."""
    payload = {"plan_type": "pro", "rate_limit": {
        "primary_window": {"used_percent": 1, "limit_window_seconds": 604800},
        "secondary_window": None,
    }}
    snapshot, _ = _explicit_creds_snapshot(monkeypatch, payload)
    assert [(w.label, w.used_percent) for w in snapshot.windows] == [("Weekly", 1.0)]


def test_codex_window_labels_follow_duration_with_positional_fallback(monkeypatch):
    # Swapped positions: labels must follow limit_window_seconds.
    payload = {"rate_limit": {
        "primary_window": {"used_percent": 4, "limit_window_seconds": 604800},
        "secondary_window": {"used_percent": 21, "limit_window_seconds": 18000},
    }}
    snapshot, _ = _explicit_creds_snapshot(monkeypatch, payload)
    assert [w.label for w in snapshot.windows] == ["Weekly", "Session"]
    # Missing / unrecognized durations keep the legacy positional labels.
    payload = {"rate_limit": {
        "primary_window": {"used_percent": 4},
        "secondary_window": {"used_percent": 21, "limit_window_seconds": 12345},
    }}
    snapshot, _ = _explicit_creds_snapshot(monkeypatch, payload)
    assert [w.label for w in snapshot.windows] == ["Session", "Weekly"]


def test_codex_snapshot_exposes_exact_raw_payload_with_one_get(monkeypatch, codex_usage_payload):
    """#79695: the decoded body rides along untouched (unknown fields included), from the single GET."""
    codex_usage_payload["future_field"] = {"nested": [1, 2]}
    snapshot, calls = _explicit_creds_snapshot(monkeypatch, codex_usage_payload)
    assert snapshot.raw == codex_usage_payload
    assert snapshot.raw["future_field"] == {"nested": [1, 2]}
    assert len(calls) == 1
    assert [w.label for w in snapshot.windows] == ["Session", "Weekly"]  # normalized limits unchanged
    # Additive: existing constructor calls stay valid and default to no raw body.
    assert account_usage.AccountUsageSnapshot(provider="anthropic", source="x", fetched_at=snapshot.fetched_at).raw is None


def test_codex_invalid_payload_fails_closed(monkeypatch):
    snapshot, _ = _explicit_creds_snapshot(monkeypatch, ["not", "a", "dict"])
    assert snapshot is None


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

    from agent import credential_pool

    monkeypatch.setattr(
        credential_pool,
        "load_pool",
        lambda provider: (_ for _ in ()).throw(AssertionError("pool must not be consulted")),
    )

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert calls[0]["headers"]["Authorization"] == "Bearer singleton-token"
    # account_id read failed → header omitted, but the singleton token is kept.
    assert "ChatGPT-Account-ID" not in calls[0]["headers"]


def test_codex_usage_retries_401_with_forced_refresh(monkeypatch, codex_usage_payload):
    credential_calls = []
    request_calls = []
    responses = [_FakeResponse({}, status_code=401), _FakeResponse(codex_usage_payload)]

    def resolve(**kwargs):
        credential_calls.append(kwargs)
        token = "fresh-token" if kwargs.get("force_refresh") else "revoked-token"
        return {"api_key": token, "base_url": "https://chatgpt.com/backend-api/codex"}

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, headers):
            request_calls.append(headers["Authorization"])
            return responses.pop(0)

    monkeypatch.setattr(account_usage, "resolve_codex_runtime_credentials", resolve)
    monkeypatch.setattr(account_usage, "_read_codex_tokens", lambda: {"tokens": {}})
    monkeypatch.setattr(account_usage.httpx, "Client", lambda timeout: Client())

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert snapshot.windows[0].label == "Session"
    assert credential_calls == [
        {"refresh_if_expiring": True},
        {"refresh_if_expiring": True, "force_refresh": True},
    ]
    assert request_calls == ["Bearer revoked-token", "Bearer fresh-token"]


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
    credential_calls = []
    request_calls = []
    client_count = 0
    payload = _usage_payload_with_resets(100, 40, 1)

    def resolve(base_url, api_key, *, force_refresh=False):
        credential_calls.append(force_refresh)
        token = "fresh-token" if force_refresh else "revoked-token"
        return token, "https://chatgpt.com/backend-api/codex", None

    class Client(_FakeResetClient):
        def get(self, url, headers):
            request_calls.append(("GET", headers["Authorization"]))
            if headers["Authorization"] == "Bearer revoked-token":
                return _FakeResponse({}, status_code=401)
            return _FakeResponse(payload)

        def post(self, url, headers=None, json=None):
            request_calls.append(("POST", headers["Authorization"]))
            return _FakeResponse({"code": "reset", "windows_reset": 2})

    def client_factory(timeout):
        nonlocal client_count
        client_count += 1
        return Client([], payload)

    monkeypatch.setattr(account_usage, "_resolve_codex_usage_credentials", resolve)
    monkeypatch.setattr(account_usage.httpx, "Client", client_factory)

    result = account_usage.redeem_codex_reset_credit()

    assert result.status == "reset"
    assert credential_calls == [False, True]
    assert request_calls == [
        ("GET", "Bearer revoked-token"),
        ("GET", "Bearer fresh-token"),
        ("POST", "Bearer fresh-token"),
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


def test_codex_usage_401_retry_refreshes_the_explicit_credential_not_another_account(monkeypatch, codex_usage_payload):
    """A live agent on pool entry B hands its own api_key in; after a 401 the retry must refresh B,
    not re-resolve and render the singleton/pool account A's usage."""
    request_calls = []
    refresh_hints = []
    responses = [_FakeResponse({}, status_code=401), _FakeResponse(codex_usage_payload)]

    class Pool:
        def try_refresh_matching(self, api_key_hint=None, credential_id=None):
            refresh_hints.append(api_key_hint)
            return SimpleNamespace(runtime_api_key="pool-B-fresh", runtime_base_url=None)

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, headers):
            request_calls.append(headers["Authorization"])
            return responses.pop(0)

    monkeypatch.setattr(account_usage, "resolve_codex_runtime_credentials",
                        lambda **kwargs: pytest.fail("must not re-resolve another account's credential"))
    monkeypatch.setattr(account_usage, "_read_codex_tokens", lambda: {"tokens": {"access_token": "singleton-A"}})
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda provider: Pool())
    monkeypatch.setattr(account_usage.httpx, "Client", lambda timeout: Client())

    snapshot = account_usage.fetch_account_usage(
        "openai-codex", base_url="https://chatgpt.com/backend-api/codex", api_key="pool-B-revoked")

    assert snapshot is not None
    assert refresh_hints == ["pool-B-revoked"]
    assert request_calls == ["Bearer pool-B-revoked", "Bearer pool-B-fresh"]


# ── Anthropic: 401 on an expired pool token → refresh once via the pool, then retry ──

_ANTHROPIC_PAYLOAD = {"five_hour": {"utilization": 0.5, "resets_at": "2026-10-11T23:00:00+00:00"},
                      "seven_day": {"utilization": 0.71, "resets_at": "2026-10-14T05:00:00+00:00"}}


def _http_401():
    request = httpx.Request("GET", "https://api.anthropic.com/api/oauth/usage")
    return httpx.HTTPStatusError("401", request=request, response=httpx.Response(401, request=request))


class _FakeAnthropicPool:
    def __init__(self, refreshed_token):
        self.refreshed_token = refreshed_token
        self.hints = []

    def try_refresh_matching(self, api_key_hint=None, credential_id=None):
        self.hints.append(api_key_hint)
        if self.refreshed_token is None:
            return None
        return SimpleNamespace(runtime_api_key=self.refreshed_token, access_token=self.refreshed_token)


def _patch_anthropic(monkeypatch, *, stored, pool, accept):
    seen = []

    def fake_get_json(url, headers, *, timeout):
        token = headers["Authorization"].removeprefix("Bearer ")
        seen.append(token)
        if token != accept:
            raise _http_401()
        return _ANTHROPIC_PAYLOAD

    monkeypatch.setattr(account_usage, "resolve_anthropic_token", lambda **_: stored)
    monkeypatch.setattr(account_usage, "_get_json", fake_get_json)
    monkeypatch.setattr(account_usage, "_load_anthropic_pool", lambda: pool)
    return seen


def test_anthropic_usage_refreshes_expired_pool_token_on_401(monkeypatch):
    pool = _FakeAnthropicPool("sk-ant-oat01-fresh")
    seen = _patch_anthropic(monkeypatch, stored="sk-ant-oat01-stale", pool=pool, accept="sk-ant-oat01-fresh")

    snapshot = account_usage._fetch_anthropic_account_usage()

    assert seen == ["sk-ant-oat01-stale", "sk-ant-oat01-fresh"]
    assert pool.hints == ["sk-ant-oat01-stale"]  # refresh targets exactly the entry that failed
    assert [w.label for w in snapshot.windows][:2] == ["Current session", "Current week"]
    assert snapshot.unavailable_reason is None


def test_anthropic_usage_401_without_refreshable_entry_reraises(monkeypatch):
    pool = _FakeAnthropicPool(None)
    seen = _patch_anthropic(monkeypatch, stored="sk-ant-oat01-stale", pool=pool, accept="never")

    with pytest.raises(httpx.HTTPStatusError):
        account_usage._fetch_anthropic_account_usage()
    assert seen == ["sk-ant-oat01-stale"]  # no blind retry with the same dead token


def test_anthropic_usage_retries_at_most_once(monkeypatch):
    pool = _FakeAnthropicPool("sk-ant-oat01-also-rejected")
    seen = _patch_anthropic(monkeypatch, stored="sk-ant-oat01-stale", pool=pool, accept="never")

    with pytest.raises(httpx.HTTPStatusError):
        account_usage._fetch_anthropic_account_usage()
    assert seen == ["sk-ant-oat01-stale", "sk-ant-oat01-also-rejected"]
    assert len(pool.hints) == 1


def test_anthropic_usage_explicit_key_is_never_refreshed_here(monkeypatch):
    # A live agent's explicit credential is owned by its own refresh path; usage must not rotate it.
    pool = _FakeAnthropicPool("sk-ant-oat01-fresh")
    seen = _patch_anthropic(monkeypatch, stored="unused", pool=pool, accept="never")

    with pytest.raises(httpx.HTTPStatusError):
        account_usage._fetch_anthropic_account_usage(api_key="sk-ant-oat01-explicit")
    assert seen == ["sk-ant-oat01-explicit"]
    assert pool.hints == []
