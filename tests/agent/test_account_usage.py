from types import SimpleNamespace

import pytest

from agent import account_usage


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

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


def test_anthropic_usage_prefers_explicit_active_pool_key(monkeypatch):
    # Regression for the pool-rotation bug: _fetch_anthropic_account_usage must
    # use the caller-supplied key (the active pool entry) instead of reading the
    # singleton token store, so /usage reflects the key currently in use.
    calls = []
    payload = {"five_hour": {"utilization": 0.21, "resets_at": 1779846359}}
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, payload),
    )
    monkeypatch.setattr(
        account_usage,
        "resolve_anthropic_token",
        lambda: (_ for _ in ()).throw(AssertionError("singleton token store should not be used")),
    )

    snapshot = account_usage.fetch_account_usage("anthropic", api_key="cc-live-agent-token")

    assert snapshot is not None
    assert snapshot.provider == "anthropic"
    assert calls[0]["url"] == "https://api.anthropic.com/api/oauth/usage"
    assert calls[0]["headers"]["Authorization"] == "Bearer cc-live-agent-token"
    assert snapshot.windows[0].used_percent == 21


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


def test_codex_usage_does_not_swap_to_pool_on_transient_resolver_error(monkeypatch, codex_usage_payload):
    """A transient refresh/network failure (non-AuthError) must NOT silently
    downgrade to a possibly-different pool account. It fails open (no snapshot)
    instead of reporting the wrong account's usage."""
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )
    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("refresh endpoint 503")),
    )

    pool_entry = SimpleNamespace(
        runtime_api_key="pooled-token-WRONG-ACCOUNT",
        runtime_base_url="https://chatgpt.com/backend-api/codex",
    )
    pool = SimpleNamespace(select=lambda: pool_entry)

    import agent.credential_pool as credential_pool

    # If the guard regressed, this pool would be consulted and return a snapshot
    # for the wrong account. It must NOT be.
    monkeypatch.setattr(credential_pool, "load_pool", lambda provider: pool)

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is None
    assert calls == []  # HTTP usage endpoint never hit with a wrong-account token


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


def test_codex_usage_treats_wham_used_percent_as_used_not_remaining(monkeypatch):
    """ChatGPT UI says "left"; /wham/usage.used_percent is already used."""
    payload = {
        "plan_type": "plus",
        "rate_limit": {
            "primary_window": {
                "used_percent": 85,
                "reset_at": 1779846359,
            },
            "secondary_window": {
                "used_percent": 14,
                "reset_at": 1780230796,
            },
        },
        "credits": {"has_credits": False},
    }
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, payload),
    )
    monkeypatch.setattr(
        account_usage,
        "resolve_codex_runtime_credentials",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("explicit auth should be used")),
    )

    snapshot = account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )

    assert snapshot is not None
    assert [window.used_percent for window in snapshot.windows] == [85, 14]
    rendered = "\n".join(account_usage.render_account_usage_lines(snapshot, markdown=True))
    assert "85% used" in rendered
    assert "14% used" in rendered
    assert "15% used" not in rendered
    assert "86% used" not in rendered
