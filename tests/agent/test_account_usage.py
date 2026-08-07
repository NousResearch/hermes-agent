from types import SimpleNamespace

import pytest

from agent import account_usage
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

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
















def test_redeem_missing_credentials_reports_unavailable(monkeypatch):
    monkeypatch.setattr(
        account_usage,
        "_resolve_codex_usage_credentials",
        lambda base_url, api_key: (_ for _ in ()).throw(RuntimeError("no creds")),
    )

    result = account_usage.redeem_codex_reset_credit()

    assert result.status == "unavailable"
    assert "hermes auth" in result.message


def test_fetch_account_usage_custom_provider(monkeypatch):
    custom_payload = {
        "provider": "antigravity-proxy",
        "source": "cloudcode.fetchAvailableModels",
        "windows": [
            {
                "label": "Gemini quota",
                "remaining_fraction": 0.95,
                "used_percent": 5.0,
                "reset_at": "2026-07-26T15:26:20Z",
            }
        ],
    }
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda *a, **kw: _FakeClient(calls, custom_payload),
    )
    monkeypatch.setattr(
        account_usage,
        "_custom_usage_config_entry",
        lambda *args, **kwargs: {
            "base_url": "https://wrong.example/anthropic",
            "api_mode": "anthropic_messages",
            "extra_headers": {"X-Wrong-Tenant": "wrong"},
        },
    )

    snapshot = account_usage.fetch_account_usage(
        "custom:antigravity-proxy",
        base_url="http://127.0.0.1:8091/anthropic",
        api_key="agy-key",
        api_mode="anthropic_messages",
    )

    assert snapshot is not None
    assert snapshot.provider == "antigravity-proxy"
    assert snapshot.windows[0].label == "Gemini quota"
    assert snapshot.windows[0].used_percent == 5.0
    assert calls[0]["url"] == "http://127.0.0.1:8091/anthropic/v1/usage"
    assert calls[0]["headers"]["x-api-key"] == "agy-key"
    assert "Authorization" not in calls[0]["headers"]
    assert "X-Wrong-Tenant" not in calls[0]["headers"]


def test_fetch_account_usage_custom_provider_uses_resolved_endpoint_with_resolved_key(monkeypatch):
    """Never combine a caller URL with credentials resolved for another endpoint."""
    calls = []

    class _MultiPortClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers):
            calls.append({"url": url, "headers": headers})
            return _FakeResponse(
                {
                    "provider": "proxy-b",
                    "windows": [{"label": "Quota B", "used_percent": 10.0}],
                }
            )

    monkeypatch.setattr(account_usage.httpx, "Client", _MultiPortClient)
    monkeypatch.setattr(
        account_usage,
        "_custom_usage_config_entry",
        lambda *args, **kwargs: {
            "base_url": "https://trusted.example/anthropic",
            "api_mode": "anthropic_messages",
            "extra_headers": {"X-Proxy-Tenant": "tenant-a"},
        },
    )
    monkeypatch.setattr(
        account_usage,
        "_peek_custom_usage_api_key",
        lambda *args, **kwargs: "trusted-key",
    )

    snapshot = account_usage.fetch_account_usage(
        "custom:proxy-b",
        base_url="https://attacker.example/anthropic",
        api_key=None,
    )

    assert snapshot is not None
    assert snapshot.provider == "proxy-b"
    assert len(calls) == 1
    assert calls[0]["url"] == "https://trusted.example/anthropic/v1/usage"
    assert calls[0]["headers"] == {
        "X-Proxy-Tenant": "tenant-a",
        "x-api-key": "trusted-key",
    }


def test_fetch_account_usage_custom_openai_mode_uses_bearer_auth(monkeypatch):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda *a, **kw: _FakeClient(
            calls,
            {"provider": "proxy", "windows": [{"label": "Quota", "used_percent": 1}]},
        ),
    )
    monkeypatch.setattr(
        account_usage,
        "_custom_usage_config_entry",
        lambda *args, **kwargs: {
            "base_url": "https://proxy.example/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr(
        account_usage,
        "_peek_custom_usage_api_key",
        lambda *args, **kwargs: "proxy-key",
    )

    snapshot = account_usage.fetch_account_usage("custom:proxy")

    assert snapshot is not None
    assert calls[0]["headers"] == {"Authorization": "Bearer proxy-key"}


def test_fetch_account_usage_live_pair_keeps_matching_runtime_extra_headers(monkeypatch):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda *a, **kw: _FakeClient(
            calls,
            {"provider": "proxy", "windows": [{"label": "Quota", "used_percent": 1}]},
        ),
    )
    monkeypatch.setattr(
        account_usage,
        "_custom_usage_config_entry",
        lambda *args, **kwargs: {
            "base_url": "https://proxy.example/v1/",
            "api_mode": "chat_completions",
            "extra_headers": {"CF-Access-Client-Secret": "cf-secret"},
        },
    )

    snapshot = account_usage.fetch_account_usage(
        "custom:proxy",
        base_url="https://proxy.example/v1",
        api_key="live-pool-key",
        api_mode="chat_completions",
    )

    assert snapshot is not None
    assert calls[0]["headers"] == {
        "CF-Access-Client-Secret": "cf-secret",
        "Authorization": "Bearer live-pool-key",
    }


def test_fetch_account_usage_anthropic_bearer_endpoint_uses_main_client_auth_contract(monkeypatch):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda *a, **kw: _FakeClient(
            calls,
            {"provider": "minimax", "windows": [{"label": "Quota", "used_percent": 1}]},
        ),
    )

    snapshot = account_usage.fetch_account_usage(
        "custom:minimax",
        base_url="https://api.minimax.io/anthropic",
        api_key="minimax-key",
        api_mode="anthropic_messages",
    )

    assert snapshot is not None
    assert calls[0]["headers"] == {"Authorization": "Bearer minimax-key"}


def test_fetch_account_usage_custom_rejects_non_finite_percentages(monkeypatch):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda *a, **kw: _FakeClient(
            calls,
            {
                "provider": "malformed-proxy",
                "windows": [
                    {"label": "NaN quota", "used_percent": float("nan")},
                    {"label": "Infinite quota", "remaining_fraction": float("inf")},
                ],
            },
        ),
    )

    snapshot = account_usage.fetch_account_usage(
        "custom:malformed",
        base_url="https://malformed.example/v1",
        api_key="live-key",
    )

    assert snapshot is not None
    assert [window.used_percent for window in snapshot.windows] == [None, None]
    assert account_usage.render_account_usage_lines(snapshot)[2:] == [
        "NaN quota: unavailable",
        "Infinite quota: unavailable",
    ]


def test_fetch_account_usage_resolves_keyed_provider_key_env(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        """
providers:
  proxy-v12:
    base_url: https://trusted.example/proxy-v12?tenant=one
    api_mode: anthropic_messages
    key_env: TEST_CUSTOM_USAGE_KEY
    extra_headers:
      X-Proxy-Tenant: tenant-one
    ssl_verify: false
""".lstrip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_CUSTOM_USAGE_KEY", "v12-provider-key")
    calls = []
    client_options = []

    def fake_client(*args, **kwargs):
        client_options.append(kwargs)
        return _FakeClient(
            calls,
            {"provider": "proxy-v12", "windows": [{"label": "Quota", "used_percent": 2}]},
        )

    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        fake_client,
    )
    token = set_hermes_home_override(home)
    try:
        snapshot = account_usage.fetch_account_usage(
            "custom",
            base_url="https://trusted.example/proxy-v12?tenant=one",
            api_key=None,
        )
    finally:
        reset_hermes_home_override(token)

    assert snapshot is not None
    assert calls == [
        {
            "url": "https://trusted.example/proxy-v12/v1/usage?tenant=one",
            "headers": {
                "X-Proxy-Tenant": "tenant-one",
                "x-api-key": "v12-provider-key",
            },
        }
    ]
    assert client_options[0]["verify"] is False


def test_custom_usage_pool_lookup_peeks_without_selecting(monkeypatch):
    events = []

    class _Pool:
        def peek(self):
            events.append("peek")
            return type("Entry", (), {"runtime_api_key": "pooled-key"})()

        def select(self):
            raise AssertionError("read-only usage must not rotate the pool")

    monkeypatch.setattr(
        "agent.credential_pool.get_custom_provider_pool_key",
        lambda *args, **kwargs: "custom:proxy",
    )
    monkeypatch.setattr("agent.credential_pool.load_pool", lambda key: _Pool())

    key = account_usage._peek_custom_usage_api_key(
        {"name": "proxy"},
        "https://proxy.example/v1",
    )

    assert key == "pooled-key"
    assert events == ["peek"]
