from datetime import datetime, timezone

from agent import account_usage
from agent.account_usage import (
    AccountUsageSnapshot,
    AccountUsageWindow,
    fetch_account_usage,
    render_account_usage_lines,
)


class _Response:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class _Client:
    def __init__(self, payload):
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        return _Response(self._payload)


class _RoutingClient:
    def __init__(self, payloads):
        self._payloads = payloads

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        return _Response(self._payloads[url])


def test_fetch_account_usage_codex(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_codex_runtime_credentials",
        lambda refresh_if_expiring=True: {
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "access-token",
        },
    )
    monkeypatch.setattr(
        "agent.account_usage._read_codex_tokens",
        lambda: {"tokens": {"account_id": "acct_123"}},
    )
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=15.0: _Client(
            {
                "plan_type": "pro",
                "rate_limit": {
                    "primary_window": {
                        "used_percent": 15,
                        "reset_at": 1_900_000_000,
                        "limit_window_seconds": 18000,
                    },
                    "secondary_window": {
                        "used_percent": 40,
                        "reset_at": 1_900_500_000,
                        "limit_window_seconds": 604800,
                    },
                },
                "credits": {"has_credits": True, "balance": 12.5},
            }
        ),
    )

    snapshot = fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert snapshot.plan == "Pro"
    assert len(snapshot.windows) == 2
    assert snapshot.windows[0].label == "Session"
    assert snapshot.windows[0].used_percent == 15.0
    assert snapshot.windows[0].reset_at == datetime.fromtimestamp(1_900_000_000, tz=timezone.utc)
    assert "Credits balance: $12.50" in snapshot.details


def test_render_account_usage_lines_includes_reset_and_provider():
    snapshot = AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime.now(timezone.utc),
        plan="Pro",
        windows=(
            AccountUsageWindow(
                label="Session",
                used_percent=25,
                reset_at=datetime.now(timezone.utc),
            ),
        ),
        details=("Credits balance: $9.99",),
    )
    lines = render_account_usage_lines(snapshot)

    assert lines[0] == "📈 Account limits"
    assert "openai-codex (Pro)" in lines[1]
    assert "Session: 75% remaining (25% used)" in lines[2]
    assert "Credits balance: $9.99" in lines[3]


def test_fetch_account_usage_openrouter_uses_limit_remaining_and_ignores_deprecated_rate_limit(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_runtime_provider",
        lambda requested, explicit_base_url=None, explicit_api_key=None: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-test",
        },
    )
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=10.0: _RoutingClient(
            {
                "https://openrouter.ai/api/v1/credits": {
                    "data": {"total_credits": 300.0, "total_usage": 10.92}
                },
                "https://openrouter.ai/api/v1/key": {
                    "data": {
                        "limit": 100.0,
                        "limit_remaining": 70.0,
                        "limit_reset": "monthly",
                        "usage": 12.5,
                        "usage_daily": 0.5,
                        "usage_weekly": 2.0,
                        "usage_monthly": 8.0,
                        "rate_limit": {"requests": -1, "interval": "10s"},
                    }
                },
            }
        ),
    )

    snapshot = fetch_account_usage("openrouter")

    assert snapshot is not None
    assert snapshot.windows == (
        AccountUsageWindow(
            label="API key quota",
            used_percent=30.0,
            detail="$70.00 of $100.00 remaining • resets monthly",
        ),
    )
    assert "Credits balance: $289.08" in snapshot.details
    assert "API key usage: $12.50 total • $0.50 today • $2.00 this week • $8.00 this month" in snapshot.details
    assert all("-1 requests / 10s" not in line for line in render_account_usage_lines(snapshot))


def test_fetch_account_usage_openrouter_omits_quota_window_when_key_has_no_limit(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_runtime_provider",
        lambda requested, explicit_base_url=None, explicit_api_key=None: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-test",
        },
    )
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=10.0: _RoutingClient(
            {
                "https://openrouter.ai/api/v1/credits": {
                    "data": {"total_credits": 100.0, "total_usage": 25.5}
                },
                "https://openrouter.ai/api/v1/key": {
                    "data": {
                        "limit": None,
                        "limit_remaining": None,
                        "usage": 25.5,
                        "usage_daily": 1.25,
                        "usage_weekly": 4.5,
                        "usage_monthly": 18.0,
                    }
                },
            }
        ),
    )

    snapshot = fetch_account_usage("openrouter")

    assert snapshot is not None
    assert snapshot.windows == ()
    assert "Credits balance: $74.50" in snapshot.details
    assert "API key usage: $25.50 total • $1.25 today • $4.50 this week • $18.00 this month" in snapshot.details


def test_fetch_all_providers_quota_discovers_openrouter_without_env_key(monkeypatch):
    """Aggregate discovery must delegate to runtime resolution, not env gates."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    calls = []
    openrouter_snapshot = AccountUsageSnapshot(
        provider="openrouter",
        source="credits_api",
        fetched_at=datetime.now(timezone.utc),
        details=("Credits balance: $12.00",),
    )

    def fake_fetch(provider):
        calls.append(provider)
        return openrouter_snapshot if provider == "openrouter" else None

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)

    snapshots = account_usage.fetch_all_providers_quota()

    assert calls == ["openai-codex", "anthropic", "openrouter", "deepseek"]
    assert snapshots == [openrouter_snapshot]


def test_fetch_account_usage_deepseek_uses_runtime_credentials_without_env_key(monkeypatch):
    """DeepSeek quota discovery must use the runtime resolver like OpenRouter."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    resolver_calls = []

    def resolve_runtime(requested, explicit_base_url=None, explicit_api_key=None):
        resolver_calls.append((requested, explicit_base_url, explicit_api_key))
        return {"api_key": "runtime-token"}

    monkeypatch.setattr(account_usage, "resolve_runtime_provider", resolve_runtime)
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout=10.0: _Client(
            {
                "is_available": True,
                "balance_infos": [{"currency": "USD", "total_balance": "12.34"}],
            }
        ),
    )

    snapshot = fetch_account_usage("deepseek")

    assert resolver_calls == [("deepseek", None, None)]
    assert snapshot is not None
    assert snapshot.provider == "deepseek"
    assert snapshot.details == ("Balance (USD): $12.34",)


def test_fetch_account_usage_deepseek_derives_balance_endpoint_from_runtime_base_url(monkeypatch):
    """A configured DeepSeek-compatible endpoint is used instead of the default host."""
    requested_urls = []

    class CapturingClient(_Client):
        def get(self, url, headers=None):
            requested_urls.append(url)
            return super().get(url, headers=headers)

    monkeypatch.setattr(
        account_usage,
        "resolve_runtime_provider",
        lambda **kwargs: {
            "api_key": "runtime-token",
            "base_url": "https://deepseek-proxy.example/v1/",
        },
    )
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout=10.0: CapturingClient(
            {"is_available": True, "balance_infos": [{"currency": "USD", "total_balance": "12.34"}]}
        ),
    )

    snapshot = fetch_account_usage("deepseek")

    assert snapshot is not None
    assert requested_urls == ["https://deepseek-proxy.example/user/balance"]


def test_fetch_all_providers_quota_continues_after_a_provider_failure(monkeypatch):
    """One failed lookup must not hide quota data returned by later providers."""
    openrouter_snapshot = AccountUsageSnapshot(
        provider="openrouter",
        source="credits_api",
        fetched_at=datetime.now(timezone.utc),
        details=("Credits balance: $12.00",),
    )

    def fake_fetch(provider):
        if provider == "openai-codex":
            raise RuntimeError("transient resolver failure")
        return openrouter_snapshot if provider == "openrouter" else None

    monkeypatch.setattr(account_usage, "fetch_account_usage", fake_fetch)

    assert account_usage.fetch_all_providers_quota() == [openrouter_snapshot]
