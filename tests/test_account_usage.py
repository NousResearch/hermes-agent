import json
from datetime import datetime, timezone

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


class _RecordingClient:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        self.calls.append((url, headers))
        return _Response(self._payload)


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


def test_anthropic_usage_uses_live_oauth_token_on_official_route(monkeypatch):
    client = _RecordingClient({"five_hour": {"utilization": 0.25}})
    monkeypatch.setattr(
        "agent.account_usage.resolve_anthropic_token",
        lambda: "sk-ant-profile-token",
    )
    monkeypatch.setattr("agent.account_usage.httpx.Client", lambda timeout: client)

    snapshot = fetch_account_usage(
        "anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key="sk-ant-live-oauth-token",
    )

    assert snapshot is not None
    assert snapshot.available
    assert client.calls == [
        (
            "https://api.anthropic.com/api/oauth/usage",
            {
                "Authorization": "Bearer sk-ant-live-oauth-token",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "anthropic-beta": "oauth-2025-04-20",
                "User-Agent": "claude-code/2.1.0",
            },
        )
    ]


def test_anthropic_usage_falls_back_to_profile_token_when_live_key_is_empty(monkeypatch):
    """A runtime URL without a key must not suppress the configured OAuth token."""
    client = _RecordingClient({"five_hour": {"utilization": 0.25}})
    monkeypatch.setattr(
        "agent.account_usage.resolve_anthropic_token",
        lambda: "cc-profile-oauth-token",
    )
    monkeypatch.setattr("agent.account_usage.httpx.Client", lambda timeout: client)

    snapshot = fetch_account_usage(
        "anthropic",
        base_url="https://api.anthropic.com/v1",
        api_key=None,
    )

    assert snapshot is not None
    assert snapshot.available
    assert client.calls[0][1]["Authorization"] == "Bearer cc-profile-oauth-token"


def test_anthropic_usage_omits_custom_route_instead_of_cross_hosting_key(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_anthropic_token",
        lambda: "sk-ant-profile-token",
    )

    def fail_client(*args, **kwargs):
        raise AssertionError("custom-route Anthropic quota must not make a request")

    monkeypatch.setattr("agent.account_usage.httpx.Client", fail_client)

    snapshot = fetch_account_usage(
        "anthropic",
        base_url="https://anthropic-proxy.example/v1",
        api_key="sk-ant-live-oauth-token",
    )

    assert snapshot is not None
    assert not snapshot.available
    assert "official Anthropic endpoint" in (snapshot.unavailable_reason or "")


def test_fetch_account_usage_supermemory_uses_resolved_self_hosted_endpoint(monkeypatch):
    client = _RecordingClient(
        {
            "features": {
                "usd_credits": {
                    "usage": 44.25,
                    "included_usage": 55,
                    "balance": 10.75,
                    "next_reset_at": 1785391522227,
                }
            }
        }
    )
    monkeypatch.setattr(
        "plugins.memory.supermemory.resolve_supermemory_connection_settings",
        lambda: {
            "api_key": "self-hosted-key",
            "base_url": "https://memory.internal.example/root",
            "api_timeout": 2.5,
        },
    )
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout: client,
    )

    snapshot = fetch_account_usage("supermemory")

    assert snapshot is not None
    assert snapshot.provider == "supermemory"
    assert snapshot.windows == (
        AccountUsageWindow(
            label="Supermemory credits",
            used_percent=80.45454545454545,
            reset_at=datetime.fromtimestamp(1785391522227 / 1000, tz=timezone.utc),
            detail="$10.75 of $55.00 remaining",
        ),
    )
    assert client.calls == [
        (
            "https://memory.internal.example/root/v3/auth/billing/usage",
            {"Authorization": "Bearer self-hosted-key", "Accept": "application/json"},
        )
    ]


def test_supermemory_billing_rejects_split_model_credentials_and_cloud_override(
    tmp_path,
    monkeypatch,
):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    (tmp_path / "supermemory.json").write_text(
        json.dumps({"base_url": "https://self-hosted.example/root"})
    )
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "self-hosted-key")
    client = _RecordingClient(
        {
            "features": {
                "usd_credits": {
                    "usage": 1,
                    "included_usage": 9,
                    "balance": 8,
                }
            }
        }
    )
    monkeypatch.setattr("agent.account_usage.httpx.Client", lambda timeout: client)

    token = set_hermes_home_override(str(tmp_path))
    try:
        snapshot = fetch_account_usage(
            "supermemory",
            base_url="https://api.supermemory.ai",
            api_key="model-provider-key",
        )
    finally:
        reset_hermes_home_override(token)

    assert snapshot is not None
    assert client.calls == [
        (
            "https://self-hosted.example/root/v3/auth/billing/usage",
            {"Authorization": "Bearer self-hosted-key", "Accept": "application/json"},
        )
    ]


def test_supermemory_billing_clamps_negative_remaining_credit(monkeypatch):
    """Overages are rendered as exhausted credit, never a negative remainder."""
    monkeypatch.setattr(
        "plugins.memory.supermemory.resolve_supermemory_connection_settings",
        lambda: {
            "api_key": "test-key",
            "base_url": "https://memory.example",
            "api_timeout": 2.5,
        },
    )
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout: _RecordingClient(
            {
                "features": {
                    "usd_credits": {
                        "usage": 60,
                        "included_usage": 50,
                        "balance": -10,
                    }
                }
            }
        ),
    )

    snapshot = fetch_account_usage("supermemory")

    assert snapshot is not None
    assert snapshot.windows[0].used_percent == 120
    assert snapshot.windows[0].detail == "$0.00 of $50.00 remaining"


def test_supermemory_billing_clamps_negative_balance_without_usage_window(monkeypatch):
    monkeypatch.setattr(
        "plugins.memory.supermemory.resolve_supermemory_connection_settings",
        lambda: {"api_key": "test-key", "base_url": "https://memory.example", "api_timeout": 2.5},
    )
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout: _RecordingClient({"features": {"usd_credits": {"balance": -10}}}),
    )

    snapshot = fetch_account_usage("supermemory")

    assert snapshot is not None
    assert snapshot.details == ("Supermemory credits balance: $0.00",)
