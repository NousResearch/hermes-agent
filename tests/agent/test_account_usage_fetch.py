from datetime import datetime, timezone

import pytest

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


class _RecordingClient:
    def __init__(self, payload, calls):
        self._payload = payload
        self._calls = calls

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        self._calls.append((url, dict(headers or {})))
        return _Response(self._payload)


class _FailingClient:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        raise RuntimeError("connection reset")


def _stub_deepinfra(monkeypatch, payload, calls=None, client=None):
    """Point the deepinfra route at a stub client; its configured base_url carries a route suffix."""
    monkeypatch.setattr(
        "agent.account_usage.resolve_runtime_provider",
        lambda requested, explicit_base_url=None, explicit_api_key=None: {
            "provider": "deepinfra",
            "base_url": "https://api.deepinfra.com/v1/openai",
            "api_key": "test-key",
        },
    )
    captured = calls if calls is not None else []
    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=10.0: client or _RecordingClient(payload, captured),
    )
    return captured


def test_fetch_account_usage_deepinfra_renders_prepaid_credit_from_the_api_origin(monkeypatch):
    calls = _stub_deepinfra(
        monkeypatch,
        {
            "email": "dan@example.com",
            "billing_address_info": {"name": "Daniel Cheah", "line1": "1 Example Street"},
            "billing_type": "balance",
            "suspended": False,
            "stripe_balance": -5.0,
            "recent": 0.04,
            "limit": 10.0,
        },
    )

    snapshot = fetch_account_usage("deepinfra")

    assert snapshot is not None
    assert snapshot.source == "payment_checklist"
    assert snapshot.windows == (
        AccountUsageWindow(
            label="Spending limit",
            used_percent=50.0,
            detail="$5.00 of $10.00 remaining",
        ),
    )
    assert snapshot.details == ("Credits balance: $5.00", "Recent spend: $0.04")
    # The checklist endpoint hangs off the API origin, not the configured /v1/openai route suffix.
    assert calls[0][0] == "https://api.deepinfra.com/payment/checklist?compute_owed=true"
    assert calls[0][1]["Authorization"] == "Bearer test-key"
    # The payload carries billing PII; only numeric billing fields may reach the snapshot.
    assert not any("Daniel" in line or "Example" in line for line in render_account_usage_lines(snapshot))


def test_fetch_account_usage_deepinfra_flags_exhausted_and_suspended_balances(monkeypatch):
    _stub_deepinfra(
        monkeypatch, {"stripe_balance": 0.0, "recent": 0.04, "suspended": True}
    )

    snapshot = fetch_account_usage("deepinfra")

    assert snapshot is not None
    assert snapshot.details[0] == "Credits balance: $0.00"
    assert snapshot.details[-1].startswith("Status: suspended")


@pytest.mark.parametrize(
    "payload",
    [
        {"suspended": False, "billing_type": "balance"},  # no balance field at all
        {"stripe_balance": None, "recent": 0.04},  # unreadable balance
    ],
)
def test_fetch_account_usage_deepinfra_fails_soft_on_unusable_payload(monkeypatch, payload):
    _stub_deepinfra(monkeypatch, payload)

    assert fetch_account_usage("deepinfra") is None


def test_fetch_account_usage_deepinfra_fails_soft_on_transport_error(monkeypatch):
    _stub_deepinfra(monkeypatch, None, client=_FailingClient())

    assert fetch_account_usage("deepinfra") is None
