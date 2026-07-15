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


# --- Copilot ---------------------------------------------------------------

import agent.account_usage as _au
from agent.account_usage import _build_copilot_snapshot, _extract_copilot_quota


def _reset_copilot_cache():
    _au._COPILOT_USAGE_CACHE["snap"] = None
    _au._COPILOT_USAGE_CACHE["fetched_at"] = 0.0


def test_extract_copilot_quota_flat_field_pairs():
    assert _extract_copilot_quota(
        {"chat_enabled_quota_remaining": 120, "chat_enabled_quota": 300}
    ) == (120, 300)
    assert _extract_copilot_quota(
        {"premium_interactions_remaining": 5, "premium_interactions_quota": 50}
    ) == (5, 50)
    assert _extract_copilot_quota(
        {"monthly_quota_remaining": 900, "monthly_quota": 1000}
    ) == (900, 1000)


def test_extract_copilot_quota_nested_snapshot_shape():
    data = {"quota_snapshots": {"premium_interactions": {"remaining": 42, "entitlement": 300}}}
    assert _extract_copilot_quota(data) == (42, 300)


def test_extract_copilot_quota_returns_none_when_absent():
    assert _extract_copilot_quota({"unrelated": 1}) == (None, None)


def test_build_copilot_snapshot_labels_and_ok_level():
    snap = _build_copilot_snapshot(
        {
            "access_type_sku": "copilot_pro",
            "chat_enabled_quota_remaining": 250,
            "chat_enabled_quota": 300,
            "quota_reset_date": "2026-08-01",
        }
    )
    assert snap.provider == "copilot"
    assert snap.compact_label == "quota 250/300 left · resets Aug 1"
    assert snap.compact_short_label == "q 250/300 · Aug 1"
    assert snap.compact_tiny_label == "q 250"
    assert snap.compact_level == "ok"  # 83% remaining
    assert snap.windows and snap.windows[0].label == "Premium interactions"


def test_build_copilot_snapshot_warn_and_error_levels():
    warn = _build_copilot_snapshot(
        {"premium_interactions_remaining": 60, "premium_interactions_quota": 300}
    )  # 20% remaining
    assert warn.compact_level == "warn"

    err = _build_copilot_snapshot(
        {"premium_interactions_remaining": 15, "premium_interactions_quota": 300}
    )  # 5% remaining
    assert err.compact_level == "error"


def test_build_copilot_snapshot_no_quota_yields_no_compact_label():
    snap = _build_copilot_snapshot({"access_type_sku": "copilot_pro"})
    assert snap.compact_label is None
    assert snap.compact_level is None


def test_fetch_account_usage_copilot(monkeypatch):
    _reset_copilot_cache()
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("tok-abc", "test"),
    )
    payload = {
        "access_type_sku": "copilot_pro",
        "chat_enabled_quota_remaining": 100,
        "chat_enabled_quota": 300,
    }
    monkeypatch.setattr(_au.httpx, "Client", lambda *a, **k: _Client(payload))

    snap = fetch_account_usage("copilot")
    assert snap is not None
    assert snap.provider == "copilot"
    assert snap.compact_label == "quota 100/300 left"


def test_fetch_account_usage_copilot_http_failure_returns_none(monkeypatch):
    _reset_copilot_cache()
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("tok-abc", "test"),
    )

    class _FailClient(_Client):
        def get(self, url, headers=None):
            return _Response({}, status_code=500)

    monkeypatch.setattr(_au.httpx, "Client", lambda *a, **k: _FailClient({}))
    assert fetch_account_usage("copilot") is None


def test_fetch_account_usage_copilot_no_token_returns_none(monkeypatch):
    _reset_copilot_cache()
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: (None, None),
    )
    assert fetch_account_usage("copilot") is None


def test_fetch_account_usage_copilot_caches_within_ttl(monkeypatch):
    _reset_copilot_cache()
    monkeypatch.setattr(
        "hermes_cli.copilot_auth.resolve_copilot_token",
        lambda: ("tok-abc", "test"),
    )
    calls = {"n": 0}
    payload = {"chat_enabled_quota_remaining": 100, "chat_enabled_quota": 300}

    class _CountingClient(_Client):
        def get(self, url, headers=None):
            calls["n"] += 1
            return _Response(payload)

    monkeypatch.setattr(_au.httpx, "Client", lambda *a, **k: _CountingClient(payload))

    first = fetch_account_usage("copilot")
    second = fetch_account_usage("copilot")
    assert first is second
    assert calls["n"] == 1  # second call served from cache


def test_fetch_account_usage_does_not_hit_copilot_for_other_providers(monkeypatch):
    """Provider gating: a non-copilot provider must never resolve a Copilot
    token or hit the Copilot endpoint."""
    _reset_copilot_cache()

    def _boom():
        raise AssertionError("resolve_copilot_token must not be called for non-copilot providers")

    monkeypatch.setattr("hermes_cli.copilot_auth.resolve_copilot_token", _boom)
    # Unknown/blank providers short-circuit before any copilot path.
    assert fetch_account_usage("") is None
    assert fetch_account_usage("auto") is None

