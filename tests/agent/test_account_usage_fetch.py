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


def test_fetch_account_usage_xai_oauth_reads_credits_config(monkeypatch):
    """SuperGrok consumer quota: weekly % + reset from the Grok CLI billing proxy, keyed by x-userid-free
    bearer auth against the xai-oauth grant — not the api.x.ai API key."""
    captured_headers = {}

    class _HeaderClient:
        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, headers=None):
            captured_headers.update(headers or {})
            assert "billing?format=credits" in url
            return _Response({
                "config": {
                    "creditUsagePercent": 4.0,
                    "currentPeriod": {
                        "type": "USAGE_PERIOD_TYPE_WEEKLY",
                        "start": "2026-09-17T10:13:12+00:00",
                        "end": "2026-09-24T10:13:12+00:00",
                    },
                    "onDemandCap": {"val": 0},
                    "onDemandUsed": {"val": 0},
                    "prepaidBalance": {"val": 8666},
                },
                "subscriptionTier": "SuperGrok",
            })

    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=15.0, follow_redirects=False: _HeaderClient(None),
    )
    import agent.account_usage as au

    def _fake_resolve(**kwargs):
        return {"provider": "xai-oauth", "api_key": "oauth-access-token"}

    monkeypatch.setattr("hermes_cli.auth_xai.resolve_xai_oauth_runtime_credentials", _fake_resolve)

    snapshot = fetch_account_usage("xai-oauth", api_key="sk-plain-api-key")

    assert snapshot is not None
    assert snapshot.provider == "xai-oauth"
    assert snapshot.source == "grok-billing-proxy"
    assert snapshot.plan == "SuperGrok"
    assert len(snapshot.windows) == 1
    assert snapshot.windows[0].label == "Weekly limit"
    assert snapshot.windows[0].used_percent == 4.0
    assert snapshot.windows[0].reset_at == datetime(2026, 9, 24, 10, 13, 12, tzinfo=timezone.utc)
    assert "Prepaid credits: $86.66" in snapshot.details
    assert captured_headers.get("X-XAI-Token-Auth") == "xai-grok-cli"
    assert captured_headers.get("Authorization") == "Bearer oauth-access-token"


def test_fetch_account_usage_xai_oauth_fails_open(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.auth_xai.resolve_xai_oauth_runtime_credentials",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("no store")),
    )
    assert fetch_account_usage("xai-oauth") is None


def test_fetch_account_usage_xai_oauth_reports_missing_config(monkeypatch):
    class _EmptyClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, headers=None):
            return _Response({})

    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=15.0, follow_redirects=False: _EmptyClient(),
    )
    monkeypatch.setattr(
        "hermes_cli.auth_xai.resolve_xai_oauth_runtime_credentials",
        lambda **kw: {"api_key": "tok"},
    )
    snapshot = fetch_account_usage("xai")

    assert snapshot is not None
    assert snapshot.windows == ()
    assert snapshot.unavailable_reason is not None


def test_fetch_account_usage_nano_gpt_scales_fraction_and_reads_balance(monkeypatch):
    """NanoGPT: percentUsed is a 0–1 fraction; balance comes from POST check-balance."""
    class _RoutingClient:
        def __init__(self, sub, balance):
            self._sub, self._balance = sub, balance

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, headers=None):
            assert "subscription/usage" in url
            return _Response(self._sub)

        def post(self, url, headers=None, json=None):
            assert "check-balance" in url
            return _Response(self._balance)

    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=15.0, follow_redirects=False: _RoutingClient(
            {
                "active": True,
                "allowOverage": True,
                "weeklyInputTokens": {"used": 56_058_595, "remaining": 3_941_405,
                                      "percentUsed": 0.9343099166666666, "resetAt": 1_789_948_800_000},
                "dailyImages": {"used": 0, "remaining": 100, "percentUsed": 0, "resetAt": 1_789_776_000_000},
            },
            {"usd_balance": "27.29969381"},
        ),
    )

    snapshot = fetch_account_usage("nano-gpt", api_key="sk-nano-test")

    assert snapshot is not None
    assert snapshot.provider == "nano-gpt"
    assert len(snapshot.windows) == 2
    assert snapshot.windows[0].label == "Weekly token limit"
    assert snapshot.windows[0].used_percent > 90  # fraction scaled, NOT 0.93%
    assert snapshot.windows[0].reset_at == datetime.fromtimestamp(1_789_948_800, tz=timezone.utc)
    assert "Balance: $27.30" in snapshot.details


def test_fetch_account_usage_ollama_cloud_scales_fractions(monkeypatch):
    """Ollama Cloud: limits.*.usage is a 0–1 fraction; spend + top model from activity."""

    class _Client:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, headers=None):
            assert url == "https://ollama.com/api/usage"
            return _Response({
                "activity": {"cost": "2.01533", "models": [
                    {"name": "glm-5.3", "request_count": 52, "cost": "1.74216"},
                    {"name": "glm-5.3-flash", "request_count": 15, "cost": "0.27317"},
                ]},
                "limits": {
                    "session": {"usage": 0.208, "models": []},
                    "weekly": {"usage": 0.631, "models": []},
                },
            })

    monkeypatch.setattr(
        "agent.account_usage.httpx.Client",
        lambda timeout=15.0, follow_redirects=False: _Client(),
    )

    snapshot = fetch_account_usage("ollama-cloud", api_key="key")

    assert snapshot is not None
    assert snapshot.provider == "ollama-cloud"
    assert [w.label for w in snapshot.windows] == ["Session usage", "Weekly usage"]
    assert snapshot.windows[0].used_percent == 20.8
    assert snapshot.windows[1].used_percent == 63.1
    assert "Spend, last 4 weeks: $2.02" in snapshot.details
    assert "Top model: glm-5.3 at $1.74" in snapshot.details


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
