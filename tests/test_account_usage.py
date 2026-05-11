import base64
import json
import time
from datetime import datetime, timedelta, timezone

from agent.account_usage import (
    AccountUsageSnapshot,
    AccountUsageWindow,
    ProviderAccountUsage,
    active_provider_account_index,
    compact_account_usage_description,
    discord_account_option_parts,
    fetch_account_usage,
    fetch_provider_account_usages,
    format_reset_countdown,
    list_account_usage_providers,
    render_account_usage_lines,
    render_provider_account_usage_lines,
    safe_display_text,
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


class _HeaderRoutingClient:
    def __init__(self, payloads):
        self._payloads = payloads
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url, headers=None):
        headers = headers or {}
        self.calls.append((url, dict(headers)))
        return _Response(self._payloads[headers.get("ChatGPT-Account-Id")])


class _Entry:
    def __init__(
        self,
        *,
        label,
        token,
        base_url="https://chatgpt.com/backend-api/codex",
        source="manual",
        id_token=None,
        account_id=None,
    ):
        self.label = label
        self.source = source
        self.auth_type = "oauth"
        self.access_token = token
        self.runtime_api_key = token
        self.runtime_base_url = base_url
        self.id_token = id_token
        self.account_id = account_id
        self.last_status = None
        self.last_error_reset_at = None


class _Pool:
    def __init__(self, entries):
        self._entries = entries

    def entries(self):
        return list(self._entries)

    def current(self):
        return None


def _jwt_with_account(account_id):
    payload = {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"hdr.{body}.sig"


def test_active_provider_account_index_defaults_to_priority_first_without_current_id():
    first = _Entry(label="first", token="stub")
    first.id = "cred-1"
    second = _Entry(label="second", token="stub")
    second.id = "cred-2"

    assert active_provider_account_index("openai-codex", pool=_Pool([first, second])) == 1


def test_list_account_usage_providers_reads_credential_pool(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.read_credential_pool",
        lambda provider_id=None: {
            "openai-codex": [{"label": "codex-a"}],
            "anthropic": [{"label": "claude-a"}],
            "custom:local": [{"label": "local"}],
            "openrouter": [],
        } if provider_id is None else [],
    )

    assert list_account_usage_providers() == ["anthropic", "custom:local", "openai-codex"]


def test_fetch_provider_account_usages_codex_fetches_every_pool_entry(monkeypatch):
    entries = [
        _Entry(label="alpha", token=_jwt_with_account("acct_alpha")),
        _Entry(label="beta", token=_jwt_with_account("acct_beta")),
    ]
    client = _HeaderRoutingClient(
        {
            "acct_alpha": {
                "plan_type": "pro",
                "rate_limit": {
                    "primary_window": {"used_percent": 10, "reset_at": 1_900_000_000},
                    "secondary_window": {"used_percent": 20, "reset_at": 1_900_500_000},
                },
            },
            "acct_beta": {
                "plan_type": "plus",
                "rate_limit": {
                    "primary_window": {"used_percent": 95, "reset_at": 1_900_000_000},
                    "secondary_window": {"used_percent": 80, "reset_at": 1_900_500_000},
                },
            },
        }
    )
    monkeypatch.setattr("agent.account_usage.load_pool", lambda provider: _Pool(entries))
    monkeypatch.setattr("agent.account_usage.httpx.Client", lambda timeout=15.0: client)

    results = fetch_provider_account_usages("openai-codex")

    assert [result.label for result in results] == ["alpha", "beta"]
    assert [result.snapshot.plan for result in results] == ["Pro", "Plus"]
    assert [result.snapshot.windows[0].used_percent for result in results] == [10.0, 95.0]
    assert [call[1]["ChatGPT-Account-Id"] for call in client.calls] == ["acct_alpha", "acct_beta"]

    rendered = "\n".join(render_provider_account_usage_lines("openai-codex", results))
    assert "Provider: openai-codex" in rendered
    assert "1. alpha" in rendered
    assert "Session: 90% left" in rendered
    assert "2. beta" in rendered
    assert "Session: 5% left" in rendered


def test_fetch_provider_account_usages_codex_uses_id_token_account_id_and_chatgpt_backend_url(monkeypatch):
    entry = _Entry(
        label="id-token-account",
        token="stub",
        id_token=_jwt_with_account("acct_from_id_token"),
        base_url="https://chatgpt.com",
    )
    client = _HeaderRoutingClient(
        {
            "acct_from_id_token": {
                "plan_type": "pro",
                "rate_limit": {"primary_window": {"used_percent": 50}},
            },
        }
    )
    monkeypatch.setattr("agent.account_usage.load_pool", lambda provider: _Pool([entry]))
    monkeypatch.setattr("agent.account_usage.httpx.Client", lambda timeout=15.0: client)

    results = fetch_provider_account_usages("openai-codex")

    assert results[0].snapshot.windows[0].used_percent == 50.0
    assert client.calls[0][0] == "https://chatgpt.com/backend-api/wham/usage"
    assert client.calls[0][1]["ChatGPT-Account-Id"] == "acct_from_id_token"


def test_codex_usage_url_normalizes_chatgpt_and_codex_api_hosts():
    from agent.account_usage import _resolve_codex_usage_url

    assert _resolve_codex_usage_url("https://chatgpt.com") == "https://chatgpt.com/backend-api/wham/usage"
    assert _resolve_codex_usage_url("https://chatgpt.com/backend-api/codex") == "https://chatgpt.com/backend-api/wham/usage"
    assert _resolve_codex_usage_url("https://codex.openai.com/api/codex") == "https://codex.openai.com/api/codex/usage"


def test_render_provider_account_usage_lines_flags_limited_accounts_with_selection_hint():
    results = [
        ProviderAccountUsage(
            provider="openai-codex",
            index=1,
            label="main",
            status="ok",
            snapshot=AccountUsageSnapshot(
                provider="openai-codex",
                source="usage_api",
                fetched_at=datetime.now(timezone.utc),
                plan="Pro",
                windows=(AccountUsageWindow(label="Session", used_percent=98),),
            ),
        ),
        ProviderAccountUsage(
            provider="openai-codex",
            index=2,
            label="backup",
            status="ok",
            snapshot=AccountUsageSnapshot(
                provider="openai-codex",
                source="usage_api",
                fetched_at=datetime.now(timezone.utc),
                plan="Plus",
                windows=(AccountUsageWindow(label="Session", used_percent=20),),
            ),
        ),
    ]

    rendered = "\n".join(
        render_provider_account_usage_lines(
            "openai-codex",
            results,
            markdown=True,
            active_index=2,
            select_hint="/account openai-codex 1",
        )
    )

    assert "Hermes Account Center" in rendered
    assert "Limited" in rendered
    assert "✅" in rendered
    assert "active" in rendered.lower()
    assert "▰" in rendered or "█" in rendered
    assert "/account openai-codex 1" in rendered


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
    assert "Session: 75% left" in lines[2]
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



def test_format_reset_countdown_is_friendly_and_deterministic():
    now = datetime(2026, 5, 5, 8, 0, tzinfo=timezone.utc)

    assert format_reset_countdown(now + timedelta(minutes=32), now=now) == "renews in 32m ⏳"
    assert format_reset_countdown(now + timedelta(hours=2, minutes=14), now=now) == "renews in 2h 14m ✨"
    assert format_reset_countdown(datetime(2026, 5, 6, 9, 0, tzinfo=timezone.utc), now=now) == "renews tomorrow at 09:00 🌙"
    assert format_reset_countdown(now + timedelta(days=3, hours=4), now=now) == "renews in 3d 4h 🗓️"
    assert format_reset_countdown(now - timedelta(seconds=1), now=now) == "renews now ✨"


def test_discord_compact_description_uses_snapshot_windows():
    snapshot = AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime.now(timezone.utc),
        windows=(
            AccountUsageWindow(label="Session", used_percent=10),
            AccountUsageWindow(label="Weekly", used_percent=20),
        ),
    )
    result = ProviderAccountUsage("openai-codex", 1, "main", "ok", snapshot)

    label, desc = discord_account_option_parts(result, active=True)

    assert label == "✓ 1. main · Healthy"
    assert desc == "Session: 90% left · Weekly: 80% left"
    assert compact_account_usage_description(snapshot) == desc


def test_markdown_sensitive_labels_are_sanitized_for_display():
    result = ProviderAccountUsage(
        provider="openai-codex",
        index=1,
        label="main_key_*[x]`",
        status="ok_*[x]`",
        snapshot=AccountUsageSnapshot(
            provider="openai-codex",
            source="usage_api",
            fetched_at=datetime.now(timezone.utc),
            plan="pro_*[x]`",
            windows=(AccountUsageWindow(label="Session_*[x]`", used_percent=10),),
        ),
    )

    rendered = "\n".join(render_provider_account_usage_lines("openai-codex", [result], markdown=True))

    assert r"main\_key\_\*\[x\]\`" in rendered
    assert r"pro\_\*\[x\]\`" in rendered
    assert safe_display_text("main_key_*[x]`", markdown=True) == r"main\_key\_\*\[x\]\`"


def test_fetch_provider_account_usages_preserves_order_when_one_entry_fails(monkeypatch):
    entries = [
        _Entry(label="alpha", token="alpha"),
        _Entry(label="beta", token="beta"),
        _Entry(label="gamma", token="gamma"),
    ]
    monkeypatch.setattr("agent.account_usage.load_pool", lambda provider: _Pool(entries))

    def fake_fetch(provider, index, entry, pool=None):
        if index == 2:
            raise RuntimeError("boom")
        return ProviderAccountUsage(
            provider,
            index,
            entry.label,
            "available",
            AccountUsageSnapshot(
                provider=provider,
                source="test",
                fetched_at=datetime.now(timezone.utc),
                windows=(AccountUsageWindow(label="Session", used_percent=index),),
            ),
        )

    monkeypatch.setattr("agent.account_usage._fetch_provider_entry_usage", fake_fetch)

    results = fetch_provider_account_usages("openai-codex")

    assert [result.label for result in results] == ["alpha", "beta", "gamma"]
    assert results[1].snapshot.unavailable_reason == "usage lookup failed: RuntimeError"


def test_fetch_provider_account_usages_timeout_returns_unavailable_snapshot(monkeypatch):
    entries = [_Entry(label="slow", token="slow")]
    monkeypatch.setattr("agent.account_usage.load_pool", lambda provider: _Pool(entries))

    def slow_fetch(provider, index, entry, pool=None):
        time.sleep(0.2)
        return ProviderAccountUsage(provider, index, entry.label, "available")

    monkeypatch.setattr("agent.account_usage._fetch_provider_entry_usage", slow_fetch)

    results = fetch_provider_account_usages("openai-codex", timeout=0.01)

    assert results[0].label == "slow"
    assert results[0].snapshot.unavailable_reason == "usage lookup timed out"


def test_fetch_provider_account_usages_unsupported_provider_is_informational(monkeypatch):
    entries = [_Entry(label="local", token="token")]
    monkeypatch.setattr("agent.account_usage.load_pool", lambda provider: _Pool(entries))

    results = fetch_provider_account_usages("custom:local")

    assert results[0].snapshot.unavailable_reason == "usage lookup not supported for this provider"


def test_fetch_provider_account_usages_syncs_codex_entry_before_lookup(monkeypatch):
    stale = _Entry(label="codex", token="stale", source="device_code")
    fresh = _Entry(label="codex", token=_jwt_with_account("acct_fresh"), source="device_code")

    class SyncPool(_Pool):
        def _sync_codex_entry_from_auth_store(self, entry):
            return fresh

    client = _HeaderRoutingClient(
        {
            "acct_fresh": {
                "plan_type": "pro",
                "rate_limit": {"primary_window": {"used_percent": 1}},
            },
        }
    )
    monkeypatch.setattr("agent.account_usage.load_pool", lambda provider: SyncPool([stale]))
    monkeypatch.setattr("agent.account_usage.httpx.Client", lambda timeout=15.0: client)

    results = fetch_provider_account_usages("openai-codex")
    assert results[0].snapshot.windows[0].used_percent == 1.0
    assert client.calls[0][1]["ChatGPT-Account-Id"] == "acct_fresh"
