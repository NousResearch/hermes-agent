import base64
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agent import account_usage


def _codex_token(account_id: str) -> str:
    payload = json.dumps(
        {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}},
        separators=(",", ":"),
    ).encode()
    encoded = base64.urlsafe_b64encode(payload).decode().rstrip("=")
    return f"header.{encoded}.signature"


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


class _RoutingClient:
    def __init__(self, payloads):
        self._payloads = payloads

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers):
        return _FakeResponse(self._payloads[url])


def _patch_openrouter(monkeypatch, credits_data, key_data):
    monkeypatch.setattr(
        account_usage,
        "resolve_runtime_provider",
        lambda requested, explicit_base_url=None, explicit_api_key=None: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-test",
        },
    )
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout=10.0: _RoutingClient(
            {
                "https://openrouter.ai/api/v1/credits": {"data": credits_data},
                "https://openrouter.ai/api/v1/key": {"data": key_data},
            }
        ),
    )


def _fetch_codex(monkeypatch, payload):
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient([], payload),
    )
    return account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )


_OPENROUTER_QUOTA_KEY = {
    "limit": 100.0,
    "limit_remaining": 70.0,
    "limit_reset": "monthly",
    "usage": 12.5,
    "usage_daily": 0.5,
    "usage_weekly": 2.0,
    "usage_monthly": 8.0,
    "rate_limit": {"requests": -1, "interval": "10s"},
}

_OPENROUTER_UNLIMITED_KEY = {
    "limit": None,
    "limit_remaining": None,
    "usage": 25.5,
    "usage_daily": 1.25,
    "usage_weekly": 4.5,
    "usage_monthly": 18.0,
}


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

    token = _codex_token("acct-live")
    snapshot = account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key=token,
    )

    assert snapshot is not None
    assert snapshot.provider == "openai-codex"
    assert snapshot.plan == "Plus"
    assert [w.label for w in snapshot.windows] == ["Session", "Weekly"]
    assert snapshot.windows[0].used_percent == 21
    assert calls[0]["url"] == "https://chatgpt.com/backend-api/wham/usage"
    assert calls[0]["headers"]["Authorization"] == f"Bearer {token}"
    assert calls[0]["headers"]["ChatGPT-Account-Id"] == "acct-live"


def test_codex_usage_rotated_token_changes_account_header(monkeypatch, codex_usage_payload):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )

    for account_id in ("acct-one", "acct-two"):
        assert account_usage.fetch_account_usage(
            "openai-codex",
            base_url="https://chatgpt.com/backend-api/codex",
            api_key=_codex_token(account_id),
        ) is not None

    assert [call["headers"]["ChatGPT-Account-Id"] for call in calls] == [
        "acct-one",
        "acct-two",
    ]


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


def test_codex_usage_pool_selection_never_pairs_stale_singleton_account_id(
    monkeypatch, codex_usage_payload
):
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
            "api_key": "opaque-pool-token",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "source": "credential_pool",
        },
    )
    monkeypatch.setattr(
        account_usage,
        "_read_codex_tokens",
        lambda: {"tokens": {"account_id": "acct-stale-singleton"}},
    )

    snapshot = account_usage.fetch_account_usage("openai-codex")

    assert snapshot is not None
    assert calls[0]["headers"]["Authorization"] == "Bearer opaque-pool-token"
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


@pytest.mark.parametrize("provider", ["", "auto", "custom", "gemini"])
def test_fetch_account_usage_returns_none_without_calling_fetchers(monkeypatch, provider):
    def _boom(*_args, **_kwargs):
        raise AssertionError("fetcher must not run")

    monkeypatch.setattr(account_usage, "_fetch_codex_account_usage", _boom)
    monkeypatch.setattr(account_usage, "_fetch_anthropic_account_usage", _boom)
    monkeypatch.setattr(account_usage, "_fetch_openrouter_account_usage", _boom)

    assert account_usage.fetch_account_usage(provider) is None


def test_anthropic_explicit_api_key_is_used_instead_of_resolver(monkeypatch):
    calls = []
    monkeypatch.setattr(
        account_usage,
        "resolve_anthropic_token",
        lambda: "profile-token-A",
    )
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(
            calls,
            {"five_hour": {"utilization": 0.1, "resets_at": "2026-07-16T02:00:00Z"}},
        ),
    )

    snapshot = account_usage.fetch_account_usage(
        "anthropic",
        api_key="sk-ant-oat-live-token-B",
    )

    assert snapshot is not None
    assert snapshot.available
    assert len(calls) == 1
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-ant-oat-live-token-B"
    assert "profile-token-A" not in calls[0]["headers"]["Authorization"]


def test_anthropic_without_explicit_key_uses_resolver_token(monkeypatch):
    calls = []
    monkeypatch.setattr(
        account_usage,
        "resolve_anthropic_token",
        lambda: "sk-ant-oat-profile-token-A",
    )
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(
            calls,
            {"five_hour": {"utilization": 0.2, "resets_at": "2026-07-16T02:00:00Z"}},
        ),
    )

    snapshot = account_usage.fetch_account_usage("anthropic")

    assert snapshot is not None
    assert snapshot.available
    assert len(calls) == 1
    assert calls[0]["headers"]["Authorization"] == "Bearer sk-ant-oat-profile-token-A"


def test_anthropic_plain_api_key_skips_structured_fields_and_http(monkeypatch):
    def _boom(*_args, **_kwargs):
        raise AssertionError("non-OAuth anthropic usage must return before HTTP")

    monkeypatch.setattr(account_usage, "resolve_anthropic_token", _boom)
    monkeypatch.setattr(account_usage.httpx, "Client", _boom)

    snapshot = account_usage.fetch_account_usage(
        "anthropic",
        api_key="sk-ant-api03-plain-console-key",
    )

    assert snapshot is not None
    assert snapshot.details_structured is False
    assert snapshot.rows == ()
    payload = account_usage.serialize_account_usage_snapshot(snapshot)
    assert "details_structured" not in payload
    assert "rows" not in payload


def test_serialize_account_usage_snapshot_is_json_safe(codex_usage_payload, monkeypatch):
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, codex_usage_payload),
    )

    snapshot = account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )

    assert snapshot is not None
    payload = account_usage.serialize_account_usage_snapshot(snapshot)
    assert payload["available"] is True
    assert payload["provider"] == "openai-codex"
    assert payload["plan"] == "Plus"
    assert payload["windows"][0]["used_percent"] == 21
    assert payload["windows"][0]["reset_at"].endswith("+00:00")
    assert "token" not in repr(payload).lower()


def test_serialize_account_usage_snapshot_sanitizes_non_finite_percentages():
    snapshot = account_usage.AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime.now(timezone.utc),
        windows=(
            account_usage.AccountUsageWindow(label="NaN", used_percent=float("nan")),
            account_usage.AccountUsageWindow(label="Infinite", used_percent=float("inf")),
            account_usage.AccountUsageWindow(label="Low", used_percent=-10),
            account_usage.AccountUsageWindow(label="High", used_percent=140),
        ),
    )

    payload = account_usage.serialize_account_usage_snapshot(snapshot)

    assert [window["used_percent"] for window in payload["windows"]] == [
        None,
        None,
        0.0,
        100.0,
    ]
    json.dumps(payload, allow_nan=False)


def test_openrouter_credits_balance_is_credits_minus_usage(monkeypatch):
    class _RoutingClient:
        def __init__(self, payloads):
            self._payloads = payloads

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, headers):
            return _FakeResponse(self._payloads[url])

    monkeypatch.setattr(
        account_usage,
        "resolve_runtime_provider",
        lambda requested, explicit_base_url=None, explicit_api_key=None: {
            "provider": "openrouter",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-test",
        },
    )
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout=10.0: _RoutingClient(
            {
                "https://openrouter.ai/api/v1/credits": {
                    "data": {"total_credits": 300.0, "total_usage": 10.92}
                },
                "https://openrouter.ai/api/v1/key": {"data": {}},
            }
        ),
    )

    snapshot = account_usage.fetch_account_usage("openrouter")

    assert snapshot is not None
    assert snapshot.credits_balance == 289.08
    assert snapshot.details[0] == "Credits balance: $289.08"


def test_codex_numeric_credits_balance_matches_details_string(monkeypatch, codex_usage_payload):
    payload = dict(codex_usage_payload)
    payload["credits"] = {"has_credits": True, "balance": 12.5}
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, payload),
    )

    snapshot = account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )

    assert snapshot is not None
    assert snapshot.credits_balance == 12.5
    assert "Credits balance: $12.50" in snapshot.details


def test_codex_unlimited_credits_leave_credits_balance_unset(monkeypatch, codex_usage_payload):
    payload = dict(codex_usage_payload)
    payload["credits"] = {"has_credits": True, "unlimited": True}
    calls = []
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(calls, payload),
    )

    snapshot = account_usage.fetch_account_usage(
        "openai-codex",
        base_url="https://chatgpt.com/backend-api/codex",
        api_key="live-agent-token",
    )

    assert snapshot is not None
    assert snapshot.credits_balance is None
    assert "Credits balance: unlimited" in snapshot.details


def test_render_account_usage_lines_ignores_credits_balance_field():
    shared = dict(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime(2026, 7, 16, 1, 2, 3, tzinfo=timezone.utc),
        plan="Plus",
        windows=(
            account_usage.AccountUsageWindow(label="Session", used_percent=27),
        ),
        details=("Credits balance: $12.50",),
    )
    without = account_usage.AccountUsageSnapshot(**shared)
    with_balance = account_usage.AccountUsageSnapshot(**shared, credits_balance=12.5)

    assert account_usage.render_account_usage_lines(without) == (
        account_usage.render_account_usage_lines(with_balance)
    )
    assert account_usage.render_account_usage_lines(without, markdown=True) == (
        account_usage.render_account_usage_lines(with_balance, markdown=True)
    )


def test_serialize_account_usage_snapshot_sanitizes_non_finite_credits_balance():
    snapshot = account_usage.AccountUsageSnapshot(
        provider="openrouter",
        source="credits_api",
        fetched_at=datetime.now(timezone.utc),
        credits_balance=float("inf"),
    )

    payload = account_usage.serialize_account_usage_snapshot(snapshot)

    assert payload["credits_balance"] is None
    json.dumps(payload, allow_nan=False)

    nan_snapshot = account_usage.AccountUsageSnapshot(
        provider="openrouter",
        source="credits_api",
        fetched_at=datetime.now(timezone.utc),
        credits_balance=float("nan"),
    )
    assert account_usage.serialize_account_usage_snapshot(nan_snapshot)[
        "credits_balance"
    ] is None


def test_serialize_account_usage_snapshot_sanitizes_non_finite_row_args():
    snapshot = account_usage.AccountUsageSnapshot(
        provider="openrouter",
        source="credits_api",
        fetched_at=datetime.now(timezone.utc),
        rows=(
            account_usage.AccountUsageRow(
                "credits_balance",
                {"value": float("nan"), "currency": "USD"},
            ),
            account_usage.AccountUsageRow(
                "api_key_usage",
                {"total": 1.0, "daily": float("nan"), "weekly": float("inf")},
            ),
        ),
    )

    payload = account_usage.serialize_account_usage_snapshot(snapshot)

    assert payload["rows"] == [{"key": "api_key_usage", "args": {"total": 1.0}}]
    assert "credits_balance" not in {row["key"] for row in payload["rows"]}
    json.dumps(payload, allow_nan=False)


_STRUCTURED_WINDOW_KEYS = (
    "label_key",
    "limit",
    "limit_remaining",
    "reset_interval",
)


def test_openrouter_structured_fields_cover_quota_and_usage(monkeypatch):
    _patch_openrouter(
        monkeypatch,
        {"total_credits": 300.0, "total_usage": 10.92},
        _OPENROUTER_QUOTA_KEY,
    )

    snapshot = account_usage.fetch_account_usage("openrouter")

    assert snapshot is not None
    assert snapshot.details_structured is True
    assert len(snapshot.windows) == 1
    quota = snapshot.windows[0]
    assert quota.label == "API key quota"
    assert quota.label_key == "api_key_quota"
    assert quota.limit == 100.0
    assert quota.limit_remaining == 70.0
    assert quota.reset_interval == "monthly"
    assert snapshot.rows == (
        account_usage.AccountUsageRow(
            "credits_balance",
            {"value": snapshot.credits_balance, "currency": "USD"},
        ),
        account_usage.AccountUsageRow(
            "api_key_usage",
            {"total": 12.5, "daily": 0.5, "weekly": 2.0, "monthly": 8.0},
        ),
    )
    assert snapshot.details[0] == "Credits balance: $289.08"
    assert (
        snapshot.details[1]
        == "API key usage: $12.50 total • $0.50 today • $2.00 this week • $8.00 this month"
    )
    assert len(snapshot.rows) == len(snapshot.details)

    payload = account_usage.serialize_account_usage_snapshot(snapshot)
    serialized_quota = payload["windows"][0]
    assert serialized_quota["label_key"] == "api_key_quota"
    assert serialized_quota["limit"] == 100.0
    assert serialized_quota["limit_remaining"] == 70.0
    assert serialized_quota["reset_interval"] == "monthly"
    for key in _STRUCTURED_WINDOW_KEYS:
        assert serialized_quota[key] is not None
    assert payload["details_structured"] is True
    assert payload["rows"] == [
        {
            "key": "credits_balance",
            "args": {"value": snapshot.credits_balance, "currency": "USD"},
        },
        {
            "key": "api_key_usage",
            "args": {"total": 12.5, "daily": 0.5, "weekly": 2.0, "monthly": 8.0},
        },
    ]
    json.dumps(payload, allow_nan=False)


def test_openrouter_without_quota_window_still_emits_credits_balance_row(monkeypatch):
    _patch_openrouter(
        monkeypatch,
        {"total_credits": 100.0, "total_usage": 25.5},
        _OPENROUTER_UNLIMITED_KEY,
    )

    snapshot = account_usage.fetch_account_usage("openrouter")

    assert snapshot is not None
    assert snapshot.windows == ()
    assert snapshot.details_structured is True
    assert snapshot.rows[0] == account_usage.AccountUsageRow(
        "credits_balance",
        {"value": 74.5, "currency": "USD"},
    )
    assert "credits_balance" in {row.key for row in snapshot.rows}

    payload = account_usage.serialize_account_usage_snapshot(snapshot)
    assert payload["windows"] == []
    assert "label_key" not in payload
    assert payload["rows"][0]["key"] == "credits_balance"


def test_codex_structured_fields_include_label_keys_and_banked_resets(
    monkeypatch, codex_usage_payload
):
    payload = dict(codex_usage_payload)
    payload["rate_limit_reset_credits"] = {"available_count": 3}
    payload["credits"] = {"has_credits": True, "balance": 12.5}

    snapshot = _fetch_codex(monkeypatch, payload)

    assert snapshot is not None
    assert [window.label_key for window in snapshot.windows] == ["session", "weekly"]
    assert snapshot.details_structured is True
    assert snapshot.rows[0] == account_usage.AccountUsageRow(
        "banked_resets", {"count": 3}
    )
    assert snapshot.rows[1] == account_usage.AccountUsageRow(
        "credits_balance",
        {"value": 12.5, "currency": "USD"},
    )
    assert len(snapshot.rows) == len(snapshot.details)
    assert snapshot.details[0] == (
        "You have 3 resets banked - use /usage reset to activate"
    )
    assert "Credits balance: $12.50" in snapshot.details


def test_codex_unlimited_credits_emit_credits_unlimited_row(
    monkeypatch, codex_usage_payload
):
    payload = dict(codex_usage_payload)
    payload["credits"] = {"has_credits": True, "unlimited": True}

    snapshot = _fetch_codex(monkeypatch, payload)

    assert snapshot is not None
    assert snapshot.credits_balance is None
    assert snapshot.details_structured is True
    assert snapshot.rows == (account_usage.AccountUsageRow("credits_unlimited"),)
    assert [row.key for row in snapshot.rows] == ["credits_unlimited"]
    assert "credits_balance" not in {row.key for row in snapshot.rows}
    assert "Credits balance: unlimited" in snapshot.details


def test_anthropic_structured_fields_include_label_keys_and_extra_usage(monkeypatch):
    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(
            [],
            {
                "five_hour": {
                    "utilization": 0.2,
                    "resets_at": "2026-07-16T02:00:00Z",
                },
                "seven_day": {
                    "utilization": 0.4,
                    "resets_at": "2026-07-20T02:00:00Z",
                },
                "seven_day_opus": {
                    "utilization": 0.1,
                    "resets_at": "2026-07-20T02:00:00Z",
                },
                "seven_day_sonnet": {
                    "utilization": 0.3,
                    "resets_at": "2026-07-20T02:00:00Z",
                },
                "extra_usage": {
                    "is_enabled": True,
                    "used_credits": 5.5,
                    "monthly_limit": 100.0,
                    "currency": "USD",
                },
            },
        ),
    )

    snapshot = account_usage.fetch_account_usage(
        "anthropic",
        api_key="sk-ant-oat-live-token-B",
    )

    assert snapshot is not None
    assert [window.label for window in snapshot.windows] == [
        "Current session",
        "Current week",
        "Opus week",
        "Sonnet week",
    ]
    assert [window.label_key for window in snapshot.windows] == [
        "current_session",
        "current_week",
        "opus_week",
        "sonnet_week",
    ]
    assert snapshot.details_structured is True
    assert snapshot.rows == (
        account_usage.AccountUsageRow(
            "extra_usage",
            {"used": 5.5, "limit": 100.0, "currency": "USD"},
        ),
    )
    assert snapshot.details == ("Extra usage: 5.50 / 100.00 USD",)


def test_render_account_usage_lines_ignores_structured_localization_fields():
    base = dict(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime(2026, 7, 16, 1, 2, 3, tzinfo=timezone.utc),
        plan="Plus",
        details=("Credits balance: $12.50",),
    )
    plain = account_usage.AccountUsageSnapshot(
        **base,
        windows=(
            account_usage.AccountUsageWindow(label="Session", used_percent=27),
        ),
    )
    structured = account_usage.AccountUsageSnapshot(
        **base,
        windows=(
            account_usage.AccountUsageWindow(
                label="Session",
                used_percent=27,
                label_key="session",
            ),
        ),
        rows=(
            account_usage.AccountUsageRow(
                "credits_balance",
                {"value": 12.5, "currency": "USD"},
            ),
        ),
        details_structured=True,
    )

    assert account_usage.render_account_usage_lines(plain) == (
        account_usage.render_account_usage_lines(structured)
    )
    assert account_usage.render_account_usage_lines(plain, markdown=True) == (
        account_usage.render_account_usage_lines(structured, markdown=True)
    )


def test_serialize_account_usage_snapshot_omits_empty_structured_fields():
    snapshot = account_usage.AccountUsageSnapshot(
        provider="openai-codex",
        source="usage_api",
        fetched_at=datetime(2026, 7, 16, 1, 2, 3, tzinfo=timezone.utc),
        windows=(
            account_usage.AccountUsageWindow(label="Session", used_percent=27),
        ),
    )

    payload = account_usage.serialize_account_usage_snapshot(snapshot)

    assert "rows" not in payload
    assert "details_structured" not in payload
    window = payload["windows"][0]
    for key in _STRUCTURED_WINDOW_KEYS:
        assert key not in window
    assert window == {
        "label": "Session",
        "used_percent": 27.0,
        "reset_at": None,
        "detail": None,
    }


def test_structured_details_coverage_matches_fetcher_kind(
    monkeypatch, codex_usage_payload
):
    _patch_openrouter(
        monkeypatch,
        {"total_credits": 300.0, "total_usage": 10.92},
        _OPENROUTER_QUOTA_KEY,
    )
    openrouter = account_usage.fetch_account_usage("openrouter")
    assert openrouter is not None
    assert openrouter.details_structured is True
    assert len(openrouter.rows) == len(openrouter.details)

    codex = _fetch_codex(monkeypatch, dict(codex_usage_payload))
    assert codex is not None
    assert codex.details_structured is True
    assert len(codex.rows) == len(codex.details)

    monkeypatch.setattr(
        account_usage.httpx,
        "Client",
        lambda timeout: _FakeClient(
            [],
            {
                "five_hour": {
                    "utilization": 0.2,
                    "resets_at": "2026-07-16T02:00:00Z",
                },
                "extra_usage": {
                    "is_enabled": True,
                    "used_credits": 1.0,
                    "monthly_limit": 10.0,
                    "currency": "EUR",
                },
            },
        ),
    )
    anthropic = account_usage.fetch_account_usage(
        "anthropic",
        api_key="sk-ant-oat-live-token-B",
    )
    assert anthropic is not None
    assert anthropic.details_structured is True
    assert len(anthropic.rows) == len(anthropic.details)

    from hermes_cli.nous_account import (
        NousPaidServiceAccessInfo,
        NousPortalAccountInfo,
        NousPortalSubscriptionInfo,
    )

    nous = account_usage.build_nous_credits_snapshot(
        NousPortalAccountInfo(
            logged_in=True,
            source="account_api",
            fresh=True,
            paid_service_access=True,
            paid_service_access_info=NousPaidServiceAccessInfo(
                subscription_credits_remaining=18.0,
                purchased_credits_remaining=12.34,
                total_usable_credits=30.34,
            ),
            subscription=NousPortalSubscriptionInfo(
                plan="Pro",
                current_period_end="2026-07-01",
            ),
        )
    )
    assert nous is not None
    assert nous.details_structured is False
    assert nous.rows == ()

    from agent.credits_tracker import CreditsState

    fixture = account_usage._snapshot_from_credits_state(
        CreditsState(
            from_header=True,
            remaining_micros=30_340_000,
            remaining_usd="30.34",
            subscription_micros=10_000_000,
            subscription_usd="10.00",
            subscription_limit_micros=20_000_000,
            subscription_limit_usd="20.00",
            purchased_micros=12_340_000,
            purchased_usd="12.34",
            denominator_kind="subscription_cap",
            paid_access=True,
        )
    )
    assert fixture is not None
    assert fixture.details_structured is False
    assert fixture.rows == ()


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
