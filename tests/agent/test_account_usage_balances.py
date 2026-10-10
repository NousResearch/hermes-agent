"""Structured account balances: what a provider says is left, as money, beside the /usage text."""

from datetime import UTC, datetime

from agent.account_usage import (
    AccountBalance,
    AccountUsageSnapshot,
    AccountUsageWindow,
    fetch_account_usage,
    render_account_usage_lines,
)
from agent.account_usage_cache import cached_account_usage, remember_account_usage
from hermes_cli.subcommands.usage import usage_snapshot_document


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _RoutingClient:
    def __init__(self, payloads):
        self._payloads = payloads

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get(self, url, headers=None):
        return _Response(self._payloads[url])


def test_openrouter_credits_become_a_usd_balance_with_the_same_usage_text(monkeypatch):
    monkeypatch.setattr(
        "agent.account_usage.resolve_runtime_provider",
        lambda requested, explicit_base_url=None, explicit_api_key=None: {
            "provider": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": "sk-test"},
    )
    monkeypatch.setattr("agent.account_usage.httpx.Client", lambda timeout=10.0: _RoutingClient({
        "https://openrouter.ai/api/v1/credits": {"data": {"total_credits": 100.0, "total_usage": 25.5}},
        "https://openrouter.ai/api/v1/key": {"data": {"limit": None, "usage": 25.5}},
    }))

    snapshot = fetch_account_usage("openrouter")

    assert snapshot is not None
    assert snapshot.balances == (AccountBalance(label="Credits balance", amount=74.5, currency="USD"),)
    lines = render_account_usage_lines(snapshot)
    assert lines[2:] == ["Credits balance: $74.50", "API key usage: $25.50 total"]
    assert snapshot.available


def test_balances_render_in_their_currency_and_reach_the_json_document():
    snapshot = AccountUsageSnapshot(
        provider="deepseek", source="test", fetched_at=datetime.now(UTC), title="Account balance",
        balances=(AccountBalance("Balance", 110.0, "CNY"), AccountBalance("Balance", 2.5, "EUR")))

    assert render_account_usage_lines(snapshot)[2:] == ["Balance: ¥110.00", "Balance: 2.50 EUR"]
    assert usage_snapshot_document(snapshot)["balances"] == [
        {"label": "Balance", "amount": 110.0, "currency": "CNY"},
        {"label": "Balance", "amount": 2.5, "currency": "EUR"},
    ]


def _snapshot(**kw):
    return AccountUsageSnapshot(provider="openrouter", source="test", fetched_at=datetime.now(UTC), **kw)


def test_a_balance_only_snapshot_is_cached(monkeypatch, tmp_path):
    """A key with no spend limit reports a balance but no percentage window; it must still land."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    snapshot = _snapshot(balances=(AccountBalance(label="Credits", amount=7.21, currency="USD"),))

    remember_account_usage("openrouter", snapshot)

    assert cached_account_usage("openrouter") is snapshot


def test_an_empty_snapshot_never_replaces_a_cached_one(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kept = _snapshot(windows=(AccountUsageWindow(label="Weekly", used_percent=40.0),))
    remember_account_usage("openrouter", kept)

    remember_account_usage("openrouter", _snapshot(details=("Credits balance: $1.00",)))

    assert cached_account_usage("openrouter") is kept
