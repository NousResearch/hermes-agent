"""Balance beside each month row: from the account-usage cache, waiting briefly only when it is cold."""

import threading
import time
from datetime import UTC, datetime

import pytest

from agent.account_usage import AccountBalance, AccountUsageSnapshot, AccountUsageWindow
from hermes_cli.usage_balance import attach_balances


@pytest.fixture(autouse=True)
def _isolated_home(monkeypatch, tmp_path):
    # The cache and its refresh throttle are keyed by profile home: a fresh home is a cold cache.
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _fetchers(monkeypatch, **fetchers):
    monkeypatch.setattr("agent.account_usage._USAGE_FETCHERS", fetchers)


def _snapshot(provider, **kw):
    return AccountUsageSnapshot(provider=provider, source="test", fetched_at=datetime.now(UTC), **kw)


def _balances(*providers, wait_s=1.0):
    rows = [{"provider": p} for p in providers]
    attach_balances(rows, wait_s=wait_s)
    return {row["provider"]: row["balance"] for row in rows}


def test_a_cold_provider_reports_its_balance_on_the_first_call(monkeypatch):
    _fetchers(monkeypatch, credits=lambda base_url, api_key: _snapshot(
        "credits", balances=(AccountBalance("Credits balance", 7.21, "USD"),)))

    balance = _balances("credits")["credits"]

    assert balance["state"] == "ready"
    assert balance["amounts"] == [{"label": "Credits balance", "amount": 7.21, "currency": "USD"}]
    assert datetime.fromisoformat(balance["fetched_at"]).tzinfo is not None


def test_providers_without_a_money_balance_say_so(monkeypatch):
    """No usage source at all, or one that reports only percentage windows: there is no balance."""
    _fetchers(monkeypatch, windows=lambda base_url, api_key: _snapshot(
        "windows", windows=(AccountUsageWindow(label="Weekly", used_percent=30.0),)))

    assert _balances("windows", "no-usage-api") == {"windows": None, "no-usage-api": None}


def test_a_failed_fetch_is_unknown_not_zero(monkeypatch):
    def broken(base_url, api_key):
        raise RuntimeError("HTTP 401")

    _fetchers(monkeypatch, broken=broken)

    assert _balances("broken") == {"broken": {"state": "unknown"}}


def test_a_slow_balance_api_never_holds_the_month_past_the_wait(monkeypatch):
    release = threading.Event()

    def slow(base_url, api_key):
        release.wait(5)
        return _snapshot("slow", balances=(AccountBalance("Balance", 1.0, "USD"),))

    _fetchers(monkeypatch, slow=slow)
    started = time.monotonic()
    try:
        assert _balances("slow", wait_s=0.05) == {"slow": {"state": "unknown"}}
        assert time.monotonic() - started < 1.0
    finally:
        release.set()
