"""Month-to-date usage counts calls when they were made, under the provider that served them.

Regressions from review of the first cut, which bucketed by ``sessions.started_at`` and grouped the
session's lifetime counters by its last provider: real SessionDB writes at controlled times.
"""
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import hermes_state_usage
from hermes_cli.usage_budget import ledger_started_at, month_usage_rows, month_window, summarize_month
from hermes_state import SessionDB

NOW = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
MONTH_START = month_window(NOW)[0].timestamp()
SEPT_29 = MONTH_START - 2 * 86400
OCT_5 = MONTH_START + 4 * 86400


@pytest.fixture
def db(tmp_path):
    store = SessionDB(db_path=tmp_path / "state.db")
    yield store
    store.close()


@pytest.fixture
def at(monkeypatch):
    """``at(ts)`` makes the next writes happen at *ts*."""
    clock = {"now": time.time()}
    monkeypatch.setattr(hermes_state_usage, "time", SimpleNamespace(time=lambda: clock["now"], monotonic=time.monotonic))
    return lambda ts: clock.update(now=ts)


def _call(db, session_id, provider, tokens, **kw):
    db.update_token_counts(session_id, input_tokens=tokens, model=f"{provider}-model", billing_provider=provider,
                           api_call_count=1, **kw)


def _by_provider(db):
    return {row["provider"]: row for row in month_usage_rows(db._conn, MONTH_START)}


def test_a_session_from_last_month_counts_only_the_calls_made_this_month(db, at):
    at(SEPT_29)
    db.create_session("long-lived", "cli")
    _call(db, "long-lived", "xiaomi", 50_000, cost_status="unknown")
    db.create_session("idle-new", "cli")  # opened this month, no calls yet
    at(OCT_5)
    _call(db, "long-lived", "xiaomi", 7_000, cost_status="unknown")

    (row,) = month_usage_rows(db._conn, MONTH_START)

    assert (row["provider"], row["tokens"], row["sessions"], row["unpriced_sessions"]) == ("xiaomi", 7_000, 1, 1)


def test_a_provider_switch_splits_the_month_by_provider(db, at):
    at(OCT_5)
    db.create_session("switched", "cli")
    _call(db, "switched", "openrouter", 100_000, estimated_cost_usd=0.4, cost_status="estimated")
    _call(db, "switched", "deepseek", 1_000, cost_status="unknown")

    rows = _by_provider(db)

    assert (rows["openrouter"]["tokens"], rows["openrouter"]["estimated_cost"]) == (100_000, 0.4)
    assert (rows["openrouter"]["unpriced_sessions"], rows["deepseek"]["unpriced_sessions"]) == (0, 1)
    assert rows["deepseek"]["tokens"] == 1_000


def test_auxiliary_calls_count_under_their_own_provider(db, at):
    at(OCT_5)
    db.create_session("s", "cli")
    _call(db, "s", "xiaomi", 1_000, cost_status="unknown")
    db.record_auxiliary_usage("s", "compression", model="gemini-flash", billing_provider="gemini",
                              input_tokens=300, output_tokens=20, estimated_cost_usd=0.001)

    rows = _by_provider(db)

    assert (rows["xiaomi"]["tokens"], rows["gemini"]["tokens"], rows["gemini"]["unpriced_sessions"]) == (1_000, 320, 0)


def test_a_ledger_started_mid_month_projects_only_the_counted_days(db, at):
    """Upgrading mid-month: the days before were never recorded by time, so the pace starts at the upgrade."""
    at(OCT_5)
    db.create_session("s", "cli")
    _call(db, "s", "xiaomi", 50_000_000, cost_status="unknown")
    ledger_start = datetime.fromtimestamp(ledger_started_at(db._conn), UTC)

    month = summarize_month(month_usage_rows(db._conn, MONTH_START), {"xiaomi": ("tokens", 300_000_000.0)}, NOW,
                            ledger_start=ledger_start)

    assert month["counted_since"] == ledger_start.isoformat()
    budget = month["providers"][0]["budget"]
    # 50M over the 5.5 counted days → ~9.09M/day over the 27 counted days of October.
    assert round(budget["projected_ratio"], 2) == round(50_000_000 / 5.5 * 27 / 300_000_000, 2)
    assert budget["runs_out_on"] is None


def test_a_ledger_older_than_the_month_counts_the_whole_month():
    month = summarize_month([], {}, NOW, ledger_start=datetime(2026, 9, 1, tzinfo=UTC))
    assert month["counted_since"] is None
