"""Daily usage chart must attribute tokens to the UTC activity day (#107861).

Cross-day sessions currently pin lifetime counters to date(started_at) and
drop pre-window sessions entirely. These cases pin the activity-day ledger
contract: in-window deltas land on today, residuals keep the old start-day
shape, and absolute writes ledger only the increment.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest


@pytest.fixture
def client(monkeypatch, _isolate_hermes_home):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")

    import hermes_state
    from hermes_cli.web_server import app, _SESSION_HEADER_NAME, _SESSION_TOKEN
    from hermes_constants import get_hermes_home

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    c = TestClient(app)
    c.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN
    return c


def _utc_day(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts if ts is not None else time.time(), tz=timezone.utc).strftime(
        "%Y-%m-%d"
    )


def _day_row(daily, day: str):
    return next((row for row in daily if row.get("day") == day), None)


def _session_db():
    from hermes_state import SessionDB

    return SessionDB()


def _set_session_fields(db, session_id: str, **fields) -> None:
    cols = ", ".join(f"{name} = ?" for name in fields)
    values = list(fields.values()) + [session_id]
    db._execute_write(
        lambda conn: conn.execute(f"UPDATE sessions SET {cols} WHERE id = ?", values)
    )


def test_cross_day_delta_lands_on_activity_day_not_start_day(client):
    """Detection 1+2: a backdated session's later delta is visible in a short window."""
    db = _session_db()
    try:
        db.create_session("cross-day-delta", source="cli", model="test-model")
        started_at = time.time() - 5 * 86400
        _set_session_fields(db, "cross-day-delta", started_at=started_at)
        db.update_token_counts(
            "cross-day-delta",
            input_tokens=1_000_000,
            api_call_count=1,
        )
    finally:
        db.close()

    resp = client.get("/api/analytics/usage?days=2")
    assert resp.status_code == 200
    data = resp.json()
    today = _utc_day()
    start_day = _utc_day(started_at)

    today_row = _day_row(data["daily"], today)
    assert today_row is not None, f"today ({today}) missing from daily={data['daily']!r}"
    assert today_row["input_tokens"] == 1_000_000
    assert today_row["api_calls"] == 1
    assert _day_row(data["daily"], start_day) is None
    assert data["totals"]["total_input"] == 1_000_000
    assert data["totals"]["total_api_calls"] == 1


def test_same_day_usage_matches_lifetime_on_today_bar(client):
    db = _session_db()
    try:
        db.create_session("same-day-control", source="cli", model="test-model")
        db.update_token_counts(
            "same-day-control",
            input_tokens=100,
            output_tokens=20,
            api_call_count=1,
            estimated_cost_usd=0.25,
        )
    finally:
        db.close()

    resp = client.get("/api/analytics/usage?days=2")
    assert resp.status_code == 200
    data = resp.json()
    today_row = _day_row(data["daily"], _utc_day())
    assert today_row is not None
    assert today_row["input_tokens"] == 100
    assert today_row["output_tokens"] == 20
    assert today_row["api_calls"] == 1
    assert today_row["estimated_cost"] == pytest.approx(0.25)
    assert today_row["sessions"] == 1
    assert data["totals"]["total_input"] == 100
    assert data["totals"]["total_output"] == 20
    assert data["totals"]["total_api_calls"] == 1
    assert data["totals"]["total_estimated_cost"] == pytest.approx(0.25)
    assert data["totals"]["total_sessions"] == 1


def test_pre_migration_residual_stays_on_started_at_day(client):
    """Fail-open: lifetime counters with no ledger still pin to date(started_at)."""
    db = _session_db()
    try:
        db.create_session("pre-migration-residual", source="cli", model="test-model")
        started_at = time.time() - 1 * 86400
        _set_session_fields(
            db,
            "pre-migration-residual",
            started_at=started_at,
            input_tokens=777,
            output_tokens=11,
            api_call_count=2,
            estimated_cost_usd=1.5,
        )
    finally:
        db.close()

    resp = client.get("/api/analytics/usage?days=30")
    assert resp.status_code == 200
    data = resp.json()
    start_row = _day_row(data["daily"], _utc_day(started_at))
    assert start_row is not None
    assert start_row["input_tokens"] == 777
    assert start_row["output_tokens"] == 11
    assert start_row["api_calls"] == 2
    assert start_row["estimated_cost"] == pytest.approx(1.5)
    assert data["totals"]["total_input"] == 777
    assert data["totals"]["total_output"] == 11
    assert data["totals"]["total_api_calls"] == 2
    assert data["totals"]["total_estimated_cost"] == pytest.approx(1.5)


def test_absolute_write_ledgers_only_the_increment(client):
    db = _session_db()
    try:
        db.create_session("absolute-increment", source="cli", model="test-model")
        started_at = time.time() - 5 * 86400
        _set_session_fields(db, "absolute-increment", started_at=started_at)
        db.update_token_counts(
            "absolute-increment",
            input_tokens=1_000_000,
            api_call_count=2,
            estimated_cost_usd=4.0,
            absolute=True,
        )
        db.update_token_counts(
            "absolute-increment",
            input_tokens=1_500_000,
            api_call_count=3,
            estimated_cost_usd=6.0,
            absolute=True,
        )
    finally:
        db.close()

    resp = client.get("/api/analytics/usage?days=2")
    assert resp.status_code == 200
    data = resp.json()
    today_row = _day_row(data["daily"], _utc_day())
    assert today_row is not None
    # Both writes happen today: increments 1_000_000 then 500_000, not 1M+1.5M.
    assert today_row["input_tokens"] == 1_500_000
    assert today_row["api_calls"] == 3
    assert today_row["estimated_cost"] == pytest.approx(6.0)
    assert data["totals"]["total_input"] == 1_500_000
    assert data["totals"]["total_api_calls"] == 3
    assert data["totals"]["total_estimated_cost"] == pytest.approx(6.0)
    assert _day_row(data["daily"], _utc_day(started_at)) is None
