"""Regression tests for durable, pre-call session budgets.

The budget must be reserved in state.db before a provider request. A fresh
agent/process resuming the same session must therefore see the spent budget.
"""

import time

from hermes_state import SessionDB


def test_api_call_budget_is_reserved_before_call_and_survives_new_instance(tmp_path):
    db_path = tmp_path / "state.db"
    first = SessionDB(db_path=db_path)
    first.create_session("budget-session", "test")

    assert first.reserve_session_api_call("budget-session", max_api_calls=2) == (True, 1, None)
    first.close()

    # Simulates a resumed session handled by another agent instance/process.
    resumed = SessionDB(db_path=db_path)
    assert resumed.reserve_session_api_call("budget-session", max_api_calls=2) == (True, 2, None)
    assert resumed.reserve_session_api_call("budget-session", max_api_calls=2) == (
        False,
        2,
        "api_call_ceiling",
    )
    assert resumed.get_session("budget-session")["api_call_count"] == 2
    resumed.close()


def test_session_duration_budget_rejects_active_but_expired_session(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("expired-session", "test")
    # The session remains active; elapsed lifetime, not inactivity, is the limit.
    db._execute_write(lambda conn: conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = ?",
        (time.time() - 61, "expired-session"),
    ))

    assert db.reserve_session_api_call(
        "expired-session", max_api_calls=10, max_session_duration_seconds=60,
    ) == (False, 0, "session_duration_ceiling")
    db.close()


def test_session_cost_budget_rejects_before_another_provider_call(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("cost-session", "test")
    db.update_token_counts("cost-session", estimated_cost_usd=15.0)

    assert db.reserve_session_api_call(
        "cost-session", max_api_calls=10, max_session_cost_usd=15.0,
    ) == (False, 0, "session_cost_ceiling")
    db.close()
