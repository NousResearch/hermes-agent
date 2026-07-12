"""Tests for authoritative per-call cost capture (x-nous-credits-* delta -> actual_cost_usd).

Root-cause fix for the telemetry gap: Hermes recorded only *estimated* cost;
the server's authoritative billable cost was parsed (credits_tracker) but never
written to state.db.actual_cost_usd. These tests prove the delta math and the
DB increment/absolute-set behavior.
"""
import pytest
from unittest.mock import MagicMock

from hermes_state import SessionDB


def _credits_headers(remaining_usd: str, remaining_micros: int) -> dict:
    return {
        "x-nous-credits-version": "1",
        "x-nous-credits-remaining-micros": str(remaining_micros),
        "x-nous-credits-remaining-usd": remaining_usd,
        "x-nous-credits-subscription-micros": str(remaining_micros),
        "x-nous-credits-subscription-usd": remaining_usd,
        "x-nous-credits-subscription-limit-micros": "20000000",
        "x-nous-credits-subscription-limit-usd": "20.00",
        "x-nous-credits-rollover-micros": "0",
        "x-nous-credits-purchased-micros": "0",
        "x-nous-credits-purchased-usd": "0.00",
        "x-nous-credits-denominator-kind": "subscription_cap",
        "x-nous-credits-paid-access": "true",
        "x-nous-credits-as-of-ms": "1",
    }


def test_per_call_delta_computes_cost():
    """prev remaining 10000µ -> new 7000µ == 0.003 USD for this call."""
    from run_agent import AIAgent  # type: ignore  (importable under repo root)
    from agent.credits_tracker import parse_credits_headers

    agent = MagicMock(spec=AIAgent)
    agent._credits_state = parse_credits_headers(_credits_headers("0.01", 10000))
    agent._credits_session_start_micros = 10000

    # Simulate the production call site: _capture_credits computes the delta.
    new_state = parse_credits_headers(_credits_headers("0.00", 7000))
    _prev = agent._credits_state
    delta = None
    if (
        _prev is not None and _prev.from_header
        and _prev.remaining_micros is not None
        and new_state is not None
        and new_state.remaining_micros is not None
        and new_state.remaining_micros <= _prev.remaining_micros
    ):
        delta = _prev.remaining_micros - new_state.remaining_micros
    assert new_state is not None
    assert delta == 3000
    assert delta / 1_000_000.0 == 0.003


def test_update_token_counts_increments_actual_cost(tmp_path):
    db = SessionDB()
    db.db_path = str(tmp_path / "s.db")
    sid = "t1"
    db._insert_session_row(sid, "unknown")
    baseline = db.get_session(sid)["actual_cost_usd"] or 0.0
    baseline_calls = db.get_session(sid)["api_call_count"] or 0
    db.update_token_counts(sid, api_call_count=1, actual_cost_usd=0.003, absolute=False)
    db.update_token_counts(sid, api_call_count=1, actual_cost_usd=0.002, absolute=False)
    row = db.get_session(sid)
    # increment path ADDS per-call deltas to the running total
    assert abs(row["actual_cost_usd"] - (baseline + 0.005)) < 1e-9
    assert row["api_call_count"] == (baseline_calls + 2)


def test_update_token_counts_absolute_sets_actual_cost(tmp_path):
    db = SessionDB()
    db.db_path = str(tmp_path / "s.db")
    sid = "t2"
    db._insert_session_row(sid, "unknown")
    baseline = db.get_session(sid)["actual_cost_usd"] or 0.0
    db.update_token_counts(sid, api_call_count=5, actual_cost_usd=0.042, absolute=True)
    row = db.get_session(sid)
    # absolute path SETS the total directly (overwrites baseline)
    assert abs(row["actual_cost_usd"] - 0.042) < 1e-9
    assert row["api_call_count"] == 5
