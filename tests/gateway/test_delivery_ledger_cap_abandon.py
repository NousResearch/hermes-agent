"""Regression tests for #100660: failed delivery obligations must leave the
recoverable pool when the attempts budget is exhausted — at mark_failed time.

The reporter's incident: Slack 429 rejections drove delivery_obligations rows
to attempts=3 state='failed', and the rows were re-sent on every gateway
restart indefinitely. Two escape routes kept the rows recoverable:

1. sweep_recoverable's abandon path only runs when the owning process is DEAD
   — a row failed by THIS process, owned by THIS process, never reaches it
   while the process lives.
2. release_runtime_claim DECREMENTS attempts, so the runtime reconnect loop
   (claim 2->3, release 3->2) could bounce a row under the cap forever.

Fix: mark_failed() abandons the row the moment a definitive rejection arrives
with attempts already at MAX_ATTEMPTS. Whatever path spent the budget, the
row leaves the recoverable pool at the point of exhaustion.

These tests drive the REAL ledger functions against real SQLite (the module's
own _connect/_initialize_schema under a temp HERMES_HOME) — no mocked SQL.
"""

import time
from pathlib import Path

import pytest


@pytest.fixture()
def ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import importlib

    import gateway.delivery_ledger as dl

    importlib.reload(dl)
    with dl._DB_LOCK, dl._transaction() as conn:
        dl._initialize_schema(conn)
    return dl


def _make_obligation(ledger, *, state="pending", attempts=0) -> str:
    """Record a pending row then drive its attempts to the desired point the
    way production does: claim (attempts+1, state->attempting) via
    mark_attempting + a manual UPDATE for the spent count."""
    oid = ledger.compute_obligation_id("s", "m1", "hello")
    ledger.record_obligation(
        obligation_id=oid,
        session_key="s",
        platform="slack",
        chat_id="C1",
        thread_id=None,
        content="hello",
    )
    # Spend attempts the way sweeps do (attempts=attempts+1 at claim time).
    with ledger._DB_LOCK, ledger._transaction() as conn:
        conn.execute(
            "UPDATE delivery_obligations SET attempts=?, state=? WHERE obligation_id=?",
            (attempts, state, oid),
        )
    return oid


def _get_row(ledger, oid):
    with ledger._DB_LOCK, ledger._transaction() as conn:
        return conn.execute(
            "SELECT state, attempts, last_error FROM delivery_obligations "
            "WHERE obligation_id=?",
            (oid,),
        ).fetchone()


class TestMarkFailedEnforcesCap:
    def test_failure_at_cap_abandons_immediately(self, ledger):
        """The #100660 witness: attempts=3 (budget spent by prior claims),
        the send 429s, mark_failed must ABANDON — not leave it recoverable
        for the next restart's sweep."""
        oid = _make_obligation(ledger, state="attempting", attempts=3)
        ledger.mark_failed(oid, error="slack 429 ratelimited")

        row = _get_row(ledger, oid)
        assert row[0] == "abandoned", (
            "a definitive rejection at the attempts cap must leave the "
            "recoverable pool immediately (#100660) — leaving it 'failed' "
            "re-sends on every restart"
        )
        assert "ratelimited" in (row[2] or "")

    def test_failure_below_cap_stays_failed(self, ledger):
        """The normal retry budget: a failure with attempts remaining keeps
        the row recoverable for the sweep paths."""
        oid = _make_obligation(ledger, state="attempting", attempts=1)
        ledger.mark_failed(oid, error="slack 429")

        row = _get_row(ledger, oid)
        assert row[0] == "failed"
        assert row[1] == 1, "mark_failed must not spend attempts"

    def test_boundary_exactly_max_abandons(self, ledger):
        """attempts == MAX_ATTEMPTS is the capped row (this send WAS the
        MAXth attempt). One below stays failed."""
        oid = _make_obligation(ledger, state="attempting",
                               attempts=ledger.MAX_ATTEMPTS)
        ledger.mark_failed(oid, error="x")
        assert _get_row(ledger, oid)[0] == "abandoned"

        oid2 = _make_obligation(ledger, state="attempting",
                                attempts=ledger.MAX_ATTEMPTS - 1)
        ledger.mark_failed(oid2, error="x")
        assert _get_row(ledger, oid2)[0] == "failed"

    def test_abandoned_row_not_reclaimed_by_sweep(self, ledger):
        """The end-to-end contract: a row abandoned by mark_failed must not
        be claimed by sweep_recoverable on the next (dead-owner) boot."""
        oid = _make_obligation(ledger, state="attempting", attempts=3)
        ledger.mark_failed(oid, error="ratelimited")

        claimed = ledger.sweep_recoverable(
            deliverable_platforms={"slack"},
            deliverable_targets={("slack", "default")},
        )
        assert not any(c["obligation_id"] == oid for c in claimed), (
            "an abandoned row must never be re-sent by the restart sweep"
        )

    def test_runtime_release_cannot_resurrect_capped_row(self, ledger):
        """Escape route 2: release_runtime_claim decrements attempts — but a
        row ABANDONED at the cap can't be released back into the failed pool
        because its state is no longer 'attempting' after mark_failed
        terminal-ized it."""
        oid = _make_obligation(ledger, state="attempting", attempts=3)
        ledger.mark_failed(oid, error="ratelimited")

        released = ledger.release_runtime_claim(oid, error="late release")
        assert released is False, (
            "release after terminal abandonment must be a no-op — otherwise "
            "the decrement could bounce the row back under the cap forever"
        )
        assert _get_row(ledger, oid)[0] == "abandoned"

    def test_delivered_unaffected(self, ledger):
        """mark_delivered still wins for the happy path."""
        oid = _make_obligation(ledger, state="attempting", attempts=2)
        ledger.mark_delivered(oid)
        assert _get_row(ledger, oid)[0] == "delivered"
