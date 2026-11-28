"""Automatic triage sweeps must not undo the unblock-loop breaker.

Regression: a worker blocks ``needs_input`` (waiting on the user), the breaker routes the card to
``triage`` at ``BLOCK_RECURRENCE_LIMIT``, the auto-decomposer "specifies" it back to ``todo`` ->
``ready``, the worker respawns with nothing new and re-blocks, and the loop the breaker exists to
stop runs once per dispatcher tick (one run + one notification each time).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_specify as spec


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claim(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


def _escalated(conn, kind="needs_input"):
    """A card the breaker sent to triage: block -> unblock -> same-cause re-block."""
    tid = kb.create_task(conn, title="gate", assignee="worker")
    _claim(conn, tid)
    assert kb.block_task(conn, tid, reason="waiting on the user", kind=kind)
    assert kb.unblock_task(conn, tid)
    _claim(conn, tid)
    assert kb.block_task(conn, tid, reason="still waiting on the user", kind=kind)
    assert kb.get_task(conn, tid).status == "triage"
    return tid


@pytest.mark.parametrize("kind", ["needs_input", "capability", None])
def test_breaker_escalation_is_skipped_by_both_sweeps(kanban_home, kind):
    with kbc.connect_closing() as conn:
        escalated = _escalated(conn, kind)
        fresh = kb.create_task(conn, title="one-liner", assignee="worker", triage=True)
        assert kb.is_breaker_escalation(conn, escalated)
        assert not kb.is_breaker_escalation(conn, fresh)
    assert decomp.list_triage_ids() == [fresh]
    assert spec.list_triage_ids() == [fresh]


def test_escalated_card_stays_parked_across_dispatch_recompute(kanban_home):
    with kbc.connect_closing() as conn:
        tid = _escalated(conn)
        kb.recompute_ready(conn)
        assert kb.get_task(conn, tid).status == "triage"
        assert kb.claim_task(conn, tid, claimer="worker") is None


def test_unblock_releases_an_escalated_triage_card(kanban_home):
    """The human exit: comment + Unblock / ``hermes kanban unblock --reason`` works on it."""
    with kbc.connect_closing() as conn:
        tid = _escalated(conn)
        assert kb.unblock_task(conn, tid)
        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        # Loop memory survives the unblock, exactly like a blocked-column unblock.
        assert (task.block_kind, task.block_recurrences) == ("needs_input", 2)
        assert not kb.is_breaker_escalation(conn, tid)


def test_unblock_still_refuses_a_plain_triage_card(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="one-liner", assignee="worker", triage=True)
        assert not kb.unblock_task(conn, tid)
        assert kb.get_task(conn, tid).status == "triage"


def test_explicit_specify_still_works_on_an_escalated_card(kanban_home):
    """Only the sweeps skip it; an operator naming the id may still promote it."""
    with kbc.connect_closing() as conn:
        tid = _escalated(conn)
        assert kb.specify_triage_task(conn, tid, body="new spec", author="operator")
        assert kb.get_task(conn, tid).status in ("todo", "ready")


def test_rerouted_to_triage_by_hand_is_sweepable_again(kanban_home):
    """Once a human moves the card on, a later manual triage drop is a normal triage card."""
    with kbc.connect_closing() as conn:
        tid = _escalated(conn)
        assert kb.unblock_task(conn, tid)
        _claim(conn, tid)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage', claim_lock=NULL WHERE id=?", (tid,))
        assert not kb.is_breaker_escalation(conn, tid)
    assert tid in decomp.list_triage_ids()
