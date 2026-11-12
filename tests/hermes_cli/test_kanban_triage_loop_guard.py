"""Regression tests for the t_f0ad4213 storm: the auto-decomposer must not
re-arm unblock-loop-escalated human-wait cards.

Storm chain (live evidence 2026-11-12): worker blocks a human-only card
``kind=needs_input`` -> at ``BLOCK_RECURRENCE_LIMIT`` ``_route_block``
(kanban_db.py) parks it in ``triage`` with a ``block_loop_detected`` event
"for a human" -> ``auto_decompose_tick``
(gateway/kanban_watchers_dispatcher.py) pulled *every* triage card,
including exactly those, through ``decompose_task`` -> ``_apply_single`` ->
``specify_triage_task`` flipped it back to ``todo`` -> dispatcher promoted
and claimed it -> same worker, same human wait, same triage. ~1
specified→promoted→claimed→block_loop_detected cycle per minute for 15
recurrences: the loop breaker's escalation queue was the decomposer's work
queue.

Fix contract tested here:
* ``list_triage_ids(exclude_loop_limited=True)`` hides cards with
  ``block_recurrences >= BLOCK_RECURRENCE_LIMIT``; the dispatcher tick uses
  it, so automation treats the escalation as terminal.
* ``specify_triage_task`` (the human re-arm path) clears the loop memory and
  records ``loop_counter_reset`` on the ``specified`` event, so a
  deliberately re-scoped card gets a fresh budget — and can still escalate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t", assignee="worker"):
    tid = kb.create_task(conn, title=title, assignee=assignee)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer=assignee) is not None
    return tid


def _escalate_to_triage(conn, tid, kind="needs_input", cycles=2):
    """Block/unblock *cycles* times with the same kind; last block routes to triage."""
    for _ in range(cycles):
        kb.block_task(conn, tid, reason="human wait", kind=kind)
        row = kb.get_task(conn, tid)
        if row.status == "triage":
            return
        kb.unblock_task(conn, tid)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="worker") is not None
    raise AssertionError("task never escalated to triage")


# ---------------------------------------------------------------------------
# list_triage_ids filter
# ---------------------------------------------------------------------------

def test_list_triage_ids_excludes_loop_limited(kanban_home: Path) -> None:
    from hermes_cli import kanban_decompose as decomp

    with kbc.connect_closing() as conn:
        fresh = kb.create_task(conn, title="fresh triage card", triage=True)
        normal = _running_task(conn, title="escalates normally")
        _escalate_to_triage(conn, normal)

        assert kb.get_task(conn, normal).status == "triage"
        assert kb.get_task(conn, normal).block_recurrences >= kb.BLOCK_RECURRENCE_LIMIT
        assert kb.get_task(conn, fresh).status == "triage"

        assert set(decomp.list_triage_ids()) == {fresh, normal}
        assert decomp.list_triage_ids(exclude_loop_limited=True) == [fresh]


def test_below_limit_card_still_decomposable(kanban_home: Path) -> None:
    """A triage card that has NOT hit the breaker (recurrences < limit) keeps
    the normal auto-decompose flow — no regression for ordinary triage."""
    from hermes_cli import kanban_decompose as decomp

    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="once-blocked triage", triage=True)
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET block_kind='needs_input', block_recurrences=1 WHERE id=?",
                (tid,),
            )
        assert decomp.list_triage_ids(exclude_loop_limited=True) == [tid]


# ---------------------------------------------------------------------------
# dispatcher tick (the automation entry point)
# ---------------------------------------------------------------------------

def test_auto_decompose_tick_skips_loop_limited(kanban_home: Path, monkeypatch) -> None:
    from gateway import kanban_watchers_dispatcher as disp
    from hermes_cli import kanban_decompose as decomp

    with kbc.connect_closing() as conn:
        fresh = kb.create_task(conn, title="fresh", triage=True)
        storm = _running_task(conn, title="stormed human-wait")
        _escalate_to_triage(conn, storm)

    attempted: list[str] = []

    class _Outcome:
        task_id: str
        ok = False
        reason = "test stub: never reaches the aux LLM"
        fanout = False
        child_ids: list[str] = []
        new_title = None

    def _fake_decompose(task_id, *, author=None):
        attempted.append(task_id)
        return _Outcome()

    monkeypatch.setattr(decomp, "decompose_task", _fake_decompose)

    settings = disp._DispatcherSettings(
        interval=60.0, max_spawn=None, max_in_progress=None,
        failure_limit=kb.DEFAULT_FAILURE_LIMIT, stale_timeout_seconds=3600,
        reconcile_orphans=False, default_assignee=None,
        max_in_progress_per_profile=None,
    )
    dispatcher = disp._KanbanDispatcher(kb, settings)
    dispatcher.auto_decompose_tick(auto_decompose_per_tick=10)

    assert fresh in attempted, "ordinary triage card must still auto-decompose"
    assert storm not in attempted, "loop-limited card must NOT be auto-re-armed"


# ---------------------------------------------------------------------------
# human re-arm: specify resets the loop memory
# ---------------------------------------------------------------------------

def test_specify_resets_loop_counter_and_records_event(kanban_home: Path) -> None:
    with kbc.connect_closing() as conn:
        tid = _running_task(conn, title="to be specified by a human")
        _escalate_to_triage(conn, tid)
        before = kb.get_task(conn, tid)
        assert before.block_recurrences >= kb.BLOCK_RECURRENCE_LIMIT

        assert kb.specify_triage_task(conn, tid, body="resolved: founder approved", author="human")

        after = kb.get_task(conn, tid)
        # todo, then recompute_ready (run at the end of specify) flips a
        # parent-free card straight to ready — both are "left triage".
        assert after.status in ("todo", "ready")
        assert (after.block_recurrences or 0) == 0
        assert after.block_kind is None
        events = [e for e in kb.list_events(conn, tid) if e.kind == "specified"]
        assert events and (events[-1].payload or {}).get("loop_counter_reset") is True


def test_rearmed_card_can_still_escalate(kanban_home: Path) -> None:
    """The reset grants a *fresh budget*, not immunity: a re-armed card that
    loops again still escalates to triage."""
    with kbc.connect_closing() as conn:
        tid = _running_task(conn, title="re-looping card")
        _escalate_to_triage(conn, tid)
        kb.specify_triage_task(conn, tid, body="re-scoped", author="human")

        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        _escalate_to_triage(conn, tid)
        row = kb.get_task(conn, tid)
        assert row.status == "triage"
        assert row.block_recurrences == kb.BLOCK_RECURRENCE_LIMIT
