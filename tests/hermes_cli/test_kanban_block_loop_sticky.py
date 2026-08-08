"""Regression: a block-loop escalation must not be resurrected by the
dispatcher's auto-decomposer.

Live defect (board default, event 2955 / run 233, task ``t_bdf1001a``;
``t_80a3542a`` showed the same class): ``block_task`` routed a repeated
``needs_input`` block (``block_recurrences >= BLOCK_RECURRENCE_LIMIT``) to
``status='triage'`` and emitted ``block_loop_detected``. But ``triage`` is an
*automation* queue, not a human hold: with ``kanban.auto_decompose`` enabled
(the default) the dispatcher treats every triage row as a decompose/specify
candidate. On the next tick it silently advanced the escalated card
``triage -> todo`` (no lifecycle event) and ``recompute_ready`` would then make
it ``ready`` — exactly the blocked-task-resurrection class this branch fixes.

The invariant pinned here: a block-loop escalation lands in a *sticky*
``blocked`` state that auto-decompose, ``recompute_ready``, ``claim_task`` and
dispatcher ticks cannot advance. Only an explicit ``unblock_task`` resumes it.
Ordinary triage cards keep auto-decomposing.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_decompose as kd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn: sqlite3.Connection, title: str = "t") -> str:
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    return tid


def _escalate_block_loop(conn: sqlite3.Connection, kind: str = "needs_input") -> str:
    """Drive a task through the real block -> unblock -> re-block loop until
    ``block_task`` trips the recurrence breaker."""
    tid = _running_task(conn, title="worker keeps asking the same question")
    assert kb.block_task(conn, tid, reason="need creds", kind=kind)
    assert kb.unblock_task(conn, tid)  # the cron that keeps spinning it
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None
    assert kb.block_task(conn, tid, reason="need creds", kind=kind)
    row = conn.execute(
        "SELECT block_recurrences FROM tasks WHERE id=?", (tid,)
    ).fetchone()
    assert int(row["block_recurrences"]) >= kb.BLOCK_RECURRENCE_LIMIT
    kinds = [
        r["kind"] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id=? ORDER BY id", (tid,)
        )
    ]
    assert "block_loop_detected" in kinds
    return tid


# ---------------------------------------------------------------------------
# The escalation itself
# ---------------------------------------------------------------------------


def test_block_loop_escalation_is_sticky_blocked_not_triage(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _escalate_block_loop(conn)
        task = kb.get_task(conn, tid)
        assert task.status == "blocked", (
            "block-loop escalation must land in a sticky blocked state, not in "
            "the triage automation queue"
        )
        assert task.block_kind == "needs_input"
        assert kb._has_sticky_block(conn, tid) is True


def test_block_loop_escalation_is_invisible_to_auto_decompose(kanban_home: Path) -> None:
    """The real dispatcher path: ``list_triage_ids`` feeds the auto-decomposer
    every tick. An escalated card must never appear in it."""
    with kb.connect_closing() as conn:
        tid = _escalate_block_loop(conn)
        ordinary = kb.create_task(conn, title="rough idea", triage=True)

    assert kd.list_triage_ids() == [ordinary], (
        "auto-decompose must not see a block-loop escalation as a triage "
        "candidate"
    )

    outcome = kd.decompose_task(tid)
    assert outcome.ok is False


def test_block_loop_escalation_survives_dispatcher_ticks(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _escalate_block_loop(conn)
        for _ in range(5):
            assert kb.recompute_ready(conn) == 0
            assert kb.get_task(conn, tid).status == "blocked"
        assert kb.claim_task(conn, tid, claimer="worker") is None


def test_block_loop_escalation_rejects_specify_and_decompose(kanban_home: Path) -> None:
    """Defense in depth: even if something parks an escalated row back in
    ``triage``, the two production promotion helpers must refuse it."""
    with kb.connect_closing() as conn:
        tid = _escalate_block_loop(conn)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='triage' WHERE id=?", (tid,))
        assert kb.specify_triage_task(conn, tid, body="spec") is False
        assert kb.decompose_triage_task(
            conn, tid, root_assignee="worker",
            children=[{"title": "child"}],
        ) is None
        assert kb.get_task(conn, tid).status == "triage"


def test_explicit_unblock_resumes_a_block_loop_escalation(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _escalate_block_loop(conn)
        assert kb.unblock_task(conn, tid) is True
        assert kb.get_task(conn, tid).status == "ready"
        assert kb._has_sticky_block(conn, tid) is False
        assert kb.claim_task(conn, tid, claimer="worker") is not None


# ---------------------------------------------------------------------------
# Legacy rows already parked in triage by the old routing
# ---------------------------------------------------------------------------


def _legacy_escalated_row(conn: sqlite3.Connection, title: str) -> str:
    """Write a row shaped exactly like the pre-fix escalation: parked in
    ``triage`` with ``block_kind`` set and a trailing ``block_loop_detected``
    event."""
    tid = kb.create_task(conn, title=title)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='triage', block_kind='needs_input', "
            "block_recurrences=? WHERE id=?",
            (kb.BLOCK_RECURRENCE_LIMIT, tid),
        )
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) "
            "VALUES (?, 'block_loop_detected', ?, ?)",
            (
                tid,
                json.dumps({
                    "reason": "need creds",
                    "kind": "needs_input",
                    "recurrences": kb.BLOCK_RECURRENCE_LIMIT,
                    "limit": kb.BLOCK_RECURRENCE_LIMIT,
                }),
                int(time.time()),
            ),
        )
    return tid


def test_migration_lifts_legacy_triage_escalations_out_of_triage(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        escalated = _legacy_escalated_row(conn, "legacy escalation")
        ordinary = kb.create_task(conn, title="ordinary idea", triage=True)
        # An escalation an operator already unblocked and re-parked in triage
        # by hand is NOT ours to move.
        cleared = _legacy_escalated_row(conn, "already handled")
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, 'unblocked', NULL, ?)",
                (cleared, int(time.time())),
            )

    kb.init_db()  # activation re-runs the migration pass

    with kb.connect_closing() as conn:
        assert kb.get_task(conn, escalated).status == "blocked"
        assert kb._has_sticky_block(conn, escalated) is True
        assert kb.get_task(conn, ordinary).status == "triage"
        assert kb.get_task(conn, cleared).status == "triage"

    assert sorted(kd.list_triage_ids()) == sorted([ordinary, cleared])


def test_migration_leaves_ordinary_triage_auto_decomposable(kanban_home: Path) -> None:
    """The legitimate flow must keep working: a plain triage card is still a
    decompose candidate and still promotes to ``todo``."""
    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="build me a thing", triage=True)

    kb.init_db()

    assert kd.list_triage_ids() == [tid]
    with kb.connect_closing() as conn:
        assert kb.specify_triage_task(conn, tid, body="spec") is True
        assert kb.get_task(conn, tid).status in ("todo", "ready")


# ---------------------------------------------------------------------------
# Legacy DBs whose ``task_events`` still carries drifted TEXT ids (#35096)
# ---------------------------------------------------------------------------


_LEGACY_EVENTS_SQL = """
CREATE TABLE task_events (
    id         TEXT PRIMARY KEY,
    task_id    TEXT NOT NULL,
    run_id     INTEGER,
    kind       TEXT NOT NULL,
    payload    TEXT,
    created_at INTEGER NOT NULL
)
"""


def _drift_task_events_to_text_ids(conn: sqlite3.Connection) -> None:
    """Rewrite ``task_events`` into the pre-#35096 TEXT-id shape.

    Rows keep their chronological order but get ``ev-1 … ev-N`` string ids —
    which sort *lexicographically*, so ``ev-9`` compares greater than
    ``ev-10``. ``_rebuild_drifted_tables`` is what repairs this on init.
    """
    rows = conn.execute(
        "SELECT task_id, run_id, kind, payload, created_at "
        "FROM task_events ORDER BY id"
    ).fetchall()
    conn.execute("DROP TABLE task_events")
    conn.execute(_LEGACY_EVENTS_SQL)
    for idx, row in enumerate(rows, start=1):
        conn.execute(
            "INSERT INTO task_events (id, task_id, run_id, kind, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f"ev-{idx}", row["task_id"], row["run_id"], row["kind"],
             row["payload"], row["created_at"]),
        )
    conn.commit()
    assert kb._table_has_drifted(conn, "task_events")


def _legacy_triage_row_with_history(
    conn: sqlite3.Connection, title: str, *, tail_kinds: list[str],
    filler: int,
) -> str:
    """A triage row with a realistic (>= 10 event) history whose final
    chronological events are ``tail_kinds``.

    ``filler`` pads the history so the caller can place ``tail_kinds`` at exact
    positions in the *global* event sequence — which is what decides whether
    the legacy TEXT ids (``ev-9`` vs ``ev-10``) sort the wrong way round.
    ``create_task`` already emits one ``created`` event, so the row carries
    ``1 + filler + len(tail_kinds)`` events in total.
    """
    tid = kb.create_task(conn, title=title)
    now = int(time.time())
    kinds = ["claimed", "heartbeat", "comment", "claimed", "reclaimed",
             "claimed", "comment", "heartbeat", "comment", "claimed"]
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status='triage', block_kind='needs_input', "
            "block_recurrences=? WHERE id=?",
            (kb.BLOCK_RECURRENCE_LIMIT, tid),
        )
        for offset, kind in enumerate(kinds[:filler] + tail_kinds):
            payload = None
            if kind == "block_loop_detected":
                payload = json.dumps({
                    "reason": "need creds", "kind": "needs_input",
                    "recurrences": kb.BLOCK_RECURRENCE_LIMIT,
                    "limit": kb.BLOCK_RECURRENCE_LIMIT,
                })
            conn.execute(
                "INSERT INTO task_events (task_id, kind, payload, created_at) "
                "VALUES (?, ?, ?, ?)",
                (tid, kind, payload, now + offset),
            )
    return tid


def test_migration_reads_event_order_after_text_id_drift_is_repaired(
    kanban_home: Path,
) -> None:
    """A legacy drifted board must not have its escalation migration decided by
    lexicographic TEXT ids.

    On a pre-#35096 board ``task_events.id`` is TEXT, so ``ORDER BY id DESC``
    picks ``ev-9`` over ``ev-10``. A task whose real history ends
    ``… block_loop_detected (ev-9), unblocked (ev-10)`` — i.e. an operator
    already cleared the escalation — then looks *still escalated*. If the
    escalation migration runs before ``_rebuild_drifted_tables`` repairs the
    ids, init converts that triage card to ``blocked``; the rebuild then
    renumbers the events, ``_has_sticky_block`` reads the true tail
    (``unblocked``) and returns False, and the very next ``recompute_ready``
    promotes the card to ``ready`` for a worker to claim. That is the
    resurrection this branch exists to prevent, reached by a different door.
    """
    with kb.connect_closing() as conn:
        # Built first so its events own global ids 1-10: the escalation lands
        # at ev-9 and the operator's unblock at ev-10, which TEXT ids sort
        # backwards ('ev-9' > 'ev-10').
        cleared = _legacy_triage_row_with_history(
            conn, "operator already unblocked", filler=7,
            tail_kinds=["block_loop_detected", "unblocked"],
        )
        still_escalated = _legacy_triage_row_with_history(
            conn, "still waiting on a human", filler=8,
            tail_kinds=["unblocked", "block_loop_detected"],
        )
        ordinary = kb.create_task(conn, title="ordinary idea", triage=True)
        for tid in (cleared, still_escalated):
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM task_events WHERE task_id=?", (tid,)
            ).fetchone()["n"]
            assert n >= 10, n
        _drift_task_events_to_text_ids(conn)
        # Pin the hazard itself: on the drifted table the event log reads
        # backwards for the cleared row.
        assert kb._latest_block_event_kind(conn, cleared) == "block_loop_detected"

    kb.init_db()  # activation: rebuild + escalation migration

    with kb.connect_closing() as conn:
        assert kb._table_has_drifted(conn, "task_events") is False
        # The cleared row was never ours to touch — it must stay exactly where
        # the operator left it, and must not be silently made claimable.
        assert kb.get_task(conn, cleared).status == "triage", (
            "a legacy row whose newest event is 'unblocked' must not be "
            "migrated out of triage on the strength of lexicographic id order"
        )
        assert kb.has_block_loop_escalation(conn, cleared) is False
        # The genuinely-escalated row still becomes a sticky block.
        assert kb.get_task(conn, still_escalated).status == "blocked"
        assert kb._has_sticky_block(conn, still_escalated) is True
        assert kb.get_task(conn, ordinary).status == "triage"

        # No dispatcher tick may hand either card back to a worker.
        for _ in range(3):
            kb.recompute_ready(conn)
        assert kb.get_task(conn, cleared).status == "triage"
        assert kb.get_task(conn, still_escalated).status == "blocked"
        assert kb.claim_task(conn, cleared, claimer="worker") is None
        assert kb.claim_task(conn, still_escalated, claimer="worker") is None
