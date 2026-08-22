"""Tests for typed block reasons + the unblock-loop breaker.

Covers the built-in fix for the kanban "blocked loop" — a worker blocks a
task, a cron unblocks it, the worker re-blocks for the same reason, repeat
forever. The fix gives ``block_task`` a typed ``kind`` and a persistent
``block_recurrences`` counter:

* ``dependency`` blocks route to ``todo`` (parent-gated, auto-resumed) and
  never enter the human ``blocked`` bucket a cron would keep unblocking.
* ``needs_input`` / ``capability`` / un-typed blocks land in ``blocked``;
  each same-cause re-block after an unblock increments ``block_recurrences``,
  and at ``BLOCK_RECURRENCE_LIMIT`` the task routes to ``triage`` for a human.
* ``unblock_task`` deliberately does NOT reset ``block_recurrences`` (the
  amnesia that let the loop run unbounded).
* A successful ``complete_task`` or material specification resets loop memory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _running_task(conn, title="t"):
    """Create a task and drive it to ``running`` so block_task can act."""
    tid = kb.create_task(conn, title=title, assignee="worker")
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    claimed = kb.claim_task(conn, tid, claimer="worker")
    assert claimed is not None
    return tid


def _make_running_again(conn, tid):
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (tid,))
    assert kb.claim_task(conn, tid, claimer="worker") is not None


# ---------------------------------------------------------------------------
# Loop breaker
# ---------------------------------------------------------------------------










def test_block_loop_detected_event_emitted(kanban_home: Path) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="x", kind="capability")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason="x", kind="capability")
        events = [e for e in kb.list_events(conn, tid)
                  if e.kind == "block_loop_detected"]
        assert events, "expected a block_loop_detected event"
        payload = events[-1].payload or {}
        assert payload.get("recurrences") == 2
        assert payload.get("kind") == "capability"


def test_distinct_reasons_do_not_share_recurrence_lineage(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(
            conn,
            tid,
            reason="Which saved payment method?",
            kind="needs_input",
        )
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)

        kb.block_task(
            conn,
            tid,
            reason="Which controlled QA email?",
            kind="needs_input",
        )

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_recurrences == 1
        assert not [
            event
            for event in kb.list_events(conn, tid)
            if event.kind == "block_loop_detected"
        ]


def test_material_specification_resets_block_recurrence_lineage(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        reason = "Which saved payment method?"
        kb.block_task(conn, tid, reason=reason, kind="needs_input")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason=reason, kind="needs_input")
        assert kb.get_task(conn, tid).status == "triage"

        assert kb.specify_triage_task(
            conn,
            tid,
            body="Owner selected the corporate card.",
            author="owner",
        )
        specified = kb.get_task(conn, tid)
        assert specified.block_kind is None
        assert specified.block_recurrences == 0

        claimed = kb.claim_task(conn, tid, claimer="worker")
        assert claimed is not None
        kb.block_task(conn, tid, reason=reason, kind="needs_input")
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_recurrences == 1


def test_block_cause_fingerprint_is_normalized_and_privacy_safe(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        reason = "  Which   Controlled QA Email?  "
        kb.block_task(conn, tid, reason=reason, kind="needs_input")
        row = conn.execute(
            "SELECT block_reason_fingerprint FROM tasks WHERE id = ?",
            (tid,),
        ).fetchone()
        fingerprint = row["block_reason_fingerprint"]
        assert fingerprint
        assert reason.strip().casefold() not in fingerprint

        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(
            conn,
            tid,
            reason="which controlled qa email?",
            kind="needs_input",
        )
        assert kb.get_task(conn, tid).status == "triage"


def test_punctuation_change_starts_distinct_recurrence_lineage(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        kb.block_task(conn, tid, reason="Use the saved card?", kind="needs_input")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)

        kb.block_task(conn, tid, reason="Use the saved card!", kind="needs_input")

        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_recurrences == 1


def test_status_only_specification_preserves_same_cause_lineage(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        reason = "Still waiting for the payment method"
        kb.block_task(conn, tid, reason=reason, kind="needs_input")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason=reason, kind="needs_input")
        assert kb.get_task(conn, tid).status == "triage"

        assert kb.specify_triage_task(conn, tid)
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        kb.block_task(conn, tid, reason=reason, kind="needs_input")
        task = kb.get_task(conn, tid)
        assert task.status == "triage"
        assert task.block_recurrences == 3


def test_assignee_only_specification_preserves_same_cause_lineage(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        reason = "Still waiting for the payment method"
        kb.block_task(conn, tid, reason=reason, kind="needs_input")
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason=reason, kind="needs_input")
        assert kb.get_task(conn, tid).status == "triage"

        assert kb.specify_triage_task(conn, tid, assignee="reviewer")
        assert kb.claim_task(conn, tid, claimer="worker") is not None
        kb.block_task(conn, tid, reason=reason, kind="needs_input")
        task = kb.get_task(conn, tid)
        assert task.status == "triage"
        assert task.block_recurrences == 3


def test_legacy_recurrence_row_recovers_cause_from_latest_event(
    kanban_home: Path,
) -> None:
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        reason = "Waiting for the same credential"
        kb.block_task(conn, tid, reason=reason, kind="capability")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET block_reason_fingerprint = NULL WHERE id = ?",
                (tid,),
            )
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason=reason, kind="capability")
        assert kb.get_task(conn, tid).status == "triage"


def test_legacy_fingerprint_tolerates_invalid_utf8_block_payload(
    kanban_home: Path,
    caplog,
) -> None:
    """Corrupt BLOB payloads must not crash the legacy cause reader."""
    with kb.connect_closing() as conn:
        tid = _running_task(conn)
        reason = "Waiting for the same credential"
        kb.block_task(conn, tid, reason=reason, kind="capability")
        with kb.write_txn(conn):
            conn.execute(
                "UPDATE tasks SET block_reason_fingerprint = NULL WHERE id = ?",
                (tid,),
            )
            conn.execute(
                "UPDATE task_events SET payload = ? "
                "WHERE task_id = ? AND kind = 'blocked'",
                (b"\x80", tid),
            )
        kb.unblock_task(conn, tid)
        _make_running_again(conn, tid)
        kb.block_task(conn, tid, reason=reason, kind="capability")
        task = kb.get_task(conn, tid)
        assert task.status == "blocked"
        assert task.block_recurrences == 1
        assert "Could not recover legacy block reason fingerprint" in caplog.text
        assert reason not in caplog.text


# ---------------------------------------------------------------------------
# Dependency routing
# ---------------------------------------------------------------------------


def test_dependency_then_parent_done_promotes(kanban_home: Path) -> None:
    """A dependency-parked child becomes ready once its parent completes."""
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="parent", assignee="worker")
        child = _running_task(conn, title="child")
        kb.link_tasks(conn, parent_id=parent, child_id=child)
        kb.block_task(conn, child, reason="wait", kind="dependency")
        assert kb.get_task(conn, child).status == "todo"
        # Finish the parent, then let recompute_ready run.
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (parent,))
        kb.claim_task(conn, parent, claimer="worker")
        kb.complete_task(conn, parent, result="done")
        kb.recompute_ready(conn)
        assert kb.get_task(conn, child).status == "ready"


# ---------------------------------------------------------------------------
# Completion resets loop memory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Validation + back-compat
# ---------------------------------------------------------------------------


