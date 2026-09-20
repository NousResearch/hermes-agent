"""Tamper detection for kanban.db writes that bypass the kernel (#110080).

The completion gate lives in the tool/CLI path only; SQLite accepts writes from
any process, so a refused worker can just run
``UPDATE tasks SET status='done' ...`` itself. These tests pin the read-side
detection: the kernel's own flow stays silent, while a raw status flip — or a
raw event INSERT/UPDATE/DELETE — surfaces as a diagnostic.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_diagnostics as kd

TAMPER_KINDS = {"out_of_band_transition", "event_chain_tampered"}


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def conn(kanban_home):
    c = kbc.connect()
    try:
        yield c
    finally:
        c.close()


def _events(conn, task_id):
    return kb.list_events(conn, task_id)


def _diags(conn, task_id):
    return kd.compute_task_diagnostics(
        kb.get_task(conn, task_id), _events(conn, task_id), kb.list_runs(conn, task_id),
    )


def _tamper_kinds(conn, task_id) -> set[str]:
    return {d.kind for d in _diags(conn, task_id)} & TAMPER_KINDS


def _claimed_task(conn) -> str:
    tid = kb.create_task(conn, title="ship it", assignee="worker")
    assert kb.claim_task(conn, tid, claimer="worker:1") is not None
    return tid


def _direct_done(conn, tid: str) -> None:
    """The incident: raw SQL once the completion gate refused the transition."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'done', completed_at = ? WHERE id = ?",
            (int(time.time()), tid),
        )


def test_kernel_completion_reports_nothing(conn):
    tid = _claimed_task(conn)
    assert kb.complete_task(conn, tid, result="shipped", summary="shipped")
    assert kb.get_task(conn, tid).status == "done"
    assert kb.verify_event_chain(tid, _events(conn, tid)) == []
    assert _tamper_kinds(conn, tid) == set()


def test_completion_gate_refusal_then_raw_update_is_flagged(conn):
    parent = kb.create_task(conn, title="parent", assignee="worker")
    assert kb.claim_task(conn, parent, claimer="worker:0") is not None
    tid = kb.create_task(conn, title="child", assignee="worker", parents=[parent])
    assert kb.claim_task(conn, tid, claimer="worker:1") is None  # parent not done
    assert kb.complete_task(conn, tid, result="done!") is False  # the gate refuses
    _direct_done(conn, tid)

    flagged = [d for d in _diags(conn, tid) if d.kind == "out_of_band_transition"]
    assert len(flagged) == 1
    assert flagged[0].severity == "critical"
    # 3 events: created, dependency_wait (create_task records the open parent), claim_rejected.
    assert flagged[0].data == {"status": "done", "expected_event": "completed", "event_count": 3}
    # The tasks row was forged; the event log itself is untouched.
    assert kb.verify_event_chain(tid, _events(conn, tid)) == []

    # Same verdict on the sqlite3.Row objects the dashboard/CLI fleet path passes.
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
    rows = list(conn.execute(
        "SELECT * FROM task_events WHERE task_id = ? ORDER BY id", (tid,),
    ).fetchall())
    assert any(d.kind == "out_of_band_transition" for d in kd.compute_task_diagnostics(row, rows, []))


def test_forged_completed_event_is_caught_by_the_chain(conn):
    """Faking the event too (raw INSERT) satisfies the status/event pairing and
    is caught by the hash chain instead."""
    tid = _claimed_task(conn)
    with kb.write_txn(conn):
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, 'completed', NULL, ?)",
            (tid, int(time.time())),
        )
    _direct_done(conn, tid)

    assert [f["kind"] for f in kb.verify_event_chain(tid, _events(conn, tid))] == ["unsigned_event"]
    assert "event_chain_tampered" in _tamper_kinds(conn, tid)
    # The forged event does satisfy the pairing rule, so the reported
    # out_of_band_transition is the row's divergence from its commitment — the
    # second rail sees the raw flip even when the log was faked to look right.
    flagged = [d for d in _diags(conn, tid) if d.kind == "out_of_band_transition"]
    assert [d.data["diverged_fields"] for d in flagged] == [["status"]]


def test_edited_event_payload_is_flagged(conn):
    tid = _claimed_task(conn)
    assert kb.complete_task(conn, tid, result="ok")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_events SET payload = '{\"forged\": true}' WHERE id = ("
            " SELECT id FROM task_events WHERE task_id = ? AND kind = 'completed'"
            " ORDER BY id DESC LIMIT 1)",
            (tid,),
        )

    assert [f["kind"] for f in kb.verify_event_chain(tid, _events(conn, tid))] == ["hash_mismatch"]
    assert "event_chain_tampered" in _tamper_kinds(conn, tid)


def test_deleted_event_is_flagged(conn):
    tid = kb.create_task(conn, title="audit log", assignee="worker")
    for kind in ("one", "two", "three"):
        kb._append_event(conn, tid, kind)
    with kb.write_txn(conn):
        conn.execute("DELETE FROM task_events WHERE task_id = ? AND kind = 'two'", (tid,))

    assert [f["kind"] for f in kb.verify_event_chain(tid, _events(conn, tid))] == ["chain_broken"]


def test_legacy_and_gc_pruned_rows_are_not_false_positives(conn):
    """Rows written before the chain (NULL hash) and a gc-pruned prefix must
    stay silent — detection may not fire on boards that predate the feature."""
    tid = _claimed_task(conn)
    assert kb.complete_task(conn, tid, result="ok")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_events SET prev_hash = NULL, event_hash = NULL WHERE task_id = ?", (tid,),
        )
        # A pre-feature row carries no commitment either: the chain columns AND
        # the card's tip/snapshot were added by the same migration, so a legacy
        # board has all four NULL.
        conn.execute(
            "UPDATE tasks SET event_chain_tip = NULL, board_state_snapshot = NULL WHERE id = ?", (tid,),
        )
    assert kb.verify_event_chain(tid, _events(conn, tid)) == []
    assert kb.verify_event_chain(tid, _events(conn, tid), task=kb.get_task(conn, tid)) == []
    assert kb.board_state_divergence(kb.get_task(conn, tid)) == {}
    # First chained row after the legacy block: still no findings, no diagnostic.
    kb.add_comment(conn, tid, "human", "LGTM")
    assert kb.verify_event_chain(tid, _events(conn, tid), task=kb.get_task(conn, tid)) == []
    assert _tamper_kinds(conn, tid) == set()

    # gc prunes the prefix but keeps the terminal event (its audit anchor).
    assert kb.gc_events(conn, older_than_seconds=1) >= 0
    assert kb.get_task(conn, tid).status == "done"
    assert kb.verify_event_chain(tid, _events(conn, tid), task=kb.get_task(conn, tid)) == []
    assert _tamper_kinds(conn, tid) == set()


def test_disclosed_triage_to_ready_raw_write_is_flagged(conn):
    """The incident behind the #110080 amendment: ``default`` moved a card
    ``triage -> ready`` with direct SQLite because no verb exits ``triage``.
    Nothing terminal is involved, so the status/event pairing cannot see it —
    the card's commitment can."""
    tid = kb.create_task(conn, title="exit triage", assignee="worker", triage=True)
    assert kb.get_task(conn, tid).status == "triage"
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))

    flagged = [d for d in _diags(conn, tid) if d.kind == "out_of_band_transition"]
    assert len(flagged) == 1
    assert flagged[0].severity == "critical"
    assert flagged[0].data["diverged_fields"] == ["status"]
    assert flagged[0].data["divergence"]["status"] == {"committed": "'triage'", "found": "'ready'"}
    # The audit log is untouched and its chain verifies: only the card row lies.
    assert kb.verify_event_chain(tid, _events(conn, tid), task=kb.get_task(conn, tid)) == []

    # Same verdict on the sqlite3.Row the dashboard/CLI fleet path passes.
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert any(d.kind == "out_of_band_transition" for d in kd.compute_task_diagnostics(row, _events(conn, tid), []))


@pytest.mark.parametrize("sql,value,field", [
    ("priority = ?", 7, "priority"),
    ("title = ?", "forged title", "title"),
    ("body = ?", "forged body", "body"),
    ("assignee = ?", "someone-else", "assignee"),
    ("result = ?", "APPROVED - all ACs met", "result"),
])
def test_direct_field_writes_are_flagged(conn, sql, value, field):
    """Every protected board field is reconciled against the kernel's commitment,
    not just the terminal status the original example used."""
    tid = kb.create_task(conn, title="field watch", assignee="worker")
    with kb.write_txn(conn):
        conn.execute(f"UPDATE tasks SET {sql} WHERE id = ?", (value, tid))

    flagged = [d for d in _diags(conn, tid) if d.kind == "out_of_band_transition"]
    assert [d.data["diverged_fields"] for d in flagged] == [[field]]


def test_kernel_field_writes_stay_silent(conn):
    """Kernel mutators — including the direct-SQL dashboard paths, which now
    emit through ``_append_event`` — re-commit what they just wrote."""
    tid = kb.create_task(conn, title="kernel edits", assignee="worker")
    assert kb.assign_task(conn, tid, "other-worker")  # 'assigned'
    assert kb.claim_task(conn, tid, claimer="other-worker:1") is not None
    with kb.write_txn(conn):  # plugins/kanban/dashboard/plugin_api._set_priority
        conn.execute("UPDATE tasks SET priority = ? WHERE id = ?", (3, tid))
        kb._append_event(conn, tid, "reprioritized", {"priority": 3})
    with kb.write_txn(conn):  # plugins/kanban/dashboard/plugin_api._patch_title_body
        conn.execute("UPDATE tasks SET title = ?, body = ? WHERE id = ?", ("new title", "new body", tid))
        kb._append_event(conn, tid, "edited", None)
    with kb.write_txn(conn):  # plugins/kanban/dashboard/plugin_api._set_status_direct
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", ("review", tid))
        kb._append_event(conn, tid, "status", {"status": "review", "requested_status": "review"})
    assert kb.complete_task(conn, tid, result="ok")
    assert kb.edit_completed_task_result(conn, tid, result="ok, reworded")
    kb.add_comment(conn, tid, "human", "LGTM")
    kb.gc_events(conn, older_than_seconds=1)

    assert kb.board_state_divergence(kb.get_task(conn, tid)) == {}
    assert kb.verify_event_chain(tid, _events(conn, tid), task=kb.get_task(conn, tid)) == []
    assert _tamper_kinds(conn, tid) == set()


def test_deleted_tail_event_is_flagged(conn):
    """Deleting the newest hashed event leaves no successor whose ``prev_hash``
    can expose the missing link, so the prefix still verifies clean — the tip
    the kernel committed on the card is the surviving witness."""
    tid = kb.create_task(conn, title="audit tail", assignee="worker")
    for kind in ("one", "two", "three"):
        kb._append_event(conn, tid, kind)
    assert kb.verify_event_chain(tid, _events(conn, tid), task=kb.get_task(conn, tid)) == []

    with kb.write_txn(conn):
        conn.execute("DELETE FROM task_events WHERE task_id = ? AND kind = 'three'", (tid,))

    # The remaining prefix is internally consistent: only the tip disagrees.
    assert kb.verify_event_chain(tid, _events(conn, tid)) == []
    findings = kb.verify_event_chain(tid, _events(conn, tid), task=kb.get_task(conn, tid))
    assert [f["kind"] for f in findings] == ["chain_tail_missing"]
    assert _tamper_kinds(conn, tid) == {"event_chain_tampered"}


def test_delete_tail_then_append_does_not_heal(conn):
    """The next legitimate append chains onto the surviving row; the tip must not
    be re-committed over the evidence."""
    tid = kb.create_task(conn, title="audit tail then append", assignee="worker")
    for kind in ("one", "two", "three"):
        kb._append_event(conn, tid, kind)
    with kb.write_txn(conn):
        conn.execute("DELETE FROM task_events WHERE task_id = ? AND kind = 'three'", (tid,))
    kb.add_comment(conn, tid, "human", "post-delete")

    assert [f["kind"] for f in kb.verify_event_chain(
        tid, _events(conn, tid), task=kb.get_task(conn, tid))] == ["chain_tail_missing"]
    assert _tamper_kinds(conn, tid) == {"event_chain_tampered"}


def test_existing_board_gains_chain_columns(kanban_home):
    db_path = kb.kanban_db_path()
    with kbc.connect_closing() as c:
        c.execute("ALTER TABLE task_events DROP COLUMN event_hash")
        c.execute("ALTER TABLE task_events DROP COLUMN prev_hash")
        c.execute("ALTER TABLE tasks DROP COLUMN board_state_snapshot")
        c.execute("ALTER TABLE tasks DROP COLUMN event_chain_tip")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))

    with kbc.connect_closing() as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(task_events)")}
        assert {"prev_hash", "event_hash"} <= cols
        task_cols = {r["name"] for r in c.execute("PRAGMA table_info(tasks)")}
        assert {"board_state_snapshot", "event_chain_tip"} <= task_cols

    with kbc.connect_closing() as c:
        tid = kb.create_task(c, title="post-migration", assignee="worker")
        assert any(e.event_hash for e in _events(c, tid))
        assert kb.verify_event_chain(tid, _events(c, tid), task=kb.get_task(c, tid)) == []
