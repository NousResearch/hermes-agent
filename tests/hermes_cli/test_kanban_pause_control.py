"""Durable, fail-closed state for the Kanban-only dispatch control."""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_control
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    connection = kbc.connect()
    try:
        yield connection
    finally:
        with contextlib.suppress(sqlite3.ProgrammingError):
            connection.close()


def test_missing_control_row_is_running_revision_zero(conn):
    before = conn.total_changes

    state = kanban_control.read_control(conn)

    assert state.state == "running"
    assert state.revision == 0
    assert state.changed_at is None
    assert state.changed_by is None
    assert state.reason is None
    assert state.error_code is None
    assert conn.total_changes == before


def test_existing_database_gains_control_schema_without_state_write(tmp_path):
    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(kb.SCHEMA_SQL.replace(kanban_control.CONTROL_SCHEMA_SQL, ""))
    legacy.close()
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))

    migrated = kbc.connect(path)
    try:
        state = kanban_control.read_control(migrated)
        tables = {
            row[0] for row in migrated.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            )
        }
        assert "kanban_dispatch_control" in tables
        assert "kanban_dispatch_control_events" in tables
        assert state.state == "running" and state.revision == 0
        assert migrated.execute(
            "SELECT COUNT(*) FROM kanban_dispatch_control",
        ).fetchone()[0] == 0
    finally:
        migrated.close()


def test_status_reads_legacy_database_without_schema_or_file_mutation(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    path = kb.kanban_db_path(board="default")
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy = sqlite3.connect(path)
    legacy.executescript(kb.SCHEMA_SQL.replace(kanban_control.CONTROL_SCHEMA_SQL, ""))
    legacy.close()
    before = path.read_bytes()

    result = kanban_control.control_status(board="default")

    assert result["state"] == "running"
    assert result["boards"][0]["revision"] == 0
    assert path.read_bytes() == before
    verify = sqlite3.connect(path)
    try:
        assert verify.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name LIKE 'kanban_dispatch_control%'",
        ).fetchone()[0] == 0
    finally:
        verify.close()


def test_pause_persists_and_records_one_matching_event(conn):
    paused = kanban_control.pause_board(
        conn, changed_by="operator@example", reason="maintenance",
    )
    conn.close()

    reopened = kbc.connect()
    try:
        observed = kanban_control.read_control(reopened)
        events = reopened.execute(
            "SELECT revision, old_state, new_state, changed_by, reason, changed_at "
            "FROM kanban_dispatch_control_events ORDER BY revision"
        ).fetchall()
    finally:
        reopened.close()

    assert paused == observed
    assert observed.state == "paused"
    assert observed.revision == 1
    assert observed.changed_by == "operator@example"
    assert observed.reason == "maintenance"
    assert len(events) == 1
    assert dict(events[0]) == {
        "revision": observed.revision,
        "old_state": "running",
        "new_state": "paused",
        "changed_by": observed.changed_by,
        "reason": observed.reason,
        "changed_at": observed.changed_at,
    }


def test_repeated_pause_is_idempotent_and_resume_advances_revision(conn):
    first = kanban_control.pause_board(conn, changed_by="one", reason="hold")
    repeated = kanban_control.pause_board(conn, changed_by="two", reason="changed text")
    resumed = kanban_control.resume_board(conn, changed_by="three", reason="clear")
    conn.close()

    reopened = kbc.connect()
    try:
        observed = kanban_control.read_control(reopened)
        events = reopened.execute(
            "SELECT revision, old_state, new_state FROM kanban_dispatch_control_events "
            "ORDER BY revision"
        ).fetchall()
    finally:
        reopened.close()

    assert repeated == first
    assert resumed == observed
    assert observed.state == "running"
    assert observed.revision == 2
    assert [(row["revision"], row["old_state"], row["new_state"]) for row in events] == [
        (1, "running", "paused"),
        (2, "paused", "running"),
    ]


@pytest.mark.parametrize(
    "corrupt_sql",
    [
        "UPDATE kanban_dispatch_control SET state='stopped' WHERE id=1",
        "UPDATE kanban_dispatch_control SET revision=-1 WHERE id=1",
    ],
)
def test_invalid_control_row_reads_unknown_without_repair(conn, corrupt_sql):
    kanban_control.pause_board(conn, changed_by="operator")
    conn.execute("PRAGMA ignore_check_constraints=ON")
    conn.execute(corrupt_sql)
    conn.execute("PRAGMA ignore_check_constraints=OFF")
    before = conn.total_changes

    state = kanban_control.read_control(conn)

    assert state.state == "unknown"
    assert state.error_code == "invalid_control_state"
    assert conn.total_changes == before


def test_schema_drift_reads_unknown_without_repair(conn):
    conn.execute("DROP TABLE kanban_dispatch_control")
    conn.execute("CREATE TABLE kanban_dispatch_control (id INTEGER PRIMARY KEY, state TEXT)")
    conn.execute("INSERT INTO kanban_dispatch_control VALUES (1, 'paused')")
    before = conn.total_changes

    state = kanban_control.read_control(conn)

    assert state.state == "unknown"
    assert state.error_code == "control_schema_invalid"
    assert conn.total_changes == before


def test_missing_audit_table_fails_control_closed(conn):
    kanban_control.pause_board(conn, changed_by="operator")
    conn.execute("DROP TABLE kanban_dispatch_control_events")

    state = kanban_control.read_control(conn)

    assert state.state == "unknown"
    assert state.error_code == "control_schema_invalid"


@pytest.mark.parametrize("column", ["revision", "changed_at"])
def test_fractional_integer_control_values_fail_closed(conn, column):
    kanban_control.pause_board(conn, changed_by="operator")
    conn.execute("PRAGMA ignore_check_constraints=ON")
    conn.execute(f"UPDATE kanban_dispatch_control SET {column}=1.5 WHERE id=1")
    conn.execute("PRAGMA ignore_check_constraints=OFF")

    state = kanban_control.read_control(conn)

    assert state.state == "unknown"
    assert state.error_code == "invalid_control_state"


def _queued_task(conn, status):
    task_id = kb.create_task(conn, title=f"{status} task", assignee="worker")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status=?, claim_lock=NULL, claim_expires=NULL, "
            "current_run_id=NULL WHERE id=?",
            (status, task_id),
        )
    return task_id


def _claim_path_snapshot(conn, task_id):
    task = dict(conn.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone())
    runs = [dict(row) for row in conn.execute(
        "SELECT * FROM task_runs WHERE task_id=? ORDER BY id", (task_id,),
    )]
    events = [dict(row) for row in conn.execute(
        "SELECT * FROM task_events WHERE task_id=? ORDER BY id", (task_id,),
    )]
    return task, runs, events


def test_paused_ready_claim_returns_typed_reason_without_mutation(conn):
    task_id = _queued_task(conn, "ready")
    kanban_control.pause_board(conn, changed_by="operator")
    before = _claim_path_snapshot(conn, task_id)

    result = kb.claim_task_result(conn, task_id, claimer="worker")

    assert result.task is None
    assert result.reason == "kanban_paused"
    assert _claim_path_snapshot(conn, task_id) == before


def test_paused_review_claim_returns_typed_reason_without_mutation(conn):
    task_id = _queued_task(conn, "review")
    kanban_control.pause_board(conn, changed_by="operator")
    before = _claim_path_snapshot(conn, task_id)

    result = kb.claim_review_task_result(conn, task_id, claimer="reviewer")

    assert result.task is None
    assert result.reason == "kanban_paused"
    assert _claim_path_snapshot(conn, task_id) == before


def test_unknown_control_fails_both_claim_lanes_closed(conn):
    ready_id = _queued_task(conn, "ready")
    review_id = _queued_task(conn, "review")
    conn.execute("DROP TABLE kanban_dispatch_control")
    conn.execute("CREATE TABLE kanban_dispatch_control (id INTEGER PRIMARY KEY, state TEXT)")
    before_ready = _claim_path_snapshot(conn, ready_id)
    before_review = _claim_path_snapshot(conn, review_id)

    ready = kb.claim_task_result(conn, ready_id, claimer="worker")
    review = kb.claim_review_task_result(conn, review_id, claimer="reviewer")

    assert ready.reason == "kanban_control_unknown"
    assert review.reason == "kanban_control_unknown"
    assert _claim_path_snapshot(conn, ready_id) == before_ready
    assert _claim_path_snapshot(conn, review_id) == before_review


def test_pause_gate_precedes_parent_demotion_and_reason_is_immutable(conn):
    parent_id = _queued_task(conn, "done")
    child_id = _queued_task(conn, "ready")
    conn.execute(
        "INSERT INTO task_links(parent_id, child_id) VALUES(?, ?)",
        (parent_id, child_id),
    )
    conn.execute("UPDATE tasks SET status='running' WHERE id=?", (parent_id,))
    conn.commit()
    kanban_control.pause_board(conn, changed_by="operator")
    before_events = conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=?", (child_id,),
    ).fetchone()[0]

    paused = kb.claim_task_result(conn, child_id, claimer="worker")
    kanban_control.resume_board(conn, changed_by="operator")

    assert paused.reason == "kanban_paused"
    child = kb.get_task(conn, child_id)
    assert child is not None and child.status == "ready"
    assert conn.execute(
        "SELECT COUNT(*) FROM task_events WHERE task_id=?", (child_id,),
    ).fetchone()[0] == before_events

    rejected = kb.claim_task_result(conn, child_id, claimer="worker")
    assert rejected.reason == "parents_not_done"
    child = kb.get_task(conn, child_id)
    assert child is not None and child.status == "todo"


def test_admission_refusal_and_ordinary_claim_loss_are_distinct(conn, monkeypatch):
    from hermes_cli import kanban_ready_monitor as monitor

    monitor.ensure_schema(conn)
    admission_task = _queued_task(conn, "ready")
    refused = kb.claim_task_result(
        conn, admission_task, claimer="worker", monitor_attempt_id="missing-attempt",
    )
    assert refused.reason == "admission_refused"
    assert conn.execute(
        "SELECT COUNT(*) FROM kanban_claim_admissions WHERE task_id=?",
        (admission_task,),
    ).fetchone()[0] == 0

    lost_task = _queued_task(conn, "ready")
    monkeypatch.setattr(kb, "_claim_and_open_run", lambda *args, **kwargs: None)
    lost = kb.claim_task_result(conn, lost_task, claimer="worker")
    assert lost.reason == "claim_lost"


def test_resume_permits_ready_and_review_compatibility_claims(conn):
    ready_id = _queued_task(conn, "ready")
    review_id = _queued_task(conn, "review")
    kanban_control.pause_board(conn, changed_by="operator")
    kanban_control.resume_board(conn, changed_by="operator")

    ready = kb.claim_task(conn, ready_id, claimer="worker")
    review = kb.claim_review_task(conn, review_id, claimer="reviewer")

    assert ready is not None and ready.status == "running"
    assert review is not None and review.status == "running"


def test_pause_does_not_modify_an_existing_running_claim(conn):
    task_id = _queued_task(conn, "ready")
    claimed = kb.claim_task(conn, task_id, claimer="worker")
    assert claimed is not None
    before = _claim_path_snapshot(conn, task_id)

    kanban_control.pause_board(conn, changed_by="operator")

    assert _claim_path_snapshot(conn, task_id) == before


def test_claim_and_pause_race_has_one_linearized_outcome(tmp_path, monkeypatch):
    path = tmp_path / "board.db"
    setup = kbc.connect(path)
    task_id = _queued_task(setup, "ready")
    setup.close()

    entered_claim = threading.Event()
    release_claim = threading.Event()
    original = kb._claim_and_open_run

    def delayed_claim(*args, **kwargs):
        entered_claim.set()
        assert release_claim.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(kb, "_claim_and_open_run", delayed_claim)
    outcomes = {}

    def claim_worker():
        claim_conn = kbc.connect(path)
        outcomes["claim"] = kb.claim_task_result(
            claim_conn, task_id, claimer="worker",
        )
        claim_conn.close()

    def pause_worker():
        pause_conn = kbc.connect(path)
        outcomes["pause"] = kanban_control.pause_board(
            pause_conn, changed_by="operator",
        )
        pause_conn.close()

    claim_thread = threading.Thread(target=claim_worker)
    pause_thread = threading.Thread(target=pause_worker)
    claim_thread.start()
    assert entered_claim.wait(5)
    pause_thread.start()
    release_claim.set()
    claim_thread.join(5)
    pause_thread.join(5)
    assert not claim_thread.is_alive() and not pause_thread.is_alive()

    verify = kbc.connect(path)
    task = kb.get_task(verify, task_id)
    state = kanban_control.read_control(verify)
    verify.close()
    assert outcomes["claim"].task is not None
    assert outcomes["pause"].state == "paused"
    assert task is not None and task.status == "running"
    assert state.state == "paused"


@pytest.mark.parametrize(
    ("status", "claim_name"),
    [("ready", "claim_task_result"), ("review", "claim_review_task_result")],
)
def test_pause_first_race_refuses_both_claim_lanes_without_mutation(
    tmp_path, monkeypatch, status, claim_name,
):
    path = tmp_path / f"{status}.db"
    setup = kbc.connect(path)
    task_id = _queued_task(setup, status)
    before = _claim_path_snapshot(setup, task_id)
    setup.close()
    entered_pause = threading.Event()
    release_pause = threading.Event()
    original_read = kanban_control.read_control

    def blocked_pause_read(connection):
        if threading.current_thread().name == "pause-writer" and not entered_pause.is_set():
            entered_pause.set()
            assert release_pause.wait(5)
        return original_read(connection)

    monkeypatch.setattr(kanban_control, "read_control", blocked_pause_read)
    outcomes = {}

    def pause_worker():
        connection = kbc.connect(path)
        outcomes["pause"] = kanban_control.pause_board(
            connection, changed_by="operator",
        )
        connection.close()

    def claim_worker():
        connection = kbc.connect(path)
        outcomes["claim"] = getattr(kb, claim_name)(
            connection, task_id, claimer="worker",
        )
        connection.close()

    pause_thread = threading.Thread(target=pause_worker, name="pause-writer")
    claim_thread = threading.Thread(target=claim_worker, name="claim-writer")
    pause_thread.start()
    assert entered_pause.wait(5)
    claim_thread.start()
    release_pause.set()
    pause_thread.join(5)
    claim_thread.join(5)
    assert not pause_thread.is_alive() and not claim_thread.is_alive()

    verify = kbc.connect(path)
    try:
        assert outcomes["pause"].state == "paused"
        assert outcomes["claim"].reason == "kanban_paused"
        assert _claim_path_snapshot(verify, task_id) == before
    finally:
        verify.close()


def test_resume_first_race_creates_exactly_one_consistent_run(tmp_path, monkeypatch):
    path = tmp_path / "resume.db"
    setup = kbc.connect(path)
    task_id = _queued_task(setup, "ready")
    kanban_control.pause_board(setup, changed_by="operator")
    setup.close()
    entered_resume = threading.Event()
    release_resume = threading.Event()
    original_read = kanban_control.read_control

    def blocked_resume_read(connection):
        if threading.current_thread().name == "resume-writer" and not entered_resume.is_set():
            entered_resume.set()
            assert release_resume.wait(5)
        return original_read(connection)

    monkeypatch.setattr(kanban_control, "read_control", blocked_resume_read)
    outcomes = {}

    def resume_worker():
        connection = kbc.connect(path)
        outcomes["resume"] = kanban_control.resume_board(
            connection, changed_by="operator",
        )
        connection.close()

    def claim_worker():
        connection = kbc.connect(path)
        outcomes["claim"] = kb.claim_task_result(
            connection, task_id, claimer="worker",
        )
        connection.close()

    resume_thread = threading.Thread(target=resume_worker, name="resume-writer")
    claim_thread = threading.Thread(target=claim_worker, name="claim-writer")
    resume_thread.start()
    assert entered_resume.wait(5)
    claim_thread.start()
    release_resume.set()
    resume_thread.join(5)
    claim_thread.join(5)
    assert not resume_thread.is_alive() and not claim_thread.is_alive()

    verify = kbc.connect(path)
    try:
        task = kb.get_task(verify, task_id)
        runs = verify.execute(
            "SELECT id, status FROM task_runs WHERE task_id=?", (task_id,),
        ).fetchall()
        assert outcomes["resume"].state == "running"
        assert outcomes["claim"].task is not None
        assert task is not None and task.status == "running"
        assert task.current_run_id == runs[0]["id"]
        assert [(row["status"]) for row in runs] == ["running"]
    finally:
        verify.close()
