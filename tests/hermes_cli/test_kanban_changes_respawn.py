"""Native changes-request handoffs must not look like duplicate PR work."""

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as dispatch


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    with kbc.connect() as db:
        yield db


def handoff(conn):
    task_id = kb.create_task(conn, title="fix review defects", assignee="builder")
    builder = kb.claim_task(conn, task_id)
    assert builder is not None
    kb.add_comment(
        conn,
        task_id,
        author="builder",
        body="https://github.com/example/repo/pull/12",
    )
    assert kb.request_review(
        conn,
        task_id,
        summary="Review exact head",
        reviewer="reviewer",
        expected_run_id=builder.current_run_id,
    )
    review = kb.claim_review_task(conn, task_id)
    assert review is not None
    assert kb.request_changes(
        conn,
        task_id,
        reason="Regression requires fix",
        expected_run_id=review.current_run_id,
    ) == (True, "builder")
    return task_id


def test_changes_requested_dispatches_original_builder_once(conn, monkeypatch):
    import hermes_cli.profiles as profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    task_id = handoff(conn)
    task = kb.get_task(conn, task_id)
    assert task is not None
    assert task.status == "ready" and task.assignee == "builder"
    assert task.block_kind is None
    assert dispatch.check_respawn_guard(conn, task_id) is None

    spawned = []

    def spawn(task, workspace):
        spawned.append(task.assignee)
        return None

    first = dispatch.dispatch_once(conn, spawn_fn=spawn)
    assert task_id in [row[0] for row in first.spawned]
    second = dispatch.dispatch_once(conn, spawn_fn=spawn)
    assert not second.spawned
    assert spawned == ["builder"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("worker_pid", 12345),
        ("claim_lock", "writer-lease"),
        ("current_run_id", 999),
        ("assignee", "other-builder"),
    ],
)
def test_native_handoff_does_not_override_writer_or_owner(conn, field, value):
    task_id = handoff(conn)
    with kb.write_txn(conn):
        conn.execute(f"UPDATE tasks SET {field} = ? WHERE id = ?", (value, task_id))
    assert dispatch.check_respawn_guard(conn, task_id) == "active_pr"


@pytest.mark.parametrize("payload", [None, "{not-json", "{}"])
def test_invalid_handoff_provenance_keeps_active_pr_guard(conn, payload):
    task_id = handoff(conn)
    with kb.write_txn(conn):
        if payload is None:
            conn.execute(
                "DELETE FROM task_events WHERE task_id = ? "
                "AND kind = 'changes_requested'",
                (task_id,),
            )
        else:
            conn.execute(
                "UPDATE task_events SET payload = ? WHERE id = ("
                "SELECT id FROM task_events WHERE task_id = ? "
                "AND kind = 'changes_requested' ORDER BY id DESC LIMIT 1)",
                (payload, task_id),
            )
    assert dispatch.check_respawn_guard(conn, task_id) == "active_pr"


def test_arbitrary_unblock_with_pr_is_not_a_review_permit(conn):
    task_id = kb.create_task(conn, title="manual recovery", assignee="builder")
    kb.add_comment(
        conn,
        task_id,
        author="reviewer",
        body="REQUEST_CHANGES https://github.com/example/repo/pull/12",
    )
    assert dispatch.check_respawn_guard(conn, task_id) == "active_pr"


def test_handoff_is_consumed_by_later_writer_run(conn):
    task_id = handoff(conn)
    writer = kb.claim_task(conn, task_id)
    assert writer is not None
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET status = 'ready', current_run_id = NULL, "
            "claim_lock = NULL, worker_pid = NULL WHERE id = ?",
            (task_id,),
        )
    assert dispatch.check_respawn_guard(conn, task_id) == "active_pr"


def test_auth_block_is_not_bypassed(conn):
    task_id = handoff(conn)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET last_failure_error = '429 rate limit exceeded' "
            "WHERE id = ?",
            (task_id,),
        )
    assert dispatch.check_respawn_guard(conn, task_id) == "blocker_auth"