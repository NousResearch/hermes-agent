"""Fail-closed finalization for dispatcher-spawned Kanban workers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import cli
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _claimed(conn, *, assignee="coder", body=None):
    task_id = kb.create_task(conn, title="work", body=body, assignee=assignee)
    task = kb.claim_task(conn, task_id)
    assert task is not None and task.current_run_id is not None
    return task_id, task.current_run_id


@pytest.mark.parametrize("assignee", ["coder", "qa"])
def test_clean_worker_exit_without_lifecycle_is_durably_failed_closed(
    kanban_home, assignee,
):
    with kb.connect() as conn:
        task_id, run_id = _claimed(conn, assignee=assignee)

        result = kb.finalize_clean_worker_exit(conn, task_id, run_id)

        assert result == "protocol_violation"
        task = kb.get_task(conn, task_id)
        run = kb.get_run(conn, run_id)
        assert task.status == "ready"
        assert task.current_run_id is None
        assert run.outcome == "crashed"
        assert run.ended_at is not None
        assert run.metadata["protocol_violation"] is True
        assert not any(e.kind == "gave_up" for e in kb.list_events(conn, task_id))


@pytest.mark.parametrize(
    ("transition", "expected"),
    [
        (lambda conn, tid, rid: kb.complete_task(conn, tid, summary="done", expected_run_id=rid), "completed"),
        (lambda conn, tid, rid: kb.block_task(conn, tid, reason="dependency", expected_run_id=rid), "blocked"),
        (lambda conn, tid, rid: kb.request_review(conn, tid, summary="review me", expected_run_id=rid), "review_requested"),
    ],
)
def test_successful_lifecycle_transition_is_preserved(
    kanban_home, transition, expected,
):
    with kb.connect() as conn:
        task_id, run_id = _claimed(conn)
        assert transition(conn, task_id, run_id)

        assert kb.finalize_clean_worker_exit(conn, task_id, run_id) == "already_finalized"
        assert kb.get_run(conn, run_id).outcome == expected


def test_transient_finalization_write_is_retried_and_verified(
    kanban_home, monkeypatch,
):
    with kb.connect() as conn:
        task_id, run_id = _claimed(conn)

    real_connect = kb.connect_closing
    attempts = 0

    def flaky_connect():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary database failure")
        return real_connect()

    monkeypatch.setattr(kb, "connect_closing", flaky_connect)
    assert cli._ensure_kanban_worker_lifecycle(
        task_id, run_id, clean_exit=True, retry_delays=(0, 0)
    ) is False
    # One failed open + one successful write + one fresh verification read.
    assert attempts == 3
    with kb.connect() as conn:
        assert kb.get_run(conn, run_id).ended_at is not None


def test_process_crash_does_not_fabricate_a_lifecycle_transition(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _claimed(conn)

    assert cli._ensure_kanban_worker_lifecycle(
        task_id, run_id, clean_exit=False, retry_delays=(0,)
    ) is True
    with kb.connect() as conn:
        assert kb.get_task(conn, task_id).status == "running"
        assert kb.get_run(conn, run_id).ended_at is None


def test_duplicate_finalization_is_idempotent(kanban_home):
    with kb.connect() as conn:
        task_id, run_id = _claimed(conn)
        assert kb.finalize_clean_worker_exit(conn, task_id, run_id) == "protocol_violation"
        event_ids = [e.id for e in kb.list_events(conn, task_id)]

        assert kb.finalize_clean_worker_exit(conn, task_id, run_id) == "protocol_violation"
        assert [e.id for e in kb.list_events(conn, task_id)] == event_ids


def test_product_signoff_task_is_never_auto_blocked_across_violation_limit(
    kanban_home,
):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="ship release",
            body="PRODUCT_SIGNOFF required from owner",
            assignee="coder",
        )
        for _ in range(kb._PROTOCOL_VIOLATION_FAILURE_LIMIT + 1):
            task = kb.claim_task(conn, task_id)
            assert task is not None and task.current_run_id is not None

            assert (
                kb.finalize_clean_worker_exit(conn, task_id, task.current_run_id)
                == "protocol_violation"
            )

            task = kb.get_task(conn, task_id)
            assert task is not None and task.status == "ready"
            assert not any(
                e.kind == "gave_up" for e in kb.list_events(conn, task_id)
            )

        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, task_id).status == "ready"


def test_finalization_and_breaker_accounting_share_one_transaction(
    kanban_home, monkeypatch,
):
    with kb.connect() as conn:
        task_id, run_id = _claimed(conn)
        conn.execute("UPDATE tasks SET max_retries = 1 WHERE id = ?", (task_id,))
        real_record_failure = kb._record_task_failure

        def require_active_transaction(inner_conn, *args, **kwargs):
            assert inner_conn.in_transaction
            return real_record_failure(inner_conn, *args, **kwargs)

        monkeypatch.setattr(kb, "_record_task_failure", require_active_transaction)

        assert kb.finalize_clean_worker_exit(conn, task_id, run_id) == "protocol_violation"
        assert kb.get_task(conn, task_id).status == "blocked"


def test_breaker_accounting_failure_rolls_back_run_finalization(
    kanban_home, monkeypatch,
):
    with kb.connect() as conn:
        task_id, run_id = _claimed(conn)
        conn.execute("UPDATE tasks SET max_retries = 1 WHERE id = ?", (task_id,))

        def fail_accounting(*_args, **_kwargs):
            raise OSError("accounting write failed")

        monkeypatch.setattr(kb, "_record_task_failure", fail_accounting)

        with pytest.raises(OSError, match="accounting write failed"):
            kb.finalize_clean_worker_exit(conn, task_id, run_id)

        assert kb.get_task(conn, task_id).status == "running"
        assert kb.get_run(conn, run_id).ended_at is None
        assert not any(
            event.kind == "protocol_violation"
            for event in kb.list_events(conn, task_id)
        )


def test_protocol_violation_limit_stays_blocked_after_recompute(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="repeated omission", assignee="qa")
        for _ in range(kb._PROTOCOL_VIOLATION_FAILURE_LIMIT):
            task = kb.claim_task(conn, task_id)
            assert task is not None and task.current_run_id is not None
            kb.finalize_clean_worker_exit(conn, task_id, task.current_run_id)

        assert kb.get_task(conn, task_id).status == "blocked"
        assert kb.recompute_ready(conn) == 0
        assert kb.get_task(conn, task_id).status == "blocked"


def test_single_query_finalizer_rejects_missing_run_identity(monkeypatch):
    worker = SimpleNamespace(_release_active_session=lambda: None, agent=None, session_id=None)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.setattr(cli, "_wait_for_oneshot_background_completions", lambda _cli: None)
    monkeypatch.setattr(cli, "_flush_one_shot_session_store", lambda _cli: None)
    monkeypatch.setattr(cli, "_notify_single_query_session_finalize", lambda _cli: None)
    monkeypatch.setattr(cli, "_run_cleanup", lambda **_kw: None)

    with pytest.raises(SystemExit) as exc:
        cli._finalize_single_query(worker)

    assert exc.value.code != 0


def test_single_query_finalizes_worker_before_cleanup(monkeypatch):
    order = []
    worker = SimpleNamespace(
        _release_active_session=lambda: order.append("release"),
        agent=None,
        session_id=None,
    )
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    monkeypatch.setattr(cli, "_wait_for_oneshot_background_completions", lambda _cli: None)
    monkeypatch.setattr(cli, "_flush_one_shot_session_store", lambda _cli: None)
    monkeypatch.setattr(cli, "_notify_single_query_session_finalize", lambda _cli: None)
    monkeypatch.setattr(
        cli,
        "_ensure_kanban_worker_lifecycle",
        lambda *a, **k: order.append("lifecycle") or True,
    )
    monkeypatch.setattr(cli, "_run_cleanup", lambda **_kw: order.append("cleanup"))

    cli._finalize_single_query(worker)

    assert order == ["lifecycle", "cleanup", "release"]


def test_single_query_finalizer_forces_nonzero_exit_on_protocol_violation(monkeypatch):
    worker = SimpleNamespace(_release_active_session=lambda: None, agent=None, session_id=None)
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    monkeypatch.setattr(cli, "_wait_for_oneshot_background_completions", lambda _cli: None)
    monkeypatch.setattr(cli, "_flush_one_shot_session_store", lambda _cli: None)
    monkeypatch.setattr(cli, "_notify_single_query_session_finalize", lambda _cli: None)
    monkeypatch.setattr(cli, "_run_cleanup", lambda **_kw: None)
    monkeypatch.setattr(cli, "_ensure_kanban_worker_lifecycle", lambda *a, **k: False)

    with pytest.raises(SystemExit) as exc:
        cli._finalize_single_query(worker)

    assert exc.value.code != 0
