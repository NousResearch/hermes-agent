"""Real-board regressions for the Kanban stop guard's run identity fence."""

from __future__ import annotations

from agent.kanban_stop import build_kanban_stop_nudge


def test_terminal_reviewer_run_cannot_nudge_or_mutate_immediate_builder_successor(
    tmp_path, monkeypatch,
):
    """A real review handoff closes the reviewer run before its successor claims.

    The stop guard receives the old reviewer's exact run id. It must inspect
    that run rather than the task's current status/run, suppress its nudge,
    and leave the already-claimed builder successor untouched. The CAS probes
    additionally prove late terminal calls from that old run cannot mutate the
    successor.
    """
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    db_path = tmp_path / "kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    with kbc.connect(db_path) as conn:
        task_id = kb.create_task(conn, title="guarded review", assignee="builder")
        implementation = kb.claim_task(conn, task_id, claimer="builder:implementation")
        assert implementation is not None
        assert kb.request_review(
            conn,
            task_id,
            summary="ready",
            reviewer="reviewer",
            expected_run_id=implementation.current_run_id,
        )
        reviewer = kb.claim_review_task(conn, task_id, claimer="reviewer:terminal")
        assert reviewer is not None
        reviewer_run_id = reviewer.current_run_id
        assert kb.request_changes(
            conn,
            task_id,
            reason="add the regression",
            expected_run_id=reviewer_run_id,
        ) == (True, "builder")
        successor = kb.claim_task(conn, task_id, claimer="builder:successor")
        assert successor is not None
        successor_run_id = successor.current_run_id
        before = kb.get_task(conn, task_id)
        assert before is not None
        assert before.status == "running"
        assert before.current_run_id == successor_run_id
        assert before.claim_lock == "builder:successor"
        events_before = list(kb.list_events(conn, task_id))

        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(reviewer_run_id))
        assert build_kanban_stop_nudge(messages=[]) is None

        # A delayed reviewer must lose every terminal CAS race to successor.
        assert not kb.complete_task(
            conn, task_id, summary="stale", expected_run_id=reviewer_run_id,
        )
        assert not kb.block_task(
            conn, task_id, reason="stale", expected_run_id=reviewer_run_id,
        )
        assert kb.request_changes(
            conn, task_id, reason="stale", expected_run_id=reviewer_run_id,
        )[0] is False
        assert not kb.request_review(
            conn, task_id, summary="stale", expected_run_id=reviewer_run_id,
        )

        after = kb.get_task(conn, task_id)
        assert after is not None
        assert after.status == "running"
        assert after.current_run_id == successor_run_id
        assert after.claim_lock == "builder:successor"
        assert kb.list_events(conn, task_id) == events_before


def test_terminal_review_requested_run_suppresses_nudge_with_real_board(
    tmp_path, monkeypatch,
):
    """The implementer's review_requested run is terminal even while card is review."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    db_path = tmp_path / "review-requested.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    with kbc.connect(db_path) as conn:
        task_id = kb.create_task(conn, title="review handoff", assignee="builder")
        implementation = kb.claim_task(conn, task_id)
        assert implementation is not None
        assert kb.request_review(
            conn, task_id, summary="ready", expected_run_id=implementation.current_run_id,
        )
        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(implementation.current_run_id))
        assert build_kanban_stop_nudge(messages=[]) is None


def test_open_real_run_keeps_stop_guard_enforcement(tmp_path, monkeypatch):
    """A live worker with no terminal call remains subject to the guard."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    db_path = tmp_path / "open-run.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    with kbc.connect(db_path) as conn:
        task_id = kb.create_task(conn, title="live worker", assignee="builder")
        worker = kb.claim_task(conn, task_id)
        assert worker is not None
        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(worker.current_run_id))
        assert build_kanban_stop_nudge(messages=[]) is not None


def test_open_outcome_is_not_cached_before_later_terminal_transition(tmp_path, monkeypatch):
    """A prior open read must not hide the same run closing later (no TOCTOU cache)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    db_path = tmp_path / "transition.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    with kbc.connect(db_path) as conn:
        task_id = kb.create_task(conn, title="transitioning worker", assignee="builder")
        worker = kb.claim_task(conn, task_id)
        assert worker is not None
        run_id = worker.current_run_id
        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
        assert build_kanban_stop_nudge(messages=[]) is not None
        assert kb.complete_task(conn, task_id, summary="done", expected_run_id=run_id)
        assert build_kanban_stop_nudge(messages=[]) is None


def test_foreign_task_run_does_not_suppress_nudge(tmp_path, monkeypatch):
    from hermes_cli import kanban_db as kb
    from hermes_cli import kanban_db_connect as kbc

    db_path = tmp_path / "foreign-run.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    with kbc.connect(db_path) as conn:
        other_id = kb.create_task(conn, title="other", assignee="builder")
        other = kb.claim_task(conn, other_id)
        assert other is not None
        assert kb.complete_task(conn, other_id, summary="done", expected_run_id=other.current_run_id)
        task_id = kb.create_task(conn, title="my task", assignee="builder")
        assert kb.claim_task(conn, task_id) is not None

        monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(other.current_run_id))
        assert build_kanban_stop_nudge(messages=[]) is not None
