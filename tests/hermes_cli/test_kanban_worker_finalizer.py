"""Tests for the Kanban worker process-exit finalizer."""

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


def test_unclosed_kanban_worker_is_blocked_by_process_finalizer(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.main as main_mod

    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="silent worker", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    assert main_mod._finalize_unclosed_kanban_worker() is True

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)
        events = kb.list_events(conn, tid)

    assert task.status == "blocked"
    assert task.current_run_id is None
    assert task.block_kind == "capability"
    assert run.status == "blocked"
    assert run.outcome == "blocked"
    assert "kanban_complete" in (run.summary or "")
    assert "kanban_block" in (run.summary or "")
    assert events[-1].kind == "blocked"


def test_process_finalizer_noops_after_worker_already_completed(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.main as main_mod

    with kb.connect_closing() as conn:
        tid = kb.create_task(conn, title="done worker", assignee="worker")
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None
        assert kb.complete_task(
            conn,
            tid,
            summary="finished normally",
            expected_run_id=run_id,
        )

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))

    assert main_mod._finalize_unclosed_kanban_worker() is False

    with kb.connect_closing() as conn:
        assert kb.get_task(conn, tid).status == "done"


def test_process_finalizer_skips_goal_mode_workers(
    kanban_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hermes_cli.main as main_mod

    with kb.connect_closing() as conn:
        tid = kb.create_task(
            conn, title="goal turn", assignee="worker", goal_mode=True,
        )
        claimed = kb.claim_task(conn, tid)
        assert claimed is not None
        run_id = claimed.current_run_id
        assert run_id is not None

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")

    assert main_mod._finalize_unclosed_kanban_worker() is False

    with kb.connect_closing() as conn:
        task = kb.get_task(conn, tid)
        run = kb.latest_run(conn, tid)

    assert task.status == "running"
    assert task.current_run_id == run_id
    assert run.status == "running"
