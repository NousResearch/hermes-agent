from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _spawned_profiles(conn, spawn_fn, **dispatch_kwargs):
    res = kbd.dispatch_once(conn, spawn_fn=spawn_fn, **dispatch_kwargs)
    return res, [who for _tid, who, _ws in res.spawned]


def test_first_review_request_with_no_reviewer_dispatches_review_assignee(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl a feature", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        kb.request_review(conn, tid, summary="done", expected_run_id=run_id)

        row = conn.execute("SELECT assignee, status FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["status"] == "review"
        assert row["assignee"] == "worker"

        spawned = []

        def spawn_fn(task, workspace, board=None):
            spawned.append(task.assignee)

        res, who = _spawned_profiles(conn, spawn_fn, review_assignee="reviewer")
        assert who == ["reviewer"]
        assert spawned == ["reviewer"]

        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["assignee"] == "reviewer"


def test_explicit_handoff_reviewer_wins_over_review_assignee_config(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl a feature", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        kb.request_review(conn, tid, summary="done", reviewer="chosen-reviewer", expected_run_id=run_id)

        spawned = []

        def spawn_fn(task, workspace, board=None):
            spawned.append(task.assignee)

        res, who = _spawned_profiles(conn, spawn_fn, review_assignee="fallback-reviewer")
        assert who == ["chosen-reviewer"]
        assert spawned == ["chosen-reviewer"]


def test_no_review_assignee_configured_preserves_stored_assignee(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl a feature", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        kb.request_review(conn, tid, summary="done", expected_run_id=run_id)

        spawned = []

        def spawn_fn(task, workspace, board=None):
            spawned.append(task.assignee)

        res, who = _spawned_profiles(conn, spawn_fn)
        assert who == ["worker"]
        assert spawned == ["worker"]


def test_review_assignee_respects_per_profile_cap(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    with kbc.connect() as conn:
        tids = []
        for i in range(3):
            tid = kb.create_task(conn, title=f"impl feature {i}", assignee="worker")
            kb.claim_task(conn, tid)
            run_id = kb.get_task(conn, tid).current_run_id
            kb.request_review(conn, tid, summary="done", expected_run_id=run_id)
            tids.append(tid)

        spawned = []

        def spawn_fn(task, workspace, board=None):
            spawned.append(task.assignee)

        res, who = _spawned_profiles(
            conn, spawn_fn,
            review_assignee="reviewer",
            max_in_progress_per_profile=1,
        )
        assert who == ["reviewer"]
        assert spawned == ["reviewer"]

        rows = conn.execute(
            "SELECT status FROM tasks WHERE id IN ({})".format(
                ",".join("?" for _ in tids)
            ),
            tids,
        ).fetchall()
        assert sum(1 for r in rows if r["status"] == "running") == 1


def test_review_assignee_invalid_profile_falls_back_to_stored_assignee(
    kanban_home: Path, monkeypatch,
) -> None:
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "worker")

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl a feature", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        kb.request_review(conn, tid, summary="done", expected_run_id=run_id)

        spawned = []

        def spawn_fn(task, workspace, board=None):
            spawned.append(task.assignee)

        res, who = _spawned_profiles(conn, spawn_fn, review_assignee="ghost-reviewer")
        assert who == ["worker"]
        assert spawned == ["worker"]


def test_stale_self_review_handoff_falls_back_to_review_assignee_config(
    kanban_home: Path, all_assignees_spawnable,
) -> None:
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="impl a feature", assignee="worker")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        kb.request_review(conn, tid, summary="done", reviewer="worker", expected_run_id=run_id)

        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["assignee"] == "worker"

        spawned = []

        def spawn_fn(task, workspace, board=None):
            spawned.append(task.assignee)

        res, who = _spawned_profiles(conn, spawn_fn, review_assignee="reviewer")
        assert who == ["reviewer"]
        assert spawned == ["reviewer"]

        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (tid,)).fetchone()
        assert row["assignee"] == "reviewer"
