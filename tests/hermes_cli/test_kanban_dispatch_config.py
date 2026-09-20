"""Dispatch configuration is shared by every Kanban dispatch surface."""

from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


def test_load_dispatch_config_normalizes_all_dispatch_limits():
    config = {
        "kanban": {
            "max_spawn": "8",
            "max_in_progress": "6",
            "failure_limit": "4",
            "dispatch_stale_timeout_seconds": "77",
            "default_assignee": "  default  ",
            "max_in_progress_per_profile": "2",
        }
    }

    resolved = kb.load_dispatch_config(config)

    assert resolved == kb.DispatchConfig(
        max_spawn=8,
        max_in_progress=6,
        failure_limit=4,
        stale_timeout_seconds=77,
        default_assignee="default",
        max_in_progress_per_profile=2,
    )


def test_concurrent_periodic_and_ui_dispatches_respect_both_caps(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    for profile in ("default", "research", "ops"):
        (home / "profiles" / profile).mkdir(parents=True)
        # Identity marker: a bare directory is not a live profile.
        (home / "profiles" / profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db(board="default")

    with kb.connect_closing(board="default") as conn:
        for index, assignee in enumerate(("research", "ops", "default", "default")):
            task_id = kb.create_task(
                conn,
                title=f"running-{index}",
                assignee=assignee,
            )
            assert kb.claim_task(conn, task_id, claimer=f"seed:{index}") is not None
        for index, assignee in enumerate(("research", "research", "research", "ops")):
            kb.create_task(
                conn,
                title=f"ready-{index}",
                assignee=assignee,
            )

    barrier = threading.Barrier(2)

    def dispatch_tick():
        barrier.wait()
        with kb.connect_closing(board="default") as conn:
            return kb.dispatch_once(
                conn,
                board="default",
                spawn_fn=lambda *_args: None,
                max_spawn=8,
                max_in_progress=6,
                failure_limit=4,
                stale_timeout_seconds=77,
                default_assignee="default",
                max_in_progress_per_profile=2,
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _source: dispatch_tick(), ("periodic", "ui")))

    with kb.connect_closing(board="default") as conn:
        global_running = conn.execute(
            "SELECT COUNT(*) FROM tasks WHERE status = 'running'"
        ).fetchone()[0]
        per_profile = {
            row["assignee"]: row["count"]
            for row in conn.execute(
                "SELECT assignee, COUNT(*) AS count FROM tasks "
                "WHERE status = 'running' GROUP BY assignee"
            )
        }

    assert global_running <= 6
    assert per_profile.get("research", 0) <= 2
    assert sum(len(result.spawned) for result in results) == 2
    assert any(result.capacity_deferred or result.skipped_locked for result in results)


def test_review_dispatch_respects_global_capacity_cap(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    for profile in ("default", "reviewer"):
        (home / "profiles" / profile).mkdir(parents=True)
        # Identity marker: a bare directory is not a live profile.
        (home / "profiles" / profile / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db(board="default")

    with kb.connect_closing(board="default") as conn:
        for index in range(2):
            task_id = kb.create_task(
                conn,
                title=f"running-{index}",
                assignee="default",
            )
            assert kb.claim_task(conn, task_id, claimer=f"seed:{index}") is not None
        review_id = kb.create_task(
            conn,
            title="review-at-capacity",
            assignee="reviewer",
        )
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (review_id,))
        conn.commit()

        result = kb.dispatch_once(
            conn,
            spawn_fn=lambda *_args: None,
            max_in_progress=2,
            max_in_progress_per_profile=1,
        )

    assert result.spawned == []
    assert result.capacity_deferred is True


def _init_capacity_board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "profiles" / "default").mkdir(parents=True)
    # Identity marker: a bare directory is not a live profile.
    (home / "profiles" / "default" / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db(board="default")


def _seed_running_default(conn):
    task_id = kb.create_task(
        conn, title="already-running", assignee="default", board="default"
    )
    assert kb.claim_task(conn, task_id, claimer="seed") is not None


def test_capacity_deferred_ignores_unassigned_ready_task(tmp_path, monkeypatch):
    _init_capacity_board(tmp_path, monkeypatch)

    with kb.connect_closing(board="default") as conn:
        _seed_running_default(conn)
        unassigned_id = kb.create_task(
            conn, title="needs-assignment", assignee=None, board="default"
        )

        result = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=lambda *_args: None,
            max_in_progress=1,
        )

    assert result.spawned == []
    assert result.skipped_unassigned == [unassigned_id]
    assert result.capacity_deferred is False


def test_capacity_deferred_ignores_nonexistent_profile_ready_task(
    tmp_path, monkeypatch
):
    _init_capacity_board(tmp_path, monkeypatch)

    with kb.connect_closing(board="default") as conn:
        _seed_running_default(conn)
        nonspawnable_id = kb.create_task(
            conn, title="missing-profile", assignee="ghost", board="default"
        )

        result = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=lambda *_args: None,
            max_in_progress=1,
        )

    assert result.spawned == []
    assert result.skipped_nonspawnable == [nonspawnable_id]
    assert result.capacity_deferred is False


def test_capacity_deferred_ignores_respawn_guarded_task_at_profile_capacity(
    tmp_path, monkeypatch
):
    _init_capacity_board(tmp_path, monkeypatch)

    with kb.connect_closing(board="default") as conn:
        _seed_running_default(conn)
        guarded_id = kb.create_task(
            conn, title="auth-blocked", assignee="default", board="default"
        )
        conn.execute(
            "UPDATE tasks SET last_failure_error = ? WHERE id = ?",
            ("401 unauthorized", guarded_id),
        )
        conn.commit()

        result = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=lambda *_args: None,
            max_in_progress=2,
            max_in_progress_per_profile=1,
        )

    assert result.spawned == []
    assert result.respawn_guarded == [(guarded_id, "blocker_auth")]
    assert result.skipped_per_profile_capped == []
    assert result.capacity_deferred is False


def test_capacity_deferred_reports_spawnable_ready_task_at_global_capacity(
    tmp_path, monkeypatch
):
    _init_capacity_board(tmp_path, monkeypatch)

    with kb.connect_closing(board="default") as conn:
        _seed_running_default(conn)
        eligible_id = kb.create_task(
            conn, title="eligible", assignee="default", board="default"
        )

        result = kb.dispatch_once(
            conn,
            board="default",
            spawn_fn=lambda *_args: None,
            max_in_progress=1,
        )

    assert result.spawned == []
    assert result.capacity_deferred is True
    assert result.skipped_unassigned == []
    assert result.skipped_nonspawnable == []
    assert result.respawn_guarded == []
    assert result.skipped_per_profile_capped == []
    assert eligible_id not in result.spawned
