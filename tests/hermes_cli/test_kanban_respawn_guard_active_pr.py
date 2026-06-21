from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db import dispatch_once as run_dispatch_once


class MonkeyPatchLike(Protocol):
    def setenv(
        self,
        name: str,
        value: str,
        prepend: str | None = None,
    ) -> None: ...

    def setattr(
        self,
        target: object,
        name: str,
        value: object,
        raising: bool = True,
    ) -> None: ...


class SpawnFn(Protocol):
    def __call__(self, task: kb.Task, workspace: str) -> None: ...


class DispatchOnceFn(Protocol):
    def __call__(
        self,
        conn: sqlite3.Connection,
        *,
        spawn_fn: SpawnFn,
    ) -> kb.DispatchResult: ...


def _init_kanban_home(tmp_path: Path, monkeypatch: MonkeyPatchLike) -> None:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    _ = kb.init_db()


def test_ready_task_with_active_pr_promotes_to_review_without_respawn_guard(
    tmp_path: Path,
    monkeypatch: MonkeyPatchLike,
    all_assignees_spawnable: None,
) -> None:
    _ = all_assignees_spawnable
    _init_kanban_home(tmp_path, monkeypatch)
    spawned_ids: list[str] = []
    dispatch_once: DispatchOnceFn = run_dispatch_once

    def fake_spawn(task: kb.Task, _workspace: str) -> None:
        spawned_ids.append(task.id)

    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="active-pr", assignee="reviewer")
        _ = kb.add_comment(
            conn,
            task_id,
            "worker",
            "Opened https://github.com/totemx-AI/hermes-agent/pull/123",
        )

        first = dispatch_once(conn, spawn_fn=fake_spawn)
        task_after_first = kb.get_task(conn, task_id)
        assert task_after_first is not None
        spawned_after_first = list(spawned_ids)
        second = dispatch_once(conn, spawn_fn=fake_spawn)
        promoted_event_rows = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? AND kind = "
            + "'promoted_to_review' AND payload = ?",
            (task_id, '{"reason": "active_pr_handoff"}'),
        ).fetchall()
        promoted_event_count = len(promoted_event_rows)
        second_tick_guarded_event_rows = conn.execute(
            "SELECT id FROM task_events WHERE task_id = ? AND kind = "
            + "'respawn_guarded'",
            (task_id,),
        ).fetchall()
        events = kb.list_events(conn, task_id)

    assert task_after_first.status == "review"
    assert first.respawn_guarded == []
    assert second.respawn_guarded == []
    assert first.spawned == []
    assert spawned_after_first == []
    assert promoted_event_count == 1
    assert second_tick_guarded_event_rows == []
    assert not any(
        event.kind == "respawn_guarded" for event in events
    )
