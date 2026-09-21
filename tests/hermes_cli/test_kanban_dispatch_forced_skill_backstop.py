"""The dispatcher refuses to spawn a card whose forced skills cannot resolve.

Pre-queue validation only covers cards created after it shipped, and only when the
assignee's profile could be enumerated at creation time (it fails open otherwise).
Cards already on a board — or created while the assignee profile did not yet exist
— would still spawn a worker that quietly runs without the skill the card pins,
burn a retry, and look like a flaky model.

The backstop runs after profile and workspace resolution (project-local skills only
resolve once the workspace is known), blocks the card for an operator, and must not
consume the circuit breaker's budget: a malformed card is not a flaky one, and
counting it would hide the real reason behind "gave up after N failures".
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.hermes_cli.test_kanban_forced_skills import install_skill


@pytest.fixture()
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "profiles" / "alpha").mkdir(parents=True)
    # Identity marker: without one the dispatcher buckets 'alpha' as nonspawnable
    # and never reaches the forced-skill backstop this file is about.
    (home / "profiles" / "alpha" / ".env").write_text("HERMES_TEST_MARKER=x\n", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    install_skill(home / "profiles" / "alpha", "writing", "translation")

    from hermes_cli import kanban_db as kb
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    return kb


def _legacy_card(kb, skills: list[str]) -> str:
    """A card carrying forced skills that pre-queue validation never saw."""
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="legacy", assignee="alpha")
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET skills = ? WHERE id = ?",
                         (json.dumps(skills), task_id))
    return task_id


def _row(kb, task_id: str):
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        return conn.execute(
            "SELECT status, consecutive_failures, workspace_path, worker_pid "
            "FROM tasks WHERE id = ?", (task_id,)).fetchone()


def _event_kinds(kb, task_id: str) -> list[str]:
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        return [e.kind for e in kb.list_events(conn, task_id)]


def _tick(spawns: list):
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd

    def _spawn(task, workspace, **_kw):
        spawns.append(task.id)
        return 4242

    with kbc.connect_closing() as conn:
        return kbd.dispatch_once(conn, spawn_fn=_spawn)


def test_malformed_card_is_blocked_without_spawning(board):
    task_id = _legacy_card(board, ["ghost-skill"])
    spawns: list = []

    result = _tick(spawns)

    assert spawns == [], "a card whose forced skills cannot resolve must not spawn"
    assert task_id not in [s[0] for s in result.spawned]
    assert task_id in [entry[0] for entry in result.blocked_malformed]
    row = _row(board, task_id)
    assert row["status"] == "blocked"
    assert row["worker_pid"] is None


def test_block_does_not_consume_the_retry_budget(board):
    """``consecutive_failures`` stays at zero: the breaker is for flaky runs, and a
    malformed card must be reported as malformed, not as "gave up after N failures"."""
    task_id = _legacy_card(board, ["ghost-skill"])

    _tick([])

    assert _row(board, task_id)["consecutive_failures"] == 0
    assert "gave_up" not in _event_kinds(board, task_id)
    assert "spawn_failed" not in _event_kinds(board, task_id)
    assert "blocked" in _event_kinds(board, task_id)


def test_backstop_runs_after_workspace_resolution(board):
    """Project-local skills only resolve once the workspace is known, so the check
    must sit downstream of workspace resolution — the resolved path is persisted."""
    task_id = _legacy_card(board, ["ghost-skill"])

    _tick([])

    assert _row(board, task_id)["workspace_path"], "workspace was resolved before the check"


def test_blocked_card_stays_blocked_and_does_not_respawn(board):
    """The block is sticky: promotion must not hand the same broken card back."""
    task_id = _legacy_card(board, ["ghost-skill"])
    spawns: list = []

    _tick(spawns)
    _tick(spawns)

    assert spawns == []
    assert _row(board, task_id)["status"] == "blocked"


def test_reason_names_the_offending_skill_and_the_profile(board):
    task_id = _legacy_card(board, ["ghost-skill"])

    _tick([])

    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        blocked = [e for e in board.list_events(conn, task_id) if e.kind == "blocked"][-1]
    reason = str(blocked.payload.get("reason") or "")
    assert "ghost-skill" in reason
    assert "alpha" in reason


def test_valid_card_still_spawns(board):
    """The backstop is a guard, not a gate: a resolvable forced skill dispatches."""
    task_id = _legacy_card(board, ["translation"])
    spawns: list = []

    _tick(spawns)

    assert spawns == [task_id]
    assert _row(board, task_id)["status"] == "running"
