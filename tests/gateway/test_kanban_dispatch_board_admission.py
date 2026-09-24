"""Board-scoped admission for the existing embedded native dispatcher."""

from __future__ import annotations

import sqlite3

import pytest

from gateway import kanban_watchers_dispatcher as kwd
from hermes_cli import config_effective
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    kb.init_db()
    kb.create_board("allowed")
    kb.create_board("excluded")
    kb.create_board("123")
    kb.create_board("true")
    return tmp_path


def _settings(*, boards, max_in_progress=8):
    return kwd._DispatcherSettings(
        interval=60,
        max_spawn=None,
        max_in_progress=max_in_progress,
        failure_limit=2,
        stale_timeout_seconds=0,
        reconcile_orphans=True,
        default_assignee=None,
        max_in_progress_per_profile=None,
        dispatch_boards=boards,
    )


def _task_state(board: str, task_id: str) -> tuple[str, int, int]:
    path = kb.kanban_db_path(board=board).resolve()
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    try:
        status, claim_lock = conn.execute(
            "SELECT status, claim_lock FROM tasks WHERE id = ?", (task_id,),
        ).fetchone()
        events = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ?", (task_id,),
        ).fetchone()[0]
        return status, int(claim_lock is not None), int(events)
    finally:
        conn.close()


def test_policy_is_presence_sensitive_and_fails_closed(board_home, monkeypatch):
    def resolve(config):
        monkeypatch.setattr(
            config_effective, "load_user_config_effective", lambda **_kw: config,
        )
        return kwd._resolve_dispatch_boards(kb)

    assert resolve({"kanban": {}}) is None
    assert resolve({"kanban": {"dispatch_boards": ["allowed"]}}) == frozenset({"allowed"})
    invalid = (None, [], "allowed", ["missing"], [""], [123], [True], ["allowed", 123])
    for value in invalid:
        assert resolve({"kanban": {"dispatch_boards": value}}) == frozenset()
    assert resolve({"kanban": []}) == frozenset()


def test_allowed_board_dispatches_without_touching_excluded_board(
    board_home, monkeypatch,
):
    from hermes_cli import profiles

    with kbc.connect(board="allowed") as conn:
        allowed_id = kb.create_task(
            conn,
            title="Allowed synthetic work",
            assignee="worker",
            workspace_kind="scratch",
            initial_status="running",
            board="allowed",
        )
    with kbc.connect(board="excluded") as conn:
        excluded_id = kb.create_task(
            conn,
            title="Unrelated work",
            assignee="worker",
            workspace_kind="scratch",
            initial_status="running",
            board="excluded",
        )
    excluded_before = _task_state("excluded", excluded_id)

    real_connect = kbc.connect
    connected: list[str | None] = []

    def guarded_connect(*args, **kwargs):
        board = kwargs.get("board")
        connected.append(board)
        if board == "excluded":
            raise AssertionError("excluded board used the mutating connection path")
        return real_connect(*args, **kwargs)

    spawned: list[tuple[str, str]] = []
    monkeypatch.setattr(kbc, "connect", guarded_connect)
    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    monkeypatch.setattr(
        kbd, "_default_spawn",
        lambda task, workspace, board=None: spawned.append((task.id, board)) or 4242,
    )
    monkeypatch.setattr(kbd, "_memory_pressure_level", lambda *_a, **_k: "ok")

    dispatcher = kwd._KanbanDispatcher(kb, _settings(boards=frozenset({"allowed"})))
    results = dispatcher.tick_once()

    assert [slug for slug, _result in results] == ["allowed"]
    assert spawned == [(allowed_id, "allowed")]
    assert "excluded" not in connected
    assert dispatcher.tick_once_for_board("excluded") is None
    assert _task_state("excluded", excluded_id) == excluded_before
    with real_connect(board="allowed") as conn:
        task = kb.get_task(conn, allowed_id)
        assert task.status == "running"
        assert task.claim_lock
        assert task.worker_pid == 4242


def test_scoped_capacity_reads_all_boards_without_connect_and_denies_uncertainty(
    board_home, monkeypatch,
):
    with kbc.connect(board="excluded") as conn:
        task_id = kb.create_task(
            conn,
            title="Existing worker",
            assignee="worker",
            workspace_kind="scratch",
            initial_status="running",
            board="excluded",
        )
        assert kb.claim_task(conn, task_id) is not None
    db_mtime = kb.kanban_db_path(board="excluded").stat().st_mtime_ns

    monkeypatch.setattr(
        kbc, "connect",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("connect must not be used")),
    )
    assert kbd.count_running_tasks_other_boards(
        "allowed", strict_read_only=True,
    ) == 1
    assert kb.kanban_db_path(board="excluded").stat().st_mtime_ns == db_mtime

    conn = sqlite3.connect(
        f"{kb.kanban_db_path(board='allowed').resolve().as_uri()}?mode=ro", uri=True,
    )
    try:
        monkeypatch.setattr(
            kbd.sqlite3,
            "connect",
            lambda *_a, **_k: (_ for _ in ()).throw(sqlite3.OperationalError("unreadable")),
        )
        assert kbd.count_running_tasks_other_boards(
            "allowed", strict_read_only=True,
        ) is None
        may_spawn, budget = kbd._tick_spawn_budget(
            conn,
            kbd.DispatchResult(),
            max_spawn=None,
            max_in_progress=8,
            board="allowed",
            strict_capacity_read=True,
        )
    finally:
        conn.close()
    assert (may_spawn, budget) == (False, None)


def test_scoped_policy_rejects_db_override_before_board_access(
    board_home, monkeypatch,
):
    with kbc.connect(board="excluded") as conn:
        excluded_id = kb.create_task(
            conn,
            title="Override target",
            assignee="worker",
            workspace_kind="scratch",
            initial_status="running",
            board="excluded",
        )
    excluded_before = _task_state("excluded", excluded_id)
    override = kb.kanban_db_path(board="excluded").resolve()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(override))
    monkeypatch.setattr(
        config_effective,
        "load_user_config_effective",
        lambda **_kw: {"kanban": {"dispatch_boards": ["allowed"]}},
    )

    assert kwd._resolve_dispatch_boards(kb) == frozenset()
    dispatcher = kwd._KanbanDispatcher(kb, _settings(boards=frozenset({"allowed"})))
    monkeypatch.setattr(
        dispatcher,
        "board_db_fingerprint",
        lambda _slug: (_ for _ in ()).throw(AssertionError("path access must be fenced")),
    )
    monkeypatch.setattr(
        kbc,
        "connect",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("connect must be fenced")),
    )
    monkeypatch.setattr(
        kbd,
        "count_running_tasks_other_boards",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("capacity must not run")),
    )

    assert dispatcher._board_slugs() == []
    assert dispatcher.tick_once_for_board("allowed") is None
    assert _task_state("excluded", excluded_id) == excluded_before
