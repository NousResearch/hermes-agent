"""Worker resource-group scheduling contracts for Kanban."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


def _spawn(*_args, **_kwargs):
    return 12345


def test_shared_resource_defers_but_disjoint_resource_runs(all_assignees_spawnable):
    conn = kbc.connect()
    try:
        running = kb.create_task(conn, title="research", assignee="alpha")
        blocked = kb.create_task(conn, title="writer", assignee="beta")
        parallel = kb.create_task(conn, title="coding", assignee="default")

        result = kbd.dispatch_once(
            conn,
            spawn_fn=_spawn,
            dry_run=True,
            worker_resource_groups={
                "gpu-0": ["alpha", "beta"],
                "gpu-1": ["default"],
                "ignored": "not-a-profile-list",
            },
        )

        assert [item[0] for item in result.spawned] == [running, parallel]
        assert result.skipped_resource_conflict == [(blocked, "beta", ("gpu-0",))]
    finally:
        conn.close()


def test_resource_groups_include_running_workers_on_other_boards(
    all_assignees_spawnable,
):
    from gateway.kanban_watchers_dispatcher import _resolve_dispatcher_settings

    groups = {"gpu-0": ["alpha", "beta"]}
    settings = _resolve_dispatcher_settings({"worker_resource_groups": groups}, kb)
    assert settings.worker_resource_groups == groups

    kb.create_board("second", name="Second")
    with kbc.connect_closing(board="second") as other:
        running = kb.create_task(other, title="research", assignee="alpha")
        assert kb.claim_task(other, running) is not None
    with kbc.connect_closing() as conn:
        blocked = kb.create_task(conn, title="writer", assignee="beta")
        result = kbd.dispatch_once(
            conn,
            spawn_fn=_spawn,
            dry_run=True,
            worker_resource_groups=groups,
        )

    assert not result.spawned
    assert result.skipped_resource_conflict == [(blocked, "beta", ("gpu-0",))]


def test_resource_admission_is_serialized_across_boards(all_assignees_spawnable):
    groups = {"gpu-0": ["alpha", "beta"]}
    kb.create_board("second", name="Second")
    with kbc.connect_closing() as conn:
        alpha = kb.create_task(conn, title="research", assignee="alpha")
    with kbc.connect_closing(board="second") as conn:
        beta = kb.create_task(conn, title="writer", assignee="beta")

    spawn_entered = Event()
    release_spawn = Event()

    def dispatch_first():
        def blocking_spawn(*_args, **_kwargs):
            spawn_entered.set()
            assert release_spawn.wait(5)
            return 12345

        with kbc.connect_closing() as conn:
            return kbd.dispatch_once(
                conn,
                spawn_fn=blocking_spawn,
                worker_resource_groups=groups,
            )

    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(dispatch_first)
        assert spawn_entered.wait(5)
        with kbc.connect_closing(board="second") as conn:
            contended = kbd.dispatch_once(
                conn,
                spawn_fn=_spawn,
                worker_resource_groups=groups,
            )
        release_spawn.set()
        admitted = first.result(timeout=5)

    assert [item[0] for item in admitted.spawned] == [alpha]
    assert contended.skipped_locked is True
    assert contended.spawned == []
    with kbc.connect_closing(board="second") as conn:
        assert kb.get_task(conn, beta).status == "ready"

        # Once admission completes, a disjoint resource remains dispatchable
        # even while the first board's worker is running.
        disjoint = kbd.dispatch_once(
            conn,
            spawn_fn=_spawn,
            worker_resource_groups={"gpu-0": ["alpha"], "gpu-1": ["beta"]},
        )
    assert [item[0] for item in disjoint.spawned] == [beta]


def test_resource_lock_open_failure_fails_closed(all_assignees_spawnable, monkeypatch):
    task_id: str
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="research", assignee="alpha")

    original_open = Path.open

    def deny_resource_lock(path, *args, **kwargs):
        if path.name == ".resource-dispatch.lock":
            raise PermissionError("resource lock denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_resource_lock)
    spawn_calls = []
    with kbc.connect_closing() as conn:
        result = kbd.dispatch_once(
            conn,
            spawn_fn=lambda *args, **kwargs: spawn_calls.append((args, kwargs)),
            worker_resource_groups={"gpu-0": ["alpha"]},
        )

    assert result.skipped_locked is True
    assert result.spawned == []
    assert spawn_calls == []
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == "ready"
