"""Worker resource-group scheduling contracts for Kanban."""
from __future__ import annotations

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


def _spawn(*_args, **_kwargs):
    return 12345


def test_shared_resource_defers_but_disjoint_resource_runs(isolated_kanban_home_with_profiles):
    conn = kbc.connect()
    try:
        running = kb.create_task(conn, title="research", assignee="alpha")
        blocked = kb.create_task(conn, title="writer", assignee="beta")
        parallel = kb.create_task(conn, title="coding", assignee="default")
        assert kb.claim_task(conn, running) is not None

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

        assert [item[0] for item in result.spawned] == [parallel]
        assert result.skipped_resource_conflict == [(blocked, "beta", ("gpu-0",))]
    finally:
        conn.close()


def test_gateway_settings_preserve_worker_resource_groups():
    from gateway.kanban_watchers_dispatcher import _resolve_dispatcher_settings

    groups = {"gpu-0": ["alpha", "beta"]}
    settings = _resolve_dispatcher_settings({"worker_resource_groups": groups}, kb)

    assert settings.worker_resource_groups == groups