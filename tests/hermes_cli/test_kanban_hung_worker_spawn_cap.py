"""Regression: gateway kanban admit must bound a hung-provider worker pool.

2026-08-18 Mac incident: gateway pid 1597 spawned 142 ``hermes -p <profile>``
workers in ~61 minutes. The postmortem watched ``delegation.max_concurrent_children``
(delegate_task batch width only). The real admit path is
``kanban_db.dispatch_once`` via the gateway dispatcher, whose
``kanban.max_spawn`` / ``kanban.max_in_progress`` knobs defaulted to None
(unlimited). Hung workers that stay ``status='running'`` already occupy
``max_spawn`` *when the cap is set*; the gate was never armed.

These tests name that defect and require a fail-closed default cap.
"""
from __future__ import annotations

import argparse
import os
import tempfile

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture()
def isolated_kanban_home(monkeypatch):
    """Fresh HERMES_HOME so CLI dispatch does not touch the live board."""
    test_home = tempfile.mkdtemp(prefix="kanban_spawn_cap_")
    os.makedirs(os.path.join(test_home, "profiles", "default"), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    yield test_home


def test_default_config_kanban_max_spawn_is_fail_closed():
    """An empty user config.yaml must not leave the dispatcher uncapped."""
    from hermes_cli.config import DEFAULT_CONFIG

    cap = DEFAULT_CONFIG["kanban"].get("max_spawn")
    assert isinstance(cap, int) and cap >= 1, (
        "DEFAULT_CONFIG['kanban']['max_spawn'] must be a positive int so a "
        f"25-line config.yaml cannot spawn 142 workers; got {cap!r}"
    )


def test_resolve_kanban_max_spawn_fail_closes_none_and_invalid():
    """None / 0 / negative / garbage all become the fail-closed default.

    This is the opposite of max_concurrent_sessions, where None disables
    the gate. Null-means-unlimited is the 142-worker path.
    """
    default = kb.resolve_kanban_max_spawn(None)
    assert isinstance(default, int) and default >= 1
    assert kb.resolve_kanban_max_spawn(0) == default
    assert kb.resolve_kanban_max_spawn(-1) == default
    assert kb.resolve_kanban_max_spawn("abc") == default
    assert kb.resolve_kanban_max_spawn("1.5") == default
    assert kb.resolve_kanban_max_spawn(8) == 8
    assert kb.resolve_kanban_max_spawn("4") == 4


def test_uncapped_max_spawn_none_drains_the_ready_queue(
    kanban_home, all_assignees_spawnable
):
    """Reproduction of the 142-worker admit path.

    ``dispatch_once(max_spawn=None)`` is unlimited. A ready-queue flood
    (the 08-17 reclamation sweep) therefore becomes one worker per task.
    """
    spawns: list[str] = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        ready_ids = [
            kb.create_task(conn, title=f"ready-{i}", assignee="alice")
            for i in range(20)
        ]
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=None)

    assert len(res.spawned) == 20, (
        f"uncapped dispatch must drain the ready queue (142-worker path); "
        f"spawned {len(res.spawned)}"
    )
    assert spawns == ready_ids


def test_hung_provider_worker_pool_cannot_exceed_default_max_spawn(
    kanban_home, all_assignees_spawnable
):
    """Named acceptance test: N hung workers occupy the fail-closed cap.

    Simulate a provider-dead pool (status=running, no heartbeat, pid
    recorded) plus a ready-queue flood. Production resolve() must refuse
    to spawn past the cap on this tick *and* the next tick.
    """
    cap = kb.resolve_kanban_max_spawn(None)
    spawns: list[str] = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)
        return 70000 + len(spawns)

    with kb.connect() as conn:
        hung = []
        for i in range(cap):
            tid = kb.create_task(conn, title=f"hung-provider-{i}", assignee="alice")
            kb.claim_task(conn, tid)
            kb._set_worker_pid(conn, tid, 60000 + i)
            hung.append(tid)
        flood = [
            kb.create_task(conn, title=f"flood-{i}", assignee="bob")
            for i in range(cap * 5)
        ]

        first = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=cap)
        assert first.spawned == [], (
            f"hung pool of {cap} must saturate the cap; spawned "
            f"{[t for t, *_ in first.spawned]}"
        )
        assert spawns == []
        for tid in flood:
            assert kb.get_task(conn, tid).status == "ready"
        for tid in hung:
            assert kb.get_task(conn, tid).status == "running"

        second = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=cap)
        assert second.spawned == [], (
            "second tick must still refuse: hung workers stay running and "
            f"must keep counting; spawned {[t for t, *_ in second.spawned]}"
        )
        assert getattr(first, "skipped_concurrency_capped", None), (
            "fail-loud: dispatch must record ready work deferred by the cap"
        )


def test_hung_pool_releases_one_slot_when_a_worker_finishes(
    kanban_home, all_assignees_spawnable
):
    """When one hung worker completes, exactly one ready task may spawn."""
    cap = kb.resolve_kanban_max_spawn(None)
    spawns: list[str] = []

    def fake_spawn(task, workspace):
        spawns.append(task.id)

    with kb.connect() as conn:
        hung = []
        for i in range(cap):
            tid = kb.create_task(conn, title=f"hung-{i}", assignee="alice")
            kb.claim_task(conn, tid)
            hung.append(tid)
        ready_a = kb.create_task(conn, title="ready-a", assignee="bob")
        ready_b = kb.create_task(conn, title="ready-b", assignee="carol")

        kb.complete_task(conn, hung[0])
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn=cap)

        assert [t for t, *_ in res.spawned] == [ready_a]
        assert spawns == [ready_a]
        assert kb.get_task(conn, ready_b).status == "ready"


def test_cli_missing_max_spawn_uses_fail_closed_default(
    isolated_kanban_home, monkeypatch
):
    """Empty kanban config must not pass max_spawn=None into dispatch_once."""
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"kanban": {}})
    captured = {}
    monkeypatch.setattr(
        kanban_db,
        "dispatch_once",
        lambda conn, **kw: (captured.update(kw), kanban_db.DispatchResult())[1],
    )
    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)
    got = captured.get("max_spawn")
    assert isinstance(got, int) and got >= 1, (
        f"CLI must fail-close missing kanban.max_spawn; got {got!r}"
    )
