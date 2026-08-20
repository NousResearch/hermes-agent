"""Memory-aware kanban dispatch guard (OOF-30 / OOF-77).

Two production incidents shared the same failure shape: no
``kanban.max_in_progress`` configured, a busy board, and a small hosted VM —
the dispatcher fanned out 26-31 concurrent workers, the host swap-thrashed,
and the whole machine (dashboard included) became unreachable.

Covers the two safeguards added in response:

1. :func:`hermes_cli.kanban_db.derive_default_max_in_progress` /
   :func:`hermes_cli.kanban_db.resolve_max_in_progress` — memory-derived
   default global concurrency cap when the operator never set one.
2. The live memory-pressure guard inside ``dispatch_once`` — critical
   pressure spawns nothing; elevated pressure spawns at most one; unknown
   imposes no restriction (fail-open).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


GIB = 1024 * 1024  # KiB per GiB


# ---------------------------------------------------------------------------
# derive_default_max_in_progress / resolve_max_in_progress
# ---------------------------------------------------------------------------


def test_derived_cap_small_vm_floors_at_two():
    # 1 GiB VM (the incident shape): 1024 // 512 = 2 workers.
    assert kb.derive_default_max_in_progress({"mem_total_kib": 1 * GIB}) == 2
    # Even tiny VMs keep a floor of 2 so boards still make progress.
    assert kb.derive_default_max_in_progress({"mem_total_kib": GIB // 4}) == 2


def test_derived_cap_scales_with_memory_and_ceilings():
    assert kb.derive_default_max_in_progress({"mem_total_kib": 2 * GIB}) == 4
    assert kb.derive_default_max_in_progress({"mem_total_kib": 4 * GIB}) == 8
    # Big iron clamps at the ceiling — explicit config is the escape hatch.
    assert kb.derive_default_max_in_progress({"mem_total_kib": 64 * GIB}) == 8


def test_derived_cap_fails_open_without_memtotal():
    assert kb.derive_default_max_in_progress({}) is None
    assert kb.derive_default_max_in_progress({"mem_total_kib": 0}) is None
    assert kb.derive_default_max_in_progress({"mem_total_kib": -5}) is None
    assert kb.derive_default_max_in_progress({"mem_total_kib": True}) is None
    assert kb.derive_default_max_in_progress({"mem_total_kib": "1048576"}) is None


def test_resolve_max_in_progress_explicit_config_wins(monkeypatch):
    monkeypatch.setattr(
        kb, "_system_memory_sample", lambda: {"mem_total_kib": 1 * GIB}
    )
    # Operator said 6 on a 1 GiB box — their call, even above the derived 2.
    assert kb.resolve_max_in_progress(6) == 6
    assert kb.resolve_max_in_progress(1) == 1


def test_resolve_max_in_progress_derives_when_unset(monkeypatch):
    monkeypatch.setattr(
        kb, "_system_memory_sample", lambda: {"mem_total_kib": 1 * GIB}
    )
    assert kb.resolve_max_in_progress(None) == 2


def test_resolve_max_in_progress_unset_and_unknown_memory_is_uncapped(monkeypatch):
    monkeypatch.setattr(kb, "_system_memory_sample", lambda: {})
    assert kb.resolve_max_in_progress(None) is None


# ---------------------------------------------------------------------------
# _memory_pressure_level
# ---------------------------------------------------------------------------


def test_pressure_level_unknown_on_empty_sample(monkeypatch):
    monkeypatch.setattr(kb, "_system_memory_sample", lambda: {})
    assert kb._memory_pressure_level() == "unknown"


def test_pressure_level_classifies_via_gateway_thresholds():
    ok = {"mem_available_kib": GIB // 2, "mem_total_kib": 1 * GIB}
    critical = {"mem_available_kib": 32 * 1024, "mem_total_kib": 1 * GIB}
    elevated = {"mem_available_kib": 100 * 1024, "mem_total_kib": 1 * GIB}
    assert kb._memory_pressure_level(ok) == "ok"
    assert kb._memory_pressure_level(critical) == "critical"
    assert kb._memory_pressure_level(elevated) == "elevated"


# ---------------------------------------------------------------------------
# dispatch_once under pressure
# ---------------------------------------------------------------------------


def _pressure_sample(level: str) -> dict:
    total = 1 * GIB
    if level == "critical":
        return {"mem_available_kib": 32 * 1024, "mem_total_kib": total}
    if level == "elevated":
        return {"mem_available_kib": 100 * 1024, "mem_total_kib": total}
    return {"mem_available_kib": total // 2, "mem_total_kib": total}


def test_dispatch_spawns_nothing_under_critical_pressure(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(
        kb, "_system_memory_sample", lambda: _pressure_sample("critical")
    )
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kb.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert not spawns
    assert not res.spawned
    assert res.memory_pressure == "critical"


def test_dispatch_critical_pressure_defers_not_drops(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Tasks skipped under pressure stay 'ready' and spawn once memory clears."""
    sample = {"value": _pressure_sample("critical")}
    monkeypatch.setattr(kb, "_system_memory_sample", lambda: sample["value"])
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kb.connect() as conn:
        task = kb.create_task(conn, title="a", assignee="alice")
        kb.dispatch_once(conn, spawn_fn=fake_spawn)
        assert not spawns
        row = kb.get_task(conn, task)
        assert row is not None and row.status == "ready"

        sample["value"] = _pressure_sample("ok")
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert spawns == [task]
    assert res.memory_pressure is None


def test_dispatch_elevated_pressure_spawns_at_most_one(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(
        kb, "_system_memory_sample", lambda: _pressure_sample("elevated")
    )
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kb.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert len(spawns) == 1
    assert res.memory_pressure == "elevated"


def test_dispatch_elevated_pressure_does_not_widen_tighter_budget(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A caller cap already at 0 remaining must not be widened to 1."""
    monkeypatch.setattr(
        kb, "_system_memory_sample", lambda: _pressure_sample("elevated")
    )
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kb.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="alice")
        kb.claim_task(conn, running)
        kb.create_task(conn, title="ready", assignee="bob")
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn, max_in_progress=1)

    assert not spawns
    assert not res.spawned


def test_dispatch_unknown_pressure_imposes_no_restriction(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(kb, "_system_memory_sample", lambda: {})
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kb.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kb.dispatch_once(conn, spawn_fn=fake_spawn)

    assert len(spawns) == 3
    assert res.memory_pressure is None


def test_dispatch_critical_pressure_still_runs_reclaim_bookkeeping(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """The guard must only stop NEW spawns — reclaim/promotion still run."""
    monkeypatch.setattr(
        kb, "_system_memory_sample", lambda: _pressure_sample("critical")
    )
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        child = kb.create_task(
            conn, title="child", assignee="alice", parents=[parent],
        )
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent,))
        res = kb.dispatch_once(conn, spawn_fn=lambda *a, **k: 42)
        row = kb.get_task(conn, child)

    # Promotion (todo -> ready once parents are done) happened despite the
    # spawn freeze.
    assert res.memory_pressure == "critical"
    assert row is not None
    assert row.status == "ready"
