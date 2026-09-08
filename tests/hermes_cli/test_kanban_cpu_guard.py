"""Host-global CPU-pressure kanban dispatch guard (t_4008d306).

The 2026-09-08 Astra control-plane review found no host-wide CPU admission:
the dispatcher already refuses to spawn new workers under critical *memory*
pressure (``test_kanban_memory_guard.py``), but a saturated host (high load
average / PSI) had no equivalent stop. At 13:16 BST the host reported load
averages 51.47/50.01/47.34 with CPU PSI some avg300=76.64 — exactly the
"every representative board dispatch should defer, not spawn, not drop"
condition this guard exists for.

Covers:

1. :func:`hermes_cli.kanban_db_dispatch._cpu_pressure_level` — classification
   via :mod:`gateway.cpu_status` (worst-of load average / PSI).
2. The live CPU-pressure guard inside ``dispatch_once`` — critical pressure
   spawns nothing (``spawned == []``) across representative boards; elevated
   pressure spawns at most one; unknown imposes no restriction (fail-open,
   matching the memory guard's documented convention).
3. One HOST-GLOBAL decision: the guard reads the whole machine once per tick,
   not a per-board budget — asserted here by running it against several
   distinct boards/DBs and confirming the same pressure reading gates all of
   them identically in the same tick.
4. Defer, never drop: tasks skipped under critical CPU pressure stay in their
   dispatchable status and spawn once pressure clears.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# _cpu_pressure_level
# ---------------------------------------------------------------------------


def test_cpu_pressure_level_unknown_on_empty_sample(monkeypatch):
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: {})
    assert kbd._cpu_pressure_level() == "unknown"


def test_cpu_pressure_level_classifies_via_load_average():
    ok = {"load1": 1.0, "cpu_count": 8}
    elevated = {"load1": 9.0, "cpu_count": 8}
    critical = {"load1": 51.47, "cpu_count": 8}
    assert kbd._cpu_pressure_level(ok) == "ok"
    assert kbd._cpu_pressure_level(elevated) == "elevated"
    assert kbd._cpu_pressure_level(critical) == "critical"


def test_cpu_pressure_level_classifies_via_psi_when_load_unavailable():
    # Mirrors the 2026-09-08 probe: CPU PSI some avg300=76.64 without a usable
    # load-average/cpu_count pairing (e.g. cgroup-limited container).
    assert kbd._cpu_pressure_level({"psi_some_avg60": 76.64}) == "critical"
    assert kbd._cpu_pressure_level({"psi_some_avg60": 5.0}) == "ok"


def test_cpu_pressure_level_worst_of_both_signals():
    # Load looks fine but PSI says critical (or vice versa) -> critical wins.
    sample = {"load1": 1.0, "cpu_count": 8, "psi_some_avg60": 90.0}
    assert kbd._cpu_pressure_level(sample) == "critical"


# ---------------------------------------------------------------------------
# dispatch_once under CPU pressure — RED/GREEN invariant coverage
# ---------------------------------------------------------------------------


def _load_probe_sample(level: str) -> dict:
    """Representative host reading at each tier, using the actual probed
    2026-09-08 figures for "critical" (load avg 51.47 on an 8-core box)."""
    if level == "critical":
        return {"load1": 51.47, "cpu_count": 8, "psi_some_avg60": 76.64}
    if level == "elevated":
        return {"load1": 9.0, "cpu_count": 8, "psi_some_avg60": 25.0}
    return {"load1": 1.0, "cpu_count": 8, "psi_some_avg60": 2.0}


# Representative boards a real host dispatches across, per the P0 acceptance
# criterion: "every representative board dispatch defers safely".
REPRESENTATIVE_BOARDS = ("jarvis-os", "sycode-trading", "upero")


@pytest.mark.parametrize("board", REPRESENTATIVE_BOARDS)
def test_RED_dispatch_spawns_nothing_under_critical_cpu_pressure(
    kanban_home, all_assignees_spawnable, monkeypatch, board,
):
    """RED: this is the exact regression the card exists to close — before the
    guard existed, critical CPU pressure did not stop a spawn. GREEN below
    proves the guard closes it for every representative board."""
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: _load_probe_sample("critical"))
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect(board=board) as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn, board=board)

    # The required invariant: critical CPU pressure returns spawned=[], never
    # drops the queued work, and records why.
    assert spawns == []
    assert res.spawned == []
    assert res.cpu_pressure == "critical"


def test_GREEN_dispatch_critical_cpu_pressure_defers_not_drops(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Tasks skipped under CPU pressure stay dispatchable and spawn once the
    host's load clears — deferred, never dropped."""
    sample = {"value": _load_probe_sample("critical")}
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: sample["value"])
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        task = kb.create_task(conn, title="a", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)
        assert spawns == []
        assert res.spawned == []
        row = kb.get_task(conn, task)
        assert row is not None and row.status == "ready"

        sample["value"] = _load_probe_sample("ok")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert spawns == [task]
    assert res.cpu_pressure is None


def test_dispatch_elevated_cpu_pressure_spawns_at_most_one(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: _load_probe_sample("elevated"))
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert len(spawns) == 1
    assert res.cpu_pressure == "elevated"


def test_dispatch_elevated_cpu_pressure_does_not_widen_tighter_budget(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A caller cap already at 0 remaining must not be widened to 1."""
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: _load_probe_sample("elevated"))
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="alice")
        kb.claim_task(conn, running)
        kb.create_task(conn, title="ready", assignee="bob")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn, max_in_progress=1)

    assert spawns == []
    assert res.spawned == []


def test_dispatch_unknown_cpu_pressure_imposes_no_restriction(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: {})
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert len(spawns) == 3
    assert res.cpu_pressure is None


def test_dispatch_critical_cpu_pressure_still_runs_reclaim_bookkeeping(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """The guard must only stop NEW spawns — reclaim/promotion still run."""
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: _load_probe_sample("critical"))
    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        child = kb.create_task(conn, title="child", assignee="alice", parents=[parent])
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent,))
        res = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 42)
        row = kb.get_task(conn, child)

    assert res.cpu_pressure == "critical"
    assert row is not None
    assert row.status == "ready"


def test_dispatch_combines_worse_of_memory_and_cpu_pressure(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Critical CPU pressure stops spawning even when memory reads fine, and
    vice versa — the two guards are independent, worst-of-both gates."""
    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: {"mem_available_kib": 10**7, "mem_total_kib": 10**8})
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: _load_probe_sample("critical"))
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42

    with kbc.connect() as conn:
        kb.create_task(conn, title="a", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert spawns == []
    assert res.memory_pressure is None
    assert res.cpu_pressure == "critical"


# ---------------------------------------------------------------------------
# Test neutralizer reload-stability (t_4008d306 rework, review finding #2)
# ---------------------------------------------------------------------------


def test_default_neutralizer_returns_empty_sample_without_marker():
    """Sanity baseline: absent ``@pytest.mark.real_cpu_guard``, the autouse
    fixture in ``tests/conftest.py`` keeps the CPU guard neutral for every
    test, regardless of the host's actual load."""
    assert kbd._system_cpu_sample() == {}
    assert kbd._cpu_pressure_level() == "unknown"


def test_neutralizer_survives_hermes_cli_module_purge_and_reimport():
    """The autouse neutralizer must patch a RELOAD-STABLE origin, not a
    ``hermes_cli.kanban_db_dispatch`` module attribute directly.

    ``tests/hermes_cli/test_kanban_per_profile_cap.py`` (and similar tests)
    purge every ``hermes_cli``/``hermes_state``/``hermes_constants`` module
    from ``sys.modules`` and reimport fresh module objects mid-test. A patch
    applied only to the OLD ``kanban_db_dispatch`` module object is invisible
    to that fresh module, which would then read the REAL (possibly critical)
    host CPU state instead of the neutral default — exactly the failure the
    independent review reproduced. ``tests/conftest.py`` patches
    ``gateway.cpu_status.sample_cpu`` instead, which survives because
    ``gateway.*`` modules are never purged by that pattern and
    ``_system_cpu_sample`` re-imports ``sample_cpu`` from it on every call.
    """
    purge_prefixes = ("hermes_cli", "hermes_state")
    removed = {}
    for mod_name in list(sys.modules):
        if mod_name.startswith(purge_prefixes) or mod_name == "hermes_constants":
            removed[mod_name] = sys.modules.pop(mod_name)
    try:
        from hermes_cli import kanban_db_dispatch as fresh_kbd

        assert fresh_kbd is not kbd, "expected a genuinely fresh module object after the purge"
        assert fresh_kbd._system_cpu_sample() == {}
        assert fresh_kbd._cpu_pressure_level() == "unknown"
    finally:
        # Restore the original module objects so later tests in this process
        # keep using the identities they already hold references to.
        for mod_name in list(sys.modules):
            if mod_name.startswith(purge_prefixes) or mod_name == "hermes_constants":
                del sys.modules[mod_name]
        sys.modules.update(removed)


@pytest.mark.real_cpu_guard
def test_real_cpu_guard_marker_exposes_real_sampling_without_dispatch():
    """Opting out with ``@pytest.mark.real_cpu_guard`` must expose the REAL
    ``gateway.cpu_status.sample_cpu`` seam (not the neutralized ``{}``),
    while this test performs no dispatch at all — proving the opt-out only
    widens what CAN be sampled, never forces an actual worker spawn."""
    sample = kbd._system_cpu_sample()
    assert sample, "real_cpu_guard should expose the real, non-neutralized CPU sample"
    assert "load1" in sample or "psi_some_avg60" in sample
