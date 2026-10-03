"""CPU-pressure admission guard for kanban dispatch (regression for #126119).

The dispatcher already refuses new workers when SYSTEM MEMORY is critical
(``_memory_pressure_level``, tests/hermes_cli/test_kanban_memory_guard.py) but
had no CPU equivalent. On a host saturated by work the dispatcher cannot see
(cron jobs, CI runners, local inference, another tenant's workers) every tick
kept admitting workers up to the static caps: ``max_in_progress`` /
``max_in_progress_per_profile`` count only the dispatcher's OWN running tasks.

This suite pins the CPU mirror of the memory guard:

* :func:`_cpu_pressure_level` — ok/elevated/critical/unknown from a load-per-
  core reading and a Linux PSI ``some avg60`` reading, worse signal wins,
  ``unknown`` only when BOTH signals are unreadable (fail open).
* The tick contract — critical spawns nothing and still runs reclaim/promote
  bookkeeping; elevated spawns at most one and never widens a tighter budget;
  ok/unknown impose no restriction.
* Visibility — ``DispatchResult.cpu_pressure`` and ``describe_suppression``.

The sample is injected, never produced by saturating the machine: these tests
must never load the host they run on.
"""

from __future__ import annotations

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
# _cpu_pressure_level — the classifier
# ---------------------------------------------------------------------------


def test_cpu_pressure_ok_below_every_threshold():
    # 0.5 load per core and 4% PSI: neither signal reaches "elevated".
    assert kbd._cpu_pressure_level(
        {"load1_per_core": 0.5, "psi_some_avg60_pct": 4.0}
    ) == "ok"


def test_cpu_pressure_elevated_from_load_per_core():
    assert kbd._cpu_pressure_level({"load1_per_core": 1.0}) == "elevated"
    # Between the tiers the worse signal decides, not the last one read.
    assert kbd._cpu_pressure_level({"load1_per_core": 1.2, "psi_some_avg60_pct": 5.0}) == "elevated"


def test_cpu_pressure_critical_from_load_per_core():
    assert kbd._cpu_pressure_level({"load1_per_core": 2.0}) == "critical"
    assert kbd._cpu_pressure_level({"load1_per_core": 0.1, "psi_some_avg60_pct": 61.0}) == "critical"


def test_cpu_pressure_worse_signal_wins():
    """Load says elevated, PSI says critical -> critical (never the better one)."""
    assert kbd._cpu_pressure_level(
        {"load1_per_core": 1.5, "psi_some_avg60_pct": 75.0}
    ) == "critical"
    # And the reverse: a calm CPU must not mask a loadavg spike.
    assert kbd._cpu_pressure_level(
        {"load1_per_core": 3.0, "psi_some_avg60_pct": 0.0}
    ) == "critical"


def test_cpu_pressure_unknown_only_when_both_signals_unreadable():
    # Non-Linux / no loadavg and no /proc/pressure: fail open, never brick dispatch.
    assert kbd._cpu_pressure_level({}) == "unknown"
    # A malformed reading is as unreadable as a missing one.
    assert kbd._cpu_pressure_level({"load1_per_core": None, "psi_some_avg60_pct": None}) == "unknown"
    assert kbd._cpu_pressure_level({"load1_per_core": "busy"}) == "unknown"
    assert kbd._cpu_pressure_level({"load1_per_core": float("nan")}) == "unknown"


def test_cpu_pressure_one_readable_signal_still_classifies():
    """PSI alone (loadavg unavailable) is enough to restrict; the converse too."""
    assert kbd._cpu_pressure_level({"psi_some_avg60_pct": 25.0}) == "elevated"
    assert kbd._cpu_pressure_level({"load1_per_core": 1.1}) == "elevated"


# ---------------------------------------------------------------------------
# kanban.cpu_pressure_thresholds — the operator override
# ---------------------------------------------------------------------------


def _set_thresholds(monkeypatch, value):
    import hermes_cli.config as cfgmod

    monkeypatch.setattr(cfgmod, "load_config_readonly", lambda: {"kanban": {"cpu_pressure_thresholds": value}})


def test_configured_thresholds_widen_the_guard(monkeypatch):
    _set_thresholds(monkeypatch, {"elevated": 4.0, "critical": 8.0})
    # 1.5 load/core is "elevated" by default but "ok" under the override.
    assert kbd._cpu_pressure_level({"load1_per_core": 1.5}) == "ok"
    assert kbd._cpu_pressure_level({"load1_per_core": 4.0}) == "elevated"


def test_configured_thresholds_default_to_shipped_values(monkeypatch):
    for value in ({}, None, {"elevated": "lots"}, {"elevated": 0}, {"elevated": True}, []):
        _set_thresholds(monkeypatch, value)
        assert kbd._cpu_pressure_level({"load1_per_core": 1.0}) == "elevated", value


def test_config_read_failure_falls_back_to_defaults(monkeypatch):
    import hermes_cli.config as cfgmod

    def _boom():
        raise OSError("config gone")

    monkeypatch.setattr(cfgmod, "load_config_readonly", _boom)
    assert kbd._cpu_pressure_level({"load1_per_core": 2.0}) == "critical"


# ---------------------------------------------------------------------------
# dispatch_once under CPU pressure
# ---------------------------------------------------------------------------


def _cpu_sample(level: str) -> dict:
    if level == "critical":
        return {"load1_per_core": 4.0, "psi_some_avg60_pct": 80.0}
    if level == "elevated":
        return {"load1_per_core": 1.2, "psi_some_avg60_pct": 25.0}
    return {"load1_per_core": 0.3, "psi_some_avg60_pct": 3.0}


def _recorder(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


def test_dispatch_spawns_nothing_under_critical_cpu(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """The reported symptom: a CPU-saturated host must stop getting workers."""
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: _cpu_sample("critical"))
    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: {})
    spawns: list = []

    with kbc.connect() as conn:
        for title in ("a", "b", "c", "d", "e", "f", "g", "h"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=_recorder(spawns))

    assert not spawns
    assert not res.spawned
    assert res.cpu_pressure == "critical"


def test_dispatch_critical_cpu_defers_not_drops(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Tasks held by the CPU guard stay 'ready' and spawn once CPU clears."""
    sample = {"value": _cpu_sample("critical")}
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: sample["value"])
    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: {})
    spawns: list = []

    with kbc.connect() as conn:
        task = kb.create_task(conn, title="a", assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=_recorder(spawns))
        assert not spawns
        row = kb.get_task(conn, task)
        assert row is not None and row.status == "ready"

        sample["value"] = _cpu_sample("ok")
        res = kbd.dispatch_once(conn, spawn_fn=_recorder(spawns))

    assert spawns == [task]
    assert res.cpu_pressure is None


def test_dispatch_elevated_cpu_spawns_at_most_one(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: _cpu_sample("elevated"))
    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: {})
    spawns: list = []

    with kbc.connect() as conn:
        for title in ("a", "b", "c", "d", "e", "f"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=_recorder(spawns))

    assert len(spawns) == 1
    assert res.cpu_pressure == "elevated"


def test_dispatch_elevated_cpu_does_not_widen_tighter_budget(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """A caller cap already exhausted must not be widened to 1 by the guard."""
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: _cpu_sample("elevated"))
    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: {})
    spawns: list = []

    with kbc.connect() as conn:
        running = kb.create_task(conn, title="running", assignee="alice")
        kb.claim_task(conn, running)
        kb.create_task(conn, title="ready", assignee="bob")
        res = kbd.dispatch_once(conn, spawn_fn=_recorder(spawns), max_in_progress=1)

    assert not spawns
    assert not res.spawned


def test_dispatch_unknown_cpu_imposes_no_restriction(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Windows dev boxes and PSI-less containers must dispatch normally."""
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: {})
    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: {})
    spawns: list = []

    with kbc.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=_recorder(spawns))

    assert len(spawns) == 3
    assert res.cpu_pressure is None


def test_dispatch_critical_cpu_still_runs_reclaim_bookkeeping(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """The guard must only stop NEW spawns — reclaim/promotion still run."""
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: _cpu_sample("critical"))
    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: {})

    with kbc.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="alice")
        child = kb.create_task(conn, title="child", assignee="alice", parents=[parent])
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (parent,))
        res = kbd.dispatch_once(conn, spawn_fn=lambda *a, **k: 42)
        row = kb.get_task(conn, child)

    assert res.cpu_pressure == "critical"
    assert row is not None
    assert row.status == "ready"


def test_cpu_guard_is_independent_of_the_memory_guard(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """CPU critical holds a tick even when memory is fine, and vice versa."""
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: _cpu_sample("critical"))
    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: {
        "mem_available_kib": 512 * 1024, "mem_total_kib": 1024 * 1024,
    })
    spawns: list = []

    with kbc.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=_recorder(spawns))

    assert not spawns
    assert res.cpu_pressure == "critical"
    assert res.memory_pressure is None


def test_both_guards_elevated_still_spawn_one(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """Two "at most one" caps must not compound into "at most zero"."""
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: _cpu_sample("elevated"))
    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: {
        "mem_available_kib": 100 * 1024, "mem_total_kib": 1024 * 1024,
    })
    spawns: list = []

    with kbc.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=_recorder(spawns))

    assert len(spawns) == 1
    assert res.cpu_pressure == "elevated"
    assert res.memory_pressure == "elevated"


def test_cpu_guard_is_off_on_a_host_that_cannot_report_load(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """The REAL sampler must fail open on this host, whatever its true load is.

    Never calls the CPU sampler for real on a saturated box: the point is that
    an unreadable signal yields no restriction.
    """
    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: {})
    monkeypatch.setattr(kbd, "_system_cpu_sample", lambda: {})
    assert kbd._cpu_pressure_level() == "unknown"

    # The live sampler itself must not raise whatever the host is.
    from gateway import cpu_status

    assert isinstance(cpu_status.sample_cpu_pressure(), dict)


# ---------------------------------------------------------------------------
# Visibility: the tick summary names the hold
# ---------------------------------------------------------------------------


def test_describe_suppression_names_cpu_pressure():
    held = kbd.describe_suppression([kb.DispatchResult(cpu_pressure="critical")])
    assert "cpu_pressure=critical" in held


def test_describe_suppression_omits_cpu_pressure_when_unrestricted():
    assert kbd.describe_suppression([kb.DispatchResult()]) == ""
