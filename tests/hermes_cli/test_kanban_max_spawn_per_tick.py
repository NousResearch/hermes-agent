"""G3 — distinct per-tick spawn budget: kanban.max_spawn_per_tick.

Approved outcome (Sahil, 2026-07-13):
    up to 5 live/running Kanban workers          -> kanban.max_in_progress (existing)
    at most 3 newly spawned workers per cycle    -> kanban.max_spawn_per_tick (NEW)
    never spawn kensei/denji/orchestrator/misa-misa

``max_spawn`` stays a live-concurrency cap and is NOT touched. ``max_spawn_per_tick``
limits only the number of starts made during a single ``dispatch_once`` call, and
applies identically under ``dry_run`` (so --dry-run reports the real cap). When the
key is absent (None) or invalid, behaviour is unchanged (no per-tick cap).

Proven with the REAL dispatch path (stub spawn_fn, not mocked dispatch).
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (no root config)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Drop cached modules so config/home resolution rebinds to this home.
    from hermes_cli import kanban_db as _kb
    _kb.init_db()
    return _kb, home


@pytest.fixture()
def no_stagger(monkeypatch):
    """No-op the same-profile stagger sleep so real-spawn tests stay fast."""
    monkeypatch.setattr(kb.time, "sleep", lambda *a, **k: None)


def _recorder():
    spawns = []

    def fake_spawn(task, workspace):
        spawns.append((task.id, task.assignee))
        return 4321  # pretend PID

    return spawns, fake_spawn


def _make_ready(_kb, conn, n, assignee="alice", prefix="r"):
    for i in range(n):
        _kb.create_task(conn, title=f"{prefix}{i}", assignee=assignee)


def _make_running(_kb, conn, n, assignee="alice", prefix="run"):
    for i in range(n):
        tid = _kb.create_task(conn, title=f"{prefix}{i}", assignee=assignee)
        _kb.claim_task(conn, tid)  # -> status='running'


# ---------------------------------------------------------------------------
# required matrix — real dispatch path
# ---------------------------------------------------------------------------
def test_zero_running_eight_ready_starts_three(kanban_home, all_assignees_spawnable, no_stagger):
    _kb, home = kanban_home
    spawns, fake_spawn = _recorder()
    with _kb.connect() as conn:
        _make_ready(_kb, conn, 8)
        _kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn_per_tick=3)
    assert len(spawns) == 3


def test_three_running_eight_ready_starts_two_ceiling(kanban_home, all_assignees_spawnable, no_stagger):
    """max_in_progress=5 ceiling with 3 running leaves room for 2, even though
    the per-tick budget (3) would otherwise allow 3."""
    _kb, home = kanban_home
    spawns, fake_spawn = _recorder()
    with _kb.connect() as conn:
        _make_running(_kb, conn, 3)
        _make_ready(_kb, conn, 8)
        _kb.dispatch_once(
            conn, spawn_fn=fake_spawn, max_in_progress=5, max_spawn_per_tick=3,
        )
    assert len(spawns) == 2


def test_five_running_eight_ready_starts_zero(kanban_home, all_assignees_spawnable, no_stagger):
    _kb, home = kanban_home
    spawns, fake_spawn = _recorder()
    with _kb.connect() as conn:
        _make_running(_kb, conn, 5)
        _make_ready(_kb, conn, 8)
        _kb.dispatch_once(
            conn, spawn_fn=fake_spawn, max_in_progress=5, max_spawn_per_tick=3,
        )
    assert len(spawns) == 0


def test_zero_running_two_ready_starts_two(kanban_home, all_assignees_spawnable, no_stagger):
    _kb, home = kanban_home
    spawns, fake_spawn = _recorder()
    with _kb.connect() as conn:
        _make_ready(_kb, conn, 2)
        _kb.dispatch_once(
            conn, spawn_fn=fake_spawn, max_in_progress=5, max_spawn_per_tick=3,
        )
    assert len(spawns) == 2


def test_per_tick_budget_alone_caps_at_three(kanban_home, all_assignees_spawnable, no_stagger):
    """No max_in_progress: the per-tick budget is the sole cap -> 3 of 8."""
    _kb, home = kanban_home
    spawns, fake_spawn = _recorder()
    with _kb.connect() as conn:
        _make_ready(_kb, conn, 8)
        _kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn_per_tick=3)
    assert len(spawns) == 3


# ---------------------------------------------------------------------------
# protected profiles never start (synthetic config, not the root config)
# ---------------------------------------------------------------------------
@pytest.fixture()
def home_with_protected_config(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    # Synthetic minimal config — NOT the live root config.
    (home / "config.yaml").write_text(
        "kanban:\n"
        "  nonspawnable_profiles:\n"
        "    - kensei\n"
        "    - denji\n"
        "    - orchestrator\n"
        "    - misa-misa\n",
        encoding="utf-8",
    )
    for prof in ["kensei", "denji", "orchestrator", "misa-misa", "worker-ok", "default"]:
        prof_dir = home / "profiles" / prof
        prof_dir.mkdir(parents=True, exist_ok=True)
        # ``profile_exists`` requires an identity marker (see
        # ``hermes_constants._PROFILE_IDENTITY_MARKERS``): a bare directory is
        # a ghost shell and resolves as a non-profile. Give each synthetic
        # profile one so the spawnability gate is exercised for the right
        # reason — the ``nonspawnable_profiles`` list — rather than tripping
        # the marker check and blocking every assignee indiscriminately.
        (prof_dir / "config.yaml").write_text("tier: 1\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    from hermes_cli import kanban_db as _kb
    _kb.init_db()
    return _kb, home


def test_protected_profiles_never_start(home_with_protected_config, no_stagger):
    _kb, home = home_with_protected_config
    spawns, fake_spawn = _recorder()
    with _kb.connect() as conn:
        for name in ["kensei", "denji", "orchestrator", "misa-misa"]:
            _kb.create_task(conn, title=f"t-{name}", assignee=name)
        _kb.create_task(conn, title="t-ok", assignee="worker-ok")
        res = _kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn_per_tick=3)
    started = [a for _, a in spawns]
    assert "worker-ok" in started
    for name in ["kensei", "denji", "orchestrator", "misa-misa"]:
        assert name not in started
    assert len(res.skipped_nonspawnable) == 4


# ---------------------------------------------------------------------------
# backward compatibility + invalid handling
# ---------------------------------------------------------------------------
def test_missing_key_is_unlimited(kanban_home, all_assignees_spawnable, no_stagger):
    """max_spawn_per_tick absent (None) -> no per-tick cap -> all ready start."""
    _kb, home = kanban_home
    spawns, fake_spawn = _recorder()
    with _kb.connect() as conn:
        _make_ready(_kb, conn, 8)
        _kb.dispatch_once(conn, spawn_fn=fake_spawn)  # no kwarg at all
    assert len(spawns) == 8


@pytest.mark.parametrize("bad", [0, -1, -5, "abc", 3.5, True, False])
def test_invalid_values_treated_as_no_cap(kanban_home, all_assignees_spawnable, no_stagger, bad):
    """Non-positive-int values coerce to 'no cap' (documented safe default),
    never crash the dispatcher."""
    _kb, home = kanban_home
    spawns, fake_spawn = _recorder()
    with _kb.connect() as conn:
        _make_ready(_kb, conn, 4)
        _kb.dispatch_once(conn, spawn_fn=fake_spawn, max_spawn_per_tick=bad)
    assert len(spawns) == 4


# ---------------------------------------------------------------------------
# dry-run parity (fixes the earlier dry-run over-report caveat for this cap)
# ---------------------------------------------------------------------------
def test_dry_run_matches_real_dispatch(kanban_home, all_assignees_spawnable, no_stagger):
    _kb, home = kanban_home
    # real
    spawns, fake_spawn = _recorder()
    with _kb.connect() as conn:
        _make_ready(_kb, conn, 8)
        _kb.dispatch_once(
            conn, spawn_fn=fake_spawn, max_in_progress=5, max_spawn_per_tick=3,
        )
    real_n = len(spawns)
    # dry-run on an identical fresh board
    with _kb.connect() as conn:
        # clear + re-seed
        with _kb.write_txn(conn):
            conn.execute("DELETE FROM tasks")
        _make_ready(_kb, conn, 8)
        res = _kb.dispatch_once(
            conn, spawn_fn=fake_spawn, dry_run=True,
            max_in_progress=5, max_spawn_per_tick=3,
        )
    dry_n = len(res.spawned)
    assert real_n == 3
    assert dry_n == real_n, f"dry-run reported {dry_n}, real dispatched {real_n}"


# ---------------------------------------------------------------------------
# config default + CLI wiring
# ---------------------------------------------------------------------------
def test_default_config_has_key_none():
    from hermes_cli import config as cfg
    assert cfg.DEFAULT_CONFIG["kanban"].get("max_spawn_per_tick", "MISSING") is None


def test_cli_dispatch_passes_max_spawn_per_tick_from_config(monkeypatch):
    test_home = tempfile.mkdtemp(prefix="kanban_per_tick_cli_")
    os.makedirs(os.path.join(test_home, "profiles", "default"), exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", test_home)
    from hermes_cli import kanban as kb_cli
    from hermes_cli import kanban_db

    fake_config = {"kanban": {"max_in_progress": 5, "max_spawn_per_tick": 3}}
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: fake_config)

    captured = {}

    # Patch the REAL dispatch entry point. ``kanban_db.dispatch_once`` is a
    # deprecated compat shim (removal 2026-09-14) that only re-exports the
    # symbol for external plugins; ``_cmd_dispatch`` resolves it through
    # ``kanban_db_dispatch`` at call time, so patching the old alias leaves
    # the real call unpatched and this test asserts against an empty dict.
    from hermes_cli import kanban_db_dispatch

    monkeypatch.setattr(
        kanban_db_dispatch, "dispatch_once",
        lambda conn, **kw: (captured.update(kw), kanban_db.DispatchResult())[1],
    )
    args = argparse.Namespace(dry_run=True, max=None, failure_limit=2, json=False)
    kb_cli._cmd_dispatch(args)
    assert captured.get("max_spawn_per_tick") == 3
    assert captured.get("max_in_progress") == 5
