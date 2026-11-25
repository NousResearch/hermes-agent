"""Regression tests: the embedded dispatcher re-reads the kanban concurrency caps.

``gateway/kanban_watchers.py`` resolved ``kanban.*`` once at boot, so changing
``kanban.max_in_progress`` (or ``max_spawn`` / ``max_in_progress_per_profile``)
needed a gateway restart — and a restart interrupts every in-flight worker.
The caps are now re-resolved on every tick and applied in place, so a change
takes effect within one dispatch interval and lowering one never touches a
running worker (#148006, #117734).

The first test is the red proof: on the pre-fix tree the dispatcher keeps the
boot value for the life of the process and records ``[12, 12]``.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


def test_raise_mid_run_applies_on_the_next_tick_without_restart(
    tmp_path, monkeypatch, caplog,
):
    """A cap raised after boot is enforced by the next tick; one line, no restart."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    # The boot read and every tick read share this mutable mapping, so a tick
    # can change the cap exactly the way an operator writing config.yaml would.
    config = {"kanban": {"dispatch_interval_seconds": 1, "max_in_progress": 12}}
    dispatchers: list = []

    class _StubDispatcher:
        """Records the settings each tick actually dispatches with."""

        def __init__(self, kb_module, settings):
            self.kb = kb_module
            self.settings = settings
            self.ticks: list = []
            dispatchers.append(self)

        def tick_once(self):
            self.ticks.append(self.settings.max_in_progress)
            if len(self.ticks) == 1:
                config["kanban"]["max_in_progress"] = 20
            else:
                runner._running = False
            return []

        def auto_decompose_tick(self, per_tick):
            return 0

        def ready_nonempty(self):
            return False

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {}

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    async def fake_sleep(delay):
        return None

    caplog.set_level(logging.INFO, logger="gateway.run")
    with (
        patch("hermes_cli.config.load_config", side_effect=lambda: config),
        patch("gateway.kanban_watchers._acquire_singleton_lock", return_value=(None, "held")),
        patch("gateway.kanban_watchers._release_singleton_lock"),
        patch("gateway.kanban_watchers._kanban_dispatch_allowed", return_value=True),
        patch("gateway.kanban_watchers._KanbanDispatcher", _StubDispatcher),
        patch("hermes_cli.kanban_db_dispatch.reap_worker_zombies", return_value=[]),
        patch("asyncio.sleep", side_effect=fake_sleep),
        patch("asyncio.to_thread", side_effect=fake_to_thread),
    ):
        asyncio.run(runner._kanban_dispatcher_watcher())

    assert len(dispatchers) == 1, "the dispatcher must not be re-constructed mid-run"
    (stub,) = dispatchers
    assert stub.ticks == [12, 20], "tick 2 must enforce the raised cap without a restart"

    live_lines = [r.getMessage() for r in caplog.records if "12 -> 20" in r.getMessage()]
    assert live_lines == [
        "kanban dispatcher: max_in_progress 12 -> 20 (live; next tick, no restart)"
    ], f"exactly one old->new INFO line per effective change; got {live_lines}"


def test_resolve_live_caps_explicit_or_memory_derived(monkeypatch):
    """Explicit config wins; unset/invalid falls back to the memory-derived cap."""
    from gateway.kanban_watchers_dispatcher import _resolve_live_caps

    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: {"mem_total_kib": 16 * 1024 * 1024})

    caps = _resolve_live_caps(
        {"max_spawn": 4, "max_in_progress": 20, "max_in_progress_per_profile": 2}
    )
    assert (caps.max_spawn, caps.max_in_progress, caps.max_in_progress_per_profile) == (4, 20, 2)
    assert caps.invalid == ()

    derived = _resolve_live_caps({"max_in_progress": 0})
    assert derived.max_in_progress == 8  # clamp(16 GiB / 512 MiB = 32, floor 2, ceiling 8)
    assert derived.invalid == (("max_in_progress", 0),)


def test_boot_settings_and_live_caps_parse_a_cap_identically(monkeypatch):
    """Boot and the per-tick view must agree on every cap value, ``max_spawn`` included.

    ``max_spawn`` used to stay raw at boot: ``0`` paused dispatch there while
    the first live apply read it as ``None`` (uncapped), so editing an
    unrelated cap resumed spawning — and ``"4"`` stayed a string at boot.
    ``< 1`` and non-integers now mean unset for all three caps at both reads.
    """
    from gateway.kanban_watchers_dispatcher import (
        _resolve_dispatcher_settings,
        _resolve_live_caps,
    )

    # No live-memory restriction, so max_in_progress is deterministic.
    monkeypatch.setattr(kbd, "_system_memory_sample", dict)

    cases = [
        ({}, (None, None, None)),
        (
            {"max_spawn": 4, "max_in_progress": 20, "max_in_progress_per_profile": 2},
            (4, 20, 2),
        ),
        ({"max_spawn": "4"}, (4, None, None)),
        ({"max_spawn": 0}, (None, None, None)),
        ({"max_spawn": -1}, (None, None, None)),
        ({"max_spawn": "many"}, (None, None, None)),
    ]
    for cfg, expected in cases:
        settings = _resolve_dispatcher_settings(cfg, kbd)
        live = _resolve_live_caps(cfg)
        assert (
            settings.max_spawn,
            settings.max_in_progress,
            settings.max_in_progress_per_profile,
        ) == expected, cfg
        assert (
            live.max_spawn,
            live.max_in_progress,
            live.max_in_progress_per_profile,
        ) == expected, cfg


def test_boot_rejects_a_bad_max_spawn_with_one_warning(caplog):
    """A rejected boot value warns once and behaves as unset, not as a pause."""
    from gateway.kanban_watchers_dispatcher import _resolve_dispatcher_settings

    caplog.set_level(logging.INFO, logger="gateway.run")
    settings = _resolve_dispatcher_settings({"max_spawn": 0}, kbd)

    assert settings.max_spawn is None
    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "max_spawn" in record.getMessage()
    ]
    assert warnings == ["kanban dispatcher: invalid kanban.max_spawn=0; ignoring"]


def test_unchanged_caps_log_nothing_and_a_change_logs_once(caplog):
    """The change line fires on a change only — never once per tick."""
    from gateway.kanban_watchers_dispatcher import _log_live_cap_change, _resolve_live_caps

    caplog.set_level(logging.INFO, logger="gateway.run")
    before = _resolve_live_caps({"max_in_progress": 20})

    _log_live_cap_change(before, _resolve_live_caps({"max_in_progress": 20}))
    assert [r for r in caplog.records if "max_in_progress" in r.getMessage()] == []

    _log_live_cap_change(before, _resolve_live_caps({"max_in_progress": 8}))
    assert [r.getMessage() for r in caplog.records if "max_in_progress" in r.getMessage()] == [
        "kanban dispatcher: max_in_progress 20 -> 8 (live; next tick, no restart)"
    ]


def test_unreadable_config_keeps_the_last_applied_caps():
    """A broken config read is fail-safe: the loop keeps the caps it has."""
    from gateway.kanban_watchers import _kanban_cfg_now

    def boom():
        raise RuntimeError("broken yaml")

    assert _kanban_cfg_now(boom) is None
    assert _kanban_cfg_now(lambda: {"kanban": "not-a-mapping"}) is None
    assert _kanban_cfg_now(lambda: {"kanban": {"max_in_progress": 4}}) == {"max_in_progress": 4}


def test_lower_mid_run_stops_spawning_and_leaves_running_workers(kanban_home, monkeypatch):
    """Lowering a cap never touches running workers; it only stops the next spawn."""
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: True)
    # Deterministic: no live-memory restriction on the spawn budget.
    monkeypatch.setattr(kbd, "_system_memory_sample", dict)

    spawns: list = []
    with kbc.connect() as conn:
        # 12 workers already running, claim bookkeeping intact so
        # reconcile_orphans leaves them alone.
        running_ids = []
        for i in range(12):
            tid = kb.create_task(conn, title=f"busy-{i}", assignee="alice")
            assert kb.claim_task(conn, tid) is not None
            running_ids.append(tid)
        for i in range(4):
            kb.create_task(conn, title=f"ready-{i}", assignee="alice")

        # Cap above the running count: the ready backlog fills the free budget.
        res = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=20,
        )
        assert len(res.spawned) == 4
        all_running = running_ids + [task_id for task_id, *_ in res.spawned]

        # Lower the cap below the running count (16 running > 8): no spawns, and
        # every previously running row keeps running with its claim intact.
        res2 = kbd.dispatch_once(
            conn, spawn_fn=_fake_spawn_factory(spawns), max_in_progress=8,
        )
        assert res2.spawned == []
        assert res2.reclaimed == 0
        assert res2.crashed == []

        rows = {
            row[0]: (row[1], row[2])
            for row in conn.execute(
                "SELECT id, status, claim_lock FROM tasks WHERE status = 'running'"
            )
        }
        for tid in all_running:
            status, claim_lock = rows[tid]
            assert status == "running"
            assert claim_lock, f"task {tid} lost its claim — a cap must not touch running workers"
