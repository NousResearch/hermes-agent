"""The dispatcher-stuck warning must not fire for work the tick held back.

The health telemetry in ``hermes kanban daemon`` (and its mirror in the
embedded gateway dispatcher) counts a tick bad when the ready queue is
non-empty and nothing spawned. A queue parked at
``kanban.max_in_progress_per_profile`` is the cap doing its configured job —
counting those ticks produced an 80-tick ``dispatcher stuck: ... Check profile
health (venv, PATH, credentials)`` streak on a perfectly healthy fleet. The
held-back buckets (``skipped_per_profile_capped``, ``respawn_guarded``,
``rate_limited``, ``skipped_locked``, ``memory_pressure``) must read as
"correctly idle", exactly like the control-plane ``claim_task`` lanes the
readiness probe already excludes — while a genuinely unexplained zero-spawn
tick must still warn, and its message must name any partial hold.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile

import pytest


@pytest.fixture()
def isolated_kanban_home_with_profiles(monkeypatch):
    """Spin up a fresh HERMES_HOME with kanban DB + alpha/beta profiles."""
    test_home = tempfile.mkdtemp(prefix="kanban_stuck_telemetry_test_")
    for prof in ("alpha", "beta", "default"):
        os.makedirs(os.path.join(test_home, "profiles", prof), exist_ok=True)
        with open(os.path.join(test_home, "profiles", prof, "config.yaml"), "w") as fh:
            fh.write("{}\n")  # identity marker: a bare dir is not a profile
    monkeypatch.setenv("HERMES_HOME", test_home)
    for mod in list(sys.modules.keys()):
        if mod.startswith("hermes_cli") or mod.startswith("hermes_state") or mod == "hermes_constants":
            del sys.modules[mod]
    from hermes_cli import kanban_db
    from hermes_cli import kanban_db_dispatch
    # The module purge above discards the conftest's memory-sample neutralizer;
    # re-apply it so the host's live memory pressure cannot cap this tick.
    monkeypatch.setattr(kanban_db_dispatch, "_system_memory_sample", dict)
    yield kanban_db


def _daemon_on_tick(monkeypatch):
    """Run ``_cmd_daemon`` against a stubbed ``run_daemon`` and hand back the
    health-telemetry ``on_tick`` callback it installed."""
    from hermes_cli import kanban_db_dispatch as kbd
    from hermes_cli import kanban_ops

    captured = {}

    def _fake_run_daemon(**kwargs):
        captured["on_tick"] = kwargs["on_tick"]

    monkeypatch.setattr(kbd, "run_daemon", _fake_run_daemon)
    args = argparse.Namespace(
        interval=5, max=None, force=True, verbose=False, pidfile=None,
    )
    assert kanban_ops._cmd_daemon(args) == 0
    return captured["on_tick"]


def _seed_ready_alpha(kb, count):
    from hermes_cli import kanban_db_connect as kbc
    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        ids = [kb.create_task(conn, title=f"a{i}", assignee="alpha")
               for i in range(count)]
    return ids


def _saturated_cap_result(kb):
    """Two real ticks with ``max_in_progress_per_profile=1``: the first spawns
    one worker (pid = this live process, so reclaim leaves it running), the
    second is the saturated shape — zero spawns, the rest capped."""
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    live_pid = os.getpid()

    def _fake_spawn(*args, **kwargs):
        return live_pid

    with kbc.connect_closing() as conn:
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn, max_in_progress_per_profile=1)
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(conn, spawn_fn=_fake_spawn, max_in_progress_per_profile=1)
    assert not res.spawned
    assert len(res.skipped_per_profile_capped) == 2
    return res


def test_capped_queue_does_not_count_bad_ticks(isolated_kanban_home_with_profiles, monkeypatch, capsys):
    """The card-level regression: a queue held only by the per-profile cap must
    not raise ``dispatcher stuck`` — the dispatcher is behaving exactly as
    configured, not sitting on a broken venv/PATH/credential."""
    kb = isolated_kanban_home_with_profiles
    _seed_ready_alpha(kb, 3)
    res = _saturated_cap_result(kb)
    on_tick = _daemon_on_tick(monkeypatch)
    for _ in range(8):  # well past HEALTH_WINDOW=6
        on_tick(res)
    err = capsys.readouterr().err
    assert "dispatcher stuck" not in err


def test_whole_tick_hold_does_not_count_bad_ticks(isolated_kanban_home_with_profiles, monkeypatch, capsys):
    """A tick held back wholesale by configuration — here the memory-pressure
    guard — is correctly idle too, not stuck."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_dispatch as kbd
    _seed_ready_alpha(kb, 2)
    res = kbd.DispatchResult(memory_pressure="critical")
    on_tick = _daemon_on_tick(monkeypatch)
    for _ in range(8):
        on_tick(res)
    err = capsys.readouterr().err
    assert "dispatcher stuck" not in err


def test_unexplained_zero_spawn_still_warns(isolated_kanban_home_with_profiles, monkeypatch, capsys):
    """Positive control: the fix must not silence the real signal. Ready,
    spawnable, unclaimed work with a zero-spawn tick and NO hold recorded is
    still exactly the broken-profile case the warning exists for."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_dispatch as kbd
    _seed_ready_alpha(kb, 2)
    res = kbd.DispatchResult()
    on_tick = _daemon_on_tick(monkeypatch)
    for _ in range(8):
        on_tick(res)
    err = capsys.readouterr().err
    assert "dispatcher stuck" in err


def test_mixed_hold_and_unexplained_warns_and_names_the_cap(
    isolated_kanban_home_with_profiles, monkeypatch, capsys,
):
    """Cap held some work back, but another profile's ready work stayed
    unexplained (its spawn keeps failing): the counter legitimately stays
    positive, and the warning must name the actual held-back reason
    (``per_profile_capped=N``) instead of only pointing at profile health."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    live_pid = os.getpid()

    def _spawn(*args, **kwargs):
        task = args[0]
        if getattr(task, "assignee", None) == "beta":
            raise RuntimeError("worker venv is gone")
        return live_pid

    with kbc.connect_closing() as conn:
        kb.create_board(slug="default", name="Test")
        for i in range(3):
            kb.create_task(conn, title=f"a{i}", assignee="alpha")
        kb.create_task(conn, title="b0", assignee="beta")

    # Tick 1: alpha capped at 1 (a0 runs), beta's spawn fails — the task
    # returns to ready WITHOUT landing in any hold bucket.
    with kbc.connect_closing() as conn:
        kbd.dispatch_once(
            conn, spawn_fn=_spawn, max_in_progress_per_profile=1, failure_limit=5,
        )
    # Tick 2: the saturated shape — zero spawns, alpha capped, beta still
    # ready and unexplained.
    with kbc.connect_closing() as conn:
        res = kbd.dispatch_once(
            conn, spawn_fn=_spawn, max_in_progress_per_profile=1, failure_limit=5,
        )
    assert not res.spawned
    assert len(res.skipped_per_profile_capped) == 2

    on_tick = _daemon_on_tick(monkeypatch)
    for _ in range(8):
        on_tick(res)
    err = capsys.readouterr().err
    assert "dispatcher stuck" in err
    assert "per_profile_capped=2" in err


def test_held_ready_task_ids_and_whole_tick_hold():
    """Unit pin of the shared predicates the health telemetry consumes: the
    per-task holds enumerate capped + guarded + rate-limited ids; the
    whole-tick holds are the board lock and the memory-pressure guard."""
    from hermes_cli import kanban_db_dispatch as kbd
    per_task = kbd.DispatchResult(
        skipped_per_profile_capped=[("t1", "alpha", 2)],
        respawn_guarded=[("t2", "active_pr")],
        rate_limited=["t3"],
    )
    assert kbd.held_ready_task_ids(per_task) == {"t1", "t2", "t3"}
    assert kbd.held_ready_task_ids(kbd.DispatchResult()) == set()
    assert kbd.held_ready_task_ids(None) == set()
    assert not kbd.whole_tick_hold(per_task)
    assert kbd.whole_tick_hold(kbd.DispatchResult(skipped_locked=True))
    assert kbd.whole_tick_hold(kbd.DispatchResult(memory_pressure="critical"))
    assert not kbd.whole_tick_hold(kbd.DispatchResult())
    assert not kbd.whole_tick_hold(None)
    # The stuck warning's suppression line must name the cap too.
    assert "per_profile_capped=1" in kbd.describe_suppression([per_task])
    assert kbd.describe_suppression([kbd.DispatchResult(), None]) == ""


def test_has_spawnable_ready_excludes_held_ids(isolated_kanban_home_with_profiles):
    """The readiness probe keeps answering "stuck" for unexplained work while
    subtracting the ids the last tick deliberately held back."""
    kb = isolated_kanban_home_with_profiles
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli import kanban_db_dispatch as kbd
    ids = _seed_ready_alpha(kb, 2)
    with kbc.connect_closing() as conn:
        assert kbd.has_spawnable_ready(conn) is True
        assert kbd.has_spawnable_ready(conn, exclude_task_ids=ids[:1]) is True
        assert kbd.has_spawnable_ready(conn, exclude_task_ids=ids) is False
