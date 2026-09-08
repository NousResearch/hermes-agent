"""Direct/standalone elevated-CPU admission (t_4008d306 rework 3).

The t_75790124 independent review found that ``dispatch_once(host_cycle=None)``
-- the direct/manual call path used by ``hermes kanban dispatch`` and the
standalone daemon (no shared :class:`~hermes_cli.kanban_db_dispatch.
HostCyclePressure`) -- classified elevated CPU pressure and set
``spawn_budget = 1``, but only accounted for it via ``DispatchResult.spawned``.
An ambiguous callback exception, or a PID-persistence failure after a real
launch, left ``spawned`` at 0 and let the very next ready/review row attempt
another launch, contradicting the documented at-most-one-worker cap.

It also found ``_call_spawn_fn`` wrapped both ``inspect.signature`` AND the
actual ``spawn_fn`` invocation in one ``except (TypeError, ValueError)``, so a
board-aware callback that itself raised ``TypeError`` was silently retried a
second time without ``board``.

This module proves the fix directly against ``dispatch_once`` with no
``host_cycle`` argument (the exact call shape of the rejected paths), by
counting real callback invocations rather than trusting ``DispatchResult``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _elevated_sample() -> dict:
    return {"load1": 9.0, "cpu_count": 8, "psi_some_avg60": 25.0}


def _mark_review(conn, task_id: str) -> None:
    conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (task_id,))


# ---------------------------------------------------------------------------
# Ambiguous callback exception: at most one real invocation, ready and review
# ---------------------------------------------------------------------------


def test_RED_direct_elevated_ready_ambiguous_exception_invokes_at_most_once(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(kbd, "_system_cpu_sample", _elevated_sample)
    calls = []

    def flaky_spawn(task, workspace, board=None):
        calls.append(task.id)
        raise RuntimeError("ambiguous: may have already forked before raising")

    with kbc.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=flaky_spawn)

    assert len(calls) <= 1, calls
    assert res.spawned == []
    assert res.cpu_pressure == "elevated"


def test_RED_direct_elevated_review_ambiguous_exception_invokes_at_most_once(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(kbd, "_system_cpu_sample", _elevated_sample)
    calls = []

    def flaky_spawn(task, workspace, board=None):
        calls.append(task.id)
        raise RuntimeError("ambiguous: may have already forked before raising")

    with kbc.connect() as conn:
        ids = [kb.create_task(conn, title=t, assignee="alice") for t in ("a", "b", "c")]
        for tid in ids:
            _mark_review(conn, tid)
        res = kbd.dispatch_once(conn, spawn_fn=flaky_spawn)

    assert len(calls) <= 1, calls
    assert res.spawned == []
    assert res.cpu_pressure == "elevated"


# ---------------------------------------------------------------------------
# PID 4242 then _set_worker_pid OSError: at most one real invocation
# ---------------------------------------------------------------------------


def test_RED_direct_elevated_ready_pid_persistence_failure_invokes_at_most_once(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(kbd, "_system_cpu_sample", _elevated_sample)
    calls = []

    def launching_spawn(task, workspace, board=None):
        calls.append(task.id)
        return 4242

    def failing_set_worker_pid(conn, task_id, pid):
        raise OSError("simulated PID persistence fault")

    monkeypatch.setattr(kbd, "_set_worker_pid", failing_set_worker_pid)

    with kbc.connect() as conn:
        for title in ("a", "b", "c"):
            kb.create_task(conn, title=title, assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=launching_spawn)

    assert len(calls) <= 1, calls
    assert res.spawned == []
    assert res.cpu_pressure == "elevated"


def test_RED_direct_elevated_review_pid_persistence_failure_invokes_at_most_once(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    monkeypatch.setattr(kbd, "_system_cpu_sample", _elevated_sample)
    calls = []

    def launching_spawn(task, workspace, board=None):
        calls.append(task.id)
        return 4242

    def failing_set_worker_pid(conn, task_id, pid):
        raise OSError("simulated PID persistence fault")

    monkeypatch.setattr(kbd, "_set_worker_pid", failing_set_worker_pid)

    with kbc.connect() as conn:
        ids = [kb.create_task(conn, title=t, assignee="alice") for t in ("a", "b", "c")]
        for tid in ids:
            _mark_review(conn, tid)
        res = kbd.dispatch_once(conn, spawn_fn=launching_spawn)

    assert len(calls) <= 1, calls
    assert res.spawned == []
    assert res.cpu_pressure == "elevated"


# ---------------------------------------------------------------------------
# Board-aware callback raising TypeError must invoke exactly once
# ---------------------------------------------------------------------------


def test_RED_direct_board_aware_callback_typeerror_invokes_exactly_once(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    """The callback accepts ``board`` (so ``inspect.signature`` succeeds) and
    raises ``TypeError`` from its OWN body. The old ``_call_spawn_fn`` caught
    that ``TypeError`` in the same handler guarding ``inspect.signature`` and
    retried the callback a second time without ``board`` -- this must not
    happen: the callback's own ``TypeError`` propagates once."""
    calls = []

    def board_aware_spawn(task, workspace, board=None):
        calls.append((board, task.id))
        raise TypeError("callback's own bug, not a signature mismatch")

    with kbc.connect() as conn:
        kb.create_task(conn, title="a", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=board_aware_spawn)

    assert len(calls) == 1, calls
    assert res.spawned == []


def test_RED_direct_board_aware_callback_typeerror_review_lane_invokes_exactly_once(
    kanban_home, all_assignees_spawnable, monkeypatch,
):
    calls = []

    def board_aware_spawn(task, workspace, board=None):
        calls.append((board, task.id))
        raise TypeError("callback's own bug, not a signature mismatch")

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="a", assignee="alice")
        _mark_review(conn, tid)
        res = kbd.dispatch_once(conn, spawn_fn=board_aware_spawn)

    assert len(calls) == 1, calls
    assert res.spawned == []


# ---------------------------------------------------------------------------
# Pre-spawn non-attempt must not consume the per-call elevated slot
# ---------------------------------------------------------------------------


def test_direct_elevated_pre_spawn_rejection_leaves_slot_available_for_later_task(
    kanban_home, monkeypatch,
):
    """A row rejected before any spawn attempt (unknown/nonspawnable profile)
    must not spend the per-call elevated reservation -- a later eligible row
    in the SAME call must still get its one spawn."""
    monkeypatch.setattr(kbd, "_system_cpu_sample", _elevated_sample)
    from hermes_cli import profiles
    monkeypatch.setattr(profiles, "profile_exists", lambda name: name == "alice")
    spawns = []

    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 4242

    with kbc.connect() as conn:
        rejected = kb.create_task(conn, title="rejected", assignee="not-a-profile")
        eligible = kb.create_task(conn, title="eligible", assignee="alice")
        res = kbd.dispatch_once(conn, spawn_fn=fake_spawn)

    assert rejected in res.skipped_nonspawnable
    assert spawns == [eligible]
    assert len(res.spawned) == 1 and res.spawned[0][0] == eligible
