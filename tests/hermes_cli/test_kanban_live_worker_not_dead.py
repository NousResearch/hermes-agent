"""A dead-pid verdict must corroborate before a live worker is booked crashed.

Board evidence for the change (board ``vx``, 2026-09-13): a single dispatcher
sweep at 21:57:40Z booked **nine** host-local ``running`` workers ``crashed``
with ``error="pid <n> not alive"``. All nine processes were alive at the
verdict, still alive minutes later, and two of them had ``/proc/<pid>/cwd`` in
the card's own workspace. The systemd journals agree on the far side: the
``hermes-worker-kanban-*-run-*.scope`` of one of them ("Consumed 3min 7.313s
CPU time") does not close until 22:09:21Z, and the gateway reaps its pid as a
zombie at 22:09:42Z — 12 minutes after the "not alive" verdict. The cost is not
only duplicated work: ``complete_task`` pins the row on ``current_run_id``, so
the first worker's ``kanban_complete`` can never land once the card is
reclaimed, and the card is closed by a second worker that has to re-derive
everything the first one already did.

The probe that produced those verdicts is ``_pid_alive`` — a *negative* test
whose False folds "the pid is gone", "the pid is a zombie" and "the probe
itself failed" into one answer. So the reclaim path now requires positive
evidence (``_worker_pid_absent``) plus a stale heartbeat before it acts, and
defers the card otherwise; a deferral keeps the card ``running`` under the
worker it already has instead of spawning a duplicate beside it.
"""

from __future__ import annotations

import os
import sys
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB (mirrors test_kanban_db)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    # These tests exercise the reclaim decision itself, not the launch-window
    # grace that shields a freshly spawned worker.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    kb.init_db()
    return home


def _claim_running(conn, *, pid: int, heartbeat_age: int | None):
    """A ``running`` card claimed by this host, with a worker pid + heartbeat.

    ``heartbeat_age=None`` opens a run that never heartbeated; an int writes the
    heartbeat that many seconds ago (0 = just now).
    """
    host = kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="live worker", assignee="vx-dev")
    assert kb.claim_task(conn, tid, claimer=f"{host}:mock") is not None
    run_id = kb._current_run_id(conn, tid)
    conn.execute(
        "UPDATE tasks SET worker_pid = ? WHERE id = ?", (pid, tid),
    )
    if heartbeat_age is not None:
        stamp = int(time.time()) - heartbeat_age
        conn.execute(
            "UPDATE tasks SET last_heartbeat_at = ? WHERE id = ?", (stamp, tid),
        )
        conn.execute(
            "UPDATE task_runs SET last_heartbeat_at = ? WHERE id = ?", (stamp, run_id),
        )
    conn.commit()
    return tid, run_id


def _events(conn, tid, kind):
    return [e for e in kb.list_events(conn, tid) if e.kind == kind]


def _crashed_ids(conn):
    crashed = kbd.detect_crashed_workers(conn)
    deferred = list(getattr(kbd.detect_crashed_workers, "_last_deferred_live", []))
    return crashed, deferred


def test_live_worker_with_a_fresh_heartbeat_is_deferred_not_booked_crashed(
    kanban_home, monkeypatch,
):
    """The incident in one assertion: probe says dead, heartbeat says alive.

    The probe's False is not evidence of death, and the heartbeat inside the
    window was written by that very process — so the card must stay ``running``
    under its own worker, with its claim, run and worker_pid intact, no crash
    booked and no failure counted.
    """
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

    with kbc.connect() as conn:
        tid, run_id = _claim_running(conn, pid=991100, heartbeat_age=5)
        before = kb.get_task(conn, tid)

        crashed, deferred = _crashed_ids(conn)

        task = kb.get_task(conn, tid)
        assert tid not in crashed, "a heartbeating worker is not a crash"
        assert deferred == [tid]
        assert task.status == "running", (
            f"deferred cards stay running, got {task.status}"
        )
        assert task.status != "done"
        assert task.worker_pid == 991100, "the live worker keeps its claim"
        assert task.claim_lock == before.claim_lock
        assert task.consecutive_failures == 0, "a deferral is not a worker failure"

        # No terminal/crash bookkeeping: the run is still open and owns the card.
        assert not _events(conn, tid, "crashed")
        assert not _events(conn, tid, "gave_up")
        run = kb.list_runs(conn, tid)[-1]
        assert run.id == run_id and run.ended_at is None
        assert kb._current_run_id(conn, tid) == run_id, (
            "the first worker's run id must stay current, or its terminal call "
            "can never land"
        )

        # The hold is recorded, names the disagreement, and outlasts a tick.
        held = _events(conn, tid, "reclaim_deferred")
        assert len(held) == 1
        payload = held[0].payload or {}
        assert payload.get("reason") == "heartbeat_fresher_than_pid_probe"
        assert payload.get("worker_pid") == 991100
        assert payload.get("heartbeat_age_seconds") == 5
        assert task.claim_expires >= int(time.time()) + kb.RECLAIM_DEFER_GRACE_SECONDS - 5


def test_positively_absent_pid_with_a_stale_heartbeat_is_still_reclaimed(
    kanban_home, monkeypatch,
):
    """The corroborated case is unchanged: a really-dead worker is reclaimed.

    Guards the other direction — a probe that is right, and a run that stopped
    heartbeating long ago, must still release the card to its source phase.
    """
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

    with kbc.connect() as conn:
        tid, _ = _claim_running(conn, pid=991101, heartbeat_age=3600)
        crashed, deferred = _crashed_ids(conn)

        task = kb.get_task(conn, tid)
        assert tid in crashed, "an absent pid with a stale heartbeat is a crash"
        assert deferred == []
        assert task.status == "ready"
        assert task.worker_pid is None
        assert _events(conn, tid, "crashed")
        assert not _events(conn, tid, "reclaim_deferred")


def test_run_with_no_heartbeat_ever_still_reclaims(kanban_home, monkeypatch):
    """A worker that never heartbeated is reclaimed once its pid is gone."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)

    with kbc.connect() as conn:
        tid, _ = _claim_running(conn, pid=991102, heartbeat_age=None)
        crashed, deferred = _crashed_ids(conn)
        assert tid in crashed
        assert deferred == []


def test_an_unreadable_proc_entry_is_not_evidence_of_death(kanban_home, monkeypatch):
    """Probe failure defers instead of reclaiming, even with a stale heartbeat.

    ``/proc/<pid>/status`` that cannot be read (EACCES/EMFILE/ENOMEM under load)
    says nothing about whether the worker is alive. Acting on it is what booked
    nine live workers ``crashed`` in one tick.
    """
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setattr(kbd, "_worker_pid_absent", lambda _pid: False)

    with kbc.connect() as conn:
        tid, _ = _claim_running(conn, pid=991103, heartbeat_age=3600)
        crashed, deferred = _crashed_ids(conn)

        assert crashed == []
        assert deferred == [tid]
        assert kb.get_task(conn, tid).status == "running"
        payload = (_events(conn, tid, "reclaim_deferred")[0].payload or {})
        assert payload.get("reason") == "pid_probe_unknown"
        assert payload.get("pid_absent") is False


@pytest.mark.skipif(sys.platform != "linux", reason="/proc probe is Linux-only")
def test_worker_pid_absent_is_tri_state(monkeypatch):
    """``_worker_pid_absent``: gone/zombie are True; unreadable and live are False."""
    live_status = "Name:\tpython\nState:\tS (sleeping)\nPPid:\t1\n"

    class _FakeStatus:
        def __init__(self, text):
            self._text = text

        def __enter__(self):
            import io

            return io.StringIO(self._text)

        def __exit__(self, *_exc):
            return False

    def _fake_open(text=None, exc=None):
        def _open(*_args, **_kwargs):
            if exc is not None:
                raise exc
            return _FakeStatus(text)

        return _open

    # A pid with no /proc entry is positively absent.
    monkeypatch.setattr(kbd, "open", _fake_open(exc=FileNotFoundError()), raising=False)
    assert kbd._worker_pid_absent(991104) is True

    # An unreadable entry is "unknown", never death.
    monkeypatch.setattr(kbd, "open", _fake_open(exc=PermissionError("nope")), raising=False)
    assert kbd._worker_pid_absent(991104) is False

    # A live process is present.
    monkeypatch.setattr(kbd, "open", _fake_open(text=live_status), raising=False)
    assert kbd._worker_pid_absent(991104) is False

    # A zombie has exited: absent for a reclaim's purposes.
    monkeypatch.setattr(
        kbd, "open", _fake_open(text="Name:\tx\nState:\tZ (zombie)\n"), raising=False,
    )
    assert kbd._worker_pid_absent(991104) is True

    # A live process (this one) is present, and a non-pid is absent.
    monkeypatch.delattr(kbd, "open", raising=False)
    assert kbd._worker_pid_absent(os.getpid()) is False
    assert kbd._worker_pid_absent(None) is True
    assert kbd._worker_pid_absent(0) is True


def test_poll_worker_exit_needs_a_positive_absence(monkeypatch):
    """The termination probe reads the same primitive, so it fails closed too."""
    monkeypatch.setattr(kbd.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(kbd, "_worker_pid_absent", lambda _pid: False)
    assert kbd._poll_worker_exit(991105) is False, (
        "an unknown probe must not read as 'the worker died'"
    )
    monkeypatch.setattr(kbd, "_worker_pid_absent", lambda _pid: True)
    assert kbd._poll_worker_exit(991105) is True
