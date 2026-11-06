"""Invariant: once a terminal run's client is provably gone, the reaper releases
that run's systemd scope unit by its exact name — never while the client lives,
never by pattern, never past the release window.

A ``--collect`` scope stays active while any straggler keeps its cgroup
non-empty, so a worker that hung after ``kanban_complete`` (or left a daemonised
grandchild) pinned ``hermes-worker-kanban-<task>-run-<run>.scope`` forever. The
terminal-worker reaper (#111791) now stops that unit after the pid is gone,
everything derived from the ledger row.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_dispatch_worker_scope as ws
from tools import process_registry

_EXACT_UNIT = re.compile(r"^hermes-worker-kanban-[A-Za-z0-9_-]+-run-\d+\.scope$")


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


@pytest.fixture
def systemd(monkeypatch):
    """Fake systemd seams; on teardown every unit ever touched must be an exact name."""
    fake = SimpleNamespace(active=True, stop_ok=True, is_active_calls=[], stop_calls=[], on_is_active=None)

    def is_active(unit):
        fake.is_active_calls.append(unit)
        if fake.on_is_active:
            fake.on_is_active(unit)
        return fake.active

    def stop(unit):
        fake.stop_calls.append(unit)
        return fake.stop_ok

    monkeypatch.setattr(ws, "_scope_unit_is_active", is_active)
    monkeypatch.setattr(process_registry, "_stop_systemd_unit", stop)
    yield fake
    for unit in fake.is_active_calls + fake.stop_calls:
        assert _EXACT_UNIT.match(unit), unit  # no glob: the pattern admits no "*", "?" or "["


def _sleeper():
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(120)"],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _completed_card_with_worker(conn, proc, *, ended_ago: int = 600) -> tuple[str, int]:
    tid = kb.create_task(conn, title="finished", assignee="coder")
    kb.claim_task(conn, tid, claimer=kb._claimer_id())
    run_id = kb._current_run_id(conn, tid)
    kbd._set_worker_pid(conn, tid, proc.pid)
    assert kb.complete_task(conn, tid, result="done", expected_run_id=run_id) is True
    conn.execute("UPDATE task_runs SET ended_at = ended_at - ? WHERE id=?", (ended_ago, run_id))
    return tid, run_id


def _completed_card_with_gone_worker(conn, *, ended_ago: int = 600) -> tuple[str, int, int]:
    proc = _sleeper()
    tid, run_id = _completed_card_with_worker(conn, proc, ended_ago=ended_ago)
    proc.kill()
    proc.wait()
    return tid, run_id, proc.pid


def _unit_from_ledger(conn, run_id: int) -> str:
    row = conn.execute("SELECT id, task_id FROM task_runs WHERE id=?", (run_id,)).fetchone()
    return f"hermes-worker-kanban-{row['task_id']}-run-{row['id']}.scope"


def _evidence(conn, run_id: int):
    row = conn.execute("SELECT worker_pid, worker_started_at FROM task_runs WHERE id=?", (run_id,)).fetchone()
    return row["worker_pid"], row["worker_started_at"]


def _events(conn, tid: str, kind: str) -> list[dict]:
    return [e.payload for e in kb.list_events(conn, tid) if e.kind == kind]


def test_gone_client_with_active_scope_stops_the_exact_unit(conn, systemd):
    tid, run_id, pid = _completed_card_with_gone_worker(conn)
    unit = _unit_from_ledger(conn, run_id)
    assert kbd._worker_scope_unit_name(tid, run_id) == unit  # facade re-export of the sibling's name

    kbd.reap_terminal_workers(conn)

    assert systemd.stop_calls == [unit]
    assert _evidence(conn, run_id) == (None, None)
    assert _events(conn, tid, "worker_scope_released") == [{"pid": pid, "released": True, "unit": unit}]


def test_inactive_scope_is_never_stopped_and_leaves_no_event(conn, systemd):
    systemd.active = False
    tid, run_id, _pid = _completed_card_with_gone_worker(conn)

    kbd.reap_terminal_workers(conn)

    assert systemd.is_active_calls == [_unit_from_ledger(conn, run_id)]
    assert systemd.stop_calls == []
    assert _events(conn, tid, "worker_scope_released") == []
    assert _evidence(conn, run_id) == (None, None)


def test_scope_is_released_only_after_sigterm_ends_the_live_client(conn, systemd):
    proc = _sleeper()
    client_exit_at_release = []
    systemd.on_is_active = lambda _unit: client_exit_at_release.append(proc.poll())
    try:
        tid, run_id = _completed_card_with_worker(conn, proc)
        unit = _unit_from_ledger(conn, run_id)

        assert kbd.reap_terminal_workers(conn) == [tid]

        assert client_exit_at_release and client_exit_at_release[0] is not None
        assert systemd.stop_calls == [unit]
        [reaped] = _events(conn, tid, "terminal_worker_reaped")
        assert reaped["scope"] == {"released": True, "unit": unit}
        assert _events(conn, tid, "worker_scope_released") == [{"pid": proc.pid, "released": True, "unit": unit}]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def test_surviving_client_never_touches_systemd_and_keeps_evidence(conn, systemd, monkeypatch):
    proc = _sleeper()
    monkeypatch.setattr(kbd, "_terminate_reclaimed_worker", lambda *a, **k: {"terminated": False})
    try:
        tid, run_id = _completed_card_with_worker(conn, proc)
        before = _evidence(conn, run_id)

        assert kbd.reap_terminal_workers(conn) == []

        assert systemd.is_active_calls == [] and systemd.stop_calls == []
        assert _evidence(conn, run_id) == before and before[0] == proc.pid
        assert _events(conn, tid, "worker_scope_released") == []
    finally:
        proc.kill()
        proc.wait()


def test_run_ended_past_release_window_never_touches_systemd(conn, systemd):
    tid, run_id, _pid = _completed_card_with_gone_worker(
        conn, ended_ago=kbd.TERMINAL_WORKER_SCOPE_RELEASE_WINDOW_SECONDS + 60)

    kbd.reap_terminal_workers(conn)

    assert systemd.is_active_calls == [] and systemd.stop_calls == []
    assert _evidence(conn, run_id) == (None, None)
    assert _events(conn, tid, "worker_scope_released") == []


def test_failed_stop_is_recorded_and_evidence_still_cleared(conn, systemd):
    systemd.stop_ok = False
    tid, run_id, pid = _completed_card_with_gone_worker(conn)
    unit = _unit_from_ledger(conn, run_id)

    kbd.reap_terminal_workers(conn)

    assert systemd.stop_calls == [unit]
    assert _events(conn, tid, "worker_scope_released") == [{"pid": pid, "released": False, "unit": unit}]
    assert _evidence(conn, run_id) == (None, None)


def test_non_host_local_claim_is_never_released_or_cleared(conn, systemd):
    _tid, run_id, pid = _completed_card_with_gone_worker(conn)
    conn.execute("UPDATE task_runs SET claim_lock = 'elsewhere:1' WHERE id=?", (run_id,))

    kbd.reap_terminal_workers(conn)

    assert systemd.is_active_calls == [] and systemd.stop_calls == []
    assert _evidence(conn, run_id)[0] == pid


def test_every_unit_touched_in_a_sweep_is_an_exact_ledger_name(conn, systemd):
    runs = [_completed_card_with_gone_worker(conn)[1] for _ in range(3)]

    kbd.reap_terminal_workers(conn)

    expected = sorted(_unit_from_ledger(conn, run_id) for run_id in runs)
    assert sorted(systemd.is_active_calls) == expected
    assert sorted(systemd.stop_calls) == expected


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        (b"active\n", True),
        (b"activating\n", True),
        (b"deactivating\n", False),
        (b"inactive\n", False),
        (b"failed\n", False),
        (b"unknown\n", False),
        (b"", False),
    ],
)
def test_scope_is_active_only_for_active_or_activating(monkeypatch, stdout, expected):
    calls = []
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemctl")
    monkeypatch.setattr(process_registry, "systemd_user_bus_env", lambda: {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/x"})

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(ws.subprocess, "run", run)
    unit = "hermes-worker-kanban-t_1-run-2.scope"

    assert ws._scope_unit_is_active(unit) is expected

    [(argv, kwargs)] = calls
    assert argv == ["/usr/bin/systemctl", "--user", "is-active", unit]
    assert kwargs["env"] == {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/x"}
    assert kwargs["timeout"] == 10


def test_scope_is_active_is_false_when_systemctl_raises_or_is_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/systemctl")

    def boom(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 10)

    monkeypatch.setattr(ws.subprocess, "run", boom)
    assert ws._scope_unit_is_active("hermes-worker-kanban-t_1-run-2.scope") is False

    monkeypatch.setattr("shutil.which", lambda name: None)
    assert ws._scope_unit_is_active("hermes-worker-kanban-t_1-run-2.scope") is False
