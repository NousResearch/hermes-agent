"""PR1 — detect_crashed_workers must short-circuit on Windows.

On Windows the recorded ``worker_pid`` is the distlib console-script launcher
shim, which exits within seconds of spawning the real (orphaned) worker child.
PID-liveness then reports a live worker dead and the dispatcher reclaims +
respawns it every tick — a respawn storm where ``spawned=N crashed=N`` repeats
for the same tasks and nothing progresses. The fix returns ``[]`` early when
``os.name == 'nt'`` and lets claim-TTL reclamation handle genuine crashes;
POSIX behavior is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB.

    Mirrors the fixture in test_kanban_db.py so the crash-detection tests build
    their conn/DB the same way the rest of the kanban suite does.
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _seed_running_task_with_dead_pid(conn, monkeypatch):
    """Create a ``running`` task whose recorded worker_pid looks dead.

    Mirrors the setup the existing detect_crashed_workers tests use: a running
    task with a claim_lock, started long enough ago to be past the crash grace
    period, and ``_pid_alive`` stubbed False so PID-based detection *would*
    reclaim it on POSIX.
    """
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    # Push "now" well past the default crash grace window so the grace guard
    # never suppresses reclaim — we want to isolate the os.name gate.
    now = 5_000_000.0
    monkeypatch.setattr(_kb.time, "time", lambda: now)

    host = _kb._claimer_id().split(":", 1)[0]
    tid = kb.create_task(conn, title="windows pid task", assignee="a")
    conn.execute(
        "UPDATE tasks SET status='running', worker_pid=?, "
        "claim_lock=?, started_at=? WHERE id=?",
        (99999, f"{host}:w", int(now) - 3600, tid),
    )
    conn.commit()
    return tid


def test_detect_crashed_workers_noop_on_windows(kanban_home, monkeypatch):
    """On Windows the launcher-shim PID is not a reliable liveness signal, so
    PID-based crash detection is skipped entirely and the running task is left
    in place (claim-TTL reclamation handles genuine crashes)."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb.os, "name", "nt")

    with kb.connect() as conn:
        tid = _seed_running_task_with_dead_pid(conn, monkeypatch)

        # Guard short-circuits: nothing reclaimed despite the dead PID.
        assert kb.detect_crashed_workers(conn) == []

        # Task is still 'running' — NOT bounced back to 'ready' (that would be
        # the respawn-storm bug). Reclaim is deferred to claim-TTL.
        assert kb.get_task(conn, tid).status == "running"


def test_detect_crashed_workers_guard_skips_liveness_on_windows(
    kanban_home, monkeypatch,
):
    """On Windows the guard returns before any PID-liveness probe, so
    ``_pid_alive`` is never consulted for the running task. This is the
    behavioral signature of the early return: PID-based detection is fully
    bypassed (claim-TTL reclaims genuine crashes instead)."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb.os, "name", "nt")

    probed: list[int] = []
    with kb.connect() as conn:
        _seed_running_task_with_dead_pid(conn, monkeypatch)
        monkeypatch.setattr(_kb, "_pid_alive", lambda pid: probed.append(pid))

        assert kb.detect_crashed_workers(conn) == []

    assert probed == [], "Windows guard must short-circuit before _pid_alive"


@pytest.mark.skipif(
    sys.platform == "win32",
    reason=(
        "Cannot simulate POSIX path semantics on a Windows host — forcing "
        "os.name='posix' makes pathlib instantiate PosixPath, which Python "
        "refuses to construct on Windows. The reclaim path is exercised on "
        "POSIX CI by the existing detect_crashed_workers_* tests; this control "
        "runs there to confirm the os.name=='nt' guard is the only suppressor."
    ),
)
def test_detect_crashed_workers_reclaims_when_not_windows(
    kanban_home, monkeypatch,
):
    """Control (POSIX runner only): the same dead-PID running task that the
    Windows guard leaves alone IS reclaimed when ``os.name != 'nt'`` — the
    early return is the sole reason for the Windows no-op, and POSIX behavior is
    unchanged by the patch."""
    import hermes_cli.kanban_db as _kb

    # On a POSIX host os.name is already 'posix'; assert that to be explicit.
    assert _kb.os.name != "nt"

    with kb.connect() as conn:
        tid = _seed_running_task_with_dead_pid(conn, monkeypatch)

        crashed = kb.detect_crashed_workers(conn)

        assert tid in crashed
        assert kb.get_task(conn, tid).status == "ready"
