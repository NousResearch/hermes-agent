"""Quota-wall total bound: unbounded ``rate_limited`` retry must hold the task (issue #114511).

TDD red step: ``test_quota_attempt_bound_holds_task`` and
``test_quota_elapsed_bound_holds_task`` FAIL on origin/main (the task is
requeued ``ready`` forever and ``check_respawn_guard`` never reports
``quota_exhausted``). The fix adds a total quota-attempt/elapsed bound that
holds the task (``blocked``) with a coordinator-visible ``gave_up`` event.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _exited_status(code: int) -> int:
    return code << 8


def _quota_hit(conn, tid, host, i):
    """Drive one real quota-wall reclaim for ``tid`` (mirrors the existing
    rate-limit regression test: claim -> dead pid -> sentinel exit)."""
    pid = 71000 + i
    kb.claim_task(conn, tid, claimer=f"{host}:w{i}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, consecutive_failures=? WHERE id=?",
        (pid, 0, tid),
    )
    conn.commit()
    kbd._record_worker_exit(pid, _exited_status(kb.KANBAN_RATE_LIMIT_EXIT_CODE))
    kbd.detect_crashed_workers(conn)


def test_quota_attempt_bound_holds_task(kanban_home, monkeypatch):
    """Past the total quota-attempt bound the task must be HELD (blocked),
    not requeued ready forever; the guard must report ``quota_exhausted``."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ATTEMPT_LIMIT", "3")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")

    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="quota-bound", assignee="a")
        for i in range(6):
            if kb.get_task(conn, tid).status == "blocked":
                break
            _quota_hit(conn, tid, host, i)

        task = kb.get_task(conn, tid)
        assert task.status == "blocked", (
            f"quota wall must hold the task past the attempt bound, got {task.status}"
        )
        assert task.consecutive_failures == 0
        kinds = [r[0] for r in conn.execute(
            "SELECT kind FROM task_events WHERE task_id=?", (tid,),
        ).fetchall()]
        assert "gave_up" in kinds
        payloads = [
            r[0] for r in conn.execute(
                "SELECT payload FROM task_events WHERE task_id=? AND kind='gave_up'",
                (tid,),
            ).fetchall()
        ]
        assert any("quota" in (p or "") for p in payloads)
        assert kbd.check_respawn_guard(conn, tid) == "quota_exhausted"


def test_quota_elapsed_bound_holds_task(kanban_home, monkeypatch):
    """A quota wall stretching past the elapsed window must also hold the
    task, even when the attempt count is still below its limit."""
    import time

    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ATTEMPT_LIMIT", "1000")
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ELAPSED_SECONDS", "300")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")

    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="quota-elapsed", assignee="a")
        _quota_hit(conn, tid, host, 0)
        # Age the first quota hit past the elapsed window.
        old = int(time.time()) - 600
        conn.execute(
            "UPDATE task_runs SET ended_at=? WHERE task_id=? AND outcome='rate_limited'",
            (old, tid),
        )
        conn.commit()
        _quota_hit(conn, tid, host, 1)

        task = kb.get_task(conn, tid)
        assert task.status == "blocked", (
            f"quota wall must hold the task past the elapsed bound, got {task.status}"
        )
        assert kbd.check_respawn_guard(conn, tid) == "quota_exhausted"


def test_quota_bound_not_tripped_below_limit(kanban_home, monkeypatch):
    """Below the bound nothing changes: ready requeue, cooldown guard."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")

    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="quota-ok", assignee="a")
        for i in range(3):
            _quota_hit(conn, tid, host, i)

        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"


def test_quota_hold_preserves_per_run_history(kanban_home, monkeypatch):
    """Holding the task must not destroy per-run history: every
    ``rate_limited`` run row and event survives the hold."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ATTEMPT_LIMIT", "3")

    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="quota-logs", assignee="a")
        for i in range(5):
            if kb.get_task(conn, tid).status == "blocked":
                break
            _quota_hit(conn, tid, host, i)

        assert kb.get_task(conn, tid).status == "blocked"
        runs = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id=? AND outcome='rate_limited'",
            (tid,),
        ).fetchone()[0]
        assert runs >= 3, f"per-run rate_limited rows must survive, got {runs}"
        events = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='rate_limited'",
            (tid,),
        ).fetchone()[0]
        assert events >= 3, f"per-run rate_limited events must survive, got {events}"


def test_unblock_starts_fresh_quota_epoch(kanban_home, monkeypatch):
    """A deliberate operator unblock clears the hold: the guard must allow a
    fresh probe instead of re-trapping on the pre-unblock quota history."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_QUOTA_ATTEMPT_LIMIT", "2")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "0")

    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="quota-epoch", assignee="a")
        for i in range(4):
            if kb.get_task(conn, tid).status == "blocked":
                break
            _quota_hit(conn, tid, host, i)
        assert kb.get_task(conn, tid).status == "blocked"

        assert kb.unblock_task(conn, tid) is True
        # Cooldown 0 respawns immediately; the exhausted pre-unblock history
        # must not re-trap the fresh epoch.
        assert kbd.check_respawn_guard(conn, tid) is None
