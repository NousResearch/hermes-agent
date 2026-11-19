"""Regressions for the dispatcher's quota-wall hold (#quota-wall hold).

A worker that exits on a provider quota wall is requeued WITHOUT counting a failure, and the
respawn guard spaces the next attempt. A fixed cooldown alone let one card bounce off the same
weekly wall every few minutes for hours: 27+ ``rate_limited`` runs on one task, 266 board-wide.
These tests pin the two things that stop it:

* the provider's own "resets <when>" boundary, when the failed run recorded one, is honoured;
* otherwise the hold doubles per consecutive quota-wall run up to a ceiling, and the task stays
  ``ready`` so the ordinary dispatcher re-spawns it when the hold lapses.

They also pin what must NOT change: a crash or an unrelated transient error is never held as a
quota wall, and any non-rate-limited run ends the streak (so one throttle cannot inherit a
multi-hour hold).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd

RATE_LIMIT = kb.KANBAN_RATE_LIMIT_EXIT_CODE
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _exited_status(code: int) -> int:
    """Raw wait-status for a WIFEXITED child with the given exit code."""
    return code << 8


def _reset_message(hours_ahead: int = 6) -> str:
    """The provider's weekly-wall sentence, in its real shape, relative to the test clock."""
    zone = ZoneInfo("Asia/Tokyo")
    when = datetime.now(zone) + timedelta(hours=hours_ahead)
    hour12 = when.hour % 12 or 12
    meridiem = "pm" if when.hour >= 12 else "am"
    return (f"You've hit your weekly limit · resets {_MONTHS[when.month - 1]} {when.day}, "
            f"{hour12}{meridiem} (Asia/Tokyo)")


def _seed_rate_limited_run(conn, tid: str, *, claimer: str, pid: int, last_output: str = "",
                           exit_code: int = RATE_LIMIT) -> int:
    """Claim, record the worker's own result, then let the crash sweep book the dead worker."""
    kb.claim_task(conn, tid, claimer=claimer)
    run_id = kb.get_task(conn, tid).current_run_id
    kb._append_event(
        conn, tid, "worker_result",
        {"pid": pid, "claim_lock": claimer, "exit_code": exit_code, "last_output": last_output},
        run_id=run_id,
    )
    conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (pid, tid))
    conn.commit()
    kbd._record_worker_exit(pid, _exited_status(exit_code))
    kbd.detect_crashed_workers(conn)
    return run_id


def _run_metadata(conn, run_id: int) -> dict:
    row = conn.execute("SELECT metadata FROM task_runs WHERE id=?", (run_id,)).fetchone()
    return json.loads(row["metadata"]) if row and row["metadata"] else {}


def _seed_history(conn, tid: str, outcomes: list[str], *, ended_at: int) -> None:
    """Insert closed runs (oldest first) so the newest one is ``outcomes[-1]``."""
    with kb.write_txn(conn):
        for index, outcome in enumerate(outcomes):
            conn.execute(
                "INSERT INTO task_runs (task_id, profile, status, outcome, started_at, ended_at) "
                "VALUES (?, 'a', ?, ?, ?, ?)",
                (tid, outcome, outcome, ended_at - (len(outcomes) - 1 - index) * 10,
                 ended_at - (len(outcomes) - 1 - index) * 10),
            )


def test_hold_honours_provider_reset_instant_instead_of_the_base_cooldown(
    kanban_home, monkeypatch,
):
    """A run that recorded the window's reopen instant is held to it, not re-probed at 5 minutes."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: None)

    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="weekly wall", assignee="sophia")
        run_id = _seed_rate_limited_run(
            conn, tid, claimer=f"{host}:w", pid=71001, last_output=_reset_message(hours_ahead=6),
        )

        task = kb.get_task(conn, tid)
        assert task.status == "ready" and task.consecutive_failures == 0
        resets_at = _run_metadata(conn, run_id)["quota_resets_at"]
        assert resets_at > time.time() + 4 * 3600, "the provider's boundary must be persisted"

        # Well past the 5-minute cooldown, still before the boundary: no re-spawn.
        monkeypatch.setattr(kb.time, "time", lambda: resets_at - 3600)
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"
        held = kbd.dispatch_once(conn, dry_run=True)
        assert tid not in [task_id for task_id, *_ in held.spawned]
        assert dict(held.respawn_guarded).get(tid) == "rate_limit_cooldown"

        # The boundary passed: the ordinary dispatcher takes it again.
        monkeypatch.setattr(kb.time, "time", lambda: resets_at + 1)
        assert kbd.check_respawn_guard(conn, tid) is None
        released = kbd.dispatch_once(conn, dry_run=True)
        assert tid in [task_id for task_id, *_ in released.spawned]


def test_hold_without_a_reset_hint_doubles_up_to_the_ceiling(kanban_home, monkeypatch):
    """No reset instant — the wait widens per consecutive quota-wall run and then stops."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_MAX_COOLDOWN_SECONDS", "2400")
    ended_at = int(time.time()) - 1

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="no hint", assignee="a")
        _seed_history(conn, tid, ["rate_limited"] * 4, ended_at=ended_at)
        conn.execute(
            "UPDATE tasks SET last_failure_error=? WHERE id=?",
            ("pid 1 exited rate-limited (quota wall) — requeued without counting a failure", tid),
        )
        conn.commit()

        # streak 4 → 300 * 2^3 = 2400 (the ceiling).
        monkeypatch.setattr(kb.time, "time", lambda: ended_at + 300)
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"
        monkeypatch.setattr(kb.time, "time", lambda: ended_at + 2399)
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"
        monkeypatch.setattr(kb.time, "time", lambda: ended_at + 2401)
        assert kbd.check_respawn_guard(conn, tid) is None

        # A single wall still waits only the base cooldown: the streak is run history, not a flag.
        with kb.write_txn(conn):
            conn.execute("DELETE FROM task_runs WHERE task_id=?", (tid,))
        _seed_history(conn, tid, ["rate_limited"], ended_at=ended_at)
        monkeypatch.setattr(kb.time, "time", lambda: ended_at + 301)
        assert kbd.check_respawn_guard(conn, tid) is None


def test_hold_never_grows_without_a_ceiling_breach_across_many_walls(kanban_home, monkeypatch):
    """Twelve quota walls hold at the ceiling instead of re-probing at the base cooldown."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_MAX_COOLDOWN_SECONDS", "21600")
    ended_at = int(time.time()) - 1

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="many walls", assignee="a")
        _seed_history(conn, tid, ["rate_limited"] * 12, ended_at=ended_at)
        conn.execute(
            "UPDATE tasks SET last_failure_error=? WHERE id=?",
            ("pid 1 exited rate-limited (quota wall) — requeued without counting a failure", tid),
        )
        conn.commit()

        monkeypatch.setattr(kb.time, "time", lambda: ended_at + 5 * 3600)
        assert kbd.check_respawn_guard(conn, tid) == "rate_limit_cooldown"
        monkeypatch.setattr(kb.time, "time", lambda: ended_at + 21601)
        assert kbd.check_respawn_guard(conn, tid) is None


@pytest.mark.parametrize("prior_outcome", ["completed", "crashed", "scheduled"])
def test_any_other_run_outcome_ends_the_streak(kanban_home, monkeypatch, prior_outcome):
    """A recovered / crashed / parked run resets the hold: history alone never parks a card."""
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")
    ended_at = int(time.time()) - 1

    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="recovered", assignee="a")
        _seed_history(conn, tid, ["rate_limited"] * 6 + [prior_outcome], ended_at=ended_at)
        conn.commit()

        monkeypatch.setattr(kb.time, "time", lambda: ended_at + 301)
        assert kbd.check_respawn_guard(conn, tid) != "rate_limit_cooldown"


def test_plain_crash_is_not_treated_as_a_quota_wall(kanban_home, monkeypatch):
    """An ordinary crash never picks up the quota hold, even when its output mentions a reset."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")

    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="crash", assignee="a")
        run_id = _seed_rate_limited_run(
            conn, tid, claimer=f"{host}:w", pid=71002,
            last_output=f"docker compose failed; the flag resets {_MONTHS[0]} 1, 3pm (Asia/Tokyo)",
            exit_code=1,
        )

        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert "quota_resets_at" not in _run_metadata(conn, run_id)
        # Booked as a crash, so the breaker counts it — not a free retry.
        assert task.consecutive_failures >= 1
        assert kbd.check_respawn_guard(conn, tid) != "rate_limit_cooldown"


def test_reset_hint_is_read_only_from_the_failed_rate_limited_run(kanban_home, monkeypatch):
    """Evidence from another attempt does not park the current one."""
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setenv("HERMES_KANBAN_RATE_LIMIT_COOLDOWN_SECONDS", "300")

    with kbc.connect() as conn:
        host = kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="stale hint", assignee="a")
        _seed_rate_limited_run(
            conn, tid, claimer=f"{host}:w0", pid=71003, last_output=_reset_message(hours_ahead=6),
        )
        first = conn.execute(
            "SELECT id FROM task_runs WHERE task_id=? ORDER BY id DESC LIMIT 1", (tid,)).fetchone()
        assert "quota_resets_at" in _run_metadata(conn, first["id"])

        # The next attempt records no hint (e.g. a plain 429 without a window statement).
        run_id = _seed_rate_limited_run(
            conn, tid, claimer=f"{host}:w1", pid=71004,
            last_output="Provider said: HTTP 429: too many requests",
        )
        assert "quota_resets_at" not in _run_metadata(conn, run_id)
