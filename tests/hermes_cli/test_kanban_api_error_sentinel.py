"""Tests for the Kanban API/auth-error sentinel (exit 76, #80456).

A worker that bails before its first successful tool-call turn on an API/auth/
model error (HTTP 401/403/503, "Model is disabled", provider-side transient)
must be classified ``api_error`` — requeued to ``ready`` WITHOUT counting a
protocol violation and WITHOUT consuming the violation streak — and the card's
``last_failure_error`` must carry the REAL provider message read from the
worker log, not the generic protocol-violation text.

Mirrors the rate-limit sentinel (exit 75) test in this file. The wait-status
macros (``os.WIFEXITED`` / ``os.WEXITSTATUS``) are shimmed so the suite runs
identically on POSIX (where the dispatcher reaps) and on Windows (where the
macros are absent from the ``os`` module but the classifier logic is the same).
"""

from __future__ import annotations

import os

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(__import__("pathlib").Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def wait_macros(monkeypatch):
    """Provide POSIX wait-status macros on every platform.

    ``_classify_worker_exit`` calls ``os.WIFEXITED`` / ``os.WEXITSTATUS`` /
    ``os.WIFSIGNALED`` / ``os.WTERMSIG`` directly. These exist on POSIX but
    not on Windows; shim them so the classifier is exercised identically
    regardless of the host running the suite. The raw wait-status encoding is
    the standard POSIX one (exit code in the high byte).
    """
    monkeypatch.setattr(
        os, "WIFEXITED", lambda status: (status & 0xFF) == 0, raising=False
    )
    monkeypatch.setattr(
        os, "WEXITSTATUS", lambda status: (status >> 8) & 0xFF, raising=False
    )
    monkeypatch.setattr(
        os, "WIFSIGNALED", lambda status: (status & 0x7F) != 0, raising=False
    )
    monkeypatch.setattr(
        os, "WTERMSIG", lambda status: status & 0x7F, raising=False
    )


def _exited_status(code: int) -> int:
    """Raw POSIX wait-status for a WIFEXITED child with the given exit code."""
    return code << 8


# ---------------------------------------------------------------------------
# Classification + requeue invariants
# ---------------------------------------------------------------------------


def test_api_error_exit_requeues_without_violation(kanban_home, monkeypatch, wait_macros):
    """Exit 76 → classified ``api_error``, task requeued to ``ready``, NO
    protocol violation recorded, NO failure counted, violation streak not
    incremented, and ``last_failure_error`` carries the real API error text
    read from the worker log."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="api-err", assignee="a")

        # Seed the worker log with a realistic 401 bail line so the reaper's
        # log reader surfaces the real message in last_failure_error.
        log_dir = _kb.worker_logs_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{tid}.log").write_text(
            "starting worker...\n"
            "AuthenticationError [HTTP 401] model is disabled\n"
            "switching to fallback provider\n",
            encoding="utf-8",
        )

        pid = 70000
        kb.claim_task(conn, tid, claimer=f"{host}:w0")
        conn.execute(
            "UPDATE tasks SET worker_pid=?, consecutive_failures=? WHERE id=?",
            (pid, 0, tid),
        )
        conn.commit()
        _kb._record_worker_exit(
            pid, _exited_status(_kb.KANBAN_API_ERROR_EXIT_CODE)
        )

        crashed = kb.detect_crashed_workers(conn)

        # api_error requeues are NOT crashes.
        assert tid not in crashed
        api_err = getattr(_kb.detect_crashed_workers, "_last_api_errors", [])
        assert tid in api_err

        task = kb.get_task(conn, tid)
        assert task.status == "ready", (
            f"should requeue ready, got {task.status}"
        )
        # No failure counted — the worker never ran the task.
        assert task.consecutive_failures == 0

        # The real provider error is surfaced, not the generic protocol text.
        assert task.last_failure_error
        assert "HTTP 401" in task.last_failure_error or "Model is disabled" in task.last_failure_error
        assert "protocol violation" not in task.last_failure_error.lower()

        # Run outcome recorded as api_error (distinct from crashed /
        # rate_limited), and a distinct api_error event was appended.
        outcomes = [
            r["outcome"] for r in conn.execute(
                "SELECT outcome FROM task_runs WHERE task_id=?", (tid,),
            ).fetchall()
        ]
        assert "api_error" in outcomes
        assert "crashed" not in outcomes
        assert "rate_limited" not in outcomes

        events = [
            r["kind"] for r in conn.execute(
                "SELECT kind FROM task_events WHERE task_id=?", (tid,),
            ).fetchall()
        ]
        assert "api_error" in events
        assert "protocol_violation" not in events

        # No protocol_violation marker stamped on the run metadata.
        meta_rows = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id=?", (tid,),
        ).fetchall()
        for r in meta_rows:
            import json
            md = json.loads(r["metadata"] or "{}")
            assert not md.get("protocol_violation"), (
                f"api_error run must not carry the protocol_violation marker: {md}"
            )


def test_api_error_does_not_consume_violation_streak(
    kanban_home, monkeypatch, wait_macros
):
    """``_protocol_violation_streak`` ignores ``api_error`` runs, exactly as
    it ignores ``rate_limited`` — a pre-tool provider bail says nothing about
    the task and must neither consume nor extend the violation budget."""
    import json

    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="streak", assignee="a")

        # Seed two api_error runs by driving the reaper twice. Neither should
        # count as a violation.
        for i in range(2):
            pid = 71000 + i
            kb.claim_task(conn, tid, claimer=f"{host}:w{i}")
            conn.execute(
                "UPDATE tasks SET worker_pid=?, consecutive_failures=0 "
                "WHERE id=?",
                (pid, tid),
            )
            conn.commit()
            _kb._record_worker_exit(
                pid, _exited_status(_kb.KANBAN_API_ERROR_EXIT_CODE)
            )
            kb.detect_crashed_workers(conn)

        # Streak must be 0 — api_error runs are neutral.
        assert _kb._protocol_violation_streak(conn, tid) == 0


def test_api_error_log_missing_falls_back_to_clear_text(
    kanban_home, monkeypatch, wait_macros
):
    """When the worker log is missing/unreadable, ``last_failure_error`` falls
    back to a clear API/auth-error message, the task is still requeued, and
    still no violation is recorded."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    # No log file written — the reader must return None and the reaper falls
    # back to the sentinel text.

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="no-log", assignee="a")
        pid = 72000
        kb.claim_task(conn, tid, claimer=f"{host}:w0")
        conn.execute(
            "UPDATE tasks SET worker_pid=?, consecutive_failures=0 WHERE id=?",
            (pid, tid),
        )
        conn.commit()
        _kb._record_worker_exit(
            pid, _exited_status(_kb.KANBAN_API_ERROR_EXIT_CODE)
        )

        crashed = kb.detect_crashed_workers(conn)
        assert tid not in crashed
        assert tid in getattr(_kb.detect_crashed_workers, "_last_api_errors", [])

        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 0
        assert task.last_failure_error
        # Fallback text names the API/auth error class, not the protocol text.
        assert "API/auth error" in task.last_failure_error or "api/auth" in task.last_failure_error.lower() or "exit 76" in task.last_failure_error
        assert "protocol violation" not in task.last_failure_error.lower()


# ---------------------------------------------------------------------------
# Non-regression: clean_exit (real protocol violation) and rate_limited
# ---------------------------------------------------------------------------


def test_clean_exit_still_protocol_violation(kanban_home, monkeypatch, wait_macros):
    """A worker exiting rc=0 WITHOUT the 76 sentinel (a real protocol
    violation — it ran turns then exited without a terminal kanban call) is
    STILL classified ``clean_exit`` → ``protocol_violation`` as today."""
    import json

    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="clean", assignee="a")
        pid = 73000
        kb.claim_task(conn, tid, claimer=f"{host}:w0")
        conn.execute(
            "UPDATE tasks SET worker_pid=?, consecutive_failures=0 WHERE id=?",
            (pid, tid),
        )
        conn.commit()
        # rc=0 → clean_exit path, NOT api_error.
        _kb._record_worker_exit(pid, _exited_status(0))

        kb.detect_crashed_workers(conn)

        task = kb.get_task(conn, tid)
        # A single violation is below the streak budget (3), so the task is
        # back at ready with the violation stamped.
        assert task.status == "ready"
        assert task.last_failure_error
        assert "protocol violation" in task.last_failure_error.lower()

        # Streak counts this run.
        assert _kb._protocol_violation_streak(conn, tid) == 1

        # The run carries the protocol_violation marker.
        meta_rows = conn.execute(
            "SELECT metadata FROM task_runs WHERE task_id=?", (tid,),
        ).fetchall()
        assert any(
            json.loads(r["metadata"] or "{}").get("protocol_violation")
            for r in meta_rows
        )


def test_rate_limited_still_neutral_and_requeued(
    kanban_home, monkeypatch, wait_macros
):
    """Exit 75 is still classified ``rate_limited`` and requeued without a
    failure — the existing sentinel path is unchanged by the new exit-76
    branch."""
    import hermes_cli.kanban_db as _kb

    monkeypatch.setattr(_kb, "_pid_alive", lambda _pid: False)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")

    with kb.connect() as conn:
        host = _kb._claimer_id().split(":", 1)[0]
        tid = kb.create_task(conn, title="rl", assignee="a")
        pid = 74000
        kb.claim_task(conn, tid, claimer=f"{host}:w0")
        conn.execute(
            "UPDATE tasks SET worker_pid=?, consecutive_failures=0 WHERE id=?",
            (pid, tid),
        )
        conn.commit()
        _kb._record_worker_exit(
            pid, _exited_status(_kb.KANBAN_RATE_LIMIT_EXIT_CODE)
        )

        crashed = kb.detect_crashed_workers(conn)
        assert tid not in crashed
        assert tid in getattr(_kb.detect_crashed_workers, "_last_rate_limited", [])

        task = kb.get_task(conn, tid)
        assert task.status == "ready"
        assert task.consecutive_failures == 0

        outcomes = [
            r["outcome"] for r in conn.execute(
                "SELECT outcome FROM task_runs WHERE task_id=?", (tid,),
            ).fetchall()
        ]
        assert "rate_limited" in outcomes
        assert "api_error" not in outcomes


# ---------------------------------------------------------------------------
# Classifier unit check
# ---------------------------------------------------------------------------


def test_classify_worker_exit_codes(wait_macros):
    """``_classify_worker_exit`` maps the three sentinels distinctly."""
    import hermes_cli.kanban_db as _kb

    _kb._recent_worker_exits.clear()
    _kb._record_worker_exit(80001, _exited_status(0))
    _kb._record_worker_exit(80002, _exited_status(_kb.KANBAN_RATE_LIMIT_EXIT_CODE))
    _kb._record_worker_exit(80003, _exited_status(_kb.KANBAN_API_ERROR_EXIT_CODE))
    _kb._record_worker_exit(80004, _exited_status(1))

    assert _kb._classify_worker_exit(80001) == ("clean_exit", 0)
    assert _kb._classify_worker_exit(80002) == ("rate_limited", _kb.KANBAN_RATE_LIMIT_EXIT_CODE)
    assert _kb._classify_worker_exit(80003) == ("api_error", _kb.KANBAN_API_ERROR_EXIT_CODE)
    assert _kb._classify_worker_exit(80004) == ("nonzero_exit", 1)
