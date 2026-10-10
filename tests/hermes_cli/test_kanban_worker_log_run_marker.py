"""A dead worker is explained by ITS OWN run, never by the run before it.

The per-task worker log is append-mode across re-runs (a re-run on unblock must not
destroy the previous attempt's evidence). That made every reap-side read of "the tail"
ambiguous: a run that died before writing anything was reported with the PREVIOUS run's
output, and — worse — its missing exit trailer was read as the previous run's, booking
the crash as a clean "worker exited rc=0 without calling kanban_complete" protocol
violation. A workspace file that shadowed ``inspect`` bricked eight consecutive runs on
one card that way; every death looked like a paperwork slip.

``_default_spawn`` now stamps a start marker into the log before the child inherits the
descriptor, and both readers take only the bytes after the last marker.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.quiet_single_query import KANBAN_WORKER_EXIT_TRAILER

PREVIOUS_RUN = (
    "the previous attempt said something\n\n"
    "Resume this session with:\n  hermes --resume x\n\n"
    f"{KANBAN_WORKER_EXIT_TRAILER}0\n"
)


@pytest.fixture
def kanban_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    monkeypatch.setattr(kb, "_pid_alive", lambda _pid: False)
    kbd._recent_worker_exits.clear()
    kb.init_db()
    return home


def _write_log(tid: str, body: str) -> Path:
    log = kb.worker_log_path(tid)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(body, encoding="utf-8")
    return log


def _marker(run_id: int) -> str:
    return f"{kbd._WORKER_RUN_MARKER}{run_id} @ 2026-10-02T00:00:00+0000 ===\n"


def _claim_dead(conn, tid: str, pid: int) -> None:
    """Claim ``tid`` for a worker that already exited and left its log behind."""
    host = kb._claimer_id().split(":", 1)[0]
    kb.claim_task(conn, tid, claimer=f"{host}:w{pid}")
    conn.execute(
        "UPDATE tasks SET worker_pid=?, worker_started_at=NULL, started_at=? WHERE id=?",
        (pid, int(time.time()) - 120, tid),
    )
    conn.commit()


def test_the_last_marker_frames_what_the_reader_sees(kanban_home):
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _write_log(tid, PREVIOUS_RUN + _marker(11) + "run eleven spoke\n")
        assert kbd._worker_final_output(tid) == "run eleven spoke"
        assert kbd._worker_current_run_text(tid) == "run eleven spoke\n"


def test_a_log_without_a_marker_still_reads_its_tail(kanban_home):
    """Logs written before markers existed keep their diagnostic value."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="t", assignee="a")
        _write_log(tid, "older dispatcher wrote this\n")
        assert kbd._worker_current_run_text(tid) is None
        assert kbd._worker_final_output(tid) == "older dispatcher wrote this"
        assert kbd._worker_log_exit_code(tid) is None


def test_a_silent_crash_is_not_booked_as_the_previous_runs_clean_exit(kanban_home):
    """The whole failure mode: the stale trailer said rc=0, so the death was read as
    'the worker exited cleanly without calling kanban_complete' (#46593)."""
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="poison", assignee="a")
        _claim_dead(conn, tid, 70011)
        _write_log(tid, PREVIOUS_RUN + _marker(12))

        assert kbd._worker_log_exit_code(tid) is None
        kbd.detect_crashed_workers(conn)

        runs = conn.execute(
            "SELECT outcome, error FROM task_runs WHERE task_id=? ORDER BY id DESC",
            (tid,),
        ).fetchall()
        assert runs, "the reap must have booked the death"
        assert runs[0]["outcome"] == "crashed", runs[0]["outcome"]
        assert "wrote no output" in (runs[0]["error"] or "")
        assert "previous attempt said something" not in (runs[0]["error"] or "")
