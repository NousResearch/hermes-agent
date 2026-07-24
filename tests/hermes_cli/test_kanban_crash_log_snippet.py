"""Tests for crash-log snippet extraction in detect_crashed_workers."""

from __future__ import annotations

import os

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with an empty kanban DB."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    # Disable crash grace period so tests see immediate reclaim.
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# _extract_crash_cause_from_log unit tests
# ---------------------------------------------------------------------------


class TestExtractCrashCauseFromLog:
    """Tests for ``_extract_crash_cause_from_log``."""

    def test_returns_error_line_from_log(self, kanban_home):
        """Picks the first error-prefixed line when scanning backwards."""
        task_id = "t_err_skill"
        log_path = kb.worker_log_path(task_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "Starting worker...\n"
            "Loading config...\n"
            "Error: Unknown skill(s): org-design-pipeline\n"
            "Error: Unknown skill(s): org-design-pipeline\n",
            encoding="utf-8",
        )
        result = kb._extract_crash_cause_from_log(task_id)
        assert result is not None
        assert "Unknown skill" in result
        assert result.startswith("Error:")

    def test_returns_traceback_line(self, kanban_home):
        """Matches 'Traceback' prefix."""
        task_id = "t_traceback"
        log_path = kb.worker_log_path(task_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "ok\nTraceback (most recent call last):\n  File ...\n",
            encoding="utf-8",
        )
        result = kb._extract_crash_cause_from_log(task_id)
        assert result is not None
        assert result.startswith("Traceback")

    def test_returns_exception_line(self, kanban_home):
        """Matches 'Exception' prefix."""
        task_id = "t_exc"
        log_path = kb.worker_log_path(task_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "ok\nException: something broke\n",
            encoding="utf-8",
        )
        result = kb._extract_crash_cause_from_log(task_id)
        assert result is not None
        assert result.startswith("Exception:")

    def test_fallback_last_nonempty_line(self, kanban_home):
        """Falls back to last non-empty line when no error prefix matches."""
        task_id = "t_generic"
        log_path = kb.worker_log_path(task_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            "line 1\nline 2\nsome other output\n",
            encoding="utf-8",
        )
        result = kb._extract_crash_cause_from_log(task_id)
        assert result is not None
        assert result == "some other output"

    def test_returns_none_when_no_log_file(self, kanban_home):
        """Returns None when the log file doesn't exist."""
        result = kb._extract_crash_cause_from_log("t_nonexistent")
        assert result is None

    def test_returns_none_for_empty_log(self, kanban_home):
        """Returns None when the log file is empty."""
        task_id = "t_empty"
        log_path = kb.worker_log_path(task_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        result = kb._extract_crash_cause_from_log(task_id)
        assert result is None

    def test_truncates_long_line(self, kanban_home):
        """Lines longer than 300 chars are truncated."""
        task_id = "t_long"
        log_path = kb.worker_log_path(task_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        long_line = "Error: " + "x" * 400
        log_path.write_text(f"{long_line}\n", encoding="utf-8")
        result = kb._extract_crash_cause_from_log(task_id)
        assert result is not None
        assert len(result) <= 300

    def test_uses_tail_bytes(self, kanban_home):
        """Only reads the last tail_bytes of the log."""
        task_id = "t_tail"
        log_path = kb.worker_log_path(task_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Write a large log with the error near the end.
        padding = "padding line\n" * 200  # ~1400 bytes
        log_path.write_text(
            f"{padding}Error: the real cause\n",
            encoding="utf-8",
        )
        result = kb._extract_crash_cause_from_log(task_id, tail_bytes=256)
        assert result is not None
        assert "the real cause" in result


# ---------------------------------------------------------------------------
# Integration: detect_crashed_workers surfaces log cause
# ---------------------------------------------------------------------------


class TestCrashWorkerLogSurfaced:
    """Verify that ``detect_crashed_workers`` enriches the error text with
    the crash cause extracted from the worker log file."""

    def test_crash_error_includes_log_cause(self, kanban_home, monkeypatch):
        """When a crashed worker has a log with an error line, the run error
        includes that line appended after the pid/exit-code message."""
        import hermes_cli.kanban_db as _kb

        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="crash with log", assignee="worker")
            kb.claim_task(conn, tid)
            kb._set_worker_pid(conn, tid, 98765)

            # Write a log file simulating the crash.
            log_path = kb.worker_log_path(tid)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(
                "Loading worker...\n"
                "Error: Unknown skill(s): org-design-pipeline\n",
                encoding="utf-8",
            )

            monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
            kb.detect_crashed_workers(conn)

            runs = kb.list_runs(conn, tid)
            crashed_runs = [r for r in runs if r.outcome == "crashed"]
            assert len(crashed_runs) >= 1
            err = crashed_runs[0].error
            assert err is not None
            # Should contain the original pid message AND the log cause.
            assert "98765" in err
            assert "Unknown skill" in err
            assert "→" in err  # arrow prefix for the log line
        finally:
            conn.close()

    def test_crash_error_without_log_still_works(self, kanban_home, monkeypatch):
        """When no log file exists, the error falls back to the default
        pid/exit-code message without crashing."""
        import hermes_cli.kanban_db as _kb

        conn = kb.connect()
        try:
            tid = kb.create_task(conn, title="crash no log", assignee="worker")
            kb.claim_task(conn, tid)
            kb._set_worker_pid(conn, tid, 98765)

            # No log file written.
            monkeypatch.setattr(_kb, "_pid_alive", lambda pid: False)
            kb.detect_crashed_workers(conn)

            runs = kb.list_runs(conn, tid)
            crashed_runs = [r for r in runs if r.outcome == "crashed"]
            assert len(crashed_runs) >= 1
            err = crashed_runs[0].error
            assert err is not None
            assert "98765" in err
            # No arrow when there's no log cause.
            assert "→" not in err
        finally:
            conn.close()
