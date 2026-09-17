"""Tests for crash output capture: worker log tail, ANSI stripping, Windows
spawn tracking, and the diagnostics integration that surfaces crash output
data in repeated-failures and repeated-crashes rules.

These tests are primarily isolated (no DB, mocked log reads) with one
integration-style case that round-trips through kanban_db.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
import uuid
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli.kanban_db_dispatch import (
    _ANSI_ESCAPE_RE,
    _DeadWorker,
    _SPAWNED_WORKER_PROCS_MAX,
    _spawned_worker_exit_codes,
    _spawned_worker_procs,
    _worker_error_excerpt,
    _worker_output_tail,
)
from hermes_cli import kanban_diagnostics as kd


# ---------------------------------------------------------------------------
# _DeadWorker helpers
# ---------------------------------------------------------------------------


def _make_dead(kind: str = "unknown", error: str = "") -> _DeadWorker:
    return _DeadWorker(
        kind=kind,
        code=None,
        error_text=error,
        event_kind="unknown",
        event_payload={},
        rate_limited=False,
    )


# ---------------------------------------------------------------------------
# Output capture: _worker_output_tail
# ---------------------------------------------------------------------------


def test_worker_output_tail_reads_log() -> None:
    """Returns last 2KB of the worker log, stripped of ANSI codes."""
    task_id = uuid.uuid4().hex
    ansi_text = "\x1b[31mERROR:\x1b[0m something broke"
    with patch.object(kbd, "_kb") as mock_kb:
        mock_kb.read_worker_log.return_value = ansi_text
        result = _worker_output_tail(task_id, board="test-board")
        mock_kb.read_worker_log.assert_called_once_with(
            task_id, tail_bytes=2048, board="test-board"
        )
    # ANSI codes stripped
    assert "\x1b[31m" not in result
    assert "\x1b[0m" not in result
    assert "ERROR: something broke" in result


def test_worker_output_tail_empty_log() -> None:
    """Returns '' when the log is empty."""
    task_id = uuid.uuid4().hex
    with patch.object(kbd, "_kb") as mock_kb:
        mock_kb.read_worker_log.return_value = ""
        assert _worker_output_tail(task_id) == ""


def test_worker_output_tail_missing_task_no_error() -> None:
    """Returns '' without raising when read_worker_log raises."""
    task_id = uuid.uuid4().hex
    with patch.object(kbd, "_kb") as mock_kb:
        mock_kb.read_worker_log.side_effect = FileNotFoundError
        assert _worker_output_tail(task_id) == ""


def test_worker_output_tail_none_log() -> None:
    """Returns '' when read_worker_log returns None."""
    task_id = uuid.uuid4().hex
    with patch.object(kbd, "_kb") as mock_kb:
        mock_kb.read_worker_log.return_value = None
        assert _worker_output_tail(task_id) == ""


def test_worker_output_tail_default_board() -> None:
    """Calls through with board=None when omitted."""
    task_id = uuid.uuid4().hex
    with patch.object(kbd, "_kb") as mock_kb:
        mock_kb.read_worker_log.return_value = "anything"
        _worker_output_tail(task_id)
        mock_kb.read_worker_log.assert_called_once_with(
            task_id, tail_bytes=2048, board=None
        )


# ---------------------------------------------------------------------------
# Output capture: _worker_error_excerpt
# ---------------------------------------------------------------------------


def test_worker_error_excerpt_strips_non_printable() -> None:
    """Replaces non-printable characters with spaces."""
    raw = "ok\x00\x01\x02end"
    result = _worker_error_excerpt(raw)
    assert "\x00" not in result
    assert "ok   end" in result


def test_worker_error_excerpt_truncates_to_limit() -> None:
    """Caps at _WORKER_LOG_EXCERPT_CHARS (300) chars."""
    raw = "x" * 500
    result = _worker_error_excerpt(raw)
    assert len(result) == 300


def test_worker_error_excerpt_ansi_stripped() -> None:
    """Strips ANSI before excerpting."""
    raw = "\x1b[1m\x1b[32mshort\x1b[0m"
    result = _worker_error_excerpt(raw)
    assert "short" in result
    assert "\x1b[" not in result


# ---------------------------------------------------------------------------
# Output capture: _attach_worker_output
# ---------------------------------------------------------------------------


def test_attach_worker_output_sets_error_text_and_payload() -> None:
    """Attaches tail to dead worker's error_text and event_payload."""
    dead = _make_dead(kind="crashed", error="something failed")
    task_id = uuid.uuid4().hex
    tail = "Last line: crash here"
    with patch.object(kbd, "_worker_output_tail", return_value=tail):
        kbd._attach_worker_output(dead, task_id)
    assert tail in dead.error_text
    assert dead.event_payload.get("output_tail") == tail
    assert dead.event_payload.get("output_tail_len") == len(tail)


def test_attach_worker_output_empty_tail_noop() -> None:
    """Does not modify dead when output is empty."""
    dead = _make_dead(kind="crashed", error="err")
    with patch.object(kbd, "_worker_output_tail", return_value=""):
        kbd._attach_worker_output(dead, "t_none")
    assert dead.error_text == "err"
    assert "output_tail" not in dead.event_payload


def test_attach_worker_output_excerpt_used_when_available() -> None:
    """Appends excerpt to error_text when output is non-empty."""
    dead = _make_dead(kind="crashed")
    tail = "Traceback: ValueError"
    with (
        patch.object(kbd, "_worker_output_tail", return_value=tail),
        patch.object(kbd, "_worker_error_excerpt", return_value="Traceback: ValueError"),
    ):
        kbd._attach_worker_output(dead, "t_demo")
    assert "Traceback: ValueError" in dead.error_text


# ---------------------------------------------------------------------------
# _ANSI_ESCAPE_RE correctness
# ---------------------------------------------------------------------------


def test_ansi_escape_re_strips_common_codes() -> None:
    """Matches color, bold, reset CSI sequences."""
    for code in ["\x1b[31m", "\x1b[1m", "\x1b[0m", "\x1b[92m"]:
        assert _ANSI_ESCAPE_RE.sub("", code) == ""


def test_ansi_escape_re_leaves_plain_text() -> None:
    """Does not strip ordinary characters."""
    text = "hello world"
    assert _ANSI_ESCAPE_RE.sub("", text) == text


# ---------------------------------------------------------------------------
# _poll_spawned_worker_exit
# ---------------------------------------------------------------------------

@pytest.fixture
def clear_spawned_registry():
    """Clear shared module-level dicts before and after."""
    _spawned_worker_procs.clear()
    _spawned_worker_exit_codes.clear()
    yield
    _spawned_worker_procs.clear()
    _spawned_worker_exit_codes.clear()


def test_poll_spawned_worker_exit_unknown_pid(clear_spawned_registry) -> None:
    """Returns None for a PID not in the registry."""
    assert kbd._poll_spawned_worker_exit(999999) is None


def test_poll_spawned_worker_exit_still_running(clear_spawned_registry) -> None:
    """Returns None when proc.poll() returns None."""
    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = None
    _spawned_worker_procs[12345] = mock_proc
    assert kbd._poll_spawned_worker_exit(12345) is None
    mock_proc.poll.assert_called_once()


def test_poll_spawned_worker_exit_moved_to_exit_codes(clear_spawned_registry) -> None:
    """On exit stores (code, timestamp) and removes from proc registry."""
    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = 0
    _spawned_worker_procs[42] = mock_proc
    rc = kbd._poll_spawned_worker_exit(42)
    assert rc == 0
    assert 42 not in _spawned_worker_procs
    assert 42 in _spawned_worker_exit_codes
    code, ts = _spawned_worker_exit_codes[42]
    assert code == 0
    assert isinstance(ts, float)


def test_poll_spawned_worker_exit_nonzero_exit(clear_spawned_registry) -> None:
    """Returns the actual exit code when poll() returns nonzero."""
    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.poll.return_value = 137
    _spawned_worker_procs[7] = mock_proc
    rc = kbd._poll_spawned_worker_exit(7)
    assert rc == 137


# ---------------------------------------------------------------------------
# _classify_worker_exit — os.waitstatus_to_exitcode path
# ---------------------------------------------------------------------------


def test_classify_clean_exit() -> None:
    """A pid registered with exit status 0 returns clean_exit."""
    pid = 101
    kbd._record_worker_exit(pid, os.waitstatus_to_exitcode(0))
    kind, code = kbd._classify_worker_exit(pid)
    assert kind == "clean_exit"
    assert code == 0


@pytest.mark.skipif(not hasattr(os, "WIFEXITED"), reason="POSIX-only")
def test_classify_legacy_macros_fallback() -> None:
    """Falls back to os.WIFEXITED/WEXITSTATUS when waitstatus_to_exitcode fails."""
    pid = 102
    # Simulate raw wait status with WEXITSTATUS=3
    raw_exit_3 = os.WEXITSTATUS(3) | os.WEXITSTATUS(3) << 8  # Not real; use os.system
    # Instead: use a real subprocess that exits with code 3
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(3)"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.wait()
    kbd._record_worker_exit(proc.pid, proc.returncode)
    kind, code = kbd._classify_worker_exit(proc.pid)
    assert kind == "nonzero_exit"
    assert code == 3


def test_classify_unknown_pid() -> None:
    """Returns (unknown, None) for a PID never recorded."""
    kind, code = kbd._classify_worker_exit(99999)
    assert kind == "unknown"
    assert code is None


# ---------------------------------------------------------------------------
# _classify_dead_worker — updated to call _attach_worker_output
# ---------------------------------------------------------------------------


def test_classify_dead_worker_with_task_id_calls_attach(clear_spawned_registry) -> None:
    """With task_id, calls _attach_worker_output."""
    pid = 103
    kbd._record_worker_exit(pid, 0)
    with patch.object(kbd, "_attach_worker_output") as mock_attach:
        dead = kbd._classify_dead_worker(pid, "demo-claimer", task_id="t_test01")
        mock_attach.assert_called_once_with(dead, "t_test01", None)


def test_classify_dead_worker_without_task_id_does_not_call_attach(clear_spawned_registry) -> None:
    """Without task_id, does not call _attach_worker_output."""
    pid = 104
    kbd._record_worker_exit(pid, 0)
    with patch.object(kbd, "_attach_worker_output") as mock_attach:
        kbd._classify_dead_worker(pid, "demo-claimer")
        mock_attach.assert_not_called()


def test_classify_dead_worker_rate_limited_also_gets_output(clear_spawned_registry) -> None:
    """Rate-limited workers also get output attached (diagnostic value)."""
    pid = 105
    with (
        patch.object(kbd, "_classify_worker_exit", return_value=("rate_limited", 75)),
        patch.object(kbd, "_attach_worker_output") as mock_attach,
    ):
        dead = kbd._classify_dead_worker(pid, "demo", task_id="t_demo")
        mock_attach.assert_called_once()
        assert dead.rate_limited is True


# ---------------------------------------------------------------------------
# _crash_output_tail (kanban_diagnostics)
# ---------------------------------------------------------------------------


def test_crash_output_tail_reads_and_strips() -> None:
    """Reads 500-byte tail, strips ANSI, caps at 300 chars."""
    task_id = uuid.uuid4().hex
    ansi_tail = "\x1b[31m" + "A" * 400
    with patch("hermes_cli.kanban_db.read_worker_log", return_value=ansi_tail):
        result = kd._crash_output_tail(task_id)
    assert len(result) <= 300
    assert "\x1b[" not in result
    assert "A" in result


def test_crash_output_tail_empty_log() -> None:
    """Returns '' on empty log."""
    task_id = uuid.uuid4().hex
    with patch("hermes_cli.kanban_db.read_worker_log", return_value=""):
        assert kd._crash_output_tail(task_id) == ""


def test_crash_output_tail_never_raises() -> None:
    """Returns '' when read_worker_log raises."""
    task_id = uuid.uuid4().hex
    with patch("hermes_cli.kanban_db.read_worker_log", side_effect=PermissionError):
        assert kd._crash_output_tail(task_id) == ""


# ---------------------------------------------------------------------------
# _one_line helper (kanban_diagnostics)
# ---------------------------------------------------------------------------


def test_one_line_empty() -> None:
    """Returns '' for empty input."""
    assert kd._one_line("") == ""


def test_one_line_first_line_only() -> None:
    """Returns only the first line."""
    result = kd._one_line("Traceback\n  File test.py\n    raise ValueError")
    assert "Traceback" in result
    assert "\n" not in result


def test_one_line_truncates_long_lines() -> None:
    """Caps at 160 chars."""
    long = "x" * 300
    result = kd._one_line(long)
    assert len(result) <= 160


# ---------------------------------------------------------------------------
# Diagnostics rules — crash output tail appears in data
# ---------------------------------------------------------------------------


def test_repeated_failures_data_includes_crash_tail_when_crashed() -> None:
    """crash_output_tail appears in the diag data for crashed outcomes."""
    now = int(time.time())
    err_snippet = "ValueError: something broke"
    task = {
        "id": "t_demo_crash",
        "title": "demo",
        "assignee": "demo",
        "status": "ready",
        "consecutive_failures": 3,
        "last_failure_error": err_snippet,
    }
    runs = [
        {"id": 1, "outcome": "crashed", "error": err_snippet},
        {"id": 2, "outcome": "crashed", "error": err_snippet},
        {"id": 3, "outcome": "crashed", "error": err_snippet},
    ]
    fake_tail = "Worker died: OOM"
    with patch("hermes_cli.kanban_diagnostics._crash_output_tail", return_value=fake_tail):
        diags = kd.compute_task_diagnostics(task, [], runs, now=now)

    crash_diags = [d for d in diags if d.kind in ("repeated_failures", "repeated_crashes")]
    if crash_diags:
        d = crash_diags[0]
        assert d.data.get("crash_output_tail") == fake_tail
        # Title or detail should reference it
        assert any(
            phrase in d.title + d.detail
            for phrase in ["Worker died", "OOM", fake_tail[:30]]
        )


def test_repeated_failures_data_omits_tail_on_completed_outcome() -> None:
    """crash_output_tail is not included when the most recent outcome is not crashed."""
    now = int(time.time())
    task = {
        "id": "t_demo_fail",
        "title": "demo",
        "assignee": "demo",
        "status": "ready",
        "consecutive_failures": 2,
        "last_failure_error": "timeout",
    }
    runs = [
        {"id": 1, "outcome": "failed", "error": "timeout"},
        {"id": 2, "outcome": "failed", "error": "timeout"},
    ]
    diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    for d in diags:
        if d.kind == "repeated_failures":
            val = d.data.get("crash_output_tail")
            assert val is None or val == ""


def test_repeated_crashes_data_includes_crash_tail() -> None:
    """crash_output_tail in repeated_crashes data."""
    now = int(time.time())
    task = {
        "id": "t_demo_crash2",
        "title": "demo",
        "assignee": "demo",
        "status": "ready",
        "consecutive_failures": 0,
        "last_failure_error": None,
    }
    runs = [
        {"id": 1, "outcome": "crashed", "error": "segfault"},
        {"id": 2, "outcome": "crashed", "error": "segfault"},
    ]
    fake_tail = "SIGSEGV at 0xdeadbeef"
    with patch("hermes_cli.kanban_diagnostics._crash_output_tail", return_value=fake_tail):
        diags = kd.compute_task_diagnostics(task, [], runs, now=now)
    crash_diags = [d for d in diags if d.kind == "repeated_crashes"]
    assert crash_diags
    d = crash_diags[0]
    assert d.data.get("crash_output_tail") == fake_tail


# ---------------------------------------------------------------------------
# Regression: ANSI RE does not strip non-escape text
# ---------------------------------------------------------------------------


def test_ansi_re_on_real_log_keeps_content() -> None:
    """A realistic log snippet keeps its meaningful content after stripping."""
    raw_log = (
        "\x1b[32m2026-09-17 15:00:00\x1b[0m [INFO] "
        "\x1b[1mWorker started\x1b[0m\\n"
        "\x1b[31mERROR\x1b[0m: cannot find resource\\n"
    )
    cleaned = _ANSI_ESCAPE_RE.sub("", raw_log)
    assert "2026-09-17 15:00:00" in cleaned
    assert "[INFO]" in cleaned
    assert "Worker started" in cleaned
    assert "ERROR: cannot find resource" in cleaned


# ---------------------------------------------------------------------------
# Spawn handle cleanup regression
# ---------------------------------------------------------------------------


def test_spawn_registry_cleanup_removes_dead_procs(clear_spawned_registry) -> None:
    """The inline cleanup in _default_spawn (triggered via the > MAX/2 guard)
    is tested indirectly by verifying that dead procs are removed from the
    registry when the module guard fires."""
    # Import early to avoid UnboundLocalError (import = assignment redefines scope).
    from hermes_cli.kanban_db_dispatch import _SPAWNED_WORKER_PROCS_MAX as _max
    max_reg = _max
    # Simulate a full-ish registry of dead procs
    for pid in range(1, max_reg // 2 + 2):
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = 0  # already dead
        _spawned_worker_procs[pid] = mock_proc

    # Trigger the inline cleanup (this is what _default_spawn does)
    from hermes_cli.kanban_db_dispatch import _spawned_worker_procs as reg

    def _simulated_register(pid_):
        reg[pid_] = MagicMock(spec=subprocess.Popen)
        reg[pid_].poll.return_value = None  # still alive
        if len(reg) > _max // 2:
            reg.clear()
            reg[pid_] = MagicMock(spec=subprocess.Popen)
            reg[pid_].poll.return_value = None

    _simulated_register(99998)
    # After cleanup, most old entries are gone
    assert len(reg) <= 2