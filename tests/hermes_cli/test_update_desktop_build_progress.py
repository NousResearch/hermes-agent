"""Desktop rebuild output streams to update.log with bounded progress."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import hermes_cli.update_cmd as update_cmd
from hermes_cli.main_dashboard import _UpdateOutputStream

SLOW_CHILD = (
    "import time\n"
    "for i in range(4):\n"
    "    print(f'build line {i}', flush=True)\n"
    "    time.sleep(0.15)\n"
)


def test_streams_lines_and_prints_progress_while_the_build_runs(monkeypatch):
    terminal = io.StringIO()
    log = io.StringIO()
    monkeypatch.setattr(sys, "stdout", _UpdateOutputStream(terminal, log))
    # The cadence is a module constant, not a parameter: production callers
    # pass nothing, so the seam belongs here rather than in the signature.
    monkeypatch.setattr(update_cmd, "_LOGGED_SUBPROCESS_PROGRESS_SECONDS", 0.1)

    result = update_cmd._run_logged_subprocess([sys.executable, "-c", SLOW_CHILD])

    assert result.returncode == 0
    assert all(f"build line {i}" in result.stdout for i in range(4))
    assert "build line 3" in log.getvalue()
    assert "desktop build running" in terminal.getvalue()
    assert "build line" not in terminal.getvalue().replace("lines captured", "")


def test_quick_child_prints_no_progress_line(monkeypatch):
    terminal = io.StringIO()
    log = io.StringIO()
    monkeypatch.setattr(sys, "stdout", _UpdateOutputStream(terminal, log))

    result = update_cmd._run_logged_subprocess([sys.executable, "-c", "print('fast')"])

    assert result.returncode == 0
    assert "fast" in log.getvalue()
    assert terminal.getvalue() == ""


def test_falls_back_to_the_log_file_when_the_mirror_is_off(monkeypatch, tmp_path: Path):
    plain = io.StringIO()
    monkeypatch.setattr(sys, "stdout", plain)
    # update_cmd re-exports get_hermes_home precisely so it can be patched here.
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: tmp_path)

    result = update_cmd._run_logged_subprocess(
        [sys.executable, "-c", "print('gateway-mode build')"]
    )

    assert result.returncode == 0
    assert "gateway-mode build" in (tmp_path / "logs" / "update.log").read_text(encoding="utf-8")
    assert plain.getvalue() == ""


def test_nonzero_exit_and_output_survive(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    # update_cmd re-exports get_hermes_home precisely so it can be patched here.
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: tmp_path)

    result = update_cmd._run_logged_subprocess(
        [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"]
    )

    assert result.returncode == 3
    assert "boom" in result.stdout
