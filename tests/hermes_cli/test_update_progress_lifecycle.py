"""Progress must be observable before a real update subprocess finishes."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time
import threading

import pytest

from hermes_cli.main_dashboard import (
    _finalize_update_output,
    _install_hangup_protection,
)
from hermes_cli.update_cmd import _log_only_write, _run_logged_subprocess


@pytest.mark.parametrize("failure", [BrokenPipeError, ValueError])
def test_closed_terminal_does_not_stop_progress_witness(tmp_path, monkeypatch, failure):
    from hermes_cli import update_cmd

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    observed = threading.Event()
    writes = []
    real_write = update_cmd._log_only_write

    def witness(text):
        real_write(text)
        writes.append(text)
        if len(writes) >= 2:
            observed.set()

    class ClosedTerminal:
        def write(self, _text):
            raise failure("fixture terminal is closed")

        def flush(self):
            raise failure("fixture terminal is closed")

    monkeypatch.setattr(update_cmd, "_log_only_write", witness)
    monkeypatch.setattr(sys, "stdout", ClosedTerminal())
    with update_cmd._update_progress_heartbeat("still ({elapsed}s)", interval_seconds=.02):
        assert observed.wait(3), "the heartbeat died before publishing two log witnesses"
    assert "still (" in (tmp_path / "logs" / "update.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("gateway_mode", [False, True])
def test_update_output_mirror_preserves_progress_in_both_modes(
    tmp_path, monkeypatch, gateway_mode
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    original = (sys.stdout, sys.stderr)
    state = _install_hangup_protection(gateway_mode=gateway_mode)
    try:
        print("visible progress", flush=True)
        _log_only_write("build progress")
        log = tmp_path / "logs" / "update.log"
        assert log.exists(), "gateway updates must publish the watchdog's witness"
        assert "visible progress" in log.read_text(encoding="utf-8")
        assert "build progress" in log.read_text(encoding="utf-8")
    finally:
        _finalize_update_output(state)
    assert (sys.stdout, sys.stderr) == original


@pytest.mark.parametrize("terminated_line", [True, False])
def test_real_build_output_reaches_log_before_child_exit(
    tmp_path: Path, monkeypatch, terminated_line
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    payload = "build is still running" + ("\n" if terminated_line else "")
    child = (
        "import pathlib,sys,time; "
        "ready,release=map(pathlib.Path,sys.argv[1:3]); "
        "sys.stdout.write(sys.argv[3]);sys.stdout.flush();ready.touch(); "
        "deadline=time.monotonic()+25\n"
        "while not release.exists() and time.monotonic()<deadline:time.sleep(.02)\n"
        "sys.exit(7)"
    )
    with ThreadPoolExecutor(max_workers=1) as workers:
        running = workers.submit(
            _run_logged_subprocess,
            [sys.executable, "-u", "-c", child, str(ready), str(release), payload],
        )
        try:
            deadline = time.monotonic() + 15
            log = tmp_path / "logs" / "update.log"
            observed = False
            while time.monotonic() < deadline:
                if ready.exists() and log.exists():
                    observed = "build is still running" in log.read_text(encoding="utf-8")
                    if observed:
                        break
                time.sleep(.02)
            assert not running.done(), "the witness must precede process completion"
            assert observed, "live subprocess output never reached the progress log"
        finally:
            release.touch()
            result = running.result(timeout=30)
        assert result.returncode == 7
        assert result.stdout == payload
