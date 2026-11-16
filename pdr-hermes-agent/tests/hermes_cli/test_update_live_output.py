"""Live output must reach disk before exit, without concealing silence."""
import io
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from hermes_cli import main_dashboard as output
from hermes_cli import update_cmd


@pytest.mark.parametrize("gateway", [False, True, None])
def test_child_progress_reaches_log_before_exit(tmp_path, monkeypatch, gateway):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    terminal = io.StringIO()
    monkeypatch.setattr(sys, "stdout", terminal)
    state = output._install_hangup_protection(gateway) if gateway is not None else None
    log = tmp_path / "logs" / "update.log"
    release = tmp_path / "release"
    ready = tmp_path / "ready"
    script = tmp_path / "child.py"
    script.write_text(
        "import os, pathlib, sys, time\n"
        f"pathlib.Path({str(ready)!r}).touch()\n"
        "os.write(1, b'progress: ' + bytes([0xe2, 0x82]))\n"
        "time.sleep(.1)\n"
        "os.write(2, bytes([0xac, 0xff]))\n"
        f"while not pathlib.Path({str(release)!r}).exists(): time.sleep(.02)\n"
        "print(' done', flush=True)\n"
        "sys.exit(7)\n", encoding="utf-8")
    results = []
    worker = threading.Thread(target=lambda: results.append(
        update_cmd._run_logged_subprocess([sys.executable, str(script)])))
    try:
        worker.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            text = log.read_text(encoding="utf-8") if log.exists() else ""
            if "progress: €�" in text:
                break
            time.sleep(.02)
        assert ready.exists(), "child fixture never started"
        assert "progress: €�" in text, "output withheld while child waits for release"
        assert worker.is_alive()
        size = log.stat().st_size
        time.sleep(.3)
        assert log.stat().st_size == size, "silence must not manufacture progress"
        assert terminal.getvalue() == ""
        if gateway is not None:
            assert state["installed"]
            print("update stage", flush=True)
            assert "update stage" in log.read_text(encoding="utf-8")
    finally:
        release.touch()
        worker.join(10)
        output._finalize_update_output(state)
    assert not worker.is_alive()
    assert results[0].returncode == 7
    assert results[0].stdout == "progress: €� done\n"
    assert results[0].stderr is None


def test_cancelled_output_reader_reaps_child(tmp_path, monkeypatch):
    pidfile = tmp_path / "pid"
    child = tmp_path / "cancel_child.py"
    child.write_text(
        "import os, pathlib, time\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()), encoding='utf-8')\n"
        "print('started', flush=True)\n"
        "time.sleep(30)\n", encoding="utf-8")

    class CancelLog:
        def write(self, text):
            raise KeyboardInterrupt

    monkeypatch.setattr(sys, "stdout", output._UpdateOutputStream(io.StringIO(), CancelLog()))
    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        update_cmd._run_logged_subprocess([sys.executable, str(child)])
    assert time.monotonic() - started < 10
    import psutil
    assert not psutil.pid_exists(int(pidfile.read_text(encoding="utf-8")))
