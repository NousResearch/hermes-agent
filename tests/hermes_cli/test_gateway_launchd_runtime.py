"""An unmanaged macOS gateway runs as a one-off launchd job through osascript, not as its caller's child.

Local Network Privacy attributes a process to its responsible app; a gateway left running after the
app that started it quits becomes responsible for itself and macOS refuses its LAN connects (a model
server on another machine answers ``No route to host``). A launchd-spawned osascript stays the
responsible code for the job's whole life (#71206).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from hermes_cli.gateway_launchd_runtime import start_gateway_job
from tests.waiting import wait_until


def _command_of(pid: int) -> str:
    return subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True, encoding="utf-8",
                          check=False, stdin=subprocess.DEVNULL).stdout.strip()


@pytest.mark.platforms("macos")
def test_gateway_job_is_a_launchd_osascript_child_that_reads_as_unsupervised(tmp_path):
    report = tmp_path / "report.json"
    release = tmp_path / "release"
    child = (
        "import json, os, pathlib, time; "
        f"pathlib.Path({str(report)!r}).write_text(json.dumps({{'ppid': os.getppid(), "
        "'xpc': os.environ.get('XPC_SERVICE_NAME'), 'marker': os.environ.get('PROBE_MARKER')}})); "
        f"[time.sleep(0.1) for _ in range(300) if not pathlib.Path({str(release)!r}).exists()]"
    )
    job = start_gateway_job([sys.executable, "-c", child], env={**os.environ, "PROBE_MARKER": "env-carried"},
                            cwd=tmp_path, log_path=tmp_path / "gateway-stdio.log")
    assert job is not None, "launchd refused the job in this user's GUI domain"
    try:
        try:
            wait_until(lambda: report.exists() or job.poll() is not None, timeout=30)
        except AssertionError:
            pass  # the assert below names what the job did instead
        log = tmp_path / "gateway-stdio.log"
        state = subprocess.run(["launchctl", "print", job.target], capture_output=True, text=True, encoding="utf-8",
                               check=False, stdin=subprocess.DEVNULL).stdout
        assert report.exists(), (f"tmp={tmp_path}\nlog={log.read_text(encoding='utf-8') if log.exists() else None!r}\n"
                                 f"launchctl print:\n{state[:3000]}")
        seen = json.loads(report.read_text(encoding="utf-8-sig"))
        wrapper = seen["ppid"]
        # The caller is not an ancestor: launchd (pid 1) spawned osascript, osascript the command.
        assert _command_of(wrapper).startswith("/usr/bin/osascript"), _command_of(wrapper)
        assert subprocess.run(["ps", "-o", "ppid=", "-p", str(wrapper)], capture_output=True, text=True, encoding="utf-8",
                              check=False, stdin=subprocess.DEVNULL).stdout.strip() == "1"
        assert seen["marker"] == "env-carried"
        # Read as unsupervised, or /restart would exit for a respawn no KeepAlive provides.
        assert seen["xpc"] == "0"
        assert not list(tmp_path.glob(".*.plist")), "the plist holding the launch env must not stay on disk"
        assert job.poll() is None
    finally:
        release.write_text("", encoding="utf-8")
    assert wait_until(lambda: job.poll() is not None, timeout=30)
    assert job.poll() == 0
