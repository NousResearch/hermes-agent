"""The backend report finds a dashboard started through the install launcher.

The install's ``hermes`` launcher execs ``python -I -c <multi-line bootstrap> dashboard ...``. ``ps``
joins that argv with spaces (macOS also spells the newlines ``\\012``), so the subcommand matcher
cannot tell where the inline source ends. These tests run a real process with exactly that argv shape
(its source sleeps; it never reaches Hermes) and read it back through the real process table.
"""

from __future__ import annotations

import shlex
import subprocess
import sys

import psutil
import pytest

from hermes_cli import gateway_migrate_backends as gmb
from tests.waiting import wait_until

# The launcher's shape: an "entry" bootstrap ending in ``from hermes_cli.main import main``.
_LAUNCHER_SOURCE = ("import os, re, sys\nimport time\ntime.sleep(120)\n"
                    "from hermes_cli.main import main\nsys.exit(main())\n")
_ARGS = ["dashboard", "--no-open", "--host", "127.0.0.1", "--port", "59119"]


@pytest.fixture
def launcher_dashboard(monkeypatch):
    proc = subprocess.Popen([sys.executable, "-I", "-c", _LAUNCHER_SOURCE, *_ARGS],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # The real scan, narrowed to this process: other backends running on the machine (another test run's,
    # whose HERMES_HOME may sit under the real home) must not reach the report's per-backend reads.
    scan = gmb._running_backends
    monkeypatch.setattr(gmb, "_running_backends", lambda: [row for row in scan() if row[0] == proc.pid])
    try:
        wait_until(lambda: psutil.Process(proc.pid).cmdline()[-1:] == ["59119"], timeout=10,
                   what="the launcher-shaped process is running")
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=10)


def test_running_backend_started_through_the_launcher_is_found(launcher_dashboard):
    rows = dict(gmb._running_backends())
    assert launcher_dashboard.pid in rows


def test_the_report_records_its_exact_argv(launcher_dashboard):
    item = next(i for i in gmb.collect_backends() if i.pid == launcher_dashboard.pid)
    assert item.running is True
    assert item.argv[-len(_ARGS):] == _ARGS
    assert item.argv[2:4] == ["-c", _LAUNCHER_SOURCE]


def test_exact_argv_of_a_gone_process_is_none():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)
    assert gmb._exact_argv(proc.pid) is None


@pytest.mark.skipif(sys.platform != "darwin", reason="the placeholder is macOS-only")
def test_the_report_shows_the_launcher_source_as_a_placeholder(launcher_dashboard):
    item = next(i for i in gmb.collect_backends() if i.pid == launcher_dashboard.pid)
    runs = next(line for line in gmb.format_report([item]) if line.lstrip().startswith("runs:"))
    assert "'<inline launcher source>' dashboard --no-open --host 127.0.0.1 --port 59119" in runs
    assert "time.sleep" not in runs


@pytest.mark.skipif(sys.platform in ("darwin", "win32"), reason="the report on Linux and other POSIX hosts")
def test_off_macos_the_runs_line_is_the_whole_argv_as_before(launcher_dashboard):
    # The report for a user who never migrates stays as it was: ``runs:`` is ``shlex.join`` of the argv.
    item = next(i for i in gmb.collect_backends() if i.pid == launcher_dashboard.pid)
    argv = psutil.Process(launcher_dashboard.pid).cmdline()
    runs = [line for line in gmb.format_report([item]) if line.lstrip().startswith("runs:")]
    assert runs == [f"    runs:    {shlex.join(argv)}"]


@pytest.fixture
def launcher_proxy():
    proc = subprocess.Popen([sys.executable, "-I", "-c", _LAUNCHER_SOURCE, "proxy", "start"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_until(lambda: psutil.Process(proc.pid).cmdline()[-1:] == ["start"], timeout=10,
                   what="the launcher-shaped non-backend process is running")
        yield proc
    finally:
        proc.kill()
        proc.wait(timeout=10)


@pytest.mark.skipif(sys.platform == "darwin", reason="macOS re-reads the argv of such rows")
def test_off_macos_the_scan_reads_no_argv_for_a_launcher_row_that_is_not_a_backend(launcher_proxy, monkeypatch):
    read: list[int] = []
    monkeypatch.setattr(gmb, "_exact_argv", lambda pid: read.append(pid), raising=False)  # main has none
    assert launcher_proxy.pid not in dict(gmb._running_backends())
    assert read == []
