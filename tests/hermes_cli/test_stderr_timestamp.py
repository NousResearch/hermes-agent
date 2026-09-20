"""Tests for hermes_cli.stderr_timestamp."""

import re
import sys

import pytest

from gateway.restart import (
    EXTERNAL_GATEWAY_SUPERVISOR_ENV,
    GATEWAY_FATAL_CONFIG_EXIT_CODE,
    GATEWAY_SERVICE_RESTART_EXIT_CODE,
)
from hermes_cli import stderr_timestamp

_STALE_GATEWAY_ARGV = [
    sys.executable,
    "-m",
    "hermes_cli.main",
    "gateway",
    "run",
    "--replace",
]
_LAUNCHD_ENV = {"PATH": "/usr/bin", "XPC_SERVICE_NAME": "ai.hermes.gateway-butler"}


def test_main_timestamps_each_stderr_line(tmp_path):
    log_path = tmp_path / "gateway.error.log"
    code = (
        "import sys\n"
        "sys.stderr.write('first failure\\n')\n"
        "sys.stderr.write('second failure without newline\\n')\n"
        "sys.stderr.write('2026-07-15 12:34:56,789 already timestamped')\n"
        "sys.exit(7)\n"
    )

    rc = stderr_timestamp.main(
        [
            "--error-log",
            str(log_path),
            "--",
            sys.executable,
            "-c",
            code,
        ]
    )

    assert rc == 7
    lines = log_path.read_text(encoding="utf-8").splitlines()
    timestamp = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}"
    assert len(lines) == 3
    assert re.fullmatch(f"{timestamp} first failure", lines[0])
    assert re.fullmatch(f"{timestamp} second failure without newline", lines[1])
    assert lines[2] == "2026-07-15 12:34:56,789 already timestamped"


def test_prepare_upgrades_stale_gateway_argv_under_launchd():
    upgraded = stderr_timestamp._prepare_child_command(
        _STALE_GATEWAY_ARGV, _LAUNCHD_ENV
    )
    assert upgraded == [*_STALE_GATEWAY_ARGV, "--external-supervisor"]


def test_prepare_keeps_existing_external_supervisor_flag():
    already = [*_STALE_GATEWAY_ARGV, "--external-supervisor"]
    assert (
        stderr_timestamp._prepare_child_command(already, _LAUNCHD_ENV) == already
    )


def test_prepare_skips_arbitrary_command_under_launchd():
    """A generic wrapper must not mark random launchd children as the gateway."""
    other = [sys.executable, "-c", "print('ok')"]
    assert stderr_timestamp._prepare_child_command(other, _LAUNCHD_ENV) == other


def test_prepare_skips_interactive_xpc_zero_even_for_gateway_argv():
    assert (
        stderr_timestamp._prepare_child_command(
            _STALE_GATEWAY_ARGV, {"PATH": "/usr/bin", "XPC_SERVICE_NAME": "0"}
        )
        == _STALE_GATEWAY_ARGV
    )
    assert (
        stderr_timestamp._prepare_child_command(_STALE_GATEWAY_ARGV, {"PATH": "/usr/bin"})
        == _STALE_GATEWAY_ARGV
    )


# The child is ``python -c <record argv>`` carrying a "gateway run" tail as inert data, which is
# exactly what the guard's real-gateway spawn check matches; it exits at once.
@pytest.mark.spawns_gateway_lookalike
def test_main_injects_flag_into_stale_gateway_child(tmp_path, monkeypatch):
    """Stale plist inner argv must grow --external-supervisor in the grandchild."""
    monkeypatch.setenv("XPC_SERVICE_NAME", "ai.hermes.gateway-butler")
    monkeypatch.delenv(EXTERNAL_GATEWAY_SUPERVISOR_ENV, raising=False)
    log_path = tmp_path / "gateway.error.log"
    marker_path = tmp_path / "argv.txt"
    code = (
        "import sys\n"
        f"from pathlib import Path\n"
        f"Path({str(marker_path)!r}).write_text("
        "'\\n'.join(sys.argv[1:]), encoding='utf-8')\n"
    )
    stale = [sys.executable, "-c", code, "-m", "hermes_cli.main", "gateway", "run", "--replace"]

    rc = stderr_timestamp.main(
        ["--error-log", str(log_path), "--", *stale]
    )

    assert rc == 0
    recorded = marker_path.read_text(encoding="utf-8").splitlines()
    assert recorded[-1] == "--external-supervisor"
    assert "gateway" in recorded and "run" in recorded


def test_main_does_not_mark_arbitrary_launchd_child(tmp_path, monkeypatch):
    monkeypatch.setenv("XPC_SERVICE_NAME", "ai.hermes.gateway-butler")
    monkeypatch.delenv(EXTERNAL_GATEWAY_SUPERVISOR_ENV, raising=False)
    log_path = tmp_path / "gateway.error.log"
    marker_path = tmp_path / "marker.txt"
    code = (
        "import os\n"
        f"from pathlib import Path\n"
        f"Path({str(marker_path)!r}).write_text("
        f"os.environ.get({EXTERNAL_GATEWAY_SUPERVISOR_ENV!r}, 'unset'), encoding='utf-8')\n"
    )

    rc = stderr_timestamp.main(
        [
            "--error-log",
            str(log_path),
            "--",
            sys.executable,
            "-c",
            code,
        ]
    )

    assert rc == 0
    assert marker_path.read_text(encoding="utf-8") == "unset"


def test_main_does_not_mark_unsupervised_child(tmp_path, monkeypatch):
    """Foreground/unsupervised starts must not inherit a fabricated marker."""
    monkeypatch.setenv("XPC_SERVICE_NAME", "0")
    monkeypatch.delenv(EXTERNAL_GATEWAY_SUPERVISOR_ENV, raising=False)
    log_path = tmp_path / "gateway.error.log"
    marker_path = tmp_path / "marker.txt"
    code = (
        "import os\n"
        f"from pathlib import Path\n"
        f"Path({str(marker_path)!r}).write_text("
        f"os.environ.get({EXTERNAL_GATEWAY_SUPERVISOR_ENV!r}, 'unset'), encoding='utf-8')\n"
    )

    rc = stderr_timestamp.main(
        [
            "--error-log",
            str(log_path),
            "--",
            sys.executable,
            "-c",
            code,
        ]
    )

    assert rc == 0
    assert marker_path.read_text(encoding="utf-8") == "unset"


@pytest.mark.spawns_gateway_lookalike
def test_main_maps_gateway_ex_config_to_clean_stop(tmp_path):
    """launchd KeepAlive.SuccessfulExit=false parks exit 0; the wrapper must
    turn gateway EX_CONFIG (78) into that clean stop without swallowing the
    please-restart code (75) or a non-gateway child's 78."""
    log_path = tmp_path / "gateway.error.log"
    gateway_tail = ["-m", "hermes_cli.main", "gateway", "run"]

    rc_config = stderr_timestamp.main(
        [
            "--error-log",
            str(log_path),
            "--",
            sys.executable,
            "-c",
            f"raise SystemExit({GATEWAY_FATAL_CONFIG_EXIT_CODE})",
            *gateway_tail,
        ]
    )
    rc_restart = stderr_timestamp.main(
        [
            "--error-log",
            str(log_path),
            "--",
            sys.executable,
            "-c",
            f"raise SystemExit({GATEWAY_SERVICE_RESTART_EXIT_CODE})",
            *gateway_tail,
        ]
    )
    rc_other = stderr_timestamp.main(
        [
            "--error-log",
            str(log_path),
            "--",
            sys.executable,
            "-c",
            f"raise SystemExit({GATEWAY_FATAL_CONFIG_EXIT_CODE})",
        ]
    )

    assert rc_config == 0
    assert rc_restart == GATEWAY_SERVICE_RESTART_EXIT_CODE
    assert rc_other == GATEWAY_FATAL_CONFIG_EXIT_CODE



@pytest.mark.macos_only
def test_wrapper_forwards_sigusr1_restart_request_to_child(tmp_path):
    """Regression for #101426: launchd owns the wrapper's PID, so ``hermes update`` sends its
    drain-aware SIGUSR1 to the wrapper. It must reach the gateway child and the wrapper must
    report the child's planned exit code — not die of the signal itself (which makes launchd
    treat the restart as a crash and apply its back-off to every sibling profile)."""
    import os
    import signal
    import subprocess
    import time

    log_path = tmp_path / "gateway.error.log"
    ready = tmp_path / "ready"
    child = (
        "import os, signal, sys, time, pathlib\n"
        f"signal.signal(signal.SIGUSR1, lambda *_: (sys.stderr.write('restart requested\\n'), sys.exit({GATEWAY_SERVICE_RESTART_EXIT_CODE})))\n"
        f"pathlib.Path({str(ready)!r}).write_text('1')\n"
        "time.sleep(20)\n"
        "sys.exit(1)\n"
    )
    wrapper = subprocess.Popen(
        [sys.executable, "-m", "hermes_cli.stderr_timestamp", "--error-log", str(log_path), "--",
         sys.executable, "-c", child],
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists():
            assert time.monotonic() < deadline, "child never started"
            time.sleep(0.05)
        os.kill(wrapper.pid, signal.SIGUSR1)
        rc = wrapper.wait(timeout=10)
    finally:
        if wrapper.poll() is None:
            wrapper.kill()

    assert rc == GATEWAY_SERVICE_RESTART_EXIT_CODE
    assert "restart requested" in log_path.read_text(encoding="utf-8")
