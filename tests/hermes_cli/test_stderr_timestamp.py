"""Tests for hermes_cli.stderr_timestamp."""

import re
import sys

from gateway.restart import EXTERNAL_GATEWAY_SUPERVISOR_ENV, is_gateway_supervisor_process
from hermes_cli import stderr_timestamp


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


def test_child_env_forwards_supervisor_marker_under_launchd():
    """launchd's XPC label on the wrapper must survive into the grandchild."""
    child_env = stderr_timestamp._child_env_for_command(
        {
            "PATH": "/usr/bin",
            "XPC_SERVICE_NAME": "ai.hermes.gateway-butler",
        }
    )
    assert child_env is not None
    assert child_env["PATH"] == "/usr/bin"
    assert child_env[EXTERNAL_GATEWAY_SUPERVISOR_ENV] == "1"
    assert is_gateway_supervisor_process(child_env) is True


def test_child_env_skips_interactive_xpc_zero():
    """Interactive macOS shells inherit XPC_SERVICE_NAME=0 — do not mark them."""
    assert (
        stderr_timestamp._child_env_for_command(
            {"PATH": "/usr/bin", "XPC_SERVICE_NAME": "0"}
        )
        is None
    )
    assert stderr_timestamp._child_env_for_command({"PATH": "/usr/bin"}) is None
    assert is_gateway_supervisor_process({"XPC_SERVICE_NAME": "0"}) is False


def test_main_forwards_supervisor_marker_to_child(tmp_path, monkeypatch):
    """The wrapper hop must set HERMES_GATEWAY_EXTERNAL_SUPERVISOR in the child."""
    monkeypatch.setenv("XPC_SERVICE_NAME", "ai.hermes.gateway-butler")
    monkeypatch.delenv(EXTERNAL_GATEWAY_SUPERVISOR_ENV, raising=False)
    log_path = tmp_path / "gateway.error.log"
    marker_path = tmp_path / "marker.txt"
    code = (
        "import os\n"
        f"from pathlib import Path\n"
        f"Path({str(marker_path)!r}).write_text("
        f"os.environ.get({EXTERNAL_GATEWAY_SUPERVISOR_ENV!r}, ''), encoding='utf-8')\n"
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
    assert marker_path.read_text(encoding="utf-8") == "1"


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
