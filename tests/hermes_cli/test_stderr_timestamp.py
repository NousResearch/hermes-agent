"""Tests for hermes_cli.stderr_timestamp."""

import re
import sys

import pytest

from gateway.restart import EXTERNAL_GATEWAY_SUPERVISOR_ENV
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


_MALLOC_STACK_LOGGING_WARNING = (
    "MallocStackLogging: can't turn off malloc stack logging because it was not enabled."
)


def test_benign_malloc_warning_exact_match_contract():
    accepted = [
        _MALLOC_STACK_LOGGING_WARNING,
        f"Python(12345) {_MALLOC_STACK_LOGGING_WARNING}",
        f"python(67890) {_MALLOC_STACK_LOGGING_WARNING}\r\n",
    ]
    for line in accepted:
        assert stderr_timestamp._is_benign_darwin_malloc_stack_logging_line(
            line, platform="darwin"
        )
        assert not stderr_timestamp._is_benign_darwin_malloc_stack_logging_line(
            line, platform="linux"
        )

    rejected = [
        f"ERROR: {_MALLOC_STACK_LOGGING_WARNING}",
        f"{_MALLOC_STACK_LOGGING_WARNING} extra context",
        "MallocStackLogging: malloc stack logging enabled.",
        f"Python(not-a-pid) {_MALLOC_STACK_LOGGING_WARNING}",
    ]
    for line in rejected:
        assert not stderr_timestamp._is_benign_darwin_malloc_stack_logging_line(
            line, platform="darwin"
        )


@pytest.mark.macos_only
def test_main_drops_only_benign_malloc_warning_on_darwin(tmp_path):
    log_path = tmp_path / "gateway.error.log"
    code = (
        "import sys\n"
        "sys.stderr.write('real failure\\n')\n"
        f"sys.stderr.write({('Python(12345) ' + _MALLOC_STACK_LOGGING_WARNING + chr(10))!r})\n"
        f"sys.stderr.write({('ERROR: ' + _MALLOC_STACK_LOGGING_WARNING + chr(10))!r})\n"
        "sys.stderr.write('another real failure\\n')\n"
    )

    rc = stderr_timestamp.main(
        ["--error-log", str(log_path), "--", sys.executable, "-c", code]
    )

    assert rc == 0
    body = log_path.read_text(encoding="utf-8")
    assert f"Python(12345) {_MALLOC_STACK_LOGGING_WARNING}" not in body
    assert "real failure" in body
    assert f"ERROR: {_MALLOC_STACK_LOGGING_WARNING}" in body
    assert "another real failure" in body


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
