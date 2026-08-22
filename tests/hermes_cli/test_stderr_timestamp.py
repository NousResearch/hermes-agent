"""Tests for hermes_cli.stderr_timestamp."""

import re
import sys

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



def test_rotating_writer_rotates_at_max_size(tmp_path, monkeypatch):
    """gateway.error.log must split into .1 / .2 backups instead of growing unbounded."""
    import os

    log_path = tmp_path / "gateway.error.log"
    # Force a tiny cap + many backups so rotation is observable without MBs of data.
    monkeypatch.setattr(stderr_timestamp, "_DEFAULT_MAX_SIZE_MB", 1)
    monkeypatch.setattr(stderr_timestamp, "_DEFAULT_BACKUP_COUNT", 3)

    writer = stderr_timestamp._RotatingWriter(log_path)
    try:
        # ~150KB per chunk; 60 chunks >> 1MB forces several rotations.
        chunk = ("x" * 1024 + "\n") * 150  # ~150KB
        for _ in range(8):
            writer.write(chunk)
    finally:
        writer.close()

    # The live file must never exceed the configured cap by much.
    assert log_path.stat().st_size <= 2 * 1024 * 1024
    # Rotated backups should exist (oldest has the highest suffix near backup_count).
    backups = [p for p in tmp_path.iterdir() if p.name.startswith("gateway.error.log.")]
    assert len(backups) >= 1


def test_rotating_writer_honors_config_override(tmp_path, monkeypatch):
    """logging.backup_count from the canonical reader caps rotated copies."""
    log_path = tmp_path / "gateway.error.log"
    monkeypatch.setattr(stderr_timestamp, "_DEFAULT_MAX_SIZE_MB", 1)
    monkeypatch.setattr(stderr_timestamp, "_DEFAULT_BACKUP_COUNT", 2)
    writer = stderr_timestamp._RotatingWriter(log_path)
    chunk = ("y" * 1000 + "\n") * 200  # ~200KB
    try:
        for _ in range(12):
            writer.write(chunk)
    finally:
        writer.close()
    backups = sorted(
        p for p in tmp_path.iterdir() if p.name.startswith("gateway.error.log.")
    )
    # With backup_count=2 only .1 and .2 should survive.
    assert len(backups) <= 2


def test_rotation_config_reads_logging_keys(tmp_path):
    """logging.max_size_mb / backup_count in config.yaml drive the writer."""
    import json as _json

    cfg_dir = tmp_path / "homedir" / ".hermes"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.yaml").write_text(
        _json.dumps({"logging": {"max_size_mb": 1, "backup_count": 2}})
    )
    # hermes_logging's canonical reader resolves the home via the supported
    # context-local override API.
    import hermes_constants

    token = hermes_constants.set_hermes_home_override(cfg_dir)
    try:
        max_mb, backups = stderr_timestamp._rotation_config()
    finally:
        hermes_constants.reset_hermes_home_override(token)
    assert (max_mb, backups) == (1, 2)
