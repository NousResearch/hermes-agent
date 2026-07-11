from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[3]
    / "ops"
    / "muncho"
    / "runtime"
    / "planned_gateway_restart.sh"
)


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)


def _fixture(tmp_path: Path, *, pid: str = "4242", sudo_rc: int = 0):
    bin_dir = tmp_path / "bin"
    active = tmp_path / "active"
    python_bin = active / ".venv" / "bin" / "python"
    bin_dir.mkdir()
    python_bin.parent.mkdir(parents=True)
    _write_executable(python_bin, "#!/bin/sh\nexit 0\n")
    _write_executable(
        bin_dir / "systemctl",
        f"#!/bin/sh\nprintf '%s\\n' '{pid}'\n",
    )
    _write_executable(
        bin_dir / "sudo",
        "#!/bin/sh\nprintf '%s\\n' \"$*\" > \"$CALL_LOG\"\n"
        f"exit {sudo_rc}\n",
    )
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CALL_LOG": str(tmp_path / "sudo-call.txt"),
    }
    return active, env


def _run(tmp_path: Path, *, pid: str = "4242", sudo_rc: int = 0):
    active, env = _fixture(tmp_path, pid=pid, sudo_rc=sudo_rc)
    return subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(active),
            "hermes-cloud-gateway.service",
            "ai-platform-brain",
            "/opt/adventico-ai-platform/hermes-home",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    ), env


def test_marker_is_written_for_exact_active_pid(tmp_path):
    result, env = _run(tmp_path)
    assert result.returncode == 0
    assert result.stdout.strip() == "PLANNED_STOP_MARKER_PASS"
    call = Path(env["CALL_LOG"]).read_text()
    assert "gateway.status" in call
    assert "write_planned_stop_marker" in call
    assert call.rstrip().endswith("4242")
    assert "HERMES_HOME=/opt/adventico-ai-platform/hermes-home" in call


def test_inactive_service_does_not_write_marker(tmp_path):
    result, env = _run(tmp_path, pid="0")
    assert result.returncode == 0
    assert result.stdout.strip() == "PLANNED_STOP_MARKER_NOT_NEEDED"
    assert not Path(env["CALL_LOG"]).exists()


def test_marker_failure_blocks_restart_path(tmp_path):
    result, _ = _run(tmp_path, sudo_rc=1)
    assert result.returncode == 4
    assert "BLOCKED_PLANNED_STOP_MARKER_WRITE_FAILED" in result.stderr


def test_invalid_pid_fails_closed(tmp_path):
    result, _ = _run(tmp_path, pid="not-a-pid")
    assert result.returncode == 2
    assert "BLOCKED_INVALID_GATEWAY_PID" in result.stderr
