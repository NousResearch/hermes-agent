"""End-to-end regression coverage for ``hermes cron status``."""

import os
import subprocess
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FOREIGN_GATEWAY_PID = 2_000_000_000
_LOCK_HOLDER = f"""
import json
import sys
from pathlib import Path
from gateway import status

ready_path = Path(sys.argv[1])
hermes_home = Path(sys.argv[2])
record = {{
    "pid": {_FOREIGN_GATEWAY_PID},
    "kind": "hermes-gateway",
    "argv": ["external-owner-not-visible-in-this-namespace"],
    "start_time": 1,
}}
status._build_pid_record = lambda: record
if not status.acquire_gateway_runtime_lock():
    raise SystemExit(1)
(hermes_home / "gateway.pid").write_text(json.dumps(record), encoding="utf-8")
ready_path.write_text("READY", encoding="utf-8")
try:
    sys.stdin.read()
finally:
    status.release_gateway_runtime_lock()
"""


def _run_console_entrypoint(
    *argv: str, env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run the packaged console-script contract in a fresh interpreter."""
    return subprocess.run(
        [
            sys.executable,
            "-c",
            # Model a separate PID namespace while retaining the real CLI,
            # lock probe, metadata parsing, output, and cleanup paths.
            "from hermes_cli import gateway as g; "
            "g.find_gateway_pids = lambda *args, **kwargs: []; "
            "from hermes_cli.main import main; raise SystemExit(main())",
            *argv,
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_cron_status_reports_gateway_owner_in_another_namespace(tmp_path):
    """A shared lock owner is healthy even when its PID is not locally visible."""
    hermes_home = tmp_path / "hermes-home"
    (hermes_home / "cron").mkdir(parents=True)
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)
    ready_path = tmp_path / "gateway-lock-ready"

    holder = subprocess.Popen(
        [sys.executable, "-u", "-c", _LOCK_HOLDER, str(ready_path), str(hermes_home)],
        cwd=_REPO_ROOT,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready_path.exists() and holder.poll() is None:
            if time.monotonic() >= deadline:
                raise AssertionError("timed out waiting for gateway lock holder")
            time.sleep(0.01)
        if not ready_path.exists():
            stderr = holder.stderr.read() if holder.stderr is not None else ""
            raise AssertionError(f"gateway lock holder failed to start:\n{stderr}")
        assert ready_path.read_text(encoding="utf-8") == "READY"
        lock_path = hermes_home / "gateway.lock"
        pid_path = hermes_home / "gateway.pid"
        lock_before = lock_path.read_text(encoding="utf-8")
        pid_before = pid_path.read_text(encoding="utf-8")

        result = _run_console_entrypoint("cron", "status", env=env)

        assert result.returncode == 0, result.stderr
        output = result.stdout.lower()
        assert "will not fire" not in output
        assert "won't fire" not in output
        assert "another namespace/container" in output
        assert lock_path.read_text(encoding="utf-8") == lock_before
        assert pid_path.read_text(encoding="utf-8") == pid_before
    finally:
        if holder.stdin is not None:
            holder.stdin.close()
        try:
            holder.wait(timeout=5)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait(timeout=5)


def test_cron_status_handles_unverifiable_lock_path_without_traceback(tmp_path):
    """Unexpected lock-path failures are reported without a false health claim."""
    hermes_home = tmp_path / "hermes-home"
    (hermes_home / "cron").mkdir(parents=True)
    (hermes_home / "gateway.lock").mkdir()
    env = os.environ.copy()
    env["HERMES_HOME"] = str(hermes_home)

    result = _run_console_entrypoint("cron", "status", env=env)

    assert result.returncode == 0, result.stderr
    assert "traceback" not in result.stderr.lower()
    output = result.stdout.lower()
    assert "unable to determine gateway ownership" in output
    assert "will fire automatically" not in output
