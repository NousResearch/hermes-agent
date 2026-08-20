import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from plugins.builder_adapter.errors import AdapterError
from plugins.builder_adapter.validation import (
    ValidationRunner,
    _UnverifiedLaunchdContainmentProbe,
)


def init_repo(path: Path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(path), "config", "user.name", "Test"], check=True
    )
    (path / "x.txt").write_text("x")
    subprocess.run(["git", "-C", str(path), "add", "x.txt"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "base"], check=True)
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def profile(command_id="safe"):
    return {
        "environment_policy": {"allow": ["PATH"], "deny": ["*_API_KEY"]},
        "commands": [
            {
                "command_id": command_id,
                "argv": ["{python}", "-c", "print('ok')"],
                "timeout_seconds": 10,
                "required": True,
            }
        ],
    }


def _living_pids(pid_file: Path) -> list[int]:
    pids = [
        int(value)
        for value in pid_file.read_text(encoding="utf-8").splitlines()
        if value
    ]
    return [
        pid
        for pid in pids
        if subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "pid="],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    ]


def _terminate_probe_pids(pid_file: Path) -> None:
    for pid in _living_pids(pid_file):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if not _living_pids(pid_file):
            return
        time.sleep(0.05)
    raise AssertionError(f"test probe cleanup failed: {_living_pids(pid_file)}")


def test_registered_validation_fails_closed_without_proven_disposable_unit(tmp_path):
    head = init_repo(tmp_path)
    runner = ValidationRunner({"profile": profile()}, python=sys.executable)
    with pytest.raises(AdapterError) as raised:
        runner.run("profile", tmp_path, head)
    assert raised.value.code == "VALIDATION_CONTAINMENT_UNAVAILABLE"
    assert "container or VM" in str(raised.value)


def test_expected_sha_mismatch_runs_nothing(tmp_path):
    init_repo(tmp_path)
    runner = ValidationRunner({"profile": profile()}, python=sys.executable)
    with pytest.raises(AdapterError) as raised:
        runner.run("profile", tmp_path, "f" * 40)
    assert raised.value.code == "HEAD_MISMATCH"


@pytest.mark.live_system_guard_bypass
@pytest.mark.parametrize(
    ("mode", "timeout_seconds", "expected_status", "descendants_escape"),
    [
        ("parent-child-grandchild", 10, 0, False),
        ("setsid-success", 10, 0, True),
        ("setsid-failure", 10, 9, True),
        ("double-fork-success", 10, 0, True),
        ("setsid-timeout", 3, 124, True),
    ],
)
def test_launchd_probe_proves_detached_descendants_escape_job_removal(
    tmp_path, mode, timeout_seconds, expected_status, descendants_escape
):
    """Negative platform proof: launchd bootout is not sufficient confinement."""
    pid_file = tmp_path / f"{mode}.pids"
    probe_script = tmp_path / "probe.py"
    probe_script.write_text(
        """
import os
import pathlib
import subprocess
import sys
import time

mode = sys.argv[1]
pid_file = pathlib.Path(sys.argv[2])

if mode == "parent-child-grandchild":
    subprocess.Popen(
        [sys.executable, __file__, "child-with-grandchild", str(pid_file)]
    )
    while not pid_file.exists():
        time.sleep(0.01)
    raise SystemExit(0)

if mode == "child-with-grandchild":
    grandchild = subprocess.Popen(
        [sys.executable, __file__, "plain-leaf", str(pid_file)]
    )
    pid_file.write_text(str(os.getpid()) + "\\n" + str(grandchild.pid) + "\\n")
    time.sleep(60)
    raise SystemExit(0)

if mode == "plain-leaf":
    time.sleep(60)
    raise SystemExit(0)

if mode == "setsid-leaf":
    os.setsid()
    pid_file.write_text(str(os.getpid()) + "\\n")
    time.sleep(60)
    raise SystemExit(0)

if mode.startswith("setsid"):
    child = subprocess.Popen(
        [sys.executable, __file__, "setsid-leaf", str(pid_file)]
    )
    while not pid_file.exists():
        time.sleep(0.01)
    if mode == "setsid-timeout":
        time.sleep(60)
    raise SystemExit(9 if mode == "setsid-failure" else 0)

if mode == "double-fork-success":
    child = os.fork()
    if child == 0:
        os.setsid()
        grandchild = os.fork()
        if grandchild > 0:
            os._exit(0)
        pid_file.write_text(str(os.getpid()) + "\\n")
        time.sleep(60)
        os._exit(0)
    os.waitpid(child, 0)
    while not pid_file.exists():
        time.sleep(0.01)
    raise SystemExit(0)

raise SystemExit(64)
""".lstrip(),
        encoding="utf-8",
    )
    probe = _UnverifiedLaunchdContainmentProbe(python=sys.executable)
    try:
        result = probe.run(
            [sys.executable, str(probe_script), mode, str(pid_file)],
            tmp_path,
            {"PATH": os.environ["PATH"]},
            timeout_seconds,
            scope_id=f"negative-{mode}",
        )
        assert result["exit_status"] == expected_status
        assert result["containment"]["job_absence_confirmed"] is True
        living = _living_pids(pid_file)
        if descendants_escape:
            assert living, (
                "platform behavior changed: review whether launchd containment "
                "can now be governed before enabling it"
            )
        else:
            assert living == []
    finally:
        if pid_file.exists():
            _terminate_probe_pids(pid_file)
