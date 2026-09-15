"""Tests for the desktop-build process lock."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import main as cli_main
from hermes_cli import main_desktop as cli_desktop


_LOCK_CHILD = r"""
import sys
import time
from pathlib import Path

from hermes_cli.main_desktop import _ProcessFileLock

lock_path = Path(sys.argv[1])
mode = sys.argv[2]
if mode == "hold":
    with _ProcessFileLock(lock_path, "test", wait=False):
        Path(sys.argv[3]).write_text("ready", encoding="utf-8")
        time.sleep(10)
else:
    with _ProcessFileLock(lock_path, "test", wait=False):
        print("acquired", flush=True)
"""


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = [str(cli_main.PROJECT_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def _run_lock_child(lock_path: Path, mode: str, ready_path: Path | None = None):
    args = [sys.executable, "-c", _LOCK_CHILD, str(lock_path), mode]
    if ready_path is not None:
        args.append(str(ready_path))
    return subprocess.run(
        args,
        cwd=cli_main.PROJECT_ROOT,
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=5,
    )


@pytest.mark.skipif(os.name == "nt", reason="the salvaged implementation uses fcntl on POSIX")
def test_process_file_lock_rejects_concurrent_process_and_releases(tmp_path):
    lock_path = tmp_path / ".update.lock"
    ready_path = tmp_path / "ready"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _LOCK_CHILD,
            str(lock_path),
            "hold",
            str(ready_path),
        ],
        cwd=cli_main.PROJECT_ROOT,
        env=_child_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready_path.exists():
            if holder.poll() is not None:
                stderr = holder.stderr.read() if holder.stderr else ""
                raise AssertionError(f"lock holder exited early: {stderr}")
            if time.monotonic() >= deadline:
                raise AssertionError("lock holder did not acquire the lock")
            time.sleep(0.01)

        contender = _run_lock_child(lock_path, "try")
        assert contender.returncode == 2
        assert "Another Hermes test is already running." in contender.stdout
    finally:
        holder.terminate()
        try:
            holder.wait(timeout=5)
        except subprocess.TimeoutExpired:
            holder.kill()
            holder.wait(timeout=5)
        if holder.stderr:
            holder.stderr.close()

    released = _run_lock_child(lock_path, "try")
    assert released.returncode == 0
    assert released.stdout.strip() == "acquired"


def test_process_lock_paths_use_the_profile_root(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    profile_home = root / "profiles" / "writing"
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    assert cli_desktop._hermes_root_for_process_locks() == root
    assert cli_desktop._desktop_build_process_lock().path == root / ".desktop-build.lock"
