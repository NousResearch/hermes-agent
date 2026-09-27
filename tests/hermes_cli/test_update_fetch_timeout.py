"""A dead-stalled network fetch ends `hermes update` with an error, never a hang (#93759, #95777),
and the stalled fetch's process *tree* is reaped as a unit instead of orphaned (#124794).

`_git_run(network=True)` bounds the wait and runs the git child in its own process group;
a timeout tree-kills the whole group and becomes a failed CompletedProcess whose stderr names
the stall, so every caller's existing fetch-failure path prints one clear line. Local git
(network=False) is unbounded.
"""

import os
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.update_cmd as update_cmd


def _timeout(cmd, **kwargs):
    if "timeout" in kwargs:
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
    return MagicMock(returncode=0, stdout="ok", stderr="")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_local_git_stays_unbounded(monkeypatch):
    monkeypatch.setattr(update_cmd, "_m", lambda: MagicMock(PROJECT_ROOT="/repo"))
    with patch.object(update_cmd.subprocess, "run", side_effect=_timeout) as run:
        assert update_cmd._git_run(["git"], ["rev-parse", "HEAD"]).returncode == 0
        assert "timeout" not in run.call_args.kwargs


def test_network_fetch_stall_becomes_a_failed_run_with_a_named_cause(monkeypatch):
    monkeypatch.setattr(update_cmd, "_m", lambda: MagicMock(PROJECT_ROOT="/repo"))
    from hermes_cli import _subprocess_compat
    monkeypatch.setattr(_subprocess_compat, "bounded_probe_run", lambda *a, **k: None)

    result = update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True)

    assert result.returncode == 124
    assert "timed out" in result.stderr and "fetch" in result.stderr
    try:
        update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True, check=True)
    except subprocess.CalledProcessError as exc:
        assert exc.returncode == 124
    else:
        raise AssertionError("check=True must raise on a timed-out fetch")


@pytest.mark.skipif(os.name != "posix", reason="process-group reap is POSIX-only")
def test_stalled_network_fetch_is_reaped_as_a_process_tree(tmp_path, monkeypatch):
    """Reproduce the #124794 shape: the "git" child spawns a grandchild and both hang.

    `subprocess.run`'s timeout kills only the direct child and leaves the grandchild
    running — on a tree:0 partial clone that grandchild is another `git fetch` of the
    promisor recursion; the hardened runner must take the whole process group down.
    """
    monkeypatch.setattr(update_cmd, "_m", lambda: MagicMock(PROJECT_ROOT=str(tmp_path)))
    monkeypatch.setattr(update_cmd, "NETWORK_GIT_TIMEOUT_SECONDS", 2)
    pid_file = tmp_path / "grandchild.pid"
    fake_git = tmp_path / "git"
    fake_git.write_text("#!/bin/sh\n(sleep 60) &\necho $! > " + str(pid_file) + "\nsleep 60\n")
    fake_git.chmod(0o755)

    result = update_cmd._git_run([str(fake_git)], ["fetch", "origin", "main"], network=True)

    assert result.returncode == 124
    assert "timed out" in result.stderr and "fetch" in result.stderr
    pid = int(pid_file.read_text().strip())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _alive(pid):
        time.sleep(0.1)
    if _alive(pid):
        os.kill(pid, 9)  # never leak the hung grandchild into the test run
        raise AssertionError(f"grandchild pid {pid} outlived the timed-out network fetch")
