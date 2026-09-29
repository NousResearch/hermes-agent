"""A dead-stalled network fetch ends `hermes update` with an error, never a hang (#93759, #95777).

`_git_run(network=True)` bounds the wait AND runs the call in its own process group
(#124794): a timed-out call tree-kills the whole git process tree instead of orphaning
nested lazy fetches, then reports a failed CompletedProcess whose stderr names the stall,
so every caller's existing fetch-failure path prints one clear line. Local git
(network=False) is unbounded.
"""

import os
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest
import psutil

import hermes_cli.update_cmd as update_cmd


def _unreachable_pid() -> int:
    """A pid that is not running, so a double's ``pid`` cannot reach a real process tree.

    A stalled network call really does ``kill_process_tree(proc)``, which on Windows ends at
    ``taskkill /T /F /PID <pid>``. With a test double that pid is a bare integer, not a retained
    handle, so a fixed literal makes the run destroy whatever unrelated process happens to hold
    it — and ``/T`` takes that process's whole subtree with it.

    Liveness is probed with ``psutil.pid_exists``, never ``os.kill(pid, 0)``: on Windows every
    signal other than the two console events goes straight to TerminateProcess, so a probe built
    on ``os.kill`` would kill the very process it was checking.
    """
    for candidate in range(4_000_000, 0, -1):
        if not psutil.pid_exists(candidate):
            return candidate
    raise RuntimeError("no unreachable pid found")


def _timeout(cmd, **kwargs):
    if "timeout" in kwargs:
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
    return MagicMock(returncode=0, stdout="ok", stderr="")


def test_network_fetch_stall_becomes_a_failed_run_with_a_named_cause(monkeypatch):
    monkeypatch.setattr(update_cmd, "_m", lambda: MagicMock(PROJECT_ROOT="/repo"))

    class _HangingProc:
        args = ["git", "fetch", "origin", "main"]
        pid = _unreachable_pid()
        returncode = 0

        def communicate(self, timeout=None):
            raise subprocess.TimeoutExpired(self.args, timeout if isinstance(timeout, (int, float)) else 30)

        def kill(self):
            pass

    with patch.object(update_cmd.subprocess, "Popen", return_value=_HangingProc()), \
         patch.object(update_cmd, "NETWORK_GIT_TIMEOUT_SECONDS", 30):
        result = update_cmd._git_run(["git"], ["fetch", "origin", "main"], network=True)

    assert result.returncode != 0
    assert "timed out" in result.stderr and "fetch" in result.stderr


def test_local_git_stays_unbounded_and_check_true_raises(monkeypatch):
    monkeypatch.setattr(update_cmd, "_m", lambda: MagicMock(PROJECT_ROOT="/repo"))
    with patch.object(update_cmd.subprocess, "run", side_effect=_timeout) as run:
        assert update_cmd._git_run(["git"], ["rev-parse", "HEAD"]).returncode == 0
        assert "timeout" not in run.call_args.kwargs


class _FakeProc:
    def __init__(self, cmd):
        self.args = list(cmd)
        self.pid = _unreachable_pid()
        self.returncode = 0

    def communicate(self, timeout=None):
        if timeout is not None and timeout < 1:
            raise subprocess.TimeoutExpired(self.args, timeout)
        return "", ""

    def kill(self):
        pass


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_timed_out_network_git_reaps_the_whole_process_tree(monkeypatch, tmp_path):
    """The regression from #124794: nested children of a stalled network call must not survive.

    On a treeless partial clone git spawns nested lazy fetches when the promisor remote stalls;
    killing only the direct child left that tree running (an 8 GB host lost its alerting layer
    to the resulting process pile-up). The network runner must tree-kill on timeout.
    """
    monkeypatch.setattr(update_cmd, "_m", lambda: MagicMock(PROJECT_ROOT=str(tmp_path)))
    monkeypatch.setattr(update_cmd, "NETWORK_GIT_TIMEOUT_SECONDS", 3)

    # The "git" stand-in spawns a child that outlives the parent's timeout.
    stall = tmp_path / "stall.py"
    stall.write_text(
        "import subprocess, sys, time\n"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(tmp_path / 'grandchild.pid')!r}, 'w').write(str(g.pid))\n"
        "time.sleep(60)\n"
    )
    result = update_cmd._git_run([sys.executable], [str(stall)], network=True)

    assert result.returncode == 124
    assert "timed out" in result.stderr

    pid_file = tmp_path / "grandchild.pid"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not pid_file.exists():
        time.sleep(0.05)
    assert pid_file.exists(), "stall child never spawned its grandchild"
    pid = int(pid_file.read_text().strip())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        pytest.fail(f"grandchild pid {pid} survived the network-git timeout (orphaned process tree)")


def test_network_fetch_env_disables_prompts_and_lazy_fetch(monkeypatch, tmp_path):
    monkeypatch.setattr(update_cmd, "_m", lambda: MagicMock(PROJECT_ROOT=str(tmp_path)))
    seen = {}

    def popen(cmd, **kwargs):
        seen.update(kwargs)
        return _FakeProc(cmd)

    monkeypatch.setattr(update_cmd.subprocess, "Popen", popen)
    update_cmd._git_run(["git"], ["fetch", "origin", "main"], cwd=tmp_path, network=True)

    # The no-prompt guard still rides along with the bound, and a fetch never lazy-fetches.
    assert seen["env"]["GIT_TERMINAL_PROMPT"] == "0"
    assert seen["env"]["GCM_INTERACTIVE"] == "Never"
    assert seen["env"]["GIT_NO_LAZY_FETCH"] == "1"
    assert seen["stdin"] is subprocess.DEVNULL
    if sys.platform != "win32":
        assert seen["process_group"] == 0
