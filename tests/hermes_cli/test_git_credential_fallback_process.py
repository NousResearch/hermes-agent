"""Each git attempt of the shared credential fallback stays bounded and windowless.

``run_git_with_credential_fallback`` runs the network git verbs of plugin installs and updates, MCP
catalog installs, profile distributions and the gateway's scheduled plugin update check. That check
runs inside the gateway's single housekeeping thread, ahead of the other housekeeping chores, so an
attempt that outlives its timeout stalls every chore behind it. On Windows ``subprocess.run`` kills
only ``git.exe`` and then waits for the captured pipes, which ``git-remote-https`` (or any other
descendant) still holds. The attempts also opened a console window from console-less callers (the
gateway and desktop backend, the #117781 class).
"""

import os
import subprocess
import sys
import time

import pytest

from hermes_cli import git_credentials

# A git stand-in whose child inherits the captured pipes and outlives it, as git-remote-https does.
_PARENT_WITH_PIPE_HOLDING_CHILD = (
    "import subprocess, sys, threading\n"
    "subprocess.Popen([sys.executable, '-c', 'import threading; threading.Event().wait(60)'])\n"
    "threading.Event().wait(60)\n"
)


@pytest.mark.platforms("any")
def test_a_completed_attempt_returns_what_subprocess_run_would():
    argv = [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"]

    ours = git_credentials.run_git_with_credential_fallback(
        argv, "", env=dict(os.environ), capture_output=True, text=True, encoding="utf-8", timeout=30)
    theirs = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8", timeout=30,
                            stdin=subprocess.DEVNULL, check=False)

    assert (ours.returncode, ours.stdout, ours.stderr) == (theirs.returncode, theirs.stdout, theirs.stderr)


@pytest.mark.platforms("any")
def test_a_timed_out_attempt_ends_its_whole_process_tree():
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        git_credentials.run_git_with_credential_fallback(
            [sys.executable, "-c", _PARENT_WITH_PIPE_HOLDING_CHILD], "", env=dict(os.environ),
            capture_output=True, text=True, timeout=2)
    assert time.monotonic() - started < 30


class _FakeGit:
    """Popen stand-in that records how it was spawned and exits cleanly."""

    spawned: list[dict] = []
    returncode = 0
    stdout = stderr = None

    def __init__(self, argv, **kwargs):
        self.spawned.append(kwargs)
        self.args = argv

    def communicate(self, input=None, timeout=None):
        return "", ""

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.platforms("any")
def test_attempts_never_open_a_console_window(monkeypatch):
    """The wiring under test is that every attempt passes ``windows_hide_flags()`` through; the
    helper is pinned so the Linux lane covers it without faking the OS."""
    _FakeGit.spawned = []
    monkeypatch.setattr(git_credentials, "windows_hide_flags", lambda: 0x08000000)
    monkeypatch.setattr(git_credentials.subprocess, "Popen", _FakeGit)

    git_credentials.run_git_with_credential_fallback(
        ["git", "ls-remote", "https://git.example.test/x.git", "HEAD"], "https://git.example.test/x.git",
        env={}, capture_output=True, text=True, timeout=30)

    assert [kwargs.get("creationflags") for kwargs in _FakeGit.spawned] == [0x08000000]
