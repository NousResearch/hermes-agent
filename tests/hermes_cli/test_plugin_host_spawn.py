"""Spawn contract of ``hermes_cli.plugin_host._spawn_host``: the plugin host child must never be
forked out of a threaded gateway (macOS kills the child pre-exec in Apple's fork handlers), must
start in the profile's working directory, and must not inherit the parent's descriptors.

Regression for the operations/metrics-dashboard/smart-router-studio 502s: the gateway's
``Popen`` fell back to fork() (this CPython lacks ``POSIX_SPAWN_CLOSEFROM`` on macOS), and the
forked child died with SIGSEGV in ``nw_settings_child_has_forked`` before exec, so every
secondary-profile plugin host answered "channel closed".
"""

import os
import signal
import socket
import sys
from pathlib import Path

import pytest

from hermes_cli.plugin_host import _spawn_host, run_no_fork

REPO_ROOT = str(Path(__file__).resolve().parents[2])


def _child_env(**extra):
    env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "TZ", "SYSTEMROOT")
           if key in os.environ}
    env.update(HOME=str(Path.home()), USERPROFILE=str(Path.home()), **extra)
    return env


def test_spawn_host_roundtrip_pipes_and_wait(tmp_path):
    proc = _spawn_host(
        [sys.executable, "-P", "-c",
         "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read()); sys.stdout.buffer.flush()"],
        _child_env(), str(tmp_path))
    proc.stdin.write(b"host-spawn-echo")
    proc.stdin.close()
    assert proc.stdout.read() == b"host-spawn-echo"
    assert proc.wait(timeout=30) == 0
    assert proc.poll() == 0


@pytest.mark.platforms("posix")
def test_spawn_host_child_chdirs_and_drops_plumbing_env(tmp_path):
    code = (
        "import os;"
        "from hermes_cli.plugin_host_child import _close_inherited_fds, _apply_spawn_cwd;"
        "_close_inherited_fds(set()); _apply_spawn_cwd();"
        "print(os.path.realpath(os.getcwd()));"
        "print(os.environ.get('HERMES_PLUGIN_HOST_CWD'))"
    )
    proc = _spawn_host([sys.executable, "-P", "-c", code],
                       _child_env(PYTHONPATH=REPO_ROOT), str(tmp_path))
    lines = proc.stdout.read().decode().splitlines()
    assert proc.wait(timeout=30) == 0
    assert lines == [os.path.realpath(tmp_path), "None"]


@pytest.mark.platforms("posix")
def test_spawn_host_child_closes_inherited_fds(tmp_path):
    junk_pipes = [os.pipe() for _ in range(3)]
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        code = (
            "import os;"
            "from hermes_cli.plugin_host_child import _close_inherited_fds;"
            "_close_inherited_fds(set());"
            "print(len(os.listdir('/dev/fd')))"
        )
        proc = _spawn_host([sys.executable, "-P", "-c", code],
                           _child_env(PYTHONPATH=REPO_ROOT), str(tmp_path))
        open_fds = int(proc.stdout.read().decode().strip())
        assert proc.wait(timeout=30) == 0
    finally:
        for r, w in junk_pipes:
            os.close(r)
            os.close(w)
        listener.close()
    # 0/1/2 plus the transient fd os.listdir itself holds while scanning.
    assert open_fds <= 4


@pytest.mark.platforms("posix")
def test_spawn_host_kill_terminates_child(tmp_path):
    proc = _spawn_host([sys.executable, "-P", "-c", "import sys; sys.stdin.buffer.read()"],
                       _child_env(), str(tmp_path))
    assert proc.poll() is None
    proc.kill()
    assert proc.wait(timeout=30) == -signal.SIGKILL
    assert proc.poll() == -signal.SIGKILL


def test_run_no_fork_captures_streams_and_status(tmp_path):
    done = run_no_fork([sys.executable, "-P", "-c",
                        "import sys; sys.stdout.write('out'); sys.stderr.write('err'); "
                        "sys.exit(3)"],
                       _child_env(), cwd=str(tmp_path))
    assert done.returncode == 3
    assert done.stdout == b"out"
    assert done.stderr == b"err"


@pytest.mark.platforms("posix")
def test_run_no_fork_timeout_kills_the_child(tmp_path):
    import subprocess

    with pytest.raises(subprocess.TimeoutExpired):
        run_no_fork([sys.executable, "-P", "-c",
                     "import threading; threading.Event().wait()"],
                    _child_env(), timeout=1, cwd=str(tmp_path))
