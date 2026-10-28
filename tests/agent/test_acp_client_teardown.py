"""Teardown of ACPClient's session process must reach the launcher's native descendants.

The npm ACP launchers (#124835) are thin wrappers that spawn a platform-native
executable as their child; a launcher-only TERM orphans that worker, and a
container's PID 1 never reaps it — 16 leaked processes ≈ 2.6 GiB RSS in the
reporter's deployment.
"""

import os
import sys

import pytest


_LAUNCHER_WITH_NATIVE_CHILD = """\
import subprocess, sys, time
# Shape of the npm ACP launcher (#124835): a thin wrapper that spawns a
# worker as a child, stays alive until terminated, and orphans that worker
# when only the wrapper itself is signalled.
w = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
print(w.pid, flush=True)
while True:
    time.sleep(0.1)
"""

# The crash-path twin: the launcher itself dies right after spawning the worker,
# so the worker is orphaned while the teardown still owes it a signal.
_LAUNCHER_THAT_DIES_AND_LEAVES_A_WORKER = """\
import subprocess, sys
w = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
print(w.pid, flush=True)
"""


@pytest.fixture
def launcher_factory(tmp_path):
    """Materialize a fake launcher script under tmp_path and spawn it exactly the
    way _spawn does: argument-list Popen (never a shell), pipes for the ACP wire,
    and the launcher leading its own process group on POSIX."""
    import subprocess as sp

    def _spawn(script: str):
        script_path = tmp_path / "fake_acp_launcher.py"
        script_path.write_text(script, encoding="utf-8")
        argv = [sys.executable, str(script_path)]
        popen_kwargs: dict = {"stdin": sp.PIPE, "stdout": sp.PIPE, "stderr": sp.PIPE}
        if sys.platform != "win32":
            popen_kwargs["process_group"] = 0
        return sp.Popen(argv, **popen_kwargs)

    return _spawn


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_terminate_process_takes_the_native_descendant_with_it(launcher_factory):
    """Terminating only the tracked launcher used to orphan the launcher's native
    child, which a container's PID 1 then never reaped — 16 leaked processes
    ≈ 2.6 GiB RSS in the reporter's deployment (#124835). The session spawn puts
    the tree in its own process group, so teardown must reach the descendant."""
    import contextlib
    import time as _time

    from agent.acp_client import ACPClient

    proc = launcher_factory(_LAUNCHER_WITH_NATIVE_CHILD)
    try:
        worker_pid = int(proc.stdout.readline().decode().strip())
        assert worker_pid > 0

        ACPClient._terminate_process(proc)

        deadline = _time.monotonic() + 5
        while _time.monotonic() < deadline:
            try:
                os.kill(worker_pid, 0)
            except OSError:
                break
            _time.sleep(0.05)  # health: allow HX013 -- polling for the descendant's death is the behaviour under test
        else:
            with contextlib.suppress(OSError):
                os.kill(worker_pid, 9)
            raise AssertionError("the native worker survived the launcher's termination")
    finally:
        with contextlib.suppress(Exception):
            proc.kill()
            proc.wait(timeout=5)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_terminate_process_never_signals_a_shared_process_group(monkeypatch):
    """A spawn that did not get its own process group must fall back to
    launcher-only termination: signalling the shared group would take unrelated
    processes down with the session (#124835)."""
    import agent.acp_client as mod

    from agent.acp_client import ACPClient

    signalled, terminated = [], []

    class _Proc:
        pid = 111

        def terminate(self):
            terminated.append(111)

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(mod.os, "name", "posix")
    monkeypatch.setattr(mod.os, "getpgid", lambda pid: 4242, raising=False)
    monkeypatch.setattr(mod.os, "killpg", lambda pgid, sig: signalled.append((pgid, sig)), raising=False)
    ACPClient._terminate_process(_Proc())  # type: ignore[arg-type]
    assert terminated == [111]
    assert signalled == []


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_terminate_process_sweeps_a_group_that_ignores_the_term(monkeypatch):
    """A descendant that ignores the graceful TERM must not outlive the session:
    after the launcher is reaped, a live group (pgid still claimed by a member)
    gets the KILL sweep (#124835)."""
    import signal as _signal

    import agent.acp_client as mod

    from agent.acp_client import ACPClient

    signalled, terminated = [], []

    class _Proc:
        pid = 111

        def terminate(self):
            terminated.append(111)

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(mod.os, "name", "posix")
    monkeypatch.setattr(mod.os, "getpgid", lambda pid: 111, raising=False)
    monkeypatch.setattr(mod.os, "killpg", lambda pgid, sig: signalled.append((pgid, sig)), raising=False)
    ACPClient._terminate_process(_Proc())  # type: ignore[arg-type]
    # TERM to the group, probe (signal 0), then the KILL sweep for the survivor.
    assert signalled == [(111, _signal.SIGTERM), (111, 0), (111, _signal.SIGKILL)]
    assert terminated == []  # the group path never degrades to launcher-only


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_terminate_process_signals_a_reaped_launchers_group(monkeypatch):
    """The crash path: the CLI died early, _session's poll() reaped it, and
    getpgid can no longer see the pid — but the group it led outlives it while
    any member lives. Bailing there would signal nothing at all (the exact
    #124835 leak); the sweep must fall back to the launcher's pid, which
    _spawn guarantees is the group id (#124835 review)."""
    import signal as _signal

    import agent.acp_client as mod

    from agent.acp_client import ACPClient

    signalled, terminated = [], []

    class _Proc:
        pid = 111

        def terminate(self):
            terminated.append(111)

        def wait(self, timeout=None):
            return 0  # already reaped: Popen returns immediately

        def kill(self):
            pass

    def _getpgid(pid):
        raise ProcessLookupError(pid)

    monkeypatch.setattr(mod.os, "name", "posix")
    monkeypatch.setattr(mod.os, "getpgid", _getpgid, raising=False)
    monkeypatch.setattr(mod.os, "killpg", lambda pgid, sig: signalled.append((pgid, sig)), raising=False)
    ACPClient._terminate_process(_Proc())  # type: ignore[arg-type]
    assert signalled == [(111, _signal.SIGTERM), (111, 0), (111, _signal.SIGKILL)]
    assert terminated == []


def test_terminate_process_sweeps_the_tree_on_windows(monkeypatch):
    """No process groups on Windows: a launcher TERM stops at the npm wrapper,
    so teardown must also run the taskkill /T /F tree sweep instead of leaving
    the native descendant running (#124835's Windows twin)."""
    from types import SimpleNamespace

    import agent.acp_client as mod
    import hermes_cli._subprocess_compat as compat

    from agent.acp_client import ACPClient

    terminated, swept = [], []

    class _Proc:
        pid = 222

        def terminate(self):
            terminated.append(222)

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    # A module-local os stand-in, NOT a global os.name patch: a global 'nt'
    # makes pathlib instantiate WindowsPath on POSIX hosts the moment a failed
    # assertion's repr touches Path, which turns a red test into an INTERNALERROR.
    monkeypatch.setattr(mod, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(compat, "kill_process_tree", lambda proc: swept.append(proc.pid))
    ACPClient._terminate_process(_Proc())  # type: ignore[arg-type]
    assert terminated == [222]  # graceful window first
    assert swept == [222]  # then the tree sweep


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_terminate_process_reaches_survivors_after_the_launcher_is_reaped(launcher_factory):
    """End-to-end crash path: the launcher exits on its own, poll() reaps it,
    and the native worker it spawned keeps running — getpgid can no longer see
    the pid. The teardown must still take the worker down; bailing on the
    ProcessLookupError is exactly the #124835 leak the tree-kill exists for."""
    import contextlib
    import time as _time

    from agent.acp_client import ACPClient

    proc = launcher_factory(_LAUNCHER_THAT_DIES_AND_LEAVES_A_WORKER)
    try:
        worker_pid = int(proc.stdout.readline().decode().strip())
        proc.wait(timeout=10)  # the launcher exits immediately...
        assert proc.poll() is not None  # ...and is reaped, like _session's poll()
        os.kill(worker_pid, 0)  # the worker outlives its dead launcher

        ACPClient._terminate_process(proc)

        deadline = _time.monotonic() + 5
        while _time.monotonic() < deadline:
            try:
                os.kill(worker_pid, 0)
            except OSError:
                break
            _time.sleep(0.05)  # health: allow HX013 -- polling for the descendant's death is the behaviour under test
        else:
            with contextlib.suppress(OSError):
                os.kill(worker_pid, 9)
            raise AssertionError("the reaped launcher's worker survived the teardown")
    finally:
        with contextlib.suppress(Exception):
            proc.kill()
            proc.wait(timeout=5)
