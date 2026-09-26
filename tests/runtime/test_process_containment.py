"""Native containment tests; all processes and homes are disposable."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest


_SCRIPT = r'''
import json, os, subprocess, sys, time
from pathlib import Path
import psutil
root = Path(sys.argv[1])
role = sys.argv[2]
def record(name):
    p = psutil.Process()
    target = root / (name + '.json')
    temp = target.with_suffix('.tmp')
    temp.write_text(json.dumps({'pid': p.pid, 'created': p.create_time()}))
    temp.replace(target)
record(role)
if role == 'owner':
    from runtime.processes import spawn_contained_process
    proc, job = spawn_contained_process([sys.executable, __file__, str(root), 'router'],
                             close_fds=False)
    (root / 'ready').write_text('ready')
    while not (root / 'stop').exists():
        time.sleep(.02)
    job.close()
    job.close()
    proc.wait(timeout=10)
    (root / 'closed').write_text('closed')
    while not (root / 'exit').exists():
        time.sleep(.02)
elif role == 'router':
    subprocess.Popen([sys.executable, __file__, str(root), 'grandchild'])
    time.sleep(90)
else:
    time.sleep(90)
'''


def _wait(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(.03)
    return bool(predicate())


def _read(path):
    assert _wait(path.exists), f"missing child receipt: {path}"
    return json.loads(path.read_text())


def _alive(identity):
    try:
        proc = psutil.Process(identity['pid'])
        if proc.create_time() != identity['created']:
            return False
        try:
            proc.wait(timeout=0)
            return False
        except psutil.TimeoutExpired:
            return True
    except psutil.NoSuchProcess:
        return False


def _kill(identity):
    if _alive(identity):
        proc = psutil.Process(identity['pid'])
        proc.kill()
        proc.wait(timeout=10)


@pytest.mark.windows_only
@pytest.mark.parametrize('stop_mode', ['graceful', 'abrupt'])
@pytest.mark.parametrize('nested', [False, True])
def test_owner_exit_kills_router_tree_not_external(tmp_path, stop_mode, nested):
    script = tmp_path / 'disposable server.py'
    script.write_text(_SCRIPT)
    env = dict(os.environ, HERMES_HOME=str(tmp_path / 'home'),
               PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    launchers = []
    outer_job = None
    with (tmp_path / 'children.log').open('w') as log:
        try:
            for role in ('control', 'owner'):
                cmd = [sys.executable, str(script), str(tmp_path), role]
                if nested and role == 'owner':
                    from runtime.processes import spawn_contained_process
                    launcher, outer_job = spawn_contained_process(cmd, env=env, stdout=log, stderr=log)
                else:
                    launcher = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
                launchers.append(launcher)
            owner = _read(tmp_path / 'owner.json')
            router = _read(tmp_path / 'router.json')
            grandchild = _read(tmp_path / 'grandchild.json')
            control = _read(tmp_path / 'control.json')
            assert _wait((tmp_path / 'ready').exists)
            assert all(_alive(i) for i in (owner, router, grandchild, control))
            print('disposable identities:', stop_mode, owner, router, grandchild, control)
            if stop_mode == 'graceful':
                (tmp_path / 'stop').write_text('stop')
                assert _wait((tmp_path / 'closed').exists), 'close killed the owner itself'
                assert _alive(owner)
                assert _wait(lambda: not _alive(router) and not _alive(grandchild))
                assert _alive(control)
                (tmp_path / 'exit').write_text('exit')
            else:
                _kill(owner)  # The venv launcher is not necessarily the owner.
            assert _wait(lambda: not _alive(owner))
            assert _wait(lambda: not _alive(router) and not _alive(grandchild)), (
                'uncontained router/grandchild survived owner exit', router, grandchild)
            assert _alive(control), 'unrelated external control was terminated'
        finally:
            if outer_job is not None:
                outer_job.close()
            # Stop only identities/descendants created by this test, including red runs.
            descendants = []
            for launcher in launchers:
                try:
                    descendants.extend(psutil.Process(launcher.pid).children(recursive=True))
                except psutil.NoSuchProcess:
                    pass
            for path in tmp_path.glob('*.json'):
                _kill(json.loads(path.read_text()))
            for proc in reversed(descendants):
                _kill({'pid': proc.pid, 'created': proc.create_time()})
            for launcher in launchers:
                if launcher.poll() is None:
                    launcher.kill()
                launcher.wait(timeout=10)





def test_spawn_contained_process_preserves_caller_kwargs_off_windows(monkeypatch):
    """Non-Windows spawning is a transparent Popen pass-through."""
    from runtime import processes

    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr(processes, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(processes.subprocess, "Popen", fake_popen)
    kwargs = {
        "env": {"TOKEN": "keep", "PATH": "/bin"},
        "cwd": "/tmp/work",
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
    }

    proc, job = processes.spawn_contained_process(["git", "status"], **kwargs)

    assert proc is not None
    assert job is None
    assert captured == {"cmd": ["git", "status"], "kwargs": kwargs}


def test_windows_spawn_ors_creationflags_and_assigns_before_resume(monkeypatch):
    """Caller flags survive, and a Windows child is resumed only after job assignment."""
    from runtime import processes

    events = []
    captured = {}

    class FakeProc:
        pid = 123
        stdin = stdout = stderr = None

    class FakeJob:
        def assign(self, proc):
            events.append(("assign", proc.pid))

        def close(self):
            events.append(("close", None))

    class FakePsProcess:
        def __init__(self, pid):
            self.pid = pid

        def resume(self):
            events.append(("resume", self.pid))

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        events.append(("popen", None))
        return FakeProc()

    fake_psutil = SimpleNamespace(Process=FakePsProcess)
    monkeypatch.setattr(processes, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(processes, "_WindowsJob", FakeJob)
    monkeypatch.setattr(processes.subprocess, "Popen", fake_popen)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)

    proc, job = processes.spawn_contained_process(
        ["child"],
        creationflags=0x08000000,
        env={"X": "1"},
    )

    assert proc.pid == 123
    assert isinstance(job, FakeJob)
    assert captured["kwargs"]["creationflags"] == 0x08000004
    assert captured["kwargs"]["env"] == {"X": "1"}
    assert events == [("popen", None), ("assign", 123), ("resume", 123)]


def test_importing_runtime_processes_does_not_import_psutil():
    """The generic runtime module stays lightweight until a Windows child is spawned."""
    repo_root = Path(__file__).resolve().parents[2]
    code = (
        "import sys; "
        "sys.modules.pop('psutil', None); "
        "import runtime.processes; "
        "assert 'psutil' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


@pytest.mark.windows_only
def test_windows_job_close_is_idempotent():
    """Closing an owner-held child-containment job twice is harmless."""
    from runtime.processes import _WindowsJob

    job = _WindowsJob()
    job.close()
    job.close()
    assert job._handle is None


@pytest.mark.windows_only
@pytest.mark.parametrize('failure', ['assign', 'resume', 'popen', 'configure'])
def test_failed_setup_never_runs_child_and_releases_handles(tmp_path, monkeypatch, failure):
    import ctypes
    from ctypes import wintypes
    from runtime import processes

    marker = tmp_path / 'child executed'
    jobs, children, handles = [], [], []
    real_init = processes._WindowsJob.__init__
    real_assign = processes._WindowsJob.assign

    def track_job(job):
        jobs.append(job)
        real_init(job)
        handles.append(job._handle)

    if failure == 'configure':
        real_dll = ctypes.WinDLL

        def failed_config_dll(*args, **kwargs):
            api = real_dll(*args, **kwargs)

            def fail_config(handle, *args):
                handles.append(handle)
                raise KeyboardInterrupt('injected cancellation configuring job')

            api.SetInformationJobObject = fail_config
            return api

        monkeypatch.setattr(ctypes, 'WinDLL', failed_config_dll)

    def assign(job, proc):
        children.append(proc)
        assert psutil.Process(proc.pid).status() == psutil.STATUS_STOPPED
        assert not marker.exists()
        # Query the actual kernel object, not implementation source/constants.
        limits = processes._ExtendedLimits()
        query = job._api.QueryInformationJobObject
        query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                          wintypes.DWORD, ctypes.c_void_p]
        query.restype = wintypes.BOOL
        assert query(job._handle, 9, ctypes.byref(limits), ctypes.sizeof(limits), None)
        flags = limits.BasicLimitInformation.LimitFlags
        assert flags & 0x2000  # KILL_ON_JOB_CLOSE
        assert not flags & (0x800 | 0x1000)  # Neither breakaway limit.
        info = job._api.GetHandleInformation
        info.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        info.restype = wintypes.BOOL
        inherited = wintypes.DWORD()
        assert info(job._handle, ctypes.byref(inherited))
        assert not inherited.value & 1  # HANDLE_FLAG_INHERIT
        if failure == 'assign':
            # Exercise a real WinAPI rejection while the child is suspended.
            assert not job._api.AssignProcessToJobObject(job._handle, None)
            raise ctypes.WinError(ctypes.get_last_error())
        real_assign(job, proc)

    def fail_resume(self):
        raise KeyboardInterrupt('injected cancellation before resume')

    monkeypatch.setattr(processes._WindowsJob, '__init__', track_job)
    monkeypatch.setattr(processes._WindowsJob, 'assign', assign)
    monkeypatch.setattr(psutil.Process, 'resume', fail_resume)
    cmd = ([str(tmp_path / 'missing.exe')] if failure == 'popen' else
           [sys.executable, '-c', 'from pathlib import Path; import sys; '
            'Path(sys.argv[1]).write_text("ran")', str(marker)])
    try:
        with pytest.raises((OSError, KeyboardInterrupt)):
            processes.spawn_contained_process(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        assert not marker.exists()
        assert jobs and jobs[0]._handle is None
        for proc in children:
            assert proc.poll() is not None, 'failed setup left a live suspended child'
            assert all(stream.closed for stream in (proc.stdin, proc.stdout, proc.stderr))
            assert proc._handle.closed, 'failed setup leaked the Popen process handle'
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.GetHandleInformation.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        api.GetHandleInformation.restype = wintypes.BOOL
        for handle in handles:
            flags = wintypes.DWORD()
            assert not api.GetHandleInformation(handle, ctypes.byref(flags))
    finally:
        for proc in children:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                stream.close()
            proc._handle.Close()
        for job in jobs:
            job.close()
