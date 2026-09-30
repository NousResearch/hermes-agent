"""Native containment tests; all processes and homes are disposable."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

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
    from hermes_cli.local_runtime.processes import spawn_server
    proc, job = spawn_server([sys.executable, __file__, str(root), 'router'],
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


@pytest.mark.platforms("windows")
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
                    from hermes_cli.local_runtime.processes import spawn_server
                    launcher, outer_job = spawn_server(cmd, env=env, stdout=log, stderr=log)
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


@pytest.mark.platforms("windows")
@pytest.mark.parametrize('failure', ['assign', 'resume', 'popen', 'configure'])
def test_failed_setup_never_runs_child_and_releases_handles(tmp_path, monkeypatch, failure):
    import ctypes
    from ctypes import wintypes
    from hermes_cli.local_runtime import processes

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
            processes.spawn_server(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
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


_POSIX_SCRIPT = r'''
import json, subprocess, sys, time
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
    from hermes_cli.local_runtime.processes import spawn_server
    proc, job = spawn_server([sys.executable, __file__, str(root), 'router'],
                             reap_with_owner=True)
    (root / 'ready').write_text('ready')
    while not ((root / 'stop_router').exists() or (root / 'exit_owner').exists()):
        time.sleep(.02)
    if (root / 'stop_router').exists():
        # A graceful stop: the owner (the router's parent) terminates it itself and
        # deliberately does NOT wait, so the corpse stays a zombie for a while.
        proc.terminate()
        time.sleep(90)
    # exit_owner: leave the router tree running and exit — the test's launcher
    # never waits, so this corpse stays a zombie instead of vanishing.
elif role == 'router':
    # The crash scenario needs a tree (router + grandchild) to verify the whole
    # tree dies; the graceful-stop scenario only watches the reaper vs the
    # zombie router, so the crash test plants the switch file before launch.
    if (root / 'with_grandchild').exists():
        subprocess.Popen([sys.executable, __file__, str(root), 'grandchild'])
    time.sleep(90)
else:
    time.sleep(90)
'''


def _dead_or_zombie(identity):
    """A terminated non-child cannot be waited out — only its own parent can reap
    the corpse — so zombie-or-gone is the strongest death observable from here."""
    try:
        proc = psutil.Process(identity['pid'])
        return proc.create_time() != identity['created'] or (
            proc.status() == psutil.STATUS_ZOMBIE or not proc.is_running())
    except psutil.NoSuchProcess:
        return True


def _zombie(identity):
    """The corpse still answers with its recorded identity — an unreaped exit,
    not a vanished or PID-reused stranger."""
    try:
        proc = psutil.Process(identity['pid'])
        return (proc.create_time() == identity['created']
                and proc.status() == psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


def _reaper_procs():
    return [p for p in psutil.process_iter(['cmdline', 'exe'])
            if p.info['cmdline'] and p.info['exe']
            and 'python' in p.info['exe'].lower()
            and 'hermes-owner-death-reaper' in ' '.join(p.info['cmdline'])]


def _launch(role, script, tmp_path, env, log):
    return subprocess.Popen([sys.executable, str(script), str(tmp_path), role],
                            env=env, stdout=log, stderr=log)


@pytest.mark.platforms("posix")
def test_owner_crash_reaps_router_tree_not_external(tmp_path):
    script = tmp_path / 'disposable posix server.py'
    script.write_text(_POSIX_SCRIPT)
    env = dict(os.environ, HERMES_HOME=str(tmp_path / 'home'),
               PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    (tmp_path / 'with_grandchild').write_text('tree')
    with (tmp_path / 'children.log').open('w') as log:
        launchers = [_launch(role, script, tmp_path, env, log)
                     for role in ('control', 'owner')]
        try:
            identities = {role: _read(tmp_path / f'{role}.json')
                          for role in ('owner', 'router', 'grandchild', 'control')}
            assert _wait((tmp_path / 'ready').exists)
            assert _wait(lambda: _reaper_procs()), 'owner-death reaper never armed'
            print('disposable identities:', identities)
            _kill(identities['owner'])
            for launcher in launchers:
                if launcher.poll() is None and launcher.pid == identities['owner']['pid']:
                    launcher.wait(timeout=10)
            # The sibling reaper takes the whole tree down within a polling cycle.
            assert _wait(lambda: _dead_or_zombie(identities['router'])
                         and _dead_or_zombie(identities['grandchild']), timeout=20), (
                'uncontained router/grandchild survived owner crash',
                identities['router'], identities['grandchild'])
            assert _wait(lambda: not _reaper_procs(), timeout=20), (
                'reaper lingered after reaping')
            assert _alive(identities['control']), 'unrelated external control was terminated'
        finally:
            for path in tmp_path.glob('*.json'):
                _kill(json.loads(path.read_text()))
            for launcher in launchers:
                if launcher.poll() is None:
                    launcher.kill()
                launcher.wait(timeout=10)


@pytest.mark.platforms("posix")
def test_reaper_exits_after_router_stops_but_owner_lives(tmp_path):
    script = tmp_path / 'disposable posix server.py'
    script.write_text(_POSIX_SCRIPT)
    env = dict(os.environ, HERMES_HOME=str(tmp_path / 'home'),
               PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    with (tmp_path / 'children.log').open('w') as log:
        owner = _launch('owner', script, tmp_path, env, log)
        try:
            identities = {role: _read(tmp_path / f'{role}.json')
                          for role in ('owner', 'router')}
            assert _wait((tmp_path / 'ready').exists)
            assert _wait(lambda: _reaper_procs()), 'owner-death reaper never armed'
            # A graceful stop takes the router down while the owner lives on
            # (the owner terminates it itself — the live-system guard rightly
            # forbids the test from signalling outside its own subtree);
            # the reaper must treat the unreaped zombie as gone and exit.
            (tmp_path / 'stop_router').write_text('stop')
            assert _wait(lambda: _dead_or_zombie(identities['router']))
            assert _wait(lambda: not _reaper_procs(), timeout=20), (
                'reaper lingered after a graceful router stop')
            assert _alive(identities['owner']), 'reaper terminated a living owner'
        finally:
            # The zombie router is reaped by the OS once its owner dies; the
            # owner itself is this test's launcher, so cleaning it up is enough.
            _kill(identities['owner'])
            if owner.poll() is None:
                owner.kill()
            owner.wait(timeout=10)


@pytest.mark.platforms("posix")
def test_reaper_harvests_tree_when_owner_lingers_as_zombie(tmp_path):
    # An owner whose parent never waits stays a zombie, and psutil keeps
    # reporting a zombie as running — so owner liveness must treat the corpse
    # as gone or the tree it exists to harvest is left running forever.
    script = tmp_path / 'disposable posix server.py'
    script.write_text(_POSIX_SCRIPT)
    env = dict(os.environ, HERMES_HOME=str(tmp_path / 'home'),
               PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    (tmp_path / 'with_grandchild').write_text('tree')
    with (tmp_path / 'children.log').open('w') as log:
        owner = _launch('owner', script, tmp_path, env, log)
        try:
            identities = {role: _read(tmp_path / f'{role}.json')
                          for role in ('owner', 'router', 'grandchild')}
            assert _wait((tmp_path / 'ready').exists)
            assert _wait(lambda: _reaper_procs()), 'owner-death reaper never armed'
            print('disposable identities:', identities)
            (tmp_path / 'exit_owner').write_text('exit')
            # Never wait() the launcher mid-test: the owner must linger as an
            # observable zombie rather than vanish, or the harvest below could
            # pass off a plain NoSuchProcess instead of the zombie branch.
            assert _wait(lambda: _zombie(identities['owner']), timeout=10), (
                'owner did not linger as an unreaped zombie', identities['owner'])
            assert _wait(lambda: _dead_or_zombie(identities['router'])
                         and _dead_or_zombie(identities['grandchild']), timeout=20), (
                'unreaped zombie owner kept its router tree running', identities)
            assert _wait(lambda: not _reaper_procs(), timeout=20), (
                'reaper lingered after reaping')
        finally:
            for path in tmp_path.glob('*.json'):
                _kill(json.loads(path.read_text()))
            if owner.poll() is None:
                owner.kill()
            owner.wait(timeout=10)
