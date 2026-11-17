"""``gateway.detached_workers``, worker side: a detachable managed worker outlives its owner, buffers
its frames and hands them to the next same-user owner over its private socket (real processes).
Without ``detach_v1`` it still dies with its owner: test_managed_worker_parent_death.py."""
import contextlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import shutil
import time

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[2]

# A worker past bootstrap whose turn keeps producing frames after its owner is gone.
WORKER = textwrap.dedent('''
    import os, sys, threading, time
    import agent.managed_worker as mw
    from agent.managed_worker_detach import DetachableChannel
    pipe = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
    channel = DetachableChannel(pipe, sys.argv[1], {'pid': os.getpid(), 'birth': 1.5,
                                                    'execution_id': 'admission-worker:x', 'generation': 3})
    controls = mw.WorkerControls(channel, route='r')
    controls.agent = type('Agent', (), {'interrupt': lambda self, *a, **k: None})()
    channel.send('ready', pid=os.getpid())
    for n in range(40):
        time.sleep(.25)
        channel.send('delta', text=f'tick-{n}')
    channel.send('result', result={'final_response': 'SURVIVED'})
    controls.finish.wait(60)
    channel.send('finished')
''')

OWNER = textwrap.dedent('''
    import subprocess, sys
    from gateway.session_managed_worker import WORKER_POPEN
    worker = subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, **WORKER_POPEN)
    print(worker.pid, worker.stdout.readline().decode().strip(), flush=True)
    worker.wait()
''')


def _alive(pid):
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _frames(reader):
    while True:
        line = reader.readline()
        if not line:
            return
        yield json.loads(line)


@pytest.mark.platforms("linux", "macos")
@pytest.mark.live_system_guard_bypass  # cleanup may signal our own descendant after init adopted it
def test_detached_worker_outlives_its_owner_and_reattaches_to_the_next_one():
    import agent.managed_worker as mw
    from gateway.control_socket import _private_runtime_dir
    short = tempfile.mkdtemp(prefix='hmw-', dir=_private_runtime_dir() or tempfile.gettempdir())  # sun_path limit
    path = os.path.join(short, 'w.sock')
    env = {**os.environ, 'PYTHONPATH': str(ROOT)}
    owner = subprocess.Popen([sys.executable, '-c', OWNER, WORKER, path], cwd=ROOT, env=env,
                             stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
    worker_pid = None
    try:
        first = owner.stdout.readline().split(maxsplit=1)
        worker_pid = int(first[0])
        assert json.loads(first[1]) == {'type': 'ready', 'pid': worker_pid}
        owner.kill()
        owner.wait(timeout=5)
        time.sleep(mw.ORPHAN_GRACE_SECONDS + 2)  # health: allow HX013 -- outliving the legacy grace IS the behaviour
        assert _alive(worker_pid), 'the detachable worker died with its owner'
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(30)
            conn.connect(path)
            reader = conn.makefile('rb')
            frames = _frames(reader)
            assert next(frames) == {'type': 'hello', 'pid': worker_pid, 'birth': 1.5,
                                    'execution_id': 'admission-worker:x', 'generation': 3}
            conn.sendall(b'{"type":"reattached","epoch":2}\n')
            seen = []
            for frame in frames:
                seen.append(frame)
                if frame['type'] == 'result':
                    conn.sendall(b'{"type":"finish"}\n')
                if frame['type'] == 'finished':
                    break
        ticks = [f['text'] for f in seen if f['type'] == 'delta']
        assert ticks and ticks[-1] == 'tick-39'  # buffered while orphaned, then live, none lost at the end
        assert {'type': 'result', 'result': {'final_response': 'SURVIVED'}} in seen
        assert seen[-1] == {'type': 'finished'}
    finally:
        owner.stdout.close()
        if owner.poll() is None:
            owner.kill()
            owner.wait(timeout=5)
        if worker_pid is not None and _alive(worker_pid):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(worker_pid, signal.SIGKILL)  # windows-footgun: ok — POSIX-only test
        shutil.rmtree(short, ignore_errors=True)
