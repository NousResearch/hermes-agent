"""Worker identity is verified by the owner through launcher chains, never asserted by the worker.

On Windows the uv venv ``python.exe`` is a trampoline: ``Popen.pid`` is the launcher and the
real interpreter is its child. The owner reserves with the handle pid, the worker reports its
own pid/birth, and the OWNER proves that pid is alive, has that birth, and descends from the
reserved handle before binding the adoption digest to the real worker.
"""
import asyncio
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, claim_session_input
from tests.gateway.fixtures.local_recovery_probe import child_env, daemon, rpc, websocket

LAUNCHER = ('import subprocess, sys\n'
            'child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=sys.stdin)\n'
            'print(child.pid, flush=True)\n'
            'sys.exit(child.wait())\n')


def _hello(pid):
    proc = psutil.Process(pid)
    return {'type': 'hello', 'pid': pid, 'birth': proc.create_time(),
            'ancestors': [p.pid for p in proc.parents()[:4]]}


def test_owner_binds_reservation_to_verified_descendant_of_the_handle_only(tmp_path):
    from gateway.session_worker_reservation import reserve_admission_worker
    launcher = subprocess.Popen([sys.executable, '-c', LAUNCHER], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    stranger = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        real = int(launcher.stdout.readline())
        # One hop on POSIX; a venv trampoline (Windows) nests a second launcher in the chain.
        assert real != launcher.pid and launcher.pid in [p.pid for p in psutil.Process(real).parents()]
        with closing(SessionDB(tmp_path / 'state.db')) as db:
            db.create_session('owned', 'cli')
            epoch = begin_runtime_epoch(db, instance_id='owner')
            authority = SimpleNamespace(db=db, epoch=epoch, profile_id=str(tmp_path), _require_admission_open=lambda: None)
            admitted = admit_session_input(db, epoch=epoch, principal_id='human', session_id='owned', request_id='input', payload={})
            claim_session_input(db, epoch=epoch, session_id='owned')
            kwargs = dict(admission_id=admitted['admission_id'], process=launcher, principal_id='human')
            # Alive, truthful birth, same user — but a sibling, not a descendant of the reserved handle.
            with pytest.raises(Exception, match='permission_denied'):
                reserve_admission_worker(authority, hello=_hello(stranger.pid), **kwargs)
            # The right pid with a forged birth is a recycled-pid claim.
            with pytest.raises(Exception, match='permission_denied'):
                reserve_admission_worker(authority, hello={**_hello(real), 'birth': _hello(real)['birth'] + 1}, **kwargs)
            # A worker claiming the owner's own pid never adopts.
            with pytest.raises(Exception, match='permission_denied'):
                reserve_admission_worker(authority, hello=_hello(os.getpid()), **kwargs)
            assert db._read_one('SELECT COUNT(*) FROM worker_executions')[0] == 0
            scope = reserve_admission_worker(authority, hello=_hello(real), **kwargs)
            assert (scope['pid'], scope['birth']) == (real, psutil.Process(real).create_time())
            assert scope['pid'] != launcher.pid
    finally:
        # The real interpreter first, while it is still the launcher's child: killing the launcher
        # first reparents it outside the test subtree, where the conftest kill guard refuses it.
        if psutil.pid_exists(real):
            psutil.Process(real).kill()
        for proc in (launcher, stranger):
            proc.stdin.close()
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)


@pytest.mark.linux_only
def test_managed_worker_completes_through_interpreter_trampoline(tmp_path):
    """The ordinary daemon spawns every child through a launcher that execs the real interpreter as
    its child (uv's Windows python.exe shape). The turn still completes in the real worker and no
    launcher or worker survives the owner's cleanup."""
    root = Path(__file__).resolve().parents[2]
    home, user = tmp_path / 'state', tmp_path / 'user'
    home.mkdir(mode=0o700)
    user.mkdir()
    from http.server import ThreadingHTTPServer
    import threading
    from tests.gateway.fixtures.local_recovery_probe import Model
    peer = ThreadingHTTPServer(('127.0.0.1', 0), Model)
    peer.requests = []
    peer.blocked, peer.release = threading.Event(), threading.Event()
    threading.Thread(target=peer.serve_forever, daemon=True).start()
    url = f'http://127.0.0.1:{peer.server_port}/v1'
    launches = tmp_path / 'launches.jsonl'
    trampoline = tmp_path / 'python-launcher'
    trampoline.write_text(f'#!{sys.executable}\n'
        'import json, os, subprocess, sys\n'
        f'child = subprocess.Popen([{sys.executable!r}, *sys.argv[1:]])\n'
        f'with open({str(launches)!r}, "a", encoding="utf-8") as f:\n'
        '    f.write(json.dumps({"launcher": os.getpid(), "child": child.pid, "argv": sys.argv[1:]}) + "\\n")\n'
        'sys.exit(child.wait())\n', encoding='utf-8')
    trampoline.chmod(0o755)
    (home / 'config.yaml').write_text(json.dumps({
        'gateway': {'multiplex_profiles': False, 'managed_workers': True},
        'model': {'provider': 'custom', 'default': 'managed-model', 'base_url': url},
        'auxiliary': {'title_generation': {'enabled': False}}, 'platform_toolsets': {'cli': []}}), encoding='utf-8')
    env = child_env()
    env.update(HOME=str(user), USERPROFILE=str(user), HERMES_HOME=str(home), PYTHONPATH=str(root),
               OPENAI_API_KEY='loopback-only', OPENAI_BASE_URL=url, PYTHONUNBUFFERED='1', UGW_TRAMPOLINE=str(trampoline))

    def query(sql, args=()):
        with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
            return db.execute(sql, args).fetchall()

    async def exercise(desc, owner):
        async with websocket(home, desc) as ws:
            created = await rpc(ws, 'session.create', request_id='tramp', source='cli', cwd=str(home),
                                model='managed-model', provider='custom', base_url=url, api_key='loopback-only',
                                toolsets=[], ignore_rules=True)
            assert 'result' in created, created
            sid = created['result']['session_id']
            submitted = await rpc(ws, 'prompt.submit', session_id=sid, input_id='tramp-input', text='TRAMPOLINE_PROBE')
            assert 'result' in submitted, submitted
            async with asyncio.timeout(90):
                while query('SELECT status FROM session_admissions WHERE request_id=?', ('tramp-input',)) in ([('queued',)], [('started',)]):
                    await asyncio.sleep(.05)
            evidence = {'admissions': query('SELECT request_id,status FROM session_admissions'),
                        'workers': query('SELECT execution_id,status FROM worker_executions'),
                        'launches': launches.read_text(encoding='utf-8') if launches.exists() else None,
                        'gateway_log': (home / 'logs' / 'gateway.log').read_text(encoding='utf-8', errors='replace')[-3000:]}
            assert query('SELECT status FROM session_admissions WHERE request_id=?', ('tramp-input',)) == [('terminal',)], evidence
            restored = await rpc(ws, 'session.resume', session_id=sid)
            assert 'RECOVERY_ACK_TRAMPOLINE_PROBE' in json.dumps(restored['result']['messages']), restored
            assert query('SELECT status FROM worker_executions WHERE session_id=?', (sid,)) == [('terminal',)], evidence
            workers = [json.loads(l) for l in launches.read_text(encoding='utf-8').splitlines()
                       if json.loads(l)['argv'][-2:] == ['-m', 'agent.managed_worker']]
            assert len(workers) == 1 and workers[0]['launcher'] != workers[0]['child'], evidence
            # The launcher chain is fully reaped: neither the launcher nor the real worker survives
            # the owner's close (which runs after the admission commits terminal; bounded wait).
            def leftovers():
                return [(p.pid, p.cmdline()) for p in psutil.Process(owner.pid).children(recursive=True)
                        if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
            async with asyncio.timeout(15):
                while leftovers():
                    await asyncio.sleep(.1)
            assert leftovers() == []
            print(json.dumps({'owner_pid': owner.pid, 'launcher_pid': workers[0]['launcher'], 'worker_pid': workers[0]['child']}))

    try:
        with daemon(root, home, env, barrier=True, fixture='trampoline_daemon.py') as (owner, desc):
            asyncio.run(exercise(desc, owner))
    finally:
        peer.release.set()
        peer.shutdown()
        peer.server_close()
