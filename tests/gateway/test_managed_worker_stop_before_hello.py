"""Stop is never acknowledged into a queue nobody drains: a managed child that stays alive
without ever sending hello is terminated on Stop and the admission settles as interrupted."""
import asyncio
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest

from gateway.session_contract import Principal, SessionRef


@pytest.mark.platforms("linux")
def test_stop_before_hello_terminates_child_and_settles_interrupted(monkeypatch):
    from gateway import session_managed_worker as managed
    spawned = []
    original = subprocess.Popen

    def wedged_child(args, **kwargs):
        # An interpreter that starts but never introduces itself (loader/stdio stall).
        child = original([sys.executable, '-c', 'import time\nwhile True: time.sleep(1)'], **kwargs)
        spawned.append(child)
        return child
    monkeypatch.setattr(managed.subprocess, 'Popen', wedged_child)
    ref = SessionRef('profile', 'session')
    checks = []
    authority = SimpleNamespace(profile_id='profile', pending_results={}, waiters={},
        adopt_agent=lambda session_id, generation, worker: None,
        authorize=lambda actor, ref, cap: checks.append(cap),
        check_approval_generation=lambda session_id, generation: checks.append(generation))
    row = {'admission_id': 'adm', 'principal_id': 'owner', 'generation': 3, 'payload': {'text': 'go'}}
    actor = Principal('owner', 'profile', frozenset({'session:control'}), 'transport')

    async def scenario():
        turn = asyncio.create_task(managed.execute_managed(authority, ref, row, policy=None))
        async with asyncio.timeout(10):
            while ref.session_id not in getattr(authority, '_managed_workers', {}):
                await asyncio.sleep(.01)
        assert managed.interrupt_managed(authority, actor, ref, 3) is True
        async with asyncio.timeout(10):
            return await turn
    assert asyncio.run(scenario()) == ''
    assert checks == ['session:control', 3]
    assert authority.pending_results['adm']['result']['interrupted'] is True
    assert ref.session_id not in authority._managed_workers
    child = spawned[0]
    assert child.poll() is not None and not psutil.pid_exists(child.pid) or psutil.Process(child.pid).status() == psutil.STATUS_ZOMBIE


CHILD = """import json, os, sys, psutil
me = psutil.Process()
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
def send(frame):
    out.write((json.dumps(frame) + '\\n').encode())
send({'type': 'hello', 'pid': me.pid, 'birth': me.create_time(), 'ancestors': [p.pid for p in me.parents()[:3]]})
for line in sys.stdin.buffer:
    frame = json.loads(line)
    with open(sys.argv[1], 'a') as log:
        log.write(json.dumps(frame.get('type', 'bootstrap')) + '\\n')
    if 'version' in frame:
        send({'type': 'result', 'result': {'final_response': 'LATE_TURN'}})
    elif frame == {'type': 'finish'}:
        send({'type': 'finished'})
        break
"""


@pytest.mark.platforms("linux")
def test_stop_acknowledged_during_spawn_never_bootstraps_the_worker(tmp_path, monkeypatch):
    """R2: a Stop the real interrupt RPC acknowledges while ``_worker_env``/``Popen`` is awaited
    (no worker registered yet, so the authority latches it for the generation) must keep the child
    from ever receiving its bootstrap; the admission settles interrupted, never as a late turn."""
    import threading
    from gateway import session_managed_worker as managed
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import Submission
    from gateway.session_controls import AuthorityConnection
    from gateway.session_policy import build_policy
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('s', source='test')
    runner = SimpleNamespace(_draining=False, _cached_agent_for=lambda route: None,
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='i', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='i'))
    authority.sessions['s'] = LiveSession(SimpleNamespace(platform=None, user_id='human', chat_id='c'), 'route')
    actor = Principal('human', 'owned', frozenset({'session:submit', 'session:control'}), 'cli')
    ref = SessionRef('owned', 's')
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    received = tmp_path / 'received.jsonl'

    class Child(subprocess.Popen):
        def __init__(self, args, **kw):
            super().__init__([sys.executable, '-c', CHILD, str(received)], **kw)
    monkeypatch.setattr(managed.subprocess, 'Popen', Child)
    entered, release = threading.Event(), threading.Event()

    def barrier(authority):
        entered.set()
        assert release.wait(10)
    monkeypatch.setattr(managed, '_worker_env', barrier)
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {}, profile_terminal=False)

    async def scenario():
        await authority.submit(actor, Submission('turn', ref, {'text': 'go'}, 'queue'))
        row = claim_session_input(db, epoch=authority.epoch, session_id='s')
        turn = asyncio.create_task(managed.execute_managed(authority, ref, row, policy))
        assert await asyncio.to_thread(entered.wait, 10)
        stopped = await AuthorityConnection.interrupt(SimpleNamespace(authority=authority, actor=actor), ref,
                                                      {'execution_generation': row['generation']})
        assert stopped['execution_state'] == 'running'
        release.set()
        async with asyncio.timeout(20):
            return row, await turn
    with db:
        row, response = asyncio.run(scenario())
    assert response == ''
    assert authority.pending_results[row['admission_id']]['result']['interrupted'] is True
    assert not received.exists() or 'bootstrap' not in received.read_text(), received.read_text()
    assert 's' not in authority.pending_stops
