"""The managed Stop shortcut retains an automation FIFO fence after actual cleanup refusal."""
import asyncio
import json
import shlex
import subprocess
import sys

import pytest

from gateway.session_controls import AuthorityConnection
from hermes_state_runtime import admit_session_input, list_session_admissions
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit
from tests.waiting import async_wait_until


def _child(root):
    command = shlex.join([sys.executable, '-c', 'import threading; threading.Event().wait(60)'])
    return f'''import json, os, sys, psutil, threading
from types import SimpleNamespace
from hermes_cli.process_identity import worker_birth
from agent.managed_worker import WorkerControls
from tools.process_registry import process_registry
from pathlib import Path
me = psutil.Process()
out = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
def send(frame): out.write((json.dumps(frame) + '\\n').encode())
send({{'type':'hello','pid':me.pid,'birth':worker_birth(me),'ancestors':[p.pid for p in me.parents()[:3]]}})
controls = WorkerControls.__new__(WorkerControls)
controls.route, controls.stopped = 'route', threading.Event()
controls.stop_processes_requested = False
controls.lock, controls.task_checks, controls.clarifications = threading.Lock(), {{}}, {{}}
controls.agent = SimpleNamespace(session_id='s', hard_interrupt=lambda *a, **kw: None)
for line in sys.stdin.buffer:
    frame = json.loads(line)
    if frame.get('version') == 1:
        proc = process_registry.spawn_local({command!r}, cwd={str(root)!r}, task_id='s', owner_task_id='s',
                                            session_key='route', persist_on_release=True)
        Path({str(root / 'spawned.json')!r}).write_text(json.dumps({{'id':proc.id,'pid':proc.pid}}), encoding='utf-8')
        process_registry.kill_process = lambda *a, **kw: {{'status':'error','error':'fixture termination refusal'}}
        send({{'type':'ready','pid':me.pid}})
    elif frame == {{'type':'stop','processes':True}}:
        try: controls.stop(processes=True)
        except RuntimeError:
            send({{'type':'error','reason':'managed_worker_failed'}})
            break
'''


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
async def test_managed_stop_refusal_keeps_automation_unknown_and_followers_queued(tmp_path, monkeypatch):
    from gateway import session_finite, session_managed_worker as managed
    from gateway.session_ancillary import process_stop_pending
    from gateway.session_policy import build_policy
    from tools import process_registry as module
    db, authority = _authority(tmp_path, monkeypatch)
    authority.sessions['s'].source.chat_id = 'fixture-chat'
    policy = build_policy({'cwd': str(tmp_path), 'model': 'm'}, {}, profile_terminal=False)
    class Child(subprocess.Popen):
        def __init__(self, args, **kw):
            super().__init__([sys.executable, '-c', _child(tmp_path)], **kw)
    monkeypatch.setattr(managed.subprocess, 'Popen', Child)
    monkeypatch.setattr(managed, 'managed_policy', lambda *a: policy)
    registry = module.ProcessRegistry()
    monkeypatch.setattr(module, 'process_registry', registry)
    kill = registry.kill_process
    monkeypatch.setattr(registry, 'kill_process', lambda *a, **kw: {'status': 'error', 'error': 'fixture refusal'})
    executed = []

    async def execute(owner, ref, row):
        executed.append(row['request_id'])
        assert row['request_id'] == 'managed-automation', 'a follower started with an unconfirmed managed process'
        return await managed.execute_managed(owner, ref, row, policy)

    monkeypatch.setattr(session_finite, 'execute_finite_admission', execute)
    try:
        admit_session_input(db, epoch=authority.epoch, principal_id='automation:route', session_id='s',
                            request_id='managed-automation', payload={'text': 'work', 'local_automation_v1': {}})
        authority._schedule(REF)
        await async_wait_until(lambda: (tmp_path / 'spawned.json').exists(), timeout=30, what='real managed background job')
        await _submit(authority, 'follower')
        connection = type('Connection', (), {'authority': authority, 'actor': ACTOR})()
        await AuthorityConnection.interrupt(connection, REF,
            {'execution_generation': authority._handle(REF).execution_generation})
        await asyncio.wait_for(authority.sessions['s'].task, 20)
        await _submit(authority, 'later-input')
        await asyncio.wait_for(authority.sessions['s'].task, 20)
        statuses = {r['request_id']: r['status'] for r in list_session_admissions(db, session_id='s')}
        assert statuses == {'managed-automation': 'unknown', 'follower': 'queued', 'later-input': 'queued'}
        assert executed == ['managed-automation']
        assert process_stop_pending(db, 's') is not None
        process = registry.get(json.loads((tmp_path / 'spawned.json').read_text(encoding='utf-8'))['id'])
        assert process is not None and not process.exited
    finally:
        monkeypatch.setattr(registry, 'kill_process', kill)
        registry.recover_from_checkpoint()
        for process in list(registry._running.values()):
            kill(process.id)
        db.close()
