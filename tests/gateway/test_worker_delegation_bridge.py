"""Authenticated worker ledger bridge publishes committed owner events."""
import asyncio
import json
import os
import queue
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from hermes_cli.process_identity import worker_birth
from persistence_assembly import open_persistence, profile_scope
from persistence.core.errors import AuthorityRefusal
from tests.fixtures.storage_holder import raw_holder
from tests.fixtures.storage_seed import seed_session
from gateway.session_contract import SessionRef
from gateway.session_worker import _claim


def test_bridge_publishes_committed_completion_and_rejects_another_principal(tmp_path, monkeypatch):
    from gateway.session_worker_delegation import worker_delegation_request
    from tools.process_registry import process_registry
    target = queue.Queue()
    monkeypatch.setattr(process_registry, 'completion_queue', target)
    handle = open_persistence(profile_scope(home=tmp_path))
    worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        seed_session(handle, 'parent', 'cli')
        epoch = handle.runtime.begin_epoch(instance_id='owner')
        authority = SimpleNamespace(epoch=epoch, profile_id='profile', _require_admission_open=lambda: None,
                                    persistence=handle)
        actor = SimpleNamespace(profile_id='profile', subject='alice', capabilities={'worker:adopt'})
        connection = SimpleNamespace(authority=authority, actor=actor)
        ref = SessionRef('profile', 'parent')
        scope = dict(profile_id='profile', session_id='parent', execution_id='worker', generation=0,
                     pid=worker.pid, birth=worker_birth(psutil.Process(worker.pid)), secret='private-worker-secret')
        claim = _claim(connection, ref, scope)
        handle.worker_executions.register(epoch=epoch, execution_id='worker', session_id='parent', generation=0,
                                          kind='compute', adoption_secret=claim)
        payload = dict(delegation_id='deleg_owned', task={'goal': 'inert'}, dispatched_at=time.time(),
                       parent_session_id='parent', session_key='', origin_ui_session_id='', origin_session_id='')
        params = scope | dict(epoch=epoch, sequence=1, operation='delegation.dispatch', payload=payload)
        asyncio.run(worker_delegation_request(connection, ref, params))
        event = dict(type='async_delegation', delegation_id='deleg_owned', parent_session_id='parent',
            session_key='', origin_ui_session_id='', origin_session_id='', status='completed',
            completed_at=time.time(), summary='durable')
        params = scope | dict(epoch=epoch, sequence=2, operation='delegation.complete',
                              payload={'event': event, 'result': {'summary': 'durable'}})
        response = asyncio.run(worker_delegation_request(connection, ref, params))
        assert response == {'value': None}
        assert target.get_nowait() == event | {'restored': True}
        assert asyncio.run(worker_delegation_request(connection, ref, params)) == response
        assert target.get_nowait() == event | {'restored': True}
        with raw_holder(handle)._read_ctx() as conn:
            row = conn.execute('SELECT event_json FROM async_delegations').fetchone()
            assert json.loads(row[0]) == event
            assert conn.execute('SELECT COUNT(*) FROM worker_receipts').fetchone()[0] == 2
        actor.subject = 'mallory'
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            asyncio.run(worker_delegation_request(connection, ref, params))
        assert target.empty()
    finally:
        worker.terminate()
        worker.wait(timeout=5)
        handle.close()
