"""Receipt recovery goes through the same live producer/principal claim as writes."""
import asyncio
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest
from gateway.session_contract import SessionRef
from gateway.session_worker import _claim, worker_request
from hermes_cli.process_identity import worker_birth
from hermes_state import SessionDB
from hermes_state_runtime import (RuntimeStoreError, begin_runtime_epoch, register_worker_execution,
                                  mutate_worker_execution)


def test_authenticated_receipt_read_preserves_all_claim_fences(tmp_path):
    worker = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.buffer.read()'],
                              stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
    try:
        with SessionDB(tmp_path / 'state.db') as db:
            db.create_session('assigned', source='cli')
            epoch = begin_runtime_epoch(db, instance_id='owner')
            authority = SimpleNamespace(db=db, epoch=epoch, profile_id='profile')
            actor = SimpleNamespace(profile_id='profile', subject='alice', capabilities={'worker:adopt'})
            connection = SimpleNamespace(authority=authority, actor=actor)
            ref = SessionRef('profile', 'assigned')
            scope = dict(profile_id='profile', session_id='assigned', execution_id='worker', generation=0,
                pid=worker.pid, birth=worker_birth(psutil.Process(worker.pid)), secret='synthetic-worker-secret')
            register_worker_execution(db, epoch=epoch, execution_id='worker', session_id='assigned', generation=0,
                                      kind='compute', adoption_secret=_claim(connection, ref, scope))
            mutation = dict(epoch=epoch, execution_id='worker', session_id='assigned', generation=0,
                            sequence=1, operation='usage.main', payload={'input_tokens': 1})
            result = mutate_worker_execution(db, **mutation)
            params = scope | mutation
            assert asyncio.run(worker_request(connection, ref, params, operation='receipt')) == result
            for changed in ({'secret': 'another-synthetic-secret'}, {'birth': scope['birth'] + 1},
                            {'pid': 1}, {'profile_id': 'foreign'}, {'generation': 3},
                            {'epoch': epoch + 1}, {'payload': {'input_tokens': 2}}, {'sequence': 2}):
                with pytest.raises(RuntimeStoreError):
                    asyncio.run(worker_request(connection, ref, params | changed, operation='receipt'))
            actor.subject = 'mallory'
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                asyncio.run(worker_request(connection, ref, params, operation='receipt'))
            actor.subject = 'alice'
            actor.capabilities = set()
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                asyncio.run(worker_request(connection, ref, params, operation='receipt'))
            assert db._read_one('SELECT COUNT(*) FROM worker_receipts')[0] == 1
            assert db.get_session('assigned')['input_tokens'] == 1
    finally:
        worker.terminate()
        worker.communicate(timeout=5)
