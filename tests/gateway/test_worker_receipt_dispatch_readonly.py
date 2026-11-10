"""Receipt dispatch must not restore/adopt even an unloaded session with catch-up due."""
import subprocess
import sys
from unittest.mock import AsyncMock, Mock

import psutil
import pytest

from gateway.session_controls import AuthorityConnection
from gateway.session_local import create_local_session
from gateway.session_local_adoption import _due
from gateway.session_worker import _claim
from hermes_cli.process_identity import worker_birth
from hermes_state_runtime import mutate_worker_execution, register_worker_execution
from tests.gateway.test_local_receipt_untracked_compression import (
    _compact_like_0_22_2, _owner,
)


@pytest.fixture
def live_worker():
    worker = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.buffer.read()'],
                              stdin=subprocess.PIPE, stdout=subprocess.DEVNULL)
    try:
        yield worker
    finally:
        worker.terminate()
        worker.communicate(timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize('invalid', [None, 'secret', 'principal', 'capabilities', 'epoch', 'payload'])
async def test_receipt_dispatch_never_adopts_or_writes(tmp_path, monkeypatch, invalid, live_worker):
    authority, owner = await _owner(tmp_path, monkeypatch, 'owner')
    ref = create_local_session(authority, owner.actor, {
        'request_id': 'receipt-read', 'source': 'tui', 'cwd': str(tmp_path), 'model': 'm', 'toolsets': []})
    connection = AuthorityConnection(authority, object(), {
        'user_id': 'owner', 'capabilities': ['worker:adopt']})
    worker = live_worker
    try:
        scope = dict(profile_id=authority.profile_id, session_id=ref.session_id,
                     execution_id='worker', generation=0, pid=worker.pid,
                     birth=worker_birth(psutil.Process(worker.pid)), secret='synthetic-worker-secret')
        register_worker_execution(authority.db, epoch=authority.epoch, execution_id='worker',
                                  session_id=ref.session_id, generation=0, kind='compute',
                                  adoption_secret=_claim(connection, ref, scope))
        mutation = dict(epoch=authority.epoch, execution_id='worker', session_id=ref.session_id,
                        generation=0, sequence=1, operation='usage.main', payload={'input_tokens': 1})
        expected = mutate_worker_execution(authority.db, **mutation)
        _compact_like_0_22_2(authority.db, ref.session_id, 'untracked-child')
        authority.sessions.pop(ref.session_id)
        assert _due(authority, ref.session_id) is not None
        params = scope | mutation
        params.update({
            'secret': {'secret': 'different-synthetic-secret'},
            'epoch': {'epoch': authority.epoch + 1},
            'payload': {'payload': {'input_tokens': 2}},
        }.get(invalid, {}))
        if invalid in {'principal', 'capabilities'}:
            await connection.close()
            connection = AuthorityConnection(authority, object(), {
                'user_id': 'intruder' if invalid == 'principal' else 'owner',
                'capabilities': [] if invalid == 'capabilities' else ['worker:adopt']})
        with authority.db._read_ctx() as conn:
            before = list(conn.iterdump())
        adoption = AsyncMock(side_effect=AssertionError('receipt invoked adoption'))
        write = Mock(side_effect=AssertionError('receipt invoked a write'))
        monkeypatch.setattr('gateway.session_local_adoption.adopt_before_restore', adoption)
        monkeypatch.setattr(authority.db, '_execute_write', write)
        reply = await connection.dispatch({'jsonrpc': '2.0', 'id': 1, 'method': 'worker.receipt', 'params': params})
        if invalid is None:
            assert reply['result'] == expected
        else:
            assert reply['error']['code'] == 4001
        adoption.assert_not_called()
        write.assert_not_called()
        with authority.db._read_ctx() as conn:
            assert list(conn.iterdump()) == before
        assert ref.session_id not in authority.sessions
        assert _due(authority, ref.session_id) is not None
    finally:
        await connection.close()
        await owner.close()
        authority.db.close()
