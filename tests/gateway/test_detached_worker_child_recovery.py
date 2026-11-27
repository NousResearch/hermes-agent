"""A surviving detached worker retains its child receipts across owner recovery."""
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.session_worker import worker_request
from gateway.session_worker_reservation import reserve_admission_worker
from persistence.core.errors import AuthorityRefusal
from tests.fixtures.storage_seed import seed_session
from tests.gateway.fixtures.storage_handles import storage_handle
from tests.gateway.test_worker_admission_reservation import hello


@pytest.fixture
def worker_process():
    process = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield process
    finally:
        process.stdin.close()
        process.wait(timeout=5)


@pytest.mark.platforms('posix')
@pytest.mark.asyncio
async def test_detached_child_continues_after_owner_recovery(tmp_path, worker_process):
    db = storage_handle(tmp_path)
    process = worker_process
    try:
        for sid in ('owned', 'unrelated'):
            seed_session(db, sid, 'cli')
        epoch = db.runtime.begin_epoch(instance_id='owner')
        authority = SimpleNamespace(epoch=epoch, profile_id=str(tmp_path), persistence=db,
                                    _require_admission_open=lambda: None, wake_after_worker=lambda sid: None,
                                    sessions={}, logical_owner=lambda sid: sid)
        actor = SimpleNamespace(subject='human', profile_id=str(tmp_path), capabilities={'worker:adopt'})
        connection = SimpleNamespace(authority=authority, actor=actor)
        admitted, _ = db.admissions.admit(epoch=epoch, principal_id='human', session_id='owned', request_id='input', payload={})
        db.admissions.claim(epoch=epoch, session_id='owned')
        root = reserve_admission_worker(authority, admission_id=admitted.admission_id, process=process,
                                        principal_id='human', hello=hello(process), detach_socket=str(tmp_path / 'listener'))
        await worker_request(connection, None, {k: v for k, v in root.items() if k != 'epoch'}, operation='adopt')
        secret = 'c' * 32
        reserved = await worker_request(connection, None, root | {'sequence': 1, 'operation': 'child.reserve',
                                                                 'payload': {'secret': secret}}, operation='persist')
        child = root | reserved['child'] | {'secret': secret}
        db.worker_executions.register(epoch=epoch, execution_id='other-worker', session_id='unrelated', generation=0,
                                      kind='compute', adoption_secret='other-secret')
        other = db.worker_executions.reserve_child(epoch=epoch, execution_id='other-worker', session_id='unrelated',
                    generation=0, sequence=1, secret='d' * 32, profile_id=str(tmp_path), pid=process.pid,
                    birth=root['birth'], principal='human', max_concurrent_children=10, max_spawn_depth=1).child
        authority.epoch = db.runtime.begin_epoch(instance_id='successor')
        db.admissions.recover(epoch=authority.epoch)
        assert db.admissions.get(admission_id=admitted.admission_id).owner_epoch == authority.epoch
        assert db.sessions.get(other.session_id).end_reason == 'agent_close'
        child['epoch'] = authority.epoch
        append = child | {'sequence': 1, 'operation': 'transcript.append',
                          'payload': {'messages': [{'role': 'assistant', 'content': 'continued'}]}}
        receipt = await worker_request(connection, None, append, operation='persist')
        assert await worker_request(connection, None, append, operation='persist') == receipt
        finish = child | {'sequence': 2, 'operation': 'execution.finish', 'payload': {}}
        finished = await worker_request(connection, None, finish, operation='persist')
        assert await worker_request(connection, None, finish, operation='persist') == finished
        assert [m.content for m in db.messages.read_transcript(child['session_id'])] == ['continued']
        with pytest.raises(AuthorityRefusal, match='stale_generation'):
            await worker_request(connection, None, append | {'sequence': 3}, operation='persist')
    finally:
        db.close()
