"""Files ACL, durable accepted copies and exact reservation evidence at real owner boundaries."""
import base64
import copy
import hashlib
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.hosted_room_documents import decode_batch, manifest
from gateway.hosted_room_driver import TaskIdentity
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.hosted_room_input_reclamation import collect_working_copies
from gateway.session_contract import SessionRef
from gateway.session_peer_documents import prepare, content
from hermes_state_runtime import RuntimeStoreError, admit_session_input
from tests.gateway.input_reclamation_fixtures import owned, close, rpc_files
from tui_gateway.hosted_room_driver import HostedRoomBinding
from tui_gateway.hosted_room_peer_documents import task_documents, transfer_documents


@pytest.mark.asyncio
async def test_source_acl_and_durable_manifest_precede_lazy_transfer(tmp_path, monkeypatch):
    db, owner = owned(tmp_path, monkeypatch)
    try:
        _, bound = rpc_files(tmp_path, owner, count=2)
        binding = HostedRoomBinding('room', 'home', 1)
        task = {'identity': TaskIdentity('room', 'task', 'thread', 'turn'), 'execution_generation': 1,
                'payload': {'target_member_id': 'member', 'target_profile': 'default', 'attachments': [i for i, _ in bound]}}
        inputs = task_documents(db.db_path, binding, task)
        wrong = copy.deepcopy(task)
        wrong['payload']['target_member_id'] = 'foreign'
        with pytest.raises(ValueError):
            task_documents(db.db_path, binding, wrong)
        # A retained exact attempt needs no source lookup just to replay acceptance.
        monkeypatch.setattr(HostedRoomAttachmentStore, 'describe', lambda *a, **kw: pytest.fail('source metadata reread'))
        assert task_documents(db.db_path, binding, task) == inputs
        changed = copy.deepcopy(task)
        changed['payload']['attachments'][0]['name'] = 'another.txt'
        with pytest.raises(ValueError, match='identity changed'):
            task_documents(db.db_path, binding, changed)
        dispatch = SimpleNamespace(room_id='room', member_id='member', document_inputs=inputs)
        encoded = transfer_documents(db.db_path, dispatch)
        assert [d['data'] for d in decode_batch(inputs, encoded)] == [d for _, d in bound]
        encoded[-1] = base64.b64encode(b'X' * bound[-1][0]['size']).decode()
        with pytest.raises(ValueError, match='digest'):
            decode_batch(inputs, encoded)
        malformed = copy.deepcopy(inputs)
        malformed[-1].update(kind='image', mime='image/png')
        with pytest.raises(ValueError, match='unsupported'):
            manifest(malformed, member_id='member')
        oversized = copy.deepcopy(inputs)
        oversized[-1]['size'] = 5_000_001
        with pytest.raises(ValueError, match='unsupported'):
            manifest(oversized, member_id='member')
    finally:
        close(db, tmp_path)


@pytest.mark.asyncio
@pytest.mark.parametrize("accept", [False, True])
async def test_verified_batch_admission_owns_copies_and_replay_never_reprepares(tmp_path, monkeypatch, accept):
    db, owner = owned(tmp_path, monkeypatch)
    try:
        _, bound = rpc_files(tmp_path, owner, count=2)
        inputs = [{**i, 'recipient_member_id': 'member', 'sha256': hashlib.sha256(data).hexdigest()} for i, data in bound]
        docs = decode_batch(manifest(inputs, member_id='member'), [base64.b64encode(data).decode() for _, data in bound])
        payload = {'text': 'review', 'api_turn_v1': {'history': None, 'run_owner_scope': 'a'*64,
            'settings': {'room_dispatch': {'member_id': 'member', 'document_inputs': inputs, 'task_id': 'task', 'execution_generation': 1}}}}
        prepared = prepare(owner, session_id='s', request_id='run', payload=payload, documents=docs)
        # A revocation after preparation cannot create accepted references.
        def refused(conn):
            raise RuntimeStoreError('permission_denied')
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            admit_session_input(db, epoch=owner.epoch, principal_id='api', session_id='s', request_id='run',
                payload=prepared.payload, input_custody=prepared.handle, _authorize_write=refused)
        assert db._conn.execute('SELECT COUNT(*) FROM input_custody_refs').fetchone()[0] == 0
        if not accept:
            db._execute_write(lambda conn: conn.execute('UPDATE input_custody_preparations SET expires_at=0'))
            assert collect_working_copies(db, epoch=owner.epoch)['removed'] == 2
            assert db._conn.execute('SELECT COUNT(*) FROM session_admissions').fetchone()[0] == 0
            return
        row = admit_session_input(db, epoch=owner.epoch, principal_id='api', session_id='s', request_id='run',
                                 payload=prepared.payload, input_custody=prepared.handle)
        replay = prepare(owner, session_id='s', request_id='run', payload=payload, documents=None)
        assert replay.payload == prepared.payload and replay.handle is None
        assert content(owner, SessionRef(owner.profile_id, 's'), replay.payload).startswith('review\n\nUse the file tools')
        changed = copy.deepcopy(payload)
        changed['api_turn_v1']['settings']['room_dispatch']['document_inputs'][0]['sha256'] = '0'*64
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            prepare(owner, session_id='s', request_id='run', payload=changed, documents=None)
        db._execute_write(lambda conn: conn.execute('UPDATE input_custody_preparations SET expires_at=0'))
        assert collect_working_copies(db, epoch=owner.epoch)['removed'] == 0
        assert db._conn.execute('SELECT COUNT(*) FROM session_admissions WHERE admission_id=?', (row['admission_id'],)).fetchone()[0] == 1
    finally:
        close(db, tmp_path)


@pytest.mark.asyncio
async def test_dead_unaccepted_reservation_can_resume_but_live_or_accepted_work_cannot(tmp_path, monkeypatch):
    from gateway.platforms.api_server_room_documents import recover_unaccepted, accepted_document_run
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
    from gateway.status import get_process_start_time
    db, owner = owned(tmp_path, monkeypatch)
    owner.runner.session_authority = owner
    store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    adapter = SimpleNamespace(gateway_runner=owner.runner, _run_idempotency_store=store)
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    try:
        started = int(get_process_start_time(child.pid))
        _, record = store.reserve('scope', 'key', 'fingerprint', 'run', {'status': 'queued'}, owner_pid=child.pid, owner_started=started)
        with pytest.raises(RuntimeStoreError, match='preparing'):
            recover_unaccepted(adapter, record, scope='scope', key='key', fingerprint='fingerprint', session_id='s')
        child.terminate(); child.wait(timeout=5)
        assert recover_unaccepted(adapter, record, scope='scope', key='key', fingerprint='fingerprint', session_id='s')
        _, record = store.reserve('scope', 'key', 'fingerprint', 'run', {'status': 'queued'}, owner_pid=child.pid, owner_started=started)
        dispatch = {'document_inputs': ['frozen identity'], 'task_id': 'task', 'execution_generation': 1}
        payload = {'text': 'x', 'api_turn_v1': {'run_owner_scope': 'scope', 'settings': {'room_dispatch': dispatch}}}
        admit_session_input(db, epoch=owner.epoch, principal_id='api', session_id='s', request_id='run', payload=payload)
        assert recover_unaccepted(adapter, record, scope='scope', key='key', fingerprint='fingerprint', session_id='s') is False
        # Losing only the HTTP reservation is not proof of nonadmission.
        store.forget('scope', 'key')
        assert accepted_document_run(adapter, run_id='run', session_id='s', dispatch=dispatch, scope='scope') == 'queued'
        with pytest.raises(RuntimeStoreError, match='conflict'):
            accepted_document_run(adapter, run_id='run', session_id='s', dispatch={'changed': True}, scope='scope')
        # Even damaged/retired live payload rows cannot erase canonical attempt evidence.
        db._execute_write(lambda conn: conn.execute("DELETE FROM session_admissions WHERE request_id='run'"))
        with pytest.raises(RuntimeStoreError, match='outcome_unknown'):
            accepted_document_run(adapter, run_id='run', session_id='s', dispatch=dispatch, scope='scope')
        with pytest.raises(RuntimeStoreError, match='outcome_unknown'):
            recover_unaccepted(adapter, record, scope='scope', key='key', fingerprint='fingerprint', session_id='s')
    finally:
        if child.poll() is None:
            child.kill(); child.wait(timeout=5)
        store.close()
        close(db, tmp_path)
