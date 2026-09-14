"""Q's real preclaim consumer reads G custody; only reverse network is inert."""
from dataclasses import asdict
import hashlib
import json

import pytest

from gateway.hosted_room_driver import TaskIdentity
from gateway.hosted_room_input_preparation import prepare_hosted_input
from gateway.session_contract import Submission
from gateway import session_hosted_transport as transport
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from tests.gateway.input_reclamation_fixtures import owned, close, rpc_files, v3_path


@pytest.mark.asyncio
async def test_attested_preclaim_uses_v3_refs_without_recapture_and_keeps_storage_faults(tmp_path, monkeypatch):
    db, authority = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, authority)
        binding = {'source_home': str(tmp_path / 'source'), 'owner': 'source-owner',
                   'target_home': authority.profile_id, 'selector': {
                       'room_id': 'room', 'member_id': 'member', 'profile': 'default'}}
        rpc.principal = transport._principal(authority, binding)
        task = TaskIdentity('room', 'task', 'thread', 'turn')
        request_id = 'hosted:' + json.dumps([asdict(task), 1], sort_keys=True, separators=(',', ':'))
        prepared = prepare_hosted_input(rpc, request_id=request_id, prompt='read',
            attachments=[item for item, _ in bound])
        receipt = await authority.submit(rpc.principal, Submission(request_id, rpc.ref, prepared.payload, 'queue'),
            _input_custody=prepared.handle)
        row = get_session_admission(db, admission_id=receipt.admission_id)
        db._execute_write(lambda conn: conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)',
            (transport._BINDING + rpc.ref.session_id, json.dumps(binding))))
        attested = {'owner': binding['owner'], 'prompt': 'read',
            'attachments': [item for item, _ in bound],
            'attachment_digests': [hashlib.sha256(data).hexdigest() for _, data in bound]}
        calls = []
        def network(saved, operation, params):
            assert saved == binding and operation == 'execute'
            assert params == {'task': asdict(task), 'execution_generation': 1}
            calls.append(operation)
            return attested
        monkeypatch.setattr(transport, '_attest', network)
        def no_capture(*args, **kwargs):
            pytest.fail('preclaim recaptured source bytes')
        monkeypatch.setattr('gateway.hosted_room_input_preparation.resolve_inputs', no_capture)
        monkeypatch.setattr('gateway.hosted_room_input_preparation._copy', no_capture)
        assert transport._check_remote_hosted_admission(authority, rpc.ref, row) is True
        assert calls == ['execute']
        digest = attested['attachment_digests'][0]
        attested['attachment_digests'][0] = '0' * 64
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            transport._check_remote_hosted_admission(authority, rpc.ref, row)
        attested['attachment_digests'][0] = digest
        v3_path(prepared).write_bytes(b'corrupted private bytes')
        with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
            transport._check_remote_hosted_admission(authority, rpc.ref, row)
    finally:
        close(db, tmp_path)
