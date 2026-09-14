"""Digest-only execution preflight must retain the v3 accepted copy identity."""
import hashlib

import pytest

from gateway.hosted_room_driver import TaskIdentity
from gateway.hosted_room_input_preparation import reconstruct_accepted_payload
from hermes_state_runtime import RuntimeStoreError, get_session_admission
from tests.gateway.input_reclamation_fixtures import owned, close, rpc_files


@pytest.mark.asyncio
async def test_attested_preflight_uses_accepted_private_copy_without_source_transfer(tmp_path, monkeypatch):
    db, owner = owned(tmp_path, monkeypatch)
    try:
        rpc, bound = rpc_files(tmp_path, owner)
        item, data = bound[0]
        receipt = await rpc._submit(dict(task=TaskIdentity('room', 'task', 'thread', 'turn'),
            execution_generation=1, prompt='read', attachments=[item], on_terminal=lambda value: None))
        row = get_session_admission(db, admission_id=receipt['admission_id'])
        assert row is not None
        def no_source_read(*args, **kwargs):
            pytest.fail('execution preflight re-transferred source bytes')
        monkeypatch.setattr('gateway.hosted_room_input_preparation.resolve_inputs', no_source_read)
        digest = hashlib.sha256(data).hexdigest()
        assert reconstruct_accepted_payload(rpc, 'read', [item], row, source_digests=[digest]) == row['payload']
        with pytest.raises(RuntimeStoreError, match='admission_conflict'):
            reconstruct_accepted_payload(rpc, 'read', [item], row, source_digests=['0' * 64])
    finally:
        close(db, tmp_path)
