"""Already-public v2 migration and read-only retries, no source-closing operation."""
from dataclasses import asdict
import hashlib
import json
import os

import pytest

from gateway.hosted_room_driver import TaskIdentity
from gateway.hosted_room_input_custody import _backing_root, _READY
from gateway.hosted_room_input_preparation import reconstruct_accepted_payload, prepare_hosted_input
from gateway.hosted_room_input_reclamation import initialize_working_copies, collect_working_copies, collect_legacy_input_aliases
from gateway.session_ingress_media import _media_root
from hermes_state_runtime import admit_session_input, get_session_admission
from tests.gateway.input_reclamation_fixtures import owned, close, rpc_files, retire_metadata


def old_document(db, item, data):
    digest = hashlib.sha256(data).hexdigest()
    backing = _backing_root(db.db_path) / digest / item['name']
    backing.parent.mkdir(parents=True, exist_ok=True)
    backing.write_bytes(data)
    alias = _media_root() / digest / item['name']
    alias.parent.mkdir(parents=True, exist_ok=True)
    os.link(backing, alias)
    return backing, alias


@pytest.mark.asyncio
@pytest.mark.parametrize('named', [False, True])
async def test_v2_retry_is_readonly_and_only_startup_drains_old_alias(tmp_path, monkeypatch, named):
    home = tmp_path / 'profiles' / 'member' if named else tmp_path
    home.mkdir(parents=True, exist_ok=True)
    db, owner = owned(home, monkeypatch, initialize=False)
    try:
        rpc, bound = rpc_files(home, owner, named=named)
        item, data = bound[0]
        backing, alias = old_document(db, item, data)
        task = TaskIdentity('room', 'task', 'thread', 'turn')
        request_id = 'hosted:' + json.dumps([asdict(task), 1], sort_keys=True, separators=(',', ':'))
        payload = {'text': 'read\n[Shared attachment] file: ' + str(alias) + '\n'}
        row = admit_session_input(db, epoch=owner.epoch, principal_id=rpc.principal.subject, session_id=rpc.ref.session_id,
            request_id=request_id, payload=payload)
        with db._read_ctx() as conn:
            marker = conn.execute('SELECT value FROM state_meta WHERE key=?', (_READY,)).fetchone()[0]
        initialize_working_copies(db, epoch=owner.epoch)
        def no_copy(*args, **kwargs):
            raise AssertionError('v2 retry must remain read only')
        monkeypatch.setattr('gateway.hosted_room_input_preparation._copy', no_copy)
        receipt = await rpc._submit(dict(task=task, execution_generation=1, prompt='read',
            attachments=[item], on_terminal=lambda value: None))
        assert receipt['admission_id'] == row['admission_id']
        assert reconstruct_accepted_payload(rpc, 'read', [item], row) == payload
        assert get_session_admission(db, admission_id=row['admission_id'])['payload'] == payload
        assert collect_working_copies(db, epoch=owner.epoch)['removed'] == 0
        retire_metadata(db, row['admission_id'])
        assert collect_working_copies(db, epoch=owner.epoch)['removed'] == 1
        assert not backing.exists() and alias.read_bytes() == data
        initialize_working_copies(db, epoch=owner.epoch)
        assert collect_legacy_input_aliases(db, epoch=owner.epoch)['removed'] == 1
        assert not alias.exists()
        # Internal tombstone classification must not mint new copies. This does
        # not assert that the public RPC may resume a removed session target.
        retired = prepare_hosted_input(rpc, request_id=request_id, prompt='read', attachments=[item])
        assert retired.handle.admission_id == row['admission_id']
        with db._read_ctx() as conn:
            assert conn.execute('SELECT value FROM state_meta WHERE key=?', (_READY,)).fetchone()[0] == marker
            assert conn.execute("SELECT count(*) FROM input_custody_copies WHERE namespace='v3'").fetchone()[0] == 0
    finally:
        close(db, home)
