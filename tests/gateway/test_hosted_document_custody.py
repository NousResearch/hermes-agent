"""Real Files materialization keeps exact accepted document paths across native GC."""
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_room_input_custody, session_hosted_attachments
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.hosted_room_driver import TaskIdentity
from gateway.session_authority import LiveSession
from gateway.session_contract import Principal
from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
from gateway.session_ingress_media import capture_native_media, release_admission_media, restore_native_media
from hermes_state_runtime import admit_session_input, claim_session_input, get_session_admission, list_session_admissions
from hermes_state_runtime import settle_session_input
from tests.gateway.test_api_media_retention import set_holder_status
from tests.gateway.test_native_media_budget import _authority
from tests.gateway.hosted_document_custody_helpers import local_documents


def _candidate(home, authority, name, data):
    source = home / 'native-source' / name
    source.parent.mkdir(exist_ok=True)
    source.write_bytes(data)
    references = capture_native_media([source])
    session_id = 'native-' + name
    authority.db.create_session(session_id, source='telegram')
    row = admit_session_input(authority.db, epoch=authority.epoch, principal_id='native-person',
        session_id=session_id, request_id='native-original',
        payload={'text': 'native', 'native_text_v1': {'media': references}})
    started = claim_session_input(authority.db, epoch=authority.epoch, session_id=session_id)
    settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'],
        generation=started['generation'], outcome='completed')
    return row, references


@pytest.mark.asyncio
@pytest.mark.parametrize('transferred', [False, True], ids=['local', 'named-owner-snapshot'])
@pytest.mark.parametrize('status', ['queued', 'unknown', 'terminal'])
@pytest.mark.parametrize('witness_available', [True, False], ids=['recorded-custody', 'deferred-custody'])
async def test_materialization_handoff_and_admission_keep_exact_bytes_and_retry(
        tmp_path, monkeypatch, transferred, status, witness_available):
    home = tmp_path / 'profiles' / 'member' if transferred else tmp_path
    home.mkdir(parents=True, exist_ok=True)
    db, authority = _authority(home, monkeypatch)
    with db:
        prepared, bound = local_documents(home, transferred=transferred)
        candidate, references = _candidate(home, authority, '0.txt', b'A' * 2048)
        principal = Principal('human', authority.profile_id,
            frozenset({'session:read', 'session:submit', 'session:control'}), 'fixture-private-owner')
        loop = asyncio.get_running_loop()
        rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='member', profile='default',
            principal=principal, authorize=lambda *args: True)
        db.create_session(rpc.ref.session_id, source='cli')
        authority.sessions[rpc.ref.session_id] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'fixture')
        if transferred:
            rpc.hosted_attachment_data = prepared.hosted_attachment_data
            def no_foreign_lookup(*args, **kwargs):
                raise AssertionError('named materialization must use the borrowed source snapshot')
            monkeypatch.setattr(HostedRoomAttachmentStore, 'read', no_foreign_lookup)
        if not witness_available:
            def unavailable(*args):
                raise OSError('fixture bookkeeping unavailable')
            monkeypatch.setattr(hosted_room_input_custody, 'record_admission_custody', unavailable)

        materialize = session_hosted_attachments.submission_payload
        observed = []

        def cleanup_before_admission():
            rows = list_session_admissions(db, session_id=rpc.ref.session_id)
            deleted = release_admission_media(db, candidate['admission_id'])
            observed.append((len(rows), deleted, Path(references[0]['path']).read_bytes()))

        def finished_materialization(*args, **kwargs):
            payload = materialize(*args, **kwargs)
            # Ordinary owner-loop callback runs before this to_thread completion
            # wakes _submit. No worker, held response, barrier, or timer is used.
            loop.call_soon_threadsafe(cleanup_before_admission)
            return payload

        monkeypatch.setattr(session_hosted_attachments, 'submission_payload', finished_materialization)
        params = dict(task=TaskIdentity('room', 'dtask:holder', 'thread', 'turn'), execution_generation=1,
            prompt='Read the two files', attachments=bound, on_terminal=lambda value: None)
        receipt = await rpc._submit(params)
        assert observed == [(0, 0, b'A' * 2048)]
        monkeypatch.setattr(session_hosted_attachments, 'submission_payload', materialize)
        queued = get_session_admission(db, admission_id=receipt['admission_id'])
        assert references[0]['path'] in queued['payload']['text']
        retry = await rpc._submit(params)
        assert retry['admission_id'] == receipt['admission_id']
        assert get_session_admission(db, admission_id=receipt['admission_id'])['payload'] == queued['payload']
        set_holder_status(authority, queued, status)
        assert release_admission_media(db, candidate['admission_id']) == 0
        assert Path(restore_native_media(references)[0]).read_bytes() == b'A' * 2048

        unique, unique_refs = _candidate(home, authority, 'unique.txt', b'unrelated')
        assert release_admission_media(db, unique['admission_id']) == int(witness_available)
        assert Path(unique_refs[0]['path']).exists() is not witness_available
        after = get_session_admission(db, admission_id=receipt['admission_id'])
        assert (after['request_id'], after['payload']) == (queued['request_id'], queued['payload'])


@pytest.mark.parametrize('named', [False, True], ids=['local', 'named'])
@pytest.mark.parametrize('status', ['queued', 'unknown', 'terminal'])
def test_unwitnessed_legacy_rows_conservatively_retain_without_rewriting(tmp_path, monkeypatch, named, status):
    home = tmp_path / 'profiles' / 'member' if named else tmp_path
    home.mkdir(parents=True, exist_ok=True)
    db, authority = _authority(home, monkeypatch)
    with db:
        candidate, references = _candidate(home, authority, 'legacy.txt', b'pre-upgrade document')
        db.create_session('old-hosted', source='cli')
        request_id = 'hosted:' + json.dumps([asdict(TaskIdentity('room', 'dtask:old', 'thread', 'turn')), 1],
            sort_keys=True, separators=(',', ':'))
        payload = {'text': 'Read\n[Shared attachment] file: ' + references[0]['path'] + '\n'}
        row = admit_session_input(db, epoch=authority.epoch, principal_id='human', session_id='old-hosted',
            request_id=request_id, payload=payload)
        set_holder_status(authority, row, status)
        key = hosted_room_input_custody._WITNESS + row['admission_id']
        # Missing, malformed, and differently scoped proof all deny deletion.
        with db._read_ctx() as conn:
            identity = conn.execute('''SELECT principal_id,target_session_id,request_id,payload_digest
                FROM session_admissions WHERE admission_id=?''', (row['admission_id'],)).fetchone()
        wrong_scope = {'version': 1, **dict(zip(hosted_room_input_custody._IDENTITY, identity)),
            'target_session_id': 'another-session'}
        for proof in (None, 'not-json', json.dumps(wrong_scope)):
            if proof is not None:
                db._execute_write(lambda conn: conn.execute(
                    'INSERT OR REPLACE INTO state_meta(key,value) VALUES(?,?)', (key, proof)))
            assert release_admission_media(db, candidate['admission_id']) == 0
            assert Path(restore_native_media(references)[0]).read_bytes() == b'pre-upgrade document'
            after = get_session_admission(db, admission_id=row['admission_id'])
            assert (after['request_id'], after['payload'], after['status']) == (request_id, payload, status)
