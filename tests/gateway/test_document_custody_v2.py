"""Actual Files/RPC handoff keeps bytes without a post-admission witness."""
import asyncio
import errno
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms, session_hosted_attachments
from gateway import hosted_room_input_custody as custody
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.hosted_room_driver import TaskIdentity
from gateway.session_authority import LiveSession
from gateway.session_contract import Principal
from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
from gateway.session_ingress_media import _media_root, release_admission_media
from hermes_state_runtime import RuntimeStoreError, get_session_admission, list_session_admissions
from tests.gateway.test_input_custody_migration import candidate
from tests.gateway.test_native_media_budget import _authority


def files(home):
    path = home / 'state.db'
    hosted_rooms.create_room(path, room_id='room', name='Room', authority_gateway_id='home',
        members=[dict(member_id='member', profile='default', handle='member')])
    store = HostedRoomAttachmentStore(path)
    prepared = []
    for index in range(2):
        data = bytes([65 + index]) * 32
        saved = store.put(room_id='room', upload_id=f'upload-{index}', kind='file',
            name=f'{index}.txt', mime='text/plain', data=data)
        prepared.append(({key: saved[key] for key in ('attachment_id', 'kind', 'name', 'size', 'mime')}, data))
    store.commit_message(room_id='room', event_id='event', manifest=[item for item, _ in prepared],
        recipient_member_ids=['member'])
    return store, [({**item, 'event_id': 'event'}, data) for item, data in prepared]


@pytest.mark.asyncio
@pytest.mark.parametrize('named', [False, True])
@pytest.mark.parametrize('fallback', [False, True])
async def test_v2_backing_precedes_handoff_and_exact_admission_without_witness(tmp_path, monkeypatch, named, fallback):
    home = tmp_path / 'profiles' / 'member' if named else tmp_path
    home.mkdir(parents=True, exist_ok=True)
    db, owner = _authority(home, monkeypatch)
    with db:
        custody.initialize_input_custody(db)
        store, prepared = files(home)
        native, native_path = candidate(home, owner, '0.txt', b'A' * 32)
        if fallback:
            def copy_only(*args):
                raise OSError(errno.EXDEV, 'fixture copy fallback')
            monkeypatch.setattr(custody.os, 'link', copy_only)
        actor = Principal('human', owner.profile_id, frozenset({'session:read', 'session:submit', 'session:control'}), 'fixture')
        loop = asyncio.get_running_loop()
        rpc = HostedRoomAuthorityRPC(owner, loop, room_id='room', member_id='member', profile='default',
            principal=actor, authorize=lambda *args: True)
        db.create_session(rpc.ref.session_id, source='cli')
        owner.sessions[rpc.ref.session_id] = LiveSession(SimpleNamespace(platform=None, user_id='human'), 'fixture')
        if named:
            rpc.hosted_attachment_data = prepared
        original = session_hosted_attachments.submission_payload
        observed = []

        def cleanup():
            rows = list_session_admissions(db, session_id=rpc.ref.session_id)
            observed.append((len(rows), release_admission_media(db, native['admission_id']), native_path.read_bytes()))

        def materialize(*args):
            result = original(*args)
            loop.call_soon_threadsafe(cleanup)
            return result

        monkeypatch.setattr(session_hosted_attachments, 'submission_payload', materialize)
        params = dict(task=TaskIdentity('room', 'task', 'thread', 'turn'), execution_generation=1,
            prompt='read', attachments=[item for item, _ in prepared], on_terminal=lambda value: None)
        receipt = await rpc._submit(params)
        assert observed == [(0, 0, b'A' * 32)]
        monkeypatch.setattr(session_hosted_attachments, 'submission_payload', original)
        saved = get_session_admission(db, admission_id=receipt['admission_id'])
        assert str(native_path) in saved['payload']['text']
        assert (await rpc._submit(params))['admission_id'] == receipt['admission_id']
        assert get_session_admission(db, admission_id=receipt['admission_id'])['payload'] == saved['payload']
        assert release_admission_media(db, native['admission_id']) == 0
        with db._read_ctx() as conn:
            assert conn.execute("SELECT count(*) FROM state_meta WHERE key LIKE 'gateway.hosted.input-custody.v1:%'").fetchone()[0] == 0
        backings = list(custody._backing_root(db.db_path).glob('*/*.txt'))
        assert len(backings) == 2
        source_blobs = [path for path in store.blob_root.iterdir() if path.is_file()]
        for backing in backings:
            alias = _media_root() / backing.parent.name / backing.name
            assert backing.read_bytes() == alias.read_bytes()
            assert all(not os.path.samefile(backing, source) for source in source_blobs)
            if backing.name == '1.txt':
                assert os.path.samefile(backing, alias) is not fallback


def test_unready_or_changed_working_copy_never_publishes_or_repairs(tmp_path, monkeypatch):
    db, owner = _authority(tmp_path, monkeypatch)
    with db:
        store, prepared = files(tmp_path)
        rpc = SimpleNamespace(authority=owner, room_id='room', member_id='member')
        manifest = [item for item, _ in prepared]
        with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
            session_hosted_attachments.submission_payload(rpc, 'read', manifest)
        assert not custody._backing_root(db.db_path).exists()
        custody.initialize_input_custody(db)
        session_hosted_attachments.submission_payload(rpc, 'read', manifest)
        backing = next(custody._backing_root(db.db_path).glob('*/0.txt'))
        alias = _media_root() / backing.parent.name / backing.name
        alias.write_bytes(b'changed private bytes')
        with pytest.raises(RuntimeStoreError, match='storage_unavailable'):
            session_hosted_attachments.submission_payload(rpc, 'read', manifest)
        assert alias.read_bytes() == b'changed private bytes'
        assert store.read(room_id='room', attachment_id=manifest[0]['attachment_id'],
            event_id='event', recipient_member_id='member').data == b'A' * 32
