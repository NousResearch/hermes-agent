import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


def test_authorized_upload_send_download_and_task_consumption(tmp_path):
    from gateway.session_hosted_attachments import upload, download, submission_payload
    from gateway import hosted_rooms
    hosted_rooms.create_room(storage_handle(tmp_path), placement="profile", room_id="room", name="Room", members=[], authority_gateway_id="gateway")
    from persistence.core.errors import AuthorityRefusal
    allowed = [True]
    def authorize(subject, room_id):
        if not allowed[0] or subject != 'owner' or room_id != 'room':
            raise AuthorityRefusal('permission_denied')
    service = SimpleNamespace(authority=SimpleNamespace(profile_id='owned', persistence=storage_handle(tmp_path)), db_path=tmp_path / 'state.db',
        store=storage_handle(tmp_path), placement='profile', authorize_room=authorize,
        _room=lambda room: {'members': [{'member_id': 'member', 'profile': 'default'}]},
        _owned_authority=lambda room: ('gateway', 1))
    actor = SimpleNamespace(subject='owner', profile_id='owned', capabilities={'session:read','session:submit'})
    params = dict(room_id='room', upload_id='upload', kind='file', name='note.txt', mime='text/plain', data_base64=base64.b64encode(b'committed bytes').decode())
    metadata = upload(service, actor, params)
    manifest = [{k: metadata[k] for k in ('attachment_id', 'kind', 'name', 'size', 'mime')}]
    from gateway.session_hosted_attachments import append_user_event
    append_user_event(service, room_id='room', event_id='event', payload={'text':'read', 'attachments':manifest}, gateway_id='gateway', epoch=1)
    bound = [{**manifest[0], 'event_id': 'event'}]
    rpc = SimpleNamespace(authority=SimpleNamespace(persistence=storage_handle(service.db_path.parent)), room_id='room', member_id='member')
    payload = submission_payload(rpc, 'read it', bound)
    path = Path(payload['text'].split('file: ')[1].split('\n')[0])
    assert path.read_bytes() == b'committed bytes'
    assert submission_payload(rpc, 'read it', bound) == payload
    assert base64.b64decode(download(service, actor, dict(room_id='room', event_id='event', attachment_id=metadata['attachment_id']))['data_base64']) == path.read_bytes()
    with pytest.raises(Exception):
        submission_payload(rpc, 'read it', [{**bound[0], 'event_id': 'wrong'}])
    allowed[0] = False
    for call in (lambda: upload(service, actor, params), lambda: download(service, actor, dict(room_id='room', attachment_id=metadata['attachment_id'], event_id='event'))):
        with pytest.raises(AuthorityRefusal, match='permission_denied'):
            call()


def test_rpc_accepts_only_bound_manifest_after_producer_authorization(hosted_owner):
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from persistence.gateway.hosted_rooms import HostedTaskIdentity
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    authority, loop, principal, _ = hosted_owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='member', profile='default', principal=principal, authorize=lambda *args: True)
    coords = dict(profile='default', source='bot_room')
    sid = rpc.create(**coords, title='Group: room')['session_id']
    from gateway.session_persistence import authority_persistence
    store = HostedRoomAttachmentStore(authority_persistence(authority))
    saved = store.put(room_id='room', upload_id='upload', kind='file', name='note.txt', mime='text/plain', data=b'exact bytes')
    manifest = [{k: saved[k] for k in ('attachment_id', 'kind', 'name', 'size', 'mime')}]
    store.commit_message(room_id='room', event_id='event', manifest=manifest, recipient_member_ids=['member'])
    args = dict(**coords, session_id=sid, prompt='read', task=HostedTaskIdentity('room','task','thread','turn'), execution_generation=1, on_terminal=lambda receipt: None, attachments=[{**manifest[0], 'event_id':'event'}])
    first = rpc.submit(**args)
    assert rpc.submit(**args)['admission_id'] == first['admission_id']
    rows = authority.persistence.admissions.list(session_id=sid, pending_only=False)
    assert len(rows) == 1
    path = Path(rows[0].payload['text'].split('file: ')[1].split('\n')[0])
    assert path.read_bytes() == b'exact bytes'


def test_a_plain_message_runs_the_attachment_setup_of_each_process(tmp_path, monkeypatch):
    from gateway import hosted_rooms
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    from gateway.session_hosted_attachments import append_user_event
    from tests.fixtures.storage_holder import hosted_attachment_setup_seam, hosted_upload, raw_holder
    store = storage_handle(tmp_path)
    hosted_rooms.create_room(store, placement="profile", room_id="room", name="Room", members=[], authority_gateway_id="gateway")
    service = SimpleNamespace(authority=SimpleNamespace(profile_id='owned', persistence=store), store=store, placement='profile',
        _room=lambda room: {'members': []}, _owned_authority=lambda room: ('gateway', 1))
    send = lambda event_id: append_user_event(service, room_id='room', event_id=event_id, payload={'text': 'hi'}, gateway_id='gateway', epoch=1)
    send('first')
    with raw_holder(store)._read_ctx() as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {'hosted_room_attachments', 'hosted_room_attachment_blobs'} <= tables
    HostedRoomAttachmentStore(store, clock=lambda: 1000.0).put(room_id='room', upload_id='old', kind='file', name='old.txt', mime='text/plain', data=b'stale bytes')
    blob = tmp_path / 'hosted-room-attachments' / 'blobs' / hosted_upload(store, 'profile', 'room', 'old')['blob_id']
    assert blob.exists()
    module, name = hosted_attachment_setup_seam()
    monkeypatch.setattr(module, name, set())
    send('second')
    assert hosted_upload(store, 'profile', 'room', 'old') is None and not blob.exists()
