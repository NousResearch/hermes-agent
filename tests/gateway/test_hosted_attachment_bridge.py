import base64
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_authorized_upload_send_download_and_task_consumption(tmp_path, monkeypatch):
    from gateway.hosted_room_input_custody import initialize_input_custody
    from hermes_state import SessionDB
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        initialize_input_custody(db)
    from gateway.session_hosted_attachments import upload, download, submission_payload
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    from gateway import hosted_rooms
    hosted_rooms.create_room(tmp_path / "state.db", room_id="room", name="Room", members=[], authority_gateway_id="gateway")
    from hermes_state_runtime import RuntimeStoreError
    allowed = [True]
    def authorize(subject, room_id):
        if not allowed[0] or subject != 'owner' or room_id != 'room':
            raise RuntimeStoreError('permission_denied')
    service = SimpleNamespace(authority=SimpleNamespace(profile_id='owned'), db_path=tmp_path / 'state.db', authorize_room=authorize,
        _room=lambda room: {'members': [{'member_id': 'member', 'profile': 'default'}]},
        _owned_authority=lambda room: ('gateway', 1))
    actor = SimpleNamespace(subject='owner', profile_id='owned', capabilities={'session:read','session:submit'})
    params = dict(room_id='room', upload_id='upload', kind='file', name='note.txt', mime='text/plain', data_base64=base64.b64encode(b'committed bytes').decode())
    metadata = upload(service, actor, params)
    manifest = [{k: metadata[k] for k in ('attachment_id', 'kind', 'name', 'size', 'mime')}]
    store = HostedRoomAttachmentStore(service.db_path)
    from gateway.session_hosted_attachments import append_user_event
    append_user_event(service, room_id='room', event_id='event', payload={'text':'read', 'attachments':manifest}, gateway_id='gateway', epoch=1)
    bound = [{**manifest[0], 'event_id': 'event'}]
    rpc = SimpleNamespace(authority=SimpleNamespace(db=SimpleNamespace(db_path=service.db_path)), room_id='room', member_id='member')
    payload = submission_payload(rpc, 'read it', bound)
    path = Path(payload['text'].split('file: ')[1].split('\n')[0])
    assert path.read_bytes() == b'committed bytes'
    assert submission_payload(rpc, 'read it', bound) == payload
    assert base64.b64decode(download(service, actor, dict(room_id='room', event_id='event', attachment_id=metadata['attachment_id']))['data_base64']) == path.read_bytes()
    with pytest.raises(Exception):
        submission_payload(rpc, 'read it', [{**bound[0], 'event_id': 'wrong'}])
    allowed[0] = False
    for call in (lambda: upload(service, actor, params), lambda: download(service, actor, dict(room_id='room', attachment_id=metadata['attachment_id'], event_id='event'))):
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            call()


def test_rpc_accepts_only_bound_manifest_after_producer_authorization(owner):
    from gateway.session_hosted_rpc import HostedRoomAuthorityRPC
    from gateway.hosted_room_driver import TaskIdentity
    from gateway.hosted_room_attachments import HostedRoomAttachmentStore
    from hermes_state_runtime import list_session_admissions
    authority, loop, principal, _ = owner
    rpc = HostedRoomAuthorityRPC(authority, loop, room_id='room', member_id='member', profile='default', principal=principal, authorize=lambda *args: True)
    coords = dict(profile='default', source='bot_room')
    sid = rpc.create(**coords, title='Group: room')['session_id']
    store = HostedRoomAttachmentStore(authority.db.db_path)
    saved = store.put(room_id='room', upload_id='upload', kind='file', name='note.txt', mime='text/plain', data=b'exact bytes')
    manifest = [{k: saved[k] for k in ('attachment_id', 'kind', 'name', 'size', 'mime')}]
    store.commit_message(room_id='room', event_id='event', manifest=manifest, recipient_member_ids=['member'])
    args = dict(**coords, session_id=sid, prompt='read', task=TaskIdentity('room','task','thread','turn'), execution_generation=1, on_terminal=lambda receipt: None, attachments=[{**manifest[0], 'event_id':'event'}])
    first = rpc.submit(**args)
    assert rpc.submit(**args)['admission_id'] == first['admission_id']
    rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
    assert len(rows) == 1
    path = Path(rows[0]['payload']['text'].split('file: ')[1].split('\n')[0])
    assert path.read_bytes() == b'exact bytes'


from tests.gateway.test_session_hosted_rpc import owner  # noqa: F401,E402
