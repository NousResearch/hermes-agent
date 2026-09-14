"""Native scratch cleanup must not retire API image context sharing its bytes."""
import base64
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway import hosted_rooms
from gateway.hosted_room_attachments import HostedRoomAttachmentStore
from gateway.platforms import base
from gateway.session_api_turn import admit_api_turn
from gateway.session_ingress_media import _media_root, admit_attachments, release_admission_media, restore_native_media
from hermes_state_runtime import RuntimeStoreError, admit_session_input, claim_session_input, settle_session_input
from tests.gateway.test_api_cutover_contract import api, owner  # noqa: F401

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=')


def settled_native(owner, paths):
    owner.db.create_session('native-candidate', source='cli')
    payload = {'text': 'native', **admit_attachments([{'path': str(path), 'mime': 'image/png'} for path in paths])}
    row = admit_session_input(owner.db, epoch=owner.epoch, principal_id='native-person',
        session_id='native-candidate', request_id='native', payload=payload)
    started = claim_session_input(owner.db, epoch=owner.epoch, session_id=row['target_session_id'])
    return settle_session_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'],
        generation=started['generation'], outcome='completed')


@pytest.mark.parametrize('settled', [False, True], ids=['queued-holder', 'terminal-holder'])
def test_api_image_survives_native_cleanup_while_unique_native_bytes_are_collected(api, owner, settled):
    from gateway.hosted_room_input_custody import initialize_input_custody
    initialize_input_custody(owner.db)
    _, _, row = admit_api_turn(api, session_id='image-holder', request_id='api-image',
        user_message=[{'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(PNG).decode()}}], conversation_history=[])
    reference = row['payload']['api_turn_v1']['media'][0]
    if settled:
        started = claim_session_input(owner.db, epoch=owner.epoch, session_id=row['target_session_id'])
        settle_session_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'],
            generation=started['generation'], outcome='completed')
        # A settled API turn is a holder, never a deletion candidate.
        assert release_admission_media(owner.db, row['admission_id']) == 0
    unique = base.get_image_cache_dir() / 'unique.png'
    unique.write_bytes(PNG + b'unique')
    native = settled_native(owner, [base.get_image_cache_dir() / Path(reference['path']).name, unique])
    shared, unique_reference = native['payload']['attachments_v1']['media']
    assert shared == reference
    assert release_admission_media(owner.db, native['admission_id']) == 1
    assert not Path(unique_reference['path']).exists()
    assert Path(restore_native_media([reference])[0]).read_bytes() == PNG


def test_hosted_batch_total_is_rejected_before_any_capture(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db_path = tmp_path / 'state.db'
    from gateway.hosted_room_input_custody import initialize_input_custody
    from hermes_state import SessionDB
    with SessionDB(db_path) as db:
        initialize_input_custody(db)
    hosted_rooms.create_room(db_path, room_id='room', name='Room', authority_gateway_id='home',
        members=[dict(member_id='member', profile='default', handle='member')])
    store = HostedRoomAttachmentStore(db_path)
    manifest = []
    for index in range(2):
        saved = store.put(room_id='room', upload_id=f'upload-{index}', kind='file', name=f'{index}.txt',
            mime='text/plain', data=bytes([65 + index]) * 2048)
        manifest.append({key: saved[key] for key in ('attachment_id', 'kind', 'name', 'size', 'mime')})
    store.commit_message(room_id='room', event_id='source', manifest=manifest, recipient_member_ids=['member'])
    bound = [{**item, 'event_id': 'source'} for item in manifest]
    rpc = SimpleNamespace(authority=SimpleNamespace(db=SimpleNamespace(db_path=db_path)), room_id='room', member_id='member')
    monkeypatch.setattr(base, 'get_inbound_media_max_bytes', lambda: 3072)
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        from gateway.hosted_room_input_preparation import resolve_inputs
        resolve_inputs(rpc, bound)
    assert not _media_root().exists()
    monkeypatch.setattr(base, 'get_inbound_media_max_bytes', lambda: 4096)
    assert [data for _, data in resolve_inputs(rpc, bound)] == [b'A' * 2048, b'B' * 2048]
