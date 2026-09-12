"""Native scratch cleanup must not retire API image context sharing its bytes."""
import base64
from pathlib import Path

import pytest

from gateway.session_api_turn import admit_api_turn
from gateway.session_ingress_media import admit_attachments, release_admission_media, restore_native_media
from hermes_state_runtime import admit_session_input, claim_session_input, settle_session_input
from tests.gateway.test_api_cutover_contract import api, owner  # noqa: F401

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=')


def api_image(api):
    return admit_api_turn(api, session_id='image-holder', request_id='api-image',
        user_message=[{'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(PNG).decode()}}], conversation_history=[])


def set_holder_status(owner, row, status):
    if status == 'queued':
        return
    started = claim_session_input(owner.db, epoch=owner.epoch, session_id=row['target_session_id'])
    assert started['admission_id'] == row['admission_id']
    if status == 'terminal':
        settle_session_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'],
            generation=started['generation'], outcome='completed')
    elif status == 'unknown':
        # A persisted unknown holder fixture, without worker loss/restart or execution.
        owner.db._execute_write(lambda conn: conn.execute(
            "UPDATE session_admissions SET status='unknown' WHERE admission_id=?", (row['admission_id'],)))


def settled_native(owner, paths):
    owner.db.create_session('native-candidate', source='cli')
    payload = {'text': 'native', **admit_attachments([{'path': str(path), 'mime': 'image/png'} for path in paths])}
    row = admit_session_input(owner.db, epoch=owner.epoch, principal_id='native-person',
        session_id='native-candidate', request_id='native', payload=payload)
    started = claim_session_input(owner.db, epoch=owner.epoch, session_id=row['target_session_id'])
    settled = settle_session_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'],
        generation=started['generation'], outcome='completed')
    return settled


@pytest.mark.parametrize('status', ['queued', 'started', 'unknown', 'terminal'])
def test_api_image_same_digest_and_filename_survives_native_cleanup(api, owner, status):
    from gateway.platforms.base import get_image_cache_dir
    _, _, row = api_image(api)
    reference = row['payload']['api_turn_v1']['media'][0]
    set_holder_status(owner, row, status)
    native = settled_native(owner, [get_image_cache_dir() / Path(reference['path']).name])
    assert native['payload']['attachments_v1']['media'] == [reference]
    assert release_admission_media(owner.db, native['admission_id']) == 0
    assert Path(restore_native_media([reference])[0]).read_bytes() == PNG
    if status == 'terminal':
        assert release_admission_media(owner.db, row['admission_id']) == 0
        assert Path(reference['path']).read_bytes() == PNG


def test_unique_native_candidate_is_collected_while_shared_api_bytes_survive(api, owner):
    from gateway.platforms.base import get_image_cache_dir
    _, _, row = api_image(api)
    reference = row['payload']['api_turn_v1']['media'][0]
    unique = get_image_cache_dir() / 'unique.png'
    unique.write_bytes(PNG + b'unique')
    native = settled_native(owner, [get_image_cache_dir() / Path(reference['path']).name, unique])
    unique_reference = native['payload']['attachments_v1']['media'][1]
    assert release_admission_media(owner.db, native['admission_id']) == 1
    assert not Path(unique_reference['path']).exists()
    assert Path(restore_native_media([reference])[0]).read_bytes() == PNG


def test_terminal_api_image_is_not_a_native_deletion_candidate(api, owner):
    _, _, row = api_image(api)
    reference = row['payload']['api_turn_v1']['media'][0]
    set_holder_status(owner, row, 'terminal')
    assert release_admission_media(owner.db, row['admission_id']) == 0
    assert Path(restore_native_media([reference])[0]).read_bytes() == PNG
