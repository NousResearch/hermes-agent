"""An accepted API image keeps its bytes while its admission is still committing."""
import base64
import os
from pathlib import Path

import pytest

IMAGE = b'\x89PNG\r\n\x1a\n' + os.urandom(4096)


def _image():
    return [{'type': 'text', 'text': 'what is this?'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(IMAGE).decode()}}]


@pytest.mark.asyncio
async def test_api_image_survives_a_sweep_before_its_admission_commits(api, owner, monkeypatch):
    from gateway import session_api_turn
    from gateway.session_api_turn import admit_api_turn, admit_api_turn_async
    from gateway.session_ingress_media import collect_unheld_api_images, restore_native_media
    from hermes_state_runtime import claim_session_input, settle_session_input
    _, _, older = admit_api_turn(api, session_id='older', request_id='older', user_message=_image(),
                                 conversation_history=[])
    started = claim_session_input(owner.db, epoch=owner.epoch, session_id='older')
    settle_session_input(owner.db, epoch=owner.epoch, admission_id=older['admission_id'],
                         generation=started['generation'], outcome='completed')
    original = session_api_turn.admit_session_input

    def admit_after_a_sweep(*args, **kwargs):
        # The window: the new request reused the content-addressed path, its row is not committed,
        # and the older chat is deleted (its references erased) and swept right now.
        if kwargs.get('request_id') == 'newer':
            owner.db.delete_session('older')
            collect_unheld_api_images(owner.db)
        return original(*args, **kwargs)
    monkeypatch.setattr(session_api_turn, 'admit_session_input', admit_after_a_sweep)
    _, _, row = await admit_api_turn_async(api, session_id='newer', request_id='newer', user_message=_image(),
                                           conversation_history=[])
    media = row['payload']['api_turn_v1']['media']
    assert media == older['payload']['api_turn_v1']['media']
    assert Path(media[0]['path']).read_bytes() == IMAGE
    assert restore_native_media(media) == [media[0]['path']]
