"""A settled API admission keeps its committed image by reference, not as a second base64 copy."""
import base64
import os
from pathlib import Path

from gateway.platforms.api_server_response_admissions import admitted_context
from gateway.session_contract import SessionRef
from gateway.session_api_turn import admit_api_turn, recover_api_turns
from gateway.session_ingress_media import release_admission_media
from hermes_state_runtime import claim_session_input, get_session_admission, settle_session_input

IMAGE = b'\x89PNG\r\n\x1a\n' + os.urandom(64 * 1024)


def _content():
    return [{'type': 'text', 'text': 'what is this?'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(IMAGE).decode()}}]


def _settle(owner, row, *, release=True):
    started = claim_session_input(owner.db, epoch=owner.epoch, session_id=row['target_session_id'])
    settle_session_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'],
                         generation=started['generation'], outcome='completed')
    if release:
        release_admission_media(owner.db, row['admission_id'])  # the drain's terminal media pass


def _stored(owner):
    with owner.db._read_ctx() as conn:
        return dict(conn.execute('SELECT request_id, payload_json FROM session_admissions').fetchall())


def test_settled_rows_drop_inline_base64_but_retry_replay_restart_and_cleanup_hold(api, owner):
    rows = [admit_api_turn(api, session_id=f'chat-{n}', request_id=f'turn-{n}', user_message=_content(),
                           conversation_history=[])[2] for n in range(4)]
    for row in rows[:2]:
        _settle(owner, row)
    stored = _stored(owner)
    assert 'data:image/png;base64' in stored['turn-3'], 'a queued row must keep executable input'
    assert all(len(stored[f'turn-{n}']) < len(IMAGE) for n in range(2)), \
        'every settled identical-image turn kept its own base64 copy'
    for row in rows[2:]:
        _settle(owner, row, release=False)  # the owner died between settlement and its media pass
    recover_api_turns(api)  # startup recovery compacts what the crash left behind
    assert all(len(payload) < len(IMAGE) for payload in _stored(owner).values())
    # An exact retry still matches the admission digest and returns the settled admission.
    retry = admit_api_turn(api, session_id='chat-0', request_id='turn-0', user_message=_content(),
                           conversation_history=[])[2]
    assert retry['admission_id'] == rows[0]['admission_id'] and retry['status'] == 'terminal'
    # A terminal Responses replay still projects the original request content.
    terminal = get_session_admission(owner.db, admission_id=rows[0]['admission_id'])
    context = admitted_context((owner, SessionRef(owner.profile_id, 'chat-0'), terminal), replay=True)
    assert context['user_message'] == _content()
    # The committed bytes stay as history context, and are collected once every chat is gone.
    path = Path(rows[0]['payload']['api_turn_v1']['media'][0]['path'])
    assert path.read_bytes() == IMAGE
    for n in range(4):
        owner.db.delete_session(f'chat-{n}')
    assert not path.exists()
