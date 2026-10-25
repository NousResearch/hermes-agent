"""A settled API admission row keeps only what exact-retry projections read."""
import base64
import json
import os

import pytest

from gateway.session_api_media import compact_settled_api_payloads
from gateway.session_api_turn import admit_api_turn, recover_api_turns
from hermes_state_runtime import claim_session_input, settle_session_input

IMAGE = b'\x89PNG\r\n\x1a\n' + os.urandom(16 * 1024)
HISTORY = [{'role': 'user', 'content': 'earlier ' + 'x' * 4096}, {'role': 'assistant', 'content': 'reply'}]


def _image(header):
    return [{'type': 'text', 'text': 'what is this?'},
            {'type': 'image_url', 'image_url': {'url': header + base64.b64encode(IMAGE).decode()}}]


def _stored(owner, admission_id):
    with owner.db._read_ctx() as conn:
        return json.loads(conn.execute('SELECT payload_json FROM session_admissions WHERE admission_id=?',
                                       (admission_id,)).fetchone()[0])


# (case, request id, message, history, startup sweep instead of the per-row pass, check(stored payload))
CASES = [
    ('mixed-case-image', 'chat:k1', _image('DATA:IMAGE/PNG;BASE64,'), [], False,
     lambda p: 'BASE64,' not in json.dumps(p)),
    ('chat-history', 'chat:k2', 'next', HISTORY, False, lambda p: p['api_turn_v1']['history'] is None),
    ('runs-history-after-crash', 'run_k3', 'next', HISTORY, True, lambda p: p['api_turn_v1']['history'] is None),
    # Responses terminal replay rebuilds the chained snapshot from the row's history: it stays.
    ('responses-history-kept', 'responses:s:k4', 'next', HISTORY, False,
     lambda p: p['api_turn_v1']['history'] == HISTORY),
]


@pytest.mark.parametrize('case,request_id,message,history,sweep,check', CASES, ids=[c[0] for c in CASES])
def test_settled_api_rows_drop_what_no_terminal_reader_needs(api, owner, case, request_id, message, history,
                                                             sweep, check):
    kwargs = dict(session_id='s-' + case, request_id=request_id, user_message=message, conversation_history=history)
    _, _, row = admit_api_turn(api, **kwargs)
    started = claim_session_input(owner.db, epoch=owner.epoch, session_id=row['target_session_id'])
    settle_session_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'],
                         generation=started['generation'], outcome='completed')
    if sweep:
        recover_api_turns(api)
    else:
        compact_settled_api_payloads(owner.db, row['admission_id'])
    assert check(_stored(owner, row['admission_id'])), _stored(owner, row['admission_id'])
    # The digest is untouched: an exact retry still returns the settled admission.
    retry = admit_api_turn(api, **kwargs)[2]
    assert retry['admission_id'] == row['admission_id'] and retry['status'] == 'terminal'
