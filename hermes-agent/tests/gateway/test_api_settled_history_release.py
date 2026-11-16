"""Settlement itself drops a text-only API row's client history, not only the next restart's sweep."""
import json

from gateway.session_api_turn import admit_api_turn
from gateway.session_ingress_media import release_admission_media
from hermes_state_runtime import claim_session_input, settle_session_input

HISTORY = [{'role': 'user', 'content': 'earlier ' + 'x' * 4096}, {'role': 'assistant', 'content': 'reply'}]


def test_settling_a_text_only_chat_turn_drops_its_history(api, owner):
    _, _, row = admit_api_turn(api, session_id='s-text', request_id='chat:t1', user_message='next',
                               conversation_history=HISTORY)
    started = claim_session_input(owner.db, epoch=owner.epoch, session_id=row['target_session_id'])
    settle_session_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'],
                         generation=started['generation'], outcome='completed')
    release_admission_media(owner.db, row['admission_id'])  # what the authority runs at settlement
    with owner.db._read_ctx() as conn:
        kind = conn.execute("SELECT json_type(payload_json, '$.api_turn_v1.history') FROM session_admissions "
                            "WHERE admission_id=?", (row['admission_id'],)).fetchone()[0]
    assert kind == 'null'
