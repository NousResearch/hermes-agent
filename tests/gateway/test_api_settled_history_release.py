"""Settlement itself drops a text-only API row's client history, not only the next restart's sweep."""
import json

from gateway.session_api_turn import admit_api_turn
from gateway.session_ingress_media import release_admission_media
from gateway.session_persistence import authority_persistence
from tests.fixtures.storage_holder import raw_holder

HISTORY = [{'role': 'user', 'content': 'earlier ' + 'x' * 4096}, {'role': 'assistant', 'content': 'reply'}]


def test_settling_a_text_only_chat_turn_drops_its_history(api, owner):
    _, _, row = admit_api_turn(api, session_id='s-text', request_id='chat:t1', user_message='next',
                               conversation_history=HISTORY)
    started = owner.persistence.admissions.claim(epoch=owner.epoch, session_id=row['target_session_id'])
    owner.persistence.admissions.settle(epoch=owner.epoch, admission_id=row['admission_id'],
                                        generation=started.generation, outcome='completed')
    release_admission_media(authority_persistence(owner), row['admission_id'])  # what the authority runs at settlement
    with raw_holder(owner.persistence)._read_ctx() as conn:
        kind = conn.execute("SELECT json_type(payload_json, '$.api_turn_v1.history') FROM session_admissions "
                            "WHERE admission_id=?", (row['admission_id'],)).fetchone()[0]
    assert kind == 'null'
