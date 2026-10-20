"""An operator notice and its canonical retry receipt commit together, without inference."""
from hermes_state_local_lineage import local_physical_target
from hermes_state_mutation_guards import require_idle


def notice_in_transaction(db, conn, session_id, payload):
    target = local_physical_target(conn, session_id)
    require_idle(db, conn, list({session_id, target}))
    db._check_transcript_write_guards(conn, target, None, reject_active_turn_lease=True)
    message = {'role': 'user', 'content': payload['text'], 'display_kind': 'internal_notification',
               'display_metadata': {'display_text': payload['text'], 'notification_category': 'operation'}}
    db._insert_message_rows(conn, target, [message])
    db._bump_session_counters(conn, target, 1, 0, unit=True)
    return {session_id, target}, {'notice_row_id': message['_row_id'],
                                  'notice_message_uid': message['message_uid'], 'text': payload['text']}
