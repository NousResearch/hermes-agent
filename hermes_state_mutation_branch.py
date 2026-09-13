"""An independent local route, transcript and frozen policy in one receipt commit."""
import json
import uuid
from datetime import datetime, timezone

from hermes_state_local import POLICY_PREFIX
from hermes_state_local_lineage import validate_local_lineage
from hermes_state_mutation_guards import require_idle
from hermes_state_runtime import RuntimeStoreError, _json


def branch_in_transaction(db, conn, session_id, payload):
    row = conn.execute('SELECT value FROM state_meta WHERE key=?',
                       (POLICY_PREFIX + session_id,)).fetchone()
    if row is None:
        raise RuntimeStoreError('runtime_coordination_required')
    saved = json.loads(row[0])
    target = validate_local_lineage(conn, saved)
    require_idle(db, conn, list({session_id, target}))
    from gateway.session import SessionEntry, SessionSource
    from gateway.config import Platform
    from gateway.session_local_recovery import local_identity
    request_id = 'branch:' + uuid.uuid4().hex
    child = local_identity(saved['profile_id'], saved['principal_id'], request_id)
    source = SessionSource(Platform.LOCAL, child, user_id=saved['principal_id'], chat_type='dm')
    from gateway.session import build_session_key, SessionStore
    route = build_session_key(source, profile=SessionStore._profile_from_session_key(saved['route']))
    now = datetime.now(timezone.utc)
    entry = SessionEntry(route, child, now, now, origin=source, platform=Platform.LOCAL)
    parent = conn.execute('SELECT * FROM sessions WHERE id=?', (target,)).fetchone()
    policy = saved['policy']
    db._publish_child_session_row(conn, parent, parent_session_id=target,
        child_session_id=child, source=policy['source'], model=policy['model'],
        model_config={'_branched_from': target}, system_prompt=None,
        cwd=policy['cwd'], profile_name=parent['profile_name'])
    conn.execute('UPDATE sessions SET session_key=?,chat_id=?,origin_json=?,system_prompt_hash=? WHERE id=?',
                 (route, child, _json(source.to_dict()), parent['system_prompt_hash'], child))
    ids, tools = db._tail_rows_after_watermark(conn,
        'SELECT id,tool_calls FROM messages WHERE session_id=? AND active=1 ORDER BY id', [target])
    db._clone_message_rows(conn, ids, session_id=child)
    from hermes_state_input_custody import copy_branch_input_refs
    copy_branch_input_refs(conn, session_id, child, physical_session=target)
    conn.execute('UPDATE sessions SET message_count=?,tool_call_count=? WHERE id=?', (len(ids), tools, child))
    if payload.get('title'):
        db._set_session_title_in_transaction(conn, child, payload['title'], source=db.TITLE_SOURCE_USER)
    receipt = dict(profile_id=saved['profile_id'], principal_id=saved['principal_id'],
                   request_id=request_id, session_id=child, route=route, entry=entry.to_dict(), policy=policy)
    conn.execute("INSERT INTO gateway_routing(scope,session_key,entry_json,updated_at) VALUES('',?,?,?)",
                 (route, _json(entry.to_dict()), now.timestamp()))
    conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (POLICY_PREFIX + child, _json(receipt)))
    return {session_id}, {'branched_session_id': child, 'copied_messages': len(ids)}
