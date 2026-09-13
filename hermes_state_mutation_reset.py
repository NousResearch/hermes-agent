"""Prepare and publish a local reset inside the mutation receipt transaction."""
import json
import uuid
import time

from hermes_state_local import POLICY_PREFIX
from hermes_state_local_lineage import advance_local_target, validate_local_lineage
from hermes_state_mutation_guards import require_idle
from hermes_state_runtime import RuntimeStoreError


def reset_in_transaction(db, conn, session_id, payload):
    saved = conn.execute('SELECT value FROM state_meta WHERE key=?',
                         (POLICY_PREFIX + session_id,)).fetchone()
    if saved is None:
        raise RuntimeStoreError('runtime_coordination_required')
    receipt = json.loads(saved[0])
    target = validate_local_lineage(conn, receipt)
    require_idle(db, conn, list({session_id, target}))
    parent = conn.execute('SELECT * FROM sessions WHERE id=?', (target,)).fetchone()
    policy = receipt['policy']
    from gateway.session import SessionEntry
    previous = SessionEntry.from_dict(receipt['entry'])
    from gateway.session_lifecycle import _now
    now = _now()
    child_id = uuid.uuid4().hex
    entry = SessionEntry(previous.session_key, child_id, now, now,
        origin=previous.origin, platform=previous.platform, chat_type=previous.chat_type,
        display_name=previous.display_name, is_fresh_reset=True)
    db._publish_child_session_row(conn, parent, parent_session_id=target,
        child_session_id=child_id, source=policy['source'], model=policy['model'],
        model_config={'_reset_from': target}, system_prompt=None,
        cwd=policy['cwd'], profile_name=parent['profile_name'])
    conn.execute("UPDATE sessions SET ended_at=?,end_reason='session_reset' WHERE id=?", (time.time(), target))
    db._bump_conversation_generation(conn, target, 'session_reset')
    advance_local_target(conn, target, child_id, entry=entry.to_dict())
    conn.execute('UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', (session_id,))
    generation = conn.execute('SELECT runtime_generation FROM sessions WHERE id=?', (session_id,)).fetchone()[0]
    # advance_local_target already advances the logical owner's revision.
    return set(), {'target_session_id': child_id, 'previous_target_session_id': target,
                   'execution_generation': generation}
