"""Main's ``gateway/session_worker_child.py`` bodies (as of the pinned main), the legacy side of the backend parity checks."""

import secrets
from persistence.core.errors import AuthorityRefusal
from hermes_state_runtime_workers import CHILD_SEPARATOR, descendant_range


def reserve_child_execution(db, conn, session_id, payload, *, epoch, execution_id, identity, principal, limits, refused):
    from hermes_state_ids import new_session_id
    from hermes_state_keys import admission_fingerprint
    from hermes_state_runtime import _secret_digest
    from hermes_state_worker_lifecycle import insert_session_row_in_transaction
    secret = payload.get('secret') if isinstance(payload, dict) else None
    if set(payload) != {'secret'} or not isinstance(secret, str) or not 16 <= len(secret) <= 256:
        raise AuthorityRefusal('invalid_params')
    max_children, max_depth = limits
    prefix = execution_id + CHILD_SEPARATOR
    live = conn.execute("SELECT COUNT(*) FROM worker_executions WHERE execution_id>=? AND execution_id<? "
                        "AND instr(substr(execution_id,?),'/')=0 AND status!='terminal'",
                        (*descendant_range(execution_id), len(prefix) + 1)).fetchone()[0]
    if refused is None and prefix.count(CHILD_SEPARATOR) > max_depth:
        refused = 'delegation_depth_exceeded'
    if refused is None and live >= max_children:
        refused = 'delegation_capacity'
    if refused is not None:
        return {'refused': refused}
    parent = conn.execute('SELECT profile_name FROM sessions WHERE id=?', (session_id,)).fetchone()
    child_sid, child_eid = new_session_id(), prefix + secrets.token_hex(8)
    insert_session_row_in_transaction(db, conn, child_sid, 'subagent', model_config={'_delegate_from': session_id},
                                      parent_session_id=session_id, profile_name=parent['profile_name'])
    scope = dict(identity, session_id=child_sid, execution_id=child_eid, generation=0, secret=secret)
    claim = admission_fingerprint(canonical_target=child_sid, payload=scope | {'principal': principal})
    conn.execute("INSERT INTO worker_executions(execution_id,session_id,kind,owner_epoch,generation,status,adoption_digest) "
                 "VALUES(?,?,'child',?,0,'running',?)", (child_eid, child_sid, epoch, _secret_digest(claim)))
    return {'child': {'session_id': child_sid, 'execution_id': child_eid, 'generation': 0}}
