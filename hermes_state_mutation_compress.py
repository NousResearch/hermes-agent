"""Commit a prepared summary, physical target, generation and receipt atomically."""
import uuid
from hermes_state_mutation_prepared import validate_prepared


def compress_in_transaction(db, conn, session_id, payload, prepared):
    snapshot = validate_prepared(db, conn, session_id, prepared)
    policy = snapshot['receipt']['policy']
    target = snapshot['target']
    from hermes_state_worker_compression import archive_on_connection, publish_on_connection
    # No worker/turn/compactor can be active: validate_prepared checks those
    # obligations on this exact connection. No separate legacy transaction runs.
    if prepared['in_place']:
        child = target
        archive_on_connection(db, conn, target, prepared['messages'])
        affected = {session_id, target}
    else:
        child = uuid.uuid4().hex
        publish_on_connection(db, conn, parent_session_id=target, child_session_id=child,
            source=policy['source'], model=policy['model'], messages=prepared['messages'],
            model_config={'_compressed_from': target}, cwd=policy['cwd'], require_compression_lease=False)
        affected = set()  # Rotation publication already advances the logical revision.
    conn.execute('UPDATE sessions SET runtime_generation=runtime_generation+1 WHERE id=?', (session_id,))
    generation = conn.execute('SELECT runtime_generation FROM sessions WHERE id=?', (session_id,)).fetchone()[0]
    return affected, {'target_session_id': child, 'previous_target_session_id': target,
                   'execution_generation': generation, 'message_count': len(prepared['messages'])}
