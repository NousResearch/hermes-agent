"""A model or reasoning pick on a session with a running turn lands for the next turn (RT-12, MP-24).

The pick is authorized and validated when it arrives, kept durably in ``state_meta`` (latest pick
wins, survives an owner restart), and committed through ``session.mutate``'s own writer by the
drain at the next claim boundary, when nothing is executing. The running turn keeps its route.
A level rides in the same slot beside the model pick and commits first, as ``apply_pending_reasoning``
runs before a model pick: a level picked with a later model pick wins, a later level replaces it.
"""
import json
import logging

from hermes_state_mutation_prepared import DEFERRED_PICK_PREFIX as _PREFIX
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)


def defer_pick(authority, actor, ref, params):
    """Record *params* (an authorized ``model`` or ``reasoning`` mutation refused only because a turn
    is running).

    Blocking SQLite: callers run it off the event loop. Acceptance writes the request's durable
    mutation receipt in the same transaction, so an exact retry returns this result instead of
    overwriting a newer pick, and the stored pick carries a stable application id."""
    from hermes_state_mutations import validate_action
    from hermes_state_runtime import _epoch, _json, mutation_receipt_key, replay_mutation_receipt
    operation = params.get('operation', 'model')
    validate_action(operation, params['payload'])
    key, digest = mutation_receipt_key(actor.subject, ref.session_id, params['request_id'], operation=operation,
                                       payload=params['payload'], expected_revision=params.get('expected_revision'),
                                       expected_generation=params.get('expected_generation'))
    pick = {'principal_id': actor.subject, 'request_id': params['request_id'], 'payload': params['payload'],
            'application_id': 'deferred-pick:' + params['request_id']}

    def write(conn):
        _epoch(conn, authority.epoch)
        replay = replay_mutation_receipt(conn, key, digest)
        if replay is not None:
            return replay
        row = conn.execute('SELECT runtime_revision, runtime_generation FROM sessions WHERE id=?',
                           (ref.session_id,)).fetchone()
        if row is None:
            raise RuntimeStoreError('not_found')
        stored = conn.execute('SELECT value FROM state_meta WHERE key=?', (_PREFIX + ref.session_id,)).fetchone()
        current = json.loads(stored[0]) if stored else {}
        if operation == 'reasoning':
            current.get('payload', {}).pop('reasoning', None)  # this level is newer than the one picked with the model
            value, picked = {**current, 'reasoning': pick}, {'effort': params['payload']['effort']}
        else:
            # A level picked with this model is newer than one deferred before it, which it drops: the
            # slot never holds both, so committing the slot's level never strips the model's own.
            older = 'reasoning' in current and 'reasoning' not in params['payload']
            value = {**pick, **({'reasoning': current['reasoning']} if older else {})}
            picked = {'model': params['payload']['model']}
        result = {'session_id': ref.session_id, 'operation': operation, 'status': 'deferred', **picked,
                  'revision': row[0], 'execution_generation': row[1]}
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
                     (_PREFIX + ref.session_id, _json(value)))
        conn.execute('INSERT INTO state_meta(key,value) VALUES(?,?)', (key, _json({'digest': digest, 'result': result})))
        return result
    return authority.db._execute_write(write)


async def reask_on_attach(authority, ref, live):
    """A guarded pick asked only the viewers present at its boundary (and an owner restart drops
    the stream): a newly attached viewer gets the ask again, re-checked at the next boundary."""
    import asyncio

    def read():
        with authority.db._read_ctx() as conn:
            return conn.execute('SELECT value FROM state_meta WHERE key=?', (_PREFIX + ref.session_id,)).fetchone()
    stored = await asyncio.to_thread(read)
    if stored is not None and 'awaiting_confirm' in json.loads(stored[0]):
        live.reask_pick = True
        authority._schedule(ref)


async def _commit(authority, ref, entry, operation):
    """Commit the deferred *entry* as an *operation* mutation. Returns None while a worker still holds
    the transcript (the pick waits for the next boundary), else the mutation's result."""
    import asyncio
    from gateway.session_mutations import write_mutation
    from hermes_state_mutation_guards import MUTATION_GUARD_REFUSALS
    from hermes_state_mutation_retirement import has_mutation_receipt
    if await asyncio.to_thread(has_mutation_receipt, authority.db, entry['principal_id'], ref.session_id,
                               entry['application_id']):
        return {'status': 'replayed'}  # committed before a restart, not yet removed: never apply it twice
    row = await asyncio.to_thread(authority.db.get_session, ref.session_id)
    params = {'session_id': ref.session_id, 'request_id': entry['application_id'],
              'expected_revision': row['runtime_revision'], 'expected_generation': row['runtime_generation'],
              'operation': operation, 'payload': entry['payload']}
    try:
        return await write_mutation(authority, ref, params, None, principal_id=entry['principal_id'])
    except MUTATION_GUARD_REFUSALS:
        return None
    except RuntimeStoreError as exc:
        if exc.reason == 'session_busy':
            return None
        return {'session_id': ref.session_id, 'operation': operation, 'status': 'refused', 'reason': exc.reason}
    except Exception:
        # Resolving the pick failed (provider auth, catalog, resolver crash). This runs before the
        # claim, so letting it escape would kill or wedge the drain on the same pick forever.
        logger.exception('Deferred %s pick on %s failed to resolve', operation, ref.session_id)
        return {'session_id': ref.session_id, 'operation': operation, 'status': 'refused',
                'reason': operation + '_resolution_failed'}


async def _commit_level(authority, ref, key, raw, pick):
    """Commit the level deferred in *pick* (stored as *raw*) and drop it from the slot. Returns the
    remaining slot value, or None when nothing is left or a worker still holds the transcript."""
    import asyncio
    from hermes_state_runtime import _json
    result = await _commit(authority, ref, pick.pop('reasoning'), 'reasoning')
    if result is None:
        return None
    if result.get('status') == 'refused':
        logger.warning('Deferred reasoning pick on %s not applied: %s', ref.session_id, result['reason'])
    rest = _json(pick) if 'application_id' in pick else None

    def settle(conn):
        if rest is None:
            conn.execute('DELETE FROM state_meta WHERE key=? AND value=?', (key, raw))
        else:
            conn.execute('UPDATE state_meta SET value=? WHERE key=? AND value=?', (rest, key, raw))
    await asyncio.to_thread(authority.db._execute_write, settle)
    return rest


async def apply_deferred_pick(authority, ref, live):
    """Commit a deferred pick. The drain calls this under the claim gate and mutation lock, before
    claiming, so no turn is running. A pick that no longer resolves is dropped and announced."""
    import asyncio
    key = _PREFIX + ref.session_id

    def read():
        with authority.db._read_ctx() as conn:
            return conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()
    stored = await asyncio.to_thread(read)
    if stored is None:
        return
    raw = stored[0]
    pick = json.loads(raw)
    if 'reasoning' in pick:
        raw = await _commit_level(authority, ref, key, raw, pick)
        if raw is None:
            return
    if 'awaiting_confirm' in pick:
        if not live.reask_pick:
            return  # asked; the user's confirmed (or a newer) selection supersedes it
        # Asked again for a newly attached viewer: re-prepared under the current fences, so a guard
        # that still applies mints a fresh token (the stored one is never replayed as consent).
        pick = {k: v for k, v in pick.items() if k != 'awaiting_confirm'}
    live.reask_pick = False
    result = await _commit(authority, ref, pick, 'model')
    if result is None:
        return
    if result.get('status') == 'confirmation_required':
        # A guarded target (cost / data policy / large context): nothing was written. The pick stays
        # pending, marked as asked, and the owner's confirmation exchange goes to every viewer; the
        # confirmed resend (``payload.confirm``) commits it through session.mutate.
        from hermes_state_runtime import _json
        await asyncio.to_thread(authority.db._execute_write, lambda conn: conn.execute(
            'UPDATE state_meta SET value=? WHERE key=? AND value=?',
            (_json({**pick, 'awaiting_confirm': result['confirm']}), key, raw)))
        current = await asyncio.to_thread(authority.db.get_session, ref.session_id) or {}
        provider = json.loads(current.get('model_config') or '{}').get('provider')
        # The committed selection rides along, so a viewer that declines paints it back.
        live.event_stream.publish(ref.session_id, {
            **result, 'payload': pick['payload'], 'committed_model': current.get('model'),
            **({'committed_provider': provider} if provider else {})}, event_type='session.updated')
        return
    await asyncio.to_thread(authority.db._execute_write, lambda conn: conn.execute(
        'DELETE FROM state_meta WHERE key=? AND value=?', (key, raw)))
    if result.get('status') == 'refused':
        logger.warning('Deferred model pick on %s not applied: %s', ref.session_id, result.get('reason'))
        # Named for the viewers that showed "applies from your next message": the target that was
        # refused, and the committed model they must show again.
        current = await asyncio.to_thread(authority.db.get_session, ref.session_id) or {}
        provider = json.loads(current.get('model_config') or '{}').get('provider')
        live.event_stream.publish(ref.session_id, {
            **result, 'target_model': pick['payload']['model'], 'model': current.get('model'),
            **({'provider': provider} if provider else {})}, event_type='session.updated')
