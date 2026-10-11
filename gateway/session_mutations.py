"""Action-specific authority mutations, shared by WS and authenticated HTTP.

No legacy slash handler is executed after a receipt: those handlers own separate
transactions. Local reset prepares its replacement in the receipt transaction;
branch/compress/model still require their own prepared runtime publication.
"""
import asyncio
from contextlib import nullcontext

from hermes_state_runtime import RuntimeStoreError, mutate_runtime_session

_METADATA = frozenset({'rename', 'archive', 'sidebar'})
_FIELDS = frozenset({'session_id', 'request_id', 'expected_revision', 'operation', 'payload'})


def _authorize_import(authority, actor, ref, params):
    """Import authorization; returns the imported session ids the write must claim."""
    if actor.profile_id != authority.profile_id or ref.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if 'session:create' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    from hermes_state_mutations import validate_action
    validate_action('import', params['payload'])
    normalized, errors = authority.db._validate_import_payload(params['payload']['sessions'])
    if errors:
        raise RuntimeStoreError('invalid_params')
    return tuple(item['session']['id'] for item in normalized)


def _authorize_control(authority, actor, ref, params):
    """Control authorization; True when the session is cold history the write must claim."""
    from hermes_state_mutation_retirement import has_mutation_receipt
    if actor.profile_id != authority.profile_id or ref.profile_id != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    if 'session:control' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    cold_history = False
    # A receipt authorizes only its original principal's exact retry; the
    # transaction still verifies the entire digest and current epoch.
    if not has_mutation_receipt(authority.db, actor.subject, ref.session_id, params['request_id']):
        ref = _owner_ref(authority, ref, params)
        if ref.session_id not in authority.sessions:
            from hermes_state_mutation_binding import authorize_history
            with authority.db._read_ctx() as conn:
                from gateway.session_local_migration import require_history_claim
                require_history_claim(authority, conn, actor, ref.session_id)
                cold_history = authorize_history(conn, actor, ref.session_id)
        if not cold_history:
            authority.authorize(actor, ref, 'session:control')
    return cold_history


def _owner_ref(authority, ref, params):
    """A delete names the listed row, which may be a later reset segment of a local conversation;
    the conversation's creation id owns its policy, FIFO and live session, so control and the
    session gate are checked on that owner (the store deletes the whole lineage either way)."""
    if params.get('operation') != 'delete':
        return ref
    from gateway.session_contract import SessionRef
    from hermes_state_local import local_lineage_owner
    with authority.db._read_ctx() as conn:
        owner = local_lineage_owner(conn, ref.session_id)
    return ref if owner == ref.session_id else SessionRef(ref.profile_id, owner)


async def mutate_session(authority, actor, ref, params):
    from gateway.session_runtime_workers import track_mutation
    authority._require_admission_open()
    # A request socket can disappear while its writer is inside to_thread. The owner task keeps
    # the session gate until commit AND publication finish, and retirement joins that same task.
    task = track_mutation(authority, _owned_mutation(authority, actor, ref, params))
    return await asyncio.shield(task)


async def _owned_mutation(authority, actor, ref, params):
    if (params.get('operation') in {'model', 'compress', 'reset', 'rewind'}
            and ref.session_id not in authority.sessions and 'request_id' in params):
        _authorize_control(authority, actor, ref, params)
    live = authority.sessions.get(_owner_ref(authority, ref, params).session_id)
    if live is None:
        return await _mutate_session(authority, actor, ref, params)
    # Prepared policy/route publication is inseparable from its durable commit
    # for admission and claim: neither may observe a half-published policy.
    if params.get('operation') == 'compress':
        # The summarization (an LLM call) runs outside the session gate, so a submit or metadata
        # edit is not parked behind it; the commit is revision/generation-fenced against anything
        # that lands meanwhile. Claims wait on ``claim_gate``, so no turn runs under it.
        async with live.claim_gate:
            result = await _mutate_session(authority, actor, ref, params, commit_lock=live.mutation_lock)
    else:
        async with live.mutation_lock:
            result = await _mutate_session(authority, actor, ref, params)
    if params.get('operation') in {'model', 'compress', 'reset'}:
        authority._schedule(ref)
    return result


async def _mutate_session(authority, actor, ref, params, *, commit_lock=None):
    if (set(params) - {'expected_generation'} != _FIELDS
            or params['session_id'] != ref.session_id):
        raise RuntimeStoreError('invalid_params')
    operation = params['operation']
    if not isinstance(operation, str):
        raise RuntimeStoreError('invalid_params')
    cold_history = False
    imported_ids = None
    if operation == 'import':
        imported_ids = _authorize_import(authority, actor, ref, params)
    else:
        cold_history = _authorize_control(authority, actor, ref, params)
    if operation == 'branch' and 'session:create' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    authority._require_admission_open()
    live = authority.sessions.get(ref.session_id)

    applied = False

    def live_guard(targets):
        nonlocal applied
        applied = True
        if operation in _METADATA or operation == 'import':
            return
        # Idleness is the ledger's call (require_idle, same transaction). The drain
        # task outlives the final settlement by one loop iteration, so its liveness
        # would refuse a session whose receipt the client already saw as terminal.
        if operation == 'delete':
            store = getattr(authority.runner, 'session_store', None)
            if store is not None and (store._routing_db is None
                    or store._routing_db.db_path != authority.db.db_path):
                raise RuntimeStoreError('runtime_coordination_required')
        if operation in {'rewind', 'reset'} and not callable(getattr(authority.runner, '_evict_cached_agent', None)):
            raise RuntimeStoreError('runtime_coordination_required')

    def authorize_write(conn):
        from hermes_state_mutation_binding import authorize_history, import_history_control
        if operation == 'import':
            import_history_control(conn, actor, imported_ids)
            return
        if not authorize_history(conn, actor, ref.session_id, claim=True):
            raise RuntimeStoreError('permission_denied')

    prepared = None
    if operation in {'model', 'compress'}:
        # Its own BEGIN IMMEDIATE, so off the loop like the commit below.
        prepared = await asyncio.to_thread(mutate_runtime_session, authority.db, epoch=authority.epoch,
            principal_id=actor.subject, session_id=ref.session_id, request_id=params['request_id'],
            expected_revision=params['expected_revision'], expected_generation=params.get('expected_generation'),
            operation=operation, payload=params['payload'], _live_guard=live_guard, _prepare_only=True)
        if 'snapshot' in prepared:
            from gateway.session_mutation_model import prepare_model
            from gateway.session_mutation_compress import prepare_compress
            prepare = {'model': prepare_model, 'compress': prepare_compress}[operation]
            prepared = await prepare(authority, live, params['payload'], prepared)
            applied = False
            if prepared.get('status') in {'preview', 'confirmation_required'}:
                # ``--preview`` is a read-only report and a guarded model target awaits the user's
                # confirmation: no receipt, no publication, no eviction.
                return {'session_id': ref.session_id, 'operation': operation, **prepared}
    async with commit_lock or nullcontext():
        if prepared is not None and 'snapshot' not in prepared:
            # Exact retry: the durable receipt is the result. Never re-prepare (compress
            # would summarize again); the runtime repairs below still run, because the
            # first attempt may have committed and then failed before publishing them.
            result = prepared
        else:
            # The receipt transaction is synchronous SQLite: run it off the event loop so a large
            # state.db or a contended writer lock never stalls other clients (main 1769024ca3b).
            result = await asyncio.to_thread(mutate_runtime_session, authority.db, epoch=authority.epoch,
                principal_id=actor.subject, session_id=ref.session_id, request_id=params['request_id'],
                expected_revision=params['expected_revision'], expected_generation=params.get('expected_generation'),
                operation=operation, payload=params['payload'], _live_guard=live_guard, _prepared=prepared,
                _authorize_write=authorize_write if cold_history or operation == 'import' else None)
        # Post-commit projections are idempotent reads of the committed receipt, so exact
        # retries repeat them (like delete's retirement); only the one-shot event is fenced.
        current = authority.db.get_session(ref.session_id)
        owns_projection = current is not None and (
            current['runtime_generation'] == result.get('execution_generation', params.get('expected_generation')))
        if operation not in {'model', 'reset', 'compress', 'rewind'} or owns_projection:
            _project_committed(authority, ref, operation, result)
        if live is not None:
            if operation == 'rewind' and owns_projection:
                authority.runner._evict_cached_agent(live.route)
            if applied:
                live.event_stream.publish(ref.session_id, result, event_type='session.updated')
    if operation == 'delete':
        # After the synchronous projections (no yield before retirement), off the loop: the
        # retired-media owner check is a messages-table pass (exact retries repeat it too).
        from hermes_state_media import collect_retired_media
        await asyncio.to_thread(collect_retired_media, authority.db)
    return result


def _project_committed(authority, ref, operation, result):
    """Publish a committed receipt's runtime projections (safe to repeat on exact retry)."""
    if operation == 'model':
        from gateway.session_local import publish_local_policy
        publish_local_policy(authority, ref.session_id)
        # A drain that ran while the receipt committed off-loop saw the new stored policy beside
        # the old live one and paused; with both now equal, queued input must run, not wait for
        # the next submit or restart.
        authority._schedule(ref)
    if operation == 'branch':
        from gateway.session_local_recovery import restore_local_session
        restore_local_session(authority, result['branched_session_id'])
    if operation in {'reset', 'compress'}:
        from gateway.session_local_recovery import restore_local_session
        restore_local_session(authority, ref.session_id)
        # A session boundary, not a resource evict: memory providers get on_session_end first.
        evict = getattr(authority.runner, '_evict_cached_agent_at_boundary', None) or authority.runner._evict_cached_agent
        evict(authority.sessions[ref.session_id].route)
        # Publish the prepared entry, not just its target ID, so fresh-reset and
        # per-session counters match cold recovery in this process too.
        from hermes_state_local import local_receipt
        from gateway.session import SessionEntry
        store = authority.runner.session_store
        with store._lock:
            entry = SessionEntry.from_dict(local_receipt(authority.db, ref.session_id)['entry'])
            entry.origin = authority.sessions[ref.session_id].source
            store._entries[entry.session_key] = entry
    if operation == 'delete':
        # Repeat local retirement on an exact retry too: publication may have
        # failed after the transaction committed. Never repeat the event.
        retire_live_sessions(authority, result['deleted_ids'])


def retire_live_sessions(authority, session_ids):
    """Drop committed deletions from this owner's memory: routes, launch secrets, live sessions
    and their cached agents. Shared by canonical delete and the owner-run bulk prune."""
    store = getattr(authority.runner, 'session_store', None)
    if store is not None:
        store.retire_runtime_sessions(session_ids)
    from gateway.session_policy import release_launch_secrets
    release_launch_secrets(authority, session_ids)
    for sid in session_ids:
        candidate = authority.sessions.pop(sid, None)
        if candidate is not None:
            evict = getattr(authority.runner, '_evict_cached_agent', None)
            if callable(evict):
                evict(candidate.route)
