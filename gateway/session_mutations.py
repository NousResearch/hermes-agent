"""Action-specific authority mutations, shared by WS and authenticated HTTP.

No legacy slash handler is executed after a receipt: those handlers own separate
transactions. Local reset prepares its replacement in the receipt transaction;
branch/compress/model still require their own prepared runtime publication.
"""
from hermes_state_runtime import RuntimeStoreError, mutate_runtime_session

_METADATA = frozenset({'rename', 'archive', 'sidebar'})
_FIELDS = frozenset({'session_id', 'request_id', 'expected_revision', 'operation', 'payload'})


async def mutate_session(authority, actor, ref, params):
    if (set(params) - {'expected_generation'} != _FIELDS
            or params['session_id'] != ref.session_id):
        raise RuntimeStoreError('invalid_params')
    operation = params['operation']
    if not isinstance(operation, str):
        raise RuntimeStoreError('invalid_params')
    cold_history = False
    if operation == 'import':
        if actor.profile_id != authority.profile_id or ref.profile_id != authority.profile_id:
            raise RuntimeStoreError('profile_mismatch')
        if 'session:create' not in actor.capabilities:
            raise RuntimeStoreError('permission_denied')
        from hermes_state_mutations import validate_action
        validate_action(operation, params['payload'])
        normalized, errors = authority.db._validate_import_payload(params['payload']['sessions'])
        if errors:
            raise RuntimeStoreError('invalid_params')
        imported_ids = tuple(item['session']['id'] for item in normalized)
    else:
        from hermes_state_mutation_retirement import has_mutation_receipt
        if actor.profile_id != authority.profile_id or ref.profile_id != authority.profile_id:
            raise RuntimeStoreError('profile_mismatch')
        if 'session:control' not in actor.capabilities:
            raise RuntimeStoreError('permission_denied')
        # A receipt authorizes only its original principal's exact retry; the
        # transaction still verifies the entire digest and current epoch.
        if not has_mutation_receipt(authority.db, actor.subject, ref.session_id, params['request_id']):
            if ref.session_id not in authority.sessions:
                from hermes_state_mutation_binding import authorize_history
                with authority.db._read_ctx() as conn:
                    from gateway.session_local_migration import require_history_claim
                    require_history_claim(authority, conn, actor, ref.session_id)
                    cold_history = authorize_history(conn, actor, ref.session_id)
            if not cold_history:
                authority.authorize(actor, ref, 'session:control')
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
        prepared = mutate_runtime_session(authority.db, epoch=authority.epoch,
            principal_id=actor.subject, session_id=ref.session_id, request_id=params['request_id'],
            expected_revision=params['expected_revision'], expected_generation=params.get('expected_generation'),
            operation=operation, payload=params['payload'], _live_guard=live_guard, _prepare_only=True)
        if 'snapshot' in prepared:
            from gateway.session_mutation_model import prepare_model
            from gateway.session_mutation_compress import prepare_compress
            prepare = {'model': prepare_model, 'compress': prepare_compress}[operation]
            prepared = await prepare(authority, live, params['payload'], prepared)
            applied = False
    if prepared is not None and 'snapshot' not in prepared:
        # Exact retry: the durable receipt is the result. Never re-prepare (compress
        # would summarize again); the runtime repairs below still run, because the
        # first attempt may have committed and then failed before publishing them.
        result = prepared
    else:
        result = mutate_runtime_session(authority.db, epoch=authority.epoch,
            principal_id=actor.subject, session_id=ref.session_id, request_id=params['request_id'],
            expected_revision=params['expected_revision'], expected_generation=params.get('expected_generation'),
            operation=operation, payload=params['payload'], _live_guard=live_guard, _prepared=prepared,
            _authorize_write=authorize_write if cold_history or operation == 'import' else None)
    # Post-commit projections are idempotent reads of the committed receipt, so exact
    # retries repeat them (like delete's retirement); only the one-shot event is fenced.
    if operation == 'model':
        from gateway.session_local import publish_local_policy
        publish_local_policy(authority, ref.session_id)
    if operation == 'branch':
        from gateway.session_local_recovery import restore_local_session
        restore_local_session(authority, result['branched_session_id'])
    if operation in {'reset', 'compress'}:
        from gateway.session_local_recovery import restore_local_session
        restore_local_session(authority, ref.session_id)
        authority.runner._evict_cached_agent(authority.sessions[ref.session_id].route)
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
        store = getattr(authority.runner, 'session_store', None)
        if store is not None:
            store.retire_runtime_sessions(result['deleted_ids'])
        for sid in result['deleted_ids']:
            candidate = authority.sessions.pop(sid, None)
            if candidate is not None:
                evict = getattr(authority.runner, '_evict_cached_agent', None)
                if callable(evict):
                    evict(candidate.route)
    if live is not None:
        if operation == 'rewind':
            authority.runner._evict_cached_agent(live.route)
        if applied:
            live.event_stream.publish(ref.session_id, result, event_type='session.updated')
    return result
