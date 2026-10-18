"""Action-specific authority mutations, shared by WS and authenticated HTTP.

No legacy slash handler is executed after a receipt: those handlers own separate
transactions. Local reset prepares its replacement in the receipt transaction;
branch/compress/model still require their own prepared runtime publication.
"""
import asyncio
import logging
from contextlib import nullcontext
from functools import partial

from hermes_state_mutation_guards import MUTATION_GUARD_REFUSALS
from hermes_state_runtime import RuntimeStoreError, mutate_runtime_session

logger = logging.getLogger(__name__)

_METADATA = frozenset({'rename', 'archive', 'sidebar'})
# Prepared off the commit like model, then published as a new frozen policy.
_POLICY = frozenset({'model', 'reasoning', 'workspace', 'editor_mcp'})
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
    if (params.get('operation') in {*_POLICY, 'compress', 'reset', 'rewind'}
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
            try:
                result = await _mutate_session(authority, actor, ref, params)
            except (RuntimeStoreError, *MUTATION_GUARD_REFUSALS) as exc:
                # A pick on a running session is for the next turn (the running turn keeps its route).
                if params.get('operation') != 'model' or getattr(exc, 'reason', 'session_busy') != 'session_busy':
                    raise
                from gateway.session_deferred_pick import defer_pick
                # Its own BEGIN IMMEDIATE: off the loop, like the receipt commit.
                return await asyncio.to_thread(defer_pick, authority, actor, ref, params)
    if params.get('operation') in {*_POLICY, 'compress', 'reset'}:
        authority._schedule(ref)
    return result


async def _mutate_session(authority, actor, ref, params, *, commit_lock=None):
    if (set(params) - {'expected_generation'} != _FIELDS
            or params['session_id'] != ref.session_id):
        raise RuntimeStoreError('invalid_params')
    operation = params['operation']
    if not isinstance(operation, str) or (operation in _POLICY and params.get('expected_generation') is None):
        raise RuntimeStoreError('invalid_params')
    from gateway.session_local_mcp import private_mutation_payload
    # Editor MCP transport specs (credentials) stay in owner memory; the store sees a fingerprint.
    payload, editor_servers = private_mutation_payload(operation, params['payload'])
    params = {**params, 'payload': payload}
    cold_history = False
    imported_ids = None
    if operation == 'import':
        imported_ids = _authorize_import(authority, actor, ref, params)
    else:
        cold_history = _authorize_control(authority, actor, ref, params)
    if operation == 'branch' and 'session:create' not in actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    authority._require_admission_open()

    def authorize_write(conn):
        from hermes_state_mutation_binding import authorize_history, import_history_control
        if operation == 'import':
            import_history_control(conn, actor, imported_ids)
            return
        if not authorize_history(conn, actor, ref.session_id, claim=True):
            raise RuntimeStoreError('permission_denied')

    return await write_mutation(authority, ref, params, editor_servers, principal_id=actor.subject,
                                authorize_write=authorize_write if cold_history or operation == 'import' else None,
                                commit_lock=commit_lock)


def _runtime_owns_authority(authority) -> bool:
    """Whether *authority* is one of its own runner's authorities, so one process coordinates
    both writes of a delete: the receipt commit to ``authority.db`` and the routing retire
    through the runner store's routing index. Raw ``db_path`` equality only ever holds for the
    launch home — the routing index is pinned to the launch home for every profile (#66887) —
    so a multiplexed secondary profile's store never compares equal and each of its deletes
    was refused with ``runtime_coordination_required`` (#138826)."""
    runner = getattr(authority, 'runner', None)
    registry = getattr(runner, 'session_authorities', None)
    if registry is not None:
        return any(a is authority for a in registry)
    return getattr(runner, 'session_authority', None) is authority


async def write_mutation(authority, ref, params, editor_servers, *, principal_id, authorize_write=None,
                         commit_lock=None):
    """The writer phase of ``session.mutate``, and of the admission-start editor MCP re-freeze
    (``refreeze_editor_mcp``): fenced prepare, the receipt commit, then its runtime projections.
    The caller authorized the change and holds the session's locks; this takes neither."""
    operation = params['operation']
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
                    or not _runtime_owns_authority(authority)):
                raise RuntimeStoreError('runtime_coordination_required')
        if operation in {'rewind', 'reset'} and not callable(getattr(authority.runner, '_evict_cached_agent', None)):
            raise RuntimeStoreError('runtime_coordination_required')

    fields = dict(epoch=authority.epoch, principal_id=principal_id, session_id=ref.session_id,
                  request_id=params['request_id'], expected_revision=params['expected_revision'],
                  expected_generation=params.get('expected_generation'), operation=operation,
                  payload=params['payload'], _live_guard=live_guard)

    async def prepare_only():
        # Its own BEGIN IMMEDIATE, so off the loop like the commit below.
        return await asyncio.to_thread(mutate_runtime_session, authority.db, **fields, _prepare_only=True)

    prepared = None
    if operation == 'branch':
        from gateway.session_local_mcp import prepare_branch_editor
        # Discovery launches the child's servers: the fences (receipt, revision, generation, idle) answer first.
        prepared, editor_servers = await prepare_branch_editor(authority, ref, editor_servers, prepare_only)
    if operation in {*_POLICY, 'compress'}:
        prepared = await prepare_only()
        if 'snapshot' in prepared:
            from gateway.session_mutation_model import prepare_model
            from gateway.session_mutation_compress import prepare_compress
            from gateway.session_mutation_policy import PREPARE
            from gateway.session_local_mcp import prepare_editor_mcp
            prepare = {'model': prepare_model, 'compress': prepare_compress, **PREPARE,
                       'editor_mcp': partial(prepare_editor_mcp, servers=editor_servers)}[operation]
            prepared = await prepare(authority, live, params['payload'], prepared)
            if prepared.get('status') in {'preview', 'noop', 'confirmation_required'}:
                # ``--preview`` and a compress that removes nothing are read-only reports, and a
                # guarded model target awaits the user's confirmation: no receipt, no publication,
                # no eviction.
                return {'session_id': ref.session_id, 'operation': operation, **prepared}
    applied = False
    async with commit_lock or nullcontext():
        if prepared is not None and 'snapshot' not in prepared:
            # Exact retry: the durable receipt is the result. Never re-prepare (compress
            # would summarize again); the runtime repairs below still run, because the
            # first attempt may have committed and then failed before publishing them.
            result = prepared
        else:
            # The receipt transaction is synchronous SQLite: run it off the event loop so a large
            # state.db or a contended writer lock never stalls other clients (main 1769024ca3b).
            result = await asyncio.to_thread(mutate_runtime_session, authority.db, **fields,
                                             _prepared=prepared, _authorize_write=authorize_write)
        await _project_result(authority, ref, live, params, result, editor_servers, applied=applied)
    if operation == 'compress' and applied and (prepared or {}).get('post_compaction'):
        from gateway.session_mutation_compress import fire_post_compaction_hook
        await fire_post_compaction_hook(authority, prepared, result)
    if operation == 'delete':
        # After the synchronous projections (no yield before retirement), off the loop: the
        # retired-media owner check is a messages-table pass (exact retries repeat it too).
        from hermes_state_media import collect_retired_media
        await asyncio.to_thread(collect_retired_media, authority.db)
    return result


async def _project_result(authority, ref, live, params, result, editor_servers, *, applied):
    """A committed receipt's projections, under the writer's commit lock. They are idempotent reads
    of the committed receipt, so exact retries repeat them (like delete's retirement); only the
    one-shot event is fenced."""
    operation = params['operation']
    current = authority.db.get_session(ref.session_id)
    owns_projection = current is not None and (
        current['runtime_generation'] == result.get('execution_generation', params.get('expected_generation')))
    if operation == 'compress' and applied and owns_projection and live is not None:
        # Before the boundary evicts the agent whose compressor holds the running count.
        from gateway.session_usage import note_compression
        note_compression(authority, ref, live)
    elif operation == 'reset' and applied and live is not None:
        live.compression_count = 0  # a fresh conversation, as main's /new builds a fresh agent
    if editor_servers is not None and (operation == 'branch' or owns_projection):
        # Before the policy publication below, so no execution sees the policy without them.
        from gateway.session_local_mcp import install_editor_mcp
        install_editor_mcp(authority, result.get('branched_session_id', ref.session_id), editor_servers)
        if operation == 'editor_mcp' and not editor_servers:
            # An empty selection: the old servers must not keep running for a viewer.
            from gateway.session_local_mcp import retire_editor_mcp
            await asyncio.to_thread(retire_editor_mcp, authority, ref.session_id)
    if operation not in {*_POLICY, 'reset', 'compress', 'rewind'} or owns_projection:
        _project_committed(authority, ref, operation, result)
    if live is not None:
        if operation == 'rewind' and owns_projection:
            authority.runner._evict_cached_agent(live.route)
        if applied:
            live.event_stream.publish(ref.session_id, result, event_type='session.updated')
            from gateway.session_local_mcp import announce_tools_changed
            announce_tools_changed(live, ref.session_id, result)


def _project_committed(authority, ref, operation, result):
    """Publish a committed receipt's runtime projections (safe to repeat on exact retry)."""
    if operation in _POLICY:
        from gateway.session_local import publish_local_policy
        publish_local_policy(authority, ref.session_id)
        # A drain that ran while the receipt committed off-loop saw the new stored policy beside
        # the old live one and paused; with both now equal, queued input must run, not wait for
        # the next submit or restart.
        authority._schedule(ref)
    if operation == 'reasoning' or (operation == 'model' and 'effort' in result):
        # The picked level is the policy's now; a typed /reasoning override must not shadow it.
        authority.runner._set_session_reasoning_override(authority.sessions[ref.session_id].route, None)
    if operation in {'model', 'reasoning', 'workspace'}:
        # Every viewer repaints its model, reasoning pill or workspace from the published policy,
        # as main's _emit_session_info did after the live change.
        from gateway.session_local import local_session_info
        authority.sessions[ref.session_id].event_stream.publish(
            ref.session_id, local_session_info(authority, ref), event_type='session.info')
    if operation == 'workspace':
        # Main's move drops the terminal environment so the next command starts in the new folder.
        from hermes_state_local import local_receipt
        from tools.terminal_tool_lifecycle import cleanup_vm
        for sid in {ref.session_id, local_receipt(authority.db, ref.session_id)['entry']['session_id']}:
            cleanup_vm(sid)
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
        retire_live_sessions(authority, result['deleted_ids'], require_fence=True)


def retire_live_sessions(authority, session_ids, *, require_fence=False):
    """Drop committed deletions from this owner's memory: routes, launch secrets, live sessions
    and their cached agents. Shared by canonical delete and the owner-run bulk prune.
    ``require_fence``: refuse ``storage_unavailable`` when the shared routing fence could not be
    written (canonical delete, whose exact retry replays the receipt and repeats this call)."""
    store = getattr(authority.runner, 'session_store', None)
    fence_error = None
    if store is not None:
        store.retire_runtime_sessions(session_ids)
        # The delete transaction fenced and dropped routes in the owning store; a multiplexed
        # secondary's routes live in the launch home's shared index, so the store fences them
        # there too (fences + drop in one transaction). Without the fence in THAT database a
        # whole-index snapshot taken before this delete and saved after it restored the route
        # (#138826). The fence is keyed by this profile's routing namespace, so the launch
        # profile never takes it for its own deletion, and it ages out with a sweep's marker.
        # A failure surfaces after the rest of the cleanup: an exact retry replays the
        # committed receipt and repeats this fence.
        routing = getattr(store, '_routing_db', None)
        if session_ids and routing is not None and routing.db_path != authority.db.db_path:
            try:
                from gateway.session import _session_key_namespace
                from hermes_state_mutation_retirement import expiring_retirements
                registry = getattr(authority.runner, 'session_authorities', None)
                profile = registry.profile_name(authority) if registry is not None else None
                store.fence_retired_routes(
                    session_ids, namespace=_session_key_namespace(profile).split(':', 1)[1],
                    expiring=expiring_retirements(authority.db, session_ids))
            except Exception as exc:
                logger.warning('could not fence shared routing rows for deleted sessions', exc_info=True)
                fence_error = exc
        # The dropped routes stay in sessions.json until the next full save, which an idle gateway
        # may not make before a prune's fence ages out; the next start would then import them back.
        # Strip them now, after the fences, whatever write_sessions_json says.
        strip = getattr(store, 'strip_retired_mirror_routes', None)
        if session_ids and callable(strip):
            try:
                strip(session_ids)
            except Exception:
                logger.warning('could not strip retired routes from sessions.json', exc_info=True)
    from gateway.session_policy import release_launch_secrets
    release_launch_secrets(authority, session_ids)
    from tools.async_delegation import allow_for_session
    for sid in session_ids:
        # A closed session's refusal of new delegations (session_viewer_close) goes with its live state.
        allow_for_session((authority.profile_id, sid))
        candidate = authority.sessions.pop(sid, None)
        if candidate is not None and not candidate.superseded:  # its route is the newer session's
            evict = getattr(authority.runner, '_evict_cached_agent', None)
            if callable(evict):
                evict(candidate.route)
    from gateway.session_local_mcp import release_editor_mcp
    release_editor_mcp(authority, session_ids)
    from gateway.session_worker_construct import release_session_prefill
    release_session_prefill(authority, session_ids)
    if fence_error is not None and require_fence:
        raise RuntimeStoreError('storage_unavailable') from fence_error
