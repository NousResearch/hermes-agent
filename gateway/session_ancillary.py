"""Desktop/Ink projections of existing owner state, plus the session-scoped process verbs.

No legacy server, manager construction, process recovery, or execution on reads. ``process.stop``
and ``process.kill`` stop only processes ``process.list`` projects for the session; ``reload.mcp``
reconciles only the requesting profile's MCP servers.
"""
import asyncio
from functools import partial
import sys

from gateway.session_busy_controls import authorize
from hermes_state_runtime import RuntimeStoreError


_SUBAGENT_FIELDS = (
    'subagent_id', 'parent_id', 'depth', 'goal', 'delegation_id', 'model',
    'started_at', 'status', 'tool_count', 'last_tool', 'accepting_steer',
)
_TAIL_BYTES = 16384


def handlers(connection):
    return {**{name: partial(read, connection, kind=kind) for name, kind in {
        'session.control.read': 'control', 'process.list': 'processes',
        'subagent.list': 'subagents', 'subagent.tail': 'tail',
    }.items()}, 'approval.pending': partial(approvals, connection),
        'approval.received': partial(approvals, connection, ack=True),
        'process.stop': partial(stop_processes, connection),
        'process.kill': partial(kill_process, connection), 'reload.mcp': partial(reload_mcp, connection)}


async def approvals(connection, ref, params, *, ack=False):
    """The Desktop's approval replay (``approval.pending``) and card ack (``approval.received``)
    over the owner's generation-bound prompt projection. Left to the legacy sidecar they answered
    ``session not found`` for every authority session, which the renderer reads as a reaped
    runtime and answers with a mid-turn re-resume."""
    fields = {'session_id', 'profile'} | ({'request_id'} if ack else set())
    if (set(params) - fields or not isinstance(ref.session_id, str) or not ref.session_id
            or (ack and (not isinstance(params.get('request_id'), str) or not params['request_id']))):
        raise RuntimeStoreError('invalid_params')
    authority = connection.authority
    if ref.session_id not in authority.sessions:
        raise RuntimeStoreError('not_found')
    authorize(connection, ref, params, 'session:read')
    live = authority.sessions[ref.session_id]
    with live.event_stream.lock:
        pending = [(route, prompt) for route, prompt in live.controls.pending.values() if prompt['kind'] == 'approval']
    if ack:
        from tools.approval import ack_gateway_approval
        route = next((route for route, prompt in pending if prompt['prompt_id'] == params['request_id']), None)
        return {'acknowledged': route is not None and ack_gateway_approval(route, params['request_id'])}
    return {'approvals': [{'request_id': prompt['prompt_id'], 'command': prompt['command'],
                           'description': prompt['description'], 'choices': list(prompt['choices']),
                           'allow_permanent': 'always' in prompt['choices']} for _, prompt in pending]}


async def read(connection, ref, params, *, kind):
    fields = {'session_id', 'profile'} | ({'subagent_id'} if kind == 'tail' else set())
    if (set(params) - fields or not isinstance(ref.session_id, str) or not ref.session_id
            or (kind == 'tail' and (not isinstance(params.get('subagent_id'), str)
                                   or not params['subagent_id']))):
        raise RuntimeStoreError('invalid_params')
    authority = connection.authority
    # Cold adoption belongs to session.resume, never an ancillary poll.
    if ref.session_id not in authority.sessions:
        raise RuntimeStoreError('not_found')
    authorize(connection, ref, params, 'session:read')
    live = authority.sessions[ref.session_id]
    with live.event_stream.lock:
        _require_current_route(authority, ref)
        if kind == 'control':
            return {'control': control_snapshot(authority, ref)}
        agent, records = _owner_objects(authority, ref)
        if kind == 'subagents':
            return {'subagents': [{key: record.get(key) for key in _SUBAGENT_FIELDS}
                                  for record in records], 'delegations': []}
        if kind == 'tail':
            return subagent_tail(records, params['subagent_id'])
        return {'processes': process_snapshot(authority, ref, agent, records)}


def _owned_targets(connection, ref, params, *, select):
    """``(registry, ids)``: the session's own processes (what ``process.list`` shows) that
    ``select(process)`` keeps; ``(None, [])`` when no process was ever registered here."""
    authority = connection.authority
    if ref.session_id not in authority.sessions:
        raise RuntimeStoreError('not_found')
    authorize(connection, ref, params, 'session:control')
    live = authority.sessions[ref.session_id]
    with live.event_stream.lock:
        _require_current_route(authority, ref)
        agent, records = _owner_objects(authority, ref)
        module = sys.modules.get('tools.process_registry')
        if module is None:
            return None, []
        registry = module.process_registry
        keys, owners = _ownership(authority, ref, agent, records)
        with registry._lock:
            return registry, [p.id for p in _owned_locked(registry, keys, owners) if select(p)]


async def kill_process(connection, ref, params):
    """Desktop's per-process Stop: one process THIS session owns (the ``process.list`` rule); any
    other id, another chat's included, is ``not_found``. The legacy sidecar handler looked the
    session up in its own table, which never holds an authority session."""
    if (set(params) - {'session_id', 'process_id', 'profile'} or not ref.session_id
            or not isinstance(params.get('process_id'), str) or not params['process_id']):
        raise RuntimeStoreError('invalid_params')
    registry, ids = _owned_targets(connection, ref, params, select=lambda p: p.id == params['process_id'])
    if not ids:
        raise RuntimeStoreError('not_found')
    return await asyncio.to_thread(registry.kill_process, ids[0])


async def reload_mcp(connection, ref, params):
    """Desktop's MCP add / toggle / repair: bring THIS profile's live MCP servers in step with its
    config (the owner's housekeeping reconcile, run now) and refresh this profile's cached agents.
    Never the sidecar's teardown of every server, which cut every chat's in-flight MCP calls."""
    if set(params) - {'session_id', 'confirm', 'rev', 'profile'}:
        raise RuntimeStoreError('invalid_params')
    if 'session:control' not in connection.actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    if params.get('confirm') is not True:
        return {'status': 'confirm_required', 'message': 'Reloading MCP refreshes every chat of this profile.'}
    from tools.mcp_oauth import suppress_interactive_oauth
    from tools.mcp_tool_discovery import reconcile_mcp_servers_with_config

    def reconcile():
        with suppress_interactive_oauth():
            return reconcile_mcp_servers_with_config()
    result = await asyncio.to_thread(reconcile)
    from gateway.session_authorities import served_profile_name
    runner = connection.authority.runner
    multiplex = bool(getattr(runner.config, 'multiplex_profiles', False))
    runner._mcp_reload_refresh_cached_agents(multiplex, served_profile_name(connection.authority.profile_id))
    changed = ', '.join(f'{k}: {", ".join(v)}' for k, v in result.items() if v) or 'no server changes'
    return {'status': 'reloaded', 'message': f'MCP servers reconciled ({changed})'}


async def stop_processes(connection, ref, params):
    """Ink ``/stop``: kill the background processes this session owns (what ``process.list``
    shows), never the registry-wide ``kill_all`` the legacy sidecar ran, which on the shared owner
    reached every chat, Desktop window and served profile (dokterdok N2). An explicit stop, so a
    ``persist_on_release`` job of THIS session is still reached (#41225)."""
    if set(params) - {'session_id', 'profile'} or not isinstance(ref.session_id, str) or not ref.session_id:
        raise RuntimeStoreError('invalid_params')
    registry, targets = _owned_targets(connection, ref, params,
                                       select=lambda p: not p.exited or p._scope_stop_pending)

    def kill():
        results = [registry.kill_process(pid, source='process.stop', consume_output=False) for pid in targets]
        return sum(r.get('status') in {'killed', 'already_exited'} and 'scope_stop_failed' not in r
                   for r in results)
    return {'killed': await asyncio.to_thread(kill) if targets else 0}


def _require_current_route(authority, ref):
    """The route still serves this conversation (a reused route must not lend its objects)."""
    live = authority.sessions[ref.session_id]
    target = physical_target(authority, ref)
    store = authority.runner.session_store
    with store._lock:
        entry = store._entries.get(live.route)
        if entry is None or entry.session_id != target:
            raise RuntimeStoreError('stale_generation')


def _owner_objects(authority, ref):
    """``(agent, subagent records)`` of this in-process session. Process-local registries cannot
    describe (or stop) another interpreter's live objects: a managed worker is refused rather than
    answered with an empty success."""
    from gateway.session_managed_worker import managed_policy
    if managed_policy(authority, ref) is not None:
        raise RuntimeStoreError('unsupported_projection')
    agent = authority.agent(ref)
    return agent, subagent_records(agent)


def physical_target(authority, ref):
    from gateway.config import Platform
    if authority.sessions[ref.session_id].source.platform == Platform.LOCAL:
        from hermes_state_local import local_receipt
        return local_receipt(authority.db, ref.session_id)['entry']['session_id']
    return ref.session_id


def control_snapshot(authority, ref):
    from hermes_cli.goals import GoalState
    from hermes_cli.loops import LoopState
    from hermes_cli.heartbeat import HeartbeatState
    # These pure serializers don't bind/import the legacy server. In particular
    # do not reuse _snapshot_control: its manager loader can write on a read.
    from tui_gateway.methods_session_control import (
        _safe_goal_snapshot, _safe_loop_snapshot, _safe_heartbeat_snapshot,
        _snapshot_revision, _snapshot_updated_at,
    )
    target = physical_target(authority, ref)
    states = []
    for kind, cls in [('goal', GoalState), ('loop', LoopState), ('heartbeat', HeartbeatState)]:
        raw = authority.db.get_meta(kind + ':' + target)
        try:
            states.append(cls.from_json(raw) if raw else None)
        except (ValueError, TypeError, AttributeError) as exc:
            raise RuntimeStoreError('storage_unavailable') from exc
    goal_state, loop_state, heartbeat_state = states
    goal = _safe_goal_snapshot(goal_state)
    # Project the persisted barrier, not GoalManager.is_waiting(), which clears
    # satisfied barriers and can query unrelated processes during a UI refresh.
    deferred = bool(goal and goal['status'] == 'active' and not goal.get('wait_barrier'))
    loop = _safe_loop_snapshot(loop_state, deferred_by_goal=bool(
        deferred and loop_state and loop_state.status == 'active'))
    heartbeat = _safe_heartbeat_snapshot(heartbeat_state)
    return {'goal': goal, 'loop': loop, 'heartbeat': heartbeat,
            'revision': _snapshot_revision(goal, loop, heartbeat),
            'updated_at': _snapshot_updated_at(*states)}


def subagent_records(parent):
    registry = sys.modules.get('tools.delegate_tool_registry')
    if parent is None or registry is None:
        return []
    with registry._active_subagents_lock:
        # Deliberately no durable-ID fallback: equal IDs cannot transfer a live
        # old parent's transcript to a replacement execution object.
        return [dict(record) for record in registry._active_subagents.values()
                if registry._is_descendant_of(record.get('agent'), parent)]


def subagent_tail(records, subagent_id):
    result = {'subagent_id': subagent_id, 'available': False, 'text': '', 'truncated': False}
    record = next((r for r in records if r.get('subagent_id') == subagent_id), None)
    path = getattr(record.get('agent'), '_live_transcript_path', None) if record else None
    if not path:
        return result
    try:
        with open(path, 'rb') as stream:
            size = stream.seek(0, 2)
            stream.seek(max(0, size - _TAIL_BYTES))
            text = stream.read(_TAIL_BYTES).decode('utf-8', errors='ignore')
    except OSError:
        return result
    return {**result, 'available': True, 'text': text, 'truncated': size > _TAIL_BYTES}


def _ownership(authority, ref, agent, records):
    """``(routing keys, owners)`` a registry process must match to belong to this session: its
    routing key AND an owner among the session, its agent and its subagents."""
    target = physical_target(authority, ref)
    keys, owners = {authority.sessions[ref.session_id].route, target}, {target}
    if agent is not None:
        owners.add(getattr(agent, 'session_id', None))
    owners.update(r.get('subagent_id') for r in records)
    owners.discard(None)
    owners.discard('')
    return keys, owners


def _owned_locked(registry, keys, owners):
    """Matching process objects; the caller holds ``registry._lock``. Read directly:
    ``list_sessions`` refreshes recovered processes and writes checkpoints."""
    return [p for p in (*registry._running.values(), *registry._finished.values())
            if p.session_key in keys and (p.owner_task_id or p.task_id) in owners]


def process_snapshot(authority, ref, agent, records):
    module = sys.modules.get('tools.process_registry')
    if module is None:
        return []
    registry = module.process_registry
    keys, owners = _ownership(authority, ref, agent, records)
    # Keep ownership and output together.
    with registry._lock:
        return [{'session_id': p.id, 'command': p.command[:200],
                 'status': 'exited' if p.exited else 'running', 'exit_code': p.exit_code,
                 'output_tail': p.output_buffer[-4000:]} for p in _owned_locked(registry, keys, owners)]
