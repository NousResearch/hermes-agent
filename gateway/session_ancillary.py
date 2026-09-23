"""Read-only Desktop/Ink projections of existing owner state.

No legacy server, manager construction, process recovery, or execution on reads.
"""
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
    return {name: partial(read, connection, kind=kind) for name, kind in {
        'session.control.read': 'control', 'process.list': 'processes',
        'subagent.list': 'subagents', 'subagent.tail': 'tail',
    }.items()}


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
        target = physical_target(authority, ref)
        store = authority.runner.session_store
        with store._lock:
            entry = store._entries.get(live.route)
            if entry is None or entry.session_id != target:
                raise RuntimeStoreError('stale_generation')
        if kind == 'control':
            return {'control': control_snapshot(authority, ref)}
        # Process-local registries cannot describe another interpreter's live
        # objects. Never turn that missing authority into a successful empty list.
        from gateway.session_managed_worker import managed_policy
        if managed_policy(authority, ref) is not None:
            raise RuntimeStoreError('unsupported_projection')
        agent = authority.agent(ref)
        records = subagent_records(agent)
        if kind == 'subagents':
            return {'subagents': [{key: record.get(key) for key in _SUBAGENT_FIELDS}
                                  for record in records], 'delegations': []}
        if kind == 'tail':
            return subagent_tail(records, params['subagent_id'])
        return {'processes': process_snapshot(authority, ref, agent, records)}


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


def process_snapshot(authority, ref, agent, records):
    module = sys.modules.get('tools.process_registry')
    if module is None:
        return []
    registry = module.process_registry
    route = authority.sessions[ref.session_id].route
    target = physical_target(authority, ref)
    owners = {target}
    if agent is not None:
        owners.add(getattr(agent, 'session_id', None))
    owners.update(r.get('subagent_id') for r in records)
    owners.discard(None)
    owners.discard('')
    # list_sessions refreshes recovered processes (and writes checkpoints).
    # Read the current objects directly, keeping ownership and output together.
    with registry._lock:
        processes = [p for p in (*registry._running.values(), *registry._finished.values())
                     if p.session_key in {route, target} and (p.owner_task_id or p.task_id) in owners]
        return [{'session_id': p.id, 'command': p.command[:200],
                 'status': 'exited' if p.exited else 'running', 'exit_code': p.exit_code,
                 'output_tail': p.output_buffer[-4000:]} for p in processes]
