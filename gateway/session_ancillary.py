"""Desktop/Ink projections of existing owner state, plus the session-scoped process verbs.

No legacy server, manager construction, process recovery, or execution on reads. ``process.stop``
and ``process.kill`` stop only processes ``process.list`` projects for the session; ``reload.mcp``
reconciles only the requesting profile's MCP servers; ``file.attach`` stages for one owned session.
"""
import asyncio
from functools import partial
from pathlib import Path
import sys

from gateway.session_busy_controls import authorize
from hermes_state_runtime import RuntimeStoreError


_SUBAGENT_FIELDS = (
    'subagent_id', 'parent_id', 'depth', 'goal', 'delegation_id', 'model',
    'started_at', 'status', 'tool_count', 'last_tool', 'accepting_steer',
)
_TAIL_BYTES = 16384
_PROCESS_TAIL_CHARS = 32768  # main's dock tail view (cli_process_dock.process_tail)


def handlers(connection):
    return {**{name: partial(read, connection, kind=kind) for name, kind in {
        'session.control.read': 'control', 'process.list': 'processes',
        'subagent.list': 'subagents', 'subagent.tail': 'tail',
    }.items()}, 'approval.pending': partial(approvals, connection),
        'approval.received': partial(approvals, connection, ack=True),
        'process.stop': partial(stop_processes, connection),
        'process.kill': partial(kill_process, connection), 'reload.mcp': partial(reload_mcp, connection),
        'file.attach': partial(attach_file, connection), 'tools.show': partial(show_tools, connection),
        'skills.reload': partial(reload_skills, connection),
        **{name: partial(subagent_control, connection, steer=name == 'subagent.steer')
           for name in ('subagent.steer', 'subagent.interrupt')}}


async def subagent_control(connection, ref, params, *, steer):
    """Desktop's per-child Steer / Stop: only a child of THIS session's live agent (the
    ``subagent.list`` rule). The sidecar handler never held an authority session."""
    fields = {'session_id', 'profile', 'subagent_id'} | ({'text'} if steer else set())
    if (set(params) - fields or not ref.session_id or any(not isinstance(params.get(k), str) or not params[k]
                                                           for k in fields - {'profile'})):
        raise RuntimeStoreError('invalid_params')
    if ref.session_id not in connection.authority.sessions:
        raise RuntimeStoreError('not_found')
    authorize(connection, ref, params, 'session:control')
    live, sub = connection.authority.sessions[ref.session_id], params['subagent_id']
    with live.event_stream.lock:
        _require_current_route(connection.authority, ref)
        child = next((r['agent'] for r in _owner_objects(connection.authority, ref)[1] if r.get('subagent_id') == sub), None)
    if not steer:
        from agent.interrupt_compat import request_hard_interrupt
        return {'found': child is not None and bool(request_hard_interrupt(child, f'Interrupted via Desktop ({sub})')),
                'subagent_id': sub}
    from tools import delegate_tool_registry as registry
    with registry._active_subagents_lock:  # linearized with the child's final steer drain
        record = registry._active_subagents.get(sub)
        queued = bool(child is not None and record and record.get('agent') is child and record.get('accepting_steer')
                      and child.steer(params['text']))
    return {'status': 'queued' if queued else 'rejected', 'subagent_id': sub, 'text': params['text']}


async def show_tools(connection, ref, params):
    """``/tools``: the session's catalog as main's CLI listed it (``get_tool_definitions`` with the
    session's toolsets, pre tool_search assembly), resolved in the owner, where the MCP and plugin
    tools live. Rows carry the first description sentence, grouped by toolset."""
    if set(params) - {'session_id', 'profile'} or not isinstance(ref.session_id, str) or not ref.session_id:
        raise RuntimeStoreError('invalid_params')
    authority = connection.authority
    if ref.session_id not in authority.sessions:
        raise RuntimeStoreError('not_found')
    authorize(connection, ref, params, 'session:read')
    from gateway.session_authorities import owner_scope
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    agent = authority.agent(ref)

    def build():
        import model_tools
        from agent.skill_utils import parse_config_string_list
        from toolsets import session_disabled_toolsets
        with owner_scope(authority):
            if agent is not None:
                enabled, disabled = getattr(agent, 'enabled_toolsets', None), getattr(agent, 'disabled_toolsets', None)
            else:
                config = policy.config(authority, secrets=False) if policy is not None else {}
                enabled = list(policy.current_toolsets(config)) if policy is not None else None
                disabled = session_disabled_toolsets(parse_config_string_list(
                    (config.get('agent') or {}).get('disabled_toolsets')) or None,
                    policy.platform if policy is not None else 'cli',
                    oneshot=policy is not None and policy.source == 'oneshot')
            tools = model_tools.get_tool_definitions(enabled_toolsets=enabled, disabled_toolsets=disabled,
                                                     quiet_mode=True, skip_tool_search_assembly=True)
            sections = {}
            for tool in sorted(tools or [], key=lambda item: item['function']['name']):
                name = tool['function']['name']
                desc = str(tool['function'].get('description', '') or '').split('\n')[0]
                if '. ' in desc:
                    desc = desc[:desc.index('. ') + 1]
                sections.setdefault(model_tools.get_toolset_for_tool(name) or 'unknown', []).append(
                    {'name': name, 'description': desc})
        return {'sections': [{'name': n, 'tools': rows} for n, rows in sorted(sections.items())],
                'total': len(tools or [])}
    return await asyncio.to_thread(build)


async def reload_skills(connection, ref, params):
    """``/reload-skills`` (and the CLI's ``/skills install``): rescan the skill dirs in the owner,
    whose skill-command map serves this session's skill and bundle invocations. The CLI renders
    main's lines from ``result`` and queues main's next-turn note itself."""
    if set(params) - {'session_id', 'profile'} or not isinstance(ref.session_id, str) or not ref.session_id:
        raise RuntimeStoreError('invalid_params')
    authority = connection.authority
    if ref.session_id not in authority.sessions:
        raise RuntimeStoreError('not_found')
    authorize(connection, ref, params, 'session:control')
    from gateway.session_authorities import owner_scope
    from gateway.session_policy import policy_for_source, policy_scope
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)

    def rescan():
        from agent.skill_commands import reload_skills as rescan_skills
        with owner_scope(authority), policy_scope(policy, authority=authority):
            return rescan_skills()
    result = await asyncio.to_thread(rescan)
    added, removed = result.get('added') or [], result.get('removed') or []
    lines = ['Reloading skills...'] + ([] if added or removed else ['No new skills detected.'])
    for label, items in (('Added skills:', added), ('Removed skills:', removed)):
        if items:
            lines.append(label)
            lines.extend(f"  - {item.get('name', '')}" for item in items)
    lines.append(f"{int(result.get('total') or 0)} skill(s) available")
    return {'output': '\n'.join(lines), 'result': result}


async def attach_file(connection, ref, params):
    """Desktop's non-image attachment: stage the file for this session (its frozen launch cwd and
    the owner profile's ``attachments/``) and hand back the ``@file:`` ref its next submit carries.
    The sidecar handler looked the session up in its own table, which never holds an authority
    session, so every file drop failed with ``session not found``.

    A browser pick arrives as ``staged_upload`` (the Webapp upload route's receipt,
    ``hermes_cli/staged_uploads.py``): it is copied out of the profile it was uploaded to, which must
    be one this connection may reach."""
    staged_upload = params.get('staged_upload')
    if (set(params) - {'session_id', 'profile', 'path', 'data_url', 'name', 'staged_upload'} or not ref.session_id
            or any(not isinstance(value, str) for key, value in params.items() if key != 'staged_upload')
            or (staged_upload is not None and not isinstance(staged_upload, dict))):
        raise RuntimeStoreError('invalid_params')
    if ref.session_id not in connection.subscriptions:
        raise RuntimeStoreError('permission_denied')
    authorize(connection, ref, params, 'session:submit')
    if staged_upload is not None and not _reachable_upload_home(connection, staged_upload.get('profile_home')):
        raise RuntimeStoreError('permission_denied')
    from gateway.session_policy import policy_for_source
    authority = connection.authority
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    if policy is None:
        raise RuntimeStoreError('permission_denied')
    # The staging helpers are bound onto the sidecar module's globals (method_ctx.bind_module).
    from tui_gateway import server
    session = {'cwd': policy.cwd, 'profile_home': authority.profile_id}

    def stage():
        if staged_upload is not None:
            # Only the receipt is read: a supplied path/data_url never stands in for a missing upload.
            return server._stage_browser_file_attachment(session, staged_upload, params.get('name', ''))
        return server._stage_session_file_attachment(
            session, raw_path=params.get('path', ''), data_url=params.get('data_url', ''),
            name=params.get('name', ''))
    try:
        stored, uploaded = await asyncio.to_thread(stage)
    except (ValueError, OverflowError) as exc:
        raise RuntimeStoreError('invalid_params') from exc
    except OSError as exc:
        raise RuntimeStoreError('storage_unavailable') from exc
    ref_path = server._attachment_ref_path(session, stored)
    return {'attached': True, 'name': stored.name, 'path': str(stored), 'ref_path': ref_path,
            'ref_text': f'@file:{server._format_ref_value(ref_path)}', 'uploaded': uploaded}


def _reachable_upload_home(connection, raw_home) -> bool:
    """A staged upload's source profile: this authority's own home or, for a host-scoped grant, a
    sibling this process serves (an owner-routed tile can receive a foreground browser pick). Never
    any other profile on disk, whatever its descriptor names."""
    from pathlib import Path
    if not isinstance(raw_home, str) or not raw_home:
        return False
    homes = {connection.authority.profile_id}
    if connection.host_scoped:
        registry = getattr(getattr(connection.authority, 'runner', None), 'session_authorities', None)
        homes |= {sibling.profile_id for sibling in (registry or ())}
    try:
        target = Path(raw_home).resolve()
        return any(Path(home).resolve() == target for home in homes)
    except (OSError, ValueError):
        return False


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
        if kind == 'control':
            # Persisted per session id, never the route's live objects: readable after a messaging
            # route moved on to another conversation.
            return {'control': control_snapshot(authority, ref)}
        _require_current_route(authority, ref)
        agent, records = _owner_objects(authority, ref, include_fallback_evicted=kind == 'subagents')
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
    other id, another chat's included, is ``process_not_found``. The legacy sidecar handler looked the
    session up in its own table, which never holds an authority session."""
    if (set(params) - {'session_id', 'process_id', 'profile'} or not ref.session_id
            or not isinstance(params.get('process_id'), str) or not params['process_id']):
        raise RuntimeStoreError('invalid_params')
    registry, ids = _owned_targets(connection, ref, params, select=lambda p: p.id == params['process_id'])
    if not ids:
        # Never ``not_found``: Desktop reads that reason as "this chat is gone" (dokterdok N64).
        raise RuntimeStoreError('process_not_found')
    return await asyncio.to_thread(registry.kill_process, ids[0])


async def reload_mcp(connection, ref, params):
    """Desktop's MCP add / toggle / repair: bring THIS profile's live MCP servers in step with its
    config (the owner's housekeeping reconcile, run now) and refresh this profile's cached agents.
    Never the sidecar's teardown of every server, which cut every chat's in-flight MCP calls. The calling
    session's MCP servers then change as a new policy generation when it is idle (``session``: applied,
    unchanged or busy); other sessions take theirs when next attached. A configured server that did not
    connect is named: a refusal when nothing took effect (no server registered or removed, no session
    generation), else ``unavailable`` beside the success. A call naming ``rev``, the revision the session
    already holds, is that same nothing-left-to-load, answered as success with ``loaded_rev``."""
    if set(params) - {'session_id', 'confirm', 'rev', 'profile'}:
        raise RuntimeStoreError('invalid_params')
    if 'session:control' not in connection.actor.capabilities:
        raise RuntimeStoreError('permission_denied')
    if params.get('confirm') is not True:
        return {'status': 'confirm_required', 'message': 'Reloading MCP refreshes every chat of this profile.'}
    from tools.mcp_oauth import suppress_interactive_oauth
    from tools import mcp_tool_discovery as discovery

    def reconcile():
        with suppress_interactive_oauth():
            return (discovery.reconcile_mcp_servers_with_config(), discovery.mcp_servers_not_connected(),
                    discovery.mcp_servers_ready())
    result, failed, ready = await asyncio.to_thread(reconcile)
    # ``added`` is what reconcile attempted: only those now registered (live or lazy) changed anything.
    result = {**result, 'added': [name for name in result.get('added', []) if name not in failed]}
    from gateway.session_authorities import served_profile_name
    runner = connection.authority.runner
    multiplex = bool(getattr(runner.config, 'multiplex_profiles', False))
    runner._mcp_reload_refresh_cached_agents(multiplex, served_profile_name(connection.authority.profile_id))
    session = await _commit_session_mcp(connection, ref, params)
    held_rev = await _held_mcp_rev(connection, ref) if params.get('rev') else ''
    # Refuse only a reload whose own work did not take effect: a call that names the revision the session
    # already holds has nothing left to load, and refusing it leaves a client that acknowledges on
    # ``reloaded`` retrying a parked server at poll rate with no way to close the handshake (#141769).
    if (failed and session != 'applied' and not (result.get('removed') or result['added'])
            and params.get('rev') != held_rev):
        raise RuntimeStoreError('mcp_servers_unavailable', field='mcp_servers',
                                detail=f'did not connect: {", ".join(failed)}')
    changed = ', '.join(f'{k}: {", ".join(v)}' for k, v in result.items() if v) or 'no server changes'
    note = {'applied': '; this chat\'s tools changed for its next turn',
            'busy': '; this chat is busy: its new tools apply once the turn finishes and reload runs again '
                    'or the chat is reopened'}.get(session or '', '')
    unavailable = f'; did not connect: {", ".join(failed)}' if failed else ''
    selected = {'session_servers': await asyncio.to_thread(_session_servers, connection.authority, ref.session_id,
                                                           ready)} if session else {}
    return {'status': 'reloaded', 'message': f'MCP servers reconciled ({changed}){note}{unavailable}',
            **({'loaded_rev': held_rev} if held_rev else {}),
            **({'session': session} if session else {}), **({'unavailable': failed} if failed else {}), **selected}


async def _held_mcp_rev(connection, ref):
    """The ``mcp_rev`` this session's committed policy already holds — the value ``config.get mtime`` shipped
    the client — or ``''`` when the call names no live local session. The policy is read on the loop (it
    walks ``authority.sessions`` and the adapter's maps, which only the loop mutates); only the config read
    and hash go to a worker thread."""
    sid = getattr(ref, 'session_id', None)
    live = connection.authority.sessions.get(sid) if sid else None
    if live is None or live.source is None:
        return ''
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(connection.authority.runner, live.source)
    return await asyncio.to_thread(_mcp_revision_in_scope, Path(connection.authority.profile_id), policy)


def _mcp_revision_in_scope(home, policy):
    from gateway.run import _profile_runtime_scope
    from gateway.session_config import mcp_revision
    from hermes_cli.config import load_config
    with _profile_runtime_scope(home):
        return mcp_revision(policy, load_config())


def _session_servers(authority, sid, ready):
    """The profile MCP servers the session's committed policy runs (its toolsets name them) that are *ready*
    (live or lazy): a server connected for the profile is not one a chat with ``no_mcp``, an explicit pick or
    a platform allowlist enabled, and one the policy names but that parked serves the chat nothing."""
    import json
    from gateway.session_policy import restore_policy
    from hermes_cli.tools_config import enabled_mcp_server_names
    from hermes_state_local import local_receipt
    policy = restore_policy(local_receipt(authority.db, sid)['policy'])
    return sorted(set(policy.toolsets) & enabled_mcp_server_names(json.loads(policy.config_json)) & set(ready))


async def _commit_session_mcp(connection, ref, params):
    """The calling local session's MCP servers as a gateway-committed generation (``commit_refresh``):
    never a running or queued turn's, never another session's."""
    from gateway.config import Platform
    authority, sid = connection.authority, getattr(ref, 'session_id', None)
    live = authority.sessions.get(sid) if sid else None
    if live is None or live.source is None or live.source.platform != Platform.LOCAL:
        return None
    authorize(connection, ref, params, 'session:control')
    from gateway.session_policy_refresh import commit_refresh
    try:
        return 'applied' if await commit_refresh(authority, ref, mcp_only=True) else 'unchanged'
    except RuntimeStoreError as exc:
        if exc.reason not in {'session_busy', 'unknown_execution', 'revision_conflict'}:
            raise
        return 'busy'


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
    killed = await asyncio.to_thread(kill) if targets else 0
    return {'killed': killed, 'delegations': await asyncio.to_thread(_interrupt_delegations, connection.authority, ref)}


def _interrupt_delegations(authority, ref):
    """Main's ``/stop`` also interrupted the background (async) delegations; here only the ones this
    session dispatched (its route or its durable id), never another chat's."""
    module = sys.modules.get('tools.async_delegation')
    if module is None:
        return 0
    return module.interrupt_for_session(session_key=authority.sessions[ref.session_id].route,
                                        parent_session_id=physical_target(authority, ref), reason='/stop')


def _require_current_route(authority, ref):
    """The route still serves this conversation (a reused route must not lend its objects)."""
    live = authority.sessions[ref.session_id]
    target = physical_target(authority, ref)
    store = authority.runner.session_store
    with store._lock:
        entry = store._entries.get(live.route)
        if entry is None or entry.session_id != target:
            raise RuntimeStoreError('stale_generation')


def _owner_objects(authority, ref, *, include_fallback_evicted=False):
    """``(agent, subagent records)`` of this in-process session. Process-local registries cannot
    describe (or stop) another interpreter's live objects: a managed worker is refused rather than
    answered with an empty success."""
    from gateway.session_managed_worker import managed_policy
    if managed_policy(authority, ref) is not None:
        raise RuntimeStoreError('unsupported_projection')
    agent = authority.agent(ref)
    session_id = physical_target(authority, ref) if include_fallback_evicted else None
    route = authority.sessions[ref.session_id].route if include_fallback_evicted else None
    session_db = getattr(authority, 'db', None) if include_fallback_evicted else None
    return agent, subagent_records(
        agent, session_id=session_id, session_db=session_db, route=route,
        include_fallback_evicted=include_fallback_evicted)


def physical_target(authority, ref):
    from gateway.config import Platform
    source = authority.sessions[ref.session_id].source  # None: a history mount
    if source is not None and source.platform == Platform.LOCAL:
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
    # Compression carries goal/loop/heartbeat to the continuation: read the current segment.
    target = physical_target(authority, ref)
    target = authority.db.get_compression_tip(target) or target
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


def _has_fallback_evicted_ancestor(agent, target_sids, expected_db, route, registry):
    if not target_sids:
        return False
    current = agent
    for _ in range(getattr(registry, '_MAX_ANCESTOR_HOPS', 8)):
        ref = getattr(current, '_delegate_parent_ref', None) if current is not None else None
        current = ref() if callable(ref) else None
        if current is None:
            return False
        marker = getattr(current, '_evicted_on_fallback', False)
        if not (marker is True or (isinstance(marker, str) and bool(marker))):
            continue
        if getattr(current, 'session_id', None) not in target_sids:
            continue
        if (expected_db is not None or getattr(current, '_session_db', None) is not None) and getattr(current, '_session_db', None) is not expected_db:
            continue
        if route is not None and marker != route:
            continue
        if route is None and isinstance(marker, str) and getattr(current, 'gateway_session_key', None) not in (None, marker):
            continue
        return True
    return False


def subagent_records(parent, session_id=None, *, session_db=None, route=None, include_fallback_evicted=True):
    registry = sys.modules.get('tools.delegate_tool_registry')
    target_sid = (session_id or getattr(parent, 'session_id', None)) if include_fallback_evicted else None
    if registry is None or (parent is None and not target_sid):
        return []
    expected_db = session_db if session_db is not None else getattr(parent, '_session_db', None)
    effective_route = route if route is not None else getattr(parent, 'gateway_session_key', None)
    # A rotating /compress moves the conversation to a new stored id; an evicted parent keeps its earlier one.
    target_sids = _compression_lineage(expected_db, target_sid)
    with registry._active_subagents_lock:
        return [dict(record) for record in registry._active_subagents.values()
                if (parent is not None and registry._is_descendant_of(record.get('agent'), parent))
                or _has_fallback_evicted_ancestor(record.get('agent'), target_sids, expected_db, effective_route, registry)
                or _owned_by_lineage(record, target_sids, expected_db)]


def _owned_by_lineage(record, target_sids, expected_db):
    """The read-only roster follows the conversation, as ``delegate_task(action="list")`` does: a
    child whose durable owner id is in this session's compression lineage, written to the same
    session database file. Without it every cached-agent rebuild (``/reasoning``, ``/model``, a
    policy refresh, idle eviction) hid still-running children from the dock while the status bar
    kept counting them. Control and tail stay pinned to the resident parent."""
    if not target_sids or record.get('owner_agent_session_id') not in target_sids:
        return False
    expected_path = getattr(expected_db, 'db_path', None)
    child_path = getattr(getattr(record.get('agent'), '_session_db', None), 'db_path', None)
    return expected_path is not None and child_path is not None and str(child_path) == str(expected_path)


def _compression_lineage(db, session_id):
    """*session_id* plus the sessions it continues by compression, walked backward so a closed sibling under a compressed parent cannot divert the walk."""
    if not session_id:
        return frozenset()
    ids, current = {session_id}, session_id
    is_child, get = getattr(db, '_is_compression_child_row', None), getattr(db, 'get_session', None)
    if callable(is_child) and callable(get):
        row = get(current)
        while row and is_child(row) and row['parent_session_id'] not in ids and len(ids) < 200:
            current = row['parent_session_id']
            ids.add(current)
            row = get(current)
        return frozenset(ids)
    lineage = getattr(db, 'get_compression_lineage', None)
    found = lineage(session_id) if callable(lineage) else None
    return frozenset(found if isinstance(found, (list, tuple)) else ()) | ids


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
                 # The classic CLI dock's elapsed / exit-age / verdict and its 32k tail view.
                 'started_epoch': p.started_at, 'exited_at': p.exited_at or None,
                 'completion_reason': p.completion_reason if p.exited else None,
                 'output_tail': p.output_buffer[-_PROCESS_TAIL_CHARS:]} for p in _owned_locked(registry, keys, owners)]
