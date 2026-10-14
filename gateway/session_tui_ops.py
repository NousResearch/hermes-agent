"""Ink's session utilities on the shared owner: ``/save`` (``session.save``), ``/reload``
(``reload.env``), ``/agents pause|resume`` (``delegation.pause``) and ``/rollback``
(``rollback.list`` / ``rollback.diff`` / ``rollback.restore``).

Main's per-TUI sidecar served these from its own session table, which never holds a shared-gateway
session, so the canonical TUI refused them. Each verb reuses main's helpers rather than re-deriving
them. ``dispatch`` already runs every handler under the owning profile's scope (``owner_scope``) and
``asyncio.to_thread`` carries that context into the worker thread.
"""
import asyncio
from datetime import datetime
import json
import uuid
from functools import partial
from pathlib import Path

from gateway.session_busy_controls import authorize
from gateway.session_tui_reads import context_breakdown
from hermes_state_runtime import RuntimeStoreError


def handlers(connection):
    return {'session.save': partial(save_session, connection),
            'reload.env': partial(reload_env, connection),
            'delegation.pause': partial(pause_delegation, connection),
            'rollback.list': partial(rollback, connection, kind='list'),
            'rollback.diff': partial(rollback, connection, kind='diff'),
            'rollback.restore': partial(rollback, connection, kind='restore'),
            'session.context_breakdown': partial(context_breakdown, connection)}


def _profile_home(authority):
    """The owning profile's home (a symbolic test-peer id falls back to the scoped home)."""
    home = Path(str(authority.profile_id))
    if home.is_absolute() and home.is_dir():
        return home
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def _multiplexed(authority):
    """Every turn runs under a per-profile secret scope (``owner_scope``'s own predicate)."""
    return bool(getattr(getattr(authority.runner, 'config', None), 'multiplex_profiles', False))


def _live_ref(connection, ref, params, fields, capability):
    if set(params) - fields or not isinstance(ref.session_id, str) or not ref.session_id:
        raise RuntimeStoreError('invalid_params')
    authorize(connection, ref, params, capability)
    return connection.authority.sessions[ref.session_id]


async def save_session(connection, ref, params):
    """``/save``: main's ``session.save`` — the stored transcript as a ``/save json`` snapshot in the
    profile's ``sessions/saved/``. Reads the conversation's CURRENT physical row (a local session's
    reset/compression child), like main's ``_submit_row_target_key``."""
    _live_ref(connection, ref, params, {'session_id', 'profile'}, 'session:read')
    authority = connection.authority
    from gateway.session_ancillary import physical_target
    target = physical_target(authority, ref)

    def write():
        from hermes_cli.session_export import load_save_snapshot, render_session_for_save
        from hermes_state import SessionExportTooLargeError
        try:
            data = load_save_snapshot(authority.db, target, 'json')
        except SessionExportTooLargeError as exc:
            raise RuntimeStoreError('export_too_large') from exc
        if not data:
            raise RuntimeStoreError('not_found')
        path = (_profile_home(authority) / 'sessions' / 'saved'
                / f'hermes_conversation_{datetime.now():%Y%m%d_%H%M%S}.json')
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(render_session_for_save(data, 'json'), encoding='utf-8')
        except OSError as exc:
            raise RuntimeStoreError('storage_unavailable') from exc
        return {'file': str(path)}
    return await asyncio.to_thread(write)


async def reload_env(connection, ref, params):
    """``/reload``: re-read the profile's ``.env`` (main's ``reload.env`` / CLI ``/reload``).

    Single-profile owner: main's ``hermes_cli.config.reload_env()`` verbatim — the owner's
    ``os.environ`` is what its turns read, exactly as the TUI's own sidecar process was. Built agents
    keep their credential pool until ``/new`` (main's documented behaviour).

    Multiplexed owner: ``os.environ`` is shared by every served profile and turns never read profile
    secrets from it, so it is left alone. Each turn already re-reads ``.env`` into its own scope; what
    can be stale is the profile's external secret-source snapshot, which is re-pulled here. The count
    is the number of keys whose value in this profile's secret scope changed.
    """
    # Addressed at one session the caller controls: a sessionless frame stays refused, as it was
    # before this verb existed (a bare grant must not rewrite the owner's environment).
    _live_ref(connection, ref, params, {'session_id', 'profile'}, 'session:control')
    authority = connection.authority

    def reload():
        if not _multiplexed(authority):
            from hermes_cli.config import reload_env as reload_process_env
            return reload_process_env()
        from agent.secret_scope import build_profile_secret_scope, invalidate_env_file_cache
        from hermes_cli.env_loader import hydrate_profile_secret_sources, reset_secret_source_cache
        home = _profile_home(authority)
        before = build_profile_secret_scope(home)
        invalidate_env_file_cache(home / '.env')
        reset_secret_source_cache(home)
        hydrate_profile_secret_sources(home)
        after = build_profile_secret_scope(home)
        return sum(before.get(key) != after.get(key) for key in before.keys() | after.keys())
    return {'updated': int(await asyncio.to_thread(reload))}


async def pause_delegation(connection, ref, params):
    """``/agents pause|resume`` and the overlay's ``p``: main's ``delegation.pause`` — block or
    unblock NEW ``delegate_task`` spawns; running children keep going.

    The gate (``tools.delegate_tool_registry``) is process-wide, as it was in main's sidecar (a
    TUI's sidecar served one TUI, but its gate covered every agent in that process). On the shared
    owner it therefore covers every chat this gateway runs, which is what an operator's kill switch
    is for; ``delegation.status`` reports the same flag to every client."""
    paused = params.get('paused', True)
    if set(params) - {'paused', 'profile', 'session_id'} or type(paused) is not bool:
        raise RuntimeStoreError('invalid_params')
    if ref.session_id:
        authorize(connection, ref, params, 'session:control')
    else:
        if 'session:control' not in connection.actor.capabilities:
            raise RuntimeStoreError('permission_denied')
        profile = params.get('profile')
        if profile is not None and not isinstance(profile, str):
            raise RuntimeStoreError('invalid_params')
        if profile:
            from hermes_cli.profiles import profile_matches_home
            if not profile_matches_home(profile, Path(connection.authority.profile_id)):
                raise RuntimeStoreError('profile_mismatch')
    from tools.delegate_tool_registry import set_spawn_paused
    return {'paused': set_spawn_paused(paused)}


_ROLLBACK_FIELDS = {'list': {'session_id', 'profile'}, 'diff': {'session_id', 'profile', 'hash'},
                    'restore': {'session_id', 'profile', 'hash', 'file_path'}}


async def rollback(connection, ref, params, *, kind):
    """``/rollback [list] | diff <n|hash> | <n|hash> [file]``: main's ``rollback.*`` over this
    session's checkpoints (the shared ``<profile>/checkpoints`` store, keyed by the frozen launch cwd).

    The manager is built from the session's frozen ``checkpoints`` policy rather than read off the
    resident agent (main's ``agent._checkpoint_mgr``, built from the same config): it carries no
    per-session state these reads need, and a restarted owner or a just-rewound session has no
    resident agent. A whole-tree restore also rewinds the transcript to before the newest user turn
    (main's ``_rewind_active_session_history``) through the owner's fenced ``rewind`` mutation."""
    fields = _ROLLBACK_FIELDS[kind]
    if any(key in params and (not isinstance(params[key], str) or (key == 'hash' and not params[key]))
           for key in ('hash', 'file_path')) or (kind != 'list' and 'hash' not in params):
        raise RuntimeStoreError('invalid_params')
    live = _live_ref(connection, ref, params, fields, 'session:read' if kind != 'restore' else 'session:control')
    authority = connection.authority
    from gateway.session_policy import policy_for_source
    policy = policy_for_source(authority.runner, live.source)
    if policy is None:
        raise RuntimeStoreError('permission_denied')
    file_path = params.get('file_path') or ''
    if kind == 'restore' and not file_path and authority._handle(ref).execution_state != 'idle':
        # Main refuses a full-history rollback mid-turn; a file-scoped restore only touches disk.
        raise RuntimeStoreError('session_busy')
    from gateway.session_ancillary import physical_target
    task_id = physical_target(authority, ref)
    from gateway.run import _checkpoint_agent_kwargs
    kwargs = _checkpoint_agent_kwargs(policy.config(authority, secrets=False))
    terminal = json.loads(policy.terminal_json)

    def run():
        from tools.checkpoint_manager import CheckpointManager
        from tui_gateway.checkpoints import _resolve_checkpoint_hash
        mgr = CheckpointManager(enabled=bool(kwargs['checkpoints_enabled']), **{
            key[len('checkpoint_'):]: value for key, value in kwargs.items() if key.startswith('checkpoint_')})
        cwd = policy.cwd
        if kind == 'list':
            if not mgr.enabled:
                return {'enabled': False, 'checkpoints': []}
            # The TUI renders ``message``; the manager calls it ``reason``.
            return {'enabled': True, 'checkpoints': [
                {'hash': c.get('hash', ''), 'timestamp': c.get('timestamp', ''), 'message': c.get('reason', '')}
                for c in mgr.list_checkpoints(cwd)]}
        # Host checkpoints are not a container-backed session's tree (main's _container_checkpoint_refusal),
        # classified under the session's frozen terminal policy, not the owner's ambient one.
        from tools.terminal_scope import reset_terminal_scope, set_terminal_scope
        token = set_terminal_scope(terminal)
        try:
            refusal = mgr.unsupported_backend_reason(task_id)
        finally:
            reset_terminal_scope(token)
        try:
            target = _resolve_checkpoint_hash(mgr, cwd, params['hash'])
        except ValueError as exc:
            if kind == 'diff':
                raise RuntimeStoreError('invalid_checkpoint') from exc
            return {'success': False, 'error': str(exc)}
        if kind == 'diff':
            if refusal:
                raise RuntimeStoreError('unsupported_terminal_backend')
            from tui_gateway.render import render_diff
            result = mgr.diff(cwd, target)
            raw = result.get('diff', '')[:4000]
            payload = {'stat': result.get('stat', ''), 'diff': raw}
            if rendered := render_diff(raw, 80):
                payload['rendered'] = rendered
            return payload
        if refusal:
            return {'success': False, 'error': refusal}
        return mgr.restore(cwd, target, file_path=file_path or None)
    result = await asyncio.to_thread(run)
    if kind == 'restore' and result.get('success') and not file_path:
        result['history_removed'] = await _rewind_last_turn(connection, ref)
    return result


async def _rewind_last_turn(connection, ref):
    """Drop the newest user turn and everything after it (main's full ``/rollback`` restore) via the
    owner's revision-fenced ``rewind`` mutation; 0 when the transcript has no user turn."""
    authority = connection.authority
    from agent.context_compressor import user_originated_turn_view
    from gateway.session_local_recovery import local_history
    rows = await asyncio.to_thread(local_history, authority, ref)
    target = next((row['row_id'] for row in reversed(rows)
                   if isinstance(row.get('row_id'), int) and user_originated_turn_view(row) is not None), None)
    if target is None:
        return 0
    from gateway.session_mutations import mutate_session
    from hermes_state_mutation_guards import MUTATION_GUARD_REFUSALS
    handle = authority._handle(ref)
    try:
        result = await mutate_session(authority, connection.actor, ref, {
            'session_id': ref.session_id, 'request_id': f'rollback:{uuid.uuid4().hex}',
            'expected_revision': handle.revision, 'expected_generation': handle.execution_generation,
            'operation': 'rewind', 'payload': {'target_message_id': target}})
    except MUTATION_GUARD_REFUSALS as exc:
        raise RuntimeStoreError('session_busy') from exc
    return int(result.get('rewound_count') or 0)
