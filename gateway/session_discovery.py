"""Authenticated, profile-bound discovery without legacy execution dispatch."""
import asyncio
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError


async def discover_commands(authority, actor, params, *, completion=False):
    if 'session:read' not in actor.capabilities or actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('permission_denied')
    allowed = {'profile', 'text', 'session_id'} if completion else {'profile', 'session_id'}
    if (not isinstance(params, dict) or set(params) - allowed
            or any(not isinstance(value, str) for value in params.values())):
        raise RuntimeStoreError('invalid_params')
    scope = None
    if params.get('session_id'):
        # Project skills resolve from the session's workspace, not the gateway process cwd.
        from gateway.session_commands import _session_workspace
        from gateway.session_contract import SessionRef
        ref = SessionRef(authority.profile_id, params['session_id'])
        authority.authorize(actor, ref, 'session:read')
        scope = _session_workspace(authority.runner, authority.sessions[ref.session_id].source)

    def discover():
        from hermes_cli.profiles import profile_matches_home
        home = Path(authority.profile_id)
        if params.get('profile') and not profile_matches_home(params['profile'], home):
            raise RuntimeStoreError('profile_mismatch')
        from gateway.run import _profile_runtime_scope
        from tui_gateway.command_discovery import command_catalog, slash_completions
        with _profile_runtime_scope(home):
            return (slash_completions(params.get('text', ''), scope=scope) if completion
                    else command_catalog(scope=scope))

    return await asyncio.to_thread(discover)
