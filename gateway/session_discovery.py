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


async def list_plugins(authority, actor, params):
    """``plugins.list`` for this profile's own plugin manager: what the gateway loaded, with each
    plugin's tool/hook/command counts and load error (the classic CLI's ``/plugins`` reads it, so the
    CLI process imports no plugin code). Plain data only."""
    if 'session:read' not in actor.capabilities or actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('permission_denied')
    if params:
        raise RuntimeStoreError('invalid_params')

    def rows():
        from gateway.run import _profile_runtime_scope
        from hermes_cli.plugins import discover_plugins, get_plugin_manager
        from hermes_cli.plugins import get_plugin_toolsets
        with _profile_runtime_scope(Path(authority.profile_id)):
            discover_plugins()  # idempotent: the profile's plugins are normally loaded already
            manager = get_plugin_manager()
            loaded, toolsets = manager.list_plugins(), get_plugin_toolsets()
            portable = sorted(manager.get_portable_mcp_servers())
        return {'plugins': [{'name': str(p['name']), 'key': str(p['key']), 'version': str(p['version'] or ''),
                             'enabled': bool(p['enabled']), 'tools': int(p['tools']), 'hooks': int(p['hooks']),
                             'commands': int(p['commands']), 'error': str(p['error'] or '')} for p in loaded],
                # The CLI's /tools and its completion validate plugin toolsets against these.
                'toolsets': [{'key': str(key), 'label': str(label), 'description': str(desc or '')}
                             for key, label, desc in toolsets],
                'portable_mcp_servers': [str(name) for name in portable]}

    return await asyncio.to_thread(rows)
