"""Path completion rooted in the owning session, or the owning profile for a fresh draft.

An empty word inspects the same base without listing files, before checkout preparation
can change what the draft's relative references mean.
"""
import asyncio
import json
from pathlib import Path

from gateway.session_busy_controls import authorize
from hermes_state_runtime import RuntimeStoreError


async def complete_path(connection, ref, params):
    if (set(params) - {'session_id', 'profile', 'word', 'cwd', 'cwd_explicit'}
            or any(not isinstance(value, str) for key, value in params.items() if key != 'cwd_explicit')
            or ('cwd_explicit' in params and type(params['cwd_explicit']) is not bool)):
        raise RuntimeStoreError('invalid_params')
    authority = connection.authority
    if 'session:read' not in connection.actor.capabilities or connection.actor.profile_id != authority.profile_id:
        raise RuntimeStoreError('permission_denied')
    word = params.get('word', '')
    policy = None
    if not ref.session_id and 'session:create' not in connection.actor.capabilities:
        # A draft can name a host cwd before any session owns it, unlike an authorized session read.
        raise RuntimeStoreError('permission_denied')
    if ref.session_id:
        if ref.session_id not in authority.sessions:
            raise RuntimeStoreError('not_found')
        authorize(connection, ref, params, 'session:read')
        from gateway.session_policy import policy_for_source
        policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
        if policy is None:
            return {'items': []}

    def listing():
        from gateway.run import _profile_runtime_scope
        from gateway.session_policy import terminal_uses_host_cwd
        from hermes_constants import profile_name_for_home
        from tools.terminal_scope import build_profile_terminal_scope
        from tui_gateway import server
        home = Path(authority.profile_id)
        with _profile_runtime_scope(home):
            terminal = json.loads(policy.terminal_json) if policy else build_profile_terminal_scope(home)
            if not terminal_uses_host_cwd(terminal):
                # Never turn a remote/sandbox path into a listing of the gateway host.
                return {'items': []}
            root = policy.cwd if policy else server._completion_cwd({
                **params, 'profile': profile_name_for_home(home) or 'default'})
            if not word:
                return {'items': [], 'sourceCwd': root}
            return {'items': server._path_completion_items(word, root, None, True)}

    return await asyncio.to_thread(listing)
