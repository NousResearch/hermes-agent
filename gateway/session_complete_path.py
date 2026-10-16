"""``complete.path`` on the shared gateway: the sidecar's path / ``@`` listing, rooted in one session's folder.

The Ink TUI asks its canonical connection for composer completions. The listing is the sidecar's own
(``tui_gateway/methods_complete.py``); its root is the session's frozen policy cwd, never a cwd the
client sends or the daemon's own folder.
"""
import asyncio
import json
from pathlib import Path

from gateway.session_busy_controls import authorize
from hermes_state_runtime import RuntimeStoreError


async def complete_path(connection, ref, params):
    if (set(params) - {'session_id', 'profile', 'word', 'cwd'}
            or any(not isinstance(value, str) for value in params.values())):
        raise RuntimeStoreError('invalid_params')
    word = params.get('word', '')
    if not word or not ref.session_id:
        return {'items': []}
    authority = connection.authority
    if ref.session_id not in authority.sessions:
        raise RuntimeStoreError('not_found')
    authorize(connection, ref, params, 'session:read')
    from gateway.session_policy import policy_for_source, terminal_uses_host_cwd
    policy = policy_for_source(authority.runner, authority.sessions[ref.session_id].source)
    if policy is None or not terminal_uses_host_cwd(json.loads(policy.terminal_json)):
        # A remote or sandboxed backend's tree is not this host's; completion is advisory, so nothing.
        return {'items': []}
    # The listing helpers are bound onto the sidecar module's globals (method_ctx.bind_module).
    from tui_gateway import server

    def listing():
        from gateway.run import _profile_runtime_scope
        with _profile_runtime_scope(Path(authority.profile_id)):
            return server._path_completion_items(word, policy.cwd, None, True)

    return {'items': await asyncio.to_thread(listing)}
