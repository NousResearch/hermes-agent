"""``plugin.<plugin_id>.<name>`` calls and session-scoped plugin events on the session authority.

A plugin method gets the gate of a plugin slash command (``session:control``); a call naming a
``session_id`` also passes that session's authorization (profile, owner of a local session, a
superseded conversation is read-only). Any client this gateway authenticated can reach it, so a plugin
method carries the trust of a plugin tool. The handler runs off the loop, in the owning profile's scope.
"""
import asyncio
import contextlib
import json
import logging

from hermes_cli.plugins_dispatch import resolve_plugin_command_result
from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)


async def call_plugin_rpc(connection, ref, method, handler, params):
    if ref.session_id is not None and not isinstance(ref.session_id, str):
        raise RuntimeStoreError('invalid_params', field='session_id')
    if ref.session_id:
        from gateway.session_busy_controls import authorize
        authorize(connection, ref, params, 'session:control')
    else:
        from gateway.session_authorities import require_serving
        require_serving(connection.authority)
        if 'session:control' not in connection.actor.capabilities:
            raise RuntimeStoreError('permission_denied')
    try:
        # Resolved in the worker: an ``async def`` handler's body must not run on the gateway loop.
        result = await asyncio.to_thread(lambda: resolve_plugin_command_result(
            handler(dict(params), session_id=ref.session_id or None)))
        return json.loads(json.dumps(result))
    except Exception as exc:  # the call's only answer: a plugin failure is the client's error, not a crash
        logger.warning('plugin RPC %s failed', method, exc_info=True)
        raise RuntimeStoreError('plugin_rpc_failed', field=method, detail=f'{type(exc).__name__}: {exc}') from exc


def publish_session_event(session_id, event_type, payload) -> bool:
    """Publish on the event stream of the live session ``session_id`` names; True when a live viewer
    took the frame (it is in the session's replay either way). False too when no authority here holds it."""
    from gateway.run_api import _ACTIVE_RUNNER
    from gateway.session_authorities import all_authorities
    for authority in all_authorities(_ACTIVE_RUNNER) if _ACTIVE_RUNNER is not None else ():
        found = _live(authority, session_id)
        if found is not None:
            key, live = found
            return live.event_stream.publish(key, payload, event_type=event_type)
    return False


def _live(authority, session_id):
    """``(conversation id, live session)`` for ``session_id``: the conversation's own id, or the current id
    of its resident agent (compression gives the agent a new one; the viewers stay on the conversation)."""
    sessions = authority.sessions
    if (live := sessions.get(session_id)) is not None:
        return session_id, live
    from gateway.session_contract import SessionRef
    for key, live in list(sessions.items()):
        with contextlib.suppress(KeyError):  # evicted since the listing
            if getattr(authority.agent(SessionRef(authority.profile_id, key)), 'session_id', None) == session_id:
                return key, live
    return None
