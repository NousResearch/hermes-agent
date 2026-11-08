"""``on_session_open``: a local session was created or opened by a client (#89385).

Fired by the canonical authority (``AuthorityConnection.resume``, which ``session.create`` ends in)
when a connection attaches to a local session it was not attached to: once per open, never for a
snapshot poll on an attachment it already holds or a reconnect's replay. The hook runs on a thread
of its own under the owning profile's scope (that profile's plugin manager), so the reply never waits
on a plugin; the session's next claimed turn waits for it (``settled``), so plugins see the open
before the model does. ``on_session_start`` stays the first-turn hook.
"""
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# (profile id, session id) -> the open delivery still running for it. In-flight work only: an entry
# leaves when its delivery ends, so this never dedupes and holds at most one task per opening session.
_PENDING = {}


def fire(authority, ref, route, policy, reason):
    """Queue the hook for *ref*'s open (*reason*: ``create`` or ``resume``). A bypass launch
    (``--safe-mode`` / ``--ignore-user-config``) loads no plugin, so it fires nothing."""
    if policy is None or policy.ignore_user_config:
        return
    from gateway.session_policy import SURFACES
    home, key = Path(authority.profile_id), (authority.profile_id, ref.session_id)
    platform = SURFACES.get(policy.source, policy.source)
    previous = _PENDING.get(key)

    def deliver_now():
        from gateway.run import _profile_runtime_scope
        from gateway.session_context import scoped_session_key
        from hermes_cli.lifecycle import invoke_hook
        try:
            with _profile_runtime_scope(home), scoped_session_key(route):
                invoke_hook('on_session_open', session_id=ref.session_id, platform=platform, reason=reason)
        except Exception:  # health: allow BLE001 -- fail-open observer: an open never fails on a plugin
            logger.warning('on_session_open delivery failed for %s', ref.session_id, exc_info=True)

    async def deliver():
        from gateway.session_commands import _on_own_thread
        if previous is not None:  # two viewers opening at once: one session's opens fire in order
            await asyncio.wait({previous})
        await _on_own_thread(deliver_now)

    task = asyncio.ensure_future(deliver())
    _PENDING[key] = task
    task.add_done_callback(lambda done: _PENDING.pop(key) if _PENDING.get(key) is done else None)


async def settled(authority, session_id):
    """Wait out an open delivery still running for *session_id* (the hook dispatcher bounds each
    callback by ``plugins.hook_callback_timeout``); awaited before a claimed turn executes."""
    task = _PENDING.get((authority.profile_id, session_id))
    if task is not None:
        await asyncio.wait({task})
