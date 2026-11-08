"""``on_session_open``: a local session was created or opened by a client (#89385).

Fired by the canonical authority (``AuthorityConnection.resume``, which ``session.create`` and the
kanban/A2A/onboarding openers end in) when a connection attaches to a local session it was not
attached to: once per open, never for a snapshot poll on an attachment it already holds or a
reconnect. The open is registered before the attach yields and delivered on a thread of its own under
the owning profile's scope (that profile's plugin manager), so the reply never waits on a plugin; the
session's next claimed turn waits for it (``settled``), so plugins see the open before the model does.
``on_session_start`` stays the first-turn hook.
"""
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# (profile id, session id) -> the open still pending or running for it. In-flight work only: an
# entry leaves when its delivery ends, so this never dedupes and holds one open per opening session.
_PENDING = {}

# Longest a turn waits on an open when ``plugins.hook_callback_timeout`` is 0 (callbacks never
# abandoned): that setting keeps a slow callback's thread, it does not let one stall the conversation.
_UNBOUNDED_TURN_WAIT_S = 5.0


def _reason(db, session_id):
    """``create`` while the conversation has no messages yet, else ``resume``. A local session's
    messages live on its current transcript (a reset moves them to a child row), named by its receipt."""
    from hermes_state_local import local_receipt
    try:
        target = local_receipt(db, session_id)['entry']['session_id']
    except Exception:  # health: allow BLE001 -- no readable receipt: the logical row is the transcript
        target = session_id
    return 'resume' if (db.get_session(target) or {}).get('message_count') else 'create'


class _Open:
    """One pending open: registered before the attach, then ``fire``d with the session's policy, or
    ``release``d (not a local session, a failed attach) so no turn waits on it."""

    def __init__(self, authority, ref):
        from hermes_cli.plugins import _resolve_hook_callback_timeout
        self.ref, self.db = ref, authority.db
        key = (ref.profile_id, ref.session_id)
        # Read here, in the owning profile's dispatch scope; the drain that waits runs in no profile's.
        self.turn_wait = _resolve_hook_callback_timeout() or _UNBOUNDED_TURN_WAIT_S
        self.gate = asyncio.get_running_loop().create_future()
        previous = _live(key)
        self.task = asyncio.ensure_future(self._run(previous and previous.task))
        _PENDING[key] = self
        self.task.add_done_callback(lambda _done: _PENDING.pop(key) if _PENDING.get(key) is self else None)

    def fire(self, policy, route):
        # A bypass launch (``--safe-mode`` / ``--ignore-user-config``) loads no plugin: nothing fires.
        if not self.gate.done():
            self.gate.set_result(None if policy is None or policy.ignore_user_config else (policy, route))

    def release(self):
        if not self.gate.done():
            self.gate.set_result(None)

    async def _run(self, previous):
        opened = await self.gate
        if previous is not None:  # two viewers opening at once: one session's opens fire in order
            await asyncio.wait({previous})
        if opened is not None:
            from gateway.session_commands import _on_own_thread
            await _on_own_thread(lambda: self._deliver(*opened))

    def _deliver(self, policy, route):
        from gateway.run import _profile_runtime_scope
        from gateway.session_context import scoped_session_key
        from gateway.session_policy import SURFACES
        from hermes_cli.lifecycle import has_hook, invoke_hook
        sid = self.ref.session_id
        try:
            with _profile_runtime_scope(Path(self.ref.profile_id)), scoped_session_key(route):
                if has_hook('on_session_open'):  # no consumer in this profile: no read, no call
                    invoke_hook('on_session_open', session_id=sid, platform=SURFACES.get(policy.source, policy.source),
                                reason=_reason(self.db, sid))
        except Exception:  # health: allow BLE001 -- fail-open observer: an open never fails on a plugin
            logger.warning('on_session_open delivery failed for %s', sid, exc_info=True)


def open_local(authority, ref):
    """Register *ref*'s open before anything can start its next turn; the caller fires or releases it."""
    return _Open(authority, ref)


def _live(key):
    """*key*'s pending open on the running loop; a retired loop's (an in-process restart) never
    resolves on this one."""
    opening = _PENDING.get(key)
    if opening is not None and opening.task.get_loop() is not asyncio.get_running_loop():
        _PENDING.pop(key, None)
        return None
    return opening


async def settled(ref):
    """Wait out an open delivery still pending or running for *ref*; awaited before a claimed turn
    executes. The wait is bounded on its own (``plugins.hook_callback_timeout``, or
    ``_UNBOUNDED_TURN_WAIT_S`` when that is 0): a hung callback delays the turn, never wedges it, and
    its delivery keeps running."""
    opening = _live((ref.profile_id, ref.session_id))
    if opening is not None:
        _done, pending = await asyncio.wait({opening.task}, timeout=opening.turn_wait)
        if pending:
            logger.warning('on_session_open for %s still running after %gs; the turn proceeds',
                           ref.session_id, opening.turn_wait)
