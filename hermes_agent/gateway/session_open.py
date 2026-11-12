"""``on_session_open``: a local session was created or opened by a client (#89385).

Fired by the canonical authority (``AuthorityConnection.resume``, which ``session.create`` and the
kanban/A2A/onboarding openers end in) when a connection attaches to a local session it was not
attached to: once per open, never for a snapshot poll on an attachment it already holds or a
reconnect. The open is registered before the attach yields and delivered on a thread of its own under
the owning profile's scope (that profile's plugin manager), so the reply never waits on a plugin; the
session's next claimed turn waits for it (``settled``), so plugins see the open before the model does.
A profile whose loaded plugins hold no consumer starts no thread.
``on_session_start`` stays the first-turn hook.
"""
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# (profile id, session id) -> its latest open delivery still queued or running. In-flight work only:
# an entry leaves when its delivery ends, so this never dedupes and holds one per opening session.
_PENDING = {}

# Longest a turn waits on an open when ``plugins.hook_callback_timeout`` is 0 (callbacks never
# abandoned): that setting keeps a slow callback's thread, it does not let one stall the conversation.
_UNBOUNDED_TURN_WAIT_S = 5.0


def _reason(db, session_id):
    """``create`` while the conversation has no messages yet, else ``resume``. A local session's
    messages live on its current transcript (a reset moves them to a child row), named by its receipt."""
    from hermes_agent.hermes_state_local import local_receipt
    try:
        target = local_receipt(db, session_id)['entry']['session_id']
    except Exception:  # health: allow BLE001 -- no readable receipt: the logical row is the transcript
        target = session_id
    return 'resume' if (db.get_session(target) or {}).get('message_count') else 'create'


class _Open:
    """One session's open delivery, standing for every open that joined it while it queued.

    Each open holds a gate (``open_local``) its attach ``fire``s with the session's policy, or
    ``release``s (not a local session, a failed attach) so no turn waits on it. Bounded: at most
    one delivery runs per (profile, session) and one queues behind it; opens arriving while that
    one still queues join it, so a hung callback (``hook_callback_timeout: 0``) holds two tasks and
    one thread however often the session is opened meanwhile. An open is setup, so the joined opens
    fire once, after the running one, with the latest gate's payload."""

    def __init__(self, ref, db, ahead):
        self.ref, self.db, self.ahead = ref, db, ahead
        # The profile's ``plugins.hook_callback_timeout``, read by the delivery (off-loop, in that
        # profile's scope, only for a session that loads plugins); 0 until then.
        self.callback_timeout = 0.0
        self.started = self.coalesced = False
        self.unsettled, self.opened = set(), None  # gates not yet fired or released; latest payload
        self.task = asyncio.ensure_future(self._run())

    def joinable(self):
        return not self.started and self.ahead is not None and not self.ahead.task.done()

    async def _run(self):
        if self.ahead is not None:  # two viewers opening at once: one session's opens fire in order
            await asyncio.wait({self.ahead.task})
        self.started, self.ahead = True, None  # a session reopened nonstop never chains finished deliveries
        while self.unsettled:
            await asyncio.wait(set(self.unsettled))
        from hermes_agent.hermes_cli.lifecycle import hook_known_unused
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        token = set_hermes_home_override(self.ref.profile_id)
        try:  # the owning profile's loaded plugins, read without discovery or config: no consumer, no thread
            unused = hook_known_unused('on_session_open')
        finally:
            reset_hermes_home_override(token)
        if self.opened is not None and not unused:
            from hermes_agent.gateway.session_commands import _on_own_thread
            await _on_own_thread(lambda: self._deliver(*self.opened))

    def _deliver(self, policy, route):
        from hermes_agent.gateway.run import _profile_runtime_scope
        from hermes_agent.gateway.session_context import scoped_session_key
        from hermes_agent.gateway.session_policy import SURFACES
        from hermes_agent.hermes_cli.lifecycle import has_hook, invoke_hook
        from hermes_agent.hermes_cli.plugins import _resolve_hook_callback_timeout
        sid = self.ref.session_id
        try:
            with _profile_runtime_scope(Path(self.ref.profile_id)), scoped_session_key(route):
                if has_hook('on_session_open'):  # plugins not loaded until now: none of them consumes it
                    self.callback_timeout = _resolve_hook_callback_timeout()
                    invoke_hook('on_session_open', session_id=sid, platform=SURFACES.get(policy.source, policy.source),
                                reason=_reason(self.db, sid))
        except Exception:  # health: allow BLE001 -- fail-open observer: an open never fails on a plugin
            logger.warning('on_session_open delivery failed for %s', sid, exc_info=True)


class _Gate:
    """One open's hold on its delivery: the delivery waits until every gate is fired or released."""

    def __init__(self, delivery):
        self.delivery, self.future = delivery, asyncio.get_running_loop().create_future()
        delivery.unsettled.add(self.future)

    def fire(self, policy, route):
        # A bypass launch (``--safe-mode`` / ``--ignore-user-config``) loads no plugin: nothing fires.
        if not self.future.done() and policy is not None and not policy.ignore_user_config:
            self.delivery.opened = (policy, route)
        self.release()

    def release(self):
        if not self.future.done():
            self.future.set_result(None)
        self.delivery.unsettled.discard(self.future)


def open_local(authority, ref):
    """Register *ref*'s open before anything can start its next turn; the caller fires or releases
    the returned gate."""
    key = (ref.profile_id, ref.session_id)
    delivery = _live(key)
    if delivery is None or not delivery.joinable():
        delivery = _PENDING[key] = _Open(ref, authority.db, delivery)
        delivery.task.add_done_callback(lambda _done, mine=delivery: _PENDING.pop(key)
                                        if _PENDING.get(key) is mine else None)
    elif not delivery.coalesced:
        delivery.coalesced = True
        logger.warning('on_session_open for %s: an earlier delivery is still running; the opens meanwhile '
                       'fire once, after it', ref.session_id)
    return _Gate(delivery)


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
    executes. The wait is bounded on its own: ``_UNBOUNDED_TURN_WAIT_S``, extended to the profile's
    ``plugins.hook_callback_timeout`` once the delivery has read a longer one. A hung callback (or
    a delivery stuck before it) delays the turn, never wedges it, and its delivery keeps running."""
    opening = _live((ref.profile_id, ref.session_id))
    if opening is None:
        return
    _done, pending = await asyncio.wait({opening.task}, timeout=_UNBOUNDED_TURN_WAIT_S)
    if pending and opening.callback_timeout > _UNBOUNDED_TURN_WAIT_S:
        _done, pending = await asyncio.wait({opening.task},
                                            timeout=opening.callback_timeout - _UNBOUNDED_TURN_WAIT_S)
    if pending:
        logger.warning('on_session_open for %s still running; its turn proceeds', ref.session_id)
