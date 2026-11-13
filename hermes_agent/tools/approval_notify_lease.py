"""Dispatch-owned approval transport; never extends a completed turn's stream lifetime.

A background child takes a lease at dispatch (``hermes_agent.tools.async_delegation``). Its prompts go through the
``background_notify`` the gateway turn runner published for the chat
(``hermes_agent/gateway/run_turn_runner_approval_transport.py``), before and after the parent turn ends, so ending
the parent never withdraws or strands them (#133514). A surface that publishes none keeps main's
behaviour while a turn runs and fails closed once none does, instead of parking an approval nobody can
see.
"""
from contextvars import ContextVar
from weakref import WeakSet

_current = ContextVar("approval_notify_lease", default=None)
_leases = WeakSet()  # Guarded by approval._lock; does not keep completed workers alive.


class NotifyLease:
    def __init__(self, session_key, callback):
        self.session_key = session_key
        self.callback = callback
        self.active = True

    def _target_locked(self):
        """The lease's own transport. A surface that published none (CLI, TUI, ACP) keeps main's
        behaviour: the session's notifier while one is registered, else nothing (fail closed).

        The lease transport is used even while the dispatching turn still runs: that turn's notifier
        projects onto its canonical execution generation, which can settle mid-send. A child prompt
        must outlive the parent, so it never borrows the parent's generation, typing or stream."""
        from hermes_agent.tools import approval
        if not self.active:
            return None
        return self.callback or approval._gateway_notify_cbs.get(self.session_key)

    def deliverable(self):
        """False only while the lease is live but nothing could show its prompt."""
        from hermes_agent.tools import approval
        with approval._lock:
            return not self.active or self._target_locked() is not None

    def notify(self, data):
        from hermes_agent.tools import approval
        with approval._lock:
            target = self._target_locked()
        if target is None:
            raise RuntimeError("Approval notification unavailable: background task ended or transport missing")
        target(data)

    def release(self):
        from hermes_agent.tools import approval
        with approval._lock:
            self.active = False
            self.callback = None
            _leases.discard(self)
            queue = approval._gateway_queues.get(self.session_key, [])
            for entry in list(queue):
                if entry.owner is self:
                    queue.remove(entry)
                    entry.cancelled = "background task ended"
                    entry.event.set()
            if not queue:
                approval._gateway_queues.pop(self.session_key, None)


def acquire(session_key):
    from hermes_agent.tools import approval
    with approval._lock:
        # Only a transport built to outlive its turn is kept; a turn-scoped callback is not.
        lease = NotifyLease(session_key, getattr(approval._gateway_notify_cbs.get(session_key), "background_notify", None))
        _leases.add(lease)
        return lease


def revoke_session_locked(session_key):
    """Called by clear_session under approval._lock before waking pending waits."""
    for lease in list(_leases):
        if lease.session_key == session_key:
            lease.active = False
            lease.callback = None
            _leases.discard(lease)


def stranded_denial(session_key, is_gateway, noun, pattern_key, description):
    """The fail-closed result for a background child's request no chat can show (its dispatching
    turn ended and its surface left it no transport), else None. ``pending_approval`` there would be
    a prompt nobody can see or ``/approve``."""
    lease = current(session_key)
    if not is_gateway or lease is None or lease.deliverable():
        return None
    from hermes_agent.tools.approval import _denied
    return _denied(
        "BLOCKED: no approval channel is available to ask the user — the gateway turn "
        f"that delivers approval prompts has ended, so this {noun} did not run "
        "and no prompt can be answered from this context. Do NOT retry or rephrase; "
        "run it from an interactive turn instead.",
        pattern_key=pattern_key, description=description, outcome="notify_failed", noun=noun)


def current(session_key):
    lease = _current.get()
    return lease if lease is not None and lease.session_key == session_key else None


def run(lease, fn):
    token = _current.set(lease)
    try:
        return fn()
    finally:
        _current.reset(token)
        lease.release()
