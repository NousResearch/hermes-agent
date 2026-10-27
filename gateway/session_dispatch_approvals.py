"""Process-local dispatch approval authority, separate from execution generations.

Lock order is event-stream -> tools.approval._lock. Queue validation runs under
only the latter and never acquires the stream lock. Delivery performs no I/O
under either lock. Revocation and consent commit share the tools lock.
"""
from dataclasses import dataclass, field
import uuid

from hermes_state_runtime import RuntimeStoreError


@dataclass(eq=False)
class DispatchApprovalOwner:
    authority: object
    session_id: str
    live: object
    route: str
    dispatch_id: str
    epoch: int
    token: str = field(default_factory=lambda: uuid.uuid4().hex)
    lease: object = None
    published: set = field(default_factory=set)

    def valid(self):
        from gateway.session_authorities import require_serving
        a = self.authority
        try:
            require_serving(a)
            if (a.epoch != self.epoch or a.retiring or a.runner._draining
                    or a.sessions.get(self.session_id) is not self.live
                    or self.live.superseded or self.live.route != self.route
                    or self.lease is None or not self.lease.active):
                return False
            # A reset can reuse a route without reusing its canonical incarnation.
            entry = a.runner.session_store.lookup_by_session_key(self.route)
            return entry is not None and a.logical_owner(entry.session_id) == self.session_id
        except RuntimeStoreError:
            return False

    def check(self):
        if not self.valid():
            raise RuntimeStoreError('stale_dispatch')

    def permits(self, request_id):
        return self.valid() and request_id in self.published

    def publish(self, data):
        from tools import approval
        from gateway.session_pending_controls import PendingControls
        with self.live.event_stream.lock, approval._lock:
            self.check()
            request_id = data['request_id']
            queue = approval._gateway_queues.get(self.route, [])
            if not any(e.owner is self.lease and e.data['request_id'] == request_id for e in queue):
                raise RuntimeStoreError('stale_dispatch')
            self.published.add(request_id)
            # The owner is trusted state, never a field accepted from a tool/model.
            PendingControls.register(self.live.controls, self.session_id, self.route, None, data,
                                     dispatch_owner=self)


def mint(turn, lease, dispatch_id):
    a, sid, generation = turn._approval_owner
    live = a.sessions[sid]
    with live.event_stream.lock:
        a.check_approval_generation(sid, generation)
        owner = DispatchApprovalOwner(a, sid, live, turn._ctx.session_key, dispatch_id, a.epoch,
                                      lease=lease)
        owner.check()
        return owner


def respond(authority, live, session_id, token, prompt_id, response):
    """Caller has authorized and attached the actor and holds the stream lock."""
    owner = live.controls.dispatch_owners.get(prompt_id)
    if (owner is None or owner.authority is not authority or owner.session_id != session_id
            or not isinstance(token, str) or token != owner.token):
        raise RuntimeStoreError('stale_dispatch')
    owner.check()
    # PendingControls snapshots and the resolver each take the non-reentrant
    # tools lock themselves. The resolver rechecks the owner under that lock,
    # atomically with revocation and committing the exact queue entry.
    return live.controls.respond(session_id, None, prompt_id, response)
