"""A delegated child's store inside a worker: owner-reserved, never self-registered (#141793).

The parent's ``RuntimeSessionStore`` asks the owner for a child on its own receipt stream; the
owner mints the child's session row and execution (gateway/session_worker_child.py) and the
child persists through the ordinary ``worker.persist`` path under its own claim. A refusal
(capacity, depth, draining) is a ValueError so ``delegate_task`` reports it as a tool error.
"""
import secrets
from functools import partial

from agent.runtime_session_store import RuntimeSessionStore, call_off_event_loop

_REFUSALS = {
    'delegation_capacity': 'the gateway owner refused another subagent: this execution already runs '
                           'delegation.max_concurrent_children of them',
    'delegation_depth_exceeded': 'the gateway owner refused a subagent this deep (delegation.max_spawn_depth)',
}


class ChildSessionStore(RuntimeSessionStore):
    binds_delegation_ledger = False

    @property
    def reserved_session_id(self):
        return self.scope['session_id']

    def create_session(self, session_id, source, **fields):
        # The owner created this row at reservation with its source, lineage and profile; only
        # the constructor facts are the child's to add.
        from hermes_state_worker_lifecycle import _CONSTRUCTOR_CONFIG
        config = {k: v for k, v in (fields.get('model_config') or {}).items() if k in _CONSTRUCTOR_CONFIG}
        return super().create_session(session_id, 'subagent', model=fields.get('model'),
                                      model_config=config or None, system_prompt=fields.get('system_prompt'))

    def close(self):
        # The child's agent closes once, after its last write: that is this execution's end.
        try:
            if not self._outbox_owner.closed and not self.failure:
                self.finish()
        finally:
            super().close()


def reserve_child_store(parent):
    """A ``ChildSessionStore`` the owner reserved under ``parent``'s verified execution."""
    secret = secrets.token_urlsafe(32)
    # Reached from delegate_task's tool callback, which can run on Relay's loop (#139029).
    result = call_off_event_loop(parent._apply, 'child.reserve', {'secret': secret})
    if 'refused' in result:
        raise ValueError(_REFUSALS.get(result['refused'], 'the gateway owner refused a subagent: ' + result['refused']))
    child = result['child']
    scope = dict(parent.scope, **child, secret=secret)
    outbox = parent.path.parent / ('child-' + child['execution_id'].rsplit(':', 1)[1])
    # Every child RPC hops off a running loop, not only this reservation: the child is built, run
    # and closed inside delegate_task's callback, and its constructor already reads the store
    # (ContextCompressor.bind_session_state), which WorkerRPC refuses on Relay's loop.
    return ChildSessionStore(partial(call_off_event_loop, parent.rpc), scope, outbox)
