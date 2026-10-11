"""A managed worker's turn lease survives its periodic refresh (the store is the agent's session_db)."""
import time
from types import SimpleNamespace

from agent.runtime_session_store import RuntimeSessionStore
from agent.turn_facade_lease import LEASE_TTL_SECONDS, DurableTurnLease


def test_managed_worker_turn_survives_lease_refresh_tick(tmp_path):
    sent = []

    def rpc(method, **params):
        sent.append((method, params['operation'], params['payload']))
        return {'value': True}

    store = RuntimeSessionStore(rpc, {'session_id': 's'}, tmp_path / 'private')
    interrupts = []
    agent = SimpleNamespace(session_id='s', interrupt=lambda message, **kwargs: interrupts.append(message))
    lease = DurableTurnLease(agent, store, 's', 'holder', expires_at=time.time() + LEASE_TTL_SECONDS)
    lease.turn_active = True
    try:
        before = lease._authority_deadline
        assert lease.refresh_tick() is None
        assert interrupts == [] and lease.interrupt_message is None
        assert sent == [('worker.persist', 'turn.renew', {'holder': 'holder', 'ttl_seconds': LEASE_TTL_SECONDS})]
        assert lease._authority_deadline >= before
    finally:
        store.close()
