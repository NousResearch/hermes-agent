"""A write refused on an event-loop thread is never sent, so it must not poison later writes."""
import asyncio

import pytest

from agent.runtime_session_store import RuntimeSessionStore, WorkerPersistenceError


def test_loop_thread_refusal_is_not_sticky(tmp_path):
    sent = []

    def rpc(method, **params):  # WorkerRPC's own guard: refuse on a running loop, before any I/O
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            sent.append(params['operation'])
            return {'value': True}
        raise WorkerPersistenceError('synchronous_rpc_on_event_loop')

    store = RuntimeSessionStore(rpc, {'session_id': 's'}, tmp_path / 'private')
    try:
        async def best_effort_on_loop():
            with pytest.raises(WorkerPersistenceError, match='synchronous_rpc_on_event_loop'):
                store.touch_session_activity('s', 1.0, description='tool running')

        asyncio.run(best_effort_on_loop())
        assert sent == [] and store.failure is None and store.journal['pending'] == []
        store.touch_session_activity('s', 2.0, description='after')  # off-loop write still lands
        assert store.flush_token_counts() is True
        assert sent == ['session.activity']
    finally:
        store.close()
