"""The on-loop guard names its caller at WARNING (#139029): a best-effort write swallows the raise
at debug, and only the sticky follow-up failures reached the logs."""
import asyncio
import logging

import pytest

SECRET = 'arg-value-never-logged'


def _no_owner(*args, **kwargs):
    raise AssertionError('the guard must fire before owner discovery')


def _guard_warnings(caplog):
    return [r.getMessage() for r in caplog.records
            if r.name == 'agent.runtime_session_store' and r.levelno == logging.WARNING]


def test_on_loop_rpc_logs_caller_frames_without_argument_values(tmp_path, monkeypatch, caplog):
    from agent.runtime_session_store import WorkerPersistenceError, WorkerRPC
    monkeypatch.setattr('hermes_cli.gateway_runtime.discover_gateway_endpoint', _no_owner)
    rpc = WorkerRPC(tmp_path)

    def persist_from_callback():
        return rpc('worker.persist', payload=SECRET)

    async def on_loop():
        persist_from_callback()

    caplog.set_level(logging.WARNING, logger='agent.runtime_session_store')
    with pytest.raises(WorkerPersistenceError, match='synchronous_rpc_on_event_loop'):
        asyncio.run(on_loop())
    [message] = _guard_warnings(caplog)
    assert 'synchronous_rpc_on_event_loop' in message and 'worker.persist' in message
    assert f'{__name__}:persist_from_callback:' in message
    assert f'{__name__}:on_loop:' in message
    assert SECRET not in message


def test_on_loop_store_write_names_the_caller_not_the_store(tmp_path, monkeypatch, caplog):
    """Through the store, the summary skips its own frames (_apply, retry_pending) so the
    swallowing best-effort caller is what the log names."""
    from agent.runtime_session_store import RuntimeSessionStore, WorkerPersistenceError, WorkerRPC
    monkeypatch.setattr('hermes_cli.gateway_runtime.discover_gateway_endpoint', _no_owner)
    store = RuntimeSessionStore(WorkerRPC(tmp_path), {'session_id': 's'}, tmp_path / 'outbox')

    def best_effort_usage():
        try:
            store.record_auxiliary_usage('s', 'approval', input_tokens=7, model=SECRET)
        except WorkerPersistenceError:
            pass

    async def on_loop():
        best_effort_usage()

    caplog.set_level(logging.WARNING, logger='agent.runtime_session_store')
    try:
        asyncio.run(on_loop())
        [message] = _guard_warnings(caplog)
        assert message.split('caller: ', 1)[1].startswith(f'{__name__}:best_effort_usage:')
        assert 'agent.runtime_session_store' not in message.split('caller: ', 1)[1]
        assert SECRET not in message
        assert store.failure == 'synchronous_rpc_on_event_loop'
    finally:
        store._outbox_owner.close()
