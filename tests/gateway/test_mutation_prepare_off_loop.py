"""The model/compress prepare transaction runs off the owner loop like its commit."""
import asyncio
import sqlite3
import threading
import time

import pytest

from tests.waiting import async_wait_until
from persistence_assembly import store_location
from tests.fixtures.storage_read import session_row


@pytest.mark.asyncio
async def test_model_prepare_waiting_on_a_contended_writer_keeps_the_loop_live(tmp_path, monkeypatch):
    """Another process takes state.db's write lock as the prepare-only BEGIN IMMEDIATE starts, so the
    prepare waits for it. That wait must not freeze every other session's loop work."""
    from types import SimpleNamespace
    from gateway import run, session_mutations
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_cli import model_switch
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {})
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model='switched', target_provider='custom', base_url='http://127.0.0.1:9/v1'))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store.persistence
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', persistence=db,
                                 epoch=db.runtime.begin_epoch(instance_id='owner'))
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='m', source='cli', cwd=str(tmp_path),
                                                            model='frozen', toolsets=[]))
    snap = session_row(db, ref.session_id)
    held, release = threading.Event(), threading.Event()

    def hold_writer():
        other = sqlite3.connect(store_location(db.scope), timeout=10)
        other.execute('BEGIN IMMEDIATE')
        held.set()
        release.wait(1.0)
        other.rollback()
        other.close()
    holder = threading.Thread(target=hold_writer)
    runtime = session_mutations.authority_persistence(authority).runtime
    real = runtime.prepare_mutation

    def contended(*args, **kwargs):
        # Another process takes the writer lock exactly as the prepare transaction starts.
        if not held.is_set():
            holder.start()
            held.wait(5)
        return real(*args, **kwargs)
    monkeypatch.setattr(runtime, 'prepare_mutation', contended)
    gaps = []

    stopped = asyncio.Event()

    async def ticker():
        # A fixed-cadence probe until the test stops it: a gap beyond the tick is a blocked loop.
        last = time.monotonic()
        while not stopped.is_set():
            await asyncio.sleep(0.02)
            now = time.monotonic()
            gaps.append(now - last)
            last = now
    # The first dispatch imports the wire-contract registry (a one-time cold import of a few hundred
    # ms on a loaded runner); warm it so the probe measures the prepare writer wait, not module load.
    import tui_gateway.contracts.registry
    tick = asyncio.create_task(ticker())
    try:
        await async_wait_until(lambda: gaps, timeout=10, what='the loop-stall probe ticking')
        response = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'switch', 'expected_revision': snap['runtime_revision'],
            'expected_generation': snap['runtime_generation'], 'operation': 'model', 'payload': {'model': 'switched'}}})
        assert response['result']['model'] == 'switched', response
        assert held.is_set() and max(gaps) < 0.5, f'the owner loop stalled {max(gaps):.2f}s behind the prepare writer'
    finally:
        stopped.set()
        tick.cancel()
        release.set()
        if holder.is_alive():
            holder.join(5)
        await owner.close()
        store.close_all_db_handles()
