"""The model/compress prepare transaction runs off the owner loop like its commit."""
import asyncio
import sqlite3
import threading
import time

import pytest


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
    import hermes_state_runtime as rt
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {})
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model='switched', target_provider='custom', base_url='http://127.0.0.1:9/v1'))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='m', source='cli', cwd=str(tmp_path),
                                                            model='frozen', toolsets=[]))
    snap = db.get_session(ref.session_id)
    held, release = threading.Event(), threading.Event()

    def hold_writer():
        other = sqlite3.connect(db.db_path, timeout=10)
        other.execute('BEGIN IMMEDIATE')
        held.set()
        release.wait(1.0)
        other.rollback()
        other.close()
    holder = threading.Thread(target=hold_writer)
    real = session_mutations.mutate_runtime_session

    def contended(*args, **kwargs):
        # Another process takes the writer lock exactly as the prepare transaction starts.
        if kwargs.get('_prepare_only') and not held.is_set():
            holder.start()
            held.wait(5)
        return real(*args, **kwargs)
    monkeypatch.setattr(session_mutations, 'mutate_runtime_session', contended)
    gaps = []

    async def ticker():
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.02)
            now = time.monotonic()
            gaps.append(now - last)
            last = now
    tick = asyncio.create_task(ticker())
    try:
        await asyncio.sleep(0.05)
        response = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'switch', 'expected_revision': snap['runtime_revision'],
            'expected_generation': snap['runtime_generation'], 'operation': 'model', 'payload': {'model': 'switched'}}})
        assert response['result']['model'] == 'switched', response
        assert held.is_set() and max(gaps) < 0.5, f'the owner loop stalled {max(gaps):.2f}s behind the prepare writer'
    finally:
        tick.cancel()
        release.set()
        if holder.is_alive():
            holder.join(5)
        await owner.close()
        store.close_all_db_handles()
