"""The claim phase's reopen of a finalized local row is a write: it runs off the owner loop."""
import asyncio
import sqlite3
import threading
import time

import pytest


@pytest.mark.asyncio
async def test_reopening_a_finalized_local_row_behind_a_contended_writer_keeps_the_loop_live(tmp_path, monkeypatch):
    """The first turn of a finalized local session reopens its row inside the claim. With another
    writer holding state.db's lock at that moment, the reopen waits for it; the loop must not."""
    from types import SimpleNamespace
    from gateway import run, session_operator
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import Submission
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    import hermes_state_runtime as rt
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    executed = []

    async def handle(event):
        executed.append(event.text)
        return 'ok'
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _handle_message=handle,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='m', source='cli', cwd=str(tmp_path),
                                                            model='frozen', toolsets=[]))
    db.end_session(ref.session_id, 'tui_shutdown')
    held, release = threading.Event(), threading.Event()

    def hold_writer():
        other = sqlite3.connect(db.db_path, timeout=10)
        other.execute('BEGIN IMMEDIATE')
        held.set()
        release.wait(1.0)
        other.rollback()
        other.close()
    holder = threading.Thread(target=hold_writer)
    check = session_operator.check_local_input

    def check_then_contend(*args, **kwargs):
        # The last preflight before the reopen: another process takes the writer lock now.
        check(*args, **kwargs)
        if not holder.is_alive() and not held.is_set():
            holder.start()
            held.wait(5)
    monkeypatch.setattr(session_operator, 'check_local_input', check_then_contend)
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
        queued = await authority.submit(owner.actor, Submission('first', ref, {'text': 'FIRST'}, 'queue'))
        async with asyncio.timeout(10):
            while rt.get_session_admission(db, admission_id=queued.admission_id)['status'] != 'terminal':
                await asyncio.sleep(0.02)
        assert executed == ['FIRST']
        assert db.get_session(ref.session_id)['end_reason'] is None
        assert max(gaps) < 0.5, f'the owner loop stalled {max(gaps):.2f}s behind the reopen writer'
    finally:
        tick.cancel()
        release.set()
        if holder.is_alive():
            holder.join(5)
        await owner.close()
        store.close_all_db_handles()
