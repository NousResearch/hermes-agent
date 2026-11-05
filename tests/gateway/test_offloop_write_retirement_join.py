"""An authority write moved off the owner loop is still the authority's work: retirement joins it
before answering True, and the shutdown close gate counts it before closing state.db.

Each case drives the real verb against a real state.db; as its write starts, a second connection
takes the writer (another process holding state.db) until after the join or the gate decided.
"""
import asyncio
import importlib
import logging
import sqlite3
import threading
from types import SimpleNamespace

import pytest


def _owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_controls import AuthorityConnection
    import hermes_state_runtime as rt
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {})
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db

    async def handle(event):
        return 'ok'
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _running_agents={}, _evict_cached_agent=lambda route: None,
                             _handle_message=handle, _cached_agent_for=lambda route: None,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=db,
                                 epoch=rt.begin_runtime_epoch(db, instance_id='owner'))
    runner.session_authority = authority
    return store, authority, AuthorityConnection(authority, object(), {'user_id': 'human'})


def _local(authority, owner, tmp_path, source):
    from gateway.session_local import create_local_session
    return create_local_session(authority, owner.actor, dict(
        request_id='m', source=source, cwd=str(tmp_path), model='frozen', toolsets=[]))


class _HeldWriter:
    """Takes state.db's writer as the write under test starts; released by the test."""

    def __init__(self, path):
        self.path, self.conn = path, None
        self.started, self.finished = threading.Event(), threading.Event()

    def wrap(self, monkeypatch, module, name):
        target = importlib.import_module(module)
        real = getattr(target, name)

        def held(*args, **kwargs):
            if not self.started.is_set():
                self.conn = sqlite3.connect(self.path, isolation_level=None, timeout=0, check_same_thread=False)
                self.conn.execute('BEGIN IMMEDIATE')
                self.started.set()
            try:
                return real(*args, **kwargs)
            finally:
                self.finished.set()
        monkeypatch.setattr(target, name, held)

    def release(self):
        if self.conn is not None:
            self.conn.execute('ROLLBACK')
            self.conn.close()
            self.conn = None


async def _acp_detach(authority, owner, tmp_path, db):
    ref = _local(authority, owner, tmp_path, 'acp')
    snapshot = await authority.attach(owner.actor, ref)
    owner.subscriptions[ref.session_id] = snapshot.subscription_id
    return asyncio.create_task(owner.close())  # the WS handler's finally


async def _acp_idle_exit(authority, owner, tmp_path, db):
    from gateway.session_contract import Submission
    ref = _local(authority, owner, tmp_path, 'acp')
    await authority.submit(owner.actor, Submission('acp', ref, {'text': 'ACP'}, 'queue'))
    return authority.sessions[ref.session_id].task


async def _claim_reopen(authority, owner, tmp_path, db):
    from gateway.session_contract import Submission
    ref = _local(authority, owner, tmp_path, 'cli')
    db.end_session(ref.session_id, 'tui_shutdown')
    await authority.submit(owner.actor, Submission('first', ref, {'text': 'FIRST'}, 'queue'))
    return authority.sessions[ref.session_id].task


async def _create_title(authority, owner, tmp_path, db):
    return asyncio.create_task(owner.dispatch({'id': 1, 'method': 'session.create', 'params': {
        'request_id': 'titled', 'source': 'cli', 'cwd': str(tmp_path), 'model': 'frozen', 'toolsets': [],
        'title': 'Late Title'}}))


CASES = [
    ('acp-detach', 'gateway.session_acp_lifecycle', 'end_idle_local_session', _acp_detach),
    ('acp-idle-exit', 'gateway.session_acp_lifecycle', 'end_idle_local_session', _acp_idle_exit),
    ('claim-reopen', 'gateway.session_local_recovery', 'reopen_local_session', _claim_reopen),
    ('create-title', 'gateway.session_local_title', 'title_new_session', _create_title),
]


@pytest.mark.asyncio
@pytest.mark.parametrize('case,module,name,drive', CASES, ids=[c[0] for c in CASES])
async def test_retirement_answers_true_only_once_its_off_loop_writes_ended(
        tmp_path, monkeypatch, case, module, name, drive):
    from gateway.run_runtime import _retire_profile_authority
    store, authority, owner = _owner(tmp_path, monkeypatch)
    db = store._db
    writer = _HeldWriter(str(db.db_path))
    writer.wrap(monkeypatch, module, name)
    task = None
    try:
        task = await drive(authority, owner, tmp_path, db)
        assert await asyncio.to_thread(writer.started.wait, 10), case
        retired = await _retire_profile_authority(authority, timeout=0.5)
        # True would release the reservation with this write still to commit into the home.
        assert not retired or writer.finished.is_set(), case
    finally:
        writer.release()
        await asyncio.to_thread(writer.finished.wait, 30)
        pending = [*getattr(authority, '_mutation_tasks', ())]
        if task is not None:
            task.cancel()
            pending.append(task)
        await asyncio.gather(*pending, return_exceptions=True)
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_close_gate_counts_a_detach_write_its_cancelled_ws_handler_left(tmp_path, monkeypatch, caplog):
    from gateway.run_runtime import managed_turn_count
    store, authority, owner = _owner(tmp_path, monkeypatch)
    db = store._db
    writer = _HeldWriter(str(db.db_path))
    writer.wrap(monkeypatch, 'gateway.session_acp_lifecycle', 'end_idle_local_session')
    task = await _acp_detach(authority, owner, tmp_path, db)
    try:
        assert await asyncio.to_thread(writer.started.wait, 10)
        task.cancel()  # uvicorn's graceful-shutdown timeout ends the WS handler
        await asyncio.gather(task, return_exceptions=True)
        live = managed_turn_count(authority.runner)
        if not live:
            db.close()  # what the gate does when nothing is counted
        caplog.set_level(logging.WARNING)
        writer.release()
        assert await asyncio.to_thread(writer.finished.wait, 30)
        assert live, 'the gate saw no live writer'
        assert not [r for r in caplog.records if '#94736' in r.getMessage()]
    finally:
        writer.release()
        await asyncio.gather(*getattr(authority, '_mutation_tasks', ()), return_exceptions=True)
        store.close_all_db_handles()
