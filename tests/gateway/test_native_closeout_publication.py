"""A restored closeout cannot publish after waiting for a conflicting route lookup."""
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import sqlite3

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, coerce_plaintext_gateway_command
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from hermes_cli.conversation_closeout import CloseoutStore, public_receipt
from hermes_state import SessionDB


async def _publication_phase(event, task, label):
    waiter = asyncio.create_task(event.wait())
    try:
        done, _ = await asyncio.wait((waiter, task), timeout=30, return_when=asyncio.FIRST_COMPLETED)
        if task in done:
            await task  # Surface the production exception rather than reporting a barrier timeout.
            pytest.fail('Native publication finished before ' + label)
        assert waiter in done, 'Timed out waiting for ' + label
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('separate_routing_db', [False, True])
@pytest.mark.parametrize('newer_intent', ['restore', 'supersede'])
async def test_waiting_native_publication_obeys_newer_intent(tmp_path, monkeypatch, separate_routing_db, newer_intent):
    import gateway.session_closeout_native as native
    import hermes_cli.conversation_closeout_jobs as jobs
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='fixture', user_id='owner', chat_type='dm')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    old = store.get_or_create_session(source)
    db = store._db_for_key(old.session_key)
    if separate_routing_db:
        routing_db = SessionDB(tmp_path / 'routing' / 'state.db')
        monkeypatch.setattr(SessionStore, '_routing_db', property(lambda self: routing_db))
        store._save_entries()
    notices, operations = [], []
    entered, release = Event(), Event()
    ready, joined = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()
    original = store._get_or_create_session_impl

    def held_lookup(*args, **kwargs):
        if not kwargs.get('force_new'):
            entered.set()
            assert release.wait(60), 'Conflicting lookup was not released'
        return original(*args, **kwargs)

    async def ack(*args):
        notices.append(args)

    runner = SimpleNamespace(_draining=False, session_store=store, async_session_store=AsyncSessionStore(store),
        _is_user_authorized_for_source=lambda src: True, _check_slash_access=lambda *args: None,
        _session_key_for_source=store._generate_session_key, _send_command_ack=ack)
    with ThreadPoolExecutor(max_workers=1) as pool:
        flights = []

        def committed_before_wait(db, target, request_id, intent, **kwargs):
            receipts = CloseoutStore(db)
            record, _ = receipts.prepare(target, request_id, intent, 'fixture:0')
            operation = record['closeout_id']
            operations.append(operation)
            receipts.record_review(operation, {'memory': 'pending', 'tasks': 'pending'}, 'fixture:0')
            record = receipts.archive(operation, settled=False)
            monkeypatch.setattr(store, '_get_or_create_session_impl', held_lookup)
            flights.append(pool.submit(store.get_or_create_session, source, touch_activity=False))
            assert entered.wait(30), 'Conflicting lookup did not start'
            slot = store._inflight_sessions[old.session_key]
            wait = slot.event.wait

            def observe_wait(timeout=None):
                loop.call_soon_threadsafe(joined.set)
                return wait(timeout)

            monkeypatch.setattr(slot.event, 'wait', observe_wait)
            ready.set()
            return public_receipt(record)

        monkeypatch.setattr(jobs, 'begin_closeout', committed_before_wait)
        task = asyncio.create_task(native.start_native_closeout(runner,
            MessageEvent(text='/new', source=source, message_id='fixture-closeout'), 'new'))
        try:
            # The join budget starts after cold route/receipt setup and actual lookup entry.
            await _publication_phase(ready, task, 'conflicting lookup setup')
            await _publication_phase(joined, task, 'publisher joining the conflicting lookup')
            receipts = CloseoutStore(db)
            operation = operations[0]
            if newer_intent == 'restore':
                db.set_session_archived(old.session_id, False)
            else:
                receipts.prepare(receipts.read(operation)['old'], 'newer-request', 'archive', 'newer-cutoff')
        finally:
            release.set()
            result = await asyncio.wait_for(task, 30)
            if flights:
                await asyncio.wait_for(asyncio.wrap_future(flights[0]), 30)
    assert 'released or superseded' in result
    assert store.peek_session_id(old.session_key) == old.session_id
    durable = store._routing_db.load_gateway_routing_entries(scope=store._routing_scope())
    assert json.loads(durable[old.session_key])['session_id'] == old.session_id
    assert db._read_one('SELECT COUNT(*) AS n FROM sessions')['n'] == 1
    assert not notices
    if newer_intent == 'restore':
        assert not db.get_session(old.session_id)['archived']


def _archived_native(tmp_path, monkeypatch, separate_routing_db):
    from hermes_cli.conversation_closeout_jobs import old_identity
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='fixture', user_id='owner', chat_type='dm')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    old = store.get_or_create_session(source)
    db = store._db_for_key(old.session_key)
    if separate_routing_db:
        routing_db = SessionDB(tmp_path / 'routing' / 'state.db')
        monkeypatch.setattr(SessionStore, '_routing_db', property(lambda self: routing_db))
        store._save_entries()
    receipts = CloseoutStore(db)
    record, _ = receipts.prepare(old_identity(db, old.session_id, profile='default'), 'fixture', 'new', 'fixture:0')
    operation = record['closeout_id']
    receipts.record_review(operation, {'memory': 'pending', 'tasks': 'pending'}, 'fixture:0')
    receipts.archive(operation, settled=False)
    return store, db, source, old, operation


@pytest.mark.parametrize('separate_routing_db', [False, True])
def test_guarded_native_route_creates_once_and_persists_independent_identity(tmp_path, monkeypatch, separate_routing_db):
    from gateway.session_closeout_publication import publish_native_route
    store, db, source, old, operation = _archived_native(tmp_path, monkeypatch, separate_routing_db)
    record, fresh = publish_native_route(store, source, operation)
    assert fresh.session_id != old.session_id
    assert record['archived'] and db.get_session(old.session_id)['archived']
    row = db.get_session(fresh.session_id)
    assert not row['archived'] and row['parent_session_id'] is None
    assert row['session_key'] == old.session_key and row['user_id'] == 'owner'
    assert publish_native_route(store, source, operation)[1] is None
    assert store.get_or_create_session(source, force_new=True,
        create_request_id=record['next_create_request_id']).session_id == fresh.session_id
    assert db._read_one('SELECT COUNT(*) AS n FROM sessions')['n'] == 2
    durable = store._routing_db.load_gateway_routing_entries(scope=store._routing_scope())
    assert json.loads(durable[old.session_key])['session_id'] == fresh.session_id


class _CommitFault:
    def __init__(self, conn, *, retry, restore):
        self.conn, self.retry, self.restore = conn, retry, restore
        self.armed = self.injected = False

    def __getattr__(self, name):
        return getattr(self.conn, name)

    def commit(self):
        if self.armed and not self.injected:
            self.injected = True
            if self.retry:
                raise sqlite3.OperationalError('database is locked')
            raise OSError('fixture commit failure')
        return self.conn.commit()

    def rollback(self):
        self.conn.rollback()
        if self.injected and self.restore is not None:
            restore, self.restore = self.restore, None
            restore()


@pytest.mark.parametrize('separate_routing_db', [False, True])
@pytest.mark.parametrize('failure', ['commit', 'retry', 'retry_restore', 'route', 'retired'])
def test_guarded_native_route_rolls_back_or_retries_exact_local_checkpoint(
        tmp_path, monkeypatch, separate_routing_db, failure):
    import hermes_state_worker_lifecycle as lifecycle
    from gateway.session_closeout_publication import publish_native_route
    store, db, source, old, operation = _archived_native(tmp_path, monkeypatch, separate_routing_db)
    routing_db = store._routing_db
    original_insert = lifecycle.insert_session_row_in_transaction
    if failure in {'commit', 'retry', 'retry_restore'}:
        restorer = SessionDB(db_path=db.db_path)
        fault = _CommitFault(db._conn, retry=failure != 'commit',
            restore=(lambda: restorer.set_session_archived(old.session_id, False)) if failure == 'retry_restore' else None)
        monkeypatch.setattr(db, '_conn', fault)
        def insert(*args, **kwargs):
            original_insert(*args, **kwargs)
            fault.armed = True
        monkeypatch.setattr(lifecycle, 'insert_session_row_in_transaction', insert)
    else:
        save = routing_db.save_gateway_routing_entry
        def route_write(*args, **kwargs):
            if failure == 'retired':
                return
            save(*args, **kwargs)
            raise OSError('fixture route failure')
        monkeypatch.setattr(routing_db, 'save_gateway_routing_entry', route_write)
    if failure in {'commit', 'route', 'retired'}:
        with pytest.raises((OSError, ValueError)):
            publish_native_route(store, source, operation)
    else:
        result = publish_native_route(store, source, operation)
        assert (result is None) == (failure == 'retry_restore')
    selected = store.peek_session_id(old.session_key)
    durable = routing_db.load_gateway_routing_entries(scope=store._routing_scope())
    assert json.loads(durable[old.session_key])['session_id'] == selected
    assert (selected != old.session_id) == (failure == 'retry')
    assert db._read_one('SELECT COUNT(*) AS n FROM sessions')['n'] == (2 if failure == 'retry' else 1)
    if failure == 'retry_restore':
        assert not db.get_session(old.session_id)['archived']
        assert CloseoutStore(db).read(operation)['_fence_released']


@pytest.mark.asyncio
async def test_restore_during_title_await_suppresses_stale_notice(tmp_path, monkeypatch):
    from gateway.session_closeout_native import _publish_native_closeout
    store, db, source, old, operation = _archived_native(tmp_path, monkeypatch, False)
    notices = []
    async def title(*args):
        db.set_session_archived(old.session_id, False)
    async def ack(*args):
        notices.append(args)
    runner = SimpleNamespace(session_store=store, _reset_titled_header=title, _send_command_ack=ack)
    receipt = public_receipt(CloseoutStore(db).read(operation))
    result = await _publish_native_closeout(runner, MessageEvent(text='/new fixture-title', source=source),
        'new', db, None, None, receipt)
    assert result is False and not notices
    assert not db.get_session(old.session_id)['archived']


@pytest.mark.parametrize('chat_type,expected', [('dm', '/restart'), ('group', 'restart gateway')])
def test_moved_restart_grammar_retains_existing_scope(chat_type, expected):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='fixture', user_id='owner', chat_type=chat_type)
    event = MessageEvent(text='restart gateway', source=source)
    coerce_plaintext_gateway_command(event)
    assert event.text == expected
