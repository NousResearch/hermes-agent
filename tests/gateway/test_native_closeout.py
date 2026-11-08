"""Native explicit closeout bypasses both busy guards without resetting old work."""
from types import SimpleNamespace
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent, coerce_plaintext_gateway_command
from gateway.run_busy import GatewayBusySessionMixin
from gateway.session import SessionSource, SessionStore
from hermes_cli.commands import is_interrupt_then_dispatch, should_bypass_active_session


@pytest.mark.parametrize('text', ['archive', 'archive this', 'archive the chat', '/archive'])
def test_explicit_archive_is_control_before_adapter_busy_queue(text):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='fixture', user_id='owner', chat_type='dm')
    event = MessageEvent(text=text, source=source)
    coerce_plaintext_gateway_command(event)
    assert event.get_command() == 'archive'
    assert should_bypass_active_session(event.get_command())
    assert not is_interrupt_then_dispatch(event.get_command())


@pytest.mark.parametrize('text', ['please explain archive', '`archive`', '> archive',
                                  'archive\n\n[Hermes closeout: ' + 'a' * 32 + ']'])
def test_mentions_quotes_and_owner_review_are_not_commands(text):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='fixture', user_id='owner', chat_type='dm')
    event = MessageEvent(text=text, source=source)
    coerce_plaintext_gateway_command(event)
    assert event.get_command() is None


def test_new_and_reset_have_distinct_interrupt_semantics():
    assert should_bypass_active_session('new') and not is_interrupt_then_dispatch('new')
    assert is_interrupt_then_dispatch('reset')


@pytest.mark.asyncio
async def test_busy_new_does_not_interrupt_old_but_reset_keeps_existing_path(monkeypatch):
    import gateway.session_closeout_native as native
    calls = []
    async def closeout(runner, event, intent):
        calls.append(intent)
        return 'preservation-started'
    monkeypatch.setattr(native, 'start_native_closeout', closeout)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='fixture', user_id='owner')
    class Runner:
        async def _interrupt_and_clear_session(self, *args, **kwargs):
            calls.append('interrupt')
        async def _handle_reset_command(self, event):
            calls.append('reset')
            return 'old-reset'
    runner = Runner()
    assert await GatewayBusySessionMixin._busy_new_command(runner, MessageEvent(text='/new', source=source), 'route', source) == 'preservation-started'
    assert calls == ['new']
    monkeypatch.setitem(sys.modules, 'gateway.run', SimpleNamespace(_INTERRUPT_REASON_RESET='reset'))
    assert await GatewayBusySessionMixin._busy_new_command(runner, MessageEvent(text='/reset', source=source), 'route', source) == 'old-reset'
    assert calls == ['new', 'interrupt', 'reset']


def test_forced_new_retries_keep_distinct_root_and_do_not_end_old(tmp_path):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='fixture', user_id='owner')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    old = store.get_or_create_session(source)
    fresh = store.get_or_create_session(source, force_new=True, create_request_id='closeout-new-fixture')
    retry = store.get_or_create_session(source, force_new=True, create_request_id='closeout-new-fixture')
    assert fresh.session_id == retry.session_id != old.session_id
    db = store._db_for_key(old.session_key)
    assert db.get_session(old.session_id)['ended_at'] is None
    assert db.get_session(fresh.session_id)['parent_session_id'] is None


def test_forced_new_waits_for_conflicting_lookup_then_creates_its_own_root(tmp_path, monkeypatch):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='fixture', user_id='owner')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    old = store.get_or_create_session(source)
    entered, joined, release = Event(), Event(), Event()
    original = store._get_or_create_session_impl

    def held_lookup(*args, **kwargs):
        if not kwargs['force_new']:
            entered.set()
            assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, '_get_or_create_session_impl', held_lookup)
    with ThreadPoolExecutor(max_workers=2) as pool:
        ordinary = pool.submit(store.get_or_create_session, source)
        assert entered.wait(5)
        slot = store._inflight_sessions[old.session_key]
        wait = slot.event.wait
        def observe_wait(timeout=None):
            joined.set()
            return wait(timeout)
        monkeypatch.setattr(slot.event, 'wait', observe_wait)
        fresh_future = pool.submit(store.get_or_create_session, source, force_new=True,
                                   create_request_id='stable-closeout-create')
        assert joined.wait(5)
        release.set()
        assert ordinary.result(timeout=5).session_id == old.session_id
        fresh = fresh_future.result(timeout=5)
    assert fresh.session_id != old.session_id
    db = store._db_for_key(old.session_key)
    assert db.get_session(fresh.session_id)['parent_session_id'] is None
    assert not db.get_session(fresh.session_id)['archived']
    assert store.get_or_create_session(source, force_new=True,
        create_request_id='stable-closeout-create').session_id == fresh.session_id


def test_equivalent_forced_creates_still_share_one_flight(tmp_path, monkeypatch):
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='fixture', user_id='owner')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    entered, joined, release = Event(), Event(), Event()
    original = store._get_or_create_session_impl
    calls = []
    def held_create(*args, **kwargs):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(store, '_get_or_create_session_impl', held_create)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(store.get_or_create_session, source, force_new=True, create_request_id='same-create')
        assert entered.wait(5)
        slot = store._inflight_sessions[store._generate_session_key(source)]
        wait = slot.event.wait
        def observe_wait(timeout=None):
            joined.set()
            return wait(timeout)
        monkeypatch.setattr(slot.event, 'wait', observe_wait)
        second = pool.submit(store.get_or_create_session, source, force_new=True, create_request_id='same-create')
        assert joined.wait(5)
        release.set()
        assert first.result(timeout=5).session_id == second.result(timeout=5).session_id
    assert calls == [1]


@pytest.mark.asyncio
async def test_queued_native_review_and_late_capture_stay_on_old_ref_after_route_moves(tmp_path, monkeypatch):
    import asyncio
    import gateway.session_closeout_native as native
    from gateway.session import AsyncSessionStore
    from gateway.session_authority import SessionAuthority
    from hermes_cli.conversation_closeout import CloseoutStore
    from hermes_state_runtime import begin_runtime_epoch, list_session_admissions
    monkeypatch.setattr(native, 'PRESERVATION_WAIT_SECONDS', 0)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='fixture', user_id='owner', chat_type='dm')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    old = store.get_or_create_session(source)
    db = store._db_for_key(old.session_key)
    epoch = begin_runtime_epoch(db, instance_id='fixture-owner')
    notices = []
    async def ack(*args):
        notices.append(args)
    runner = SimpleNamespace(_draining=False, session_store=store, async_session_store=AsyncSessionStore(store),
        _is_user_authorized_for_source=lambda src: True, _check_slash_access=lambda *args: None,
        _session_key_for_source=store._generate_session_key, _send_command_ack=ack)
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='fixture-owner', db=db, epoch=epoch)
    runner.session_authority = authority
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    result = await native.start_native_closeout(runner, MessageEvent(text='/new', source=source, message_id='fixture-new'), 'new')
    assert 'Preserving' in result
    # Move the native route before the asynchronous review admission executes.
    fresh = store.get_or_create_session(source, force_new=True, create_request_id='peer-new-fixture')
    await asyncio.gather(*authority._closeout_jobs)
    old_rows = list_session_admissions(db, session_id=old.session_id)
    assert len(old_rows) == 1 and old_rows[0]['status'] == 'queued'
    assert list_session_admissions(db, session_id=fresh.session_id) == []
    operation = old_rows[0]['request_id'].removeprefix('closeout-review-')
    receipts = CloseoutStore(db, epoch=epoch)
    assert receipts.claim_capture(operation, old.session_id)
    receipts.record_capture(operation, old.session_id, memory='accepted', source_id='late-source')
    assert receipts.read(operation)['memory_source_id'] == 'late-source'
    assert db.get_session(old.session_id)['archived'] and not db.get_session(fresh.session_id)['archived']
    assert store.peek_session_id(old.session_key) == fresh.session_id


@pytest.mark.asyncio
@pytest.mark.parametrize('newer_intent', ['restore', 'supersede'])
async def test_native_publication_drops_released_or_superseded_committed_receipt(tmp_path, monkeypatch, newer_intent):
    import asyncio
    import gateway.session_closeout_native as native
    import hermes_cli.conversation_closeout_jobs as jobs
    from gateway.session import AsyncSessionStore
    from gateway.session_authority import SessionAuthority
    from hermes_cli.conversation_closeout import CloseoutStore
    from hermes_state_runtime import begin_runtime_epoch
    monkeypatch.setattr(native, 'PRESERVATION_WAIT_SECONDS', 0)
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='fixture', user_id='owner', chat_type='dm')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    old = store.get_or_create_session(source)
    db = store._db_for_key(old.session_key)
    epoch = begin_runtime_epoch(db, instance_id='fixture-owner')
    notices, creates, snapshots = [], [], []
    async def ack(*args):
        notices.append(args)
    runner = SimpleNamespace(_draining=False, session_store=store, async_session_store=AsyncSessionStore(store),
        _is_user_authorized_for_source=lambda src: True, _check_slash_access=lambda *args: None,
        _session_key_for_source=store._generate_session_key, _send_command_ack=ack)
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='fixture-owner', db=db, epoch=epoch)
    runner.session_authority = authority
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    original_create = store.get_or_create_session
    def observe_create(*args, **kwargs):
        if kwargs.get('force_new'):
            creates.append(1)
        return original_create(*args, **kwargs)
    monkeypatch.setattr(store, 'get_or_create_session', observe_create)
    committed, release = asyncio.Event(), asyncio.Event()
    original_begin = jobs.begin_closeout
    def held_publication(*args, **kwargs):
        publish = kwargs['publish']
        async def paused(receipt):
            snapshots.append(receipt)
            committed.set()
            await release.wait()
            return await publish(receipt)
        return original_begin(*args, **{**kwargs, 'publish': paused})
    monkeypatch.setattr(jobs, 'begin_closeout', held_publication)
    await native.start_native_closeout(runner,
        MessageEvent(text='/new', source=source, message_id='fixture-closeout'), 'new')
    try:
        await asyncio.wait_for(committed.wait(), 5)
        assert db.get_session(old.session_id)['archived']
        receipts = CloseoutStore(db, epoch=epoch)
        operation = snapshots[0]['closeout_id']
        if newer_intent == 'restore':
            db.set_session_archived(old.session_id, False)
        else:
            replacement, created = receipts.prepare(snapshots[0]['old'], 'newer-request', 'archive', 'newer-cutoff')
            assert created and replacement['closeout_id'] != operation
    finally:
        release.set()
    await asyncio.gather(*authority._closeout_jobs)
    assert creates == [] and notices == []
    assert store.peek_session_id(old.session_key) == old.session_id
    if newer_intent == 'restore':
        assert not db.get_session(old.session_id)['archived']
        assert receipts.read(operation)['_fence_released']
    # The publication cancellation never discards an independently received source ID.
    assert receipts.claim_capture(operation, old.session_id)
    receipts.record_capture(operation, old.session_id, memory='accepted', source_id='late-source')
    assert receipts.read(operation)['memory_source_id'] == 'late-source'


@pytest.mark.asyncio
async def test_cached_native_closeout_does_not_announce_archival_after_restore(tmp_path, monkeypatch):
    import gateway.session_closeout_native as native
    import hermes_cli.conversation_closeout_jobs as jobs
    from gateway.session import AsyncSessionStore
    from hermes_cli.conversation_closeout import CloseoutStore, public_receipt
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='fixture', user_id='owner', chat_type='dm')
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    old = store.get_or_create_session(source)
    db = store._db_for_key(old.session_key)
    notices = []
    async def ack(*args):
        notices.append(args)
    runner = SimpleNamespace(_draining=False, session_store=store, async_session_store=AsyncSessionStore(store),
        _is_user_authorized_for_source=lambda src: True, _check_slash_access=lambda *args: None,
        _session_key_for_source=store._generate_session_key, _send_command_ack=ack)
    def restored_before_reply(db, old, request_id, intent, **kwargs):
        receipts = CloseoutStore(db)
        record, _ = receipts.prepare(old, request_id, intent, 'old:0')
        operation = record['closeout_id']
        receipts.record_review(operation, {'memory': 'pending', 'tasks': 'pending'}, 'old:0')
        committed = receipts.archive(operation, settled=False)
        db.set_session_archived(old['stored_session_id'], False)
        return public_receipt(committed)
    monkeypatch.setattr(jobs, 'begin_closeout', restored_before_reply)
    result = await native.start_native_closeout(runner,
        MessageEvent(text='/new', source=source, message_id='fixture-closeout'), 'new')
    assert 'released or superseded' in result
    assert store.peek_session_id(old.session_key) == old.session_id and not notices
    assert not db.get_session(old.session_id)['archived']
