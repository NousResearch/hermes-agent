"""A watch notice pinned to a session that ended at a user boundary (/new) is dropped, not requeued
forever: the session authority refuses it ``admission_conflict`` on every drain tick (#141562).
Real GatewayRunner, DiscordAdapter, SessionAuthority and drain over SQLite; no agent turn runs."""
import asyncio
import logging
import queue

import pytest

from tests.module_clock import install_module_clock
from tests.waiting import async_wait_until


async def _gateway(monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_results import execution_result
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from plugins.platforms.discord.adapter import DiscordAdapter

    monkeypatch.setenv('DISCORD_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)

    async def send(chat_id, content, reply_to=None, metadata=None):
        return SendResult(success=True, message_id='sent')

    async def turn(event, source, quick_key, run_generation):
        captured = execution_result.get()
        if captured is not None:
            captured['result'] = {'final_response': 'reply', 'messages': []}
        return 'reply'
    monkeypatch.setattr(adapter, 'send', send)
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    # One real inbound turn commits the route a notice is admitted through.
    await adapter.handle_message(MessageEvent(text='start the build', source=source, message_id='m1'))
    key = runner.session_store._generate_session_key(source)
    await async_wait_until(lambda: _settled(authority, runner.session_store.peek_session_id(key)),
                           timeout=15, what='the inbound turn to settle')
    # Admitted notices stay queued from here on: these tests assert the admission, not its turn.
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    return runner, authority, runner.session_store.lookup_by_session_key(key)


def _settled(authority, session_id):
    from hermes_state_runtime import list_session_admissions
    rows = list_session_admissions(authority.db, session_id=session_id, pending_only=False) if session_id else []
    return bool(rows) and all(row['status'] not in ('queued', 'started') for row in rows)


def _notice(entry, pinned_session_id, kind='watch_match'):
    return {
        'type': kind, 'event_id': f'{kind}-1', 'session_id': 'proc_watch', 'started_at': 1.0,
        'session_key': entry.session_key, 'parent_session_id': pinned_session_id,
        'platform': 'discord', 'chat_type': 'dm', 'chat_id': '42', 'user_id': '42',
        'command': 'tail -f build.log', 'pattern': 'READY', 'output': 'READY', 'suppressed': 0,
        'seq': 1, 'interval': 60, 'elapsed': 60, 'omitted': 0,
    }


def _restart_wait(runner):
    """The restart's after-turn wait: the authority refuses new automation ``runtime_draining``."""
    from gateway.drain_control import open_restart_wait
    runner._running = runner._draining = True
    open_restart_wait(runner)


def _queued_admissions(authority, session_id):
    from hermes_state_runtime import list_session_admissions
    return [row for row in list_session_admissions(authority.db, session_id=session_id, pending_only=False)
            if row['principal_id'].startswith('automation:')]


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['watch_match', 'heartbeat'])
async def test_notice_pinned_to_a_reset_session_is_dropped_once(monkeypatch, caplog, kind):
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        retired = entry.session_id
        live = runner.session_store.reset_session(entry.session_key).session_id
        assert live != retired
        pending = queue.Queue()
        pending.put(_notice(entry, retired, kind))

        with caplog.at_level(logging.WARNING, logger='gateway.run'):
            await runner._drain_watch_notifications(pending)

        assert pending.empty(), 'a notice the closed session can never admit must not be requeued'
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, warnings
        assert 'permanently gone' in warnings[0]
        assert retired in warnings[0] and live in warnings[0] and 'admission_conflict' in warnings[0]
        assert not _queued_admissions(authority, live)
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_notice_pinned_to_a_compression_ancestor_still_admits(monkeypatch, caplog):
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        parent, child = entry.session_id, entry.session_id + '-compressed'
        runner.session_store._db_for_key(entry.session_key).publish_compression_child(
            parent_session_id=parent, child_session_id=child, source='discord',
            require_compression_lease=False, model='offline', model_config={},
            system_prompt='offline', messages=[{'role': 'user', 'content': 'retained'}],
        )
        assert runner.session_store.advance_compression_session(entry.session_key, parent, child)
        pending = queue.Queue()
        pending.put(_notice(entry, parent))

        with caplog.at_level(logging.WARNING, logger='gateway.run'):
            await runner._drain_watch_notifications(pending)

        assert pending.empty()
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        admitted = _queued_admissions(authority, parent) + _queued_admissions(authority, child)
        assert [row['status'] for row in admitted] == ['queued']
    finally:
        authority.db.close()


async def _foreign_pin(runner):
    """A live session of another chat: the authority refuses the notice, but nothing proves the pin
    gone (its owner may still appear, d6635ffe03b), so the refusal stays retryable."""
    from gateway.config import Platform
    adapter = runner.adapters[Platform.DISCORD]
    other = adapter.build_source(chat_id='43', chat_type='dm', user_id='42')
    return runner.session_store.get_or_create_session(other).session_id


def _route_gone(runner, entry):
    """A route that no longer holds a session (/stop suspended it): refused ``not_found`` for good."""
    assert runner.session_store.suspend_session(entry.session_key)
    return entry.session_id


@pytest.mark.asyncio
@pytest.mark.parametrize('refusal', ['permanent', 'foreign', 'transient'])
async def test_refused_notice_backs_off_and_only_a_permanent_refusal_drops(monkeypatch, caplog, refusal):
    """A refused watch notice is retried on a bounded backoff, never on every 2 s tick (#140899). A
    proven-permanent refusal (``not_found``: the route holds no session) drops it at the cap. A
    notice pinned to another chat's live session (``admission_conflict``) or refused by a restart's
    after-turn wait (``runtime_draining``) is never dropped: it waits however long it takes."""
    from gateway import run_notifications
    from gateway.run_notifications import _MAX_COMPLETION_INJECTION_ATTEMPTS
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        pinned = {'permanent': lambda: _route_gone(runner, entry), 'transient': lambda: entry.session_id}.get(
            refusal)
        pinned = pinned() if pinned else await _foreign_pin(runner)
        if refusal == 'transient':
            _restart_wait(runner)
        pending = queue.Queue()
        pending.put(_notice(entry, pinned))
        clock = install_module_clock(monkeypatch, run_notifications)

        with caplog.at_level(logging.WARNING, logger='gateway.run'):
            for _ in range(2000):  # ~33 simulated minutes of drain ticks
                if pending.empty():
                    break
                await runner._drain_watch_notifications(pending)
                clock.now += 1

        messages = [r.getMessage() for r in caplog.records]
        refused = sum('not admitted' in m for m in messages)
        if refusal == 'permanent':
            assert pending.empty()
            assert refused == _MAX_COMPLETION_INJECTION_ATTEMPTS
            assert sum('dropping the queued event' in m for m in messages) == 1
            return
        assert not pending.empty(), 'a notice whose refusal may still clear is never dropped'
        assert 20 < refused < 60, 'retries are spaced out (capped at 60 s), not attempted every tick'
        assert not [m for m in messages if 'dropping the queued event' in m]
        if refusal == 'transient':
            runner._draining = False
            clock.now += 120
            await runner._drain_watch_notifications(pending)
            assert pending.empty()
            assert [row['status'] for row in _queued_admissions(authority, entry.session_id)] == ['queued']
    finally:
        authority.db.close()


_REAL_SLEEP = asyncio.sleep


async def _run_async_watcher(runner, monkeypatch, clock, *, ticks):
    """Run the real async watcher for ``ticks`` passes, one simulated second apart."""
    seen = []

    async def tick(delay):
        seen.append(delay)
        clock.now += 1
        runner._running = len(seen) < ticks
        await _REAL_SLEEP(0)
    monkeypatch.setattr(asyncio, 'sleep', tick)
    runner._running = True
    await runner._async_delegation_watcher(interval=0)


def _persist_delegation(entry, pinned, delegation_id):
    import time
    from tools import async_delegation as delegation
    evt = dict(_notice(entry, pinned, 'async_delegation'), delegation_id=delegation_id,
               summary='review findings', status='completed', dispatched_at=time.time())
    delegation._persist_dispatch(evt)
    delegation._persist_completion(evt, {'status': 'completed', 'summary': 'review findings'})
    return evt


@pytest.mark.asyncio
async def test_permanently_refused_completion_converges_to_dropped_across_restarts(monkeypatch):
    """A proven-permanent refusal (``not_found``: the route holds no session) spends a durable
    delivery attempt in the ledger, so the count survives restarts: three processes that each
    replay the row and retry it for a few minutes converge to ``dropped`` instead of each refunding
    it (#141153)."""
    from gateway import run_notifications
    from tools import async_delegation as delegation
    from tools.process_registry import process_registry
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        _persist_delegation(entry, _route_gone(runner, entry), 'refused-1')
        clock = install_module_clock(monkeypatch, run_notifications)
        for _process in range(3):
            # Each process starts from the durable ledger alone: a fresh queue, replayed rows.
            monkeypatch.setattr(process_registry, 'completion_queue', queue.Queue())
            delegation._offered.clear()
            delegation.restore_undelivered_completions(process_registry.completion_queue)
            await _run_async_watcher(runner, monkeypatch, clock, ticks=150)
            row = delegation.get_durable_delegation('refused-1')
            if row['delivery_state'] != 'pending':
                break

        assert (row['delivery_state'], row['delivery_attempts']) == ('dropped', delegation._MAX_DELIVERY_ATTEMPTS)
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_live_pinned_completion_survives_a_long_restart_wait(monkeypatch):
    """A completion pinned to the LIVE session that the authority refuses only because a restart's
    after-turn wait is open (``runtime_draining``, up to ``restart_after_turn_timeout``) spends no
    delivery attempt however long the wait lasts, and is delivered once admission reopens."""
    from gateway import run_notifications
    from tools import async_delegation as delegation
    from tools.process_registry import process_registry
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        evt = _persist_delegation(entry, entry.session_id, 'live-1')
        monkeypatch.setattr(process_registry, 'completion_queue', queue.Queue())
        process_registry.completion_queue.put(dict(evt))
        clock = install_module_clock(monkeypatch, run_notifications)
        _restart_wait(runner)

        await _run_async_watcher(runner, monkeypatch, clock, ticks=1800)  # 30 simulated minutes
        row = delegation.get_durable_delegation('live-1')
        assert (row['delivery_state'], row['delivery_attempts']) == ('pending', 0)

        runner._draining = False
        await _run_async_watcher(runner, monkeypatch, clock, ticks=120)
        assert delegation.get_durable_delegation('live-1')['delivery_state'] == 'delivered'
        assert [r['status'] for r in _queued_admissions(authority, entry.session_id)] == ['queued']
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_foreign_pinned_completion_spends_nothing_across_restarts(monkeypatch):
    """A completion pinned to another chat's live session is refused ``admission_conflict`` but
    nothing proves its owner gone (d6635ffe03b): replayed by three processes for minutes each, it
    spends no delivery attempt, unlike a proven-permanent refusal (#141153)."""
    from gateway import run_notifications
    from tools import async_delegation as delegation
    from tools.process_registry import process_registry
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        _persist_delegation(entry, await _foreign_pin(runner), 'foreign-1')
        clock = install_module_clock(monkeypatch, run_notifications)
        for _process in range(3):
            monkeypatch.setattr(process_registry, 'completion_queue', queue.Queue())
            delegation._offered.clear()
            delegation.restore_undelivered_completions(process_registry.completion_queue)
            await _run_async_watcher(runner, monkeypatch, clock, ticks=150)
        row = delegation.get_durable_delegation('foreign-1')
        assert (row['delivery_state'], row['delivery_attempts']) == ('pending', 0)
    finally:
        authority.db.close()
