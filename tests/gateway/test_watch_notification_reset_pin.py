"""A watch notice pinned to a session that ended at a user boundary (/new) is dropped, not requeued
forever: the session authority refuses it ``admission_conflict`` on every drain tick (#141562).
Real GatewayRunner, DiscordAdapter, SessionAuthority and drain over SQLite; no agent turn runs."""
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
        assert f'session {retired} was closed at a user boundary' in warnings[0]
        assert retired in warnings[0] and live in warnings[0] and 'admission_conflict' in warnings[0]
        assert not _queued_admissions(authority, live)
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('pin, cause', [
    ('no-such-session', 'session no-such-session no longer exists'),
    ('worker', 'session worker is a subagent transcript with no owning conversation'),
])
async def test_notice_pinned_to_a_non_owner_is_dropped_with_its_cause(monkeypatch, caplog, pin, cause):
    """A pinned row the store no longer has, or a subagent transcript with no ``_delegate_from`` owner,
    is as gone as a closed session; the warning names that cause instead of blaming a /new."""
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        if pin == 'worker':
            runner.session_store._db_for_key(entry.session_key).create_session('worker', source='subagent')
        pending = queue.Queue()
        pending.put(_notice(entry, pin))

        with caplog.at_level(logging.WARNING, logger='gateway.run'):
            await runner._drain_watch_notifications(pending)

        assert pending.empty()
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, warnings
        assert 'permanently gone' in warnings[0] and cause in warnings[0]
        assert 'user boundary' not in warnings[0]
    finally:
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['new_session', 'tui_close'])
async def test_watch_notice_pinned_to_a_viewer_closed_local_session_is_kept(tmp_path, monkeypatch, caplog, reason):
    """A local conversation the viewer closed keeps its own work as a note on that session (LC-7): even a
    user-boundary close is not gone for the watch drain, refused or admitted."""
    from gateway import session_automation
    from hermes_state_local import end_idle_local_session
    from hermes_state_runtime import RuntimeStoreError
    from tests.gateway.test_session_end_delegations import _local, _notes
    runner, authority, ref, _actor = await _local(tmp_path, monkeypatch)
    try:
        route = authority.sessions[ref.session_id].route
        target = runner.session_store.lookup_by_session_key(route).session_id
        assert end_idle_local_session(authority.db, epoch=authority.epoch, session_id=ref.session_id,
                                      target_id=target, reason=reason)
        notice = {'type': 'watch_match', 'event_id': 'evt-1', 'session_id': 'proc_watch', 'started_at': 1.0,
                  'session_key': route, 'parent_session_id': target, 'command': 'make',
                  'pattern': 'READY', 'output': 'RESULT_proc_watch', 'suppressed': 0}
        pending = queue.Queue()
        pending.put(notice)
        admit = session_automation.admit_process_notice

        async def refuse(*_args):
            raise RuntimeStoreError('admission_conflict')
        monkeypatch.setattr(session_automation, 'admit_process_notice', refuse)

        with caplog.at_level(logging.WARNING, logger='gateway.run'):
            await runner._drain_watch_notifications(pending)
            assert pending.get_nowait() is notice, 'a refused notice for a closed local session is retried'
            monkeypatch.setattr(session_automation, 'admit_process_notice', admit)
            pending.put({k: v for k, v in notice.items() if not k.startswith('_watch_')})
            await runner._drain_watch_notifications(pending)

        assert pending.empty()
        assert not [r for r in caplog.records if 'permanently gone' in r.getMessage()]
        await authority._drain(ref)
        assert [m for m in _notes(authority, target) if 'RESULT_proc_watch' in str(m['content'])]
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


@pytest.mark.asyncio
async def test_durable_caller_still_gets_the_refusal_for_a_gone_pin(monkeypatch, caplog):
    """Completion and delegation callers refund their claim on ``WakeNotAccepted`` after their own
    pre-flight; the gone-pin drop is the watch drain's alone and never swallows that refusal."""
    from gateway.wake import WakeNotAccepted
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        retired = entry.session_id
        runner.session_store.reset_session(entry.session_key)

        with caplog.at_level(logging.WARNING, logger='gateway.run'):
            with pytest.raises(WakeNotAccepted, match='admission_conflict'):
                await runner._inject_watch_notification(
                    'notice', _notice(entry, retired), raise_not_accepted=True)

        assert not [r for r in caplog.records if 'permanently gone' in r.getMessage()]
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_refused_notice_pinned_to_a_compression_ancestor_is_retried(monkeypatch, caplog):
    """A compression ancestor is not gone: whatever refused its notice, the notice stays queued."""
    from gateway import session_automation
    from hermes_state_runtime import RuntimeStoreError
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        parent, child = entry.session_id, entry.session_id + '-compressed'
        runner.session_store._db_for_key(entry.session_key).publish_compression_child(
            parent_session_id=parent, child_session_id=child, source='discord',
            require_compression_lease=False, model='offline', model_config={},
            system_prompt='offline', messages=[{'role': 'user', 'content': 'retained'}],
        )
        assert runner.session_store.advance_compression_session(entry.session_key, parent, child)

        async def refuse(*_args):
            raise RuntimeStoreError('admission_conflict')
        monkeypatch.setattr(session_automation, 'admit_process_notice', refuse)
        notice = _notice(entry, parent)
        pending = queue.Queue()
        pending.put(notice)

        with caplog.at_level(logging.WARNING, logger='gateway.run'):
            await runner._drain_watch_notifications(pending)

        assert pending.get_nowait() is notice
        assert not [r for r in caplog.records if 'permanently gone' in r.getMessage()]
    finally:
        authority.db.close()


async def _foreign_pin(runner):
    """A live session of another chat: the authority refuses the notice, but nothing proves the pin
    gone, so the refusal stays retryable."""
    from gateway.config import Platform
    adapter = runner.adapters[Platform.DISCORD]
    other = adapter.build_source(chat_id='43', chat_type='dm', user_id='42')
    return runner.session_store.get_or_create_session(other).session_id


@pytest.mark.asyncio
async def test_unproven_refusal_requeues_with_backoff_until_the_cap(monkeypatch, caplog):
    from gateway import run_notifications
    from gateway.run_notifications import _MAX_COMPLETION_INJECTION_ATTEMPTS
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        pending = queue.Queue()
        pending.put(_notice(entry, await _foreign_pin(runner)))
        clock = install_module_clock(monkeypatch, run_notifications)

        with caplog.at_level(logging.WARNING, logger='gateway.run'):
            await runner._drain_watch_notifications(pending)
            assert not pending.empty(), 'an unproven refusal is retried'
            for _ in range(200):
                if pending.empty():
                    break
                clock.now += 60
                await runner._drain_watch_notifications(pending)

        messages = [r.getMessage() for r in caplog.records]
        assert pending.empty()
        assert sum('not admitted' in m for m in messages) == _MAX_COMPLETION_INJECTION_ATTEMPTS
        assert sum('dropping the queued event' in m for m in messages) == 1
        assert not [m for m in messages if 'permanently gone' in m]
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_unpinned_refusal_requeues(monkeypatch, caplog):
    """A notice that names no pinned session proves nothing gone: its refusal is retried."""
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        notice = _notice(entry, '')
        pending = queue.Queue()
        pending.put(notice)

        with caplog.at_level(logging.WARNING, logger='gateway.run'):
            await runner._drain_watch_notifications(pending)

        assert pending.get_nowait() is notice
        assert notice['_watch_injection_attempts'] == 1
        assert not [r for r in caplog.records if 'permanently gone' in r.getMessage()]
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_unproven_heartbeat_refusal_requeues_the_original_tick(monkeypatch, caplog):
    """A liveness tick keeps its own lifecycle: a refusal the pre-flight cannot prove permanent puts
    the original tick back (a later tick folds from it); only a gone pin drops it."""
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        tick = _notice(entry, await _foreign_pin(runner), 'heartbeat')
        pending = queue.Queue()
        pending.put(tick)

        with caplog.at_level(logging.WARNING, logger='gateway.run'):
            await runner._drain_watch_notifications(pending)

        assert pending.get_nowait() is tick
        assert not [r for r in caplog.records if 'permanently gone' in r.getMessage()]
        assert not _queued_admissions(authority, entry.session_id)
    finally:
        authority.db.close()
