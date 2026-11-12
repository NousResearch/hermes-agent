"""Voice-origin async completion must retain voice intent without changing the wake event type."""
import asyncio
import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_envelope import restore_native
from hermes_state_runtime import claim_session_input, list_session_admissions, settle_session_input
from tests.gateway.test_automation_compression_media import _admit_human, _discord_owner
from tests.gateway.test_completion_admission import pending




async def _telegram_owner(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import initialize_session_authority, SessionAuthority
    from plugins.platforms.telegram.adapter import TelegramAdapter
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', '42')
    runner = GatewayRunner(GatewayConfig())
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    runner.adapters = {Platform.TELEGRAM: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)
    return runner, adapter, authority, adapter.build_source(chat_id='42', chat_type='dm', user_id='42')


@pytest.fixture(autouse=True)
def isolate_voice_mode(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    monkeypatch.setattr(GatewayRunner, '_VOICE_MODE_PATH', tmp_path / 'voice-mode.json')


def test_voice_origin_completion_final_delivery(tmp_path, monkeypatch):
    async def exercise():
        from gateway.run import GatewayRunner
        monkeypatch.setattr(GatewayRunner, '_VOICE_MODE_PATH', tmp_path / 'voice-mode.json')
        runner, _adapter, authority, source = await _telegram_owner(tmp_path, monkeypatch)
        try:
            runner._voice_mode[runner._voice_key_for_source(source)] = 'voice_only'
            receipt = await _admit_human(runner, authority, MessageEvent(
                text='transcribed request', source=source, message_id='voice-request',
                message_type=MessageType.VOICE))
            sid = receipt.ref.session_id
            row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
            from gateway.session_ingress import dispatch_admission_scope
            key = runner.session_store._generate_session_key(source)
            with dispatch_admission_scope(sid, row['admission_id']):
                completion = pending(key, 'voice-completion')
            settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'],
                                 generation=row['generation'], outcome='completed')
            assert await runner._deliver_async_delegation_group([
                dict(completion, parent_session_id=sid)])
            rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
            event = restore_native(rows[-1]['payload'], runner)
            assert event.internal and event.message_type == MessageType.TEXT
            runner._send_voice_reply = AsyncMock()
            await runner._hmwa_deliver_turn_response(
                event, event.source, SimpleNamespace(session_id=sid), key, 1,
                {}, [], 'final result', '', False)
            runner._send_voice_reply.assert_awaited_once_with(event, 'final result')
        finally:
            authority.db.close()
    asyncio.run(exercise())


@pytest.mark.parametrize('kind,mode,extra', [
    (MessageType.TEXT, 'voice_only', False),
    (MessageType.VOICE, 'off', False),
    (MessageType.VOICE, 'voice_only', True),
])
def test_completion_voice_fails_closed_for_text_off_and_ambiguous(tmp_path, monkeypatch, kind, mode, extra):
    async def exercise():
        from gateway.run import GatewayRunner
        root = tmp_path / (kind.value + mode + str(extra))
        root.mkdir()
        monkeypatch.setattr(GatewayRunner, '_VOICE_MODE_PATH', root / 'voice-mode.json')
        runner, _adapter, authority, source = await _discord_owner(root, monkeypatch)
        try:
            runner._voice_mode[runner._voice_key_for_source(source)] = mode
            receipt = await _admit_human(runner, authority, MessageEvent(
                text='request', source=source, message_id='request', message_type=kind))
            sid = receipt.ref.session_id
            row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
            settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'],
                                 generation=row['generation'], outcome='completed')
            if extra:
                await _admit_human(runner, authority, MessageEvent(
                    text='later request', source=source, message_id='later', message_type=MessageType.TEXT))
            key = runner.session_store._generate_session_key(source)
            assert await runner._deliver_async_delegation_group([
                dict(pending(key, 'negative-' + kind.value + mode + str(extra)), parent_session_id=sid)])
            rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
            event = restore_native(rows[-1]['payload'], runner)
            assert not runner._should_send_voice_reply(event, 'final', [])
        finally:
            authority.db.close()
    asyncio.run(exercise())


@pytest.mark.parametrize("kinds,mode,expected", [
    ((MessageType.VOICE, MessageType.TEXT), "voice_only", True),
    ((MessageType.TEXT, MessageType.VOICE), "voice_only", False),
    ((MessageType.VOICE, MessageType.TEXT), "off", False),
])
def test_dispatch_origin_survives_later_turn(tmp_path, monkeypatch, kinds, mode, expected):
    async def exercise():
        from gateway.run import GatewayRunner
        from gateway.session_ingress import dispatch_admission_scope
        from tools import async_delegation as delegation
        monkeypatch.setattr(GatewayRunner, '_VOICE_MODE_PATH', tmp_path / 'voice-mode.json')
        runner, _adapter, authority, source = await _telegram_owner(tmp_path, monkeypatch)
        try:
            runner._voice_mode[runner._voice_key_for_source(source)] = mode
            first = await _admit_human(runner, authority, MessageEvent(text='first', source=source, message_id='first', message_type=kinds[0]))
            sid = first.ref.session_id
            row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
            key = runner.session_store._generate_session_key(source)
            with dispatch_admission_scope(sid, row['admission_id']):
                event = pending(key, 'bound-completion')
            settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'], generation=row['generation'], outcome='completed')
            await _admit_human(runner, authority, MessageEvent(text='second', source=source, message_id='second', message_type=kinds[1]))
            assert await runner._deliver_async_delegation_group([dict(event, parent_session_id=sid)])
            rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
            restored = restore_native(rows[-1]['payload'], runner)
            assert runner._should_send_voice_reply(restored, 'final', []) is expected
        finally:
            authority.db.close()
    asyncio.run(exercise())


def test_batch_voice_requires_every_dispatch_to_be_voice(tmp_path, monkeypatch):
    async def exercise():
        from gateway.run import GatewayRunner
        from gateway.session_ingress import dispatch_admission_scope
        from gateway.session_automation import _delegation_voice_origin, producer_identity
        monkeypatch.setattr(GatewayRunner, '_VOICE_MODE_PATH', tmp_path / 'voice-mode.json')
        runner, _adapter, authority, source = await _discord_owner(tmp_path, monkeypatch)
        try:
            key = runner.session_store._generate_session_key(source)
            ids = []
            for index, kind in enumerate((MessageType.VOICE, MessageType.TEXT)):
                receipt = await _admit_human(runner, authority, MessageEvent(text=str(index), source=source, message_id=str(index), message_type=kind))
                sid = receipt.ref.session_id
                row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
                with dispatch_admission_scope(sid, row['admission_id']):
                    evt = pending(key, 'batch-' + str(index))
                ids.append(producer_identity(runner, evt))
                settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'], generation=row['generation'], outcome='completed')
            assert _delegation_voice_origin(authority, sid, key, ids[0])
            assert not _delegation_voice_origin(authority, sid, key, ids[1])
            assert not _delegation_voice_origin(authority, sid, 'other-route', ids[0])
            assert not _delegation_voice_origin(authority, 'other-owner', key, ids[0])
            runner._voice_mode[runner._voice_key_for_source(source)] = 'voice_only'
            events = [dict(type='async_delegation', session_key=key, delegation_id='batch-' + str(i), summary='done', status='completed', parent_session_id=sid) for i in range(2)]
            assert await runner._deliver_async_delegation_group(events)
            rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
            restored = restore_native(rows[-1]['payload'], runner)
            assert not runner._should_send_voice_reply(restored, 'final', [])
        finally:
            authority.db.close()
    asyncio.run(exercise())


def test_admitted_voice_waits_for_settlement(tmp_path, monkeypatch):
    async def exercise():
        from gateway.run import GatewayRunner
        from gateway.session_ingress import executing_admission
        runner, _adapter, authority, source = await _telegram_owner(tmp_path, monkeypatch)
        try:
            runner._voice_mode[runner._voice_key_for_source(source)] = 'voice_only'
            event = MessageEvent(text='wake', source=source, internal=True)
            event._trusted_voice_origin = True
            runner._send_voice_reply = AsyncMock()
            token = executing_admission.set(True)
            try:
                await runner._hmwa_deliver_turn_response(event, source, SimpleNamespace(session_id='s'), 'key', 1, {}, [], 'answer', '', False)
            finally:
                executing_admission.reset(token)
            runner._send_voice_reply.assert_not_awaited()
            assert event._deferred_voice_reply == 'answer'
        finally:
            authority.db.close()
    asyncio.run(exercise())


def test_deferred_voice_delivers_once_only_after_explicit_settlement():
    from gateway.session_ingress import deliver_deferred_voice
    async def exercise():
        runner = SimpleNamespace(_send_voice_reply=AsyncMock())
        authority = SimpleNamespace(runner=runner)
        event = SimpleNamespace(_deferred_voice_reply='answer', _deferred_voice_receipt='receipt')
        runner._send_voice_reply.assert_not_awaited()
        await deliver_deferred_voice(authority, event, committed_receipt="receipt")
        await deliver_deferred_voice(authority, event, committed_receipt="receipt")
        runner._send_voice_reply.assert_awaited_once_with(event, 'answer')
    asyncio.run(exercise())


def test_restored_discord_voice_uses_verified_guild_or_fails_closed(tmp_path, monkeypatch):
    async def exercise():
        runner, adapter, authority, source = await _discord_owner(tmp_path, monkeypatch)
        try:
            source.chat_type = 'group'
            source.guild_id = '123'
            event = MessageEvent(text='wake', source=source, internal=True)
            event._trusted_voice_origin = True
            adapter.play_in_voice_channel = AsyncMock()
            adapter.send_voice = AsyncMock()
            adapter.is_in_voice_channel = lambda guild: True
            monkeypatch.setattr(runner, '_voice_ack_guild_for', lambda source: 123)
            await runner._deliver_voice_reply(event, ['audio'])
            adapter.play_in_voice_channel.assert_awaited_once_with(123, 'audio')
            monkeypatch.setattr(runner, '_voice_ack_guild_for', lambda source: 999)
            await runner._deliver_voice_reply(event, ['audio'])
            adapter.send_voice.assert_not_awaited()
            assert adapter.play_in_voice_channel.await_count == 1
        finally:
            authority.db.close()
    asyncio.run(exercise())


def test_queued_terminal_origin_overrides_opening_event(tmp_path, monkeypatch):
    async def exercise():
        runner, _adapter, authority, source = await _discord_owner(tmp_path, monkeypatch)
        try:
            runner._voice_mode[runner._voice_key_for_source(source)] = 'voice_only'
            event = MessageEvent(text='voice opener', source=source, message_type=MessageType.VOICE)
            event._terminal_voice_origin = False
            assert not runner._should_send_voice_reply(event, 'text answer', [], already_sent=True)
            event.message_type = MessageType.TEXT
            event._terminal_voice_origin = True
            assert runner._should_send_voice_reply(event, 'voice answer', [], already_sent=True)
        finally:
            authority.db.close()
    asyncio.run(exercise())


def test_disconnected_restored_discord_does_not_upload(tmp_path, monkeypatch):
    async def exercise():
        runner, adapter, authority, source = await _discord_owner(tmp_path, monkeypatch)
        try:
            source.chat_type, source.guild_id = 'group', '123'
            event = MessageEvent(text='wake', source=source, internal=True)
            event._trusted_voice_origin = True
            monkeypatch.setattr(runner, '_voice_ack_guild_for', lambda _: 123)
            adapter.is_in_voice_channel = lambda _: False
            adapter.play_in_voice_channel = AsyncMock()
            adapter.send_voice = AsyncMock()
            await runner._deliver_voice_reply(event, ['audio'])
            adapter.send_voice.assert_not_awaited()
        finally:
            authority.db.close()
    asyncio.run(exercise())


def test_adapter_respects_queued_terminal_text(tmp_path, monkeypatch):
    async def exercise():
        _runner, adapter, authority, source = await _discord_owner(tmp_path, monkeypatch)
        try:
            adapter._should_auto_tts_for_chat = lambda _: True
            event = MessageEvent(text='opener', source=source, message_type=MessageType.VOICE)
            event._terminal_voice_origin = False
            assert not adapter._wants_auto_tts(event, 'session', asyncio.Event(), 'answer', [])
        finally:
            authority.db.close()
    asyncio.run(exercise())


def test_live_waiter_deferred_voice_uses_owner_scope(monkeypatch):
    from contextlib import contextmanager
    from gateway import session_ingress
    state = []
    @contextmanager
    def scope():
        state.append('enter')
        try:
            yield
        finally:
            state.append('exit')
    async def get_scope(authority, home):
        assert home == 'owner-home'
        return scope()
    async def send(event, response):
        assert state == ['enter']
        state.append('send')
    async def exercise():
        monkeypatch.setattr(session_ingress, '_execution_scope', get_scope)
        authority = SimpleNamespace(runner=SimpleNamespace(_send_voice_reply=send))
        event = SimpleNamespace(_deferred_voice_reply='answer', _deferred_voice_receipt='receipt')
        await session_ingress.deliver_deferred_voice(authority, event, 'owner-home', committed_receipt='receipt')
        assert state == ['enter', 'send', 'exit']
    asyncio.run(exercise())


@pytest.mark.parametrize('opening,terminal,mode,already_sent,runner_expected,adapter_expected', [
    (MessageType.VOICE, False, 'voice_only', False, False, False),
    (MessageType.VOICE, False, 'all', False, True, False),
    (MessageType.VOICE, True, 'voice_only', False, False, True),
    (MessageType.TEXT, True, 'voice_only', False, True, False),
    (MessageType.TEXT, True, 'voice_only', True, True, False),
    (MessageType.VOICE, False, 'voice_only', True, False, False),
    (MessageType.VOICE, True, 'voice_only', True, True, False),
])
def test_queued_voice_has_exactly_one_delivery_owner(tmp_path, monkeypatch, opening, terminal, mode, already_sent, runner_expected, adapter_expected):
    async def exercise():
        runner, adapter, authority, source = await _discord_owner(tmp_path, monkeypatch)
        try:
            adapter._should_auto_tts_for_chat = lambda _: True
            runner._voice_mode[runner._voice_key_for_source(source)] = mode
            event = MessageEvent(text='opener', source=source, message_type=opening)
            event._terminal_voice_origin = terminal
            event._terminal_voice_event = MessageEvent(text='queued', source=source, message_type=MessageType.VOICE if terminal else MessageType.TEXT)
            assert runner._should_send_voice_reply(event, 'answer', [], already_sent=already_sent) is runner_expected
            adapter_wants = not already_sent and adapter._wants_auto_tts(event, 'session', asyncio.Event(), 'answer', [])
            assert adapter_wants is adapter_expected
        finally:
            authority.db.close()
    asyncio.run(exercise())


def test_nontelegram_admitted_voice_keeps_streaming(tmp_path, monkeypatch):
    async def exercise():
        from gateway.session_ingress import executing_admission
        from gateway.streaming_tts_consumer import StreamingTTSConsumer
        from tools import tts_tool
        runner, adapter, authority, source = await _discord_owner(tmp_path, monkeypatch)
        try:
            adapter._should_auto_tts_for_chat = lambda _: True
            monkeypatch.setattr(runner, '_delivery_adapter_for', lambda _: adapter)
            started = []
            monkeypatch.setattr(tts_tool, '_load_tts_config', dict)
            monkeypatch.setattr(StreamingTTSConsumer, '__init__', lambda self, **kw: (setattr(self, '_streamer', object()), None)[1])
            monkeypatch.setattr(StreamingTTSConsumer, 'start', lambda self: started.append(True))
            token = executing_admission.set(True)
            try:
                runner._run_agent_start_streaming_tts(source, MessageType.VOICE, None, [None])
            finally:
                executing_admission.reset(token)
            assert started == [True]
        finally:
            authority.db.close()
    asyncio.run(exercise())


def test_recovered_voice_never_speaks_stale_answer():
    from gateway.session_ingress import deliver_deferred_voice
    async def exercise():
        runner = SimpleNamespace(_send_voice_reply=AsyncMock())
        event = SimpleNamespace(_deferred_voice_reply='stale answer', _deferred_voice_committed_response='stale answer')
        await deliver_deferred_voice(SimpleNamespace(runner=runner), event, committed_receipt='different')
        runner._send_voice_reply.assert_not_awaited()
    asyncio.run(exercise())


def test_real_dispatch_propagates_voice_origin_to_telegram_wake(tmp_path, monkeypatch):
    async def exercise():
        from gateway.session_ingress import dispatch_admission_scope
        from tools import async_delegation as delegation
        import threading
        runner, _adapter, authority, source = await _telegram_owner(tmp_path, monkeypatch)
        finished = threading.Event()
        monkeypatch.setattr(delegation, '_capture_routing_origin', dict)
        try:
            runner._voice_mode[runner._voice_key_for_source(source)] = 'voice_only'
            receipt = await _admit_human(runner, authority, MessageEvent(
                text='voice request', source=source, message_id='dispatch', message_type=MessageType.VOICE))
            sid = receipt.ref.session_id
            row = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
            key = runner.session_store._generate_session_key(source)
            def work():
                finished.set()
                return {'status': 'completed', 'summary': 'done'}
            with dispatch_admission_scope(sid, row['admission_id']):
                handle = delegation.dispatch_async_delegation(goal='test', context=None, toolsets=None,
                    role='leaf', model=None, session_key=key, parent_session_id=sid, runner=work)
            assert handle['status'] == 'dispatched'
            assert await asyncio.to_thread(finished.wait, 5)
            for _ in range(100):
                with authority.db._read_ctx() as conn:
                    record = conn.execute('SELECT state,event_json FROM async_delegations WHERE delegation_id=?',
                                          (handle['delegation_id'],)).fetchone()
                if record and record[0] == 'completed':
                    break
                await asyncio.sleep(.02)
            import json
            assert record[0] == 'completed'
            settle_session_input(authority.db, epoch=authority.epoch, admission_id=row['admission_id'],
                                 generation=row['generation'], outcome='completed')
            evt = json.loads(record[1])
            assert await runner._deliver_async_delegation_group([evt])
            rows = list_session_admissions(authority.db, session_id=sid, pending_only=False)
            event = restore_native(rows[-1]['payload'], runner)
            assert runner._should_send_voice_reply(event, 'final', [])
        finally:
            authority.db.close()
    asyncio.run(exercise())


@pytest.mark.parametrize("committed_text,expected", [("answer", 1), ("changed", 0)])
def test_streamed_waiter_voice_binds_committed_result(monkeypatch, committed_text, expected):
    from threading import RLock
    from gateway import session_settlement_recovery as recovery, session_ingress as ingress, session_results
    async def exercise():
        event = SimpleNamespace(_deferred_voice_reply='answer', _deferred_voice_receipt='receipt')
        runner = SimpleNamespace(_send_voice_reply=AsyncMock())
        authority = SimpleNamespace(runner=runner, db=object(), pending_results={'a': {'result': {'final_response': 'answer'}}},
            pending_stops={}, pending_stop_causes={}, pending_steers={}, adopted={}, deferred_resets={},
            pending_deliveries={}, native_waiters=set(), sessions={}, waiter_replies={'a': (None,event,None,None,None)},
            _run_deferred_reset=AsyncMock(), _resolve_settled_waiter=AsyncMock())
        live = SimpleNamespace(event_stream=SimpleNamespace(lock=RLock(), execution={}), route='route')
        ref = SimpleNamespace(session_id='s')
        async def settle(*args):
            args[-1].update(settled={'outcome':'completed'}, response=None)
        monkeypatch.setattr(recovery, 'settle_with_retry', settle)
        monkeypatch.setattr(recovery, 'release_media_after_delivery', lambda *a: None)
        monkeypatch.setattr('gateway.session_queued_controls.publish_transfers', lambda *a: None)
        monkeypatch.setattr(ingress, 'deliver_settled', AsyncMock())
        monkeypatch.setattr(ingress, 'fire_post_delivery', AsyncMock())
        monkeypatch.setattr(ingress, '_execution_scope', AsyncMock(return_value=__import__('contextlib').nullcontext()))
        monkeypatch.setattr(session_results, 'admission_result', lambda *a: {'result': {'final_response': committed_text}, '_voice_receipt': 'receipt' if committed_text == 'answer' else 'other'})
        await recovery.settle_and_deliver(authority, live, ref, {'admission_id':'a','generation':1}, None, 'completed')
        assert runner._send_voice_reply.await_count == expected
        if expected:
            runner._send_voice_reply.assert_awaited_once_with(event, 'answer')
    asyncio.run(exercise())


@pytest.mark.parametrize("committed_text,expected", [("answer", 1), ("changed", 0)])
def test_late_native_waiter_gets_deferred_voice_once(monkeypatch, committed_text, expected):
    from gateway import session_ingress as ingress, session_results
    async def exercise():
        event = SimpleNamespace(_deferred_voice_reply='answer', _deferred_voice_receipt='receipt')
        runner = SimpleNamespace(_send_voice_reply=AsyncMock())
        authority = SimpleNamespace(runner=runner, db=object(), pending_deliveries={'a': (None,event,None,None,None)}, native_waiters={'a'})
        monkeypatch.setattr(session_results, 'admission_result', lambda *a: {'result': {'final_response': committed_text}, '_voice_receipt': 'receipt' if committed_text == 'answer' else 'other'})
        monkeypatch.setattr(ingress, '_execution_scope', AsyncMock(return_value=__import__('contextlib').nullcontext()))
        monkeypatch.setattr(ingress, 'close_settled_turn', AsyncMock())
        await ingress.deliver_settled(authority, 'a')
        await ingress.deliver_settled(authority, 'a')
        assert runner._send_voice_reply.await_count == expected
        if expected:
            runner._send_voice_reply.assert_awaited_once_with(event, 'answer')
    asyncio.run(exercise())


def test_eventless_followup_retains_voice_origin():
    from tests.gateway.test_queued_followup_processing_hooks import HookRecordingAdapter, _make_runner, _source, SESSION_KEY
    async def exercise():
        adapter = HookRecordingAdapter()
        runner = _make_runner(adapter)
        runner._delivery_adapter_for = lambda source: None
        runner._refresh_agent_cache_message_count = AsyncMock()
        runner._run_agent = AsyncMock(return_value={'final_response':'answer', 'messages':[]})
        ctx = SimpleNamespace(source=_source(), session_id='s', session_key=SESSION_KEY, run_generation=None,
            _interrupt_depth=0, history=[], _status_thread_metadata=None, channel_prompt=None, context_prompt=None)
        result = await runner._run_agent_queued_followup(ctx, adapter, 'steer', None, {}, {'interrupted':True}, None)
        assert 'queued_terminal_voice_origin' not in result
    asyncio.run(exercise())


def test_deferred_display_is_bound_to_raw_committed_result(tmp_path, monkeypatch):
    async def exercise():
        from gateway.session_ingress import executing_admission, deliver_deferred_voice
        from gateway import session_results
        runner, _adapter, authority, source = await _telegram_owner(tmp_path, monkeypatch)
        try:
            runner._voice_mode[runner._voice_key_for_source(source)] = 'voice_only'
            event = MessageEvent(text='wake', source=source, internal=True)
            event._trusted_voice_origin = True
            runner._send_voice_reply = AsyncMock()
            monkeypatch.setattr(session_results, '_redacted', lambda value: value.replace('secret', '[REDACTED]') if isinstance(value, str) else value)
            token = executing_admission.set(True)
            try:
                await runner._hmwa_deliver_turn_response(event, source, SimpleNamespace(session_id='s'), 'key', 1, {'final_response': 'raw secret'}, [], 'reasoning\nclean secret', '', False)
            finally:
                executing_admission.reset(token)
            receipt = event._deferred_voice_proof[2]
            event._deferred_voice_receipt = receipt
            await deliver_deferred_voice(authority, event, committed_receipt=receipt)
            runner._send_voice_reply.assert_awaited_once_with(event, 'reasoning\nclean [REDACTED]')
        finally:
            authority.db.close()
    asyncio.run(exercise())


def test_collision_does_not_authorize_stale_voice(tmp_path, monkeypatch):
    from gateway.session_ingress import deliver_deferred_voice
    async def exercise():
        runner = SimpleNamespace(_send_voice_reply=AsyncMock())
        event = SimpleNamespace(_deferred_voice_reply='old display', _deferred_voice_committed_response='[REDACTED]')
        await deliver_deferred_voice(SimpleNamespace(runner=runner), event, committed_response='[REDACTED]')
        runner._send_voice_reply.assert_not_awaited()
    asyncio.run(exercise())


def test_voice_result_read_failure_does_not_lose_text(monkeypatch):
    from gateway import session_ingress as ingress, session_results
    async def exercise():
        event = SimpleNamespace(_deferred_voice_reply='voice')
        authority = SimpleNamespace(runner=SimpleNamespace(_send_voice_reply=AsyncMock()), db=object(), pending_deliveries={'a': (object(),event,'key','text',None)}, native_waiters=set())
        monkeypatch.setattr(session_results, 'admission_result', lambda *a: (_ for _ in ()).throw(RuntimeError('db failed')))
        monkeypatch.setattr(ingress, '_execution_scope', AsyncMock(return_value=__import__('contextlib').nullcontext()))
        deliver = AsyncMock(return_value=True)
        monkeypatch.setattr(ingress, 'deliver_response', deliver)
        monkeypatch.setattr(ingress, '_release_turn_marker', AsyncMock())
        monkeypatch.setattr(ingress, 'close_producer_turn', AsyncMock())
        await ingress.deliver_settled(authority, 'a')
        deliver.assert_awaited_once()
    asyncio.run(exercise())


def test_distinct_raw_results_with_identical_redaction_cannot_mint_receipt():
    import hashlib, hmac
    from gateway.session_ingress import bind_deferred_voice, deliver_deferred_voice
    async def exercise():
        key = b'private random key'
        event = SimpleNamespace(_deferred_voice_reply='old display', _deferred_voice_proof=(key, hmac.digest(key, b'secret one', hashlib.sha256), 'receipt'))
        pending = {}
        bind_deferred_voice(event, {'final_response': 'secret two'}, pending)
        assert '_voice_receipt' not in pending
        runner = SimpleNamespace(_send_voice_reply=AsyncMock())
        await deliver_deferred_voice(SimpleNamespace(runner=runner), event, committed_receipt=pending.get('_voice_receipt'))
        runner._send_voice_reply.assert_not_awaited()
    asyncio.run(exercise())


def test_waiter_voice_db_failure_still_resolves_and_cleans_up(monkeypatch):
    from threading import RLock
    from gateway import session_settlement_recovery as recovery, session_ingress as ingress, session_results
    async def exercise():
        event = SimpleNamespace(_deferred_voice_reply='voice', _deferred_voice_receipt='receipt')
        authority = SimpleNamespace(runner=SimpleNamespace(_send_voice_reply=AsyncMock()), db=object(), pending_results={'a': {'result': {'final_response': 'answer'}}}, pending_stops={}, pending_stop_causes={}, pending_steers={}, adopted={}, deferred_resets={'route': 'reset'}, pending_deliveries={}, native_waiters=set(), sessions={}, waiter_replies={'a': (None,event,None,None,None)}, _run_deferred_reset=AsyncMock(), _resolve_settled_waiter=AsyncMock())
        live = SimpleNamespace(event_stream=SimpleNamespace(lock=RLock(), execution={}), route='route')
        async def settle(*args):
            args[-1].update(settled={'outcome': 'completed'}, response='answer')
        monkeypatch.setattr(recovery, 'settle_with_retry', settle)
        monkeypatch.setattr(recovery, 'release_media_after_delivery', lambda *a: None)
        monkeypatch.setattr('gateway.session_queued_controls.publish_transfers', lambda *a: None)
        monkeypatch.setattr(ingress, 'deliver_settled', AsyncMock())
        monkeypatch.setattr(ingress, 'fire_post_delivery', AsyncMock())
        monkeypatch.setattr(session_results, 'admission_result', lambda *a: (_ for _ in ()).throw(RuntimeError('db failed')))
        assert await recovery.settle_and_deliver(authority, live, SimpleNamespace(session_id='s'), {'admission_id': 'a', 'generation': 1}, 'answer', 'completed')
        authority._resolve_settled_waiter.assert_awaited_once_with('a', 'answer')
        authority._run_deferred_reset.assert_awaited_once_with('route', 'reset')
        assert not authority.waiter_replies
        authority.runner._send_voice_reply.assert_not_awaited()
    asyncio.run(exercise())


def test_queued_trusted_completion_defers_and_uses_terminal_route(tmp_path, monkeypatch):
    async def exercise():
        from gateway.session_ingress import executing_admission, bind_deferred_voice, deliver_committed_voice
        from gateway.session_results import finish_result, admission_result
        from tests.gateway.test_queued_followup_processing_hooks import HookRecordingAdapter, _make_runner, _source, SESSION_KEY
        runner, _adapter, authority, source = await _telegram_owner(tmp_path, monkeypatch)
        try:
            receipt = await _admit_human(runner, authority, MessageEvent(text="opener", source=source, message_id="open"))
            row = claim_session_input(authority.db, epoch=authority.epoch, session_id=receipt.ref.session_id)
            hook = HookRecordingAdapter()
            queued_runner = _make_runner(hook)
            queued_runner._refresh_agent_cache_message_count = AsyncMock()
            queued_runner._run_agent = AsyncMock(return_value={"final_response": "answer", "messages": []})
            queued_runner._prepare_profile_scoped_inbound_message_text = AsyncMock(return_value="completion")
            terminal = MessageEvent(text="completion", source=source, message_id="queued", internal=True)
            terminal._trusted_voice_origin = True
            ctx = SimpleNamespace(source=_source(), session_id="s", session_key=SESSION_KEY, run_generation=None,
                _interrupt_depth=0, history=[], _status_thread_metadata=None, channel_prompt=None, context_prompt=None)
            merged = await queued_runner._run_agent_queued_followup(ctx, hook, "completion", terminal, {}, {"interrupted": True}, None)
            assert merged["_queued_terminal_voice_event"] is terminal
            opener = MessageEvent(text="opener", source=source, message_type=MessageType.VOICE)
            opener._terminal_voice_origin = merged["queued_terminal_voice_origin"]
            opener._terminal_voice_event = merged.pop("_queued_terminal_voice_event")
            runner._voice_mode[runner._voice_key_for_source(source)] = "voice_only"
            runner._send_voice_reply = AsyncMock()
            token = executing_admission.set(True)
            try:
                await runner._hmwa_deliver_turn_response(opener, source, SimpleNamespace(session_id="s"), "key", 1, merged, [], "answer", "", False)
            finally:
                executing_admission.reset(token)
            runner._send_voice_reply.assert_not_awaited()
            pending_result = {"result": {"final_response": "answer", "messages": []}, "usage": {}}
            bind_deferred_voice(opener, merged, pending_result)
            assert admission_result(authority.db, row["admission_id"]) is None
            await deliver_committed_voice(authority, row["admission_id"], opener)
            runner._send_voice_reply.assert_not_awaited()
            # Recreate the volatile delivery state consumed by a premature check.
            opener._deferred_voice_reply = "answer"
            opener._deferred_voice_event = terminal
            opener._deferred_voice_receipt = pending_result["_voice_receipt"]
            finish_result(authority.db, epoch=authority.epoch, row=row, response="answer", outcome="completed", result=pending_result)
            assert admission_result(authority.db, row["admission_id"])["_voice_receipt"] == pending_result["_voice_receipt"]
            await deliver_committed_voice(authority, row["admission_id"], opener)
            await deliver_committed_voice(authority, row["admission_id"], opener)
            runner._send_voice_reply.assert_awaited_once_with(terminal, "answer")
        finally:
            authority.db.close()
    asyncio.run(exercise())
