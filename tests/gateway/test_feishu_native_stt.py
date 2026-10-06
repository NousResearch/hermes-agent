"""Feishu native transcripts and ordered fallback; all text fixtures are synthetic."""
import asyncio
import json
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from plugins.platforms.feishu.adapter import FeishuAdapter


def source():
    return SessionSource(platform=Platform.FEISHU, chat_id='synthetic-chat', chat_type='dm', user_id='synthetic-user')


async def inbound(native, download='ok', mentions=None):
    adapter = FeishuAdapter.__new__(FeishuAdapter)
    adapter.config = PlatformConfig()
    adapter.platform = Platform.FEISHU
    adapter._bot_open_id = adapter._bot_user_id = adapter._bot_name = ''
    adapter._download_feishu_message_resource = AsyncMock(
        return_value=('/tmp/synthetic.ogg', 'audio/ogg') if download == 'ok' else ('', ''),
        side_effect=RuntimeError('synthetic download failure') if download == 'exception' else None,
    )
    adapter.get_chat_info = AsyncMock(return_value={})
    adapter._resolve_sender_profile = AsyncMock(return_value=dict(user_id='synthetic-user', user_name='', user_id_alt=None))
    adapter._dispatch_inbound_event = AsyncMock()
    message = SimpleNamespace(message_type='audio', content=json.dumps(dict(file_key='synthetic-key', speech_to_text=native)), message_id='synthetic-id', chat_id='synthetic-chat', mentions=mentions or [])
    await adapter._process_inbound_message(data=None, message=message, sender_id=SimpleNamespace(open_id='synthetic-user'), chat_type='p2p', message_id='synthetic-id')
    return adapter._dispatch_inbound_event.call_args.args[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('download', ['ok', 'empty', 'exception'])
async def test_native_preserves_original_and_real_attachment(download):
    event = await inbound('  合成原生文字\n', download)
    assert event.text == '  合成原生文字\n'
    assert event.message_type is MessageType.VOICE
    assert event.media_urls == (['/tmp/synthetic.ogg'] if download == 'ok' else [])
    assert event.media_text_inlined == ([True] if download == 'ok' else [])
    assert event.voice_parts[0].text == event.text
    assert event.voice_parts[0].index == (0 if download == 'ok' else None)


def runner(enabled=True, echo=True):
    from gateway.run import GatewayRunner
    obj = GatewayRunner.__new__(GatewayRunner)
    obj.config = GatewayConfig(stt_enabled=enabled)
    obj._peek_session_state = lambda key: None
    obj._should_echo_stt_transcripts = lambda: echo
    adapter = SimpleNamespace(send=AsyncMock(), _pending_messages={})
    obj._delivery_adapter_for = lambda src: adapter
    return obj, adapter


async def normal(obj, event):
    return await obj._prepare_inbound_message_text(event=event, source=event.source, history=[], session_key='synthetic-session')


@pytest.mark.asyncio
@pytest.mark.parametrize('native', ['  合成文字\n', None, '', ' \n ', 7, {'text': '合成对象'}])
async def test_normal_uses_native_or_fallback_once(monkeypatch, native):
    from tools import transcription_tools as stt
    obj, adapter = runner()
    provider = Mock(return_value={'success': True, 'transcript': '合成兜底'})
    fallback = Mock(side_effect=AssertionError('unexpected local fallback'))
    monkeypatch.setattr(stt, 'transcribe_audio', provider)
    monkeypatch.setattr(stt, 'transcribe_audio_local_fallback', fallback)
    event = await inbound(native)
    expected = native if isinstance(native, str) and native.strip() else '"合成兜底"'
    assert await normal(obj, event) == expected
    assert await normal(obj, event) == expected
    assert provider.call_count == (0 if expected == native else 1)
    assert fallback.call_count == 0
    assert adapter.send.await_count == (0 if expected == native else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize('native', ['  合成禁用时文字\n', None])
async def test_disabled_keeps_native_in_normal_and_clarify_without_asr(monkeypatch, native):
    from tools import transcription_tools as stt
    obj, adapter = runner(enabled=False)
    provider = Mock(side_effect=AssertionError('STT disabled'))
    fallback = Mock(side_effect=AssertionError('STT disabled'))
    monkeypatch.setattr(stt, 'transcribe_audio', provider)
    monkeypatch.setattr(stt, 'transcribe_audio_local_fallback', fallback)
    event = await inbound(native)
    text = await normal(obj, event)
    assert text == native if native else 'voice message' in text
    assert await obj._prepare_clarify_reply_text(event) == (native or '')
    assert provider.call_count == fallback.call_count == adapter.send.await_count == 0


def mixed_event():
    from gateway.platforms.event import VoicePart
    return MessageEvent(text='合成A\n\n合成正文\n\n合成A', message_type=MessageType.VOICE, source=source(),
        media_urls=['/tmp/native.ogg', '/tmp/repeated.ogg', '/tmp/repeated.ogg', '/tmp/native2.ogg', '/tmp/failed.ogg'],
        media_types=['audio/ogg'] * 5, media_text_inlined=[True, False, False, True, False],
        voice_parts=[VoicePart(kind='audio', text='合成A', index=0), VoicePart(text='合成正文'),
            VoicePart(kind='audio', index=1), VoicePart(kind='audio', index=2),
            VoicePart(kind='audio', text='合成A', index=3), VoicePart(kind='audio', index=4)])


@pytest.mark.asyncio
async def test_mixed_voice_enrichment_keeps_order_and_clarify_excludes_failure(monkeypatch):
    from tools import transcription_tools as stt
    obj, adapter = runner()
    event = mixed_event()
    calls = []
    def recognize(path, *args):
        calls.append(path)
        if path.endswith('failed.ogg'):
            return {'success': True, 'transcript': '  '}
        if calls.count('/tmp/repeated.ogg') == 1 and path.endswith('repeated.ogg'):
            return {'success': False}
        return {'success': True, 'transcript': '合成重复'}
    provider = Mock(side_effect=recognize)
    fallback = Mock(return_value={'success': True, 'transcript': '合成重复'})
    monkeypatch.setattr(stt, 'transcribe_audio', provider)
    monkeypatch.setattr(stt, 'transcribe_audio_local_fallback', fallback)
    text = await obj._enrich_inbound_voice(event, event.source, event.text, event.media_urls)
    assert text.startswith('合成A\n\n合成正文\n\n"合成重复"\n\n"合成重复"\n\n合成A\n\n')
    assert 'inaudible' in text
    assert await obj._prepare_clarify_reply_text(event) == '合成A\n\n合成正文\n\n合成重复\n\n合成重复\n\n合成A'
    assert await normal(obj, event) == text
    assert provider.call_count == 3
    assert fallback.call_count == 1
    assert adapter.send.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['priority', 'busy', 'peek', 'queue-drain'])
async def test_busy_native_without_attachment_delivers_every_part(monkeypatch, entry):
    from gateway.platforms.event import VoicePart
    from tools import transcription_tools as stt
    obj, adapter = runner()
    obj._draining = False
    obj._config = None
    event = MessageEvent(text='合成同文', message_type=MessageType.VOICE, source=source(),
        voice_parts=[VoicePart(kind='audio', text='合成同文'), VoicePart(kind='audio', text='合成同文')])
    spy = Mock(side_effect=AssertionError('native must never ASR'))
    monkeypatch.setattr(stt, 'transcribe_audio', spy)
    monkeypatch.setattr(stt, 'transcribe_audio_local_fallback', spy)
    agent = SimpleNamespace(interrupt=Mock())
    if entry == 'priority':
        await obj._hm_busy_interrupt(event, event.source, agent, 'key')
        delivered = agent.interrupt.call_args.args[0]
    elif entry == 'busy':
        await obj._interrupt_running_agent_for_busy_event(event, adapter, agent)
        delivered = agent.interrupt.call_args.args[0]
    elif entry == 'peek':
        adapter._pending_messages['key'] = event
        await obj._run_agent_fire_pending_interrupt(adapter, agent, event.source, 'key', asyncio.Event(), [None], log_context='Synthetic', log=lambda: None)
        assert adapter._pending_messages['key'] is event
        delivered = agent.interrupt.call_args.args[0]
    else:
        obj._queue_or_replace_pending_event('key', event)
        adapter.get_pending_message = lambda key: adapter._pending_messages.pop(key, None)
        drained, delivered = await obj._run_agent_drain_pending({'final_response': 'synthetic'}, adapter, event.source, 'key')
        assert drained is event
    assert delivered == '合成同文\n\n合成同文'
    assert await normal(obj, event) == delivered
    assert spy.call_count == adapter.send.await_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['native-no-file', 'mixed-ok', 'partial-failure', 'other-media', 'disabled'])
@pytest.mark.parametrize('route', ['busy', 'priority'])
async def test_steer_requires_entire_voice_and_otherwise_queues(monkeypatch, case, route):
    from gateway.platforms.event import VoicePart
    from tools import transcription_tools as stt
    obj, adapter = runner(enabled=case != 'disabled')
    event = await inbound('合成native', 'empty')
    if case != 'native-no-file':
        event.media_urls.append('/tmp/fallback.ogg')
        event.media_types.append('audio/ogg')
        event.media_text_inlined.append(False)
        event.voice_parts.append(VoicePart(kind='audio', index=0))
    if case == 'other-media':
        event.media_urls.append('/tmp/image.png')
        event.media_types.append('image/png')
        event.media_text_inlined.append(False)
    provider = Mock(return_value={'success': case != 'partial-failure', 'transcript': '合成fallback'})
    fallback = Mock(return_value={'success': False, 'error': 'synthetic failure'})
    monkeypatch.setattr(stt, 'transcribe_audio', provider)
    monkeypatch.setattr(stt, 'transcribe_audio_local_fallback', fallback)
    agent = SimpleNamespace(steer=Mock(return_value=True))
    if route == 'busy':
        outcome = await obj._resolve_busy_steer_or_redirect(event, 'key', 'steer', agent)
        if not outcome.steered:
            obj._queue_or_replace_pending_event('key', event)
    else:
        await obj._prepare_busy_steer_text(event)
        obj._hm_busy_steer(event, agent, 'key')
    should_steer = case in {'native-no-file', 'mixed-ok'}
    assert agent.steer.call_count == int(should_steer)
    if should_steer:
        expected = '合成native' + ('\n\n"合成fallback"' if case == 'mixed-ok' else '')
        assert agent.steer.call_args.args[0].endswith(expected)
    else:
        assert adapter._pending_messages['key'] is event
        assert event.media_urls[0] == '/tmp/fallback.ogg'
    assert provider.call_count == int(case not in {'native-no-file', 'disabled'})
    assert fallback.call_count == int(case == 'partial-failure')


@pytest.mark.asyncio
@pytest.mark.parametrize('route', ['pending', 'feishu-batch'])
async def test_merge_after_echo_keeps_parts_indices_and_only_processes_new(monkeypatch, route):
    from gateway.platforms.base import merge_pending_message_event
    from tools import transcription_tools as stt
    obj, adapter = runner()
    provider = Mock(return_value={'success': True, 'transcript': '合成ASR'})
    monkeypatch.setattr(stt, 'transcribe_audio', provider)
    monkeypatch.setattr(stt, 'transcribe_audio_local_fallback', Mock(side_effect=AssertionError('no fallback')))
    first = await inbound(None)
    assert await normal(obj, first) == '"合成ASR"'
    native = await inbound('合成native', 'empty')
    repeated_native = await inbound('合成native', 'empty')
    last = await inbound(None)
    last.media_urls[0] = '/tmp/new.ogg'
    if route == 'pending':
        slot = {'key': first}
        for event in [native, repeated_native, last]:
            merge_pending_message_event(slot, 'key', event)
        merged = slot['key']
    else:
        batcher = FeishuAdapter.__new__(FeishuAdapter)
        batcher._pending_media_batches = {'synthetic-chat:media:voice': first}
        batcher._text_batch_key = lambda event: 'synthetic-chat'
        batcher._schedule_media_batch_flush = lambda key: None
        for event in [native, repeated_native, last]:
            await batcher._enqueue_media_event(event)
        merged = batcher._pending_media_batches['synthetic-chat:media:voice']
    assert await normal(obj, merged) == '"合成ASR"\n\n合成native\n\n合成native\n\n"合成ASR"'
    assert [p.index for p in merged.voice_parts] == [0, None, None, 1]
    assert merged.media_urls == ['/tmp/synthetic.ogg', '/tmp/new.ogg']
    assert provider.call_count == adapter.send.await_count == 2
    assert await normal(obj, merged) == await normal(obj, merged)
    assert provider.call_count == adapter.send.await_count == 2


@pytest.mark.asyncio
async def test_pending_merge_without_media_keeps_duplicate_native_and_legacy_text(monkeypatch):
    from gateway.platforms.base import merge_pending_message_event
    obj, _ = runner()
    first = MessageEvent(text='合成开头', source=source())
    slot = {'key': first}
    for event in [await inbound('合成同文', 'empty'), await inbound('合成同文', 'empty'), MessageEvent(text='合成结尾', source=source())]:
        merge_pending_message_event(slot, 'key', event)
    assert await normal(obj, slot['key']) == '合成开头\n\n合成同文\n\n合成同文\n\n合成结尾'


@pytest.mark.asyncio
@pytest.mark.parametrize('route', ['pending', 'feishu-batch'])
async def test_concurrent_processing_and_append_do_not_duplicate_asr(monkeypatch, route):
    import threading
    from gateway.platforms.base import merge_pending_message_event
    from tools import transcription_tools as stt
    obj, adapter = runner()
    event = await inbound(None)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    def recognize(path, *args):
        if path.endswith('synthetic.ogg'):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5), 'test release missing'
        return {'success': True, 'transcript': '合成ASR'}
    provider = Mock(side_effect=recognize)
    monkeypatch.setattr(stt, 'transcribe_audio', provider)
    monkeypatch.setattr(stt, 'transcribe_audio_local_fallback', Mock(side_effect=AssertionError('no fallback')))
    async def prepare():
        return await obj._transcribe_and_echo_pending_voice(event, adapter, event.source, '', log_context='Synthetic')
    first = asyncio.create_task(prepare())
    second = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        second = asyncio.create_task(prepare())
        await asyncio.sleep(0)
        native = await inbound('合成追加native', 'empty')
        new = await inbound(None)
        new.media_urls[0] = '/tmp/new.ogg'
        if route == 'pending':
            slot = {'key': event}
            merge_pending_message_event(slot, 'key', native)
            merge_pending_message_event(slot, 'key', new)
        else:
            batcher = FeishuAdapter.__new__(FeishuAdapter)
            batcher._pending_media_batches = {'key:media:voice': event}
            batcher._text_batch_key = lambda evt: 'key'
            batcher._schedule_media_batch_flush = lambda key: None
            await batcher._enqueue_media_event(native)
            await batcher._enqueue_media_event(new)
        assert not hasattr(event, '_gateway_pending_stt_text')
    finally:
        release.set()
    results = await asyncio.gather(first, second)
    assert [r[0] for r in results] == ['"合成ASR"\n\n合成追加native\n\n"合成ASR"'] * 2
    assert provider.call_count == adapter.send.await_count == 2
    assert getattr(event, '_gateway_voice_task', None) is None
    assert '_gateway_voice_task' not in asdict(event)


@pytest.mark.asyncio
@pytest.mark.parametrize('route', ['runner-fifo', 'feishu-batch', 'base-wait'])
async def test_incoming_inflight_event_is_preserved_separately(monkeypatch, route):
    import threading
    from tools import transcription_tools as stt
    obj, adapter = runner()
    original = await inbound('合成前一条', 'empty')
    incoming = await inbound(None)
    incoming.source.profile = 'synthetic-b'
    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), threading.Event()
    def recognize(path, *args):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return {'success': True, 'transcript': '合成在途'}
    provider = Mock(side_effect=recognize)
    monkeypatch.setattr(stt, 'transcribe_audio', provider)
    processing = asyncio.create_task(obj._transcribe_pending_audio_event_once(incoming))
    merge_task = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        own_task = incoming._gateway_voice_task
        assert '_gateway_voice_task' not in asdict(incoming)
        if route == 'runner-fifo':
            adapter._pending_messages['key'] = original
            state = SimpleNamespace(conversation=SimpleNamespace(queued_events=[]))
            obj._peek_session_state = obj._session_state = lambda key: state
            obj._hm_merge_pending_for_source(incoming.source, 'key', incoming)
            assert adapter._pending_messages['key'] is original
            assert state.conversation.queued_events == [incoming]
            assert len(original.voice_parts) == 1
        elif route == 'feishu-batch':
            batcher = FeishuAdapter.__new__(FeishuAdapter)
            batcher._pending_media_batches = {'key:media:voice': original}
            batcher._text_batch_key = lambda evt: 'key'
            batcher._schedule_media_batch_flush = lambda key: None
            batcher._handle_message_with_guards = AsyncMock()
            await batcher._enqueue_media_event(incoming)
            assert batcher._handle_message_with_guards.call_args.args[0] is original
            assert batcher._pending_media_batches['key:media:voice'] is incoming
            assert len(original.voice_parts) == 1
        else:
            from gateway.platforms.base import BasePlatformAdapter
            base = FeishuAdapter.__new__(FeishuAdapter)
            base.platform = Platform.FEISHU
            base._canonicalize = lambda event: True
            base._pending_messages = {'key': original}
            base._busy_session_handler = None
            base._is_queue_text_debounce_candidate = lambda event: False
            # No clarify/control prompt exists in this synthetic session.
            merge_task = asyncio.create_task(BasePlatformAdapter._handle_message_while_active(base, incoming, 'key'))
            await asyncio.sleep(0)
            assert not merge_task.done()
            assert len(original.voice_parts) == 1
        assert incoming._gateway_voice_task is own_task
    finally:
        release.set()
        await processing
        if merge_task:
            await merge_task
    if route == 'base-wait':
        assert await obj._transcribe_pending_audio_event_once(original) == ('合成前一条\n\n"合成在途"', ['合成在途'])
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_cancel_one_profile_retains_completed_parts_and_other_event(monkeypatch):
    import threading
    from gateway.platforms.event import VoicePart
    from tools import transcription_tools as stt
    obj, adapter = runner()
    event = await inbound(None)
    event.source.profile = 'synthetic-a'
    event.media_urls.append('/tmp/wait.ogg')
    event.media_types.append('audio/ogg')
    event.voice_parts.append(VoicePart(kind='audio', index=1))
    other = await inbound(None)
    other.media_urls[0] = '/tmp/other.ogg'
    other.source.profile = 'synthetic-b'
    loop = asyncio.get_running_loop()
    entered, release = asyncio.Event(), threading.Event()
    def recognize(path, *args):
        if path.endswith('wait.ogg'):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
        return {'success': True, 'transcript': '合成完成'}
    provider = Mock(side_effect=recognize)
    monkeypatch.setattr(stt, 'transcribe_audio', provider)
    waiting = asyncio.create_task(obj._transcribe_pending_audio_event_once(event))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        other_task = asyncio.create_task(obj._transcribe_pending_audio_event_once(other))
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert getattr(event, '_gateway_voice_task', None) is None
        assert event.voice_parts[0].result == ('合成完成', '"合成完成"')
        assert event.voice_parts[1].result is None
        assert not hasattr(event, '_gateway_pending_stt_text')
        assert await other_task == ('"合成完成"', ['合成完成'])
    finally:
        release.set()
    assert await obj._transcribe_pending_audio_event_once(event) == ('"合成完成"\n\n"合成完成"', ['合成完成', '合成完成'])
    assert [c.args[0] for c in provider.call_args_list].count('/tmp/synthetic.ogg') == 1


@pytest.mark.asyncio
async def test_append_during_echo_delivers_complete_event(monkeypatch):
    from gateway.platforms.base import merge_pending_message_event
    from tools import transcription_tools as stt
    obj, adapter = runner()
    event = await inbound(None)
    added = await inbound('合成发送中追加', 'empty')
    monkeypatch.setattr(stt, 'transcribe_audio', Mock(return_value={'success': True, 'transcript': '合成ASR'}))
    async def send(*args, **kwargs):
        merge_pending_message_event({'key': event}, 'key', added)
    adapter.send.side_effect = send
    text, transcripts = await obj._transcribe_and_echo_pending_voice(event, adapter, event.source, '', log_context='Synthetic')
    assert text == '"合成ASR"\n\n合成发送中追加'
    assert event._gateway_pending_stt_text == text
    assert transcripts == ['合成ASR']
    assert adapter.send.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('voice_first', [True, False])
async def test_merge_with_uploaded_audio_keeps_file_note_without_asr(monkeypatch, voice_first):
    from gateway.platforms.base import merge_pending_message_event
    from tools import transcription_tools as stt
    obj, adapter = runner()
    voice = await inbound('合成native')
    uploaded = MessageEvent(text='合成文件说明', message_type=MessageType.AUDIO, source=source(),
        media_urls=['/tmp/upload.wav'], media_types=['audio/wav'])
    spy = Mock(side_effect=AssertionError('native/uploaded file must not ASR'))
    monkeypatch.setattr(stt, 'transcribe_audio', spy)
    monkeypatch.setattr(stt, 'transcribe_audio_local_fallback', spy)
    first, second = (voice, uploaded) if voice_first else (uploaded, voice)
    slot = {'key': first}
    merge_pending_message_event(slot, 'key', second)
    text = await normal(obj, slot['key'])
    assert '/tmp/upload.wav' in text
    assert '/tmp/synthetic.ogg' not in text
    assert text.endswith('合成native\n\n合成文件说明' if voice_first else '合成文件说明\n\n合成native')
    assert spy.call_count == adapter.send.await_count == 0


@pytest.mark.asyncio
async def test_mention_context_is_not_mistaken_for_native_transcript(monkeypatch):
    from tools import transcription_tools as stt
    obj, adapter = runner()
    mention = SimpleNamespace(key='@_user_1', name='合成被提及人', id=SimpleNamespace(open_id='ou_synthetic'))
    event = await inbound(None, mentions=[mention])
    provider = Mock(return_value={'success': True, 'transcript': '合成兜底'})
    monkeypatch.setattr(stt, 'transcribe_audio', provider)
    text = await normal(obj, event)
    assert text.endswith('\n\n"合成兜底"')
    assert '合成被提及人' in text
    assert provider.call_count == adapter.send.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('entry', ['priority', 'busy', 'peek', 'queue-drain'])
async def test_busy_mixed_voice_retains_failed_audio_and_does_not_repeat_asr(monkeypatch, entry):
    from tools import transcription_tools as stt
    obj, adapter = runner()
    obj._draining = False
    event = mixed_event()
    provider = Mock(side_effect=lambda path, *args: {'success': False, 'error': 'synthetic'} if path.endswith('failed.ogg') else {'success': True, 'transcript': '合成兜底'})
    fallback = Mock(return_value={'success': False, 'error': 'synthetic'})
    monkeypatch.setattr(stt, 'transcribe_audio', provider)
    monkeypatch.setattr(stt, 'transcribe_audio_local_fallback', fallback)
    agent = SimpleNamespace(interrupt=Mock())
    if entry == 'priority':
        await obj._hm_busy_interrupt(event, event.source, agent, 'key')
        text = agent.interrupt.call_args.args[0]
    elif entry == 'busy':
        await obj._interrupt_running_agent_for_busy_event(event, adapter, agent)
        text = agent.interrupt.call_args.args[0]
    elif entry == 'peek':
        adapter._pending_messages['key'] = event
        await obj._run_agent_fire_pending_interrupt(adapter, agent, event.source, 'key', asyncio.Event(), [None], log_context='Synthetic', log=lambda: None)
        text = agent.interrupt.call_args.args[0]
    else:
        obj._queue_or_replace_pending_event('key', event)
        adapter.get_pending_message = lambda key: adapter._pending_messages.pop(key, None)
        drained, text = await obj._run_agent_drain_pending({'final_response': 'synthetic'}, adapter, event.source, 'key')
        assert drained is event
    assert text.startswith('合成A\n\n合成正文\n\n"合成兜底"\n\n"合成兜底"\n\n合成A\n\n')
    assert '/tmp/failed.ogg' in text
    assert await normal(obj, event) == text
    assert provider.call_count == 3
    assert fallback.call_count == 1
    assert adapter.send.await_count == 2


def test_legacy_event_without_voice_parts_keeps_old_merge_contract():
    from gateway.platforms.base import merge_pending_message_event
    legacy = MessageEvent(text='合成旧事件', source=source(), reply_expected=False)
    slot = {'key': legacy}
    merge_pending_message_event(slot, 'key', MessageEvent(text='合成追加', source=source(), reply_expected=True), merge_text=True)
    assert slot['key'] is legacy
    assert legacy.text == '合成旧事件\n合成追加'
    assert legacy.voice_parts is None
    assert legacy.reply_expected is True


@pytest.mark.asyncio
async def test_merge_keeps_late_echo_attempt_on_the_same_part(monkeypatch):
    from gateway.platforms.base import merge_pending_message_event
    from tools import transcription_tools as stt
    obj, adapter = runner()
    first = await inbound('合成native', 'empty')
    incoming = await inbound(None)
    provider = Mock(return_value={'success': True, 'transcript': '合成ASR'})
    monkeypatch.setattr(stt, 'transcribe_audio', provider)
    _, transcripts = await obj._transcribe_pending_audio_event_once(incoming)
    merge_pending_message_event({'key': first}, 'key', incoming)
    # A completion callback can still attempt the original event's echo after merge.
    await obj._echo_pending_stt_transcripts_once(incoming, adapter, incoming.source, transcripts)
    assert await normal(obj, first) == '合成native\n\n"合成ASR"'
    assert adapter.send.await_count == provider.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('native', [None, '合成native'])
async def test_clarify_uses_spoken_words_not_generated_mention_hint(monkeypatch, native):
    from tools import transcription_tools as stt
    obj, adapter = runner()
    mention = SimpleNamespace(key='@_user_1', name='合成被提及人', id=SimpleNamespace(open_id='ou_synthetic'))
    event = await inbound(native, mentions=[mention])
    provider = Mock(return_value={'success': False, 'error': 'synthetic failure'})
    fallback = Mock(return_value={'success': False, 'error': 'synthetic failure'})
    monkeypatch.setattr(stt, 'transcribe_audio', provider)
    monkeypatch.setattr(stt, 'transcribe_audio_local_fallback', fallback)
    assert await obj._prepare_clarify_reply_text(event) == (native or '')
    text = await normal(obj, event)
    assert '合成被提及人' in text
    assert native in text if native else '/tmp/synthetic.ogg' in text
    assert provider.call_count == fallback.call_count == int(native is None)
    assert adapter.send.await_count == 0


def fifo_base():
    from gateway.platforms.base import BasePlatformAdapter
    base = FeishuAdapter.__new__(FeishuAdapter)
    BasePlatformAdapter.__init__(base, PlatformConfig(), Platform.FEISHU)
    return base


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', ['frozen', 'quiescing'])
async def test_voice_commit_gate_busy_handler_freezes_before_false(phase):
    from hermes_constants import get_hermes_home
    base = fifo_base()
    a, b = await inbound('合成A', 'empty'), await inbound('合成B', 'empty')
    key = base._event_session_key(b)
    base._pending_messages[key] = a
    base._active_sessions[key] = asyncio.Event()
    entered, release = asyncio.Event(), asyncio.Event()
    async def busy(event, session_key):
        entered.set()
        await release.wait()
        return False
    base.set_busy_session_handler(busy)
    caller = asyncio.create_task(base._handle_message_while_active(b, key))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        if phase == 'frozen':
            await base.cancel_background_tasks()
            assert base._voice_merge_frozen
        else:
            base._voice_merge_quiescing = True
        base._active_sessions.pop(key, None)
        release.set()
        await asyncio.wait_for(caller, 3)
        assert b._gateway_accepted == (phase == 'quiescing')
        if phase == 'frozen':
            assert not base._pending_messages
        else:
            assert base._pending_messages[key] is a
            assert [event for _, event in base._voice_merge_events.values()] == [b]
        await base.cancel_background_tasks()
        files = list((get_hermes_home() / 'pending_messages').glob('*.json'))
        assert len(files) == 1
        expected = '合成A' if phase == 'frozen' else '合成A\n\n合成B'
        assert json.loads(files[0].read_text())['data'] == {'text': expected}
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('kind,phase', [
    ('voice', 'frozen'), ('text-voice', 'frozen'), ('legacy-in-voice', 'frozen'),
    ('voice', 'quiescing'), ('text-voice', 'quiescing'), ('legacy-in-voice', 'quiescing'),
    ('voice', 'drain'), ('text-voice', 'drain'), ('voice', 'after-asr'),
    ('text-voice', 'open'), ('legacy', 'frozen'), ('photo', 'frozen'),
    ('legacy', 'open'), ('photo', 'open'),
    ('voice', 'no-pending'),
])
async def test_voice_commit_gate_submission_contract(monkeypatch, kind, phase):
    from gateway import shutdown_flush
    base = fifo_base()
    a, b = await inbound('合成A', 'empty'), await inbound('合成B', 'empty')
    if kind in {'legacy', 'photo', 'legacy-in-voice'}:
        b = MessageEvent(text='合成B', message_type=MessageType.PHOTO if kind == 'photo' else MessageType.TEXT, source=source())
        if kind != 'legacy-in-voice':
            a = MessageEvent(text='合成A', message_type=MessageType.TEXT, source=source())
    if kind == 'text-voice':
        b.message_type = MessageType.TEXT
        base._busy_text_mode = 'queue'
    key = base._event_session_key(b)
    if phase != 'no-pending':
        base._pending_messages[key] = a
    writes = []
    original_write = shutdown_flush._write_payload
    def write(directory, payload):
        writes.append(payload['data']['text'])
        return original_write(directory, payload)
    monkeypatch.setattr(shutdown_flush, '_write_payload', write)
    release, entered = asyncio.Event(), asyncio.Event()
    async def asr():
        entered.set()
        await release.wait()
    child = caller = closing = None
    try:
        pending_asr = phase == 'open' and kind in {'legacy', 'photo'}
        if phase in {'drain', 'after-asr', 'no-pending'} or pending_asr:
            child = asyncio.create_task(asr())
            if pending_asr:
                a._gateway_voice_task = child
            else:
                b._gateway_voice_task = child
            await asyncio.wait_for(entered.wait(), 3)
        if phase in {'drain', 'after-asr'}:
            caller = asyncio.create_task(base._merge_voice_followup_in_order(b, key))
            await asyncio.sleep(0)
            assert len(base._voice_merge_events) == 1
            if phase == 'drain':
                closing = asyncio.create_task(base.cancel_background_tasks())
                await asyncio.sleep(0)
                assert base._voice_merge_quiescing and not base._voice_merge_frozen
            else:
                base._voice_merge_frozen = True
            release.set()
            await asyncio.wait_for(caller, 3)
            if phase == 'after-asr':
                assert base._pending_messages[key] is a
                assert not b._gateway_accepted
                assert len(base._voice_merge_events) == 1
            else:
                await asyncio.wait_for(closing, 3)
                assert b._gateway_accepted
                assert not base._voice_merge_events and not base._pending_messages
        else:
            base._voice_merge_quiescing = phase == 'quiescing'
            base._voice_merge_frozen = phase == 'frozen'
            if phase == 'no-pending' or pending_asr:
                await asyncio.wait_for(base._handle_message_while_active(b, key), 2)
                assert not child.done()
                if phase == 'no-pending':
                    assert base._pending_messages[key] is b
            else:
                await base._queue_active_followup(b, key)
            if phase == 'quiescing':
                await base._merge_voice_followup_in_order(b, key)
                assert base._pending_messages[key] is a
                assert a.text == '合成A'
                assert [event for _, event in base._voice_merge_events.values()] == [b]
                assert not base._voice_merge_lanes
            refused = phase == 'frozen' and kind not in {'legacy', 'photo'}
            assert b._gateway_accepted == (not refused)
            if refused:
                assert base._pending_messages[key] is a
                assert not base._text_debounce_store()
            if phase == 'open' and not pending_asr:
                assert base._pending_messages[key].text == '合成A\n\n合成B'
                assert not base._text_debounce_store()
        await base.cancel_background_tasks()
        saved = list(writes)
        await base.cancel_background_tasks()
        assert writes == saved and len(writes) == 1
        if phase in {'quiescing', 'drain', 'open', 'after-asr'} and not pending_asr:
            assert writes == ['合成A\n\n合成B']
        if phase == 'frozen' and kind not in {'legacy', 'photo'}:
            assert writes == ['合成A']
        assert not base._pending_messages and not base._voice_merge_events
    finally:
        release.set()
        await asyncio.gather(*(t for t in [child, caller, closing] if t is not None), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['ordered', 'cancel-b', 'cancel-c', 'cancel-bc', 'asr-error'])
async def test_voice_merge_fifo_keeps_three_events(monkeypatch, case):
    import threading
    from tools import transcription_tools as stt
    obj, _ = runner()
    base = fifo_base()
    a, b, c = await inbound('合成A', 'empty'), await inbound(None), await inbound('合成C', 'empty')
    key = base._event_session_key(b)
    base._pending_messages[key] = a
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    def recognize(path, *args):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), 'test release missing'
        if case == 'asr-error':
            raise RuntimeError('synthetic ASR error')
        return {'success': True, 'transcript': '合成B'}
    provider = Mock(side_effect=recognize)
    monkeypatch.setattr(stt, 'transcribe_audio', provider)
    processing = asyncio.create_task(obj._transcribe_pending_audio_event_once(b))
    callers = []
    try:
        await asyncio.wait_for(entered.wait(), 3)
        callers.append(asyncio.create_task(base._handle_message_while_active(b, key)))
        await asyncio.sleep(0)
        callers.append(asyncio.create_task(base._handle_message_while_active(c, key)))
        await asyncio.sleep(0)
        if case in {'cancel-b', 'cancel-bc'}:
            callers[0].cancel()
        if case in {'cancel-c', 'cancel-bc'}:
            callers[1].cancel()
        await asyncio.sleep(0)
    finally:
        release.set()
        await processing
        await asyncio.gather(*callers, return_exceptions=True)
        # Detached admitted work must also complete when both callers were cancelled.
        admitted = getattr(base, '_voice_merge_lanes', {}).get(key, (None, set()))[1]
        await asyncio.wait_for(asyncio.gather(*admitted), 3)
    rendered, _ = await obj._transcribe_pending_audio_event_once(a)
    expected_b = b.voice_parts[0].result[1]
    assert rendered == '合成A\n\n' + expected_b + '\n\n合成C'
    assert a.voice_parts[1] is b.voice_parts[0]
    assert b._gateway_accepted and c._gateway_accepted
    assert provider.call_count == 1
    assert not getattr(base, '_voice_merge_lanes', {})


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['session', 'profile', 'adapter'])
async def test_voice_merge_fifo_isolates_lanes(monkeypatch, boundary):
    import threading
    from tools import transcription_tools as stt
    obj, _ = runner()
    base = fifo_base()
    a, b = await inbound('合成A', 'empty'), await inbound(None)
    c = await inbound('合成C', 'empty')
    other = fifo_base() if boundary == 'adapter' else base
    if boundary == 'session':
        c.source.chat_id = 'synthetic-other-chat'
    elif boundary == 'profile':
        c.source.profile = 'synthetic-other-profile'
    key, other_key = base._event_session_key(b), other._event_session_key(c)
    assert (key != other_key) == (boundary != 'adapter')
    base._pending_messages[key] = a
    other._pending_messages.setdefault(other_key, await inbound('合成另一A', 'empty'))
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    def recognize(path, *args):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), 'test release missing'
        return {'success': True, 'transcript': '合成B'}
    provider = Mock(side_effect=recognize)
    monkeypatch.setattr(stt, 'transcribe_audio', provider)
    processing = asyncio.create_task(obj._transcribe_pending_audio_event_once(b))
    caller = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        caller = asyncio.create_task(base._handle_message_while_active(b, key))
        await asyncio.sleep(0)
        await asyncio.wait_for(other._handle_message_while_active(c, other_key), 3)
        assert not release.is_set()
        assert other._pending_messages[other_key].voice_parts[-1] is c.voice_parts[0]
        assert len(a.voice_parts) == 1
    finally:
        release.set()
        await processing
        if caller:
            await caller
    assert provider.call_count == 1
    assert not getattr(base, '_voice_merge_lanes', {}) and not getattr(other, '_voice_merge_lanes', {})


@pytest.mark.asyncio
async def test_voice_shutdown_preserves_blocked_lane(monkeypatch, tmp_path):
    import threading
    from tools import transcription_tools as stt
    from hermes_constants import get_hermes_home
    obj, _ = runner()
    base = fifo_base()
    a, b, c = await inbound('合成A', 'empty'), await inbound(None), await inbound('合成C', 'empty')
    path = tmp_path / '未识别B.ogg'
    path.write_bytes(b'synthetic audio boundary fixture')
    b.media_urls[0] = str(path)
    key = base._event_session_key(b)
    base._pending_messages[key] = a
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    def recognize(*args):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(10), 'test release missing'
        return {'success': True, 'transcript': '合成B晚结果'}
    monkeypatch.setattr(stt, 'transcribe_audio', Mock(side_effect=recognize))
    processing = asyncio.create_task(obj._transcribe_pending_audio_event_once(b))
    callers, admitted = [], []
    try:
        await asyncio.wait_for(entered.wait(), 3)
        for event in [b, c]:
            callers.append(asyncio.create_task(base._handle_message_while_active(event, key)))
            await asyncio.sleep(0)
        admitted = list(base._voice_merge_lanes[key][1])
        await asyncio.wait_for(base.cancel_background_tasks(), 5)
        files = list((get_hermes_home() / 'pending_messages').glob('*.json'))
        assert len(files) == 1
        text = json.loads(files[0].read_text())['data']['text']
        assert text == f'合成A\n\n[未识别语音：{path}]\n\n合成C'
        assert all(task.done() for task in admitted)
        assert processing.done()
    finally:
        release.set()
        await asyncio.gather(processing, *callers, *admitted, return_exceptions=True)
    assert not base._pending_messages
    assert not base._voice_merge_lanes


@pytest.mark.asyncio
async def test_voice_merge_fifo_retains_lane_with_waiter(monkeypatch):
    import threading
    from tools import transcription_tools as stt
    obj, _ = runner()
    base = fifo_base()
    a, b, c = await inbound('合成A', 'empty'), await inbound(None), await inbound(None)
    c.media_urls[0] = '/tmp/synthetic-c.ogg'
    d = await inbound('合成D', 'empty')
    key = base._event_session_key(b)
    base._pending_messages[key] = a
    entered = {name: asyncio.Event() for name in ['b', 'c']}
    release = {name: threading.Event() for name in ['b', 'c']}
    loop = asyncio.get_running_loop()
    def recognize(path, *args):
        name = 'c' if path.endswith('synthetic-c.ogg') else 'b'
        loop.call_soon_threadsafe(entered[name].set)
        assert release[name].wait(5), 'test release missing'
        return {'success': True, 'transcript': '合成' + name.upper()}
    provider = Mock(side_effect=recognize)
    monkeypatch.setattr(stt, 'transcribe_audio', provider)
    processing = [asyncio.create_task(obj._transcribe_pending_audio_event_once(event)) for event in [b, c]]
    callers = []
    try:
        await asyncio.wait_for(asyncio.gather(*(signal.wait() for signal in entered.values())), 3)
        for event in [b, c]:
            callers.append(asyncio.create_task(base._handle_message_while_active(event, key)))
            await asyncio.sleep(0)
        lane = getattr(base, '_voice_merge_lanes', {}).get(key)
        release['b'].set()
        await callers[0]
        callers[1].cancel()
        await asyncio.gather(callers[1], return_exceptions=True)
        callers.append(asyncio.create_task(base._handle_message_while_active(d, key)))
        await asyncio.sleep(0)
        # The cancelled waiter still owns its place; D must not create a second lane.
        assert getattr(base, '_voice_merge_lanes', {}).get(key) is lane
        assert obj._render_voice_parts(a) == '合成A\n\n"合成B"'
    finally:
        for signal in release.values():
            signal.set()
        await asyncio.gather(*processing)
        await asyncio.gather(*callers, return_exceptions=True)
        admitted = getattr(base, '_voice_merge_lanes', {}).get(key, (None, set()))[1]
        await asyncio.wait_for(asyncio.gather(*admitted), 3)
    assert obj._render_voice_parts(a) == '合成A\n\n"合成B"\n\n"合成C"\n\n合成D'
    assert a.voice_parts[2] is c.voice_parts[0]
    assert provider.call_count == 2
    assert not getattr(base, '_voice_merge_lanes', {})


@pytest.mark.asyncio
async def test_voice_merge_fifo_does_not_wait_when_no_merge_needed(monkeypatch):
    import threading
    from tools import transcription_tools as stt
    obj, _ = runner()
    base = fifo_base()
    b = await inbound(None)
    key = base._event_session_key(b)
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    def recognize(path, *args):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), 'test release missing'
        return {'success': True, 'transcript': '合成B'}
    monkeypatch.setattr(stt, 'transcribe_audio', Mock(side_effect=recognize))
    processing = asyncio.create_task(obj._transcribe_pending_audio_event_once(b))
    caller = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        caller = asyncio.create_task(base._handle_message_while_active(b, key))
        await asyncio.wait_for(asyncio.shield(caller), 2)
        assert base._pending_messages[key] is b
        assert not release.is_set()
    finally:
        release.set()
        await processing
        if caller:
            await caller


@pytest.mark.asyncio
async def test_voice_shutdown_rejects_late_intake():
    base = fifo_base()
    await base.cancel_background_tasks()
    late = await inbound('合成晚到', 'empty')
    key = base._event_session_key(late)
    await base._handle_message_while_active(late, key)
    assert not late._gateway_accepted
    assert not base._pending_messages
    base.set_message_handler(AsyncMock())
    await base.handle_message(late)
    assert not late._gateway_accepted
    assert not base._background_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['drain', 'asr-error', 'caller-cancel', 'deadline-cancel'])
async def test_voice_shutdown_recovers_ordered_history(monkeypatch, tmp_path, case):
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB
    from gateway.shutdown_flush import recover_pending_to_db
    from tools import transcription_tools as stt
    obj, _ = runner()
    base = fifo_base()
    base.gateway_runner = obj
    monkeypatch.setattr(obj, '_adapter_disconnect_timeout_secs', lambda: 0.3)
    a, b, c = await inbound('合成A', 'empty'), await inbound(None), await inbound('合成C', 'empty')
    path = tmp_path / '恢复B.ogg'
    path.write_bytes(b'synthetic fixture')
    b.media_urls[0] = str(path)
    key = base._event_session_key(b)
    base._pending_messages[key] = a
    entered, release = asyncio.Event(), asyncio.Event()
    async def recognize(text, paths):
        entered.set()
        await release.wait()
        if case == 'asr-error':
            raise RuntimeError('synthetic ASR exception')
        return '合成识别B', ['合成识别B']
    # Only the ASR boundary is controlled; the Part processor, shutdown and recovery are real.
    monkeypatch.setattr(obj, '_enrich_message_with_transcription', recognize)
    processing = asyncio.create_task(obj._transcribe_pending_audio_event_once(b))
    callers, admitted = [], []
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        for event in [b, c]:
            callers.append(asyncio.create_task(base._handle_message_while_active(event, key)))
            await asyncio.sleep(0)
        admitted = list(base._voice_merge_lanes[key][1])
        if case == 'caller-cancel':
            for caller in callers:
                caller.cancel()
            await asyncio.gather(*callers, return_exceptions=True)
        closing = asyncio.create_task(base.cancel_background_tasks())
        await asyncio.sleep(0)
        if case == 'deadline-cancel':
            # The production outer wrapper detaches and cancels its cleanup task on deadline.
            assert not await obj._wait_or_detach(closing, 0.01)
        else:
            release.set()
        await asyncio.gather(closing, return_exceptions=True)
        files = list((get_hermes_home() / 'pending_messages').glob('*.json'))
        assert len(files) == 1
        expected_b = '合成识别B' if case in {'drain', 'caller-cancel'} else f'[未识别语音：{path}]'
        expected = f'合成A\n\n{expected_b}\n\n合成C'
        assert json.loads(files[0].read_text())['data'] == {'text': expected}
        db = SessionDB(tmp_path / 'recovery.db')
        try:
            db.create_session('shutdown-history', source='feishu')
            assert recover_pending_to_db(db, session_resolver=lambda *args, **kw: ('shutdown-history', db)) == 1
            assert [m['content'] for m in db.get_messages('shutdown-history')] == [expected]
            assert recover_pending_to_db(db) == 0
        finally:
            db.close()
        await base.cancel_background_tasks()
        assert not list((get_hermes_home() / 'pending_messages').glob('*.json'))
        assert all(t.done() for t in admitted)
        assert processing.done()
        assert not base._pending_messages and not base._voice_merge_lanes
    finally:
        release.set()
        await asyncio.gather(processing, *callers, *admitted, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)


@pytest.mark.asyncio
async def test_voice_shutdown_retains_content_until_each_save_succeeds(monkeypatch, tmp_path, caplog):
    from hermes_constants import get_hermes_home
    from gateway import shutdown_flush
    base = fifo_base()
    a = await inbound('合成A', 'empty')
    b = await inbound('合成另一会话', 'empty')
    b.source.profile = 'synthetic-other-profile'
    key, other_key = base._event_session_key(a), base._event_session_key(b)
    base._pending_messages = {key: a, other_key: b}
    write = shutdown_flush._write_payload
    def failing_write(directory, payload):
        if payload['session_key'] == key:
            raise OSError('synthetic disk failure')
        return write(directory, payload)
    monkeypatch.setattr(shutdown_flush, '_write_payload', failing_write)
    await base.cancel_background_tasks()
    files = list((get_hermes_home() / 'pending_messages').glob('*.json'))
    assert len(files) == 1 and json.loads(files[0].read_text())['session_key'] == other_key
    assert key in base._pending_messages and other_key not in base._pending_messages
    assert 'retaining recovery content' in caplog.text
    # A later result cannot overwrite the frozen recovery copy on retry.
    a.voice_parts[0].text = '合成晚改动'
    await base.cancel_background_tasks()
    assert len(list((get_hermes_home() / 'pending_messages').glob('*.json'))) == 1
    monkeypatch.setattr(shutdown_flush, '_write_payload', write)
    await base.cancel_background_tasks()
    await base.cancel_background_tasks()
    files = list((get_hermes_home() / 'pending_messages').glob('*.json'))
    assert len(files) == 2
    texts = {json.loads(p.read_text())['data']['text'] for p in files}
    assert texts == {'合成A', '合成另一会话'}
    assert not base._pending_messages
    assert not base._voice_shutdown_contents


@pytest.mark.asyncio
async def test_voice_shutdown_isolates_shield_and_keeps_quiescing_arrival(monkeypatch, caplog):
    from hermes_constants import get_hermes_home
    obj, _ = runner()
    base = fifo_base()
    base.gateway_runner = obj
    monkeypatch.setattr(obj, '_adapter_disconnect_timeout_secs', lambda: 0.1)
    a, b, c = await inbound('合成A', 'empty'), await inbound(None), await inbound('合成C', 'empty')
    other = await inbound(None)
    other.source.profile = 'synthetic-other-profile'
    key = base._event_session_key(b)
    base._pending_messages[key] = a
    entered, isolated, release, other_release = (asyncio.Event() for _ in range(4))
    async def recognize(text, paths):
        if asyncio.current_task() is getattr(other, '_gateway_voice_task', None):
            await other_release.wait()
            return '合成另一识别', ['合成另一识别']
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # Model a cancellation-resistant await; a real CPU worker also cannot be stopped.
            isolated.set()
            await release.wait()
        return '合成晚识别B', ['合成晚识别B']
    monkeypatch.setattr(obj, '_enrich_message_with_transcription', recognize)
    processing = asyncio.create_task(obj._transcribe_pending_audio_event_once(b))
    unrelated = asyncio.create_task(obj._transcribe_pending_audio_event_once(other))
    caller, closing, child = None, None, None
    try:
        await asyncio.wait_for(entered.wait(), 3)
        child = b._gateway_voice_task
        caller = asyncio.create_task(base._handle_message_while_active(b, key))
        await asyncio.sleep(0)
        closing = asyncio.create_task(base.cancel_background_tasks())
        await asyncio.sleep(0)
        # Admission while quiescing retains C without creating a new lane task.
        await base._handle_message_while_active(c, key)
        assert c._gateway_accepted
        await closing
        assert isolated.is_set() and not child.done()
        assert not unrelated.done() and not other._gateway_voice_task.cancelling()
        files = list((get_hermes_home() / 'pending_messages').glob('*.json'))
        assert len(files) == 1
        assert json.loads(files[0].read_text())['data']['text'] == '合成A\n\n[未识别语音：/tmp/synthetic.ogg]\n\n合成C'
        assert 'Isolated' in caplog.text
        release.set()
        await processing
        await asyncio.gather(caller, return_exceptions=True)
        await base.cancel_background_tasks()
        assert len(list((get_hermes_home() / 'pending_messages').glob('*.json'))) == 1
        assert not base._pending_messages and not base._voice_merge_lanes
    finally:
        release.set()
        other_release.set()
        await asyncio.gather(processing, unrelated, *(t for t in [caller, closing, child] if t is not None), return_exceptions=True)


@pytest.mark.asyncio
async def test_routed_voice_rehomes_attachment_before_recognition(monkeypatch, tmp_path):
    import hermes_constants
    from gateway.platforms.event import VoicePart
    from tools import transcription_tools as stt

    launch = tmp_path / 'launch'
    routed = tmp_path / 'routed'
    old = launch / 'cache' / 'audio' / 'voice.ogg'
    old.parent.mkdir(parents=True)
    old.write_bytes(b'synthetic transport audio')
    new = routed / 'cache' / 'audio' / 'voice.ogg'
    monkeypatch.setattr(hermes_constants, 'get_hermes_home', lambda: routed)
    monkeypatch.setattr(hermes_constants, 'get_routing_process_hermes_home', lambda: launch)
    obj, adapter = runner()
    event = MessageEvent(text='', message_type=MessageType.VOICE, source=source(),
        media_urls=[str(old)], media_types=['audio/ogg'],
        voice_parts=[VoicePart(kind='audio', index=0)])
    recognized = []
    def recognize(path, *args):
        recognized.append(path)
        assert path == str(new)
        assert new.read_bytes() == b'synthetic transport audio'
        return {'success': True, 'transcript': '合成路由语音'}
    monkeypatch.setattr(stt, 'transcribe_audio', recognize)
    assert await normal(obj, event) == '"合成路由语音"'
    assert recognized == [str(new)]
    assert event.media_urls == [str(new)]
    assert not old.exists()
