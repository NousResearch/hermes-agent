"""Caller-owned outbox delivery, using real adapters and PTB with no network.

Text/wrapper cases port the behavioral coverage of PR #84095; the transport
records Bot API requests rather than mocking adapter fallback methods.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from urllib.parse import parse_qs

import httpx
import pytest
import pytest_asyncio
from telegram import Bot
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut
from telegram.ext import Application
from telegram.request import BaseRequest, HTTPXRequest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.telegram.adapter import TelegramAdapter


class RecordingRequest(BaseRequest):
    def __init__(self):
        self.calls = []
        self.errors = []

    @property
    def read_timeout(self):
        return 30

    async def initialize(self):
        pass

    async def shutdown(self):
        pass

    async def do_request(self, url, method, request_data=None, **kwargs):
        endpoint = url.rsplit('/', 1)[-1]
        self.calls.append((endpoint, request_data.parameters))
        error = self.errors.pop(0) if self.errors else None
        if error is not None:
            raise error
        result = {'message_id': len(self.calls), 'date': 1, 'chat': {'id': 12345, 'type': 'private'}}
        if endpoint == 'sendMediaGroup':
            count = len(request_data.parameters['media'])
            if not 2 <= count <= 10:
                return 400, json.dumps({
                    'ok': False, 'error_code': 400,
                    'description': 'Bad Request: media group must include 2-10 items',
                }).encode()
            result = [dict(result, message_id=i + 1) for i in range(count)]
        return 200, json.dumps({'ok': True, 'result': result}).encode()


@pytest.fixture
def delivery(monkeypatch):
    request = RecordingRequest()
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token='123:test-token', extra={'rich_messages': True}))
    adapter._bot = Bot('123:test-token', request=request)
    sleep = AsyncMock()
    monkeypatch.setattr('plugins.platforms.telegram.adapter.asyncio.sleep', sleep)
    return adapter, request, sleep


class RecordingMessageTransport(httpx.MockTransport):
    """HTTP edge for sendMessage probes, retaining real HTTPXRequest decoding."""

    def __init__(self, errors=()):
        self.calls = []
        self.errors = list(errors)
        super().__init__(self.respond)

    def respond(self, request):
        self.calls.append(request)
        if self.errors:
            raise self.errors.pop(0)
        assert request.url.path.endswith('/sendMessage')
        return httpx.Response(200, json={'ok': True, 'result': {
            'message_id': len(self.calls), 'date': 1, 'chat': {'id': 12345, 'type': 'private'},
        }})


@pytest_asyncio.fixture
async def httpx_adapter():
    requests = []

    def build(transport):
        request = HTTPXRequest(httpx_kwargs={'transport': transport, 'trust_env': False})
        updates = HTTPXRequest(httpx_kwargs={'transport': httpx.MockTransport(
            lambda _: pytest.fail('unexpected getUpdates request')), 'trust_env': False})
        requests.extend((request, updates))
        app = Application.builder().token('123:test-token').request(request).get_updates_request(updates).build()
        adapter = TelegramAdapter(PlatformConfig(enabled=True, token='123:test-token'))
        adapter._app, adapter._bot = app, app.bot
        return adapter

    yield build
    for request in requests:
        await request.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('flag', [True, False, None, 1, 'true'])
async def test_blocked_image_url_does_not_become_successful_text(httpx_adapter, flag):
    # The actual safety policy rejects this hostname before DNS, not a stub.
    wire = RecordingMessageTransport()
    adapter = httpx_adapter(wire)
    url = 'http://metadata.google.internal/private.png'
    result = await adapter.send_image(
        '12345', url, metadata={'single_external_attempt': flag, 'notify': True})
    if flag is True:
        assert not result.success, 'native image refusal must not masquerade as delivered media'
        assert result.message_id is None
        assert result.error and not result.retryable
        assert wire.calls == []
    else:
        assert result.success
        assert [request.url.path.rsplit('/', 1)[-1] for request in wire.calls] == ['sendMessage']
        assert parse_qs(wire.calls[0].content.decode())['text'] == [adapter.format_message(url)]


@pytest.mark.asyncio
@pytest.mark.parametrize('flag', [True, False, None, 1, 'true', 'false'])
@pytest.mark.parametrize('error', [NetworkError('connection lost after write'), RetryAfter(1),
                                   TimedOut('Pool timeout: request was not sent')])
async def test_text_retry_is_explicitly_caller_owned(delivery, flag, error):
    adapter, request, sleep = delivery
    assert adapter.supports_single_external_attempt is True
    request.errors = [error]
    metadata = {'notify': True}
    if flag is not None:
        metadata['single_external_attempt'] = flag
    result = await adapter.send('12345', 'hello', metadata=metadata)
    if flag is True:
        assert not result.success
        assert len(request.calls) == 1
        sleep.assert_not_awaited()
    else:
        assert result.success
        assert len(request.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize('flag', [True, False, None, 1, 'true', 'false'])
@pytest.mark.parametrize('failure', [
    SendResult(False, error='connection reset', retryable=True),
    SendResult(False, error='format rejected', error_kind='bad_format'),
    SendResult(False, error='message_too_long', error_kind='too_long'),
    SendResult(False, error='rate limited', retry_after=1),
])
async def test_shared_wrapper_returns_first_result_unchanged(delivery, failure, flag):
    adapter, _, sleep = delivery
    adapter.send = AsyncMock(return_value=failure)
    result = await adapter._send_with_retry('12345', 'hello', metadata={'single_external_attempt': flag})
    assert result is failure
    if flag is True:
        adapter.send.assert_awaited_once()
        sleep.assert_not_awaited()
    else:
        assert adapter.send.await_count > 1


@pytest.mark.asyncio
@pytest.mark.parametrize(('error', 'changed_field'), [
    (BadRequest("can't parse entities"), 'parse_mode'),
    (BadRequest('Message to be replied not found'), 'reply_parameters'),
])
async def test_definitive_text_rejection_allows_correction(delivery, error, changed_field):
    adapter, request, _ = delivery
    request.errors = [error]
    result = await adapter.send('12345', 'hello _world_', reply_to='999',
                                metadata={'single_external_attempt': True, 'notify': True})
    assert result.success
    assert len(request.calls) == 2
    assert changed_field in request.calls[0][1]
    assert changed_field not in request.calls[1][1]


@pytest.mark.asyncio
@pytest.mark.parametrize(('text', 'accepted'), [
    ('😀' * 2048, True), ('😀' * 2049, False),
    ('a' * 4096, True), ('a' * 4097, False),
])
async def test_multichunk_text_is_rejected_before_delivery(httpx_adapter, text, accepted):
    wire = RecordingMessageTransport()
    adapter = httpx_adapter(wire)
    result = await adapter._send_with_retry(
        '12345', text, metadata={'single_external_attempt': True, 'notify': True})
    assert result.success is accepted
    if accepted:
        assert len(wire.calls) == 1
        assert parse_qs(wire.calls[0].content.decode())['text'] == [text]
    else:
        assert result.error_kind == 'too_long'
        assert wire.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize('error', [
    httpx.ReadError('lost after write'), httpx.WriteError('write outcome uncertain'),
    httpx.ReadTimeout('response lost'), httpx.ConnectError('connect failed'),
])
async def test_fallback_ips_only_retry_connection_establishment(httpx_adapter, error):
    from plugins.platforms.telegram.telegram_network import TelegramFallbackTransport

    ips = ['149.154.166.110', '149.154.167.220']
    transport = TelegramFallbackTransport(ips)
    await transport._primary.aclose()
    transport._primary = RecordingMessageTransport()
    first, second = RecordingMessageTransport([error]), RecordingMessageTransport()
    transport._fallbacks = dict(zip(ips, (first, second)))
    adapter = httpx_adapter(transport)
    result = await adapter._send_with_retry(
        '12345', 'hello', metadata={'single_external_attempt': True, 'notify': True})
    connect_failure = isinstance(error, httpx.ConnectError)
    assert result.success is connect_failure
    assert len(first.calls) == 1
    assert len(second.calls) == int(connect_failure)
    assert transport._primary.calls == []
    for request in first.calls + second.calls:
        assert request.url.path.endswith('/sendMessage')
        assert request.headers['host'] == 'api.telegram.org'


_MEDIA = [
    ('send_image_file', 'image_path', 'photo.png'),
    ('send_document', 'file_path', 'report.txt'),
    ('send_video', 'video_path', 'clip.mp4'),
    ('send_voice', 'audio_path', 'clip.ogg'),
    ('send_voice', 'audio_path', 'clip.mp3'),
    ('send_voice', 'audio_path', 'clip.wav'),
    ('send_image', 'image_url', 'https://example.com/photo.png'),
    ('send_animation', 'animation_url', 'https://example.com/movie.gif'),
    ('send_multiple_images', 'images', None),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(('method', 'argument', 'source'), _MEDIA)
@pytest.mark.parametrize('single', [True, False])
@pytest.mark.parametrize('error', [NetworkError('connection lost after write'),
                                   BadRequest('Message to be replied not found')])
async def test_media_failure_does_not_hide_another_send(
    delivery, tmp_path, monkeypatch, method, argument, source, single, error,
):
    """Ported media/fallback contract, through real PTB and all native siblings."""
    adapter, request, _ = delivery
    monkeypatch.setattr('plugins.platforms.telegram.adapter._probe_voice_duration_seconds', lambda _: None)
    monkeypatch.setattr('plugins.platforms.telegram.adapter._probe_video_geometry', lambda _: {})
    monkeypatch.setattr('tools.url_safety.is_safe_url', lambda _: True)
    downloads = []

    class Download:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            pass

        async def get(self, url):
            downloads.append(url)
            return SimpleNamespace(content=b'photo', raise_for_status=lambda: None)

    monkeypatch.setattr('tools.url_safety.create_ssrf_safe_async_client', lambda **kw: Download())
    if source is None:
        value = [('https://example.com/photo.png', 'caption'), ('https://example.com/other.png', '')]
    elif source.startswith('https:'):
        value = source
    else:
        path = tmp_path / source
        path.write_bytes(b'media')
        value = str(path)
    request.errors = [error]
    result = await getattr(adapter, method)(
        '12345', **{argument: value}, metadata={
            'single_external_attempt': single, 'notify': True,
            'thread_id': '20197', 'telegram_dm_topic_reply_fallback': True,
            'telegram_reply_to_message_id': '462',
        })
    assert result.success is (not single)
    fallback_calls = 3 if method == 'send_multiple_images' and not isinstance(error, BadRequest) else 2
    assert len(request.calls) == (1 if single else fallback_calls)
    if single:
        assert not result.retryable
        assert downloads == []


@pytest.mark.asyncio
@pytest.mark.parametrize('kind', ['text', 'voice', 'rich'])
@pytest.mark.parametrize('error', [NetworkError('unsupported response: cannot parse entities'),
                                   TimedOut('parse response timed out'), BadRequest("can't parse entities")])
async def test_only_definitive_rejections_allow_format_correction(delivery, tmp_path, monkeypatch, kind, error):
    adapter, request, _ = delivery
    request.errors = [error]
    metadata = {'single_external_attempt': True, 'notify': True}
    if kind == 'voice':
        monkeypatch.setattr('plugins.platforms.telegram.adapter._probe_voice_duration_seconds', lambda _: None)
        path = tmp_path / 'voice.ogg'
        path.write_bytes(b'voice')
        result = await adapter.send_voice('12345', str(path), caption='hello _world_', metadata=metadata)
    else:
        content = '| A | B |\n|---|---|\n| a | b |' if kind == 'rich' else 'hello _world_'
        result = await adapter._send_with_retry('12345', content, metadata=metadata)
    definitive = isinstance(error, BadRequest)
    assert request.calls[0][0] == {'text': 'sendMessage', 'rich': 'sendRichMessage', 'voice': 'sendVoice'}[kind]
    assert result.success is definitive
    assert len(request.calls) == (2 if definitive else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize(('urls', 'endpoints'), [
    (['photo.png'] * 11, ['sendMediaGroup', 'sendMediaGroup', 'sendPhoto']),
    (['movie.gif', 'photo.png'], ['sendAnimation', 'sendMediaGroup', 'sendPhoto']),
    (['movie.gif'] * 2, ['sendAnimation'] * 2),
    (['photo.png'] * 10, ['sendMediaGroup']),
    (['photo.png'] * 2, ['sendMediaGroup']),
    (['movie.gif'], ['sendAnimation']),
    (['photo.png'], ['sendMediaGroup', 'sendPhoto']),
])
@pytest.mark.parametrize('single', [True, False])
async def test_image_batches_do_not_hide_multiple_operations(delivery, monkeypatch, urls, endpoints, single):
    adapter, request, _ = delivery
    monkeypatch.setattr('tools.url_safety.is_safe_url', lambda _: True)
    images = [(f'https://example.com/{url}', '') for url in urls]
    result = await adapter.send_multiple_images('12345', images, metadata={'single_external_attempt': single, 'notify': True})
    requires_multiple = len(urls) > 10 or (len(urls) > 1 and 'movie.gif' in urls)
    if single and requires_multiple:
        assert not result.success
        assert result.error_kind == 'too_long'
        assert request.calls == []
    elif single and urls == ['photo.png']:
        # Bot API requires 2–10 album items. A definitive refusal is allowed;
        # single_external_attempt does not promise singleton album success.
        assert not result.success
        assert not result.retryable
        assert [endpoint for endpoint, _ in request.calls] == ['sendMediaGroup']
    else:
        assert result.success
        assert [endpoint for endpoint, _ in request.calls] == endpoints


@pytest.mark.asyncio
@pytest.mark.parametrize('single', [True, False])
async def test_album_sdk_failure_does_not_fan_out(delivery, monkeypatch, single):
    import telegram

    adapter, request, _ = delivery
    monkeypatch.delattr(telegram, 'InputMediaPhoto')
    monkeypatch.setattr('tools.url_safety.is_safe_url', lambda _: True)
    result = await adapter.send_multiple_images(
        '12345', [('https://example.com/photo.png', '')] * 2,
        metadata={'single_external_attempt': single, 'notify': True})
    assert result.success is (not single)
    assert len(request.calls) == (0 if single else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize('rich', [False, True])
@pytest.mark.parametrize('single', [True, False])
async def test_success_does_not_schedule_a_second_chat_action(delivery, rich, single):
    adapter, request, _ = delivery
    content = '| A | B |\n|---|---|\n| a | b |' if rich else 'hello'
    result = await adapter.send('12345', content, metadata={'single_external_attempt': single})
    await asyncio.gather(*getattr(adapter, '_telegram_typing_retrigger_tasks', {}).values())
    assert result.success
    assert [endpoint for endpoint, _ in request.calls] == (
        ['sendRichMessage' if rich else 'sendMessage'] + ([] if single else ['sendChatAction']))
