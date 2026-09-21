"""Expiring artifacts are checked at dispatch, not when a send is scheduled."""
import asyncio
import hashlib
import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from cron import scheduler_delivery as delivery


def receipt(path, *, seconds=3600, mode='valid'):
    now = datetime.now(timezone.utc)
    body = b'<!doctype html><p>approval draft</p>'
    data = {'version': 1, 'not_before': (now-timedelta(seconds=1)).isoformat(),
            'expires_at': (now+timedelta(seconds=seconds)).isoformat(),
            'sha256': hashlib.sha256(body).hexdigest()}
    if mode == 'future':
        data['not_before'] = (now+timedelta(hours=1)).isoformat()
    if mode == 'naive':
        data['expires_at'] = '2099-01-01T00:00:00'
    header = '<!-- hermes-delivery-expiry ' + json.dumps(data) + ' -->\n'
    if mode == 'missing':
        header = ''
    if mode == 'malformed':
        header = '<!-- hermes-delivery-expiry bad -->\n'
    path.write_bytes(header.encode() + body + (b'changed' if mode == 'tamper' else b''))
    return [(str(path), False)]


@pytest.fixture
def loop():
    loop = asyncio.new_event_loop()
    ready = threading.Event()
    def run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()
    worker = threading.Thread(target=run)
    worker.start()
    assert ready.wait(5)
    yield loop
    loop.call_soon_threadsafe(loop.stop)
    worker.join(5)
    loop.close()


@pytest.mark.parametrize('lane', ['batch', 'single', 'standalone', 'native'])
@pytest.mark.parametrize('mode', ['expired', 'valid', 'missing', 'malformed', 'future', 'naive', 'tamper'])
def test_actual_send_checks_receipt(tmp_path, monkeypatch, loop, lane, mode):
    name = 'review.expiry.png' if lane == 'native' else 'review.expiry.html'
    files = receipt(tmp_path/name, seconds=-1 if mode == 'expired' else 3600, mode=mode)
    sent = []
    async def send(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(success=True)
    if lane == 'standalone':
        async def standalone(*args, **kwargs):
            sent.append(kwargs)
            return {'success': True}
        monkeypatch.setattr('tools.send_message_tool._send_to_platform', standalone)
        target = SimpleNamespace(job={'id': 'isolated'}, where='discord:123', platform='discord',
                                 pconfig=None, chat_id='123', thread_id=None)
        _, error = delivery._standalone_send(target, '', files)
        errors = [error] if error else []
    else:
        adapter = SimpleNamespace(send_document=send, send_image_file=send)
        if lane == 'batch':
            adapter.send_multiple_documents = send
        errors = delivery._send_media_via_adapter(adapter, '123', files, None, loop, {'id': 'isolated'})
    assert bool(sent) == (mode == 'valid'), errors
    assert bool(errors) == (mode != 'valid')


@pytest.mark.parametrize('lane', ['batch', 'standalone'])
def test_expiry_cancels_transport_retry_wait(tmp_path, monkeypatch, loop, lane):
    files = receipt(tmp_path/'review.expiry.html', seconds=3)
    attempts = []
    cancelled = threading.Event()
    async def retrying(*args, **kwargs):
        attempts.append('first')
        try:
            await asyncio.sleep(6)
            attempts.append('retry')
            return {'success': True}
        except asyncio.CancelledError:
            cancelled.set()
            raise
    if lane == 'standalone':
        monkeypatch.setattr('tools.send_message_tool._send_to_platform', retrying)
        target = SimpleNamespace(job={'id': 'isolated'}, where='discord:123', platform='discord',
                                 pconfig=None, chat_id='123', thread_id=None)
        _, error = delivery._standalone_send(target, '', files)
        errors = [error] if error else []
    else:
        adapter = SimpleNamespace(send_multiple_documents=retrying)
        errors = delivery._send_media_via_adapter(adapter, '123', files, None, loop, {'id': 'isolated'})
    assert attempts == ['first']
    assert cancelled.is_set()
    assert errors and 'expir' in str(errors).lower()


@pytest.mark.parametrize('expired', [False, True])
def test_deliver_result_checks_whole_payload_before_any_send(tmp_path, monkeypatch, loop, expired):
    from gateway.config import Platform
    home = tmp_path/'home'
    home.mkdir()
    (home/'config.yaml').write_text('platforms:\n  discord:\n    enabled: true\n    token: synthetic-test-token\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    files = receipt(tmp_path/'review.expiry.html', seconds=-1 if expired else 3600)
    calls = []
    async def send(*args, **kwargs):
        calls.append('text')
        return SimpleNamespace(success=True, message_id='123', raw_response={})
    async def documents(*args, **kwargs):
        calls.append('document')
        return SimpleNamespace(success=True, message_id='124', raw_response={})
    async def fallback(*args, **kwargs):
        calls.append('fallback')
        return {'success': True, 'message_id': '125'}
    monkeypatch.setattr('tools.send_message_tool._send_to_platform', fallback)
    adapter = SimpleNamespace(send=send, send_multiple_documents=documents)
    error = delivery._deliver_result(
        {'id': 'isolated', 'deliver': 'discord:123'},
        'Approval review\nMEDIA:'+files[0][0],
        adapters={Platform.DISCORD: adapter}, loop=loop)
    assert calls == ([] if expired else ['text', 'document']), error
    assert bool(error) is expired


def test_queued_send_rechecks_after_scheduling_delay(tmp_path, monkeypatch, loop):
    import agent.async_utils as async_utils
    path = tmp_path/'queued.expiry.html'
    files = receipt(path)
    sent = []
    async def send(**kwargs):
        sent.append(kwargs)
        return SimpleNamespace(success=True)
    schedule = async_utils.safe_schedule_threadsafe
    def delayed(coro, event_loop):
        # Receipt changes after the caller assembled its coroutine, before dispatch.
        receipt(path, seconds=-1)
        return schedule(coro, event_loop)
    monkeypatch.setattr(async_utils, 'safe_schedule_threadsafe', delayed)
    errors = delivery._send_media_via_adapter(
        SimpleNamespace(send_multiple_documents=send), '123', files, None, loop, {'id': 'isolated'})
    assert not sent
    assert errors and 'expired' in str(errors)


def test_ordinary_attachment_and_transport_error_keep_existing_semantics(tmp_path):
    from cron.delivery_expiry import dispatch_before_expiry, DeliveryExpired
    plain = tmp_path/'plain.html'
    plain.write_text('<!doctype html>ordinary report')
    async def good():
        return 'sent'
    assert asyncio.run(dispatch_before_expiry([(str(plain), False)], good)) == 'sent'
    files = receipt(tmp_path/'report.expiry.html')
    async def failed():
        raise TimeoutError('transport timeout, not expiry')
    with pytest.raises(TimeoutError, match='transport timeout'):
        asyncio.run(dispatch_before_expiry(files, failed))
