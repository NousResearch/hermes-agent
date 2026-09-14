"""PR #19121: senderless group authorization and visible media failures.

Real adapter parsing, AES decryption, disk cache and gateway ingress; only the
HTTP transport, outbound network send and final agent boundary are substituted.
"""
import asyncio
import base64
import os
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import quote

import httpx
import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.run import GatewayRunner
from plugins.platforms.wecom.adapter import WeComAdapter


@pytest.fixture
def delivery(monkeypatch):
    for key in ('WECOM_ALLOWED_USERS', 'WECOM_ALLOW_ALL_USERS', 'WECOM_GROUP_POLICY',
                'WECOM_GROUP_ALLOW_FROM', 'GATEWAY_ALLOWED_USERS', 'GATEWAY_ALLOW_ALL_USERS'):
        monkeypatch.delenv(key, raising=False)
    # Address validation still executes, but the fake transport never opens a socket.
    monkeypatch.setattr('tools.url_safety.is_safe_url', lambda url: True)

    def make(extra=None, failure=None):
        config = PlatformConfig(enabled=True, extra={
            'group_policy': 'allowlist', 'group_allow_from': ['group-ok'],
            'attachment_text_merge_delay_seconds': 0, **(extra or {}),
        })
        adapter = WeComAdapter(config)
        adapter._text_batch_delay_seconds = 0
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(platforms={Platform.WECOM: config})
        runner.adapters = {Platform.WECOM: adapter}
        runner.pairing_store = None
        runner._scale_to_zero_note_real_inbound = lambda: None
        adapter.send = AsyncMock(return_value=SendResult(success=True))
        admitted = []

        async def dispatch(event):
            result = await runner._hm_admit_event(event)
            if result is not None:
                admitted.append(result[0])
        adapter.set_message_handler(dispatch)
        adapter.send_typing = AsyncMock()
        adapter.stop_typing = AsyncMock()
        plain = b'%PDF-1.7\nfile content\n'
        key = os.urandom(32)
        pad = 16 - len(plain) % 16
        encryptor = Cipher(algorithms.AES(key), modes.CBC(key[:16])).encryptor()
        encrypted = encryptor.update(plain + bytes([pad]) * pad) + encryptor.finalize()

        def response(request):
            if failure == 'download' and request.url.path == '/bad.pdf':
                return httpx.Response(503)
            data = b'not ciphertext' if failure == 'decrypt' and request.url.path == '/bad.pdf' else encrypted
            return httpx.Response(200, content=data, headers={
                'content-type': 'application/pdf', 'content-disposition': 'attachment; filename="example.pdf"',
            })
        adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(response))
        ref = {'url': 'https://files.example/bad.pdf', 'aeskey': quote(base64.b64encode(key).decode().rstrip('='), safe='')}
        payload = {'headers': {'req_id': 'req-1'}, 'body': {
            'msgid': 'msg-1', 'chatid': 'group-ok', 'chattype': 'group',
            'from': {'userid': ''}, 'msgtype': 'file', 'file': ref,
        }}
        return adapter, runner, admitted, payload, plain
    return make


@pytest.mark.asyncio
@pytest.mark.parametrize('extra,sender,chat_type,env,expected', [
    ({}, '', 'group', None, True),
    ({'group_allow_from': ['wecom:group:GROUP-OK']}, '', 'group', None, True),
    ({'group_allow_from': ['other']}, '', 'group', None, False),
    ({'group_policy': 'open'}, '', 'group', None, False),
    ({'group_policy': 'pairing'}, '', 'group', None, False),
    ({'group_policy': 'disabled'}, '', 'group', None, False),
    ({'groups': {'group-ok': {'allow_from': ['alice']}}}, '', 'group', None, False),
    ({'groups': {'*': {'allow_from': ['alice']}}}, '', 'group', None, False),
    ({}, '', 'group', 'alice', False),
    ({}, '', 'dm', None, False),
    ({}, 'alice', 'group', None, True),
])
async def test_senderless_files_require_explicit_group_authority(delivery, monkeypatch, extra, sender, chat_type, env, expected):
    if env:
        monkeypatch.setenv('WECOM_ALLOWED_USERS', env)
    adapter, runner, admitted, payload, plain = delivery(extra)
    payload['body']['from']['userid'] = sender
    payload['body']['chattype'] = chat_type
    try:
        await adapter._on_message(payload)
        if adapter._background_tasks:
            await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks)), timeout=5)
        assert bool(admitted) is expected
        adapter.send.assert_not_awaited()
        if expected:
            assert Path(admitted[0].media_urls[0]).read_bytes() == plain
            assert admitted[0].source.user_id == (sender or None)
            if not sender:
                assert not runner._is_user_authorized(admitted[0].source, allow_adapter_delegation=False)
    finally:
        await adapter._http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize('failure,kind,with_text,with_good_file,authorized', [
    ('download', 'file', False, False, True),
    ('decrypt', 'file', False, False, True),
    ('download', 'appmsg', False, False, True),
    ('decrypt', 'file', True, False, True),
    ('download', 'file', False, True, True),
    ('download', 'file', False, False, False),
])
async def test_failed_media_is_reported_without_losing_valid_content(delivery, failure, kind, with_text, with_good_file, authorized):
    adapter, runner, admitted, payload, plain = delivery(
        {} if authorized else {'group_policy': 'open'}, failure=failure)
    body = payload['body']
    if kind == 'appmsg':
        body['msgtype'] = kind
        body[kind] = {'file': body.pop('file')}
    if with_text:
        body['text'] = {'content': '请总结附件'}
    if with_good_file:
        body['quote'] = {'msgtype': 'file', 'file': {**body['file'], 'url': 'https://files.example/good.pdf'}}
    try:
        await adapter._on_message(payload)
        if adapter._background_tasks:
            await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks)), timeout=5)
        assert adapter.send.await_count == int(authorized)
        assert bool(admitted) == (authorized and (with_text or with_good_file))
        if authorized:
            assert '失败' in adapter.send.await_args.args[1]
            assert adapter.send.await_args.kwargs['reply_to'] == body['msgid']
        if admitted:
            assert admitted[0].text == ('请总结附件' if with_text else '')
            assert len(admitted[0].media_urls) == int(with_good_file)
            if with_good_file:
                assert Path(admitted[0].media_urls[0]).read_bytes() == plain
        # Only the successful sibling may have reached the real disk cache.
        from hermes_constants import get_hermes_home
        cached = [p for p in Path(get_hermes_home()).rglob('*') if p.is_file() and p.read_bytes() == plain]
        assert bool(cached) == with_good_file
    finally:
        await adapter._http_client.aclose()


@pytest.mark.asyncio
async def test_senderless_group_grant_uses_receiving_profile(delivery):
    primary, runner, _, _, _ = delivery()
    secondary, _, _, _, _ = delivery({'group_allow_from': ['secondary-group']})
    runner._profile_adapters = {'secondary': {Platform.WECOM: secondary}}
    try:
        for receiving, chat, routed_profile, expected in (
            (primary, 'group-ok', 'secondary', True),
            (secondary, 'group-ok', None, False),
            (secondary, 'secondary-group', None, True),
        ):
            source = receiving.build_source(chat_id=chat, chat_type='group', user_id=None)
            source.profile = routed_profile
            assert runner._is_user_authorized(source) is expected
        # No connected receiving adapter must not borrow a primary bot's allowlist.
        source = secondary.build_source(chat_id='secondary-group', chat_type='group', user_id=None)
        source.profile = 'missing-profile'
        runner._profile_adapters.clear()
        assert not runner._is_user_authorized(source)
    finally:
        await primary._http_client.aclose()
        await secondary._http_client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("authorized", [True, False])
async def test_failed_file_does_not_interrupt_an_active_turn(delivery, authorized):
    adapter, runner, admitted, payload, _ = delivery(
        {} if authorized else {"group_policy": "open"}, failure="decrypt")
    source = adapter.build_source(chat_id="group-ok", chat_type="group", user_id=None)
    key = adapter._source_session_key(source)
    active = asyncio.Event()
    adapter._active_sessions[key] = active
    adapter._heal_stale_session_lock = lambda key: None  # the long-running agent is outside this test
    adapter.set_busy_session_handler(runner._handle_active_session_busy_message)
    try:
        await adapter._on_message(payload)
        assert adapter.send.await_count == int(authorized)
        assert not active.is_set()
        assert not adapter._pending_messages
        assert not admitted
    finally:
        await adapter._http_client.aclose()
