"""Outbound callback delivery must work before any inbound message or listener."""
from pathlib import Path

import httpx
import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platform_registry import platform_registry
from hermes_cli.plugins import PluginManager
from hermes_cli.plugins_manifest import parse_manifest_file


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['success', 'api_error', 'exception', 'refresh', 'scoped', 'long'])
async def test_fresh_callback_delivery(monkeypatch, mode):
    from plugins.platforms.wecom import callback_adapter as cb
    from tools.send_message_tool import _send_to_platform
    from tools.send_message_targets import resolve_send_target

    root = Path(__file__).resolve().parents[2] / 'plugins/platforms/wecom'
    manifest = parse_manifest_file(root / 'plugin.yaml', root, 'bundled', 'platforms')
    manager = PluginManager()
    calls = []
    clients = []
    real_client = httpx.AsyncClient

    def transport(request):
        calls.append(request)
        if mode == 'exception':
            raise httpx.ConnectError('fixture offline')
        if request.url.path.endswith('/gettoken'):
            return httpx.Response(200, json={'errcode': 0, 'access_token': 'fixture-token', 'expires_in': 7200})
        if mode == 'api_error':
            return httpx.Response(200, json={'errcode': 81013, 'errmsg': 'fixture refusal'})
        if mode == 'refresh' and len(calls) == 2:
            return httpx.Response(200, json={'errcode': 40001})
        return httpx.Response(200, json={'errcode': 0, 'msgid': 'fixture-message'})

    def client(**kwargs):
        result = real_client(transport=httpx.MockTransport(transport), **kwargs)
        clients.append(result)
        return result

    async def forbidden(*args, **kwargs):
        pytest.fail('Outbound delivery must not start the callback listener')

    monkeypatch.setattr(cb.httpx, 'AsyncClient', client)
    monkeypatch.setattr(cb.WecomCallbackAdapter, 'connect', forbidden)
    manager._register_deferred_platform(manifest)
    try:
        entry = platform_registry.get('wecom_callback')
        assert entry is not None, 'secondary platform must resolve in a fresh process'
        recipient = 'fixture-corp:alice' if mode == 'scoped' else 'alice'
        target, thread, error = resolve_send_target('wecom_callback', recipient)
        assert (target, thread, error) == (recipient, None, None)
        extra = {
            'corp_id': 'fixture-corp', 'corp_secret': 'fixture-secret', 'agent_id': '123',
        }
        if mode == 'scoped':
            extra = {'apps': [dict(extra, name='other', corp_id='other', agent_id='999'), dict(extra, name='selected')]}
        message = 'x' * 3000 if mode == 'long' else 'hello'
        result = await _send_to_platform(Platform.WECOM_CALLBACK, PlatformConfig(enabled=True, extra=extra), target, message)
        if mode in {'api_error', 'exception'}:
            assert result.get('error'), result
        else:
            assert result.get('success') is True, result
            assert result.get('message_id') == 'fixture-message'
            assert len(calls) == (4 if mode in {'refresh', 'long'} else 2)
            import json
            assert json.loads(calls[-1].content)['agentid'] == 123
            if mode == 'long':
                import re
                payloads = [json.loads(r.content)['text']['content'] for r in calls if r.method == 'POST']
                delivered = ''.join(re.sub(r' \(\d+/\d+\)$', '', text) for text in payloads)
                assert all(len(text) <= 2048 for text in payloads)
                assert delivered == message
        assert clients and all(c.is_closed for c in clients)
    finally:
        manager.unload(manifest)


@pytest.mark.asyncio
@pytest.mark.parametrize('target,mapping,expected_agent', [
    ('alice', {}, None),
    ('unknown:alice', {}, None),
    ('b:', {}, None),
    ('b:alice', {}, 2),
    ('alice', {'b:alice': 'b'}, 2),
    ('alice', {'a:alice': 'a', 'b:alice': 'b'}, None),
    ('b:alice', {'b:alice': 'b'}, 2),
])
async def test_live_callback_routing_never_guesses_first_app(monkeypatch, target, mapping, expected_agent):
    import json
    from types import SimpleNamespace
    from plugins.platforms.wecom.callback_adapter import WecomCallbackAdapter
    from tools import send_message_tool as send
    from tools.send_message_targets import resolve_send_target

    root = Path(__file__).resolve().parents[2] / 'plugins/platforms/wecom'
    manifest = parse_manifest_file(root / 'plugin.yaml', root, 'bundled', 'platforms')
    manager = PluginManager()
    config = PlatformConfig(enabled=True, extra={'apps': [
        {'name': 'a', 'corp_id': 'a', 'corp_secret': 'fixture', 'agent_id': '1'},
        {'name': 'b', 'corp_id': 'b', 'corp_secret': 'fixture', 'agent_id': '2'},
    ]})
    adapter = WecomCallbackAdapter(config)
    adapter._user_app_map.update(mapping)
    calls = []

    def transport(request):
        calls.append(request)
        if request.method == 'GET':
            return httpx.Response(200, json={'errcode': 0, 'access_token': 'fixture', 'expires_in': 7200})
        return httpx.Response(200, json={'errcode': 0, 'msgid': 'fixture'})

    adapter._http_client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    monkeypatch.setattr(send, '_live_adapter', lambda platform: (SimpleNamespace(), adapter))
    try:
        manager._register_deferred_platform(manifest)
        resolved, _, error = resolve_send_target('wecom_callback', target)
        assert error is None
        result = await send._send_to_platform(Platform.WECOM_CALLBACK, config, resolved, 'hello')
        if expected_agent is None:
            assert result.get('error'), result
            assert calls == []
        else:
            assert result.get('success'), result
            assert json.loads(calls[-1].content)['agentid'] == expected_agent
    finally:
        await adapter.aclose_http_client()
        manager.unload(manifest)


@pytest.mark.asyncio
@pytest.mark.parametrize('target,extra', [
    ('alice', {}),
    ('unknown:alice', {'corp_id': 'corp', 'corp_secret': 'fixture', 'agent_id': '1'}),
    ('corp:', {'corp_id': 'corp', 'corp_secret': 'fixture', 'agent_id': '1'}),
    ('alice', {'apps': [{'name': 'a', 'corp_id': 'a'}, {'name': 'b', 'corp_id': 'b'}]}),
])
async def test_ambiguous_or_missing_app_never_opens_client(monkeypatch, target, extra):
    from plugins.platforms.wecom.adapter import _callback_standalone_send
    from plugins.platforms.wecom.callback_adapter import WecomCallbackAdapter

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid routing must not create a transport')

    monkeypatch.setattr(WecomCallbackAdapter, '_ensure_http_client', forbidden)
    result = await _callback_standalone_send(PlatformConfig(enabled=True, extra=extra), target, 'hello')
    assert result.get('error')
