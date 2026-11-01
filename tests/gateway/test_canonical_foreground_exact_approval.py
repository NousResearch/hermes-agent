"""Canonical foreground waits reject historical route-only native consent."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import SendResult
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.run_turn_runner import TurnRunner
from gateway.run_turn_runner_approval_transport import approval_transport
from gateway.session_authority import initialize_session_authority
from gateway.session_results import execution_result
from gateway.turn_context import TurnContext
from tools import approval
from tools.approval_gateway_wait import _await_gateway_decision


@pytest.mark.asyncio
@pytest.mark.parametrize('capability', [None, False, 'yes'])
async def test_canonical_foreground_protected_before_notify_and_successor(monkeypatch, tmp_path, capability):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    from plugins.platforms.slack.adapter import SlackAdapter
    # Unknown/custom adapters must explicitly prove exact callback binding, not just render buttons.
    class CustomAdapter(WhatsAppAdapter):
        supports_exec_approval_request_binding = capability

        async def send_exec_approval(self, **kwargs):
            pytest.fail('unbound native card was delivered')

    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    monkeypatch.setattr(GatewayRunner, '_VOICE_MODE_PATH', tmp_path / 'voice.json')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = CustomAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    runner.adapters = {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='foreground')
    sent, decisions = [], []
    loop = asyncio.get_running_loop()
    route = None

    async def send(chat_id, text, **kwargs):
        sent.append(text)
        request_id = approval.get_pending_gateway_approval(route)['request_id']
        assert f'/approve request:{request_id}' in text
        assert f'/deny request:{request_id}' in text
        assert approval.resolve_gateway_approval(route, 'once') == 0
        assert approval.resolve_gateway_approval(route, 'once', resolve_all=True) == 0
        assert approval.resolve_gateway_approval(route, 'once', request_id='historical') == 0
        # Exercise a real sibling route-only callback, with the interaction/auth seam already admitted.
        slack = SlackAdapter.__new__(SlackAdapter)
        slack._approval_resolved = {'old-card': False}
        monkeypatch.setattr(slack, '_begin_interaction', AsyncMock(return_value=(
            '', next(k for k, v in slack._APPROVAL_CHOICES.items() if v == 'once'),
            route, {}, 'old-card', 'channel', 'owner', 'user')))
        monkeypatch.setattr(slack, '_finalize_interactive_message', AsyncMock())
        await slack._handle_approval_action(AsyncMock(), {}, {})
        assert approval.has_blocking_approval(route)
        assert approval.resolve_gateway_approval(route, 'once', request_id=request_id) == 1
        return SendResult(success=True, message_id='text')

    async def turn(event, *_args, **_kwargs):
        entry = runner.session_store.lookup_by_session_key(route)
        ctx = TurnContext(session_key=route, session_id=entry.session_id, source=event.source,
            _loop_for_step=loop, _status_adapter=adapter, _status_chat_id=event.source.chat_id,
            _status_thread_metadata={}, _run_still_current=lambda: True)
        presenter = TurnRunner(runner, ctx)
        assert presenter._approval_owner is not None
        transport = approval_transport(presenter)
        # This callback runs before notify/render: historical buttons must already be inert.
        def before_notify(data):
            assert approval.resolve_gateway_approval(route, 'once') == 0
            transport(data)
        approval.register_gateway_notify(route, transport)
        try:
            decisions.append(await asyncio.to_thread(_await_gateway_decision, route, before_notify,
                {'command': 'fixture', 'description': 'test'}))
        finally:
            approval.unregister_gateway_notify(route)
        execution_result.get()['result'] = {'final_response': 'done', 'messages': []}
        return 'done'

    monkeypatch.setattr(adapter, 'send', send)
    monkeypatch.setattr(runner, '_handle_message_with_agent', turn)
    event = MessageEvent(text='foreground', source=adapter.build_source(
        chat_id='15551234567@s.whatsapp.net', user_id='15551234567@s.whatsapp.net', chat_type='dm'), message_id='first')
    route = runner._session_key_for_source(event.source)
    try:
        for message_id in ('first', 'successor'):
            event.message_id = message_id
            owned, reply = await runner._hm_authority_admission(event, event.source, route, False)
            assert owned and reply == 'done'
        assert len(sent) == 2
        assert all(d['choice'] == 'once' and d['resolved'] for d in decisions), decisions
    finally:
        approval.clear_session(route)
        authority.db.close()
