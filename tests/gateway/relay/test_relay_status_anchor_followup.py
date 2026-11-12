"""Runner typing sets outside the adapter bracket keep a relay Slack status on the turn's own message.

The in-band queued follow-up (``_run_agent_queued_followup``) and proxy mode (``_run_agent_via_proxy``)
each set the status themselves. With metadata that names no thread (flat replies, a top-level DM), the
relay adapter anchored it on the chat's newest message while the turn's clear targets the turn's own
message, leaving the newer one "is working". Real ``GatewayRunner`` + ``RelayAdapter`` over the stub
connector; only the model and the proxy's HTTP boundary are faked.
"""
import asyncio
import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from hermes_agent.gateway.config import GatewayConfig, Platform, PlatformConfig
from hermes_agent.gateway.platforms.event import MessageEvent, MessageType
from hermes_agent.gateway.session import SessionSource

FLAT_REPLIES = {'slack': {'reply_in_thread': False}}
SESSION_KEY = 'agent:main:slack:dm:T1:D9'


class _TwoTurnAgent:
    calls: list = []

    def __init__(self, **_kwargs):
        self.tools = []

    def run_conversation(self, message, conversation_history=None, task_id=None, **_kwargs):
        type(self).calls.append(message)
        return {'final_response': f'done-{len(type(self).calls)}', 'messages': [], 'api_calls': 1}


def _event(message_id, text='hi'):
    source = SessionSource(platform=Platform.SLACK, chat_id='D9', chat_type='dm', user_id='U1', scope_id='T1')
    return MessageEvent(text=text, source=source, message_type=MessageType.TEXT, message_id=message_id)


@pytest.mark.asyncio
async def test_queued_followup_status_stays_on_its_own_message(monkeypatch, tmp_path):
    from hermes_agent.gateway.relay.adapter import RelayAdapter
    from hermes_agent.gateway.run import GatewayRunner
    from tests.gateway.relay.stub_connector import StubConnector
    from tests.gateway.relay.test_relay_slack_prompt_dm_root import _slack_desc

    fake_run_agent = types.ModuleType('run_agent')
    fake_run_agent.AIAgent = _TwoTurnAgent
    monkeypatch.setitem(sys.modules, 'run_agent', fake_run_agent)
    gateway_run = importlib.import_module('hermes_agent.gateway.run')
    monkeypatch.setattr(gateway_run, '_hermes_home', tmp_path)
    monkeypatch.setattr(gateway_run, '_resolve_runtime_agent_kwargs', lambda: {'api_key': '***'})
    _TwoTurnAgent.calls = []

    stub = StubConnector(_slack_desc())
    adapter = RelayAdapter(PlatformConfig(extra=FLAT_REPLIES), _slack_desc(), transport=stub)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    runner.adapters[Platform.RELAY] = adapter
    runner._delivery_adapter_for = lambda _source: adapter
    first, queued, newest = _event('1700000000.000900'), _event('1700000000.000901', 'follow-up'), _event('1700000000.000902')
    for event in (first, queued, newest):  # a newer message reached the chat after the queued one
        adapter._capture_scope(event)
    adapter._pending_messages[SESSION_KEY] = queued

    result = await runner._run_agent(message='first', context_prompt='', history=[], source=first.source,
                                     session_id='sess-anchor', session_key=SESSION_KEY,
                                     event_message_id=first.message_id)

    assert _TwoTurnAgent.calls == ['first', 'follow-up'] and result['final_response'] == 'done-2'
    typing = [(f['metadata'] or {}).get('thread_id') for f in stub.sent if f['op'] == 'typing']
    assert '1700000000.000902' not in typing
    assert '1700000000.000901' in typing


class _CancelledPost:
    """The proxy's HTTP boundary: the turn is cancelled (``/stop``) while the request is in flight."""

    def post(self, *_args, **_kwargs):
        raise asyncio.CancelledError

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


@pytest.mark.asyncio
async def test_a_cancelled_proxy_turn_leaves_no_status(monkeypatch, tmp_path):
    from hermes_agent.gateway.relay.adapter import RelayAdapter
    from hermes_agent.gateway.run import GatewayRunner
    from tests.gateway.relay.stub_connector import StubConnector
    from tests.gateway.relay.test_relay_slack_prompt_dm_root import _slack_desc

    stub = StubConnector(_slack_desc())
    adapter = RelayAdapter(PlatformConfig(extra=FLAT_REPLIES), _slack_desc(), transport=stub)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    runner.adapters[Platform.RELAY] = adapter
    runner._delivery_adapter_for = lambda _source: adapter
    runner._get_proxy_url = lambda: 'http://proxy.invalid:8642'
    monkeypatch.setenv('GATEWAY_PROXY_KEY', 'fixture')
    turn, newer = _event('1700000000.000910'), _event('1700000000.000911')
    for event in (turn, newer):  # the queued turn starts after a newer message reached the chat
        adapter._capture_scope(event)

    aiohttp = types.SimpleNamespace(ClientSession=MagicMock(return_value=_CancelledPost()), ClientTimeout=MagicMock())
    with patch.dict(sys.modules, {'aiohttp': aiohttp}), pytest.raises(asyncio.CancelledError):
        await runner._run_agent_via_proxy(message='hi', context_prompt='', history=[], source=turn.source,
                                          session_id='sess-proxy', session_key=SESSION_KEY,
                                          event_message_id=turn.message_id)
    await runner._hmwa_stop_typing_for_turn(turn, turn.source)  # the turn's clear; no reply was posted

    frames = [((f['metadata'] or {}).get('thread_id'), f.get('content') != '') for f in stub.sent if f['op'] == 'typing']
    assert frames == [('1700000000.000910', True), ('1700000000.000910', False)]  # set, then cleared, on one message
