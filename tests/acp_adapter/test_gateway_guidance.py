"""The default gateway-backed ACP entry point must use the actual execution owner."""
import asyncio
import json
import sqlite3
import threading

import pytest
from acp.schema import ClientCapabilities

from acp_adapter.gateway_server import GatewayACPAgent
from agent.active_turn_guidance import CAPABILITY
from tests.acp_adapter.test_gateway_sessions import editor
from tests.gateway.fixtures.guidance_model import GuidanceModel
from tests.waiting import async_wait_until
from hermes_cli.gateway_client import GatewayClientError


@pytest.mark.asyncio
async def test_guidance_requires_explicit_client_and_owner_negotiation():
    agent = GatewayACPAgent()
    calls = []
    class Client:
        async def rpc(self, method, **params):
            calls.append(method)
            return {'capabilities': []}
    async def client():
        return Client()
    agent._client = client
    assert await agent._guidance.capability(ClientCapabilities()) is None
    assert calls == []
    capable = ClientCapabilities(field_meta={CAPABILITY: {'version': 1}})
    assert await agent._guidance.capability(capable) is None
    assert calls == ['runtime.describe']
    agent._guidance.observe_owner({'capabilities': ['active-turn-guidance-v1']})
    assert agent._guidance.negotiated
    agent._guidance.observe_owner({'capabilities': []})
    assert not agent._guidance.negotiated


@pytest.mark.asyncio
async def test_terminal_receipt_transport_loss_keeps_primary_result_available():
    agent = GatewayACPAgent()
    agent._guidance.negotiated = True
    agent._guidance.track('s', 'a', 'host')
    async def unavailable():
        raise GatewayClientError('receipt connection lost')
    agent._client = unavailable
    assert await agent._guidance.terminal('s', 'a') is None
    assert agent._guidance.prompts == {}


@pytest.mark.platforms('linux')
@pytest.mark.asyncio
@pytest.mark.parametrize('daemon', [{'platform_toolsets': {'acp': ['terminal']}}], indirect=True)
async def test_default_acp_transports_guidance_and_terminal_receipts(daemon, tmp_path, model_peer):
    model_peer.RequestHandlerClass = GuidanceModel
    model_peer.blocked, model_peer.release = threading.Event(), threading.Event()
    try:
        async with editor(daemon, tmp_path) as acp:
            initialized = await acp.rpc('initialize', protocolVersion=1,
                clientCapabilities={'_meta': {CAPABILITY: {'version': 1}}})
            assert initialized['result']['agentCapabilities']['_meta'][CAPABILITY]['version'] == 1, initialized
            sid = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']['sessionId']
            for metadata in (None, 'invalid', [], {}, {'promptId': 4}):
                refused = await acp.rpc('session/prompt', sessionId=sid,
                    prompt=[{'type': 'text', 'text': 'invalid primary'}], _meta={CAPABILITY: metadata})
                assert refused.get('error', {}).get('code') == -32602, refused
            primary = asyncio.create_task(acp.rpc('session/prompt', sessionId=sid,
                prompt=[{'type': 'text', 'text': 'original primary'}], _meta={CAPABILITY: {'promptId': 'host-primary'}}))
            try:
                await async_wait_until(lambda: any(f.get('method') == '_ai.nous/guidanceTurn' for f in acp.frames),
                                       timeout=20, what='owner-confirmed ACP turn')
                target = next(f['params'] for f in acp.frames if f.get('method') == '_ai.nous/guidanceTurn')
                assert target['promptId'] == 'host-primary'
                assert await asyncio.to_thread(model_peer.blocked.wait, 10)
                params = {'sessionId': sid, 'runtimeInstanceId': target['runtimeInstanceId'],
                          'expectedTurnId': target['turnId'], 'messageId': 'm', 'text': '/model literal instruction'}
                admitted = await acp.rpc('_ai.nous/steer', **params)
                assert admitted['result']['state'] == 'pending_incorporation', admitted
                model_peer.release.set()
                result = await primary
                final = result['result']['_meta'][CAPABILITY]
                assert final['state'] == 'ended'
                assert final['receipts'][0]['state'] == 'incorporated', final
                assert result['result']['usage']['inputTokens'] == 27, result
                assert result['result']['usage']['outputTokens'] == 8, result
                status = dict(params); status.pop('text')
                assert (await acp.rpc('_ai.nous/guidanceStatus', **status))['result']['state'] == 'incorporated'
                assert len(model_peer.requests) == 2
                assert model_peer.requests[0]['messages'][-1]['content'] == 'original primary'
                assert model_peer.requests[1]['messages'][-1]['content'] == '/model literal instruction'
                with sqlite3.connect(daemon[0] / 'state.db') as db:
                    assert db.execute('SELECT count(*) FROM session_admissions').fetchone()[0] == 1
                print(json.dumps({'identity': target, 'primary_result': result}))
            finally:
                model_peer.release.set()
                await asyncio.gather(primary, return_exceptions=True)
    finally:
        model_peer.release.set()
