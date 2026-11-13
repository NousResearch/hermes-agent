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
from hermes_state import SessionDB


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
@pytest.mark.parametrize('resume_alias', [False, True])
async def test_default_acp_transports_guidance_and_terminal_receipts(daemon, tmp_path, model_peer, resume_alias):
    model_peer.RequestHandlerClass = GuidanceModel
    model_peer.blocked, model_peer.release = threading.Event(), threading.Event()
    try:
        async with editor(daemon, tmp_path) as acp:
            initialized = await acp.rpc('initialize', protocolVersion=1,
                clientCapabilities={'_meta': {CAPABILITY: {'version': 1}}})
            assert initialized['result']['agentCapabilities']['_meta'][CAPABILITY]['version'] == 1, initialized
            canonical = sid = (await acp.rpc('session/new', cwd=str(tmp_path), mcpServers=[]))['result']['sessionId']
            if resume_alias:
                sid = 'editor-tip'
                with SessionDB(db_path=daemon[0] / 'state.db') as db:
                    db.append_message(canonical, 'user', 'before compression')
                    assert db.try_acquire_compression_lock(canonical, 'fixture')
                    db.publish_compression_child(parent_session_id=canonical, child_session_id=sid, source='acp',
                        messages=[{'role': 'user', 'content': 'summary'}, {'role': 'assistant', 'content': 'retained'}],
                        compression_lock_holder='fixture')
                    db.release_compression_lock(canonical, 'fixture')
                resumed = await acp.rpc('session/load', sessionId=sid, cwd=str(tmp_path), mcpServers=[])
                assert 'result' in resumed, resumed
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
                assert target['sessionId'] == sid
                assert await asyncio.to_thread(model_peer.blocked.wait, 10)
                params = {'sessionId': sid, 'runtimeInstanceId': target['runtimeInstanceId'],
                          'expectedTurnId': target['turnId'], 'messageId': 'm', 'text': '/model literal instruction'}
                admitted = await acp.rpc('_ai.nous/steer', **params)
                assert admitted['result']['state'] == 'pending_incorporation', admitted
                assert admitted['result']['sessionId'] == sid
                model_peer.release.set()
                result = await primary
                final = result['result']['_meta'][CAPABILITY]
                assert final['state'] == 'ended'
                assert final['receipts'][0]['state'] == 'incorporated', final
                assert final['sessionId'] == sid
                assert final['receipts'][0]['sessionId'] == sid
                assert result['result']['usage']['inputTokens'] == 27, result
                assert result['result']['usage']['outputTokens'] == 8, result
                status = dict(params); status.pop('text')
                receipt = (await acp.rpc('_ai.nous/guidanceStatus', **status))['result']
                assert receipt['state'] == 'incorporated'
                assert receipt['sessionId'] == sid
                assert len(model_peer.requests) == 2
                assert model_peer.requests[0]['messages'][-1]['content'] == 'original primary'
                assert model_peer.requests[1]['messages'][-1]['content'] == '/model literal instruction'
                with sqlite3.connect(daemon[0] / 'state.db') as db:
                    rows = db.execute('SELECT target_session_id FROM session_admissions').fetchall()
                    assert rows == [(canonical,)]
                print(json.dumps({'identity': target, 'primary_result': result}))
            finally:
                model_peer.release.set()
                await asyncio.gather(primary, return_exceptions=True)
    finally:
        model_peer.release.set()
