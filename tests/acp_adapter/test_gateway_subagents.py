"""Canonical gateway lifecycle events reach the default ACP implementation over the SDK."""
from types import SimpleNamespace

import pytest
import acp

from acp_adapter.server import HermesACPAgent
from gateway.session_events import SessionEvents
from gateway.session_subagent_events import publish_subagent_progress
from tests.acp_adapter.test_subagents import connected


@pytest.mark.asyncio
@pytest.mark.parametrize('capabilities', [{}, {'subagents': True}, {'subagents': {}}])
async def test_default_gateway_projects_children_with_parent_only_controls(capabilities):
    agent = HermesACPAgent()
    stream = SessionEvents()
    frames = []
    stream.observers.add(frames.append)
    turn = SimpleNamespace(_publish_execution=lambda kind, payload: stream.publish('root', payload, event_type=kind))
    async with connected(agent) as client:
        await client.request('initialize', {'protocolVersion': 1, 'clientCapabilities': capabilities})
        for event, identity in [
            ('subagent.start', {'subagent_id': 'first', 'goal': 'Research'}),
            ('subagent.start', {'subagent_id': 'nested', 'parent_id': 'first', 'goal': 'Verify'}),
            ('subagent.complete', {'subagent_id': 'nested', 'parent_id': 'first', 'status': 'completed'}),
            ('subagent.complete', {'subagent_id': 'first', 'status': 'interrupted'}),
        ]:
            publish_subagent_progress(turn, event, None, {**identity, 'private_transcript': 'never forward'})
        for frame in frames:
            await agent._project(frame['params'])
        # Same SDK sender ordering: drain the notifications before this response.
        await client.request('initialize', {'protocolVersion': 1, 'clientCapabilities': capabilities})
        updates = [p for p in client.updates if p['update']['sessionUpdate'] == 'subagent_update']
        assert all('private_transcript' not in str(f) for f in frames)
        if capabilities != {'subagents': {}}:
            assert updates == []
            return
        assert len(updates) == 4
        first, nested = [p['update']['sessionId'] for p in updates[:2]]
        assert [p['sessionId'] for p in updates] == ['root', first, first, 'root']
        assert updates[2]['update']['sessionId'] == nested
        assert updates[2]['update']['state'] == {'state': 'idle', 'stopReason': 'end_turn'}
        assert updates[3]['update']['state'] == {'state': 'idle', 'stopReason': 'cancelled'}
        for method, extra in [('session/prompt', {'prompt': [{'type': 'text', 'text': 'no'}]}),
                              ('session/load', {'cwd': '.', 'mcpServers': []}),
                              ('session/resume', {'cwd': '.', 'mcpServers': []}),
                              ('session/fork', {'cwd': '.', 'mcpServers': []}),
                              ('session/set_model', {'modelId': 'other'}),
                              ('session/set_mode', {'modeId': 'default'}),
                              ('session/set_config_option', {'configId': 'x', 'value': 'y'})]:
            rejected = await client.request(method, {'sessionId': first, **extra})
            assert rejected['error']['code'] == -32602
        with pytest.raises(acp.RequestError, match='managed by their parent'):
            await agent.cancel(first)


def test_turn_callback_publishes_before_complete_notice_returns():
    from gateway.config import Platform
    from tests.gateway.test_local_viewer_events_turn import _turn
    published = []
    turn = _turn(Platform.LOCAL, published)
    turn.progress_callback('subagent.start', subagent_id='first', goal='Research')
    turn.progress_callback('subagent.complete', subagent_id='first', status='completed')
    assert [kind for kind, payload in published] == ['subagent.start', 'subagent.complete']
    assert published[0][1]['subagent_id'] == published[1][1]['subagent_id'] == 'first'
