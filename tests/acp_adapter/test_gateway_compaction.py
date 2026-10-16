"""Default gateway ACP projection, core outcomes and durable manual completion."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from acp_adapter.compaction import manual_compaction
from acp_adapter.server import HermesACPAgent
from agent.compaction_lifecycle import _CompactionLifecycle
from agent.conversation_compression import COMPACTION_STATUS, COMPACTION_HEARTBEAT_STATUS
from gateway.session_compaction_events import CompactionProgress, valid_update
from gateway.session_events import SessionEvents
from tests.acp_adapter.test_compaction import connected


@pytest.mark.asyncio
@pytest.mark.parametrize('capabilities', [{}, {'session': {'compaction': True}}, {'session': {'compaction': {}}}])
async def test_gateway_projects_core_lifecycle_without_summary(capabilities):
    agent = HermesACPAgent()
    stream, frames = SessionEvents(), []
    stream.observers.add(frames.append)
    progress = CompactionProgress(lambda update: stream.publish('root', update, event_type='context.compaction'))
    core = SimpleNamespace(status_callback=None, compaction_callback=None, _interrupt_requested=False)
    progress.install(core)
    async with connected(agent) as client:
        await client.request('initialize', {'protocolVersion': 1, 'clientCapabilities': capabilities})
        core.status_callback('lifecycle', COMPACTION_STATUS)
        lifecycle = _CompactionLifecycle(core, False)
        core.status_callback('lifecycle', COMPACTION_HEARTBEAT_STATUS)
        lifecycle.commit_status = 'committed'
        lifecycle.complete()
        progress.close(core)
        core.status_callback('lifecycle', COMPACTION_HEARTBEAT_STATUS)
        for frame in frames:
            await agent._project(frame['params'])
        await client.request('initialize', {'protocolVersion': 1, 'clientCapabilities': capabilities})
        updates = [p['update'] for p in client.updates if p['update']['sessionUpdate'] == 'compaction_update']
        assert all(valid_update(f['params']['payload']) for f in frames)
        if capabilities != {'session': {'compaction': {}}}:
            assert updates == []
            return
        assert [u['status'] for u in updates] == ['in_progress', 'in_progress', 'completed']
        assert len({u['compactionId'] for u in updates}) == 1
        assert all(set(u) == {'sessionUpdate', 'compactionId', 'status'} for u in updates)


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['completed', 'failed', 'cancelled'])
async def test_manual_compaction_waits_for_mutation_scope(outcome):
    import asyncio
    sender = AsyncMock()
    agent = SimpleNamespace(session_updates=SimpleNamespace(supports_session=lambda feature: True, send=sender),
                            _editor_id=lambda sid: sid)
    error = {'failed': RuntimeError, 'cancelled': asyncio.CancelledError}.get(outcome)

    async def run():
        async with manual_compaction(agent, 'root', enabled=True):
            assert sender.await_count == 1  # The durable mutation has not returned yet.
            if error:
                raise error()
    if error:
        with pytest.raises(error):
            await run()
    else:
        await run()
    updates = [c.args[1] for c in sender.await_args_list]
    assert [u['status'] for u in updates] == ['in_progress', outcome]
    assert updates[0]['compactionId'] == updates[1]['compactionId']


def test_worker_frame_is_closed_and_bounded():
    for update in [{'compactionId': 'x' * 65, 'status': 'completed'},
                   {'compactionId': 'x', 'status': 'made-up'},
                   {'compactionId': 'x', 'status': 'completed', 'summary': 'private'}]:
        assert not valid_update(update)
