"""Explicit /v1/runs selections reuse the strict, per-turn session-chat runtime lock."""
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

import gateway.run  # Initialize the launcher before the per-test home I/O guard.
from tests.gateway.test_api_server_runs import _make_adapter, _create_runs_app
from tests.waiting import async_wait_until


async def _run(adapter, agent, body):
    with patch.object(adapter, '_create_agent', return_value=agent) as create:
        async with TestClient(TestServer(_create_runs_app(adapter))) as client:
            response = await client.post('/v1/runs', json=body)
            assert response.status == 202, await response.text()
            run_id = (await response.json())['run_id']
            await async_wait_until(
                lambda: adapter._run_statuses.get(run_id, {}).get('status') in ('completed', 'failed'))
    return create, adapter._run_statuses[run_id]


def _agent(provider, model):
    agent = MagicMock()
    agent.provider, agent.model = provider, model
    agent._hermes_api_runtime = {'provider': 'custom'}
    agent.run_conversation.return_value = {'final_response': 'ok', 'completed': True}
    return agent


@pytest.mark.asyncio
@pytest.mark.parametrize('locked', [True, False])
async def test_run_passes_the_lock_to_the_agent(locked):
    create, status = await _run(_make_adapter(), _agent('custom', 'local-model'), {
        'input': 'Reply ok', 'provider': 'custom', 'model': 'local-model', 'require_model_lock': locked})
    assert create.call_args.kwargs.get('confirmed_runtime_lock', False) is locked
    assert create.call_args.kwargs['requested_provider'] == 'custom'
    assert create.call_args.kwargs['requested_model'] == 'local-model'
    assert status['status'] == 'completed'
    assert status['runtime'].get('model_lock') == ('confirmed' if locked else None)


@pytest.mark.asyncio
async def test_locked_run_served_by_another_runtime_fails_instead_of_completing():
    # The agent ended the turn on another provider/model (a fallback): the lock refuses the turn.
    _, status = await _run(_make_adapter(), _agent('openai-codex', 'gpt-6.1-sol'), {
        'input': 'Reply ok', 'provider': 'custom', 'model': 'local-model', 'require_model_lock': True})
    assert status['status'] == 'failed'
    assert 'local-model' in status.get('error', '') and 'gpt-6.1-sol' in status.get('error', '')


@pytest.mark.asyncio
async def test_run_lock_without_a_selection_is_refused():
    adapter = _make_adapter()
    with patch.object(adapter, '_create_agent') as create:
        async with TestClient(TestServer(_create_runs_app(adapter))) as client:
            response = await client.post('/v1/runs', json={'input': 'Reply ok', 'require_model_lock': True})
            assert response.status == 400
            create.assert_not_called()
