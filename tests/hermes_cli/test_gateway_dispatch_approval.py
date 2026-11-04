"""The actual terminal response producer must preserve exactly one owner."""
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest
from hermes_cli.gateway_chat_view import GatewayChatView
from hermes_cli.gateway_client import GatewayClient, GatewayClientError
import asyncio
import json
from tui_gateway.contracts.canonical_sessions import CanonicalApprovalParams, CanonicalPendingPrompt
from pydantic import ValidationError


@pytest.mark.asyncio
@pytest.mark.parametrize('owner', [{'dispatch_owner': 'lease'}, {'execution_generation': 3}])
async def test_resumed_terminal_approval_response(owner):
    client = SimpleNamespace(rpc=AsyncMock(return_value={'status': 'resolved'}))
    prompt = {'kind': 'approval', 'prompt_id': 'p', 'choices': ['once', 'deny'], **owner}
    CanonicalPendingPrompt.model_validate(prompt)
    view = GatewayChatView(client, {'stored_session_id': 's', 'prompts': [prompt]})
    assert await view.command('/approve p once')
    client.rpc.assert_awaited_once_with('approval.respond', session_id='s', prompt_id='p', choice='once', **owner)
    CanonicalApprovalParams.model_validate(client.rpc.call_args.kwargs)


@pytest.mark.parametrize('owner', [{}, {'dispatch_owner': ''}, {'dispatch_owner': None},
    {'dispatch_owner': 'lease', 'execution_generation': 3}, {'execution_generation': True},
    {'execution_generation': '3'}])
def test_contract_refuses_malformed_approval_owner(owner):
    with pytest.raises(ValidationError):
        CanonicalApprovalParams.model_validate({'session_id': 's', 'prompt_id': 'p', 'choice': 'once', **owner})


@pytest.mark.asyncio
async def test_terminal_refuses_ambiguous_owner():
    client = SimpleNamespace(rpc=AsyncMock())
    view = GatewayChatView(client, {'stored_session_id': 's', 'prompts': [
        {'kind': 'approval', 'prompt_id': 'p', 'dispatch_owner': 'lease', 'execution_generation': 3}]})
    with pytest.raises(GatewayClientError):
        await view.command('/approve p once')
    client.rpc.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('owner', [{'dispatch_owner': 'lease'}, {'execution_generation': 3}])
async def test_terminal_response_reaches_real_client_wire(owner):
    frames = []
    async def send(raw):
        frame = json.loads(raw)
        frames.append(frame)
        client.pending[frame['id']].set_result({'status': 'resolved'})
    client = GatewayClient(SimpleNamespace(send=send))
    prompt = {'kind': 'approval', 'prompt_id': 'p', **owner}
    view = GatewayChatView(client, {'stored_session_id': 's', 'prompts': [prompt]})
    await asyncio.wait_for(view.command('/approve p once'), 1)
    assert frames == [{'jsonrpc': '2.0', 'id': 1, 'method': 'approval.respond',
                       'params': {'session_id': 's', 'prompt_id': 'p', 'choice': 'once', **owner}}]
    CanonicalApprovalParams.model_validate(frames[0]['params'])
