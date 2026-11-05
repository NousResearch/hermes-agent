"""The classic CLI view submits a ``send`` directive from slash.exec as its own durable turn."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.asyncio
async def test_send_directive_prints_notice_and_submits_through_the_view(capsys):
    from hermes_cli.gateway_chat_commands import run_command
    reply = {'type': 'send', 'message': 'PLAN PROMPT BODY', 'display': '/plan add a cache',
             'notice': 'Planning: add a cache'}
    view = SimpleNamespace(client=SimpleNamespace(rpc=AsyncMock(return_value=reply)), session_id='s',
                           submit=AsyncMock())
    await run_command(view, '/plan', 'add a cache')
    assert view.client.rpc.await_args.kwargs == {'session_id': 's', 'command': 'plan add a cache'}
    view.submit.assert_awaited_once_with('PLAN PROMPT BODY')
    printed = capsys.readouterr().out
    assert 'Planning: add a cache' in printed and '● /plan add a cache' in printed
    assert 'PLAN PROMPT BODY' not in printed  # model-facing body is never painted


@pytest.mark.asyncio
async def test_exec_output_still_prints(capsys):
    from hermes_cli.gateway_chat_commands import run_command
    view = SimpleNamespace(client=SimpleNamespace(rpc=AsyncMock(return_value={'type': 'exec', 'output': 'curator ok'})),
                           session_id='s', submit=AsyncMock())
    await run_command(view, '/curator', '')
    assert 'curator ok' in capsys.readouterr().out and not view.submit.called
