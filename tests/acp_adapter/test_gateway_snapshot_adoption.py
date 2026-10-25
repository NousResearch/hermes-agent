"""Every fresh ``session.resume`` the editor installs advances the event watermark, so a ``session.info``
queued behind it (another viewer's mode change) is skipped by the pump. Each such install projects the
snapshot's own edit policy, so the editor never shows Default while the owner auto-allows edits."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from acp_adapter.gateway_commands import run_command
from acp_adapter.gateway_server import GatewayACPAgent

OWNER = {'edit_approval_policy': 'session'}  # another viewer already chose Don't Ask (seq 11)


def _set_model(agent):
    return agent.set_session_model('p:m', 's')


async def _deferred_confirm(agent):
    async def yes(ask):
        return True
    agent._model_confirmer = lambda sid: yes
    await agent._confirm_deferred_pick('s', 's', {'payload': {'model': 'm'}, 'confirm': 'tok'})


def _command(text):
    command, _, arg = text.partition(' ')
    return lambda agent: run_command(agent, 's', command, arg)


@pytest.mark.asyncio
@pytest.mark.parametrize('install', [_set_model, _deferred_confirm, _command('new'), _command('compress'),
                                     _command('model p:m'), _command('undo'), _command('retry')],
                         ids=['set_session_model', 'deferred_confirm', 'new', 'compress', 'model', 'undo', 'retry'])
async def test_a_mode_change_queued_behind_a_new_watermark_still_repaints_the_editor(install, monkeypatch):
    agent = GatewayACPAgent()
    agent._conn = AsyncMock()
    agent._snapshots['s'] = {'pending': [], 'replay_epoch': 'e', 'last_sequence': 10,
                             'execution_generation': 1, 'revision': 1}
    agent._modes['s'] = 'default'

    async def rpc(method, **params):
        if method == 'session.resume':
            return {'session_id': 's', 'pending': [], 'prompts': [], 'replay_epoch': 'e', 'last_sequence': 12,
                    'execution_generation': 1, 'revision': 3, 'info': dict(OWNER),
                    'messages': [{'row_id': 1, 'role': 'user', 'content': 'hello'}]}
        if method == 'session.mutate':
            return {'session_id': 's', 'operation': params['operation'], 'revision': 3, 'model': 'm',
                    'provider': 'p', 'rewound_count': 1, 'target_message': {'content': 'hello'}}
        if method == 'session.stats':
            return {'context_length': 1000, 'context_tokens': 0}
        raise AssertionError(method)

    gateway = type('Gateway', (), {})()
    gateway.events, gateway.rpc = asyncio.Queue(), rpc
    agent._gateway = gateway

    async def client():
        return gateway
    monkeypatch.setattr(agent, '_client', client)
    from hermes_cli import models
    monkeypatch.setattr(models, 'parse_model_input', lambda m, d: ('p', 'm'))
    await install(agent)
    # The owner's session.info for the other viewer's change (seq 11) is delivered after the resume.
    await gateway.events.put({'params': {'session_id': 's', 'type': 'session.info', 'replay_epoch': 'e', 'seq': 11,
                                         'payload': dict(OWNER)}})
    await gateway.events.put(EOFError('end of the test stream'))  # the pump records it and stops
    await asyncio.wait_for(agent._events(), 5)
    assert isinstance(agent._failure, EOFError)
    modes = [c.kwargs['update'].current_mode_id for c in agent._conn.session_update.await_args_list
             if getattr(c.kwargs['update'], 'session_update', None) == 'current_mode_update']
    assert agent._snapshots['s']['last_sequence'] == 12
    assert agent._modes['s'] == 'dont_ask' and modes == ['dont_ask']
