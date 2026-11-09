"""A lent bot's real terminal approval travels through both canonical gateways."""
import asyncio
import json
import shlex

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.gateway.test_session_group_peer_daemons import (
    Model, _create, _events, _invite, _register, _room_link, _send, _wait_rpc,
    gateways as gateways_fixture,
)

gateways = gateways_fixture
pytestmark = pytest.mark.live_system_guard_bypass


class _ApprovalModel(Model):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        messages = body.get('messages', [])
        last_user = max((i for i, m in enumerate(messages) if m['role'] == 'user'), default=-1)
        finished = not messages or any(m['role'] == 'tool' for m in messages[last_user + 1:])
        message = {'role': 'assistant', 'content': 'PEER_REPLY' if finished else None}
        if not finished:
            message['tool_calls'] = [{'index': 0, 'id': 'approved-tool', 'type': 'function', 'function': {
                'name': 'terminal', 'arguments': json.dumps({'command': self.server.command})}}]
        choice = {'index': 0, 'delta' if body.get('stream') else 'message': message,
                  'finish_reason': 'stop' if finished else 'tool_calls'}
        payload = json.dumps({'id': 'fixture', 'choices': [choice]})
        if body.get('stream'):
            payload = 'data: ' + payload + '\n\ndata: [DONE]\n\n'
        encoded = payload.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream' if body.get('stream') else 'application/json')
        self.send_header('Content-Length', str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@pytest.mark.platforms('posix')
def test_lent_bot_approval_survives_reregistration_and_runs_each_command_once(gateways):
    g = gateways(handler=_ApprovalModel)
    config = json.loads((g.target / 'config.yaml').read_text())
    config['platform_toolsets']['api_server'] = ['terminal']
    config['approvals'] = {'mode': 'manual'}
    (g.target / 'config.yaml').write_text(json.dumps(config))
    effects = g.target / 'effects'
    g.target_model.command = (
        f'rm -rf {shlex.quote(str(g.target / "disposable"))}; '
        f'echo ran >> {shlex.quote(str(effects))}')

    async def exercise(home_desc, target_desc):
        async with websocket(g.home, home_desc) as home_ws, websocket(g.target, target_desc) as target_ws:
            catalog = (await _room_link(target_ws))['catalog']
            room = await _create(home_ws, catalog)
            grant = (await _invite(target_ws, room))['grant']
            assert (await _register(home_ws, catalog, grant))['result']['registered']
            previous = None
            for turn in range(1, 3):
                await _send(home_ws, f'ask-{turn}', '@reviewer run the command')
                actions = await _wait_rpc(home_ws, 'groups.state',
                    lambda state: state['driver_status']['pending_actions'], timeout=30, room_id='linked')
                action, = actions
                assert action['approval']['command'] == g.target_model.command
                assert action['approval']['choices'] == ['once', 'deny']
                assert action['execution_generation'] == 1  # Each hosted task is a first attempt.
                assert (effects.read_text().splitlines() if effects.exists() else []) == ['ran'] * (turn - 1)
                params = {key: action[key] for key in (
                    'member_id', 'task_id', 'execution_generation', 'request_id')}
                fresh = (await _invite(target_ws, room))['grant']
                assert fresh != grant
                assert (await _register(home_ws, catalog, fresh))['result']['registered']
                grant = fresh
                if previous is not None:
                    stale = await rpc(home_ws, 'groups.approve', room_id='linked', choice='once', **previous)
                    assert stale['error']['code'] == 4001
                    assert stale['error']['message'] == 'room approval is no longer pending'
                answered = await rpc(home_ws, 'groups.approve', room_id='linked', choice='once', **params)
                assert answered.get('result', {}).get('approved'), answered
                previous = params
                replies = await _events(home_ws, 'message.member', count=turn)
                assert [r['payload']['text'] for r in replies] == ['PEER_REPLY'] * turn
                assert effects.read_text().splitlines() == ['ran'] * turn
                assert (await rpc(home_ws, 'groups.state', room_id='linked'))['result']['driver_status']['pending_actions'] == []

    with daemon(g.root, g.target, g.target_env, barrier=False) as (_, target_desc):
        with daemon(g.root, g.home, g.home_env, barrier=False) as (_, home_desc):
            asyncio.run(exercise(home_desc, target_desc))
