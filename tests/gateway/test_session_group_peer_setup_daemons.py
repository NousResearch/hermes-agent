"""Two real gateways recover interrupted create/invite/register setup without durable client state."""
import asyncio
import json

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.gateway.test_session_group_peer_daemons import (
    Model, _create, _events, _grant_status, _invite, _last_user_text, _revoked, _room_link, _send, _wait_rpc, gateways as gateways_fixture)

gateways = gateways_fixture

# This test signals only the child handles returned by its isolated daemon fixture.
pytestmark = pytest.mark.live_system_guard_bypass

# What Desktop asks for: the longest dispatch lifetime and renewal horizon a member gateway grants.
_LIFETIMES = {'ttl_seconds': 86400, 'status_ttl_seconds': 30 * 86400}


class _ToolModel(Model):
    """One real terminal call held by an event, then the usual model reply."""

    def do_GET(self):
        if self.path != '/hold':
            self.send_error(404)
            return
        held, release = self.server.gates['tool']
        held.set()
        release.wait(60)
        self.send_response(200)
        self.send_header('Content-Length', '2')
        self.end_headers()
        self.wfile.write(b'ok')

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('messages'):
            self.server.requests.append(body)
        finished = not body.get('messages') or any(m.get('role') == 'tool' for m in body['messages'])
        message = {'role': 'assistant', 'content': self.server.reply if finished else None}
        if not finished:
            message['tool_calls'] = [{'index': 0, 'id': 'held-tool', 'type': 'function', 'function': {
                'name': 'terminal', 'arguments': json.dumps({
                    'command': f'curl -s http://127.0.0.1:{self.server.server_port}/hold', 'timeout': 60})}}]
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


def test_a_setup_stopped_between_its_steps_leaves_a_visible_member_that_running_it_again_seats(gateways):
    g = gateways('tool', handler=_ToolModel)
    config = json.loads((g.target / 'config.yaml').read_text())
    config['platform_toolsets']['api_server'] = ['terminal']
    (g.target / 'config.yaml').write_text(json.dumps(config))
    setup = {}

    def register(home_ws, grant):
        return rpc(home_ws, 'groups.peer.register', room_id='linked', member_id='reviewer',
                   target_url=setup['url'], target_profile='default', grant=grant, catalog=setup['catalog'])

    async def routes(home_ws):
        return (await rpc(home_ws, 'groups.state', room_id='linked'))['result']['driver_status']['peer_routes']

    async def create_then_invite(home_desc, target_desc):
        async with websocket(g.home, home_desc) as home_ws, websocket(g.target, target_desc) as target_ws:
            link = await _room_link(target_ws)
            catalog = link['catalog']
            room = await _create(home_ws, catalog, peer_id=catalog['installation_id'])
            setup.update(room=room, catalog=catalog, url=link['endpoint']['url'])
            # Desktop quits with the grant in hand: nothing registers it, and nothing else holds it.
            setup['orphan'] = (await _invite(target_ws, setup['room'], **_LIFETIMES))['grant']

    async def run_again_with_the_member_gateway_away(home_desc, target_desc, target_proc):
        async with websocket(g.home, home_desc) as home_ws:
            # After the home restarts, the unjoined member still fails visibly.
            assert await routes(home_ws) == []
            await _send(home_ws, 'early', '@reviewer EARLY')
            failed, = await _events(home_ws, 'turn.failed')
            assert failed['payload']['error'] == 'peer room route is unavailable', failed
            # The orphaned grant does not block inviting the same member of the same room again.
            async with websocket(g.target, target_desc) as target_ws:
                setup['grant'] = (await _invite(target_ws, setup['room'], **_LIFETIMES))['grant']
            assert setup['grant'] != setup['orphan']
            # An unreachable member stays unjoined; refusal must not echo its grant.
            target_proc.kill()
            await asyncio.to_thread(target_proc.wait, 10)
            refused = await register(home_ws, setup['grant'])
            assert refused['error']['message'] == 'peer_unreachable', refused
            assert setup['grant'] not in str(refused)
            assert await routes(home_ws) == []

    async def register_after_the_member_gateway_restarts(home_desc):
        async with websocket(g.home, home_desc) as home_ws:
            # The grant outlives its gateway's restart: registering it again seats the member.
            registered = await register(home_ws, setup['grant'])
            assert registered['result']['registered'], registered
            assert await routes(home_ws) == [{'room_id': 'linked', 'member_id': 'reviewer', 'status': 'ready'}]
            await _send(home_ws, 'ask', '@reviewer PEER_PROOF')
            held, release = g.target_model.gates['tool']
            assert await asyncio.to_thread(held.wait, 30)

            activity = await _wait_rpc(home_ws, 'groups.state',
                lambda state: next((a for a in state['driver_status']['member_activity']
                                    if a.get('tool') == 'terminal'), None), timeout=15, room_id='linked')
            assert activity['member_id'] == 'reviewer' and activity['progress_at'] > 0
            assert (activity['thread_id'], activity['round_index']) == ('thread', 0)
            assert activity['started_at'] > 0
            release.set()
            reply, = await _events(home_ws, 'message.member')
            assert (reply['payload']['member_id'], reply['payload']['text']) == ('reviewer', 'PEER_REPLY'), reply
            assert reply['actor']['connection_id'] == setup['catalog']['installation_id'], reply
            assert not any('EARLY' in _last_user_text(r) for r in g.target_model.requests)
            initial = [r for r in g.target_model.requests if not any(m.get('role') == 'tool' for m in r['messages'])]
            assert sum('PEER_PROOF' in _last_user_text(r) for r in initial) == 1
            assert (await rpc(home_ws, 'groups.state', room_id='linked'))['result']['driver_status']['member_activity'] == []
            # Disband revokes the whole member scope, including the unregistered orphan.
            assert (await asyncio.to_thread(_grant_status, setup['url'], setup['orphan']))[0] == 200
            disbanded = await rpc(home_ws, 'groups.disband', room_id='linked')
            assert 'tombstone' in disbanded['result'], disbanded
            # The home owes the revoke and its worker pays it: wait for it, bounded.
            for grant in (setup['orphan'], setup['grant']):
                await _revoked(setup['url'], grant)

    with daemon(g.root, g.target, g.target_env, barrier=False) as (target_proc, target_desc):
        with daemon(g.root, g.home, g.home_env, barrier=False) as (home_proc, home_desc):
            asyncio.run(create_then_invite(home_desc, target_desc))
            home_proc.kill()
        with daemon(g.root, g.home, g.home_env, barrier=False) as (_, home_desc):
            asyncio.run(run_again_with_the_member_gateway_away(home_desc, target_desc, target_proc))
            with daemon(g.root, g.target, g.target_env, barrier=False):
                asyncio.run(register_after_the_member_gateway_restarts(home_desc))
