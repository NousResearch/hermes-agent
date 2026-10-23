"""Two real gateways recover interrupted create/invite/register setup without durable client state."""
import asyncio

import pytest

from tests.gateway.fixtures.local_recovery_probe import daemon, rpc, websocket
from tests.gateway.test_session_group_peer_daemons import (
    _create, _events, _grant_status, _invite, _last_user_text, _revoked, _room_link, _send, gateways as gateways_fixture)

gateways = gateways_fixture

# This test signals only the child handles returned by its isolated daemon fixture.
pytestmark = pytest.mark.live_system_guard_bypass

# What Desktop asks for: the longest dispatch lifetime and renewal horizon a member gateway grants.
_LIFETIMES = {'ttl_seconds': 86400, 'status_ttl_seconds': 30 * 86400}


def test_a_setup_stopped_between_its_steps_leaves_a_visible_member_that_running_it_again_seats(gateways):
    g = gateways()
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
            reply, = await _events(home_ws, 'message.member')
            assert (reply['payload']['member_id'], reply['payload']['text']) == ('reviewer', 'PEER_REPLY'), reply
            assert reply['actor']['connection_id'] == setup['catalog']['installation_id'], reply
            assert not any('EARLY' in _last_user_text(r) for r in g.target_model.requests)
            assert sum('PEER_PROOF' in _last_user_text(r) for r in g.target_model.requests) == 1
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
