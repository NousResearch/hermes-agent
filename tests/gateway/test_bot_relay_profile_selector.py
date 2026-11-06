"""A Desktop shared socket's relay verbs carry a routed profile selector."""
import pytest
from tools import bot_relay

pytest_plugins = ['tests.gateway.test_bot_roster_fleet']


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['default', 'alpha'])
async def test_shared_host_socket_routes_mailboxes_without_rejecting_profile(fleet, name):
    desktop = fleet.connect('default', scope='host')
    home = fleet.homes[name]
    result = await fleet.call(desktop, 'bot_relay.roster.sync', profile=name,
                              agents=[{'connection_id': 'pro', 'profile': 'cowork'}])
    assert result['count'] == 1
    assert bot_relay.read_remote_roster(home)[0]['profile'] == 'cowork'
    if name != 'default':
        assert bot_relay.read_remote_roster(fleet.homes['default']) == []
    assert await fleet.call(desktop, 'bot_relay.outbox.drain', profile=name) == {'envelopes': []}
    key = 'a' * 32
    assert await fleet.call(desktop, 'bot_relay.reply', profile=name, id=key, reply='fixture answer') == {'ok': True}
    assert (home/'bot_relay/replies'/f'{key}.json').exists()


@pytest.mark.asyncio
async def test_profile_ticket_cannot_sync_a_sibling_roster(fleet):
    from gateway.session_authorities import owner_scope
    connection = fleet.connect('alpha')
    with owner_scope(connection.authority):
        reply = await connection.dispatch({'id': 1, 'method': 'bot_relay.roster.sync',
            'params': {'profile': 'beta', 'agents': [{'connection_id': 'pro', 'profile': 'cowork'}]}})
    assert reply['error']['message'] == 'profile_mismatch'
    assert bot_relay.read_remote_roster(fleet.homes['beta']) == []
