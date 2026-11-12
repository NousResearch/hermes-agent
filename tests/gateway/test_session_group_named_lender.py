"""A gateway lends ANY profile it serves to another gateway's Group Chat, not only ``default``.

The named profile is lent through the shared listener's ``/p/<profile>`` mirror: its grant answers
there and nowhere else, and a granted run executes in that profile's own home, config, secrets and
``state.db``.
"""
import asyncio
import hashlib
import sqlite3
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer, TestClient

from agent import secret_scope
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from gateway.session_group_peers import dispatch_target, room_link
from hermes_agent.hermes_state import SessionDB
from tests.waiting import async_wait_until

ROOM = dict(room_id='lent', home_install_id='install:home', authority_gateway_id='install:home',
            authority_epoch=1, member_id='reviewer')


class _Authorities:
    """The served-profile registry the shared listener reads: one store per served home."""

    def __init__(self, homes):
        self.dbs = {str(home.resolve()): SessionDB(home / 'state.db') for home in homes}

    def for_home(self, home):
        db = self.dbs.get(str(home.resolve()))
        return None if db is None else SimpleNamespace(db=db)

    def active(self):
        return None


@pytest.fixture
def lender(tmp_path, monkeypatch):
    """A multiplexed lender: default + profiles/helper, each with its own config, .env and state.db."""
    from gateway.run import _profile_runtime_scope
    root = tmp_path / 'hermes'
    helper = root / 'profiles' / 'helper'
    helper.mkdir(parents=True)
    (root / 'config.yaml').write_text('approvals:\n  mode: manual\nagent:\n  max_turns: 90\n')
    (root / '.env').write_text('ROOMLINK_PROBE_KEY=default-secret\n')
    (helper / 'config.yaml').write_text('approvals:\n  mode: manual\nagent:\n  max_turns: 17\n')
    (helper / '.env').write_text('ROOMLINK_PROBE_KEY=helper-secret\n')
    monkeypatch.setenv('HERMES_HOME', str(root))
    monkeypatch.delenv('HERMES_ROOM_LINK_URL', raising=False)
    monkeypatch.delenv('ROOMLINK_PROBE_KEY', raising=False)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={'key': 'lender-gateway-api-key'}))
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    authorities = _Authorities((root, helper))
    adapter.gateway_runner = SimpleNamespace(
        config=GatewayConfig(multiplex_profiles=True), session_authorities=authorities,
        adapters={Platform.API_SERVER: adapter})
    previous = secret_scope.is_multiplex_active()
    secret_scope.set_multiplex_active(True)

    def authority(home):
        return SimpleNamespace(profile_id=str(home), runner=adapter.gateway_runner)

    def invite(home, **room):
        # As dispatch_group_control runs it: inside the connection's profile scope.
        with _profile_runtime_scope(home):
            return dispatch_target(authority(home), 'groups.peer.invite', {**ROOM, **room})

    def link(home):
        with _profile_runtime_scope(home):
            return room_link(authority(home))

    yield SimpleNamespace(root=root, helper=helper, adapter=adapter, invite=invite, link=link,
                          authorities=authorities)
    secret_scope.set_multiplex_active(previous)
    adapter._run_idempotency_store.close()
    for db in authorities.dbs.values():
        db.close()


def _app(adapter):
    """The shared listener as ``connect()`` builds it: the prefix middleware + /p/<profile>/ mirrors."""
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    for method, path, handler in adapter._http_route_table():
        app.router.add_route(method, path, handler)
        app.router.add_route(method, f'/p/{{profile}}{path}', handler)
    return app


def test_a_named_profile_is_lent_through_its_own_mirror_and_mints_its_own_grants(lender, monkeypatch):
    from gateway.hosted_room_peer import decode_room_grant, gateway_room_grant_secret
    monkeypatch.setenv('HERMES_ROOM_LINK_URL', 'https://lender.example.test/hermes/')
    link = lender.link(lender.helper)
    assert link['enabled'] is True and link['profile'] == 'helper', link
    catalog = link['catalog']
    assert catalog['execution_policy']['target_profile'] == 'helper'
    # The catalog is the helper's own config, not the launch profile's.
    assert catalog['execution_policy']['max_iterations'] == 17
    assert catalog['endpoint'] == {'available': True, 'transport_security': 'tls',
                                   'url': 'https://lender.example.test/hermes/p/helper'}
    assert lender.link(lender.root)['catalog']['endpoint']['url'] == 'https://lender.example.test/hermes'

    invitation = lender.invite(lender.helper)
    assert invitation['target_profile'] == 'helper' and invitation['catalog'] == catalog
    claims = decode_room_grant(gateway_room_grant_secret(), invitation['grant'], permission='status')
    assert claims['target_profile'] == 'helper'
    assert claims['execution_policy_digest'] == catalog['execution_policy']['policy_digest']

    # A profile that advertises its own endpoint keeps it as configured.
    monkeypatch.delenv('HERMES_ROOM_LINK_URL')
    (lender.helper / 'config.yaml').write_text(
        'approvals:\n  mode: manual\ngateway:\n  room_link_url: https://helper.example.test/link\n')
    assert lender.link(lender.helper)['endpoint']['url'] == 'https://helper.example.test/link'


def test_a_single_profile_gateway_never_advertises_the_root_gateways_listener(lender, monkeypatch):
    # `hermes -p helper gateway run`: its own process. The root gateway's listener neither serves
    # helper nor holds this process's grant secret, so its /p/helper mirror would never answer.
    secret_scope.set_multiplex_active(False)
    monkeypatch.setenv('HERMES_HOME', str(lender.helper))
    (lender.root / 'config.yaml').write_text(
        'approvals:\n  mode: manual\ngateway:\n  room_link_url: https://root.example.test/link\n')
    assert lender.link(lender.helper) == {'enabled': False, 'reason': 'endpoint_required'}


@pytest.mark.asyncio
async def test_a_granted_run_executes_in_the_lent_profile_and_its_grant_answers_nowhere_else(lender, monkeypatch):
    from gateway.hosted_room_peer import PROTOCOL_VERSION
    from hermes_constants import get_hermes_home
    from hermes_cli.config import load_config
    seen = []

    class Agent:
        def run_conversation(self, **kwargs):
            seen.append({'home': get_hermes_home(), 'secret': secret_scope.get_secret('ROOMLINK_PROBE_KEY'),
                         'max_turns': load_config()['agent']['max_turns'], 'prompt': kwargs['user_message']})
            return {'final_response': 'helper reply'}

    monkeypatch.setattr(lender.adapter, '_create_agent', lambda **kwargs: Agent())
    async with TestClient(TestServer(_app(lender.adapter))) as client:
        monkeypatch.setenv('HERMES_ROOM_LINK_URL', str(client.make_url('')).rstrip('/'))
        catalog = lender.link(lender.helper)['catalog']
        grant = lender.invite(lender.helper)['grant']
        default_grant = lender.invite(lender.root, member_id='host')['grant']
        auth = {'Authorization': f'HermesRoom {grant}'}

        async def get(path, headers=auth):
            response = await client.get(path, headers=headers)
            return response.status, await response.json()

        status, body = await get('/p/helper/v1/room-members/capabilities')
        assert status == 200 and body['target_profile'] == 'helper' and body['catalog'] == catalog, body
        # No cross-profile use: the helper's grant on the default route, the default's on the helper's.
        assert (await get('/v1/room-members/capabilities'))[1]['error']['code'] == 'invalid_room_grant'
        assert (await get('/p/helper/v1/room-members/capabilities',
                          {'Authorization': f'HermesRoom {default_grant}'}))[1]['error']['code'] == 'invalid_room_grant'

        prompt = 'review the helper side'
        dispatch = {**ROOM, 'protocol_version': PROTOCOL_VERSION, 'target_install_id': catalog['installation_id'],
                    'target_profile': 'helper', 'task_id': 'task-1', 'execution_generation': 1,
                    'source_event_seq': 1, 'cancellation_scope_id': 'cancel-1', 'prompt': prompt,
                    'prompt_digest': hashlib.sha256(prompt.encode()).hexdigest(),
                    'capability_digest': catalog['catalog_digest'],
                    'execution_policy_digest': catalog['execution_policy']['policy_digest'], 'trace_id': 'trace-1'}
        headers = {**auth, 'Idempotency-Key': 'room:task-1:1'}
        payload = {'input': prompt, 'hosted_room_dispatch': dispatch}
        refused = await client.post('/v1/runs', json=payload, headers=headers)
        assert refused.status == 401 and (await refused.json())['error']['code'] == 'invalid_room_grant'
        accepted = await client.post('/p/helper/v1/runs', json=payload, headers=headers)
        assert accepted.status == 202, await accepted.text()
        run_id = (await accepted.json())['run_id']
        await async_wait_until(lambda: lender.adapter._run_statuses.get(run_id, {}).get('status') == 'completed',
                               what='lent run completes')
        run = (await get(f'/p/helper/v1/runs/{run_id}'))[1]
        assert run['status'] == 'completed' and run['output'] == 'helper reply', run
        assert (await get(f'/v1/runs/{run_id}'))[0] == 401

    assert seen == [{'home': lender.helper, 'secret': 'helper-secret', 'max_turns': 17, 'prompt': prompt}]
    # The member's hidden group session lives in the helper's own state.db, not the default's.
    def rooms(home):
        with sqlite3.connect(home / 'state.db') as conn:
            return conn.execute("SELECT title FROM sessions WHERE source='bot_room'").fetchall()
    assert rooms(lender.helper) == [('Group: lent',)] and rooms(lender.root) == []
