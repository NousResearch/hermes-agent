"""Copy retirement end to end: canonical setup, the participant's retire route and the home's live publisher."""

import asyncio
import json
import time
from contextvars import ContextVar
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_links as links
from gateway import hosted_room_peer as peer
from gateway import hosted_room_replica_retirement as retirement
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_replication as publisher
from gateway import hosted_rooms as rooms
from gateway.config import PlatformConfig
from gateway.platforms import api_server, api_server_room_grants
from gateway.session_controls import AuthorityConnection
from tests.gateway.fixtures.passive_copy import (  # noqa: F401
    API_KEY, HOME, HOME_SECRET, KEY, MEMBERS, TARGET, api, disband, enroll, invite, member, notice, prepare)
from tests.gateway.test_session_group_replication import call, gateway  # noqa: F401
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError

BASE = "/v1/group-replicas/"


async def signed_retirement(http, outgoing, *, path=None, changed=None, signature=None):
    from gateway import hosted_room_proof as proof
    path = path or BASE + 'retire'
    body = json.dumps({'notice': {**outgoing.payload(), **(changed or {})},
                       'signature': signature or outgoing.value}, separators=(',', ':')).encode()
    headers = {'Content-Type': 'application/json'}
    authorization, key, mac, wire = proof.request_proof(outgoing.proof_grant, installation_id=outgoing.target_install_id,
                                       method='POST', path=path, body=body, headers=headers)
    response = await http.post(path, data=wire, headers={**headers, 'Authorization': authorization})
    plaintext = proof.verify_response(key, mac, response.status, await response.read(),
        response.headers[proof.RESPONSE_HEADER], response.headers[proof.RESPONSE_NONCE_HEADER])
    async def text():
        return plaintext.decode()
    async def parsed():
        return json.loads(plaintext)
    return SimpleNamespace(status=response.status, text=text, json=parsed)


def home_row(source):
    return retirement.home_status(source, room_id="room")[0]


def retired_row(target):
    with rooms._transaction(target) as conn:
        row = conn.execute(f"SELECT enrollment_id,retired_at,stored_seq FROM {retirement.RETIREMENT_TABLE} "
                           "WHERE room_id='room'").fetchone()
        return dict(row) if row else None


async def connected(api, http, monkeypatch):
    """A registered copy route on the home and a publisher that holds the home's retirement key."""
    invitation = await invite(http, replication=True)
    token = invitation['grant']
    client = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key="", timeout_seconds=2,
                               proof_install_id=invitation['catalog']['installation_id'])
    probe = await asyncio.to_thread(client.probe, grant=token)
    client.proof_install_id = probe['catalog']['installation_id']
    links.save_room_link(api.source, links.make_stored_link(
        room_id="room", member_id="reviewer", target_url=str(http.make_url("/")), target_profile="default",
        grant=token, catalog=peer.GatewayRoomCatalog.from_mapping(probe["catalog"]),
        cancellation_scope_id="retirement-test", trace_id="retirement-test"))
    monkeypatch.setattr(publisher, "gateway_room_grant_secret", lambda: HOME_SECRET)
    return token, client, make_publisher(api.source, monkeypatch)


def make_publisher(source, monkeypatch):
    with monkeypatch.context() as scoped:
        scoped.setattr(rooms, "local_authority_gateway_id", lambda: HOME)
        return publisher.HostedRoomReplicationPublisher(source)


def setup(api, http):
    return prepare(api.source, endpoint=str(http.make_url("/")), enrollment_id="http-enrollment")


async def close_home(api, client, token):
    """What the canonical Disband does: revoke every member grant, forget the routes, then disband."""
    await asyncio.to_thread(client.revoke_grant, grant=token)
    rooms.delete_room_link_records(api.source, room_id="room")
    disband(api.source)


async def drain(pub):
    pub._scan(time.monotonic())
    work = [item for item in pub._routes if isinstance(item, publisher._RetirementWork)]
    assert len(work) == 1
    await asyncio.to_thread(pub._publish_retirement, work[0])


def with_profile_routes(app):
    for route in list(app.router.routes()):
        if route.resource.canonical.startswith(("/v1/room-members/", BASE)):
            app.router.add_route(route.method, "/p/{profile}" + route.resource.canonical, route.handler)
    return app


@pytest.mark.asyncio
async def test_neither_a_room_grant_nor_the_api_key_can_retire_a_copy(api, monkeypatch):
    async with TestClient(TestServer(api.app)) as http:
        token, client, _ = await connected(api, http, monkeypatch)
        enrollment = setup(api, http)
        enroll(api.target, enrollment)
        body = {"room_id": "room", "enrollment_id": enrollment["enrollment_id"]}
        for headers in ({"Authorization": "HermesRoom " + token}, {"Authorization": "Bearer " + API_KEY}, {}):
            response = await http.post(BASE + "retire", json=body, headers=headers)
            assert response.status == 403
        retirement.revoke_target_enrollment(api.target, **body)
        await close_home(api, client, token)
        with pytest.raises(PeerRunsHTTPError) as denied:
            await asyncio.to_thread(client.retire_replica, notice(api.source, enrollment))
        assert denied.value.status_code == 403
        assert retired_row(api.target) is None


@pytest.mark.asyncio
async def test_the_participant_confirms_before_copying_and_nothing_is_revealed_before_disband(api, monkeypatch):
    async with TestClient(TestServer(api.app)) as http:
        _, _, pub = await connected(api, http, monkeypatch)
        enrollment = setup(api, http)
        await asyncio.to_thread(pub._publish_one, KEY)
        assert home_row(api.source)["state"] == "prepared"
        with pytest.raises(replicas.ReplicaNotFoundError):
            replicas.copy_state(api.target, room_id="room")
        enroll(api.target, enrollment)
        await asyncio.to_thread(pub._publish_one, KEY)
        assert home_row(api.source)["state"] == "enrolled"
        assert replicas.copy_state(api.target, room_id="room")["last_seq"] == 1
        loads = []
        with pytest.raises(retirement.RetirementConflictError):
            retirement.materialize_notice(api.source, enrollment_id=enrollment["enrollment_id"], local_gateway_id=HOME,
                                          secret_loader=lambda: loads.append(True))
        assert loads == [] and retirement.pending_notice_ids(api.source, local_gateway_id=HOME) == []


@pytest.mark.asyncio
async def test_a_closed_copy_is_retired_after_a_lost_reply_and_a_publisher_restart(api, monkeypatch):
    lose = [True]

    @web.middleware
    async def lose_retire_reply(request, handler):
        response = await handler(request)
        if request.path == BASE + "retire" and response.status == 200 and lose[0]:
            lose[0] = False
            request.transport.close()
        return response

    api.app.middlewares.append(lose_retire_reply)
    async with TestClient(TestServer(api.app)) as http:
        token, client, pub = await connected(api, http, monkeypatch)
        enrollment = setup(api, http)
        enroll(api.target, enrollment)
        await asyncio.to_thread(pub._publish_one, KEY)
        await close_home(api, client, token)
        with pytest.raises(PeerRunsHTTPError):
            await asyncio.to_thread(client.probe, grant=token)
        await drain(pub)
        original = retired_row(api.target)
        assert original["stored_seq"] == 1 and home_row(api.source)["state"] == "ready"
        assert home_row(api.source)["last_error"] == "retirement_delivery_unconfirmed"
        await drain(make_publisher(api.source, monkeypatch))
        assert retired_row(api.target) == original and home_row(api.source)["state"] == "acknowledged"
        state = replicas.copy_state(api.target, room_id="room")
        assert (state["safety_status"], state["disbanded_at"]) == ("retired", None)
        with pytest.raises(retirement.RetirementConflictError):
            enroll(api.target, enrollment)


@pytest.mark.asyncio
async def test_a_late_enrollment_still_retires_a_copy_with_no_history(api, monkeypatch):
    async with TestClient(TestServer(api.app)) as http:
        token, client, pub = await connected(api, http, monkeypatch)
        enrollment = setup(api, http)
        await close_home(api, client, token)
        await drain(pub)
        assert home_row(api.source)["state"] == "ready"
        with pytest.raises(retirement.RetirementConflictError):
            setup(api, http)
        enroll(api.target, enrollment)
        await drain(make_publisher(api.source, monkeypatch))
        assert retired_row(api.target)["stored_seq"] == 0 and home_row(api.source)["state"] == "acknowledged"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    {"room_id": "other"}, {"target_install_id": "install:other"}, {"authority_gateway_id": "install:other"},
    {"enrollment_id": "other"}, {"authority_epoch": 2}, {"extra": True},
])
async def test_the_closing_value_refuses_any_other_scope_and_never_echoes(api, monkeypatch, change):
    async with TestClient(TestServer(api.app)) as http:
        token, client, _ = await connected(api, http, monkeypatch)
        enrollment = setup(api, http)
        enroll(api.target, enrollment)
        await close_home(api, client, token)
        outgoing = notice(api.source, enrollment)
        response = await signed_retirement(http, outgoing, changed=change)
        assert response.status in {400, 403, 409}
        assert outgoing.value not in await response.text()
        assert retired_row(api.target) is None


@pytest.mark.asyncio
async def test_the_closing_value_opens_nothing_else_and_only_the_installation_endpoint(api, monkeypatch):
    with_profile_routes(api.app)
    async with TestClient(TestServer(api.app)) as http:
        token, client, _ = await connected(api, http, monkeypatch)
        enrollment = setup(api, http)
        enroll(api.target, enrollment)
        await close_home(api, client, token)
        outgoing = notice(api.source, enrollment)
        headers = {"Authorization": "HermesReplicaRetirement " + outgoing.value}
        for path in ("/v1/room-members/invitations", "/v1/room-members/replica", "/v1/room-members/work-records"):
            assert (await http.post(path, json={"enrollment": enrollment}, headers=headers)).status in {401, 403}
        assert (await http.get("/v1/room-members/capabilities", headers=headers)).status in {401, 403}
        response = await signed_retirement(http, outgoing, path='/p/default' + BASE + 'retire')
        assert response.status == 400
        assert (await response.json())["error"]["code"] == "installation_endpoint_required"
        scoped = PeerRunsHTTPClient(base_url=str(http.make_url("/p/default")), api_key="")
        with pytest.raises(PeerRunsHTTPError, match="installation"):
            await asyncio.to_thread(scoped.retire_replica, outgoing)
        response = await signed_retirement(http, outgoing, signature='ed25519-v2.' + 'A' * 86)
        assert response.status == 403
        result = await asyncio.to_thread(client.retire_replica, outgoing)
        assert result["retired"] is True and outgoing.value not in json.dumps(result)


@pytest.mark.asyncio
async def test_the_canonical_setup_methods_check_owner_operator_and_profile(gateway):
    owner, member_only = gateway.owner, AuthorityConnection(gateway.authority, object(), {'user_id': 'owner'})
    stranger = AuthorityConnection(gateway.authority, object(), {'user_id': 'stranger'})
    created = await call(owner, 'groups.create', room_id='room', name='Workshop',
                         members=[MEMBERS[0], member('reviewer', target='install:participant')])
    assert created['room']['room_id'] == 'room'
    base = dict(room_id='room', target_install_id='install:participant', endpoint='http://127.0.0.1:9')
    assert await call(stranger, 'groups.replication.prepare', **base) == 'permission_denied'
    assert await call(owner, 'groups.replication.prepare', **base, extra=1) == 'invalid_params'
    assert await call(owner, 'groups.replication.prepare', **{**base, 'endpoint': 'http://peer.example'}) == 'invalid_params'
    assert await call(owner, 'groups.replication.prepare', **{**base, 'target_install_id': 'install:else'}) == (
        'replica_retirement_conflict')
    prepared = (await call(owner, 'groups.replication.prepare', **base))['enrollment']
    assert set(prepared) == {'enrollment_id', 'room_id', 'authority_gateway_id', 'authority_epoch',
                             'target_install_id', 'roster_sha256', 'commitment'}
    assert (await call(owner, 'groups.replication.prepare', **base))['enrollment'] == prepared
    for method, params in (('groups.replication.enroll', {'enrollment': prepared}),
                           ('groups.replication.revoke', {'room_id': 'room', 'enrollment_id': 'e'})):
        assert await call(member_only, method, **params) == 'permission_denied'
    # This gateway is the home here: its own enrollment names another installation.
    assert await call(owner, 'groups.replication.enroll', enrollment=prepared) == 'replica_retirement_conflict'
    assert await call(owner, 'groups.replication.revoke', room_id='room', enrollment_id='e') == (
        'replica_retirement_conflict')
    gateway.service.runtime.status = lambda: {'running': False}
    assert await call(owner, 'groups.replication.prepare', **base) == 'runtime_coordination_required'


@pytest.mark.asyncio
async def test_a_participant_operator_enrolls_and_revokes_on_the_canonical_surface(gateway, tmp_path):
    local = rooms.local_authority_gateway_id()
    home_db = tmp_path / 'other-home.db'
    rooms.create_room(home_db, room_id='room', name='Workshop', members=[MEMBERS[0], member('reviewer', target=local)],
                      authority_gateway_id=HOME)
    prepared = retirement.prepare_home_enrollment(home_db, room_id='room', target_install_id=local,
                                                  endpoint='https://participant.example', local_gateway_id=HOME,
                                                  secret=HOME_SECRET)
    enrolled = await call(gateway.owner, 'groups.replication.enroll', enrollment=prepared)
    assert enrolled == {**prepared, 'state': 'active'}
    assert await call(gateway.owner, 'groups.replication.enroll', enrollment=prepared) == enrolled
    revoked = await call(gateway.owner, 'groups.replication.revoke', room_id='room',
                         enrollment_id=prepared['enrollment_id'])
    assert revoked == {'room_id': 'room', 'enrollment_id': prepared['enrollment_id'], 'state': 'revoked'}
    assert await call(gateway.owner, 'groups.replication.enroll', enrollment=prepared) == (
        'replica_retirement_not_authorized')


@pytest.mark.asyncio
async def test_canonical_disband_retires_the_copy_through_the_running_publisher(gateway, tmp_path, monkeypatch):
    """The home's real Disband path and publisher thread, the participant's real routes and stores."""
    target_db = tmp_path / 'participant.db'
    real_id, real_db = rooms.local_authority_gateway_id, rooms.default_db_path
    identity, database = ContextVar('participant_identity', default=None), ContextVar('participant_db', default=None)
    monkeypatch.setattr(rooms, 'local_authority_gateway_id', lambda: identity.get() or real_id())
    monkeypatch.setattr(rooms, 'default_db_path', lambda: database.get() or real_db())
    observed = []

    @web.middleware
    async def as_participant(request, handler):
        tokens = identity.set(TARGET), database.set(target_db)
        profile = api_server._api_request_profile.set(request.match_info.get('profile') or 'default')
        try:
            if request.path.endswith('/grants/revoke'):
                row = home_row(gateway.db.db_path)
                loads = []
                try:
                    retirement.materialize_notice(gateway.db.db_path, enrollment_id=row['enrollment_id'],
                                                  local_gateway_id=real_id(), secret_loader=lambda: loads.append(1))
                except retirement.RetirementConflictError:
                    observed.append(('revealed_before_revoke', False, row['state'], loads))
            response = await handler(request)
            if request.path == BASE + 'retire':
                observed.append(('retire', response.status))
            return response
        finally:
            api_server._api_request_profile.reset(profile)
            identity.reset(tokens[0])
            database.reset(tokens[1])

    adapter = api_server.APIServerAdapter(PlatformConfig(enabled=True, extra={'key': API_KEY}))
    app = web.Application(middlewares=[as_participant])
    for method, path, handler in api_server_room_grants._http_routes(adapter):
        app.router.add_route(method, path, handler)
    service = gateway.service
    service.runtime = SimpleNamespace(start=lambda: None, stop=lambda **_: True, wakeup=lambda: None,
                                      status=lambda: {'running': True, 'stopping': False, 'blocked_rooms': []})

    async def until(predicate, seconds=20):
        deadline = asyncio.get_running_loop().time() + seconds
        while not predicate():
            assert asyncio.get_running_loop().time() < deadline, 'the retirement workflow did not settle'
            await asyncio.sleep(0.05)

    async with TestClient(TestServer(app)) as http:
        url = str(http.make_url('/')).rstrip('/')
        invitation = await invite(http, replication=True, home_install_id=real_id(), authority_gateway_id=real_id())
        catalog = invitation['catalog']
        pinned = {'kind': 'peer', 'peer_id': 'participant', 'installation_id': catalog['installation_id'],
                  'profile': 'default', 'capability_digest': catalog['catalog_digest']}
        await call(gateway.owner, 'groups.create', room_id='room', name='Workshop', members=[
            MEMBERS[0], {'member_id': 'reviewer', 'profile': 'default', 'handle': 'reviewer', 'target': pinned}])
        registered = await call(gateway.owner, 'groups.peer.register', room_id='room', member_id='reviewer',
                                target_url=url, target_profile='default', grant=invitation['grant'], catalog=catalog)
        assert registered['registered'], registered
        enrollment = (await call(gateway.owner, 'groups.replication.prepare', room_id='room',
                                 target_install_id=TARGET, endpoint=url))['enrollment']
        enroll(target_db, enrollment)
        try:
            service.start()
            await until(lambda: home_row(gateway.db.db_path)['state'] == 'enrolled')
            await until(lambda: service.replication.status('room')['routes'][0]['status'] == 'acked')
            disbanded = await call(gateway.owner, 'groups.disband', room_id='room')
            assert disbanded['tombstone']['disbanded_at'] is not None
            await until(lambda: home_row(gateway.db.db_path)['state'] == 'acknowledged')
        finally:
            assert service.stop(timeout=5)
        assert retired_row(target_db)['enrollment_id'] == enrollment['enrollment_id']
        state = replicas.copy_state(target_db, room_id='room')
        assert (state['safety_status'], state['disbanded_at']) == ('retired', None)
        with rooms._transaction(target_db) as conn:
            assert 'room.disbanded' not in {r[0] for r in conn.execute("SELECT kind FROM hosted_room_replica_events")}
        assert ('revealed_before_revoke', False, 'enrolled', []) in observed
        assert ('retire', 200) in observed


@pytest.mark.asyncio
@pytest.mark.parametrize('retirement_of_grant', ['expired', 'revoked'])
async def test_signed_retirement_survives_grant_retirement_and_route_removal(api, monkeypatch, retirement_of_grant):
    async with TestClient(TestServer(api.app)) as http:
        token, client, pub = await connected(api, http, monkeypatch)
        enrollment = setup(api, http)
        enroll(api.target, enrollment)
        await asyncio.to_thread(pub._publish_one, KEY)
        if retirement_of_grant == 'revoked':
            await asyncio.to_thread(client.revoke_grant, grant=token)
        else:
            now = time.time() + 7200
            monkeypatch.setattr(time, 'time', lambda: now)
        with pytest.raises(PeerRunsHTTPError):
            await asyncio.to_thread(client.probe, grant=token)
        rooms.delete_room_link_records(api.source, room_id='room')
        disband(api.source)
        restarted = make_publisher(api.source, monkeypatch)
        await drain(restarted)
        assert home_row(api.source)['state'] == 'acknowledged'
        assert retired_row(api.target)['stored_seq'] == 1
        assert not api.adapter._active_run_tasks and not api.adapter._run_statuses


@pytest.mark.asyncio
async def test_substituted_retirement_endpoint_never_gets_private_seed_or_fabricates_ack(api, monkeypatch):
    from gateway.hosted_room_proof import SCHEME
    server = TestServer(api.app)
    await server.start_server()
    http = TestClient(server)
    await http.start_server()
    token, client, pub = await connected(api, http, monkeypatch)
    enrollment = setup(api, http)
    enroll(api.target, enrollment)
    await asyncio.to_thread(pub._publish_one, KEY)
    await close_home(api, client, token)
    outgoing = notice(api.source, enrollment)
    port = server.port
    await http.close()
    received = []
    async def substituted(request):
        received.append((request.headers.get('Authorization'), await request.read()))
        return web.json_response({'retired': True, **outgoing.payload(), 'commitment': enrollment['commitment']})
    replacement = web.Application()
    replacement.router.add_post(BASE + 'retire', substituted)
    other = TestServer(replacement, port=port)
    await other.start_server()
    try:
        await drain(make_publisher(api.source, monkeypatch))
        assert home_row(api.source)['state'] == 'ready' and retired_row(api.target) is None
        assert received and received[0][0].startswith(SCHEME)
        assert token not in repr(received) and token.split('.')[1] not in repr(received)
        with rooms._transaction(api.source) as conn:
            row = conn.execute(f'SELECT * FROM {retirement.HOME_TABLE} WHERE enrollment_id=?',
                               (enrollment['enrollment_id'],)).fetchone()
            import base64
            seed = base64.urlsafe_b64encode(retirement._signing_seed(HOME_SECRET, row)).decode().rstrip('=')
        assert seed not in repr(received)
        assert b'ed25519-v2.' not in received[0][1]
    finally:
        await other.close()


@pytest.mark.asyncio
async def test_only_named_profile_route_still_proves_installation_for_exact_retirement(api, monkeypatch):
    from tests.gateway.fixtures.passive_copy import save_link
    async with TestClient(TestServer(api.app)) as http:
        _, _, pub = await connected(api, http, monkeypatch)
        await asyncio.to_thread(pub._publish_one, KEY)
        # A preserved installation copy outlives its member route/profile. Enrollment
        # must retain usable installation proof even when only a named route remains.
        named = save_link(api.source, profile='review', url=str(http.make_url('/')),
                          secret=api.adapter._room_grant_secret())
        enrollment = setup(api, http)
        enroll(api.target, enrollment)
        rooms.delete_room_link_records(api.source, room_id='room')
        disband(api.source)
        outgoing = notice(api.source, enrollment)
        assert peer.unverified_room_grant_claims(outgoing.proof_grant)['target_profile'] == 'review'
        assert outgoing.proof_grant == named.grant
        now = time.time() + 7200
        monkeypatch.setattr(time, 'time', lambda: now)
        await drain(make_publisher(api.source, monkeypatch))
        assert home_row(api.source)['state'] == 'acknowledged'
        assert retired_row(api.target)['stored_seq'] == 1
        assert not api.adapter._active_run_tasks and not api.adapter._run_statuses


@pytest.mark.asyncio
async def test_retirement_without_retained_installation_proof_stays_visibly_pending(api, monkeypatch):
    async with TestClient(TestServer(api.app)) as http:
        enrollment = setup(api, http)
        enroll(api.target, enrollment)
        disband(api.source)
        monkeypatch.setattr(publisher, 'gateway_room_grant_secret', lambda: HOME_SECRET)
        monkeypatch.setattr('tui_gateway.hosted_room_peer_http._open_roomlink_url',
                            lambda *args, **kwargs: pytest.fail('unverified retirement endpoint was contacted'))
        await drain(make_publisher(api.source, monkeypatch))
        status = home_row(api.source)
        assert status['state'] == 'ready'
        assert status['last_error'] == 'retirement_target_proof_unavailable'
        assert retired_row(api.target) is None
