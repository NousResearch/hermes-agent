"""Succession on the canonical surface and over HTTP: capabilities, codes, owners and signed peer requests."""
import time

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_custody as custody
from gateway import hosted_room_identity as identity
from gateway import hosted_room_replicas as replicas
from gateway import hosted_room_succession as succession
from gateway import hosted_rooms as rooms
from gateway.platforms import api_server_room_succession
from gateway.session_controls import AuthorityConnection
from tests.gateway.fixtures.passive_copy import HOME, MEMBERS, append
from tests.gateway.test_session_group_replication import IDENTITY, call, gateway  # noqa: F401
from tui_gateway.contracts import groups_bot_relay as contract

PEER_SECRET = b"p" * 32


def copy_with_custody(gateway, tmp_path, *, successor=True):
    """This gateway keeps a copy of HOME's room, configured with it (and one more peer) as custodians."""
    source = tmp_path / "home.db"
    rooms.create_room(source, room_id="room", name="Workshop", members=MEMBERS, authority_gateway_id=HOME)
    me = rooms.local_authority_gateway_id()
    peer_id = "install:" + "e" * 32
    for install_id, key, endpoint in ((me, identity.local_public_key(), "https://me.example.test"),
                                      (peer_id, identity.local_public_key(secret=PEER_SECRET),
                                       "https://peer.example.test")):
        custody.enroll_custodian(source, room_id="room", install_id=install_id, public_key=key, endpoint=endpoint,
                                 name="Mac mini" if install_id == me else "Home VPS", role="custodian", active=True,
                                 allowed=successor, designated=successor and install_id == me)
    custody.maintain_configuration(source, room_id="room", local_gateway_id=HOME,
                                   public_key=identity.local_public_key(secret=b"h" * 32),
                                   endpoint="https://host.example.test", name="Studio", owner_name="Dana")
    append(source, "hello", "Café")
    page = rooms.read_events(source, room_id="room")
    replicas.ingest_page(gateway.authority.db.db_path, room_id="room", room_name="Workshop", members=MEMBERS,
                         page=page)
    return peer_id


async def raw(connection, method, **params):
    return await connection.dispatch({"id": 1, "method": method, "params": params})


@pytest.mark.asyncio
async def test_succession_methods_are_canonical_and_contracted(gateway):
    methods = (await call(gateway.owner, "groups.capabilities"))["methods"]
    assert {"groups.succession.status", "groups.succession.prepare", "groups.succession.promote",
            "groups.succession.keep", "groups.succession.branch_log"} <= set(methods)
    assert await call(gateway.owner, "groups.succession.status", room_id="nowhere") == "room_not_found"


@pytest.mark.asyncio
async def test_status_reads_this_computers_copy_and_errors_name_their_parameters(gateway, tmp_path):
    copy_with_custody(gateway, tmp_path)
    status = await call(gateway.owner, "groups.succession.status", room_id="room")
    contract.GroupsSuccessionStatusResult.model_validate(status)
    assert status["state"] == "ok" and status["this_install"]["role"] == "backup"
    assert status["host"]["name"] == "Studio" and status["owner"] == {"name": "Dana"}
    elsewhere = await raw(gateway.owner, "groups.succession.prepare", room_id="room",
                          target_install_id="install:" + "e" * 32)
    assert elsewhere["error"]["code"] == 4001
    assert elsewhere["error"]["data"] == {"reason": "target_not_local",
                                          "target": {"install_id": "install:" + "e" * 32, "name": "Home VPS"}}
    # This computer's operator acts for the owner, but its own consent is still missing.
    assert await call(gateway.owner, "groups.succession.prepare", room_id="room",
                      target_install_id=rooms.local_authority_gateway_id()) == "target_not_ready"
    member_only = AuthorityConnection(gateway.authority, object(), {"user_id": "someone"})
    assert await call(member_only, "groups.succession.prepare", room_id="room",
                      target_install_id=rooms.local_authority_gateway_id()) == "not_owner"
    assert await call(gateway.owner, "groups.succession.promote", room_id="room",
                      target_install_id=rooms.local_authority_gateway_id(), preview_id="x") == "invalid_params"


def owner_row(gateway, room_id="room"):
    with rooms._transaction(gateway.authority.db.db_path) as conn:
        row = conn.execute("SELECT value FROM state_meta WHERE key=?", ("gateway.hosted.owner.v1:" + room_id,)).fetchone()
    return row[0] if row else None


@pytest.mark.asyncio
async def test_consenting_here_records_the_caller_as_the_groups_owner_on_this_computer(gateway):
    from types import SimpleNamespace
    gateway.adapter.gateway_runner = SimpleNamespace(session_authority=gateway.authority)
    await call(gateway.owner, "groups.peer.invite", **IDENTITY, successor=True)
    with rooms._transaction(gateway.authority.db.db_path) as conn:
        subject = succession.owner_subject_locked(conn, "room")
    assert subject == gateway.owner.actor.subject
    # The same owner row a host keeps: messaging and reading a copy find the owner here too.
    assert owner_row(gateway) == gateway.owner.actor.subject


@pytest.mark.asyncio
async def test_a_backup_only_consent_records_the_owner_and_a_copy_id_is_never_a_new_group(gateway, tmp_path):
    from types import SimpleNamespace
    gateway.adapter.gateway_runner = SimpleNamespace(session_authority=gateway.authority)
    await call(gateway.owner, "groups.peer.invite", **{k: v for k, v in IDENTITY.items() if k != "member_id"},
               custody_only=True)
    assert owner_row(gateway) == gateway.owner.actor.subject
    copy_with_custody(gateway, tmp_path)
    created = await call(gateway.owner, "groups.create", room_id="room", name="Workshop", members=MEMBERS)
    assert created == "permission_denied"


@pytest.mark.asyncio
async def test_a_shared_messaging_chat_never_acts_for_the_owner(gateway, tmp_path):
    from dataclasses import replace
    copy_with_custody(gateway, tmp_path)
    shared = AuthorityConnection(gateway.authority, object(), {"user_id": "dana"}, operator=True)
    shared.actor = replace(gateway.owner.actor, transport_id="messaging:shared:grant-1")
    here = rooms.local_authority_gateway_id()
    for method, params in (("groups.succession.prepare", {"target_install_id": here}),
                           ("groups.succession.move", {"target_install_id": here}),
                           ("groups.succession.move_now", {}), ("groups.succession.keep", {"install_id": here}),
                           ("groups.succession.continue_anyway", {}),
                           ("groups.succession.branch_log", {"branch_id": "epoch-1-at-1"})):
        assert await call(shared, method, room_id="room", **params) == "permission_denied", method
    assert await call(shared, "groups.succession.handover_all", reason="sleep") == "permission_denied"
    assert (await call(shared, "groups.succession.status", room_id="room"))["state"] == "ok"
    private = AuthorityConnection(gateway.authority, object(), {"user_id": "dana"}, operator=True)
    private.actor = replace(gateway.owner.actor, transport_id="messaging:private:grant-2")
    assert await call(private, "groups.succession.prepare", room_id="room",
                      target_install_id=here) == "target_not_ready"


def signed_query(room_id, requester, *, secret):
    unsigned = {"room_id": room_id, "requester_install_id": requester, "issued_at": time.time(), "nonce": "n1"}
    return {**unsigned, "signature": identity.sign(succession.QUERY, unsigned, secret=secret)}


@pytest.mark.asyncio
async def test_a_configured_computer_asks_over_http_and_anyone_else_is_refused(gateway, tmp_path):
    from types import SimpleNamespace
    gateway.adapter.gateway_runner = SimpleNamespace(session_authority=gateway.authority)
    peer_id = copy_with_custody(gateway, tmp_path)
    app = web.Application()
    for method, path, handler in api_server_room_succession.http_routes(gateway.adapter):
        app.router.add_route(method, path, handler)
    async with TestClient(TestServer(app)) as client:
        answered = await client.post("/v1/room-members/succession/query",
                                     json=signed_query("room", peer_id, secret=PEER_SECRET))
        assert answered.status == 200
        answer = await answered.json()
        assert answer["hosting"] is False and answer["authority"]["gateway_id"] == HOME
        assert answer["responder_install_id"] == rooms.local_authority_gateway_id()
        forged = await client.post("/v1/room-members/succession/query",
                                   json=signed_query("room", peer_id, secret=b"z" * 32))
        assert forged.status == 403 and (await forged.json())["error"]["code"] == "not_owner"
        stale = signed_query("room", peer_id, secret=PEER_SECRET)
        stale["issued_at"] -= 3600
        refused = await client.post("/v1/room-members/succession/query", json=stale)
        assert refused.status == 403 and (await refused.json())["error"]["code"] == "invalid_succession_request"
