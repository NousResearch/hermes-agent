"""Regression boundaries from the 2026-09-04 RoomLink reviews A and B."""

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_grant_state, hosted_room_links, hosted_rooms
from gateway.config import PlatformConfig
from gateway.hosted_room_peer import GatewayRoomCatalog, decode_room_grant, issue_room_grant
from gateway.platforms import api_server_room_grants
from gateway.platforms.api_server import APIServerAdapter
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient, PeerRunsHTTPError
from tui_gateway.hosted_room_peer_transport import PeerMemberRoute
from tui_gateway.hosted_room_service import HostedRoomService


@pytest.fixture
def target(tmp_path, monkeypatch, request):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    profile = getattr(request, "param", "reviewer")
    home = tmp_path / ".hermes" / "profiles" / profile
    home.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-key"}))
    adapter.gateway_runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False))
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    for method, path, handler in api_server_room_grants._http_routes(adapter):
        app.router.add_route(method, "/p/{profile}" + path, handler)
    app.router.add_post("/p/{profile}/v1/runs", adapter._handle_runs)
    stores = hosted_room_grant_state.grant_state_db_paths()
    return SimpleNamespace(app=app, adapter=adapter, stores=stores, home=home, profile=profile)


def grant(target, *, issued_at=None):
    return issue_room_grant(
        target.adapter._room_grant_secret(), grant_id="retry-id", room_id="room-1",
        home_install_id="install-home", authority_gateway_id="install-home",
        authority_epoch=1, member_id="reviewer", target_profile="reviewer",
        target_install_id=hosted_rooms.local_authority_gateway_id(),
        issued_at=issued_at or time.time() - 10, ttl_seconds=3600,
    )


def claims(target, token):
    return decode_room_grant(target.adapter._room_grant_secret(), token, permission="status")


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["reviewer", "custom"], indirect=True)
async def test_named_single_profile_invitation_through_admission_and_revocation(target, monkeypatch):
    from gateway.hosted_room_driver import TaskIdentity
    from tui_gateway.hosted_room_driver import HostedRoomBinding
    from tui_gateway.hosted_room_peer_transport import build_member_dispatch

    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "Reviewed."}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(target.adapter, "_create_agent", lambda **kwargs: agent)
    async with TestClient(TestServer(target.app)) as http:
        prefix = f"/p/{target.profile}/v1/room-members"
        invitation = await http.post(prefix + "/invitations", headers={"Authorization": "Bearer test-key"}, json={
            "room_id": "room-1", "home_install_id": "install-home",
            "authority_gateway_id": "install-home", "authority_epoch": 1, "member_id": "reviewer",
        })
        body = await invitation.json()
        assert invitation.status == 201, body
        assert body["target_profile"] == target.profile
        token = body["grant"]
        auth = {"Authorization": f"HermesRoom {token}"}
        assert (await http.get(prefix + "/capabilities", headers=auth)).status == 200
        assert (await http.get("/p/foreign/v1/room-members/capabilities", headers=auth)).status == 404
        refreshed = await http.post(prefix + "/grants/refresh", headers=auth, json={})
        assert refreshed.status == 200, await refreshed.text()
        renewed = (await refreshed.json())["grant"]
        catalog = GatewayRoomCatalog.from_mapping(body["catalog"])
        route = PeerMemberRoute(
            home_install_id="install-home", member_id="reviewer",
            target_install_id=catalog.installation_id, target_profile=target.profile,
            capability_digest=catalog.catalog_digest,
            execution_policy_digest=catalog.execution_policy.policy_digest,
            cancellation_scope_id="cancel-1", trace_id="trace-1", grant=renewed,
        )
        task = TaskIdentity("room-1", "task-1", "thread-1", "turn-1")
        dispatch = build_member_dispatch(
            binding=HostedRoomBinding("room-1", "install-home", 1), route=route,
            room_id="room-1", task_id=task.task_id, target_profile=target.profile,
            execution_generation=1, source_event_seq=1, prompt="Review.", trace_id="trace-1",
        )
        started = await http.post(f"/p/{target.profile}/v1/runs", json={
            "input": "Review.", "hosted_room_dispatch": dispatch.as_mapping(),
        }, headers={"Authorization": f"HermesRoom {renewed}", "Idempotency-Key": "room:task-1:1"})
        assert started.status == 202, await started.text()
        pending = list(target.adapter._active_run_tasks.values())
        if pending:
            await asyncio.wait_for(asyncio.gather(*pending), timeout=10)
        assert agent.run_conversation.called
        assert (await http.post(prefix + "/grants/revoke-exact", headers=auth, json={})).status == 200
        renewed_auth = {"Authorization": f"HermesRoom {renewed}"}
        assert (await http.get(prefix + "/capabilities", headers=renewed_auth)).status == 200
        assert (await http.post(prefix + "/grants/revoke", headers=renewed_auth, json={})).status == 200
        assert (await http.get(prefix + "/capabilities", headers=renewed_auth)).status == 403


@pytest.mark.asyncio
@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("failed_store", [0, 1])
async def test_partial_revoke_keeps_home_retirement_retryable(target, tmp_path, monkeypatch, exact, failed_store):
    # Isolate storage classification from the independently tested profile-resolution bug.
    import contextvars
    from gateway.platforms import api_server
    monkeypatch.setattr(api_server, "_api_request_profile", contextvars.ContextVar("review-profile", default="reviewer"))
    target.app._middlewares.clear()
    token = grant(target)
    decoded = claims(target, token)
    hosted_room_grant_state.reserve_grant_state(target.stores, claims=decoded, expires_at=decoded["expires_at"])
    method = "revoke_room_grant_id" if exact else "revoke_room_grant_scope"
    original = getattr(hosted_rooms, method)
    fail = True
    def revoke(db, **kwargs):
        if fail and db == target.stores[failed_store]:
            raise OSError("injected enforcing-store failure")
        return original(db, **kwargs)
    monkeypatch.setattr(hosted_rooms, method, revoke)
    async with TestClient(TestServer(target.app)) as http:
        peer = PeerRunsHTTPClient(base_url=str(http.make_url("")), api_key="", target_profile="reviewer")
        home = HostedRoomService(SimpleNamespace(), db_path=tmp_path / "home.db")
        catalog = GatewayRoomCatalog.from_mapping(api_server_room_grants._local_room_catalog(
            target.adapter, "reviewer", decoded["target_install_id"])[1])
        route = PeerMemberRoute(
            home_install_id="install-home", member_id="reviewer", target_install_id=catalog.installation_id,
            target_profile="reviewer", capability_digest=catalog.catalog_digest,
            execution_policy_digest=catalog.execution_policy.policy_digest, grant=token,
            cancellation_scope_id="cancel-1", trace_id="trace-1",
        )
        home.register_peer_route(room_id="room-1", member_id="reviewer", route=route,
                                 client=peer, target_url=peer.base_url, catalog=catalog)
        operation = (lambda: peer.revoke_grant_exact(grant=token)) if exact else (lambda: home.revoke_room_routes("room-1"))
        with pytest.raises(PeerRunsHTTPError) as error:
            await asyncio.to_thread(operation)
        assert error.value.status_code == 503
        assert hosted_room_links.load_room_link(home.db_path, room_id="room-1", member_id="reviewer") is not None
        assert not hosted_rooms.room_grant_is_revoked(target.stores[failed_store], claims=decoded)
        assert hosted_rooms.room_grant_is_revoked(target.stores[1 - failed_store], claims=decoded)
        fail = False
        await asyncio.to_thread(operation)
        for db in target.stores:
            assert hosted_rooms.room_grant_is_revoked(db, claims=decoded)
        if not exact:
            assert hosted_room_links.load_room_link(home.db_path, room_id="room-1", member_id="reviewer") is None


@pytest.mark.asyncio
async def test_exact_http_revoke_isolates_same_id_signed_tokens(target, monkeypatch):
    import contextvars
    from gateway.platforms import api_server
    monkeypatch.setattr(api_server, "_api_request_profile", contextvars.ContextVar("review-profile", default="reviewer"))
    target.app._middlewares.clear()
    old, new = grant(target, issued_at=time.time() - 20), grant(target)
    old_claims, new_claims = claims(target, old), claims(target, new)
    hosted_room_grant_state.reserve_grant_state(target.stores, claims=new_claims, expires_at=new_claims["expires_at"])
    async with TestClient(TestServer(target.app)) as http:
        response = await http.post("/p/reviewer/v1/room-members/grants/revoke-exact", json={},
                                   headers={"Authorization": f"HermesRoom {old}"})
        assert response.status == 200
        for db in target.stores:
            assert hosted_rooms.room_grant_is_revoked(db, claims=old_claims)
            assert not hosted_rooms.room_grant_is_revoked(db, claims=new_claims)
        response = await http.get("/p/reviewer/v1/room-members/capabilities",
                                  headers={"Authorization": f"HermesRoom {new}"})
        assert response.status == 200
        response = await http.get("/p/reviewer/v1/room-members/capabilities",
                                  headers={"Authorization": f"HermesRoom {old}="})
        assert response.status == 403


def test_decode_survives_scheduled_compat_removal(target, monkeypatch):
    from gateway import hosted_room_peer
    token = grant(target)
    manifest = json.loads((Path(__file__).resolve().parents[2] / "compat_manifest.json").read_text())
    for entry in manifest["entries"]:
        if entry["facade"] == hosted_room_peer.__name__:
            monkeypatch.delattr(hosted_room_peer, entry["name"], raising=False)
    assert decode_room_grant(target.adapter._room_grant_secret(), token, permission="status")["grant_id"] == "retry-id"
