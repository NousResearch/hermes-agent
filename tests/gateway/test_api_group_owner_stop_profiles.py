"""A default-profile API key must not gain control of named-profile group work."""

import asyncio
from unittest.mock import MagicMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server_run_scope import room_run_scope_key
from tests.gateway.test_api_group_owner_stop import STOP, command, dispatch, participation
from tests.gateway.test_api_server_runs import _make_slow_agent
from tests.gateway.test_multiplex_toolsets_profile_isolation import (
    LOKAJ_KEY, OWNER_KEY, _make_adapter, _make_app, hermes_root as hermes_root,
)


@pytest.mark.asyncio
async def test_real_named_profile_admission_does_not_grant_default_connection_owner_control(hermes_root, monkeypatch):
    value = _make_adapter(multiplex=True)
    agent, ready, interrupted = _make_slow_agent()
    monkeypatch.setattr(value, "_create_agent", MagicMock(return_value=agent))
    root_headers = {"Authorization": f"Bearer {OWNER_KEY}"}
    named_headers = {"Authorization": f"Bearer {LOKAJ_KEY}"}
    try:
        async with TestClient(TestServer(_make_app(value))) as client:
            assert (await client.get("/p/lokaj/v1/toolsets", headers=root_headers)).status == 401
            response = await client.post("/p/lokaj/v1/room-members/invitations", headers=named_headers, json={
                "room_id": "named-room", "home_install_id": "install:remote-home",
                "authority_gateway_id": "install:remote-home", "authority_epoch": 1, "member_id": "writer",
            })
            invitation = await response.json()
            assert response.status == 201, invitation
            payload = dispatch(invitation, room_id="named-room")
            payload.update(home_install_id="install:remote-home", authority_gateway_id="install:remote-home",
                           target_profile="lokaj")
            headers = {"Authorization": "HermesRoom " + invitation["grant"], "Idempotency-Key": "room:task-1:1"}
            accepted = await client.post("/p/lokaj/v1/runs", headers=headers,
                json={"input": payload["prompt"], "hosted_room_dispatch": payload})
            accepted_body = await accepted.json()
            assert accepted.status == 202, accepted_body
            assert await asyncio.to_thread(ready.wait, 2)
            run_id = accepted_body["run_id"]
            assert value._run_idempotency_store.path == hermes_root / "runs_idempotency.db"
            assert (await client.post(f"/p/lokaj/v1/runs/{run_id}/stop", headers=root_headers)).status == 401
            assert (await client.post(f"/v1/runs/{run_id}/stop", headers=root_headers)).status == 404
            stopped = await client.post(STOP, headers=root_headers, json=command(participation(payload)))
            body = await stopped.json()
            assert stopped.status == 403, body
            assert body["error"]["code"] == "participant_owner_required"
            assert not value._run_idempotency_store.is_scope_frozen(room_run_scope_key(participation(payload)))
            assert not interrupted.is_set()
    finally:
        agent.interrupt()
        tasks = list(value._active_run_tasks.values())
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)
        value._run_idempotency_store.close()
