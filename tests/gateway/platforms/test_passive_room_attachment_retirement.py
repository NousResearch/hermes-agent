"""Inspection/replication credentials cannot retire normal staged input."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_room_peer as peer
from gateway.platforms import api_server_room_attachments as attachments
from tests.gateway.test_api_server_room_replicas import HOME, TARGET, KEY, setup  # noqa: F401
from tests.gateway.platforms.test_api_server_room_attachments import _dispatch, _manifest
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPClient


@pytest.mark.asyncio
@pytest.mark.parametrize("surface,work_records", [
    ("http", False), ("http", True), ("rpc", False), ("rpc", True), ("status", False),
])
async def test_inspection_grant_cannot_retire_normal_staged_input(setup, surface, work_records):
    _, target, app = setup
    scope: dict[str, Any] = dict(room_id="room", home_install_id=HOME, authority_gateway_id=HOME,
                                 authority_epoch=1, member_id="reviewer")
    async with TestClient(TestServer(app)) as http:
        owner = PeerRunsHTTPClient(base_url=str(http.make_url("/")), api_key=KEY, timeout_seconds=2)
        normal = await asyncio.to_thread(owner.issue_invitation, **scope, grant_id="normal")
        claims = peer.decode_room_grant(peer.gateway_room_grant_secret(), normal["grant"], permission="status")
        manifest = _manifest()
        dispatch = _dispatch(manifest, **scope, target_install_id=TARGET,
                             capability_digest=normal["catalog"]["catalog_digest"],
                             execution_policy_digest=claims["execution_policy_digest"])
        headers = {"Authorization": f"HermesRoom {normal['grant']}"}
        staged = await http.post("/v1/room-members/attachments", headers=headers,
                                json={"hosted_room_dispatch": dispatch.as_mapping(), "attachments": manifest})
        assert staged.status == 201, await staged.text()
        uploaded = await http.put(f"/v1/room-members/attachments/task-1/1/{manifest[0]['attachment_id']}",
                                  headers=headers, data=b"hello")
        assert uploaded.status == 201, await uploaded.text()
        spool = attachments._spool(str(target))
        staged_path = Path(spool.materialize(dispatch)[0]["path"])
        assert staged_path.is_relative_to(target.parent)
        assert staged_path.read_bytes() == b"hello"
        params = dict(**scope, grant_id="inspection", replication=True, passive_only=True,
                      work_records=work_records)
        if surface == "http":
            inspection = await asyncio.to_thread(owner.issue_invitation, **params)
            token = inspection["grant"]
        elif surface == "rpc":
            from tui_gateway import server
            reply = server._methods["groups.peer.invite"](1, params)
            assert "error" not in reply, reply
            token = reply["result"]["grant"]
        else:
            token = peer.issue_room_grant(
                peer.gateway_room_grant_secret(), **scope, grant_id="inspection",
                target_install_id=TARGET, target_profile="default", permissions=("status",),
                execution_policy_digest=claims["execution_policy_digest"])
        inspection_claims = peer.decode_room_grant(peer.gateway_room_grant_secret(), token, permission="status")
        assert "attachment.stage" not in inspection_claims["permissions"]
        response = await http.delete("/v1/room-members/attachments/task-1/1",
                                     headers={"Authorization": f"HermesRoom {token}"})
        assert response.status == 401, await response.json()
        assert staged_path.read_bytes() == b"hello"
        spool.prepare(dispatch, manifest)  # The same attempt has not been fenced off.
        allowed = await http.delete("/v1/room-members/attachments/task-1/1", headers=headers)
        assert allowed.status == 200, await allowed.json()
        assert not staged_path.exists()
