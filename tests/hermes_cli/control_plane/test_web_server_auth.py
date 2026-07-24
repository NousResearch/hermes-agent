from __future__ import annotations

from httpx import ASGITransport, AsyncClient
import pytest

from hermes_cli.control_plane.nodes import NodeRegistry
from hermes_cli import web_server


@pytest.mark.asyncio
async def test_node_observation_uses_route_auth_while_operator_routes_stay_protected(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    issuance = NodeRegistry().enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )
    credential = issuance.credential
    assert credential

    previous_required = getattr(web_server.app.state, "auth_required", None)
    previous_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = True
    web_server.app.state.bound_host = None
    try:
        async with AsyncClient(
            transport=ASGITransport(app=web_server.app),
            base_url="http://test",
        ) as client:
            body = {
                "schema_version": 1,
                "report_sequence": 1,
                "observed_at": 100,
                "health_state": "healthy",
                "capabilities": {"os": "linux"},
            }

            accepted = await client.post(
                "/api/control-plane/v1/nodes/node-1/observations",
                json={**body, "credential": credential},
            )
            assert accepted.status_code == 200
            assert accepted.json()["report_sequence"] == 1

            rejected = await client.post(
                "/api/control-plane/v1/nodes/node-1/observations",
                json={**body, "report_sequence": 2, "credential": "invalid"},
            )
            assert rejected.status_code == 401
            assert rejected.json()["error"]["code"] == "invalid_node_credential"

            protected = await client.get("/api/control-plane/v1/nodes/node-1")
            assert protected.status_code == 401
            assert protected.json()["error"] == "unauthenticated"

            enrollment = await client.post(
                "/api/control-plane/v1/nodes",
                json={
                    "enrollment_key": "request-2",
                    "node_id": "node-2",
                    "role": "worker",
                    "owner": "ops",
                    "actor": "operator:alice",
                },
            )
            assert enrollment.status_code == 401
    finally:
        web_server.app.state.auth_required = previous_required
        web_server.app.state.bound_host = previous_host
