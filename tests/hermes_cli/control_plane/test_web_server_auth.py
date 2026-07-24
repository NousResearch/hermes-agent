from __future__ import annotations

import json

from httpx import ASGITransport, AsyncClient
import pytest

from hermes_cli.control_plane.nodes import NodeRegistry
from hermes_cli import web_server


async def _asgi_post(path, chunks, *, content_length=...):
    headers = [(b"content-type", b"application/json")]
    if content_length is not ...:
        headers.append((b"content-length", str(content_length).encode()))
    sent = []
    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
        for index, chunk in enumerate(chunks)
    ]

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await web_server.app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 1),
            "server": ("test", 80),
        },
        receive,
        send,
    )
    start = next(
        message for message in sent if message["type"] == "http.response.start"
    )
    body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    return start["status"], json.loads(body)


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

            unknown = await client.post(
                "/api/control-plane/v1/nodes/unknown/observations",
                json={**body, "report_sequence": 2, "credential": "invalid"},
            )
            assert unknown.status_code == rejected.status_code
            assert unknown.json() == rejected.json()

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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "limit"),
    [
        (
            "/api/control-plane/v1/nodes/node-1/authenticate",
            web_server._NODE_CREDENTIAL_BODY_LIMITS["authenticate"],
        ),
        (
            "/api/control-plane/v1/nodes/node-1/observations",
            web_server._NODE_CREDENTIAL_BODY_LIMITS["observations"],
        ),
    ],
)
async def test_public_node_routes_reject_declared_and_streamed_oversize(path, limit):
    status, body = await _asgi_post(path, [b"{}"], content_length=limit + 1)
    assert status == 413
    assert body["error"]["code"] == "payload_too_large"

    oversized = b'{"credential":"' + (b"x" * limit) + b'"}'
    for declared_length in (..., 2):
        status, body = await _asgi_post(
            path,
            [oversized[:32], oversized[32:]],
            content_length=declared_length,
        )
        assert status == 413
        assert body["error"]["code"] == "payload_too_large"


@pytest.mark.asyncio
@pytest.mark.parametrize("content_length", ["invalid", -1])
async def test_public_node_routes_reject_invalid_content_length(content_length):
    status, body = await _asgi_post(
        "/api/control-plane/v1/nodes/node-1/authenticate",
        [b"{}"],
        content_length=content_length,
    )
    assert status == 400
    assert body["error"]["code"] == "invalid_request"


@pytest.mark.asyncio
async def test_operator_route_remains_auth_protected_without_public_body_guard():
    limit = web_server._NODE_CREDENTIAL_BODY_LIMITS["observations"]
    previous_required = getattr(web_server.app.state, "auth_required", None)
    previous_host = getattr(web_server.app.state, "bound_host", None)
    web_server.app.state.auth_required = True
    web_server.app.state.bound_host = None
    try:
        status, body = await _asgi_post(
            "/api/control-plane/v1/nodes",
            [b"x" * (limit + 1)],
            content_length=limit + 1,
        )
        assert status == 401
        assert body["error"] == "unauthenticated"
    finally:
        web_server.app.state.auth_required = previous_required
        web_server.app.state.bound_host = previous_host


def test_public_node_openapi_contract_documents_limits_and_auth_failures():
    schema = web_server.app.openapi()
    for suffix in ("authenticate", "observations"):
        operation = schema["paths"][
            f"/api/control-plane/v1/nodes/{{node_id}}/{suffix}"
        ]["post"]
        assert {"401", "413"} <= operation["responses"].keys()
        assert "security" not in operation

    operator = schema["paths"]["/api/control-plane/v1/nodes"]["post"]
    assert "413" not in operator["responses"]
