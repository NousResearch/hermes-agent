"""HTTP authentication boundaries exercised through the real MCP/ASGI app."""

from contextlib import asynccontextmanager
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

import mcp_serve


@asynccontextmanager
async def _client(config):
    app = mcp_serve.create_streamable_http_app(
        mcp_serve.create_mcp_server(), auth_config=config, json_response=True,
    )
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8666",
            headers={"Accept": "application/json, text/event-stream"},
        ) as client:
            yield client


def _initialize():
    return {
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "auth-boundary-test", "version": "1"},
        },
    }


def test_auth_path_mismatch_fails_before_serving():
    with pytest.raises(ValueError, match="path"):
        mcp_serve.create_streamable_http_app(
            mcp_serve.create_mcp_server(),
            auth_config=mcp_serve.McpHttpAuthConfig(psk="test-secret", path="/mcp"),
            path="/rpc",
        )


@pytest.mark.asyncio
async def test_custom_auth_path_is_the_served_protected_endpoint():
    config = mcp_serve.McpHttpAuthConfig(psk="test-secret", path="/rpc")
    async with _client(config) as client:
        denied = await client.post("/rpc", json=_initialize())
        accepted = await client.post(
            "/rpc", json=_initialize(), headers={"Authorization": "Bearer test-secret"},
        )
        old_path = await client.post("/mcp", json=_initialize())
    assert denied.status_code == 401
    assert accepted.status_code == 200
    assert accepted.json()["result"]["serverInfo"]["name"] == "hermes"
    assert old_path.status_code == 404


@pytest.mark.asyncio
async def test_non_ascii_credentials_are_rejected_without_server_error():
    config = mcp_serve.McpHttpAuthConfig(
        psk="test-secret", allow_query_token=True, oauth_compatible=True,
    )
    async with _client(config) as client:
        query = await client.post("/mcp", params={"access_token": "wrong-é"}, json=_initialize())
        client_id = await client.post(
            "/mcp/token", data={"grant_type": "client_credentials", "client_id": "wrong-é"},
        )
        secret = await client.post("/mcp/token", data={
            "grant_type": "client_credentials", "client_id": "hermes-mcp",
            "client_secret": "wrong-é",
        })
    assert [query.status_code, client_id.status_code, secret.status_code] == [401, 401, 401]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_verifier", ["", "too-short", "é" * 43])
async def test_malformed_verifier_consumes_authorization_code(bad_verifier):
    verifier = "v" * 43
    redirect_uri = "https://client.example/callback"
    config = mcp_serve.McpHttpAuthConfig(
        psk="test-secret", oauth_compatible=True, allowed_redirect_uris=[redirect_uri],
    )
    async with _client(config) as client:
        authorization = await client.get("/mcp/authorize", params={
            "client_id": "hermes-mcp", "response_type": "code", "redirect_uri": redirect_uri,
            "code_challenge_method": "S256",
            "code_challenge": mcp_serve._pkce_challenge(verifier, "S256"),
        })
        assert authorization.status_code == 302
        code = parse_qs(urlparse(authorization.headers["location"]).query)["code"][0]
        grant = {
            "grant_type": "authorization_code", "client_id": "hermes-mcp",
            "client_secret": "test-secret", "redirect_uri": redirect_uri, "code": code,
        }
        failed = await client.post("/mcp/token", data={**grant, "code_verifier": bad_verifier})
        replay = await client.post("/mcp/token", data={**grant, "code_verifier": verifier})
    assert failed.status_code == 400
    assert failed.json()["error"] == "invalid_grant"
    assert replay.status_code == 400
    assert replay.json()["error"] == "invalid_grant"
