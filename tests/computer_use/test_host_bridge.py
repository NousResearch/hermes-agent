"""Unit and ASGI tests for the CUA host bridge transport hardening."""

from __future__ import annotations

import asyncio
import contextlib

import pytest
from mcp import types
from starlette.testclient import TestClient

from tools.computer_use.host_bridge import (
    CUA_HOST_SCOPE,
    _CUA_HOST_CLIENT_ID,
    StaticBearerTokenVerifier,
    create_host_bridge_app,
)

_VALID_TOKEN = "t" * 40


@contextlib.asynccontextmanager
async def _fake_child(*_args):
    class _Stub:
        async def list_tools(self, params=None):
            return types.ListToolsResult(tools=[])

        async def call_tool(self, name, arguments=None):
            return types.CallToolResult(content=[], isError=False)

    yield _Stub()


def _build_app():
    # The lifespan consumes the child context exactly once, so every app under
    # test needs its own freshly created context-manager instance.
    return create_host_bridge_app(
        child_session_context=_fake_child(),
        bearer_token=_VALID_TOKEN,
        allowed_hosts=["localhost:8765"],
        allowed_origins=["http://localhost:8765"],
    )


# --- LAYER A: StaticBearerTokenVerifier unit behaviour -----------------------


def test_verifier_accepts_minimum_length_token():
    assert StaticBearerTokenVerifier("a" * 32) is not None


def test_verifier_rejects_short_token():
    with pytest.raises(ValueError, match="at least 32 bytes"):
        StaticBearerTokenVerifier("a" * 31)


def test_verifier_rejects_control_characters():
    with pytest.raises(ValueError, match="control characters"):
        StaticBearerTokenVerifier("a" * 31 + "\n")


def test_verify_token_accepts_configured_token():
    verifier = StaticBearerTokenVerifier("a" * 32)
    token = asyncio.run(verifier.verify_token("a" * 32))
    assert token is not None
    assert token.scopes == [CUA_HOST_SCOPE]
    assert token.client_id == _CUA_HOST_CLIENT_ID


def test_verify_token_rejects_wrong_token():
    verifier = StaticBearerTokenVerifier("a" * 32)
    assert asyncio.run(verifier.verify_token("b" * 32)) is None


def test_verify_token_rejects_control_character_guess():
    verifier = StaticBearerTokenVerifier("a" * 32)
    # A control character can never be the configured token, so the guess is
    # failed cleanly instead of raising.
    assert asyncio.run(verifier.verify_token("a" * 31 + "\n")) is None


# --- LAYER B: ASGI middleware ordering through TestClient --------------------


def _initialize_body():
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {"name": "t", "version": "1"},
        },
    }


def test_missing_token_is_unauthorized():
    with TestClient(_build_app(), base_url="http://localhost:8765") as client:
        assert client.get("/mcp").status_code == 401


def test_wrong_token_is_unauthorized():
    with TestClient(_build_app(), base_url="http://localhost:8765") as client:
        response = client.get("/mcp", headers={"Authorization": "Bearer " + "x" * 40})
        assert response.status_code == 401


def test_put_method_rejected_before_auth():
    with TestClient(_build_app(), base_url="http://localhost:8765") as client:
        # The valid token proves the 405 comes from the outermost method
        # filter, not from failed authentication further in.
        response = client.put("/mcp", headers={"Authorization": f"Bearer {_VALID_TOKEN}"})
        assert response.status_code == 405
        assert response.headers["allow"] == "DELETE, GET, POST"


def test_wrong_host_rejected_before_auth():
    # Regression test for the transport-security-first ordering: a valid token
    # with a disallowed Host must be classified as a transport violation (421),
    # not surface as auth noise (401) and not pass through.
    with TestClient(_build_app(), base_url="http://evil.example.com:8765") as client:
        response = client.get("/mcp", headers={"Authorization": f"Bearer {_VALID_TOKEN}"})
        assert response.status_code == 421


def test_initialize_response_is_marked_no_store():
    with TestClient(_build_app(), base_url="http://localhost:8765") as client:
        response = client.post(
            "/mcp",
            json=_initialize_body(),
            headers={
                "Authorization": f"Bearer {_VALID_TOKEN}",
                "Accept": "application/json, text/event-stream",
            },
        )
        assert response.status_code == 200
        assert "result" in response.json()
        # Rejections (401/405/421) are produced by outer middleware, upstream
        # of the innermost no-store wrapper mandated by the layering; the
        # session-manager payloads that must not be cached are stamped here.
        assert response.headers["cache-control"] == "no-store"