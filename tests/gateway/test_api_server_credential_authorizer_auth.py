"""Adversarial contract tests for plugin-issued API credentials."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.api_credentials import (
    APIServerOperation,
    AgentProfileId,
    AuthorizedAPICredential,
    CredentialAuthorizationRequest,
    CredentialScopeId,
)
from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_credential_authorizer import _CredentialAuthorizerRunner
from gateway.platforms import api_server as api_server_module
from gateway.platforms import api_server_runs as api_server_runs_module
from gateway.platforms import api_server_credential_authorizer as credential_authorizer_module
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_constants import get_hermes_home

OPERATOR_KEY = "operator-key-1234567890"
from tests.gateway.api_server_credential_authorizer_test_support import (
    OPERATOR_KEY, AsyncAuthorizer, Authorizer, SyncAuthorizer, _adapter, _auth_app,
    _create_owned_session, _credential_app, _principal, _principal_with_operations,
    _wait_for_run,
)

def test_credential_contract_is_strict_and_immutable():
    request = CredentialAuthorizationRequest(
        bearer="transient-secret",
        method="GET",
        canonical_route="/v1/capabilities",
        operation=APIServerOperation.CAPABILITIES_READ,
    )
    principal = _principal(request.operation)

    assert request.operation is APIServerOperation.CAPABILITIES_READ
    assert "transient-secret" not in repr(request)
    assert principal.allowed_operations == frozenset({request.operation})
    with pytest.raises((AttributeError, TypeError)):
        principal.runtime_profile = "other"

@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_profile", "../other"),
        ("agent_profile_id", "agent-one"),
        ("credential_scope_id", CredentialScopeId("scope\nother") if False else "bad"),
        ("allowed_operations", {APIServerOperation.CAPABILITIES_READ}),
    ],
)
def test_malformed_principal_fields_are_rejected(field, value):
    values = {
        "principal_id": "principal-one",
        "runtime_profile": "default",
        "agent_profile_id": AgentProfileId("agent-one"),
        "credential_scope_id": CredentialScopeId("device-one"),
        "allowed_operations": frozenset({APIServerOperation.CAPABILITIES_READ}),
    }
    values[field] = value
    with pytest.raises((TypeError, ValueError)):
        AuthorizedAPICredential(**values)

def test_plugin_context_exposes_one_explicit_authorizer_surface():
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="first"), manager)
    authorizer = Authorizer(lambda _request: None)

    handle = context.register_api_server_credential_authorizer(authorizer)

    assert manager.get_api_server_credential_authorizer() is authorizer
    handle.dispose()
    assert manager.get_api_server_credential_authorizer() is None

@pytest.mark.asyncio
async def test_static_operator_key_has_precedence_over_plugin_authorizer():
    calls = []
    adapter = _adapter(Authorizer(lambda request: calls.append(request)))

    async def handler(request):
        assert adapter._check_auth(request) is None
        return web.json_response({"ok": True})

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        response = await client.get(
            "/v1/capabilities", headers={"Authorization": f"Bearer {OPERATOR_KEY}"}
        )

    assert response.status == 200
    assert calls == []

@pytest.mark.asyncio
async def test_authorizer_receives_transient_token_and_canonical_route_metadata_once():
    calls = []

    def authorize(request):
        calls.append(request)
        return _principal(request.operation)

    adapter = _adapter(Authorizer(authorize))

    async def handler(request):
        assert adapter._check_auth(request) is None
        return web.json_response({"ok": True})

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        response = await client.get(
            "/v1/capabilities?ignored=yes",
            headers={"Authorization": "Bearer transient-secret"},
        )

    assert response.status == 200
    assert calls == [
        CredentialAuthorizationRequest(
            bearer="transient-secret",
            method="GET",
            canonical_route="/v1/capabilities",
            operation=APIServerOperation.CAPABILITIES_READ,
        )
    ]

@pytest.mark.asyncio
@pytest.mark.parametrize("result", [None, {"runtime_profile": "default"}])
async def test_invalid_revoked_or_non_contract_principal_is_rejected(result):
    adapter = _adapter(Authorizer(lambda _request: result))

    async def handler(_request):
        raise AssertionError("handler must not run")

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        response = await client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer revoked-secret"}
        )

    assert response.status == 401

@pytest.mark.asyncio
async def test_wrong_operation_is_rejected_before_handler():
    adapter = _adapter(
        Authorizer(lambda _request: _principal(APIServerOperation.SESSIONS_CREATE))
    )

    async def handler(_request):
        raise AssertionError("handler must not run")

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        response = await client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer limited-secret"}
        )

    assert response.status == 403

@pytest.mark.asyncio
async def test_ineligible_response_route_never_reaches_authorizer():
    calls = []
    adapter = _adapter(Authorizer(lambda request: calls.append(request)))

    async def handler(request):
        return adapter._check_auth(request) or web.json_response({"ok": True})

    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_post("/v1/responses", handler)
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer transient-secret"},
        )

    assert response.status == 401
    assert calls == []

@pytest.mark.asyncio
async def test_url_profile_conflict_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda **_kwargs: [("default", Path("/default")), ("worker", Path("/worker"))],
    )
    adapter = _adapter(
        Authorizer(
            lambda request: _principal(request.operation, profile="default")
        ),
        multiplex=True,
    )

    async def handler(_request):
        raise AssertionError("handler must not run")

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        response = await client.get(
            "/p/worker/v1/capabilities",
            headers={"Authorization": "Bearer transient-secret"},
        )

    assert response.status == 403

@pytest.mark.asyncio
async def test_authorizer_exception_is_generic_and_redacted(caplog):
    token = "credential-that-must-not-appear"
    exception_text = "issuer database says credential-that-must-not-appear was revoked"

    def authorize(_request):
        raise RuntimeError(exception_text)

    adapter = _adapter(Authorizer(authorize))

    async def handler(_request):
        raise AssertionError("handler must not run")

    caplog.set_level(logging.WARNING)
    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        response = await client.get(
            "/v1/capabilities", headers={"Authorization": f"Bearer {token}"}
        )
        body = await response.json()

    assert response.status == 401
    assert body["error"]["code"] == "gateway_auth_failed"
    assert token not in caplog.text
    assert exception_text not in caplog.text

@pytest.mark.asyncio
async def test_real_profile_scope_is_entered_before_handler(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    worker_home = default_home / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda **_kwargs: [("default", default_home), ("worker", worker_home)],
    )
    adapter = _adapter(
        Authorizer(lambda request: _principal(request.operation, profile="worker")),
        multiplex=True,
    )

    async def handler(request):
        assert adapter._check_auth(request) is None
        return web.json_response({"home": str(get_hermes_home())})

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        response = await client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer derived-secret"}
        )
        body = await response.json()

    assert response.status == 200
    assert Path(body["home"]).resolve() == worker_home.resolve()

@pytest.mark.asyncio
async def test_unserved_and_wrong_profile_principals_fail_closed(monkeypatch):
    served = [("default", Path("/default")), ("worker", Path("/worker"))]
    monkeypatch.setattr("hermes_cli.profiles.profiles_to_serve", lambda **_kwargs: served)
    selected = ["missing"]
    adapter = _adapter(Authorizer(
        lambda request: _principal(request.operation, profile=selected[0])
    ), multiplex=True)

    async def handler(_request):
        raise AssertionError("handler must not run")

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        unserved = await client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer unserved"}
        )
        selected[0] = "default"
        wrong = await client.get(
            "/p/worker/v1/capabilities", headers={"Authorization": "Bearer wrong-profile"}
        )

    assert unserved.status == wrong.status == 403
