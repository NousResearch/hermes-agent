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

def test_ambiguous_authorizers_fail_adapter_startup_resolution(monkeypatch):
    manager = PluginManager()
    for name in ("first", "second"):
        PluginContext(PluginManifest(name=name), manager).register_api_server_credential_authorizer(
            Authorizer(lambda _request: None)
        )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    adapter = _adapter(None)

    assert adapter._load_api_credential_authorizer() is False
    assert adapter.fatal_error_code == "api_credential_authorizer_ambiguous"
    assert adapter.fatal_error_retryable is False

def test_sync_authorizer_is_rejected_at_registration_time():
    """register_api_server_credential_authorizer must raise immediately for non-async authorize."""
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="sync-only"), manager)
    with pytest.raises(ValueError, match="async"):
        ctx.register_api_server_credential_authorizer(SyncAuthorizer(lambda _request: None))
    # Nothing registered — manager sees no authorizer.
    assert manager.get_api_server_credential_authorizer() is None

def test_sync_authorizer_is_rejected_with_clear_startup_error(monkeypatch):
    """Startup also rejects a sync authorizer that somehow bypassed registration validation."""
    manager = PluginManager()
    # Bypass registration gate by injecting directly (simulates a stale/corrupt registration).
    manager._api_credential_authorizers.append(
        (SyncAuthorizer(lambda _request: None), "sync-only")
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    adapter = _adapter(None)

    assert adapter._load_api_credential_authorizer() is False
    assert adapter.fatal_error_code == "api_credential_authorizer_not_async"
    assert adapter.fatal_error_retryable is False

@pytest.mark.asyncio
async def test_connect_rejects_sync_authorizer_through_actual_startup_load(monkeypatch):
    manager = PluginManager()
    manager._api_credential_authorizers.append(
        (SyncAuthorizer(lambda _request: None), "sync-only")
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    adapter = _adapter(None)
    monkeypatch.setattr(adapter, "_api_key_passes_startup_guard", lambda: True)

    assert await adapter.connect() is False
    assert adapter.fatal_error_code == "api_credential_authorizer_not_async"
    assert adapter._runner is None
    assert adapter._site is None

@pytest.mark.asyncio
@pytest.mark.parametrize("lifecycle", ["dispose", "targeted_unload"])
async def test_live_adapter_revokes_authorizer_after_actual_registration_removal(
    lifecycle, monkeypatch
):
    manager = PluginManager()
    context = PluginContext(PluginManifest(name="credential-plugin"), manager)
    handle = context.register_api_server_credential_authorizer(
        Authorizer(lambda request: _principal(request.operation))
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    adapter = _adapter(None)
    assert adapter._load_api_credential_authorizer()
    runner = adapter._credential_authorizer_runner()

    async def handler(request):
        return adapter._check_auth(request) or web.json_response({"ok": True})

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        before = await client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer credential"}
        )
        if lifecycle == "dispose":
            handle.dispose()
        else:
            assert manager.unload("credential-plugin")
        after = await client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer credential"}
        )

    assert before.status == 200
    assert after.status == 401
    assert adapter._credential_authorizer_runner() is runner

@pytest.mark.asyncio
async def test_live_adapter_uses_force_reloaded_authorizer_and_revokes_old_one(monkeypatch):
    manager = PluginManager()
    old = Authorizer(
        lambda request: _principal(request.operation) if request.bearer == "old" else None
    )
    PluginContext(PluginManifest(name="credential-plugin"), manager).register_api_server_credential_authorizer(old)
    manager._discovered = True
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    adapter = _adapter(None)
    assert adapter._load_api_credential_authorizer()
    runner = adapter._credential_authorizer_runner()

    replacement = Authorizer(
        lambda request: _principal(request.operation) if request.bearer == "new" else None
    )

    def reload_registration():
        PluginContext(PluginManifest(name="credential-plugin"), manager).register_api_server_credential_authorizer(
            replacement
        )

    monkeypatch.setattr(manager, "_discover_and_load_inner", reload_registration)

    async def handler(request):
        return adapter._check_auth(request) or web.json_response({"ok": True})

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        before = await client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer old"}
        )
        manager.discover_and_load(force=True)
        stale = await client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer old"}
        )
        fresh = await client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer new"}
        )

    assert before.status == fresh.status == 200
    assert stale.status == 401
    assert adapter._credential_authorizer_runner() is runner

@pytest.mark.asyncio
async def test_live_adapter_rejects_new_authorizer_ambiguity(monkeypatch):
    manager = PluginManager()
    PluginContext(PluginManifest(name="first"), manager).register_api_server_credential_authorizer(
        Authorizer(lambda request: _principal(request.operation))
    )
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    adapter = _adapter(None)
    assert adapter._load_api_credential_authorizer()

    async def handler(request):
        return adapter._check_auth(request) or web.json_response({"ok": True})

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        before = await client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer credential"}
        )
        PluginContext(PluginManifest(name="second"), manager).register_api_server_credential_authorizer(
            Authorizer(lambda request: _principal(request.operation))
        )
        ambiguous = await client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer credential"}
        )

    assert before.status == 200
    assert ambiguous.status == 401

@pytest.mark.asyncio
async def test_authorizer_disposal_during_request_revokes_inflight_result(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()

    async def authorize(request):
        entered.set()
        await release.wait()
        return _principal(request.operation)

    manager = PluginManager()
    handle = PluginContext(
        PluginManifest(name="credential-plugin"), manager
    ).register_api_server_credential_authorizer(AsyncAuthorizer(authorize))
    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", lambda: manager)
    adapter = _adapter(None)
    assert adapter._load_api_credential_authorizer()

    async def handler(request):
        return adapter._check_auth(request) or web.json_response({"ok": True})

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        pending = asyncio.create_task(client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer credential"}
        ))
        await asyncio.wait_for(entered.wait(), timeout=1)
        handle.dispose()
        release.set()
        response = await asyncio.wait_for(pending, timeout=1)

    assert response.status == 401

@pytest.mark.asyncio
async def test_authorizer_removal_before_atomic_handler_admission_fails_closed():
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="admission-race"), manager)
    authorizer = Authorizer(
        lambda _request: _principal(APIServerOperation.CAPABILITIES_READ)
    )
    registration = ctx.register_api_server_credential_authorizer(authorizer)
    adapter = _adapter(None)
    adapter._api_credential_authorizer_manager = manager

    original_profile_check = adapter._credential_profile_is_served

    def remove_before_admission(profile):
        registration.dispose()
        return original_profile_check(profile)

    adapter._credential_profile_is_served = remove_before_admission

    async def handler(_request):
        raise AssertionError("retired authority must not enter the handler")

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        response = await client.get(
            "/v1/capabilities",
            headers={"Authorization": "Bearer credential"},
        )

    assert response.status == 401
