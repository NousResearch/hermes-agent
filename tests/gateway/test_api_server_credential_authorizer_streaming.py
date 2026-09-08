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

@pytest.mark.asyncio
async def test_run_events_revalidates_credential_before_later_event_delivery():
    authorized = [True]
    principal = _principal(APIServerOperation.RUN_EVENTS_READ)
    authorizer = Authorizer(lambda _request: principal if authorized[0] else None)
    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="stream-authority"), manager)
    ctx.register_api_server_credential_authorizer(authorizer)
    adapter = _adapter(None)
    adapter._api_credential_authorizer_manager = manager
    run_id = "run-authority-lifetime"
    adapter._run_owners[run_id] = adapter._credential_owner_key(principal)
    queue = asyncio.Queue()
    adapter._run_streams[run_id] = queue

    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    async with TestClient(TestServer(app)) as client:
        response = await client.get(
            f"/v1/runs/{run_id}/events",
            headers={"Authorization": "Bearer credential"},
        )
        assert response.status == 200
        authorized[0] = False
        await queue.put({"event": "message.delta", "delta": "must-not-leak"})
        await queue.put(None)
        body = await asyncio.wait_for(response.read(), timeout=2)

    assert b"must-not-leak" not in body

@pytest.mark.asyncio
async def test_run_events_rechecks_registration_after_suspended_reauthorization():
    calls = 0
    reauthorization_started = asyncio.Event()
    release_reauthorization = asyncio.Event()
    principal = _principal(APIServerOperation.RUN_EVENTS_READ)

    async def authorize(_request):
        nonlocal calls
        calls += 1
        if calls > 1:
            reauthorization_started.set()
            await release_reauthorization.wait()
        return principal

    manager = PluginManager()
    ctx = PluginContext(PluginManifest(name="stream-replacement-race"), manager)
    registration = ctx.register_api_server_credential_authorizer(
        AsyncAuthorizer(authorize)
    )
    adapter = _adapter(None)
    adapter._api_credential_authorizer_manager = manager
    run_id = "run-registration-race"
    adapter._run_owners[run_id] = adapter._credential_owner_key(principal)
    queue = asyncio.Queue()
    adapter._run_streams[run_id] = queue

    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    async with TestClient(TestServer(app)) as client:
        response = await client.get(
            f"/v1/runs/{run_id}/events",
            headers={"Authorization": "Bearer credential"},
        )
        await queue.put({"event": "message.delta", "delta": "must-not-leak"})
        await asyncio.wait_for(reauthorization_started.wait(), timeout=1)
        registration.dispose()
        await queue.put(None)
        release_reauthorization.set()
        body = await asyncio.wait_for(response.read(), timeout=2)

    assert b"must-not-leak" not in body
