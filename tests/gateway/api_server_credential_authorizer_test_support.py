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
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from hermes_constants import get_hermes_home

OPERATOR_KEY = "operator-key-1234567890"


class Authorizer:
    def __init__(self, fn):
        self.fn = fn

    async def authorize(self, request):
        result = self.fn(request)
        if inspect.isawaitable(result):
            return await result
        return result


class SyncAuthorizer:
    def __init__(self, fn):
        self.fn = fn

    def authorize(self, request):
        return self.fn(request)


class AsyncAuthorizer:
    def __init__(self, fn):
        self.fn = fn

    async def authorize(self, request):
        return await self.fn(request)


def _principal(
    operation: APIServerOperation,
    *,
    profile: str = "default",
    scope: str = "device-one",
):
    return AuthorizedAPICredential(
        principal_id="principal-one",
        runtime_profile=profile,
        agent_profile_id=AgentProfileId("agent-one"),
        credential_scope_id=CredentialScopeId(scope),
        allowed_operations=frozenset({operation}),
    )


def _adapter(authorizer, *, multiplex: bool = False) -> APIServerAdapter:
    adapter = APIServerAdapter(
        PlatformConfig(enabled=True, extra={"key": OPERATOR_KEY})
    )
    adapter._api_credential_authorizer = authorizer

    class Runner:
        config = GatewayConfig(multiplex_profiles=multiplex)

    adapter.gateway_runner = Runner()
    return adapter


def _auth_app(adapter: APIServerAdapter, handler) -> web.Application:
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_get("/v1/capabilities", handler)
    app.router.add_get("/p/{profile}/v1/capabilities", handler)
    return app


def _credential_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_delete("/api/sessions/{session_id}", adapter._handle_delete_session)
    app.router.add_get(
        "/api/sessions/{session_id}/messages", adapter._handle_session_messages
    )
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    return app


async def _wait_for_run(adapter: APIServerAdapter, run_id: str) -> None:
    task = adapter._active_run_tasks.get(run_id)
    if task is not None:
        await asyncio.wait_for(task, timeout=2)


def _principal_with_operations(*operations, profile="default", scope="device-one"):
    principal = _principal(operations[0], profile=profile, scope=scope)
    return AuthorizedAPICredential(
        principal_id=principal.principal_id,
        runtime_profile=principal.runtime_profile,
        agent_profile_id=principal.agent_profile_id,
        credential_scope_id=principal.credential_scope_id,
        allowed_operations=frozenset(operations),
    )


def _create_owned_session(adapter, principal, session_id="owned-session"):
    owner = adapter._credential_owner_key(principal)
    adapter._ensure_session_db().create_session(
        session_id, "api_server", credential_owner=owner
    )
    return session_id
