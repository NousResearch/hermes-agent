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
async def test_same_origin_cross_profile_session_lookup_isolated_with_same_generated_id(
    tmp_path, monkeypatch
):
    default_home = tmp_path / ".hermes"
    worker_home = default_home / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda **_kwargs: [("default", default_home), ("worker", worker_home)],
    )
    monkeypatch.setattr(api_server_module.time, "time", lambda: 777)
    monkeypatch.setattr(
        api_server_module.uuid, "uuid4", lambda: type("U", (), {"hex": "f" * 32})()
    )

    def authorize(request):
        profile = "worker" if request.bearer.startswith("worker") else "default"
        return _principal_with_operations(
            APIServerOperation.SESSIONS_CREATE,
            APIServerOperation.SESSIONS_RESOLVE,
            profile=profile,
            scope=request.bearer,
        )

    adapter = _adapter(Authorizer(authorize), multiplex=True)
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_post("/p/{profile}/api/sessions", adapter._handle_create_session)
    app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
    app.router.add_get("/p/{profile}/api/sessions/{session_id}", adapter._handle_get_session)

    async with TestClient(TestServer(app)) as client:
        default_create = await client.post(
            "/api/sessions", json={"title": "default-title"}, headers={"Authorization": "Bearer default-token"}
        )
        worker_create = await client.post(
            "/api/sessions", json={"title": "worker-title"}, headers={"Authorization": "Bearer worker-token"}
        )
        session_id = (await default_create.json())["session"]["id"]
        assert session_id == (await worker_create.json())["session"]["id"]
        default_get = await client.get(
            f"/api/sessions/{session_id}", headers={"Authorization": "Bearer default-token"}
        )
        worker_get = await client.get(
            f"/p/worker/api/sessions/{session_id}", headers={"Authorization": "Bearer worker-token"}
        )
        default_payload = await default_get.json()
        worker_payload = await worker_get.json()

    assert default_get.status == worker_get.status == 200
    assert default_payload["session"]["title"] == "default-title"
    assert worker_payload["session"]["title"] == "worker-title"

@pytest.mark.asyncio
async def test_session_creation_and_resolution_are_credential_scope_isolated():
    active_scope = ["scope-a"]
    operations = frozenset({
        APIServerOperation.SESSIONS_CREATE,
        APIServerOperation.SESSIONS_RESOLVE,
    })

    def authorize(_request):
        principal = _principal(APIServerOperation.SESSIONS_CREATE, scope=active_scope[0])
        return AuthorizedAPICredential(
            principal_id=principal.principal_id,
            runtime_profile=principal.runtime_profile,
            agent_profile_id=principal.agent_profile_id,
            credential_scope_id=principal.credential_scope_id,
            allowed_operations=operations,
        )

    adapter = _adapter(Authorizer(authorize))
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)

    async with TestClient(TestServer(app)) as client:
        created = await client.post(
            "/api/sessions",
            json={"source": "attacker-selected"},
            headers={"Authorization": "Bearer rotating-token-a"},
        )
        created_body = await created.json()
        session_id = created_body["session"]["id"]

        active_scope[0] = "scope-b"
        foreign_list = await client.get(
            "/api/sessions", headers={"Authorization": "Bearer rotating-token-b"}
        )
        foreign_body = await foreign_list.json()

        admin_list = await client.get(
            "/api/sessions", headers={"Authorization": f"Bearer {OPERATOR_KEY}"}
        )
        admin_body = await admin_list.json()

    row = adapter._ensure_session_db().get_session(session_id)
    assert created.status == 201
    assert row["source"] == "api_server"
    assert row["credential_owner"].startswith("api-credential:")
    assert foreign_body["data"] == []
    assert any(item["id"] == session_id for item in admin_body["data"])

@pytest.mark.asyncio
async def test_credential_session_list_does_not_project_foreign_compression_tip(tmp_path):
    from hermes_state import SessionDB

    principal = _principal(APIServerOperation.SESSIONS_RESOLVE)
    adapter = _adapter(Authorizer(lambda _request: principal))
    adapter._session_db = SessionDB(tmp_path / "state.db")
    owner = adapter._credential_owner_key(principal)
    root_id = _create_owned_session(adapter, principal, "owned-root")
    adapter._session_db.set_session_title(root_id, "Owned title")
    adapter._session_db.append_message(root_id, "user", "OWNED_PREVIEW")
    adapter._session_db.end_session(root_id, "compression")
    adapter._session_db.create_session(
        "foreign-tip",
        "api_server",
        parent_session_id=root_id,
        credential_owner="api-credential:foreign",
    )
    adapter._session_db.set_session_title("foreign-tip", "FOREIGN_TITLE")
    adapter._session_db.append_message("foreign-tip", "user", "FOREIGN_PREVIEW")

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.get(
            "/api/sessions", headers={"Authorization": "Bearer credential"}
        )
        body = await response.json()

    assert response.status == 200
    assert [row["id"] for row in body["data"]] == [root_id]
    assert "FOREIGN_TITLE" not in str(body)
    assert "FOREIGN_PREVIEW" not in str(body)
    assert adapter._session_db.get_session(root_id)["credential_owner"] == owner

@pytest.mark.asyncio
async def test_credential_message_read_rejects_foreign_recreation_before_atomic_read(
    tmp_path, monkeypatch
):
    """A credential-authorized request cannot disclose a replacement owner's transcript."""
    from hermes_state import SessionDB

    principal = _principal(APIServerOperation.SESSIONS_RESOLVE)
    adapter = _adapter(Authorizer(lambda _request: principal))
    adapter._session_db = SessionDB(tmp_path / "state.db")
    owner = adapter._credential_owner_key(principal)
    session_id = _create_owned_session(adapter, principal, "message-race")
    adapter._session_db.append_message(session_id, "user", "owned message")
    racing_db = SessionDB(tmp_path / "state.db")
    real_read = adapter._session_db.resolve_owned_session_messages
    replaced = False

    def replace_then_read(*args, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            assert racing_db.delete_session(session_id)
            racing_db.create_session(
                session_id, "api_server", credential_owner="api-credential:foreign"
            )
            racing_db.append_message(session_id, "user", "FOREIGN_SECRET")
        return real_read(*args, **kwargs)

    monkeypatch.setattr(
        adapter._session_db, "resolve_owned_session_messages", replace_then_read
    )

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.get(
            f"/api/sessions/{session_id}/messages",
            headers={"Authorization": "Bearer credential"},
        )
        body = await response.json()

    assert replaced is True
    assert owner != racing_db.get_session(session_id)["credential_owner"]
    assert response.status == 404
    assert body["error"]["code"] == "session_not_found"
    assert "FOREIGN_SECRET" not in str(body)

@pytest.mark.asyncio
async def test_credential_message_read_requires_owner_on_resolved_compression_tip(tmp_path):
    from hermes_state import SessionDB

    principal = _principal(APIServerOperation.SESSIONS_RESOLVE)
    adapter = _adapter(Authorizer(lambda _request: principal))
    adapter._session_db = SessionDB(tmp_path / "state.db")
    owner = adapter._credential_owner_key(principal)
    adapter._session_db.create_session("root", "api_server", credential_owner=owner)
    adapter._session_db.end_session("root", "compression")
    adapter._session_db.create_session(
        "foreign-tip",
        "api_server",
        parent_session_id="root",
        credential_owner="api-credential:foreign",
    )
    adapter._session_db.append_message("foreign-tip", "user", "FOREIGN_SECRET")

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.get(
            "/api/sessions/root/messages",
            headers={"Authorization": "Bearer credential"},
        )
        body = await response.json()

    assert response.status == 404
    assert body["error"]["code"] == "session_not_found"
    assert "foreign-tip" not in str(body)
    assert "FOREIGN_SECRET" not in str(body)

@pytest.mark.asyncio
async def test_static_admin_message_read_keeps_ordinary_path(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    adapter = _adapter(None)
    adapter._session_db = SessionDB(tmp_path / "state.db")
    adapter._session_db.create_session("legacy", "api_server")
    adapter._session_db.append_message("legacy", "user", "legacy message")
    monkeypatch.setattr(
        adapter._session_db,
        "resolve_owned_session_messages",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("static path must not use credential-only read")
        ),
    )

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.get(
            "/api/sessions/legacy/messages",
            headers={"Authorization": f"Bearer {OPERATOR_KEY}"},
        )
        body = await response.json()

    assert response.status == 200
    assert body["session_id"] == "legacy"
    assert [message["content"] for message in body["data"]] == ["legacy message"]

@pytest.mark.asyncio
async def test_static_admin_deletion_respects_only_credential_owned_active_lease(tmp_path):
    from hermes_state import SessionDB

    adapter = _adapter(None)
    adapter._session_db = SessionDB(tmp_path / "state.db")
    adapter._session_db.create_session("legacy", "api_server")
    adapter._session_db.create_session(
        "credential-owned", "api_server", credential_owner="api-credential:owner"
    )
    for session_id in ("legacy", "credential-owned"):
        assert adapter._session_db.try_acquire_session_turn_lease(
            session_id, f"pid={os.getpid()}:turn={session_id}", ttl_seconds=300
        )

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        headers = {"Authorization": f"Bearer {OPERATOR_KEY}"}
        legacy = await client.delete("/api/sessions/legacy", headers=headers)
        protected = await client.delete("/api/sessions/credential-owned", headers=headers)
        protected_body = await protected.json()

    assert legacy.status == 200
    assert adapter._session_db.get_session("legacy") is None
    assert protected.status == 409
    assert protected_body["error"]["code"] == "session_busy"
    assert adapter._session_db.get_session("credential-owned") is not None

@pytest.mark.asyncio
async def test_session_titles_are_unique_per_owner_without_foreign_id_disclosure():
    operations = (
        APIServerOperation.SESSIONS_CREATE,
        APIServerOperation.SESSIONS_RESOLVE,
    )
    adapter = _adapter(Authorizer(
        lambda request: _principal_with_operations(
            *operations, scope="scope-b" if request.bearer == "owner-b" else "scope-a"
        )
    ))

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        first = await client.post(
            "/api/sessions", json={"title": "Shared title"},
            headers={"Authorization": "Bearer owner-a"},
        )
        foreign_id = (await first.json())["session"]["id"]
        second = await client.post(
            "/api/sessions", json={"title": "Shared title"},
            headers={"Authorization": "Bearer owner-b"},
        )
        duplicate = await client.post(
            "/api/sessions", json={"title": "Shared title"},
            headers={"Authorization": "Bearer owner-b"},
        )
        duplicate_body = await duplicate.json()

    assert first.status == second.status == 201
    assert duplicate.status == 409
    assert duplicate_body["error"]["code"] == "session_title_exists"
    assert foreign_id not in str(duplicate_body)

@pytest.mark.asyncio
async def test_static_admin_session_conflicts_keep_legacy_status_codes_and_messages():
    adapter = _adapter(None)

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        headers = {"Authorization": f"Bearer {OPERATOR_KEY}"}
        created = await client.post(
            "/api/sessions", json={"id": "legacy-id", "title": "Legacy title"},
            headers=headers,
        )
        duplicate_id = await client.post(
            "/api/sessions", json={"id": "legacy-id"}, headers=headers,
        )
        duplicate_title = await client.post(
            "/api/sessions", json={"id": "other-id", "title": "Legacy title"},
            headers=headers,
        )
        duplicate_id_body = await duplicate_id.json()
        duplicate_title_body = await duplicate_title.json()

    assert created.status == 201
    assert duplicate_id.status == 409
    assert duplicate_id_body["error"]["message"] == "Session already exists: legacy-id"
    assert duplicate_id_body["error"]["code"] == "session_exists"
    assert duplicate_title.status == 400
    assert duplicate_title_body["error"]["message"] == (
        "Title already in use by session legacy-id"
    )
    assert duplicate_title_body["error"]["code"] == "invalid_title"

@pytest.mark.asyncio
async def test_static_admin_retains_legacy_unowned_session_access():
    adapter = _adapter(Authorizer(lambda _request: None))
    db = adapter._ensure_session_db()
    db.create_session("legacy-session", source="api_server")

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.get(
            "/api/sessions", headers={"Authorization": f"Bearer {OPERATOR_KEY}"}
        )
        body = await response.json()

    assert response.status == 200
    assert any(row["id"] == "legacy-session" for row in body["data"])
