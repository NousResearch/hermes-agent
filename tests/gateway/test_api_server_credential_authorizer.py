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
from gateway.platforms.api_server import APIServerAdapter, _CredentialAuthorizerRunner
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


@pytest.mark.asyncio
async def test_credential_run_requires_explicit_preexisting_owned_session_before_id_allocation(
    monkeypatch,
):
    principal = _principal(APIServerOperation.RUNS_CREATE)
    adapter = _adapter(Authorizer(lambda _request: principal))
    monkeypatch.setattr(
        api_server_runs_module.uuid,
        "uuid4",
        lambda: (_ for _ in ()).throw(AssertionError("run ID allocation must not start")),
    )

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.post(
            "/v1/runs",
            json={"input": "hello"},
            headers={
                "Authorization": "Bearer credential",
                "X-Hermes-Session-Key": "declared-but-not-a-session-id",
            },
        )
        body = await response.json()

    assert response.status == 400
    assert body["error"]["code"] == "credential_session_required"


@pytest.mark.asyncio
async def test_credential_run_fails_closed_after_configured_durable_store_open_failure(
    tmp_path, monkeypatch,
):
    principal = _principal(APIServerOperation.RUNS_CREATE)
    adapter = _adapter(Authorizer(lambda _request: principal))
    adapter._run_idempotency_store.close()
    unopenable = tmp_path / "directory-not-database"
    unopenable.mkdir()
    adapter._run_idempotency_store = RunIdempotencyStore(str(unopenable))
    session_id = _create_owned_session(adapter, principal)
    monkeypatch.setattr(
        api_server_runs_module.uuid,
        "uuid4",
        lambda: (_ for _ in ()).throw(AssertionError("run ID allocation must not start")),
    )

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.post(
            "/v1/runs",
            json={"input": "hello", "session_id": session_id},
            headers={"Authorization": "Bearer credential"},
        )
        body = await response.json()

    assert adapter._run_idempotency_store.durability_state == "degraded"
    assert response.status == 503
    assert body["error"]["code"] == "run_storage_unavailable"


@pytest.mark.asyncio
async def test_credential_run_allows_explicit_in_memory_store_for_tests(monkeypatch):
    principal = _principal(APIServerOperation.RUNS_CREATE)
    adapter = _adapter(Authorizer(lambda _request: principal))
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore.in_memory()
    session_id = _create_owned_session(adapter, principal)
    agent = MagicMock(session_id=session_id)
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.post(
            "/v1/runs",
            json={"input": "hello", "session_id": session_id},
            headers={"Authorization": "Bearer credential"},
        )

    assert adapter._run_idempotency_store.durability_state == "memory"
    assert response.status == 202


@pytest.mark.asyncio
async def test_credential_run_compression_preserves_owner_for_effective_session_and_second_run(
    tmp_path, monkeypatch
):
    from hermes_state import SessionDB
    from run_agent import AIAgent
    from agent import conversation_loop

    principal = _principal_with_operations(
        APIServerOperation.RUNS_CREATE, APIServerOperation.RUN_STATUS_READ
    )
    adapter = _adapter(Authorizer(lambda _request: principal))
    adapter._session_db = SessionDB(tmp_path / "state.db")
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore.in_memory()
    owner = adapter._credential_owner_key(principal)
    parent_id = _create_owned_session(adapter, principal, "compression-parent")
    adapter._session_db.set_session_title(parent_id, "Owned compression")
    adapter._session_db.create_session(
        "foreign-title", "api_server", credential_owner="api-credential:foreign"
    )
    adapter._session_db.set_session_title("foreign-title", "Owned compression")
    histories = []
    agents = []
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    def create_agent(**kwargs):
        assert kwargs["credential_owner"] == owner
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=adapter._session_db,
            session_id=kwargs["session_id"],
            platform="api_server",
            skip_context_files=True,
            skip_memory=True,
        )
        agent._credential_owner = kwargs["credential_owner"]

        def run_conversation(self, *, user_message, conversation_history, task_id):
            histories.append((self.session_id, list(conversation_history)))
            if self.session_id == parent_id:
                compressor = MagicMock()
                compressor.compress.return_value = [
                    {"role": "user", "content": "[CONTEXT COMPACTION] owned summary"}
                ]
                compressor.compression_count = 1
                compressor.last_prompt_tokens = compressor.last_completion_tokens = 0
                compressor._last_summary_error = None
                compressor._last_compress_aborted = False
                compressor._last_summary_auth_failure = False
                compressor._last_aux_model_failure_model = None
                compressor._last_aux_model_failure_error = None
                self.context_compressor = compressor
                self.compression_in_place = False
                oversized = [
                    {
                        "role": "user" if index % 2 == 0 else "assistant",
                        "content": f"old-{index}-" + ("x" * 500),
                    }
                    for index in range(20)
                ]
                self._compress_context(
                    oversized,
                    "system",
                    approx_tokens=120_000,
                    force=True,
                    task_id=task_id,
                )
            else:
                self._session_db.append_message(self.session_id, "user", user_message)
            self._session_db.append_message(self.session_id, "assistant", f"reply: {user_message}")
            return {"final_response": f"reply: {user_message}", "messages": []}

        if len(agents) < 2:
            agent.run_conversation = types.MethodType(run_conversation, agent)
        agents.append(agent)
        return agent

    monkeypatch.setattr(adapter, "_create_agent", create_agent)

    def run_conversation_loop(
        agent, user_message, _system_message, conversation_history, _task_id, *_args, **_kwargs
    ):
        histories.append((agent.session_id, list(conversation_history)))
        agent._session_db.append_message(agent.session_id, "user", user_message)
        agent._session_db.append_message(agent.session_id, "assistant", f"reply: {user_message}")
        return {"final_response": f"reply: {user_message}", "messages": []}

    monkeypatch.setattr(conversation_loop, "run_conversation", run_conversation_loop)
    headers = {"Authorization": "Bearer credential"}
    async with TestClient(TestServer(_credential_app(adapter))) as client:
        first = await client.post(
            "/v1/runs", json={"input": "first", "session_id": parent_id}, headers=headers
        )
        first_body = await first.json()
        await _wait_for_run(adapter, first_body["run_id"])
        first_status = await client.get(
            f"/v1/runs/{first_body['run_id']}", headers=headers
        )
        effective_session_id = (await first_status.json())["session_id"]

        second = await client.post(
            "/v1/runs",
            json={"input": "second", "session_id": effective_session_id},
            headers=headers,
        )
        second_body = await second.json()
        await _wait_for_run(adapter, second_body["run_id"])

        resumed_from_parent = await client.post(
            "/v1/runs",
            json={"input": "third", "session_id": parent_id},
            headers=headers,
        )
        resumed_body = await resumed_from_parent.json()
        await _wait_for_run(adapter, resumed_body["run_id"])
        resumed_status = await client.get(
            f"/v1/runs/{resumed_body['run_id']}", headers=headers
        )
        resumed_session_id = (await resumed_status.json())["session_id"]

    assert first.status == second.status == resumed_from_parent.status == 202
    assert effective_session_id != parent_id
    assert agents[0].session_id == effective_session_id
    assert agents[1].session_id == effective_session_id
    assert agents[2].session_id == effective_session_id
    assert resumed_session_id == effective_session_id
    assert histories[1][0] == effective_session_id
    assert histories[2][0] == effective_session_id
    assert any(message["content"] == "reply: first" for message in histories[1][1])
    assert any(message["content"] == "reply: second" for message in histories[2][1])
    assert adapter._session_db.get_session(parent_id)["credential_owner"] == owner
    child = adapter._session_db.get_session(effective_session_id)
    assert child["credential_owner"] == owner
    assert child["parent_session_id"] == parent_id
    assert child["title"] == "Owned compression"
    assert adapter._session_db.resolve_session_by_title(
        "Owned compression", credential_owner=owner
    ) == effective_session_id
    assert all(
        adapter._session_db.get_session(message["session_id"])["credential_owner"] == owner
        for message in adapter._session_db.get_messages(effective_session_id)
    )


def test_run_store_home_resolution_failure_is_degraded(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.config.get_hermes_home",
        lambda: (_ for _ in ()).throw(RuntimeError("home unavailable")),
    )
    store = RunIdempotencyStore()
    try:
        assert store.durability_state == "degraded"
        assert store.durable is False
    finally:
        store.close()


@pytest.mark.parametrize("db_path", ["", ":memory:"])
def test_run_store_implicit_sqlite_memory_or_temp_path_is_degraded(db_path):
    store = RunIdempotencyStore(db_path)
    try:
        assert store.durability_state == "degraded"
        assert store.durable is False
    finally:
        store.close()


@pytest.mark.asyncio
async def test_credential_admission_rejects_temporary_sqlite_store(monkeypatch):
    principal = _principal(APIServerOperation.RUNS_CREATE)
    adapter = _adapter(Authorizer(lambda _request: principal))
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore("")
    session_id = _create_owned_session(adapter, principal)
    monkeypatch.setattr(
        api_server_runs_module.uuid,
        "uuid4",
        lambda: (_ for _ in ()).throw(AssertionError("run ID allocation must not start")),
    )

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.post(
            "/v1/runs",
            json={"input": "hello", "session_id": session_id},
            headers={"Authorization": "Bearer credential"},
        )
        body = await response.json()

    assert response.status == 503
    assert body["error"]["code"] == "run_storage_unavailable"


@pytest.mark.asyncio
async def test_credential_session_replacement_after_preflight_is_fenced(
    tmp_path, monkeypatch
):
    """The owner validated at lease acquisition remains canonical through history and model work."""
    from hermes_state import SessionDB
    from hermes_state_errors import SessionTurnLeaseLostError

    principal = _principal(APIServerOperation.RUNS_CREATE)
    adapter = _adapter(Authorizer(lambda _request: principal))
    adapter._session_db = SessionDB(tmp_path / "state.db")
    session_id = _create_owned_session(adapter, principal)
    deleting_db = SessionDB(tmp_path / "state.db")
    owner = adapter._credential_owner_key(principal)
    replacement_attempted = asyncio.Event()

    async def history(sid):
        with pytest.raises(SessionTurnLeaseLostError):
            deleting_db.delete_session(sid)
        replacement_attempted.set()
        return []

    monkeypatch.setattr(adapter, "_conversation_history_for_session", history)
    seen = {}
    agent = MagicMock(session_id=session_id)

    def run_conversation(**_kwargs):
        seen["owner_during_model"] = deleting_db.get_session(session_id)["credential_owner"]
        return {"final_response": "done"}

    agent.run_conversation.side_effect = run_conversation
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.post(
            "/v1/runs",
            json={"input": "hello", "session_id": session_id},
            headers={"Authorization": "Bearer credential"},
        )
        body = await response.json()
        task = adapter._active_run_tasks.get(body.get("run_id"))
        if task is not None:
            await asyncio.wait_for(task, timeout=1)

    assert response.status == 202
    assert replacement_attempted.is_set()
    assert seen["owner_during_model"] == owner
    assert deleting_db.get_session(session_id)["credential_owner"] == owner


@pytest.mark.asyncio
async def test_cancelled_run_holds_session_lease_until_executor_worker_exits(
    tmp_path, monkeypatch
):
    from hermes_state import SessionDB
    from hermes_state_errors import SessionTurnLeaseLostError

    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    principal = _principal(APIServerOperation.RUNS_CREATE)
    adapter = _adapter(Authorizer(lambda _request: principal))
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore.in_memory()
    session_id = _create_owned_session(adapter, principal)
    session_db = adapter._ensure_session_db()
    assert session_db is not None
    owner = adapter._credential_owner_key(principal)
    contender = SessionDB(hermes_home / "state.db")
    worker_started = threading.Event()
    release_worker = threading.Event()
    session_db_closed = threading.Event()
    real_close = session_db.close

    def close_session_db():
        session_db_closed.set()
        real_close()

    monkeypatch.setattr(session_db, "close", close_session_db)
    agent = MagicMock(session_id=session_id)

    def run_conversation(**_kwargs):
        worker_started.set()
        assert release_worker.wait(timeout=5)
        return {"final_response": "done"}

    agent.run_conversation.side_effect = run_conversation
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)
    contender_acquired = False
    try:
        async with TestClient(TestServer(_credential_app(adapter))) as client:
            response = await client.post(
                "/v1/runs",
                json={"input": "hello", "session_id": session_id},
                headers={"Authorization": "Bearer credential"},
            )
            body = await response.json()
            task = adapter._active_run_tasks[body["run_id"]]
            assert await asyncio.to_thread(worker_started.wait, 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert adapter.active_agent_work_count() == 1
            await adapter.disconnect()

            assert session_db_closed.is_set() is False
            with pytest.raises(SessionTurnLeaseLostError):
                contender.delete_session(session_id)
            contender_acquired = contender.try_acquire_session_turn_lease(
                session_id, "contender", expected_credential_owner=owner
            )
            assert contender_acquired is False
            release_worker.set()
            for _ in range(100):
                contender_acquired = contender.try_acquire_session_turn_lease(
                    session_id, "contender", expected_credential_owner=owner
                )
                if contender_acquired:
                    break
                await asyncio.sleep(0.01)
            assert contender_acquired is True
            await asyncio.wait_for(asyncio.to_thread(session_db_closed.wait, 1), timeout=2)
    finally:
        release_worker.set()
        if contender_acquired:
            contender.release_session_turn_lease(session_id, "contender")
        contender.close()


@pytest.mark.asyncio
async def test_credential_session_recreated_foreign_before_lease_is_rejected(
    tmp_path, monkeypatch
):
    """A row swapped after preflight but before the transactional claim never reaches history/model."""
    from hermes_state import SessionDB

    principal = _principal(APIServerOperation.RUNS_CREATE)
    adapter = _adapter(Authorizer(lambda _request: principal))
    adapter._session_db = SessionDB(tmp_path / "state.db")
    session_id = _create_owned_session(adapter, principal)
    racing_db = SessionDB(tmp_path / "state.db")
    real_acquire = adapter._session_db.try_acquire_session_turn_lease
    replaced = False

    def replace_then_acquire(*args, **kwargs):
        nonlocal replaced
        if not replaced:
            replaced = True
            assert racing_db.delete_session(session_id)
            racing_db.create_session(
                session_id, source="api_server", credential_owner="api-credential:foreign"
            )
        return real_acquire(*args, **kwargs)

    monkeypatch.setattr(
        adapter._session_db, "try_acquire_session_turn_lease", replace_then_acquire
    )
    monkeypatch.setattr(
        adapter, "_conversation_history_for_session",
        AsyncMock(side_effect=AssertionError("history must not load")),
    )
    monkeypatch.setattr(
        adapter, "_create_agent", MagicMock(side_effect=AssertionError("model must not start"))
    )

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.post(
            "/v1/runs",
            json={"input": "hello", "session_id": session_id},
            headers={"Authorization": "Bearer credential"},
        )
        body = await response.json()

    assert replaced is True
    assert response.status == 404
    assert body["error"]["code"] == "session_not_found"


@pytest.mark.asyncio
async def test_foreign_session_created_during_run_id_allocation_cannot_be_attached(
    monkeypatch,
):
    principal = _principal(APIServerOperation.RUNS_CREATE)
    adapter = _adapter(Authorizer(lambda _request: principal))
    owned_session = _create_owned_session(adapter, principal)
    candidate = f"run_{'a' * 32}"
    monkeypatch.setattr(
        api_server_runs_module.uuid,
        "uuid4",
        lambda: type("U", (), {"hex": "a" * 32})(),
    )
    real_has_run_id = adapter._run_idempotency_store.has_run_id

    def create_foreign_session_then_check(run_id):
        adapter._ensure_session_db().create_session(
            run_id, "api_server", credential_owner="api-credential:foreign"
        )
        return real_has_run_id(run_id)

    monkeypatch.setattr(
        adapter._run_idempotency_store, "has_run_id", create_foreign_session_then_check
    )
    seen = {}
    agent = MagicMock(session_id=owned_session)
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0

    def create_agent(**kwargs):
        seen.update(kwargs)
        return agent

    monkeypatch.setattr(adapter, "_create_agent", create_agent)
    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.post(
            "/v1/runs",
            json={"input": "hello", "session_id": owned_session},
            headers={"Authorization": "Bearer credential"},
        )
        body = await response.json()
        task = adapter._active_run_tasks.get(body.get("run_id"))
        if task is not None:
            await asyncio.wait_for(task, timeout=1)

    assert response.status == 202
    assert body["run_id"] == candidate
    assert seen["session_id"] == owned_session


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
async def test_authorizer_timeout_is_bounded():
    """Async authorizer respects the configured deadline."""
    async def stalled_async(_request):
        await asyncio.sleep(60)

    adapter = _adapter(AsyncAuthorizer(stalled_async))
    adapter._API_CREDENTIAL_AUTH_TIMEOUT_SECONDS = 0.01

    async def handler(_request):
        raise AssertionError("handler must not run")

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        response = await asyncio.wait_for(
            client.get(
                "/v1/capabilities",
                headers={"Authorization": "Bearer stalled-secret"},
            ),
            timeout=1,
        )
    assert response.status == 401


@pytest.mark.asyncio
async def test_lingering_async_authorizer_saturates_dedicated_capacity_without_queueing():
    """A cancellation-resistant async task holds the slot; the second request gets 401 immediately."""
    release = asyncio.Event()
    calls = []

    async def authorize(_request):
        calls.append("started")
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await release.wait()

    adapter = _adapter(AsyncAuthorizer(authorize))
    adapter._API_CREDENTIAL_AUTH_TIMEOUT_SECONDS = 0.01
    adapter._API_CREDENTIAL_AUTH_MAX_INFLIGHT = 1

    async def handler(_request):
        raise AssertionError("handler must not run")

    first_task = None
    try:
        async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
            first_task = asyncio.create_task(client.get(
                "/v1/capabilities", headers={"Authorization": "Bearer first"}))
            done, _ = await asyncio.wait({first_task}, timeout=0.5)
            if not done:
                release.set()
                await asyncio.wait_for(first_task, timeout=1)
            assert done, "request must return after deadline even with cancellation-resistant authorizer"
            first = first_task.result()
            # Slot is still held — second request must fail immediately.
            second = await asyncio.wait_for(client.get(
                "/v1/capabilities", headers={"Authorization": "Bearer second"}), timeout=0.2)
            assert first.status == second.status == 401
            assert calls == ["started"]
    finally:
        release.set()
        if first_task is not None and not first_task.done():
            await asyncio.wait_for(first_task, timeout=1)


@pytest.mark.asyncio
async def test_cancellation_resistant_async_authorizer_returns_at_deadline_and_saturates_capacity():
    """Deadline is observed, cancellation-resistant task holds its slot until it actually finishes,
    and saturation persists across adapter disconnect/reconnect, plugin unload, and a second adapter."""
    release = asyncio.Event()
    cancellation_seen = asyncio.Event()
    calls = []

    async def authorize(_request):
        calls.append("started")
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()

    adapter = _adapter(AsyncAuthorizer(authorize))
    adapter._API_CREDENTIAL_AUTH_TIMEOUT_SECONDS = 0.01
    adapter._API_CREDENTIAL_AUTH_MAX_INFLIGHT = 1

    async def handler(_request):
        raise AssertionError("handler must not run")

    first_task = None
    try:
        async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
            first_task = asyncio.create_task(client.get(
                "/v1/capabilities", headers={"Authorization": "Bearer first"}))
            done, _ = await asyncio.wait({first_task}, timeout=0.2)
            if not done:
                release.set()
                await asyncio.wait_for(first_task, timeout=1)
            assert done, "request deadline waited for cancellation-resistant authorizer cleanup"
            first = first_task.result()
            assert cancellation_seen.is_set()

            # Slot is held — saturation persists while the task has not finished.
            second = await asyncio.wait_for(client.get(
                "/v1/capabilities", headers={"Authorization": "Bearer second"}), timeout=0.2)
            assert first.status == second.status == 401
            assert calls == ["started"]

        # Simulate adapter disconnect — runner must NOT be discarded.
        runner_before = adapter._api_credential_authorizer_runner
        await adapter.disconnect()
        assert adapter._api_credential_authorizer_runner is runner_before, (
            "disconnect() must not clear the process-level runner reference"
        )

        # Saturation persists across the disconnect for the same runner.
        # (Slot still held because the lingering task has not exited yet — release.set() below.)
        from gateway.platforms.api_server import _CredentialAuthorizerSaturated
        with pytest.raises(_CredentialAuthorizerSaturated):
            runner_before._acquire()

        # A second adapter using the same process runner also sees saturation.
        adapter2 = _adapter(AsyncAuthorizer(authorize))
        adapter2._api_credential_authorizer_runner = runner_before
        with pytest.raises(_CredentialAuthorizerSaturated):
            runner_before._acquire()

    finally:
        release.set()
        if first_task is not None and not first_task.done():
            await asyncio.wait_for(first_task, timeout=1)


@pytest.mark.asyncio
async def test_plugin_manager_shutdown_cancels_runner_and_cannot_replace_it():
    manager = PluginManager()
    created = []

    def factory(capacity):
        runner = _CredentialAuthorizerRunner(capacity)
        created.append(runner)
        return runner

    runner = manager.get_api_server_credential_authorizer_runner(capacity=1, factory=factory)
    started = asyncio.Event()

    async def authorize(_request):
        started.set()
        await asyncio.sleep(60)

    in_flight = asyncio.create_task(runner.run(authorize, object(), timeout=60))
    await asyncio.wait_for(started.wait(), timeout=1)
    manager.shutdown()
    manager.shutdown()

    with pytest.raises(asyncio.CancelledError):
        await in_flight
    with pytest.raises(RuntimeError, match="shut down"):
        manager.get_api_server_credential_authorizer_runner(capacity=1, factory=factory)
    assert created == [runner]


@pytest.mark.asyncio
async def test_authorizer_runner_cancels_child_when_request_owner_is_cancelled():
    runner = _CredentialAuthorizerRunner(1)
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()

    async def authorize(_request):
        started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()

    owner = asyncio.create_task(runner.run(authorize, object(), timeout=60))
    await asyncio.wait_for(started.wait(), timeout=1)
    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    try:
        await asyncio.wait_for(cancelled.wait(), timeout=0.1)
        child_was_cancelled = True
    except asyncio.TimeoutError:
        child_was_cancelled = False
    finally:
        runner.close()
        release.set()
        for _ in range(20):
            if runner._active == 0:
                break
            await asyncio.sleep(0)
    assert child_was_cancelled
    assert runner._active == 0


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


def test_generated_run_id_allocation_exhaustion_is_bounded(monkeypatch):
    adapter = _adapter(None)
    api_server_runs_module._initialize_run_state(adapter, store_factory=lambda: MagicMock())
    adapter._run_idempotency_scope = lambda _request=None: "scope"
    assert adapter._ensure_session_db() is not None
    attempts = []

    def collide_uuid():
        attempts.append(1)
        if len(attempts) > 100:
            raise AssertionError("run ID allocation did not terminate within a bounded budget")
        return type("U", (), {"hex": "deadbeef" * 4})()

    monkeypatch.setattr(api_server_runs_module.uuid, "uuid4", collide_uuid)
    adapter._run_owners[f"run_{'deadbeef' * 4}"] = "already-owned"

    reserved = api_server_runs_module._reserve_generated_run_id(adapter, "scope")

    assert reserved is None
    assert 0 < len(attempts) <= 100


def test_generated_run_id_skips_every_live_and_durable_run_namespace(monkeypatch):
    adapter = _adapter(None)
    assert adapter._ensure_session_db() is not None
    live_status_id = f"run_{'a' * 32}"
    durable_id = f"run_{'b' * 32}"
    available = f"run_{'c' * 32}"
    adapter._run_statuses[live_status_id] = {"status": "completed"}
    adapter._run_idempotency_store.reserve(
        "foreign-scope", "foreign-key", "fingerprint", durable_id,
        {"status": "completed", "run_id": durable_id},
    )
    values = iter(("a" * 32, "b" * 32, "c" * 32))
    monkeypatch.setattr(
        api_server_runs_module.uuid, "uuid4",
        lambda: type("U", (), {"hex": next(values)})(),
    )

    reserved = api_server_runs_module._reserve_generated_run_id(adapter, "scope")

    assert reserved == available
    assert adapter._run_owners == {available: "scope"}


@pytest.mark.asyncio
async def test_generated_run_id_endpoint_exhaustion_is_generic(monkeypatch):
    principal = _principal(APIServerOperation.RUNS_CREATE)
    adapter = _adapter(Authorizer(lambda _request: principal))
    session_id = _create_owned_session(adapter, principal)
    collided = f"run_{'d' * 32}"
    adapter._run_idempotency_store.reserve(
        "foreign-scope", "foreign-key", "fingerprint", collided,
        {"status": "completed", "run_id": collided},
    )
    attempts = []

    def collide_uuid():
        attempts.append(1)
        if len(attempts) > 100:
            raise AssertionError("run ID allocation did not terminate within a bounded budget")
        return type("U", (), {"hex": "d" * 32})()

    monkeypatch.setattr(api_server_runs_module.uuid, "uuid4", collide_uuid)
    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.post(
            "/v1/runs", json={"input": "hello", "session_id": session_id},
            headers={"Authorization": "Bearer owner"},
        )
        body = await response.json()

    assert response.status == 503
    assert body["error"]["code"] == "run_id_allocation_failed"
    assert collided not in str(body)
    assert 0 < len(attempts) <= 100


@pytest.mark.asyncio
async def test_generated_run_id_endpoint_fails_closed_when_durable_status_store_unavailable(
    tmp_path, monkeypatch
):
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    monkeypatch.setattr(
        "hermes_cli.profiles.profiles_to_serve",
        lambda **_kwargs: [("default", default_home)],
    )
    principal = _principal(APIServerOperation.RUNS_CREATE)
    adapter = _adapter(Authorizer(lambda _request: principal))
    session_id = _create_owned_session(adapter, principal)
    monkeypatch.setattr(
        adapter._run_idempotency_store, "has_run_id",
        lambda _run_id: (_ for _ in ()).throw(OSError("status store unavailable")),
    )

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.post(
            "/v1/runs", json={"input": "hello", "session_id": session_id},
            headers={"Authorization": "Bearer owner"},
        )
        body = await response.json()

    assert response.status == 503
    assert body["error"]["code"] == "run_id_allocation_failed"
    assert "status store" not in str(body).lower()


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
async def test_concurrent_credential_runs_cannot_claim_nonexistent_shared_session_id(monkeypatch):
    adapter = _adapter(Authorizer(
        lambda request: _principal(APIServerOperation.RUNS_CREATE, scope=request.bearer)
    ))
    created_agents = []
    monkeypatch.setattr(adapter, "_create_agent", lambda **kw: created_agents.append(kw))

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        async def submit(scope):
            return await client.post(
                "/v1/runs", json={"input": "hello", "session_id": "client-shared"},
                headers={"Authorization": f"Bearer {scope}"},
            )

        responses = await asyncio.gather(submit("scope-a"), submit("scope-b"))
        bodies = await asyncio.gather(*(response.json() for response in responses))

    assert [response.status for response in responses] == [404, 404]
    assert all(body["error"]["code"] == "session_not_found" for body in bodies)
    assert adapter._ensure_session_db().get_session("client-shared") is None
    assert created_agents == []


@pytest.mark.asyncio
async def test_credential_run_accepts_only_a_previously_owner_stamped_session(monkeypatch):
    operations = (APIServerOperation.SESSIONS_CREATE, APIServerOperation.RUNS_CREATE)
    adapter = _adapter(Authorizer(
        lambda request: _principal_with_operations(*operations, scope=request.bearer)
    ))
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        created = await client.post(
            "/api/sessions", json={}, headers={"Authorization": "Bearer owner-a"}
        )
        session_id = (await created.json())["session"]["id"]
        own = await client.post(
            "/v1/runs", json={"input": "hello", "session_id": session_id},
            headers={"Authorization": "Bearer owner-a"},
        )
        foreign = await client.post(
            "/v1/runs", json={"input": "hello", "session_id": session_id},
            headers={"Authorization": "Bearer owner-b"},
        )

    assert own.status == 202
    assert foreign.status == 404


@pytest.mark.asyncio
async def test_concurrent_server_id_collision_creates_only_one_owned_session(monkeypatch):
    adapter = _adapter(Authorizer(
        lambda request: _principal(APIServerOperation.SESSIONS_CREATE, scope=request.bearer)
    ))
    monkeypatch.setattr(api_server_module.time, "time", lambda: 1234)
    monkeypatch.setattr(
        api_server_module.uuid, "uuid4", lambda: type("U", (), {"hex": "a" * 32})()
    )

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        responses = await asyncio.gather(*(
            client.post(
                "/api/sessions", json={},
                headers={"Authorization": f"Bearer owner-{index}"},
            )
            for index in range(2)
        ))
        bodies = await asyncio.gather(*(response.json() for response in responses))

    assert sorted(response.status for response in responses) == [201, 409]
    conflict_body = bodies[[response.status for response in responses].index(409)]
    assert conflict_body["error"]["code"] == "session_exists"
    assert "api_1234_aaaaaaaa" not in str(conflict_body)
    assert adapter._ensure_session_db().get_session("api_1234_aaaaaaaa")["credential_owner"].startswith(
        "api-credential:"
    )


@pytest.mark.asyncio
async def test_same_generated_session_id_is_isolated_across_served_profiles(
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
    monkeypatch.setattr(api_server_module.time, "time", lambda: 1234)
    monkeypatch.setattr(
        api_server_module.uuid, "uuid4", lambda: type("U", (), {"hex": "b" * 32})()
    )
    adapter = _adapter(Authorizer(
        lambda request: _principal(
            APIServerOperation.SESSIONS_CREATE,
            profile="worker" if request.bearer == "worker-owner" else "default",
            scope=request.bearer,
        )
    ), multiplex=True)

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        default = await client.post(
            "/api/sessions", json={},
            headers={"Authorization": "Bearer default-owner"},
        )
        worker = await client.post(
            "/api/sessions", json={},
            headers={"Authorization": "Bearer worker-owner"},
        )
        default_id = (await default.json())["session"]["id"]
        worker_id = (await worker.json())["session"]["id"]

    assert default.status == worker.status == 201
    assert default_id == worker_id


@pytest.mark.asyncio
async def test_generated_run_id_collision_is_reserved_before_cross_owner_publication(monkeypatch):
    principals = {
        owner: _principal(APIServerOperation.RUNS_CREATE, scope=owner)
        for owner in ("owner-a", "owner-b")
    }
    adapter = _adapter(Authorizer(lambda request: principals[request.bearer]))
    sessions = {
        owner: _create_owned_session(adapter, principal, f"session-{owner}")
        for owner, principal in principals.items()
    }
    values = iter(("a" * 32, "a" * 32, "b" * 32))
    monkeypatch.setattr(
        api_server_runs_module.uuid, "uuid4",
        lambda: type("U", (), {"hex": next(values)})(),
    )
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        first = await client.post(
            "/v1/runs", json={"input": "one", "session_id": sessions["owner-a"]},
            headers={"Authorization": "Bearer owner-a"},
        )
        second = await client.post(
            "/v1/runs", json={"input": "two", "session_id": sessions["owner-b"]},
            headers={"Authorization": "Bearer owner-b"},
        )
        first_body, second_body = await asyncio.gather(first.json(), second.json())

    assert first.status == second.status == 202
    assert first_body["run_id"] == f"run_{'a' * 32}"
    assert second_body["run_id"] == f"run_{'b' * 32}"
    assert adapter._run_owners[first_body["run_id"]] != adapter._run_owners[second_body["run_id"]]


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
async def test_authorizer_runner_singleton_survives_reconnect_and_unload():
    """The process-level runner must not be discarded on adapter disconnect or plugin unload."""
    adapter = _adapter(AsyncAuthorizer(lambda _request: asyncio.sleep(0)))
    first = adapter._credential_authorizer_runner()
    second = adapter._credential_authorizer_runner()
    assert first is second

    runner_ref = first
    await adapter.disconnect()
    # Disconnect must NOT clear the adapter's runner reference (it's the process singleton).
    assert adapter._api_credential_authorizer_runner is runner_ref, (
        "disconnect() cleared the process-level runner — this breaks saturation invariants"
    )

    # Calling the accessor again still returns the same object.
    third = adapter._credential_authorizer_runner()
    assert third is runner_ref


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
async def test_concurrent_manager_get_or_create_returns_same_runner():
    """Concurrent calls to get_api_server_credential_authorizer_runner must return the same instance."""
    manager = PluginManager()
    results = []

    async def fetch():
        runner = manager.get_api_server_credential_authorizer_runner(
            capacity=4, factory=_CredentialAuthorizerRunner
        )
        results.append(runner)

    await asyncio.gather(*[fetch() for _ in range(16)])
    assert len(set(id(r) for r in results)) == 1, (
        "get_api_server_credential_authorizer_runner returned multiple distinct runner instances"
    )


@pytest.mark.asyncio
async def test_adapter_retrieval_failure_fails_auth_closed(monkeypatch):
    """If the process runner cannot be retrieved, auth must fail closed — no local fallback."""
    def _boom():
        raise RuntimeError("plugin manager unavailable")

    monkeypatch.setattr("hermes_cli.plugins.get_plugin_manager", _boom)
    adapter = _adapter(AsyncAuthorizer(lambda _request: asyncio.sleep(0)))
    # Clear any cached runner so the accessor must call get_plugin_manager.
    adapter._api_credential_authorizer_runner = None

    async def handler(_request):
        raise AssertionError("handler must not run")

    async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
        response = await client.get(
            "/v1/capabilities", headers={"Authorization": "Bearer any-bearer"}
        )
    assert response.status == 401, (
        "runner retrieval failure must yield 401, not a local-fallback 200"
    )


@pytest.mark.asyncio
async def test_slot_recovers_after_lingering_task_finishes():
    """After a cancellation-resistant task truly exits, its capacity slot is released."""
    release = asyncio.Event()
    admitted = asyncio.Event()
    invocations = 0

    async def authorize(request):
        nonlocal invocations
        invocations += 1
        if invocations > 1:
            return _principal(request.operation)
        admitted.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await release.wait()

    adapter = _adapter(AsyncAuthorizer(authorize))
    adapter._API_CREDENTIAL_AUTH_TIMEOUT_SECONDS = 0.01
    adapter._API_CREDENTIAL_AUTH_MAX_INFLIGHT = 1

    async def handler(request):
        return web.json_response({"ok": True})

    first_task = None
    try:
        async with TestClient(TestServer(_auth_app(adapter, handler))) as client:
            first_task = asyncio.create_task(client.get(
                "/v1/capabilities", headers={"Authorization": "Bearer first"}))
            await asyncio.wait_for(admitted.wait(), timeout=1)
            # Wait for deadline to fire (returns 401, slot still held by lingering task).
            done, _ = await asyncio.wait({first_task}, timeout=0.5)
            if not done:
                release.set()
                await asyncio.wait_for(first_task, timeout=1)
            assert done
            assert (await first_task).status == 401

            # Now release the lingering task and wait briefly for done callback.
            release.set()
            await asyncio.sleep(0.05)

            # Slot must now be free — next request should be authorized normally.
            second = await asyncio.wait_for(client.get(
                "/v1/capabilities", headers={"Authorization": "Bearer second"}), timeout=0.5)
            assert invocations == 2
            assert second.status == 200
    finally:
        release.set()
        if first_task is not None and not first_task.done():
            await asyncio.wait_for(first_task, timeout=1)


@pytest.mark.asyncio
async def test_run_status_mapping_and_owner_enforcement(monkeypatch):
    active_scope = ["scope-a"]
    operations = (APIServerOperation.RUNS_CREATE, APIServerOperation.RUN_STATUS_READ)
    principal = _principal_with_operations(*operations, scope="scope-a")
    adapter = _adapter(Authorizer(
        lambda _request: _principal_with_operations(*operations, scope=active_scope[0])
    ))
    session_id = _create_owned_session(adapter, principal)
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        created = await client.post(
            "/v1/runs", json={"input": "hello", "session_id": session_id},
            headers={"Authorization": "Bearer owner-a"},
        )
        run_id = (await created.json())["run_id"]
        own = await client.get(
            f"/v1/runs/{run_id}", headers={"Authorization": "Bearer owner-a"}
        )
        active_scope[0] = "scope-b"
        foreign = await client.get(
            f"/v1/runs/{run_id}", headers={"Authorization": "Bearer owner-b"}
        )

    assert own.status == 200
    assert foreign.status == 404


@pytest.mark.asyncio
async def test_same_origin_cross_profile_run_lookup_isolated_for_status_and_events(
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

    operations = (
        APIServerOperation.RUNS_CREATE,
        APIServerOperation.RUN_STATUS_READ,
        APIServerOperation.RUN_EVENTS_READ,
    )

    def authorize(request):
        profile = "worker" if request.bearer.startswith("worker") else "default"
        return _principal_with_operations(*operations, profile=profile, scope=request.bearer)

    adapter = _adapter(Authorizer(authorize), multiplex=True)
    default_principal = authorize(type("Request", (), {"bearer": "default-token"})())
    default_session = _create_owned_session(adapter, default_principal, "default-session")
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)

    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_post("/p/{profile}/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/p/{profile}/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_get("/p/{profile}/v1/runs/{run_id}/events", adapter._handle_run_events)

    async with TestClient(TestServer(app)) as client:
        created = await client.post(
            "/v1/runs",
            json={"input": "hello", "session_id": default_session},
            headers={"Authorization": "Bearer default-token"},
        )
        run_id = (await created.json())["run_id"]

        default_status = await client.get(
            f"/v1/runs/{run_id}",
            headers={"Authorization": "Bearer default-token"},
        )
        worker_status = await client.get(
            f"/p/worker/v1/runs/{run_id}",
            headers={"Authorization": "Bearer worker-token"},
        )
        worker_events = await client.get(
            f"/p/worker/v1/runs/{run_id}/events",
            headers={"Authorization": "Bearer worker-token"},
        )
        default_events = await client.get(
            f"/v1/runs/{run_id}/events",
            headers={"Authorization": "Bearer default-token"},
        )
        default_events_text = await default_events.text()

    assert created.status == 202
    assert default_status.status == 200
    assert worker_status.status == 404
    assert worker_events.status == 404
    assert default_events.status == 200
    assert f'"run_id": "{run_id}"' in default_events_text


@pytest.mark.asyncio
async def test_same_origin_cross_profile_concurrent_generated_run_collision_retries(
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
    operations = (APIServerOperation.RUNS_CREATE,)

    def authorize(request):
        profile = "worker" if request.bearer == "worker-token" else "default"
        return _principal_with_operations(*operations, profile=profile, scope=request.bearer)

    adapter = _adapter(Authorizer(authorize), multiplex=True)
    default_principal = authorize(type("Request", (), {"bearer": "default-token"})())
    default_session = _create_owned_session(adapter, default_principal, "default-session")
    worker_principal = authorize(type("Request", (), {"bearer": "worker-token"})())
    with adapter._profile_scope("worker"):
        worker_session = _create_owned_session(adapter, worker_principal, "worker-session")
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)
    values = iter(("a" * 32, "a" * 32, "b" * 32))
    monkeypatch.setattr(
        api_server_runs_module.uuid, "uuid4",
        lambda: type("U", (), {"hex": next(values)})(),
    )
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_post("/p/{profile}/v1/runs", adapter._handle_runs)

    async with TestClient(TestServer(app)) as client:
        default_response, worker_response = await asyncio.gather(
            client.post(
                "/v1/runs",
                json={"input": "default", "session_id": default_session},
                headers={"Authorization": "Bearer default-token"},
            ),
            client.post(
                "/p/worker/v1/runs",
                json={"input": "worker", "session_id": worker_session},
                headers={"Authorization": "Bearer worker-token"},
            ),
        )
        default_body, worker_body = await asyncio.gather(
            default_response.json(), worker_response.json()
        )

    assert default_response.status == worker_response.status == 202
    assert {default_body["run_id"], worker_body["run_id"]} == {
        f"run_{'a' * 32}", f"run_{'b' * 32}"
    }


@pytest.mark.asyncio
async def test_credential_idempotency_isolated_by_owner(monkeypatch):
    active_scope = ["scope-a"]
    adapter = _adapter(Authorizer(
        lambda _request: _principal(APIServerOperation.RUNS_CREATE, scope=active_scope[0])
    ))
    session_a = _create_owned_session(
        adapter, _principal(APIServerOperation.RUNS_CREATE, scope="scope-a"), "session-a"
    )
    session_b = _create_owned_session(
        adapter, _principal(APIServerOperation.RUNS_CREATE, scope="scope-b"), "session-b"
    )
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)
    headers = lambda token: {"Authorization": f"Bearer {token}", "Idempotency-Key": "same-key"}

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        first = await client.post(
            "/v1/runs", json={"input": "one", "session_id": session_a}, headers=headers("owner-a")
        )
        first_id = (await first.json())["run_id"]
        replay = await client.post(
            "/v1/runs", json={"input": "one", "session_id": session_a},
            headers=headers("owner-a-rotated"),
        )
        replay_body = await replay.json()
        conflict = await client.post(
            "/v1/runs", json={"input": "changed", "session_id": session_a},
            headers=headers("owner-a")
        )
        active_scope[0] = "scope-b"
        isolated = await client.post(
            "/v1/runs", json={"input": "different", "session_id": session_b},
            headers=headers("owner-b"),
        )
        isolated_id = (await isolated.json())["run_id"]

    assert replay_body == {
        "run_id": first_id, "session_id": session_a,
        "status": replay_body["status"], "replayed": True,
    }
    assert conflict.status == 409
    assert isolated.status == 202
    assert isolated_id != first_id


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


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["bind", "cleanup"])
async def test_run_agent_restores_executor_auth_context_after_setup_or_cleanup_failure(
    failure, monkeypatch
):
    adapter = _adapter(None)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    leaked = api_server_module._CredentialAuthContext(
        _principal(APIServerOperation.RUNS_CREATE), "owner-key"
    )
    token = api_server_module._api_request_auth_context.set(leaked)
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)
    if failure == "bind":
        monkeypatch.setattr(
            adapter, "_bind_api_server_session",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("bind failed")),
        )
    else:
        monkeypatch.setattr(
            "gateway.session_context.clear_session_vars",
            lambda _tokens: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
        )
    try:
        with pytest.raises(RuntimeError, match=failure):
            await adapter._run_agent("hello", [])
        observed = await loop.run_in_executor(
            None, api_server_module._api_request_auth_context.get
        )
    finally:
        api_server_module._api_request_auth_context.reset(token)
        executor.shutdown(wait=False, cancel_futures=True)

    assert observed is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_cleanup", ["ownership", "declared_binding"])
async def test_run_agent_clears_all_session_context_after_each_cleanup_failure(
    failed_cleanup, monkeypatch
):
    from gateway import session_context

    adapter = _adapter(None)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    agent = MagicMock(session_id="rotated-session")
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)
    calls = []

    def clear_ownership(_agent):
        calls.append("ownership")
        if failed_cleanup == "ownership":
            raise RuntimeError("ownership cleanup failed")

    def bind_declared(_session_id, _session_key):
        calls.append("declared_binding")
        if failed_cleanup == "declared_binding":
            raise RuntimeError("declared binding cleanup failed")

    monkeypatch.setattr(api_server_module, "_clear_turn_process_ownership", clear_ownership)
    monkeypatch.setattr(adapter, "_bind_declared_conversation", bind_declared)

    def observe_session_context():
        return {
            var.name: session_context.get_session_env(var.name, "missing")
            for var in session_context._SESSION_VARS
        }

    try:
        with pytest.raises(RuntimeError, match=failed_cleanup.replace("_", " ")):
            await adapter._run_agent(
                "hello", [], session_id="bound-session",
                gateway_session_key="declared-key", bind_declared_conversation=True,
            )
        observed = await loop.run_in_executor(None, observe_session_context)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    assert sorted(calls) == ["declared_binding", "ownership"]
    assert set(observed.values()) == {""}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_cleanup",
    [
        "ownership", "declared_binding", "notify", "room_policy",
        "session_vars", "approval_session", "auth_context",
    ],
)
async def test_runs_executor_clears_every_context_after_each_cleanup_failure(
    failed_cleanup, monkeypatch
):
    from gateway import hosted_room_execution_policy, session_context
    from tools import approval, approval_context

    adapter = _adapter(None)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    leaked = api_server_module._CredentialAuthContext(
        _principal(APIServerOperation.RUNS_CREATE), "owner-key"
    )
    policy = hosted_room_execution_policy.execution_policy_mapping(
        target_profile="default", config={"agent": {}, "approvals": {}}
    )
    run = api_server_runs_module._RunLaunch(
        owner=adapter,
        run_id="run_cleanup_failure",
        queue=asyncio.Queue(),
        session_id="session_cleanup_failure",
        gateway_session_key="declared-key",
        declared_selected=True,
        user_message="hello",
        conversation_history=[],
        agent_kwargs={
            "room_dispatch": {"room_id": "room-one"},
            "room_execution_policy": policy,
        },
        request_profile=None,
        request_auth_context=leaked,
        browser_control_principal="browser-principal",
        browser_control_transport_family="cloud",
    )
    agent = MagicMock(session_id="rotated-session")
    agent.run_conversation.return_value = {"final_response": "done"}
    calls = []

    def cleanup(name, real=lambda *_args: None):
        def wrapped(*args):
            calls.append(name)
            if failed_cleanup == name:
                raise RuntimeError(f"{name} cleanup failed")
            return real(*args)
        return wrapped

    monkeypatch.setattr(
        api_server_module, "_clear_turn_process_ownership", cleanup("ownership")
    )
    monkeypatch.setattr(
        adapter, "_bind_declared_conversation", cleanup("declared_binding")
    )
    real_unregister = approval.unregister_gateway_notify
    monkeypatch.setattr(
        approval, "unregister_gateway_notify",
        cleanup("notify", real_unregister),
    )
    monkeypatch.setattr(
        hosted_room_execution_policy, "reset_room_execution_policy",
        cleanup("room_policy", hosted_room_execution_policy.reset_room_execution_policy),
    )
    monkeypatch.setattr(
        session_context, "clear_session_vars",
        cleanup("session_vars", session_context.clear_session_vars),
    )
    monkeypatch.setattr(
        approval_context, "reset_current_session_key",
        cleanup("approval_session", approval_context.reset_current_session_key),
    )

    real_auth_context = api_server_module._api_request_auth_context

    class AuthContext:
        def get(self):
            return real_auth_context.get()

        def set(self, value):
            return real_auth_context.set(value)

        def reset(self, token):
            calls.append("auth_context")
            if failed_cleanup == "auth_context":
                raise RuntimeError("auth_context cleanup failed")
            return real_auth_context.reset(token)

    monkeypatch.setattr(api_server_module, "_api_request_auth_context", AuthContext())

    def observe_contexts():
        return {
            "session": {var.name: var.get() for var in session_context._SESSION_VARS},
            "async_delivery": session_context._SESSION_ASYNC_DELIVERY.get(),
            "approval": approval_context._approval_session_key.get(),
            "room": hosted_room_execution_policy.current_room_execution_policy(),
            "auth": real_auth_context.get(),
        }

    try:
        with pytest.raises(RuntimeError, match=f"{failed_cleanup} cleanup failed"):
            await loop.run_in_executor(
                None,
                lambda: api_server_runs_module._run_agent_sync(
                    adapter, run, agent, lambda _event: None,
                    _api_server=api_server_module,
                ),
            )
        observed = await loop.run_in_executor(None, observe_contexts)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        real_unregister(run.approval_session_key)

    assert sorted(calls) == sorted([
        "ownership", "declared_binding", "notify", "room_policy",
        "session_vars", "approval_session", "auth_context",
    ])
    assert set(observed["session"].values()) == {""}
    assert observed["async_delivery"] is session_context._UNSET
    assert observed["approval"] == ""
    assert observed["room"] is None
    assert observed["auth"] is None


@pytest.mark.asyncio
async def test_run_events_reject_same_profile_cross_scope_run_ids(monkeypatch):
    active_scope = ["scope-a"]
    operations = frozenset({
        APIServerOperation.RUNS_CREATE,
        APIServerOperation.RUN_EVENTS_READ,
    })

    def authorize(_request):
        principal = _principal(APIServerOperation.RUNS_CREATE, scope=active_scope[0])
        return AuthorizedAPICredential(
            principal_id=principal.principal_id,
            runtime_profile=principal.runtime_profile,
            agent_profile_id=principal.agent_profile_id,
            credential_scope_id=principal.credential_scope_id,
            allowed_operations=operations,
        )

    adapter = _adapter(Authorizer(authorize))
    session_id = _create_owned_session(
        adapter, _principal(APIServerOperation.RUNS_CREATE, scope="scope-a")
    )
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)

    async with TestClient(TestServer(app)) as client:
        created = await client.post(
            "/v1/runs",
            json={"input": "hello", "session_id": session_id},
            headers={"Authorization": "Bearer rotating-token-a"},
        )
        run_id = (await created.json())["run_id"]
        active_scope[0] = "scope-b"
        foreign = await client.get(
            f"/v1/runs/{run_id}/events",
            headers={"Authorization": "Bearer rotating-token-b"},
        )

    assert created.status == 202
    assert foreign.status == 404


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


@pytest.mark.asyncio
async def test_run_agent_preclear_failure_clears_session_and_browser_context_on_reused_worker(
    monkeypatch,
):
    from gateway import session_context

    adapter = _adapter(None)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    agent = MagicMock(session_id="bound-session")
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)
    monkeypatch.setattr(
        session_context,
        "clear_session_vars",
        lambda _tokens: (_ for _ in ()).throw(RuntimeError("pre-clear failed")),
    )

    def observe():
        return (
            {var.name: var.get() for var in session_context._SESSION_VARS},
            session_context._SESSION_ASYNC_DELIVERY.get(),
        )

    try:
        with pytest.raises(RuntimeError, match="pre-clear failed"):
            await adapter._run_agent(
                "hello", [], session_id="bound-session",
                gateway_session_key="bound-key",
            )
        values, async_delivery = await loop.run_in_executor(None, observe)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    assert set(values.values()) == {""}
    assert async_delivery is session_context._UNSET


@pytest.mark.asyncio
async def test_run_agent_auth_reset_failure_sets_none_on_reused_worker(monkeypatch):
    adapter = _adapter(None)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)
    leaked = api_server_module._CredentialAuthContext(
        _principal(APIServerOperation.RUNS_CREATE), "owner-key"
    )
    token = api_server_module._api_request_auth_context.set(leaked)
    real_var = api_server_module._api_request_auth_context
    agent = MagicMock(session_id="bound-session")
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    monkeypatch.setattr(adapter, "_create_agent", lambda **_kwargs: agent)

    class ResetFailure:
        def get(self):
            return real_var.get()

        def set(self, value):
            return real_var.set(value)

        def reset(self, _token):
            raise RuntimeError("auth reset failed")

    monkeypatch.setattr(api_server_module, "_api_request_auth_context", ResetFailure())
    try:
        with pytest.raises(RuntimeError, match="auth reset failed"):
            await adapter._run_agent("hello", [], session_id="bound-session")
        observed = await loop.run_in_executor(None, real_var.get)
    finally:
        real_var.reset(token)
        executor.shutdown(wait=False, cancel_futures=True)

    assert observed is None


def test_expected_profile_key_reset_failure_forces_safe_baseline(monkeypatch):
    adapter = _adapter(None)
    real_var = api_server_module._api_request_profile

    class ResetFailure:
        def get(self):
            return real_var.get()

        def set(self, value):
            return real_var.set(value)

        def reset(self, _token):
            raise RuntimeError("profile reset failed")

    monkeypatch.setattr(api_server_module, "_api_request_profile", ResetFailure())
    monkeypatch.setattr(adapter, "_expected_api_key", lambda: "key")

    try:
        with pytest.raises(RuntimeError, match="profile reset failed"):
            adapter._expected_api_key_for_profile("worker")
        assert real_var.get() is None
    finally:
        real_var.set(None)


@pytest.mark.asyncio
async def test_agent_request_reservation_reset_failure_forces_safe_baseline(monkeypatch):
    adapter = _adapter(None)
    real_var = api_server_module._api_agent_request_reservation

    class ResetFailure:
        def get(self):
            return real_var.get()

        def set(self, value):
            return real_var.set(value)

        def reset(self, _token):
            raise RuntimeError("reservation reset failed")

    monkeypatch.setattr(api_server_module, "_api_agent_request_reservation", ResetFailure())
    monkeypatch.setattr(adapter, "_check_auth", lambda _request: None)

    @api_server_module._admit_api_agent_request
    async def handler(_adapter, _request):
        return web.Response(status=204)

    with pytest.raises(RuntimeError, match="reservation reset failed"):
        await handler(adapter, MagicMock())
    assert real_var.get() is None
    assert adapter._pending_agent_requests == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failed_name",
    ["profile", "auth", "browser_principal", "browser_family"],
)
async def test_scoped_request_resets_each_context_independently(failed_name, monkeypatch):
    variables = {
        "profile": api_server_module._api_request_profile,
        "auth": api_server_module._api_request_auth_context,
        "browser_principal": api_server_module._api_request_browser_control_principal,
        "browser_family": api_server_module._api_request_browser_control_transport_family,
    }
    baselines = {name: variable.get() for name, variable in variables.items()}

    class ResetFailure:
        def __init__(self, real, name):
            self.real = real
            self.name = name

        def get(self):
            return self.real.get()

        def set(self, value):
            return self.real.set(value)

        def reset(self, token):
            if self.name == failed_name:
                raise RuntimeError(f"{self.name} reset failed")
            return self.real.reset(token)

    monkeypatch.setattr(
        api_server_module, "_api_request_profile", ResetFailure(variables["profile"], "profile")
    )
    monkeypatch.setattr(
        api_server_module, "_api_request_auth_context", ResetFailure(variables["auth"], "auth")
    )
    monkeypatch.setattr(
        api_server_module,
        "_api_request_browser_control_principal",
        ResetFailure(variables["browser_principal"], "browser_principal"),
    )
    monkeypatch.setattr(
        api_server_module,
        "_api_request_browser_control_transport_family",
        ResetFailure(variables["browser_family"], "browser_family"),
    )
    adapter = _adapter(None)

    async def handler(_request):
        return web.Response(status=204)

    with pytest.raises(RuntimeError, match=f"{failed_name} reset failed"):
        await adapter._run_scoped_request(
            MagicMock(), handler, "default",
            api_server_module._CredentialAuthContext(
                _principal(APIServerOperation.RUNS_CREATE), "owner-key"
            ),
        )

    assert variables[failed_name].get() is None
    assert all(
        variable.get() == baselines[name]
        for name, variable in variables.items()
        if name != failed_name
    )
