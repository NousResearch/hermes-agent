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
            headers={"Authorization": "Bearer credential"},
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
async def test_credential_run_rejects_client_supplied_gateway_session_key(monkeypatch):
    principal = _principal(APIServerOperation.RUNS_CREATE)
    adapter = _adapter(Authorizer(lambda _request: principal))
    session_id = _create_owned_session(adapter, principal)
    monkeypatch.setattr(
        adapter,
        "_create_agent",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("credential run must reject before agent creation")
        ),
    )

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.post(
            "/v1/runs",
            json={"input": "hello", "session_id": session_id},
            headers={
                "Authorization": "Bearer credential",
                "X-Hermes-Session-Key": "caller-controlled-memory-scope",
            },
        )
        body = await response.json()

    assert response.status == 403
    assert body["error"]["code"] == "credential_operation_forbidden"

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

def test_gateway_peer_refresh_never_transfers_credential_ownership(tmp_path):
    from hermes_state import SessionDB
    from hermes_state_errors import SessionTurnLeaseLostError

    db = SessionDB(tmp_path / "state.db")
    db.create_session("root", "api_server", credential_owner="owner-a")
    db.end_session("root", "compression")
    db.create_session(
        "tip", "api_server", parent_session_id="root", credential_owner="owner-b"
    )

    with pytest.raises(SessionTurnLeaseLostError, match="credential ownership changed"):
        db.record_gateway_session_peer(
            "tip",
            source="api_server",
            session_key="declared-key",
            credential_owner="owner-a",
            include_compression_ancestors=True,
        )

    assert db.get_session("root")["credential_owner"] == "owner-a"
    assert db.get_session("tip")["credential_owner"] == "owner-b"

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
async def test_credential_run_cancelled_before_coroutine_start_releases_admission(
    tmp_path, monkeypatch
):
    from hermes_state import SessionDB

    principal = _principal(APIServerOperation.RUNS_CREATE)
    adapter = _adapter(Authorizer(lambda _request: principal))
    adapter._session_db = SessionDB(tmp_path / "state.db")
    session_id = _create_owned_session(adapter, principal)
    started = False

    async def never_started(*_args, **_kwargs):
        nonlocal started
        started = True

    monkeypatch.setattr(api_server_runs_module, "_execute_run", never_started)
    real_create_task = asyncio.create_task

    def cancel_run_task(coro):
        task = real_create_task(coro)
        if getattr(coro, "cr_code", None) is never_started.__code__:
            task.cancel()
        return task

    monkeypatch.setattr(asyncio, "create_task", cancel_run_task)

    async with TestClient(TestServer(_credential_app(adapter))) as client:
        response = await client.post(
            "/v1/runs",
            json={"input": "hello", "session_id": session_id},
            headers={"Authorization": "Bearer credential"},
        )
        body = await response.json()
        await asyncio.sleep(0)

    run_id = body["run_id"]
    assert response.status == 202
    assert started is False
    assert adapter._run_statuses[run_id]["status"] == "cancelled"
    assert run_id not in adapter._active_run_tasks
    assert adapter._session_db.try_acquire_session_turn_lease(
        session_id,
        f"pid={os.getpid()}:replacement",
        expected_credential_owner=adapter._credential_owner_key(principal),
    )

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
