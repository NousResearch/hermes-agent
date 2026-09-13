"""HTTP run owners can acknowledge an exact unknown API admission after restart."""
import asyncio
import json
from contextlib import suppress
from contextvars import ContextVar
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.session import SessionStore
from gateway.session_authority import initialize_session_authority
from gateway.session_api import restore_api_session
from hermes_state import SessionDB
from hermes_state_runtime import claim_session_input, list_session_admissions


_OWNER = ContextVar("test_api_unknown_owner", default="owner")
_API_KEYS = {"owner": "checkpoint-two-api-key", "other": "other-valid-api-key"}
_AUTH_HEADERS = {"Authorization": f"Bearer {_API_KEYS['owner']}", "X-Test-Owner": "owner"}
_OTHER_AUTH_HEADERS = {
    "Authorization": f"Bearer {_API_KEYS['other']}", "X-Test-Owner": "other"
}


def _runs_client(adapter):
    @web.middleware
    async def select_owner(request, handler):
        token = _OWNER.set(request.headers.get("X-Test-Owner", "owner"))
        try:
            return await handler(request)
        finally:
            _OWNER.reset(token)

    adapter._expected_api_key = lambda: _API_KEYS[_OWNER.get()]
    app = web.Application(middlewares=[select_owner])
    for method, path, handler in adapter._http_route_table():
        if path.startswith("/v1/runs"):
            app.router.add_route(method, path, handler)
    return TestClient(TestServer(app))


def _adapter(runner, db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter.gateway_runner = runner
    adapter._session_db = db
    return adapter


async def _restarted_api_runs(tmp_path, monkeypatch, *, idempotent=True):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    db = SessionDB(tmp_path / "state.db")
    executed = []

    async def answer(event):
        from gateway.session_results import execution_result

        executed.append(event.text)
        result = {"final_response": "ACK_" + event.text, "messages": []}
        execution_result.get()["result"] = result
        return result["final_response"]

    runner = SimpleNamespace(
        _session_db=db,
        session_store=store,
        _draining=False,
        _handle_message=answer,
    )
    first = await initialize_session_authority(
        runner, profile_id="default", instance_id="api-owner-1"
    )
    adapter = _adapter(runner, db)
    runner._adapter_for_source = lambda source: adapter

    # Exercise the public admission handler so run ownership and the canonical
    # principal/request-id binding are the same ones the resolution route sees.
    first._schedule = lambda ref: None
    client = _runs_client(adapter)
    await client.start_server()
    tasks = []
    try:
        admitted = []
        inputs = (("run-a", "HEAD"), ("run-b", "FOLLOWER")) if idempotent else (
            ("run-a", "SAME BODY"), ("run-b", "SAME BODY"))
        for run_id, text in inputs:
            headers = dict(_AUTH_HEADERS)
            if idempotent:
                headers["Idempotency-Key"] = run_id
            response = await client.post(
                "/v1/runs",
                json={"input": text, "session_id": "api-session"},
                headers=headers,
            )
            assert response.status == 202, await response.text()
            admitted.append(await response.json())
        assert admitted[0]["run_id"] != admitted[1]["run_id"]
        tasks = list(adapter._active_run_tasks.values())
        await asyncio.sleep(0)

        rows = list_session_admissions(db, session_id="api-session")
        head = claim_session_input(db, epoch=first.epoch, session_id="api-session")
        assert head["admission_id"] == rows[0]["admission_id"]

        restarted = await initialize_session_authority(
            runner, profile_id="default", instance_id="api-owner-2"
        )
        restore_api_session(restarted, "api-session")
        rows = list_session_admissions(db, session_id="api-session")
        assert [row["status"] for row in rows] == ["unknown", "queued"]
        await client.close()

        # A process restart constructs a new API adapter.  Keeping the original
        # adapter here would preserve _run_owners and conceal the recovery gap.
        fresh = _adapter(runner, db)
        runner._adapter_for_source = lambda source: fresh
        client = _runs_client(fresh)
        await client.start_server()
        return client, (adapter, fresh), store, restarted, rows[0], rows[1], executed, tasks
    except Exception:
        await client.close()
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
        store.close_all_db_handles()
        raise


async def _close_fixture(client, adapters, store, tasks):
    for task in tasks:
        task.cancel()
    for task in tasks:
        with suppress(asyncio.CancelledError):
            await task
    await client.close()
    for adapter in adapters:
        adapter._response_store.close()
        adapter._run_idempotency_store.close()
    adapters[-1].gateway_runner.session_authority.db.close()
    store.close_all_db_handles()


@pytest.mark.asyncio
async def test_http_resolve_unknown_releases_follower_once_without_replaying_head(
    tmp_path, monkeypatch
):
    client, adapters, store, owner, unknown, follower, executed, tasks = (
        await _restarted_api_runs(tmp_path, monkeypatch)
    )
    try:
        body = {
            "admission_id": unknown["admission_id"],
            "execution_generation": unknown["generation"],
        }
        path = "/v1/runs/" + unknown["request_id"] + "/resolve-unknown"
        unauthenticated = await client.post(path, json=body)
        assert unauthenticated.status == 401
        assert (await unauthenticated.json())["error"]["code"] == "gateway_auth_failed"

        capabilities = json.loads((await adapters[-1]._handle_capabilities(
            SimpleNamespace(headers=_AUTH_HEADERS, remote="127.0.0.1"))).text)
        assert capabilities["features"]["run_unknown_resolution"] is True
        assert capabilities["endpoints"]["run_unknown_resolution"]["path"] == "/v1/runs/{run_id}/resolve-unknown"

        response = await client.post(path, json=body, headers=_AUTH_HEADERS)
        assert response.status == 200, await response.text()
        result = await response.json()
        assert result["run_id"] == unknown["request_id"]
        assert result["admission_id"] == unknown["admission_id"]
        assert result["status"] == "terminal" and result["outcome"] == "interrupted"

        await owner.sessions["api-session"].task
        assert executed == ["FOLLOWER"]
        statuses = {
            row["admission_id"]: (row["status"], row["outcome"])
            for row in list_session_admissions(owner.db, session_id="api-session", pending_only=False)
        }
        assert statuses[unknown["admission_id"]] == ("terminal", "interrupted")
        assert statuses[follower["admission_id"]] == ("terminal", "completed")

        again = await client.post(path, json=body, headers=_AUTH_HEADERS)
        assert again.status == 409
        assert (await again.json())["error"]["code"] == "stale_generation"
        assert executed == ["FOLLOWER"]
    finally:
        await _close_fixture(client, adapters, store, tasks)


@pytest.mark.asyncio
async def test_http_resolve_unknown_refuses_stale_or_foreign_admission(
    tmp_path, monkeypatch
):
    client, adapters, store, owner, unknown, follower, executed, tasks = (
        await _restarted_api_runs(tmp_path, monkeypatch)
    )
    try:
        run_path = "/v1/runs/" + unknown["request_id"] + "/resolve-unknown"
        stop = await client.post(
            "/v1/runs/" + unknown["request_id"] + "/stop",
            headers=_AUTH_HEADERS,
        )
        assert stop.status == 409
        assert (await stop.json())["error"]["code"] == "unknown_execution"
        attempts = (
            ({
                "admission_id": unknown["admission_id"],
                "execution_generation": unknown["generation"] + 1,
            }, "stale_generation"),
            ({
                "admission_id": follower["admission_id"],
                "execution_generation": unknown["generation"],
            }, "not_found"),
            ({
                "admission_id": unknown["admission_id"],
                "execution_generation": False,
            }, "stale_generation"),
            ({
                "admission_id": unknown["admission_id"],
                "execution_generation": unknown["generation"],
                "extra": True,
            }, "invalid_params"),
            ([unknown["admission_id"], unknown["generation"]], "invalid_params"),
        )
        for body, code in attempts:
            response = await client.post(run_path, json=body, headers=_AUTH_HEADERS)
            assert response.status == 409
            assert (await response.json())["error"]["code"] == code

        await asyncio.sleep(0.05)
        rows = {
            row["admission_id"]: row["status"]
            for row in list_session_admissions(owner.db, session_id="api-session")
        }
        assert rows == {
            unknown["admission_id"]: "unknown",
            follower["admission_id"]: "queued",
        }
        assert executed == []
    finally:
        await _close_fixture(client, adapters, store, tasks)


@pytest.mark.asyncio
async def test_fresh_adapter_recovers_non_idempotent_owner_but_refuses_other_credential(
    tmp_path, monkeypatch
):
    client, adapters, store, owner, unknown, follower, executed, tasks = (
        await _restarted_api_runs(tmp_path, monkeypatch, idempotent=False)
    )
    try:
        path = "/v1/runs/" + unknown["request_id"] + "/resolve-unknown"
        body = {
            "admission_id": unknown["admission_id"],
            "execution_generation": unknown["generation"],
        }

        foreign = await client.post(path, json=body, headers=_OTHER_AUTH_HEADERS)
        assert foreign.status == 404
        assert (await foreign.json())["error"]["code"] == "run_not_found"

        response = await client.post(path, json=body, headers=_AUTH_HEADERS)
        assert response.status == 200, await response.text()
        assert (await response.json())["outcome"] == "interrupted"
    finally:
        await _close_fixture(client, adapters, store, tasks)
