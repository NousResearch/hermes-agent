"""HTTP run owners can acknowledge an exact unknown API admission after restart."""
import asyncio
import json
import sqlite3
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import asdict
from types import SimpleNamespace

import pytest
from aiohttp import WSServerHandshakeError, web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.session import SessionStore
from gateway.session_authority import initialize_session_authority
from gateway.session_api import restore_api_session
from hermes_state import SessionDB
from hermes_state_runtime import (
    RuntimeStoreError, cancel_session_input, claim_session_input, list_session_admissions,
    resolve_unknown_session_input,
)


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


async def _restarted_api_runs(tmp_path, monkeypatch, *, idempotent=True, control_only_restart=False):
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

        if control_only_restart:
            # The real API execution appends this user row before invoking the provider. The
            # simplified executor above does not, so seed the same transcript boundary here.
            db.append_message("api-session", "user", "HEAD")
            runner.control_only = True
        restarted = await initialize_session_authority(
            runner, profile_id="default", instance_id="api-owner-2", control_only=control_only_restart
        )
        if not control_only_restart:
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


def _control_client(adapter):
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
        app.router.add_route(method, path, handler)
    return TestClient(TestServer(app))


@pytest.mark.asyncio
async def test_control_only_denies_turn_ingress_and_resolves_unknown_without_running_fifo(
    tmp_path, monkeypatch
):
    """The API control plane remains useful while every turn path stays closed."""
    client, adapters, store, owner, unknown, follower, executed, tasks = (
        await _restarted_api_runs(tmp_path, monkeypatch, control_only_restart=True)
    )
    await client.close()
    client = _control_client(adapters[-1])
    await client.start_server()
    try:
        before_admissions = [
            (row["admission_id"], row["status"], row["generation"])
            for row in list_session_admissions(owner.db, session_id="api-session", pending_only=False)
        ]
        with owner.db._read_ctx() as conn:
            before_transcript = [tuple(row) for row in conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id", ("api-session",)
            )]

        unauthenticated = await client.get("/health")
        assert unauthenticated.status == 401
        health = await client.get("/health", headers=_AUTH_HEADERS)
        assert health.status == 200
        assert (await health.json())["status"] == "control_only"
        detailed = await client.get("/health/detailed", headers=_AUTH_HEADERS)
        assert (await detailed.json())["mode"] == "control_only"

        run_path = "/v1/runs/" + unknown["request_id"]
        status = await client.get(run_path, headers=_AUTH_HEADERS)
        assert status.status == 200
        evidence = await status.json()
        assert evidence["status"] == "unknown"
        assert evidence["admission_id"] == unknown["admission_id"]
        assert evidence["execution_generation"] == unknown["generation"]

        inference_requests = [
            ("POST", "/v1/runs", {"input": "do not execute"}),
            ("POST", "/v1/responses", {"input": "do not execute"}),
            ("POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "do not execute"}]}),
            ("POST", "/api/sessions", {"session_id": "new"}),
            ("POST", "/api/sessions/api-session/chat", {"message": "do not execute"}),
            ("POST", "/api/sessions/api-session/chat/stream", {"message": "do not execute"}),
            ("POST", "/api/sessions/api-session/fork", {"message": "do not execute"}),
            ("POST", "/api/cron/fire", {"prompt": "do not execute"}),
            ("POST", "/api/platforms/telegram/events", {"text": "do not execute"}),
        ]
        for method, path, body in inference_requests:
            denied = await client.request(method, path, json=body, headers=_AUTH_HEADERS)
            assert denied.status == 503, (path, await denied.text())
            assert "do not execute" not in await denied.text()
        with pytest.raises(WSServerHandshakeError) as websocket_denied:
            await client.ws_connect("/api/sessions/api-session/events", headers=_AUTH_HEADERS)
        assert websocket_denied.value.status == 503
        bad_auth = await client.post("/v1/runs", json={"input": "secret"})
        assert bad_auth.status == 401
        assert "secret" not in await bad_auth.text()

        resolve_path = run_path + "/resolve-unknown"
        wrong = await client.post(resolve_path, headers=_AUTH_HEADERS, json={
            "admission_id": unknown["admission_id"],
            "execution_generation": unknown["generation"] + 1,
        })
        assert wrong.status == 409
        assert (await wrong.json())["error"]["code"] == "stale_generation"
        foreign = await client.post(resolve_path, headers=_OTHER_AUTH_HEADERS, json={
            "admission_id": unknown["admission_id"],
            "execution_generation": unknown["generation"],
        })
        assert foreign.status == 404

        body = {"admission_id": unknown["admission_id"],
                "execution_generation": unknown["generation"]}
        resolved = await client.post(resolve_path, headers=_AUTH_HEADERS, json=body)
        assert resolved.status == 200
        receipt = await resolved.json()
        assert receipt["outcome"] == "interrupted"
        retry = await client.post(resolve_path, headers=_AUTH_HEADERS, json=body)
        assert retry.status == 200
        assert await retry.json() == receipt

        still_closed = await client.post("/v1/runs", json={"input": "still forbidden"}, headers=_AUTH_HEADERS)
        assert still_closed.status == 503
        health_after_resolution = await client.get("/health", headers=_AUTH_HEADERS)
        assert (await health_after_resolution.json())["mode"] == "control_only"
        after_admissions = [
            (row["admission_id"], row["status"], row["generation"])
            for row in list_session_admissions(owner.db, session_id="api-session", pending_only=False)
        ]
        with owner.db._read_ctx() as conn:
            after_transcript = [tuple(row) for row in conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id", ("api-session",)
            )]
        assert after_admissions == [
            (unknown["admission_id"], "terminal", unknown["generation"]),
            (follower["admission_id"], "queued", follower["generation"]),
        ]
        assert before_admissions[0][1] == "unknown" and before_admissions[1][1] == "queued"
        assert before_transcript == after_transcript
        assert executed == []
        assert owner.sessions == {}
        assert after_transcript == [("user", "HEAD")]
        with owner.db._read_ctx() as conn:
            marker_count = conn.execute(
                "SELECT COUNT(*) FROM state_meta WHERE key LIKE 'control_only_resolution_boundary:%'"
            ).fetchone()[0]
        assert marker_count == 1

        # The transcript boundary is deferred until a later serving boot, and is written before
        # that authority can claim any queued successor.
        owner.runner.control_only = False
        await initialize_session_authority(
            owner.runner, profile_id="default", instance_id="api-owner-3"
        )
        with owner.db._read_ctx() as conn:
            serving_transcript = [tuple(row) for row in conn.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id", ("api-session",)
            )]
            marker_count = conn.execute(
                "SELECT COUNT(*) FROM state_meta WHERE key LIKE 'control_only_resolution_boundary:%'"
            ).fetchone()[0]
        assert serving_transcript[0] == ("user", "HEAD")
        assert serving_transcript[-1][0] == "assistant"
        assert "turn did not complete" in serving_transcript[-1][1]
        assert marker_count == 0
    finally:
        await _close_fixture(client, adapters, store, tasks)


@pytest.mark.asyncio
async def test_deleted_control_only_resolution_survives_serving_restart(tmp_path, monkeypatch):
    client, adapters, store, owner, unknown, follower, executed, tasks = (
        await _restarted_api_runs(tmp_path, monkeypatch, control_only_restart=True)
    )
    try:
        path = "/v1/runs/" + unknown["request_id"] + "/resolve-unknown"
        body = {"admission_id": unknown["admission_id"],
                "execution_generation": unknown["generation"]}
        response = await client.post(path, headers=_AUTH_HEADERS, json=body)
        assert response.status == 200
        receipt = await response.json()
        assert receipt["outcome"] == "interrupted"
        cancel_session_input(owner.db, epoch=owner.epoch, admission_id=follower["admission_id"])
        marker = "control_only_resolution_boundary:" + unknown["admission_id"]

        # A failure after retirement must roll back both the tombstone and its boundary removal.
        owner.db._execute_write(lambda c: c.execute(
            "CREATE TRIGGER deny_admission_delete BEFORE DELETE ON session_admissions "
            "BEGIN SELECT RAISE(ABORT, 'delete failed'); END"))
        with pytest.raises(sqlite3.IntegrityError, match="delete failed"):
            owner.db.delete_session("api-session")
        with owner.db._read_ctx() as conn:
            assert conn.execute("SELECT value FROM state_meta WHERE key=?", (marker,)).fetchone()[0] == "api-session"
        assert owner.db.get_session("api-session") is not None
        owner.db._execute_write(lambda c: c.execute("DROP TRIGGER deny_admission_delete"))
        assert owner.db.delete_session("api-session")
        replay = resolve_unknown_session_input(
            owner.db, epoch=owner.epoch, admission_id=unknown["admission_id"],
            generation=unknown["generation"],
        )
        assert {"run_id": unknown["request_id"], **asdict(owner._receipt(replay))} == receipt

        owner.runner.control_only = False
        serving = await initialize_session_authority(
            owner.runner, profile_id="default", instance_id="api-owner-3"
        )
        assert owner.db.get_session("api-session") is None
        assert owner.db.get_messages("api-session") == []
        replay = resolve_unknown_session_input(
            owner.db, epoch=serving.epoch, admission_id=unknown["admission_id"],
            generation=unknown["generation"],
        )
        assert {"run_id": unknown["request_id"], **asdict(serving._receipt(replay))} == receipt
        assert executed == []
    finally:
        await _close_fixture(client, adapters, store, tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ["wrong_scope", "wrong_scope_retired", "orphan"])
async def test_invalid_deferred_boundary_fails_closed(tmp_path, monkeypatch, corruption):
    client, adapters, store, owner, unknown, _follower, _executed, tasks = (
        await _restarted_api_runs(tmp_path, monkeypatch, control_only_restart=True)
    )
    try:
        path = "/v1/runs/" + unknown["request_id"] + "/resolve-unknown"
        response = await client.post(path, headers=_AUTH_HEADERS, json={
            "admission_id": unknown["admission_id"],
            "execution_generation": unknown["generation"],
        })
        assert response.status == 200
        marker = "control_only_resolution_boundary:" + unknown["admission_id"]
        if corruption.startswith("wrong_scope"):
            owner.db._execute_write(lambda c: c.execute(
                "UPDATE state_meta SET value=? WHERE key=?", ("another-session", marker)))
            if corruption == "wrong_scope_retired":
                cancel_session_input(owner.db, epoch=owner.epoch, admission_id=_follower["admission_id"])
                assert owner.db.delete_session("api-session")
                with owner.db._read_ctx() as conn:
                    assert conn.execute("SELECT value FROM state_meta WHERE key=?", (marker,)).fetchone()[0] == "another-session"
        else:
            owner.db._execute_write(lambda c: c.execute(
                "UPDATE state_meta SET key=? WHERE key=?", (marker + "-missing", marker)))
        owner.runner.control_only = False
        with pytest.raises(RuntimeStoreError, match="invalid_control_only_resolution_boundary"):
            await initialize_session_authority(
                owner.runner, profile_id="default", instance_id="api-owner-3"
            )
    finally:
        await _close_fixture(client, adapters, store, tasks)


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

        # A lost-ack retry of the exact resolution replays its committed receipt; nothing runs twice.
        again = await client.post(path, json=body, headers=_AUTH_HEADERS)
        assert again.status == 200, await again.text()
        assert await again.json() == result
        await owner.sessions["api-session"].task
        assert executed == ["FOLLOWER"]
        stale = await client.post(path, json=body | {"execution_generation": unknown["generation"] + 1},
                                  headers=_AUTH_HEADERS)
        assert stale.status == 409 and (await stale.json())["error"]["code"] == "stale_generation"
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
            assert response.status == (400 if code == "invalid_params" else 409)
            assert (await response.json())["error"]["code"] == code

        # A refused resolution must not release the FIFO: let any drain it scheduled run to the end
        # (the restored drain parks on the unknown head) before reading what it left behind.
        drain = owner.sessions["api-session"].task
        if drain is not None:
            await asyncio.wait_for(drain, 10)
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
    client, adapters, store, _owner, unknown, _follower, _executed, tasks = (
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


@pytest.mark.asyncio
async def test_http_poll_reports_unknown_head_as_non_terminal_until_resolved(
    tmp_path, monkeypatch
):
    """The documented discovery path: a client polls GET /v1/runs/{id}, learns the head is
    unknown (not finished), reads admission_id/execution_generation from that body and
    resolves it. ``interrupted`` is in TERMINAL_STATUSES and documented as terminal, so a
    poller cannot tell a paused head from a finished run if the projection reuses it."""
    from gateway.platforms.api_server_run_idempotency import TERMINAL_STATUSES

    client, adapters, store, owner, unknown, _follower, executed, tasks = (
        await _restarted_api_runs(tmp_path, monkeypatch)
    )
    try:
        path = "/v1/runs/" + unknown["request_id"]
        polled = await client.get(path, headers=_AUTH_HEADERS)
        assert polled.status == 200, await polled.text()
        body = await polled.json()
        print("POLL_BEFORE_RESOLVE", json.dumps({k: body.get(k) for k in (
            "status", "admission_id", "execution_generation")}, sort_keys=True))
        assert body["status"] not in TERMINAL_STATUSES, body
        assert body["status"] == "unknown", body
        assert body["admission_id"] == unknown["admission_id"]
        assert body["execution_generation"] == unknown["generation"]

        resolved = await client.post(path + "/resolve-unknown", headers=_AUTH_HEADERS, json={
            "admission_id": body["admission_id"],
            "execution_generation": body["execution_generation"],
        })
        assert resolved.status == 200, await resolved.text()
        await owner.sessions["api-session"].task
        assert executed == ["FOLLOWER"]

        after = await (await client.get(path, headers=_AUTH_HEADERS)).json()
        print("POLL_AFTER_RESOLVE", json.dumps({"status": after.get("status")}))
        assert after["status"] in TERMINAL_STATUSES, after
    finally:
        await _close_fixture(client, adapters, store, tasks)
