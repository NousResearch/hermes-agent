"""HTTP approval retries must never answer the following prompt (#89754)."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

from gateway.platforms import api_server_runs
from gateway.platforms.api_server import _api_request_profile
from tests.gateway.test_api_server_runs import _make_adapter
from tools import approval
from tools.approval_gateway_wait import _ApprovalEntry


@pytest_asyncio.fixture
async def api(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    adapter = _make_adapter("test-only-key")
    monkeypatch.setattr(adapter, "_expected_api_key", lambda: "test-only-key")

    @web.middleware
    async def profile(request, handler):
        token = _api_request_profile.set(request.headers.get("Test-Profile"))
        try:
            return await handler(request)
        finally:
            _api_request_profile.reset(token)

    app = web.Application(middlewares=[profile])
    for method, path, handler in api_server_runs._http_routes(adapter):
        app.router.add_route(method, path, handler)
    async with TestClient(TestServer(app)) as client:
        yield adapter, client
    adapter._run_idempotency_store.close()
    adapter._response_store.close()
    adapter._close_cached_session_dbs()


def _headers(**extra):
    return {"Authorization": "Bearer test-only-key", "Idempotency-Key": "receipt", **extra}


def _pending(adapter, run_id, monkeypatch, profile=None):
    token = _api_request_profile.set(profile)
    try:
        adapter._run_owners[run_id] = adapter._run_idempotency_scope(
            make_mocked_request("POST", f"/v1/runs/{run_id}/approval"))
    finally:
        _api_request_profile.reset(token)
    adapter._run_statuses[run_id] = {"run_id": run_id, "status": "waiting_for_approval"}
    adapter._run_approval_sessions[run_id] = run_id
    first, second = (_ApprovalEntry({"command": "test"}) for _ in range(2))
    monkeypatch.setitem(approval._gateway_queues, run_id, [first, second])
    return first, second


@pytest.mark.asyncio
async def test_retry_replays_original_and_authenticates_before_replay(api, monkeypatch):
    adapter, client = api
    first, second = _pending(adapter, "retry", monkeypatch)
    url = "/v1/runs/retry/approval"
    original = await client.post(url, json={"choice": "once"}, headers=_headers())
    assert original.status == 200
    duplicate = await client.post(url, json={"choice": "once"}, headers=_headers())
    assert duplicate.status == 200
    assert await duplicate.json() == await original.json()
    assert first.event.is_set()
    assert not second.event.is_set()
    denied = await client.post(url, json={"choice": "once"}, headers={"Idempotency-Key": "receipt"})
    assert denied.status == 401
    conflict = await client.post(url, json={"choice": "deny"}, headers=_headers())
    assert conflict.status == 409
    assert not second.event.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [{"all": True}, {"resolve_all": True},
                                     {"request_id": "other"}, {"execution_generation": 2}])
async def test_every_approval_input_conflicts(api, monkeypatch, changed):
    adapter, client = api
    _, second = _pending(adapter, "inputs", monkeypatch)
    url = "/v1/runs/inputs/approval"
    assert (await client.post(url, json={"choice": "once"}, headers=_headers())).status == 200
    response = await client.post(url, json={"choice": "once", **changed}, headers=_headers())
    assert response.status == 409
    assert not second.event.is_set()


@pytest.mark.asyncio
async def test_same_key_is_independent_between_profiles_and_run_urls(api, monkeypatch):
    adapter, client = api
    for profile in ("alpha", "beta"):
        for suffix in ("one", "two"):
            run = profile + suffix
            first, second = _pending(adapter, run, monkeypatch, profile)
            response = await client.post(f"/v1/runs/{run}/approval", json={"choice": "once"},
                                         headers=_headers(**{"Test-Profile": profile}))
            # Same profile/key on another URL conflicts; a different profile is independent.
            assert response.status == (200 if suffix == "one" else 409)
            assert first.event.is_set() == (suffix == "one")
            assert not second.event.is_set()


@pytest.mark.asyncio
async def test_cors_preflight_allows_generation_and_idempotency_headers(api, monkeypatch):
    from gateway.platforms.api_server import cors_middleware

    adapter, _ = api
    origin = "https://approval-client.example"
    monkeypatch.setattr(adapter, "_cors_origins", (origin,))
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    for method, path, handler in api_server_runs._http_routes(adapter):
        app.router.add_route(method, path, handler)
    async with TestClient(TestServer(app)) as client:
        response = await client.options("/v1/runs/example/approval", headers={
            "Origin": origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "X-Hermes-Expected-Generation, Idempotency-Key"})
        assert response.status == 200
        assert response.headers["Access-Control-Allow-Origin"] == origin
        allowed = {header.strip().lower() for header in response.headers["Access-Control-Allow-Headers"].split(",")}
        assert "x-hermes-expected-generation" in allowed
        assert "idempotency-key" in allowed


@pytest.mark.asyncio
async def test_meta_auth_and_generation_refuse_before_mutation(api, monkeypatch):
    adapter, client = api
    assert (await client.get("/v1/runs/meta")).status == 401
    response = await client.get("/v1/runs/meta", headers=_headers())
    assert response.status == 200
    meta = await response.json()
    assert meta["idempotency"] == "v1"
    assert meta["idempotency_ttl_seconds"] > 0
    assert str(UUID(meta["server_generation"])) == meta["server_generation"]
    first, _ = _pending(adapter, "generation", monkeypatch)
    for generation, status in ((str(uuid4()), 409), ("", 400), ("bad", 400), ("x" * 257, 400)):
        headers = _headers(**{"X-Hermes-Expected-Generation": generation})
        for path, body in (("/v1/runs", {"input": "test"}),
                           ("/v1/runs/generation/approval", {"choice": "once"})):
            refused = await client.post(path, json=body, headers=headers)
            assert refused.status == status
            if status == 409:
                assert (await refused.json())["error"]["code"] == "server_generation_mismatch"
    assert not first.event.is_set()
    accepted = await client.post("/v1/runs/generation/approval", json={"choice": "once"},
        headers=_headers(**{"X-Hermes-Expected-Generation": meta["server_generation"]}))
    assert accepted.status == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", [{"provider": "other"}, {"model_options": {"reasoning_effort": "high"}}])
async def test_run_full_body_digest_and_profile_namespace(api, monkeypatch, changed):
    adapter, client = api
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    create = MagicMock(return_value=agent)
    monkeypatch.setattr(adapter, "_create_agent", create)
    body = {"input": "test", "provider": "initial", "model_options": {"reasoning_effort": "low"}}
    runs = []
    for profile in ("alpha", "beta"):
        headers = _headers(**{"Test-Profile": profile})
        response = await client.post("/v1/runs", json=body, headers=headers)
        assert response.status == 202, await response.text()
        runs.append((await response.json())["run_id"])
        conflict = await client.post("/v1/runs", json={**body, **changed}, headers=headers)
        assert conflict.status == 409
        assert (await conflict.json())["error"]["code"] == "idempotency_key_conflict"
        replay = await client.post("/v1/runs", json=body, headers=headers)
        assert replay.status == 202
        assert (await replay.json())["run_id"] == runs[-1]
    assert runs[0] != runs[1]
    foreign = await client.get(f"/v1/runs/{runs[0]}", headers=_headers(**{"Test-Profile": "beta"}))
    assert foreign.status == 404
    await asyncio.gather(*list(adapter._active_run_tasks.values()))
    assert create.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("ambiguous", [False, True])
async def test_overlapping_or_ambiguous_resolution_is_never_repeated(api, monkeypatch, ambiguous):
    from gateway.platforms import api_server_run_admissions as admissions

    adapter, client = api
    first, second = _pending(adapter, "overlap", monkeypatch)
    entered, release = asyncio.Event(), asyncio.Event()
    original = admissions._resolve_approval
    calls = 0

    async def controlled(*args, **kwargs):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        response = await original(*args, **kwargs)
        if ambiguous:
            raise OSError("test resolution reply lost")
        return response

    monkeypatch.setattr(admissions, "_resolve_approval", controlled)
    url = "/v1/runs/overlap/approval"
    pending = asyncio.create_task(client.post(url, json={"choice": "once"}, headers=_headers()))
    await entered.wait()
    try:
        duplicate = await client.post(url, json={"choice": "once"}, headers=_headers())
        assert duplicate.status == 409
        assert not first.event.is_set()
    finally:
        release.set()
    result = await pending
    assert result.status == (500 if ambiguous else 200)
    retry = await client.post(url, json={"choice": "once"}, headers=_headers())
    assert retry.status == (409 if ambiguous else 200)
    assert calls == 1
    assert first.event.is_set()
    assert not second.event.is_set()


def test_receipt_capacity_expires_completed_but_never_ambiguous(monkeypatch):
    from gateway.platforms import api_server_run_admissions as admissions

    state = admissions.ApprovalAdmissions()
    monkeypatch.setattr(state, "MAX_RECEIPTS", 2)
    now = [100.0]
    monkeypatch.setattr(admissions.time, "monotonic", lambda: now[0])
    assert state.reserve("profile", "pending", "fp")[0] == "reserved"
    _, complete = state.reserve("profile", "complete", "fp")
    state.complete("profile", "complete", complete, web.json_response({"resolved": 1}))
    assert state.reserve("profile", "new", "fp")[0] == "full"
    now[0] += state.TTL_SECONDS + 1
    assert state.reserve("profile", "pending", "fp")[0] == "replay"
    assert state.reserve("profile", "pending", "changed")[0] == "conflict"
    assert state.reserve("profile", "new", "fp")[0] == "reserved"
    assert state.generation != admissions.ApprovalAdmissions().generation


@pytest.mark.asyncio
@pytest.mark.parametrize("detach_error", [None, "invalid_params", "worker_control_backpressure"])
async def test_authority_retry_preserves_next_prompt_and_execution_fence(api, owner, monkeypatch, detach_error):
    from gateway.session_results import execution_result

    adapter, client = api
    monkeypatch.setenv("HERMES_HOME", str(Path(owner.db.db_path).parent))
    adapter.gateway_runner = owner.runner
    adapter._session_db = owner.db
    owner.runner._adapter_for_source = lambda source: adapter
    entries = [_ApprovalEntry({"command": "test"}) for _ in range(2)]
    registered = [asyncio.Event(), asyncio.Event()]
    started, allow_prompt = asyncio.Event(), asyncio.Event()
    generation = []

    async def turn(event):
        sid = event.source.chat_id
        live = owner.sessions[sid]
        current = owner.db.get_session(sid)["runtime_generation"]
        generation.append(current)
        started.set()
        await allow_prompt.wait()
        for entry, ready in zip(entries, registered):
            monkeypatch.setitem(approval._gateway_queues, live.route, [entry])
            owner.register_approval(sid, current, live.route, entry.data)
            ready.set()
            await asyncio.to_thread(entry.event.wait)
        execution_result.get()["result"] = {"final_response": "done"}
        return "done"

    owner.runner._handle_message = turn
    try:
        response = await client.post("/v1/runs", json={"input": "test"}, headers=_headers())
        assert response.status == 202, await response.text()
        run = (await response.json())["run_id"]
        await started.wait()
        url = f"/v1/runs/{run}/approval"
        body = {"request_id": entries[0].data["request_id"],
                "execution_generation": generation[0], "choice": "once"}
        missing = await client.post(url, json=body, headers=_headers())
        assert missing.status == 409
        assert (await missing.json())["error"]["code"] == "approval_not_pending"
        allow_prompt.set()
        await registered[0].wait()
        detach = owner.detach
        if detach_error:
            from hermes_state_runtime import RuntimeStoreError

            async def failed_detach(*args, **kwargs):
                await detach(*args, **kwargs)
                raise RuntimeStoreError(detach_error)

            monkeypatch.setattr(owner, "detach", failed_detach)
        first = await client.post(url, json=body, headers=_headers())
        monkeypatch.setattr(owner, "detach", detach)
        assert first.status == {None: 200, "invalid_params": 400, "worker_control_backpressure": 409}[detach_error]
        await registered[1].wait()
        retry = await client.post(url, json=body, headers=_headers())
        if detach_error:
            assert retry.status == 409
            assert (await retry.json())["error"]["code"] == "approval_resolution_pending"
        else:
            assert retry.status == 200
            assert await retry.json() == await first.json()
        assert not entries[1].event.is_set()
        body["request_id"] = entries[1].data["request_id"]
        body["execution_generation"] += 1
        stale = await client.post(url, json=body, headers=_headers(**{"Idempotency-Key": "second"}))
        assert stale.status == 409
        assert (await stale.json())["error"]["code"] == "stale_generation"
        assert not entries[1].event.is_set()
    finally:
        allow_prompt.set()
        for entry in entries:
            entry.event.set()
        await asyncio.gather(*list(adapter._active_run_tasks.values()))


@pytest.mark.asyncio
@pytest.mark.parametrize(("changed", "status", "code"), [
    ({"execution_generation": -1}, 409, "stale_generation"),
    ({"unexpected": True}, 400, "invalid_params"),
])
async def test_authority_preflight_refusals_do_not_exhaust_receipts(api, owner, monkeypatch, changed, status, code):
    adapter, client = api
    monkeypatch.setenv("HERMES_HOME", str(Path(owner.db.db_path).parent))
    monkeypatch.setattr(adapter._run_admissions, "MAX_RECEIPTS", 2)
    adapter.gateway_runner = owner.runner
    adapter._session_db = owner.db
    owner.runner._adapter_for_source = lambda source: adapter
    entry = _ApprovalEntry({"command": "test"})
    ready = asyncio.Event()
    generation = []

    async def turn(event):
        sid = event.source.chat_id
        live = owner.sessions[sid]
        current = owner.db.get_session(sid)["runtime_generation"]
        generation.append(current)
        monkeypatch.setitem(approval._gateway_queues, live.route, [entry])
        owner.register_approval(sid, current, live.route, entry.data)
        ready.set()
        await asyncio.to_thread(entry.event.wait)
        return "done"

    owner.runner._handle_message = turn
    try:
        response = await client.post("/v1/runs", json={"input": "test"}, headers=_headers())
        assert response.status == 202, await response.text()
        run = (await response.json())["run_id"]
        await ready.wait()
        url = f"/v1/runs/{run}/approval"
        body = {"request_id": entry.data["request_id"],
                "execution_generation": generation[0], "choice": "once"}
        for index in range(4):
            refused = await client.post(url, json={**body, **changed},
                headers=_headers(**{"Idempotency-Key": f"refusal-{index}"}))
            assert refused.status == status, await refused.text()
            assert (await refused.json())["error"]["code"] == code
            assert not entry.event.is_set()
        # A corrected payload may also reuse a refused key.
        accepted = await client.post(url, json=body,
            headers=_headers(**{"Idempotency-Key": "refusal-0"}))
        assert accepted.status == 200, await accepted.text()
        assert (await accepted.json())["status"] == "resolved"
        assert entry.event.is_set()
    finally:
        entry.event.set()
        await asyncio.gather(*list(adapter._active_run_tasks.values()))


@pytest.mark.asyncio
async def test_revoked_room_authorization_cannot_replay_receipt(api, monkeypatch):
    from gateway.platforms.api_server_room_grants import RoomGrantReauthorizationRequired

    adapter, client = api
    revoked = False
    claims = {field: "test" for field in ("room_id", "home_install_id", "authority_gateway_id",
              "authority_epoch", "member_id", "target_install_id", "target_profile")}

    def grant(request, *, permission):
        assert permission == "approve"
        if revoked:
            raise RoomGrantReauthorizationRequired("test revocation")
        return claims

    monkeypatch.setattr(adapter, "_room_grant_token", lambda request: "test-room-grant")
    monkeypatch.setattr(adapter, "_room_grant_claims", grant)
    first, second = _pending(adapter, "room", monkeypatch)
    body = {"choice": "once", "request_id": first.data["request_id"]}
    url = "/v1/runs/room/approval"
    accepted = await client.post(url, json=body, headers=_headers())
    assert accepted.status == 200
    replay = await client.post(url, json=body, headers=_headers())
    assert replay.status == 200
    revoked = True
    refused = await client.post(url, json=body,
        headers=_headers(**{"X-Hermes-Expected-Generation": str(uuid4())}))
    assert refused.status == 403
    assert (await refused.json())["error"]["code"] == "room_reauthorization_required"
    assert not second.event.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize(("initial_body", "status", "code"), [
    ({"choice": "once"}, 409, "approval_not_pending"),
    ({"choice": "invalid"}, 400, "invalid_approval_choice"),
])
async def test_non_mutating_failure_does_not_spend_approval_key(api, monkeypatch, initial_body, status, code):
    adapter, client = api
    first, second = _pending(adapter, "late-prompt", monkeypatch)
    monkeypatch.setitem(approval._gateway_queues, "late-prompt", [])
    url, body = "/v1/runs/late-prompt/approval", {"choice": "once"}
    missing = await client.post(url, json=initial_body, headers=_headers())
    assert missing.status == status
    assert (await missing.json())["error"]["code"] == code
    monkeypatch.setitem(approval._gateway_queues, "late-prompt", [first, second])
    resolved = await client.post(url, json=body, headers=_headers())
    assert resolved.status == 200
    duplicate = await client.post(url, json=body, headers=_headers())
    assert duplicate.status == 200
    assert await duplicate.json() == await resolved.json()
    assert first.event.is_set()
    assert not second.event.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("forgotten", [False, True])
async def test_generation_is_checked_after_auth_before_run_lookup(api, monkeypatch, forgotten):
    adapter, client = api
    if forgotten:
        _pending(adapter, "missing", monkeypatch)
        adapter._run_statuses.pop("missing")
        adapter._run_owners.pop("missing")
        adapter._run_approval_sessions.pop("missing")
    stale = {"X-Hermes-Expected-Generation": str(uuid4())}
    for url, body in (("/v1/runs", {"input": "test"}),
                      ("/v1/runs/missing/approval", {"choice": "once"})):
        denied = await client.post(url, json=body, headers=stale)
        assert denied.status == 401
        mismatch = await client.post(url, json=body, headers=_headers(**stale))
        assert mismatch.status == 409
        assert (await mismatch.json())["error"]["code"] == "server_generation_mismatch"
    missing = await client.post("/v1/runs/missing/approval", json={"choice": "once"}, headers=_headers())
    assert missing.status == 404
