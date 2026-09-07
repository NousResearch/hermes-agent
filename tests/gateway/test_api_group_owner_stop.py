"""Known participant containment through real auth, HTTP and durable run state."""

import asyncio
import hashlib
import json
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway import hosted_rooms as rooms
from gateway.hosted_room_peer import decode_room_grant
from gateway.platforms import api_server_runs as runs
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from gateway.platforms.api_server_run_scope import ROOM_RUN_SCOPE_FIELDS, room_run_scope_key
from tests.gateway.test_api_server_runs import _make_adapter, _make_slow_agent

KEY = "owner-stop-test-key-not-a-real-secret"
OWNER = {"Authorization": f"Bearer {KEY}"}
STOP = "/v1/group-participants/stop"


@pytest_asyncio.fixture
async def adapter(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    value = _make_adapter(KEY)
    try:
        yield value
    finally:
        from agent.interrupt_compat import request_hard_interrupt
        for agent in list(value._active_run_agents.values()):
            request_hard_interrupt(agent)
        for session in list(value._run_approval_sessions.values()):
            runs._unregister_approval_notify(session)
        tasks = list(value._active_run_tasks.values())
        if tasks:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=3)
        value._run_idempotency_store.close()


def app_for(adapter):
    app = web.Application()
    for method, path, handler in runs._http_routes(adapter):
        app.router.add_route(method, path, handler)
        if path.startswith(STOP):
            app.router.add_route(method, "/p/{profile}" + path, handler)
    app.router.add_post("/v1/room-members/invitations", adapter._handle_room_member_invitation)
    return app


async def invite(cli, room_id="room-owner-stop", member_id="writer"):
    response = await cli.post("/v1/room-members/invitations", headers=OWNER, json={
        "room_id": room_id, "home_install_id": "install:unavailable-home",
        "authority_gateway_id": "install:unavailable-home", "authority_epoch": 1,
        "member_id": member_id,
    })
    result = await response.json()
    assert response.status == 201, result
    return result


def dispatch(invited, room_id="room-owner-stop", task="task-1"):
    prompt = "PRIVATE_GROUP_PROMPT: wait for the owner's decision."
    catalog = invited["catalog"]
    return {
        "protocol_version": 2, "room_id": room_id, "home_install_id": "install:unavailable-home",
        "authority_gateway_id": "install:unavailable-home", "authority_epoch": 1,
        "member_id": "writer", "target_install_id": catalog["installation_id"], "target_profile": "default",
        "task_id": task, "execution_generation": 1, "source_event_seq": 1,
        "cancellation_scope_id": "cancel-1", "trace_id": "trace-1",
        "prompt": prompt, "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(),
        "capability_digest": catalog["catalog_digest"],
        "execution_policy_digest": catalog["execution_policy"]["policy_digest"],
    }


def participation(payload):
    return {key: payload[key] for key in ROOM_RUN_SCOPE_FIELDS}


def command(identity, command_id="owner-stop-1"):
    return {"participant": identity, "command_id": command_id, "confirm": True}


async def submit(cli, invited, payload):
    return await cli.post("/v1/runs", json={"input": payload["prompt"], "hosted_room_dispatch": payload},
                          headers={"Authorization": f"HermesRoom {invited['grant']}",
                                   "Idempotency-Key": f"room:{payload['task_id']}:1"})


def seed(adapter, identity, *, run_id="run-known", status="running", local=False):
    scope = room_run_scope_key(identity)
    adapter._run_idempotency_store.reserve(scope, f"key-{run_id}", "fingerprint", run_id,
        {"run_id": run_id, "status": status, "output": "PRIVATE_OUTPUT_MUST_NOT_LEAK"}, owner_pid=999999999)
    if local:
        adapter._run_owners[run_id] = scope
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": status}
    return scope


def identity():
    return {"room_id": "room-owner-stop", "home_install_id": "install:unavailable-home",
            "authority_gateway_id": "install:unavailable-home", "authority_epoch": 1,
            "member_id": "writer", "target_install_id": rooms.local_authority_gateway_id(),
            "target_profile": "default"}


@pytest.mark.asyncio
@pytest.mark.parametrize("revoke", [False, True])
async def test_owner_stops_real_scoped_run_and_blocks_new_work_without_its_home(adapter, monkeypatch, revoke):
    agent, ready, interrupted = _make_slow_agent()
    create = MagicMock(return_value=agent)
    monkeypatch.setattr(adapter, "_create_agent", create)
    async with TestClient(TestServer(app_for(adapter))) as cli:
        invited = await invite(cli)
        payload = dispatch(invited)
        participant = participation(payload)
        accepted = await submit(cli, invited, payload)
        body = await accepted.json()
        assert accepted.status == 202, body
        run_id = body["run_id"]
        assert await asyncio.to_thread(ready.wait, 3)
        # Owner containment must not weaken ordinary run-ID ownership.
        assert (await cli.post(f"/v1/runs/{run_id}/stop", headers=OWNER)).status == 404
        if revoke:
            claims = decode_room_grant(adapter._room_grant_secret(), invited["grant"], permission="status")
            rooms.revoke_room_grant_scope(rooms.default_db_path(), claims=claims, expires_at=claims["status_expires_at"])
        stopped = await cli.post(STOP, headers=OWNER, json=command(participant))
        reply = await stopped.json()
        assert stopped.status == 200, reply
        assert reply["admissions_frozen"] is True
        assert reply["participant"] == participant
        # The bounded store snapshot may precede the worker's terminal update;
        # only readback after the worker settles can assert no recorded work.
        assert reply["work_state"] in {"stopping", "unresolved", "no_active_recorded_runs"}
        assert await asyncio.to_thread(interrupted.wait, 1)
        tasks = list(adapter._active_run_tasks.values())
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=3)
        stored = adapter._run_idempotency_store.status_for_run(room_run_scope_key(participant), run_id)
        assert stored["status"]["status"] == "cancelled"
        repeated = await cli.post(STOP, headers=OWNER, json=command(participant))
        repeated_body = await repeated.json()
        assert repeated.status == 200
        assert repeated_body["frozen_at"] == reply["frozen_at"]
        assert repeated_body["work_state"] == "no_active_recorded_runs"
        assert "PRIVATE_" not in json.dumps(repeated_body)
        assert "owner_pid" not in json.dumps(repeated_body)
        assert (await cli.get(STOP + "/owner-stop-1", headers=OWNER)).status == 200

        fresh = await invite(cli)
        refused = await submit(cli, fresh, dispatch(fresh, task="task-after-freeze"))
        refusal = await refused.json()
        assert refused.status == 409, refusal
        assert refusal["error"]["code"] == "group_work_frozen"
        assert create.call_count == 1
        replay = await submit(cli, fresh, dispatch(fresh))
        replay_body = await replay.json()
        assert replay.status == 202, replay_body
        assert replay_body["run_id"] == run_id and replay_body["replayed"] is True
        assert create.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("authorization", [None, "Bearer wrong", "HermesRoom scoped", "HermesReplicaRetirement cleanup"])
async def test_owner_actions_reject_non_owner_credentials_before_freezing(adapter, authorization):
    participant = identity()
    scope = seed(adapter, participant)
    headers = {} if authorization is None else {"Authorization": authorization}
    async with TestClient(TestServer(app_for(adapter))) as cli:
        for path in (STOP, STOP + "/cmd"):
            result = await cli.request("POST" if path == STOP else "GET", path,
                                       headers=headers, json=command(participant) if path == STOP else None)
            assert result.status == 401
    assert adapter._run_idempotency_store.is_scope_frozen(scope) is False


@pytest.mark.asyncio
async def test_missing_owner_key_never_uses_the_manual_listener_auth_bypass(adapter):
    participant = identity()
    seed(adapter, participant)
    adapter._api_key = ""
    async with TestClient(TestServer(app_for(adapter))) as cli:
        assert (await cli.post(STOP, json=command(participant))).status == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["profile_prefix", "named_home", "wrong_store", "memory_store"])
async def test_non_installation_context_cannot_claim_owner_control(adapter, monkeypatch, tmp_path, kind):
    participant = identity()
    seed(adapter, participant)
    path, expected = STOP, 403
    if kind == "profile_prefix":
        path = "/p/default" + STOP
    elif kind == "named_home":
        profile = tmp_path / "root" / "profiles" / "other"
        profile.mkdir(parents=True)
        monkeypatch.setenv("HERMES_HOME", str(profile))
    else:
        adapter._run_idempotency_store.close()
        location = ":memory:" if kind == "memory_store" else str(tmp_path / "elsewhere.db")
        adapter._run_idempotency_store = RunIdempotencyStore(location)
        expected = 503
    async with TestClient(TestServer(app_for(adapter))) as cli:
        response = await cli.post(path, headers=OWNER, json=command(participant))
        assert response.status == expected, await response.json()


@pytest.mark.asyncio
async def test_unknown_participant_is_not_a_successful_empty_stop(adapter):
    async with TestClient(TestServer(app_for(adapter))) as cli:
        response = await cli.post(STOP, headers=OWNER, json=command(identity()))
        assert response.status == 404, await response.json()


@pytest.mark.asyncio
async def test_foreign_owner_stays_unresolved_and_summary_excludes_private_output(adapter):
    participant = identity()
    seed(adapter, participant)
    other = {**participant, "room_id": "other-group"}
    other_scope = seed(adapter, other, run_id="run-other")
    async with TestClient(TestServer(app_for(adapter))) as cli:
        response = await cli.post(STOP, headers=OWNER, json=command(participant))
        body = await response.json()
        assert response.status == 200, body
        assert body["admissions_frozen"] is True and body["work_state"] == "unresolved"
        assert [run["run_id"] for run in body["runs"]] == ["run-known"]
        assert "PRIVATE_" not in json.dumps(body)
        assert "owner_pid" not in json.dumps(body)
        assert adapter._run_idempotency_store.is_scope_frozen(other_scope) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["confirm", "extra", "wrong_target", "raw_scope"])
async def test_invalid_owner_selection_does_not_write_a_freeze(adapter, change):
    participant = identity()
    scope = seed(adapter, participant)
    body = command(participant)
    if change == "confirm":
        body["confirm"] = "true"
    elif change == "extra":
        body["run_id"] = "run-known"
    elif change == "wrong_target":
        body["participant"] = {**participant, "target_install_id": "install:other"}
    else:
        body["participant"] = {"scope": scope}
    async with TestClient(TestServer(app_for(adapter))) as cli:
        response = await cli.post(STOP, headers=OWNER, json=body)
        assert response.status == 400, await response.json()
    assert adapter._run_idempotency_store.is_scope_frozen(scope) is False
