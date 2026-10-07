"""Room Stop fences one admission identity without starting an uncertain generation."""

import asyncio
import hashlib
import time
from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.hosted_room_peer import decode_room_grant, issue_room_grant
from gateway.platforms import api_server
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore


def _adapter(path):
    adapter = api_server.APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "test-room-key"}))
    adapter._run_idempotency_store.close()
    adapter._run_idempotency_store = RunIdempotencyStore(str(path))
    return adapter


def _app(adapter):
    app = web.Application()
    app.router.add_post("/v1/room-members/invitations", adapter._handle_room_member_invitation)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_post("/v1/runs/stop", adapter._handle_stop_run)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


async def _invitation(cli):
    response = await cli.post(
        "/v1/room-members/invitations", headers={"Authorization": "Bearer test-room-key"},
        json={"room_id": "room-1", "home_install_id": "home", "authority_gateway_id": "home",
              "authority_epoch": 1, "member_id": "member-1"})
    assert response.status == 201
    invitation = await response.json()
    catalog, prompt = invitation["catalog"], "Run this generation once."
    dispatch = {
        "protocol_version": 2, "room_id": "room-1", "home_install_id": "home",
        "authority_gateway_id": "home", "authority_epoch": 1, "member_id": "member-1",
        "target_install_id": catalog["installation_id"], "target_profile": "default",
        "task_id": "task-1", "execution_generation": 1, "source_event_seq": 1,
        "cancellation_scope_id": "cancel-1", "prompt": prompt,
        "prompt_digest": hashlib.sha256(prompt.encode()).hexdigest(),
        "capability_digest": catalog["catalog_digest"],
        "execution_policy_digest": catalog["execution_policy"]["policy_digest"], "trace_id": "trace-1"}
    return invitation["grant"], {"input": prompt, "hosted_room_dispatch": dispatch}


def _headers(grant):
    return {"Authorization": f"HermesRoom {grant}", "Idempotency-Key": "room:task-1:1"}


@pytest.mark.asyncio
async def test_stop_only_grant_fences_exact_scope_through_recovery_horizon(tmp_path, monkeypatch):
    adapter = _adapter(tmp_path / "runs.db")
    async with TestClient(TestServer(_app(adapter))) as cli:
        grant, body = await _invitation(cli)
        claims = decode_room_grant(adapter._room_grant_secret(), grant, permission="stop")
        coordinates = {field: claims[field] for field in (
            "room_id", "home_install_id", "authority_gateway_id", "authority_epoch", "member_id",
            "target_install_id", "target_profile", "execution_policy_digest")}
        # A valid stop horizon survives expired dispatch access and current catalog drift.
        stop_grant = issue_room_grant(
            adapter._room_grant_secret(), grant_id="stop-only", **coordinates,
            issued_at=time.time() - 120, ttl_seconds=60, status_ttl_seconds=3600, permissions=("stop",))
        original_dispatch = body["hosted_room_dispatch"]
        for field, value in (
                ("room_id", "other-room"), ("member_id", "other-member"),
                ("authority_epoch", 2), ("target_profile", "other-profile"), ("task_id", "other-task")):
            wrong_body = {**body, "hosted_room_dispatch": {**original_dispatch, field: value}}
            denied = await cli.post("/v1/runs/stop", headers=_headers(stop_grant), json=wrong_body)
            assert denied.status == 401
        token = api_server._api_request_profile.set("other-profile")
        try:
            # Direct invocation proves the target-profile guard as well as signed coordinates.
            request = MagicMock(headers=_headers(stop_grant), match_info={})
            from unittest.mock import AsyncMock
            request.json = AsyncMock(return_value=body)
            assert (await adapter._handle_stop_run(request)).status == 401
        finally:
            api_server._api_request_profile.reset(token)
        refused = await cli.post("/v1/runs", headers=_headers(stop_grant), json=body)
        assert refused.status == 401
        assert adapter._run_idempotency_store._conn.execute("SELECT COUNT(*) FROM run_idempotency").fetchone()[0] == 0
        from gateway.platforms import api_server_room_grants
        monkeypatch.setattr(api_server_room_grants, "_local_room_catalog", MagicMock(side_effect=AssertionError("catalog used")))
        first = await cli.post("/v1/runs/stop", headers=_headers(stop_grant), json=body)
        assert first.status == 200
        cancelled = await first.json()
        assert cancelled["status"] == "cancelled" and cancelled["admission_cancelled"]
        second = await cli.post("/v1/runs/stop", headers=_headers(stop_grant), json=body)
        assert (await second.json())["run_id"] == cancelled["run_id"]
    store = adapter._run_idempotency_store
    row = store._conn.execute("SELECT scope, idempotency_key FROM run_idempotency").fetchone()
    store.close()
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    try:
        from gateway.platforms import api_server_run_idempotency
        monkeypatch.setattr(api_server_run_idempotency.time, "time", lambda: claims["status_expires_at"] + 86400)
        outcome, record = store.reserve(*row, "late-payload", "late-run", {"status": "queued"})
        assert outcome == "reused" and record["run_id"] == cancelled["run_id"]
        # The identical client key in another authenticated namespace stays independent.
        assert store.reserve("other-scope", row[1], "late-payload", "other-run", {"status": "queued"})[0] == "created"
    finally:
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before_reserve", "after_reserve", "foreign_owner"])
async def test_stop_racing_admission_never_acknowledges_live_work(tmp_path, monkeypatch, phase):
    owner = _adapter(tmp_path / "runs.db")
    controller = _adapter(tmp_path / "runs.db")
    entered, release = asyncio.Event(), asyncio.Event()
    ready, interrupted = asyncio.Event(), asyncio.Event()
    loop = asyncio.get_running_loop()
    import threading
    finish = threading.Event()
    agent = MagicMock()

    def run(**kwargs):
        loop.call_soon_threadsafe(ready.set)
        assert finish.wait(10)
        return {"interrupted": True}

    def interrupt(*args, **kwargs):
        loop.call_soon_threadsafe(interrupted.set)
        finish.set()

    agent.run_conversation.side_effect = run
    agent.interrupt.side_effect = interrupt
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    create = MagicMock(return_value=agent)
    monkeypatch.setattr(owner, "_create_agent", create)
    if phase == "before_reserve":
        original_history = owner._conversation_history_for_session

        async def pause_history(*args, **kwargs):
            entered.set()
            await release.wait()
            return await original_history(*args, **kwargs)

        monkeypatch.setattr(owner, "_conversation_history_for_session", pause_history)
    elif phase == "after_reserve":
        async def pause_mailbox(*args, **kwargs):
            entered.set()
            await release.wait()
            return None

        monkeypatch.setattr(owner, "_admit_to_live_bot_chat", pause_mailbox)
    try:
        async with TestClient(TestServer(_app(owner))) as cli, TestClient(TestServer(_app(controller))) as control:
            grant, body = await _invitation(cli)
            original = asyncio.create_task(cli.post("/v1/runs", headers=_headers(grant), json=body))
            await asyncio.wait_for(ready.wait() if phase == "foreign_owner" else entered.wait(), 10)
            stop = await control.post("/v1/runs/stop", headers=_headers(grant), json=body)
            assert stop.status == 200
            status = await stop.json()
            assert status["status"] == ("cancelled" if phase == "before_reserve" else "stopping")
            release.set()
            admission = await original
            assert admission.status == 202
            run_id = (await admission.json())["run_id"]
            assert run_id == status["run_id"]
            if phase == "foreign_owner":
                await asyncio.wait_for(interrupted.wait(), 10)
            deadline = loop.time() + 10
            while True:
                polled = await control.get(f"/v1/runs/{run_id}", headers=_headers(grant))
                receipt = await polled.json()
                if receipt["status"] == "cancelled":
                    break
                assert loop.time() < deadline, receipt
                await asyncio.sleep(0.02)
            assert create.call_count == (1 if phase == "foreign_owner" else 0)
            assert agent.run_conversation.call_count == (1 if phase == "foreign_owner" else 0)
            # A stopped existing run must remain fenced after ordinary retention too.
            owner._run_idempotency_store._conn.execute(
                "UPDATE run_idempotency SET updated_at=0, retention_until=1")
            owner._run_idempotency_store._conn.commit()
            late = await cli.post("/v1/runs", headers=_headers(grant), json=body)
            assert (await late.json())["run_id"] == run_id
    finally:
        release.set()
        finish.set()
        owner._run_idempotency_store.close()
        controller._run_idempotency_store.close()
