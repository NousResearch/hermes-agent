"""Legacy body run IDs share durable admission, never another run's identity."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from tests.gateway.test_api_server_runs import (
    _create_runs_app,
    _make_adapter,
    _use_idempotency_db,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"Idempotency-Key": "explicit-key"}])
async def test_client_run_id_http_admission_survives_retries_and_collisions(tmp_path, headers):
    adapter = _make_adapter(api_key="test-owner")
    other = _make_adapter(api_key="test-other-owner")
    for instance in (adapter, other):
        _use_idempotency_db(instance, tmp_path / "runs.db")
    requested = "run_" + "a" * 32
    body = {"run_id": requested, "input": "isolated test"}
    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    try:
        with patch.object(adapter, "_create_agent", return_value=agent), patch.object(
            other, "_create_agent", return_value=agent
        ):
            async with TestClient(TestServer(_create_runs_app(adapter))) as client, TestClient(
                TestServer(_create_runs_app(other))
            ) as other_client:
                owner_headers = {**headers, "Authorization": "Bearer test-owner"}
                other_headers = {**headers, "Authorization": "Bearer test-other-owner"}
                first = await client.post("/v1/runs", json=body, headers=owner_headers)
                assert first.status == 202
                assert (await first.json())["run_id"] == requested
                await asyncio.gather(*list(adapter._active_run_tasks.values()))
                assert agent.run_conversation.call_count == 1
                with patch.object(adapter, "_concurrency_limited_response", return_value=web.json_response(
                    {"error": "full"}, status=429
                )):
                    replay = await client.post("/v1/runs", json=body, headers=owner_headers)
                    assert replay.status == 202
                    assert (await replay.json())["run_id"] == requested
                for changed in ({**body, "input": "different"}, {**body, "run_id": "run_" + "b" * 32}):
                    # A body-only new ID is a new request, not a conflicting key.
                    if not headers and changed["run_id"] != requested:
                        continue
                    conflict = await client.post("/v1/runs", json=changed, headers=owner_headers)
                    assert conflict.status == 409
                for changed_headers in (
                    {**owner_headers, "Idempotency-Key": "another-key"},
                    {**owner_headers, "X-Hermes-Session-Key": "another-session"},
                ):
                    conflict = await client.post("/v1/runs", json=body, headers=changed_headers)
                    assert conflict.status == 409
                owner_scope = adapter._run_owners[requested]
                persisted = adapter._run_idempotency_store.status_for_run(owner_scope, requested)
                assert persisted["status"]["status"] == "completed"
                # `other` has no live owner cache: this must reach durable reservation.
                collision = await other_client.post("/v1/runs", json=body, headers=other_headers)
                assert collision.status == 409
                assert other._run_idempotency_store.status_for_run(owner_scope, requested) == persisted
                hidden = await other_client.get(f"/v1/runs/{requested}", headers=other_headers)
                assert hidden.status == 404
                # The collision must close its SQLite transaction so the next HTTP
                # admission can look up and reserve, without corrupting the first run.
                follow = await other_client.post("/v1/runs", json={**body, "run_id": "run_" + "c" * 32},
                                                 headers=other_headers)
                assert follow.status == 202
                await asyncio.gather(*list(other._active_run_tasks.values()))
                assert agent.run_conversation.call_count == 2
                original = await client.get(f"/v1/runs/{requested}", headers=owner_headers)
                assert original.status == 200
                assert (await original.json())["status"] == "completed"
                for invalid in ("../bad", "", 123, "run_" + "A" * 32):
                    rejected = await client.post("/v1/runs", json={**body, "run_id": invalid}, headers=owner_headers)
                    assert rejected.status == 400
                assert agent.run_conversation.call_count == 2
    finally:
        for instance in (adapter, other):
            await asyncio.gather(*list(instance._active_run_tasks.values()), return_exceptions=True)
            instance._run_idempotency_store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"Idempotency-Key": "explicit-key"}])
async def test_simultaneous_identical_client_ids_replay_after_history_loading(tmp_path, headers):
    adapter = _make_adapter(api_key="test-owner")
    _use_idempotency_db(adapter, tmp_path / "runs.db")
    arrived = 0
    both_arrived = asyncio.Event()

    async def load_history(_session_id):
        nonlocal arrived
        arrived += 1
        if arrived == 2:
            both_arrived.set()
        await asyncio.wait_for(both_arrived.wait(), timeout=5)
        return []

    agent = MagicMock()
    agent.run_conversation.return_value = {"final_response": "done"}
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    body = {"run_id": "run_" + "d" * 32, "session_id": "test-session", "input": "same request"}
    try:
        with patch.object(adapter, "_create_agent", return_value=agent), patch.object(
            adapter, "_conversation_history_for_session", side_effect=load_history
        ):
            async with TestClient(TestServer(_create_runs_app(adapter))) as client:
                replies = await asyncio.gather(*[
                    client.post("/v1/runs", json=body, headers={**headers, "Authorization": "Bearer test-owner"})
                    for _ in range(2)
                ])
                assert [reply.status for reply in replies] == [202, 202]
                assert [(await reply.json())["run_id"] for reply in replies] == [body["run_id"]] * 2
                assert sum(reply.headers.get("Idempotency-Replayed") == "true" for reply in replies) == 1
                await asyncio.gather(*list(adapter._active_run_tasks.values()))
                assert agent.run_conversation.call_count == 1
    finally:
        await asyncio.gather(*list(adapter._active_run_tasks.values()), return_exceptions=True)
        adapter._run_idempotency_store.close()


@pytest.mark.parametrize("scope,key", [("profile-b", "key-a"), ("profile-a", "key-b")])
def test_client_run_id_collision_keeps_original_owner_and_store_usable(tmp_path, scope, key):
    store = RunIdempotencyStore(str(tmp_path / "runs.db"))
    status = {"status": "queued", "run_id": "run_" + "a" * 32}
    try:
        assert store.reserve("profile-a", "key-a", "fingerprint", status["run_id"], status)[0] == "created"
        assert store.reserve(scope, key, "fingerprint", status["run_id"], status) == ("conflict", None)
        assert not store._conn.in_transaction
        assert store.lookup("profile-a", "key-a", "fingerprint")[0] == "reused"
        assert store.lookup(scope, key, "fingerprint") == ("missing", None)
        assert store.owns_run("profile-a", status["run_id"])
        assert not store.owns_run("profile-b", status["run_id"])
        next_status = {"status": "queued", "run_id": "run_" + "b" * 32}
        assert store.reserve(scope, key, "fingerprint", next_status["run_id"], next_status)[0] == "created"
    finally:
        store.close()
