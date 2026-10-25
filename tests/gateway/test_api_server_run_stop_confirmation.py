"""A stop acknowledgement is not proof that the agent worker has stopped."""

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter, _make_slow_agent, _use_idempotency_db


async def _wait_status(cli, run_id, expected, headers=None):
    for _ in range(80):
        status = await (await cli.get(f"/v1/runs/{run_id}", headers=headers)).json()
        if status["status"] == expected:
            return status
        await asyncio.sleep(0.025)
    raise AssertionError(f"run did not reach {expected}: {status}")


@pytest.mark.asyncio
async def test_stop_interrupt_receipt_and_confirmation_survive_store_reopen(tmp_path):
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
    adapter = _make_adapter(api_key="sk-stop")
    db_path = tmp_path / "runs.db"
    _use_idempotency_db(adapter, db_path)
    agent, ready, interrupted = _make_slow_agent()
    release = threading.Event()
    def run_conversation(**kwargs):
        ready.set()
        interrupted.wait(timeout=5)
        release.wait(timeout=5)
        return {"interrupted": True, "final_response": "stopped"}

    agent.run_conversation.side_effect = run_conversation
    headers = {"Authorization": "Bearer sk-stop", "Idempotency-Key": "stop-key"}
    try:
        with patch.object(adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
                admitted = await cli.post("/v1/runs", json={"input": "stop"}, headers=headers)
                assert admitted.status == 202
                run_id = (await admitted.json())["run_id"]
                assert await asyncio.to_thread(ready.wait, 3)
                assert (await cli.post(f"/v1/runs/{run_id}/stop", headers=headers)).status == 200
                reopened = RunIdempotencyStore(str(db_path))
                try:
                    import json
                    row = reopened._conn.execute(
                        "SELECT status_json FROM run_idempotency WHERE run_id=?", (run_id,)
                    ).fetchone()
                    persisted = json.loads(row[0])
                    assert persisted["status"] == "stopping"
                    assert persisted["stop_interrupt_requested"] is True
                    assert persisted["stop_confirmed"] is False
                    release.set()
                    assert (await _wait_status(cli, run_id, "cancelled", headers=headers))["stop_confirmed"] is True
                    row = reopened._conn.execute(
                        "SELECT status_json FROM run_idempotency WHERE run_id=?", (run_id,)
                    ).fetchone()
                    assert json.loads(row[0])["stop_confirmed"] is True
                finally:
                    reopened.close()
    finally:
        release.set()
        adapter._run_idempotency_store.close()


@pytest.mark.asyncio
async def test_repeated_stop_preserves_first_request_and_interrupt_receipt():
    adapter = _make_adapter()
    ready, release = threading.Event(), threading.Event()
    agent = MagicMock()
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    def run_conversation(**kwargs):
        ready.set()
        release.wait(timeout=5)
        return {"interrupted": True, "final_response": "stopped"}

    agent.run_conversation.side_effect = run_conversation
    try:
        with patch.object(adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
                admitted = await cli.post("/v1/runs", json={"input": "repeat stop"})
                run_id = (await admitted.json())["run_id"]
                assert await asyncio.to_thread(ready.wait, 3)
                assert (await cli.post(f"/v1/runs/{run_id}/stop")).status == 200
                first = await (await cli.get(f"/v1/runs/{run_id}")).json()
                assert (await cli.post(f"/v1/runs/{run_id}/stop")).status == 200
                second = await (await cli.get(f"/v1/runs/{run_id}")).json()
                assert second["stop_requested_at"] == first["stop_requested_at"]
                assert second["stop_interrupt_requested"] is True
                agent.interrupt.assert_called_once()
                release.set()
                confirmed = await _wait_status(cli, run_id, "cancelled")
                assert confirmed["stop_confirmed"] is True
    finally:
        release.set()


@pytest.mark.asyncio
async def test_cancelled_handler_does_not_confirm_live_executor_thread():
    adapter = _make_adapter()
    ready, release = threading.Event(), threading.Event()
    agent = MagicMock()
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    def run_conversation(**kwargs):
        ready.set()
        release.wait(timeout=5)
        return {"interrupted": True, "final_response": "late"}

    agent.run_conversation.side_effect = run_conversation
    try:
        with patch.object(adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
                admitted = await cli.post("/v1/runs", json={"input": "worker outlives handler"})
                run_id = (await admitted.json())["run_id"]
                assert await asyncio.to_thread(ready.wait, 3)
                assert (await cli.post(f"/v1/runs/{run_id}/stop")).status == 200
                adapter._active_run_tasks[run_id].cancel()
                await asyncio.sleep(0.05)
                status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                assert status["status"] == "cancelled"
                assert status["stop_confirmed"] is False
                assert not release.is_set()
    finally:
        release.set()


@pytest.mark.asyncio
async def test_stop_is_confirmed_only_after_worker_returns_interrupted():
    adapter = _make_adapter()
    agent, ready, _ = _make_slow_agent()
    with patch.object(adapter, "_create_agent", return_value=agent):
        async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
            admitted = await cli.post("/v1/runs", json={"input": "stop"})
            run_id = (await admitted.json())["run_id"]
            assert await asyncio.to_thread(ready.wait, 3)
            stopped = await cli.post(f"/v1/runs/{run_id}/stop")
            assert stopped.status == 200
            assert (await stopped.json())["status"] == "stopping"
            confirmed = await _wait_status(cli, run_id, "cancelled")
            assert confirmed["stop_confirmed"] is True
            assert isinstance(confirmed["stop_requested_at"], (int, float))


@pytest.mark.asyncio
async def test_uncooperative_completed_worker_never_claims_stop_confirmed():
    adapter = _make_adapter()
    ready, release = threading.Event(), threading.Event()
    agent = MagicMock()
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0

    def run_conversation(**kwargs):
        ready.set()
        release.wait(timeout=5)
        return {"final_response": "still completed"}

    agent.run_conversation.side_effect = run_conversation
    try:
        with patch.object(adapter, "_create_agent", return_value=agent):
            async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
                admitted = await cli.post("/v1/runs", json={"input": "ignore stop"})
                run_id = (await admitted.json())["run_id"]
                assert await asyncio.to_thread(ready.wait, 3)
                stopped = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stopped.status == 200
                stopping = await (await cli.get(f"/v1/runs/{run_id}")).json()
                assert stopping["status"] == "stopping"
                assert stopping["stop_confirmed"] is False
                release.set()
                completed = await _wait_status(cli, run_id, "completed")
                assert completed["stop_confirmed"] is False
    finally:
        release.set()
