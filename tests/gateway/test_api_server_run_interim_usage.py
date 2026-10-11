"""Run-correlated metering must be visible while a worker is still running."""

import asyncio
import threading
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter, _use_idempotency_db


@pytest.mark.asyncio
async def test_missing_later_call_usage_cannot_finalize_earlier_snapshot():
    adapter = _make_adapter(api_key="sk-meter")
    agent = MagicMock()
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_cache_read_tokens = 0
    agent.session_cache_write_tokens = 0
    agent.session_api_calls = 0

    def run_conversation(**kwargs):
        agent.session_prompt_tokens = 10
        agent.session_completion_tokens = 5
        agent.session_total_tokens = 15
        agent.session_api_calls = 1
        agent._run_usage_callback()
        agent.session_api_calls = 2  # provider call without usage
        return {"final_response": "unverified"}

    agent.run_conversation.side_effect = run_conversation
    headers = {"Authorization": "Bearer sk-meter"}
    async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            accepted = await cli.post("/v1/runs", json={"input": "partial"}, headers=headers)
            assert accepted.status == 202
            run_id = (await accepted.json())["run_id"]
            for _ in range(80):
                status = await (await cli.get(f"/v1/runs/{run_id}", headers=headers)).json()
                if status["status"] == "completed":
                    break
                await asyncio.sleep(0.025)
            assert status["status"] == "completed"
            assert status["run_usage"]["total_tokens"] == 15
            assert status["run_usage_final"] is False


@pytest.mark.asyncio
async def test_unmetered_completion_does_not_claim_final_run_usage():
    adapter = _make_adapter(api_key="sk-meter")
    agent = MagicMock()
    agent.session_prompt_tokens = 0
    agent.session_completion_tokens = 0
    agent.session_total_tokens = 0
    agent.session_cache_read_tokens = 0
    agent.session_cache_write_tokens = 0
    agent.session_api_calls = 0
    agent.run_conversation.return_value = {"final_response": "done"}
    headers = {"Authorization": "Bearer sk-meter"}
    async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            accepted = await cli.post("/v1/runs", json={"input": "unmetered"}, headers=headers)
            assert accepted.status == 202
            run_id = (await accepted.json())["run_id"]
            for _ in range(80):
                status = await (await cli.get(f"/v1/runs/{run_id}", headers=headers)).json()
                if status["status"] == "completed":
                    break
                await asyncio.sleep(0.025)
            assert status["status"] == "completed"
            assert status["run_usage_final"] is False
            assert status["run_usage"]["total_tokens"] == 0


@pytest.mark.asyncio
async def test_run_status_exposes_completed_call_usage_before_turn_finishes(tmp_path):
    from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
    adapter = _make_adapter(api_key="sk-meter")
    db_path = tmp_path / "runs.db"
    _use_idempotency_db(adapter, db_path)
    agent = MagicMock()
    agent.session_prompt_tokens = 20
    agent.session_completion_tokens = 5
    agent.session_total_tokens = 25
    agent.session_cache_read_tokens = 3
    agent.session_cache_write_tokens = 1
    agent.session_api_calls = 2
    reported = threading.Event()
    release = threading.Event()

    def run_conversation(**kwargs):
        agent.session_prompt_tokens = 31
        agent.session_completion_tokens = 9
        agent.session_total_tokens = 40
        agent.session_cache_read_tokens = 5
        agent.session_api_calls = 3
        agent._run_usage_callback()
        reported.set()
        release.wait(timeout=5)
        return {"final_response": "done"}

    agent.run_conversation.side_effect = run_conversation
    app = _create_runs_app(adapter)
    headers = {"Authorization": "Bearer sk-meter", "Idempotency-Key": "usage-test-key"}
    try:
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=agent):
                accepted = await cli.post("/v1/runs", json={"input": "meter"}, headers=headers)
                assert accepted.status == 202
                run_id = (await accepted.json())["run_id"]
                for _ in range(80):
                    status = await (await cli.get(f"/v1/runs/{run_id}", headers=headers)).json()
                    if status.get("run_usage"):
                        break
                    await asyncio.sleep(0.025)
                assert status["status"] == "running"
                assert reported.is_set()
                assert status["run_usage"] == {
                    "input_tokens": 11, "output_tokens": 4, "total_tokens": 15,
                    "cache_read_tokens": 2, "cache_write_tokens": 0,
                }
                assert status["run_usage_final"] is False
                reopened = RunIdempotencyStore(str(db_path))
                try:
                    stored = reopened._conn.execute(
                        "SELECT status_json FROM run_idempotency WHERE run_id=?", (run_id,)
                    ).fetchone()
                    assert stored is not None
                    import json
                    persisted = json.loads(stored[0])
                    assert persisted["run_usage"] == status["run_usage"]
                    assert persisted["run_usage_final"] is False
                finally:
                    reopened.close()
                release.set()
                for _ in range(80):
                    status = await (await cli.get(f"/v1/runs/{run_id}", headers=headers)).json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.025)
                assert status["status"] == "completed"
                assert status["run_usage"] == {
                    "input_tokens": 11, "output_tokens": 4, "total_tokens": 15,
                    "cache_read_tokens": 2, "cache_write_tokens": 0,
                }
                assert status["run_usage_final"] is True
    finally:
        release.set()
