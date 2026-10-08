"""/v1/runs reports the session a mid-run context compression moved the conversation to.

When compression rotates the agent onto a continuation session while a run is answering, the
answer is written to that continuation session. The terminal event, the polled status and the
status recovered after a gateway restart must name it, not the session the run was admitted to.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_runs import (
    _create_runs_app,
    _make_adapter,
    _use_idempotency_db,
)


class TestRunSessionRotation:
    @pytest.mark.asyncio
    async def test_mid_run_rotation_reports_the_session_holding_the_answer(self, tmp_path):
        """Context compression can rotate the agent onto a continuation session while a run is
        answering; the answer is then written there, not to the admitted session. The terminal
        event, the polled status and the status recovered after a restart must all name that
        continuation session, while an idempotent retry still resolves to the original run."""
        path = tmp_path / "idem.db"
        adapter = _make_adapter()
        _use_idempotency_db(adapter, path)
        agent = MagicMock()
        agent.session_id = "conversation-parent"
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0

        def rotate_then_answer(**kwargs):
            agent.session_id = "conversation-continuation"  # what compression does mid-turn
            return {"final_response": "answer"}

        agent.run_conversation.side_effect = rotate_then_answer
        request = {"input": "hello", "session_id": "conversation-parent"}
        headers = {"Idempotency-Key": "rotation-run"}

        async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
            with patch.object(adapter, "_create_agent", return_value=agent) as create:
                started = await cli.post("/v1/runs", json=request, headers=headers)
                run_id = (await started.json())["run_id"]
                body = await (await cli.get(f"/v1/runs/{run_id}/events")).text()
                status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                replay = await cli.post("/v1/runs", json=request, headers=headers)
                replayed_run_id = (await replay.json())["run_id"]

        assert create.call_args.kwargs["session_id"] == "conversation-parent"
        completed = next(
            event for line in body.split("\n") if line.startswith("data: ")
            for event in [json.loads(line.removeprefix("data: "))]
            if event.get("event") == "run.completed")
        assert completed["output"] == "answer"
        assert completed["session_id"] == "conversation-continuation"
        assert status["status"] == "completed"
        assert status["session_id"] == "conversation-continuation"
        assert replay.status == 202
        assert replay.headers["Idempotency-Replayed"] == "true"
        assert replayed_run_id == run_id
        agent.run_conversation.assert_called_once()
        adapter._run_idempotency_store.close()

        restarted = _make_adapter()
        _use_idempotency_db(restarted, path)
        async with TestClient(TestServer(_create_runs_app(restarted))) as cli:
            recovered = await (await cli.get(f"/v1/runs/{run_id}")).json()
        assert recovered["status"] == "completed"
        assert recovered["output"] == "answer"
        assert recovered["session_id"] == "conversation-continuation"
