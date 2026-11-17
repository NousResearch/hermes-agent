"""POST /v1/runs ``output_schema``: the run record names the validated object (``result``) and the
verdict, and a run without a schema keeps its exact legacy shape."""
import asyncio
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter

_SCHEMA = {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}


async def _run(body, answer):
    adapter = _make_adapter()
    seen = {}

    def run_conversation(**kwargs):
        seen.update(kwargs)
        return {"final_response": answer}

    agent = MagicMock()
    agent.run_conversation.side_effect = run_conversation
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            accepted = await cli.post("/v1/runs", json=body)
            assert accepted.status == 202, await accepted.text()
            run_id = (await accepted.json())["run_id"]
            for _ in range(200):
                status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                if status["status"] not in {"queued", "running"}:
                    break
                await asyncio.sleep(0.01)
    return status, seen


@pytest.mark.asyncio
async def test_schema_run_exposes_validated_result_on_poll():
    status, seen = await _run({"input": "count the files", "output_schema": _SCHEMA},
                              'Here you go:\n```json\n{"count": 3}\n```')

    assert "OUTPUT CONTRACT" in seen["user_message"]
    assert status["status"] == "completed"
    assert status["result"] == {"count": 3} and status["schema_valid"] is True


@pytest.mark.asyncio
async def test_schema_run_with_invalid_answer_reports_errors_not_a_result():
    status, _ = await _run({"input": "count", "output_schema": _SCHEMA}, '{"count": "three"}')

    assert status["result"] is None and status["schema_valid"] is False
    assert any("count" in error for error in status["schema_errors"])


@pytest.mark.asyncio
async def test_run_without_schema_keeps_its_shape():
    status, seen = await _run({"input": "count"}, "three")

    assert seen["user_message"] == "count" and status["output"] == "three"
    assert not {"result", "schema_valid", "schema_errors"} & status.keys()


@pytest.mark.asyncio
async def test_invalid_schema_is_refused_before_any_run():
    adapter = _make_adapter()
    async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent") as create:
            refused = await cli.post("/v1/runs", json={"input": "x", "output_schema": {"type": 7}})
            assert refused.status == 400
            assert (await refused.json())["error"]["code"] == "invalid_output_schema"
            create.assert_not_called()
