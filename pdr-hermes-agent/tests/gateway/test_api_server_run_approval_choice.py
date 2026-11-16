"""POST /v1/runs/{id}/approval normalizes the answer through tools.approval before its allow-list check."""

from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter, _make_slow_agent
from tools import approval as approval_mod
from tools import approval_gateway_wait


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", ["Approve", " ONCE "])
async def test_run_approval_resolves_spelling_variants_of_once_as_once(answer):
    adapter = _make_adapter()
    async with TestClient(TestServer(_create_runs_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent") as mock_create:
            mock_agent, agent_ready, _ = _make_slow_agent()
            mock_create.return_value = mock_agent
            run_id = (await (await cli.post("/v1/runs", json={"input": "hello"})).json())["run_id"]
            assert agent_ready.wait(timeout=3.0)
            pending = approval_gateway_wait._ApprovalEntry({"command": "rm -rf build", "pattern_keys": ["rm"]})
            with approval_mod._lock:
                approval_mod._gateway_queues[run_id] = [pending]
            try:
                resp = await cli.post(f"/v1/runs/{run_id}/approval", json={"choice": answer})

                assert resp.status == 200
                assert pending.result == "once"
            finally:
                await cli.post(f"/v1/runs/{run_id}/stop")
