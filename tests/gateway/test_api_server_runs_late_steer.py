"""A steer accepted after the finalizer's last drain is handed back as ``pending_steer`` (#132359)."""

import asyncio
import threading
from unittest.mock import patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agent.interrupt_control import InterruptControlMixin
from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter


class _Agent(InterruptControlMixin):
    session_prompt_tokens = session_completion_tokens = session_total_tokens = 0

    def __init__(self):
        self._pending_steer = None
        self._pending_steer_lock = threading.Lock()

    def run_conversation(self, *_args, **_kwargs):
        self.steer("during the turn")
        result = {"final_response": "done", "pending_steer": self._drain_pending_steer()}
        self.steer("after the final drain")
        return result


@pytest.mark.asyncio
async def test_steer_after_final_drain_surfaces_as_pending_steer():
    adapter = _make_adapter()
    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent", return_value=_Agent()):
            start_resp = await cli.post("/v1/runs", json={"input": "hello"})
            run_id = (await start_resp.json())["run_id"]
            for _ in range(40):
                if adapter._run_statuses.get(run_id, {}).get("status") == "completed":
                    break
                await asyncio.sleep(0.05)

    assert adapter._run_statuses[run_id]["status"] == "completed"
    assert adapter._run_statuses[run_id]["pending_steer"] == "during the turn\nafter the final drain"
