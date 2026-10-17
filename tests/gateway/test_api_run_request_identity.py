"""Client-addressed run identity survives compression, replay, and adapter restart."""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent.compression_facade import _rebind_caller_session_context
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
from hermes_state import SessionDB
from tools.mcp_tool_handlers import _call_tool_racing_stdio_death


@pytest.mark.asyncio
@pytest.mark.parametrize("resumed", [False, True])
async def test_compressed_request_identity_survives_callback_replay_restart(tmp_path, resumed):
    db = SessionDB(tmp_path / "state.db")
    db.create_session(session_id="original", source="api_server", model="test")
    initial = "original"
    if resumed:
        db.end_session("original", "compression")
        db.create_session(session_id="tip", source="api_server", model="test", parent_session_id="original")
        initial = "tip"
    received = []

    async def call_tool(name, *, arguments, meta):
        received.append(meta)
        return "ok"

    server = SimpleNamespace(_config={"forward_session_context": True}, session=SimpleNamespace(call_tool=call_tool))

    class RotatingAgent:
        session_prompt_tokens = session_completion_tokens = session_total_tokens = 0
        model = "test"
        provider = "test"

        def __init__(self, **kwargs):
            self.session_id = kwargs["session_id"]

        def run_conversation(self, **kwargs):
            assert self.session_id == initial
            asyncio.run(_call_tool_racing_stdio_death(server, "harness", "inspect", {}))
            db.end_session(self.session_id, "compression")
            db.create_session(session_id="rotated", source="api_server", model="test", parent_session_id=self.session_id)
            self.session_id = "rotated"
            _rebind_caller_session_context(self)
            asyncio.run(_call_tool_racing_stdio_death(server, "harness", "inspect", {}))
            return {"final_response": "done"}

    path = tmp_path / "runs.db"
    run_id = None
    try:
        for restarting in (False, True):
            adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "owner-key"}))
            adapter._session_db = db
            adapter._run_idempotency_store.close()
            adapter._run_idempotency_store = RunIdempotencyStore(str(path))
            app = web.Application()
            app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
            app.router.add_post("/v1/runs", adapter._handle_runs)
            app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
            try:
                async with TestClient(TestServer(app)) as client:
                    headers = {"Authorization": "Bearer owner-key", "Idempotency-Key": "original-request"}
                    capabilities = await client.get("/v1/capabilities", headers=headers)
                    assert (await capabilities.json())["features"]["runs_idempotency"]["request_session_id"] is True
                    with patch.object(adapter, "_create_agent", side_effect=RotatingAgent) as create:
                        admission = await client.post("/v1/runs", headers=headers,
                                                      json={"session_id": "original", "input": "inspect"})
                        assert admission.status == 202, await admission.text()
                        body = await admission.json()
                        if restarting:
                            assert body["run_id"] == run_id and body["replayed"]
                            create.assert_not_called()
                        else:
                            run_id = body["run_id"]
                            await asyncio.wait_for(asyncio.gather(*adapter._background_tasks), timeout=10)
                        response = await client.get(f"/v1/runs/{run_id}", headers=headers)
                        receipt = await response.json()
                        assert receipt["status"] == "completed", receipt
                        assert receipt["session_id"] == initial
                        assert receipt["request_session_id"] == "original"
            finally:
                adapter._run_idempotency_store.close()
        assert [meta["com.nousresearch.hermes/session_id"] for meta in received] == [initial, "rotated"]
        assert all(meta["com.nousresearch.hermes/request_session_id"] == "original" for meta in received)
        assert all(meta["com.nousresearch.hermes/session_key"] == run_id for meta in received)
    finally:
        db.close()
