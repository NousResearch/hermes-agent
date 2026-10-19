"""Authenticated API runs stamp transfer provenance into the actual tool context."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_runs import _create_runs_app, _make_adapter
from tools.registry import _current_tool_invocation_context


@pytest.mark.asyncio
async def test_authenticated_run_stamps_transfer_instruction_and_conversation():
    adapter = _make_adapter(api_key="sk-secret")
    captured = {}
    agent = MagicMock()
    agent.run_conversation.side_effect = lambda **kwargs: (
        captured.update(context=_current_tool_invocation_context()) or {"final_response": "done"})
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            denied = await cli.post("/v1/runs", json={"input": "Transfer ABC-1 to this chat",
                "session_id": "same-conversation"})
            assert denied.status == 401
            admitted = await cli.post("/v1/runs", json={"input": "Transfer ABC-1 to this chat",
                "session_id": "same-conversation"}, headers={"Authorization": "Bearer sk-secret"})
            assert admitted.status == 202
            run_id = (await admitted.json())["run_id"]
            for _ in range(40):
                response = await cli.get(f"/v1/runs/{run_id}",
                                         headers={"Authorization": "Bearer sk-secret"})
                assert response.status == 200
                if (await response.json())["status"] == "completed":
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError("authenticated run did not complete")
    context = captured["context"]
    assert context.platform == "api_server"
    assert context.user_message == "Transfer ABC-1 to this chat"
    assert context.api_control_principal.startswith("principal:")
    assert context.api_control_transport_family
    assert context.api_conversation_key == "same-conversation"
