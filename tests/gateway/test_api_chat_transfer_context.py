"""Authenticated API runs stamp transfer provenance into the actual tool context."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer
from aiohttp import web

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


@pytest.mark.asyncio
async def test_run_transfer_target_uses_selected_body_session_over_declared_header():
    adapter = _make_adapter(api_key="sk-secret")
    captured = {}
    agent = MagicMock()
    agent.run_conversation.side_effect = lambda **kwargs: (
        captured.update(context=_current_tool_invocation_context()) or {"final_response": "done"})
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    app = _create_runs_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            admitted = await cli.post("/v1/runs", json={"input": "Transfer ABC-1 to this chat",
                "session_id": "selected-conversation"}, headers={
                    "Authorization": "Bearer sk-secret",
                    "X-Hermes-Session-Key": "other-declared-conversation"})
            assert admitted.status == 202
            run_id = (await admitted.json())["run_id"]
            for _ in range(40):
                response = await cli.get(f"/v1/runs/{run_id}",
                                         headers={"Authorization": "Bearer sk-secret"})
                if (await response.json())["status"] == "completed":
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError("authenticated run did not complete")
    context = captured["context"]
    assert context.session_id == "selected-conversation"
    assert context.api_conversation_key == "selected-conversation"


@pytest.mark.asyncio
async def test_authenticated_chat_completion_stamps_selected_transfer_target():
    adapter = _make_adapter(api_key="sk-secret")
    captured = {}
    agent = MagicMock()
    agent.run_conversation.side_effect = lambda **kwargs: (
        captured.update(context=_current_tool_invocation_context()) or {"final_response": "done"})
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    app = _create_runs_app(adapter)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            denied = await cli.post("/v1/chat/completions", json={"messages": [
                {"role": "user", "content": "Transfer ABC-1 to this chat"}]},
                headers={"X-Hermes-Session-Id": "selected-conversation"})
            assert denied.status == 401
            admitted = await cli.post("/v1/chat/completions", json={"messages": [
                {"role": "user", "content": "Transfer ABC-1 to this chat"}]}, headers={
                    "Authorization": "Bearer sk-secret",
                    "X-Hermes-Session-Id": "selected-conversation"})
            assert admitted.status == 200, await admitted.text()
    context = captured["context"]
    assert context.platform == "api_server"
    assert context.user_message == "Transfer ABC-1 to this chat"
    assert context.api_control_principal.startswith("principal:")
    assert context.api_conversation_key == context.session_id == "selected-conversation"


@pytest.mark.asyncio
async def test_native_api_transfer_provenance_revoked_after_key_rotation(api, owner):
    from gateway.session_api_turn import prepare_api_execution
    from gateway.session_contract import SessionRef
    from hermes_state_runtime import list_session_admissions

    async def handle(event):
        from gateway.session_results import execution_result
        execution_result.get()["result"] = {"final_response": "done", "messages": []}
        return "done"

    owner.runner._handle_message = handle
    api._api_key = "first-secret-long-enough"
    app = web.Application()
    app.router.add_post("/v1/chat/completions", api._handle_chat_completions)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post("/v1/chat/completions", json={"messages": [
            {"role": "user", "content": "Transfer ABC-1 to this chat"}]}, headers={
                "Authorization": "Bearer first-secret-long-enough",
                "X-Hermes-Session-Id": "selected-conversation"})
        assert response.status == 200, await response.text()
    saved = list_session_admissions(owner.db, session_id="selected-conversation", pending_only=False)[0]["payload"]
    assert saved["api_turn_v1"]["transfer_auth"]["principal"].startswith("principal:")
    api._api_key = "rotated-secret-long-enough"
    restored = prepare_api_execution(owner, SessionRef("default", "selected-conversation"), saved)
    assert restored["transfer_auth"] is None


@pytest.mark.asyncio
async def test_authenticated_native_api_admission_rebinds_transfer_instruction(api, owner):
    from gateway.run import GatewayRunner
    from gateway.session import SessionContext
    from gateway.session_context import clear_session_vars
    from gateway.session_results import execution_result

    captured = {}
    async def handle(event):
        ctx = SessionContext(source=event.source, connected_platforms=[], home_channels={},
                             session_key="selected-conversation", session_id="selected-conversation")
        tokens = GatewayRunner._set_session_env(owner.runner, ctx)
        try:
            captured["context"] = _current_tool_invocation_context()
        finally:
            clear_session_vars(tokens)
        execution_result.get()["result"] = {"final_response": "done", "messages": []}
        return "done"

    owner.runner._handle_message = handle
    api._api_key = "sk-secret-long-enough"
    app = web.Application()
    app.router.add_post("/v1/chat/completions", api._handle_chat_completions)
    async with TestClient(TestServer(app)) as cli:
        response = await cli.post("/v1/chat/completions", json={"messages": [
            {"role": "user", "content": "Transfer ABC-1 to this chat"}]}, headers={
                "Authorization": "Bearer sk-secret-long-enough",
                "X-Hermes-Session-Id": "selected-conversation"})
        assert response.status == 200, await response.text()
    context = captured["context"]
    assert context.user_message == "Transfer ABC-1 to this chat"
    assert context.api_control_principal.startswith("principal:")
    assert context.api_conversation_key == context.session_id == "selected-conversation"
