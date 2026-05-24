"""Integration tests for SSE stateless chat completions streaming approvals.

Covers:
1. Streaming /v1/chat/completions with stream=True
2. Capturing event: approval.request mid-stream (triggered by mock/stub agent making tool approval callback)
3. Resolving the approval programmatically via POST /v1/runs/{id}/approval with choice
4. Verifying agent unblocks and finishes execution cleanly.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)


def _create_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    return app


@pytest.mark.asyncio
async def test_sse_approvals_integration():
    # Configure with an API key if we use custom headers, OR do not pass X-Hermes-Session-Key.
    # Actually, we don't need a custom X-Hermes-Session-Key or API key for a simple stateless test,
    # as long as we don't supply X-Hermes-Session-Key without setting APIServerAdapter's api_key,
    # or we can configured API key in APIServerAdapter.
    # Let's configure it with skip/no key API Server, but NOT pass X-Hermes-Session-Key to avoid 403.
    # If we don't pass X-Hermes-Session-Key, a session_id is automatically derived.
    config = PlatformConfig(enabled=True, extra={})
    adapter = APIServerAdapter(config)
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        def mock_run_conversation(user_message=None, conversation_history=None, task_id=None):
            from gateway.session_context import get_session_env
            from tools.approval import check_all_command_guards, get_current_session_key

            active_key = next(iter(adapter._run_approval_sessions.values()))
            assert get_current_session_key() == active_key
            assert get_session_env("HERMES_SESSION_KEY") == active_key
            assert get_session_env("HERMES_SESSION_PLATFORM") == "api_server"

            res = check_all_command_guards(
                command="rm -rf /tmp/test_dir",
                env_type="local",
            )
            choice = "once" if res.get("approved") else "deny"
            return {"final_response": f"Agent finished with choice: {choice}"}

        mock_agent = MagicMock()
        mock_agent.run_conversation.side_effect = mock_run_conversation
        mock_agent.session_prompt_tokens = 5
        mock_agent.session_completion_tokens = 10
        mock_agent.session_total_tokens = 15

        with patch.object(adapter, "_create_agent", return_value=mock_agent):
            payload = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "run dangerous command"}],
                "stream": True,
            }

            resp = await cli.post("/v1/chat/completions", json=payload)
            assert resp.status == 200

            run_id = None
            approval_seen = False
            
            reader = resp.content
            
            while True:
                line_bytes = await reader.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8").strip()
                if not line:
                    continue
                
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        data_json = json.loads(data_str)
                        if "id" in data_json:
                            run_id = data_json["id"]
                    except json.JSONDecodeError:
                        pass
                elif line.startswith("event: approval.request"):
                    approval_seen = True
                    data_line_bytes = await reader.readline()
                    data_line = data_line_bytes.decode("utf-8").strip()
                    assert data_line.startswith("data:")
                    approval_data = json.loads(data_line[5:].strip())
                    assert approval_data["command"] == "rm -rf /tmp/test_dir"
                    
                    assert run_id is not None
                    
                    # 2. While blocked, dispatch a programmatic HTTP POST to approve the command
                    approval_payload = {
                        "choice": "once",
                    }
                    app_resp = await cli.post(f"/v1/runs/{run_id}/approval", json=approval_payload)
                    assert app_resp.status == 200
                    app_json = await app_resp.json()
                    assert app_json["choice"] == "once"
                    assert app_json["resolved"] == 1

            assert approval_seen is True
            assert run_id in adapter._run_statuses
            assert adapter._run_statuses[run_id]["status"] == "completed"
            assert "Agent finished with choice: once" in adapter._run_statuses[run_id]["output"]


def test_parallel_delegate_workers_keep_gateway_approval_context():
    from gateway.session_context import clear_session_vars, get_session_env, set_session_vars
    from tools.approval import (
        get_current_session_key,
        reset_current_session_key,
        set_current_session_key,
    )
    from tools.delegate_tool import delegate_task

    session_key = "api-server-session"
    approval_token = set_current_session_key(session_key)
    session_tokens = set_session_vars(platform="api_server", session_key=session_key)
    seen = []

    def fake_run_single_child(task_index, goal, child=None, parent_agent=None, **_kwargs):
        seen.append((
            task_index,
            get_current_session_key(),
            get_session_env("HERMES_SESSION_KEY"),
            get_session_env("HERMES_SESSION_PLATFORM"),
        ))
        return {
            "task_index": task_index,
            "status": "success",
            "summary": goal,
            "api_calls": 1,
            "duration_seconds": 0,
        }

    parent_agent = SimpleNamespace(_delegate_depth=0, _interrupt_requested=False)
    try:
        with patch("tools.delegate_tool._load_config", return_value={"max_iterations": 1}), \
             patch("tools.delegate_tool._get_max_concurrent_children", return_value=2), \
             patch("tools.delegate_tool._resolve_delegation_credentials", return_value={
                 "model": None,
                 "provider": None,
                 "base_url": None,
                 "api_key": None,
                 "api_mode": None,
                 "command": None,
                 "args": None,
             }), \
             patch("tools.delegate_tool._build_child_agent", return_value=SimpleNamespace(_delegate_role="leaf")), \
             patch("tools.delegate_tool._run_single_child", side_effect=fake_run_single_child):
            result = json.loads(delegate_task(
                tasks=[
                    {"goal": "one"},
                    {"goal": "two"},
                ],
                parent_agent=parent_agent,
            ))
    finally:
        reset_current_session_key(approval_token)
        clear_session_vars(session_tokens)

    assert "error" not in result
    assert len(seen) == 2
    assert {row[1:] for row in seen} == {
        (session_key, session_key, "api_server"),
    }


def test_delegated_child_timeout_worker_keeps_gateway_approval_context():
    from gateway.session_context import clear_session_vars, get_session_env, set_session_vars
    from tools.approval import (
        get_current_session_key,
        reset_current_session_key,
        set_current_session_key,
    )
    from tools.delegate_tool import _run_single_child

    session_key = "api-server-child-session"
    approval_token = set_current_session_key(session_key)
    session_tokens = set_session_vars(platform="api_server", session_key=session_key)
    seen = []

    child = MagicMock()
    child.get_activity_summary.return_value = {
        "current_tool": None,
        "api_call_count": 0,
        "max_iterations": 1,
        "last_activity_desc": "",
    }

    def fake_child_run(**_kwargs):
        seen.append((
            get_current_session_key(),
            get_session_env("HERMES_SESSION_KEY"),
            get_session_env("HERMES_SESSION_PLATFORM"),
        ))
        return {"final_response": "done", "completed": True, "api_calls": 1}

    child.run_conversation.side_effect = fake_child_run
    parent_agent = SimpleNamespace(_current_task_id=None, _touch_activity=lambda _desc: None)

    try:
        result = _run_single_child(
            task_index=0,
            goal="child context check",
            child=child,
            parent_agent=parent_agent,
        )
    finally:
        reset_current_session_key(approval_token)
        clear_session_vars(session_tokens)

    assert result["status"] == "completed"
    assert seen == [(session_key, session_key, "api_server")]
