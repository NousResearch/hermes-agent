"""Tests for /v1/runs endpoints: start, status, events, and stop.

Covers:
- POST /v1/runs — start a run (202)
- GET /v1/runs/{run_id} — poll run status
- GET /v1/runs/{run_id}/events — SSE event stream
- POST /v1/runs/{run_id}/stop — interrupt a running agent
- Auth, error handling, and cleanup
"""

import asyncio
import json
import threading
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
from tools import approval as approval_mod
from gateway.tool_channel_state import (
    clear_tool_channel_state,
    register_tool_notify,
    submit_tool_request,
    unregister_tool_notify,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_adapter(api_key: str = "", extra: dict | None = None) -> APIServerAdapter:
    """Create an adapter with optional API key."""
    cfg_extra = dict(extra or {})
    if api_key:
        cfg_extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=cfg_extra)
    adapter = APIServerAdapter(config)
    return adapter


def _create_runs_app(adapter: APIServerAdapter) -> web.Application:
    """Create an aiohttp app with /v1/runs routes registered."""
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/runs/{run_id}/tool_result", adapter._handle_run_tool_result)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


def _make_slow_agent(**kwargs):
    """Create a mock agent that blocks in run_conversation until interrupted.

    Returns (mock_agent, agent_ready_event, interrupt_event) where
    agent_ready_event is set once run_conversation starts, and
    interrupt_event is set when interrupt() is called.
    """
    ready = threading.Event()
    interrupted = threading.Event()

    mock_agent = MagicMock()

    def _do_interrupt(message=None):
        interrupted.set()

    mock_agent.interrupt = MagicMock(side_effect=_do_interrupt)

    def _slow_run(user_message=None, conversation_history=None, task_id=None):
        ready.set()
        # Block until interrupt() is called
        interrupted.wait(timeout=10)
        return {"final_response": "interrupted"}

    mock_agent.run_conversation.side_effect = _slow_run
    mock_agent.session_prompt_tokens = 0
    mock_agent.session_completion_tokens = 0
    mock_agent.session_total_tokens = 0

    return mock_agent, ready, interrupted


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


@pytest.fixture
def split_adapter():
    return _make_adapter(extra={
        "split_runtime": {
            "enabled": True,
            "routed_toolsets": ["file"],
            "request_timeout_seconds": 1,
        }
    })


# ---------------------------------------------------------------------------
# POST /v1/runs — start a run
# ---------------------------------------------------------------------------


class TestStartRun:
    @pytest.mark.asyncio
    async def test_start_returns_202(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                assert data["status"] == "started"
                assert data["run_id"].startswith("run_")

                status_resp = await cli.get(f"/v1/runs/{data['run_id']}")
                assert status_resp.status == 200
                status = await status_resp.json()
                assert status["run_id"] == data["run_id"]
                assert status["status"] in {"queued", "running", "completed"}
                assert status["object"] == "hermes.run"

    @pytest.mark.asyncio
    async def test_start_invalid_json_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                data="not json",
                headers={"Content-Type": "application/json"},
            )
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_missing_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"model": "test"})
            assert resp.status == 400
            data = await resp.json()
            assert "input" in data["error"]["message"]

    @pytest.mark.asyncio
    async def test_start_empty_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": ""})
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_start_invalid_history_does_not_allocate_run(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "hello", "conversation_history": {"role": "user"}},
            )
        assert resp.status == 400
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_start_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": "hello"})
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_start_with_valid_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "ok"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert resp.status == 202


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id} — poll run status
# ---------------------------------------------------------------------------


class TestRunStatus:
    @pytest.mark.asyncio
    async def test_status_completed_run_includes_output_and_usage(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 4
                mock_agent.session_completion_tokens = 2
                mock_agent.session_total_tokens = 6
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    assert status_resp.status == 200
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "completed"
                assert status["output"] == "done"
                assert status["usage"]["total_tokens"] == 6
                assert status["last_event"] == "run.completed"

    @pytest.mark.asyncio
    async def test_status_reflects_explicit_session_id(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "space-session"},
                )
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(20):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                mock_agent.run_conversation.assert_called_once()
                assert mock_agent.run_conversation.call_args.kwargs["task_id"] == "space-session"
                assert status["session_id"] == "space-session"

    @pytest.mark.asyncio
    async def test_status_not_found_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_nonexistent")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_status_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_any")
        assert resp.status == 401


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id}/events — SSE event stream
# ---------------------------------------------------------------------------


class TestRunEvents:
    @pytest.mark.asyncio
    async def test_events_stream_returns_completed(self, adapter):
        """Events stream should receive run.completed when agent finishes."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "Hello!"}
                mock_agent.session_prompt_tokens = 10
                mock_agent.session_completion_tokens = 5
                mock_agent.session_total_tokens = 15
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Subscribe to events
                events_resp = await cli.get(f"/v1/runs/{run_id}/events")
                assert events_resp.status == 200
                body = await events_resp.text()

                # Should contain run.completed
                assert "run.completed" in body
                assert "Hello!" in body



    @pytest.mark.asyncio
    async def test_approval_response_without_pending_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                data = await resp.json()
                run_id = data["run_id"]

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 409
                approval_data = await approval_resp.json()
                assert approval_data["error"]["code"] in {
                    "approval_not_active",
                    "approval_not_pending",
                }

    @pytest.mark.asyncio
    async def test_approval_string_false_does_not_resolve_all(self, adapter):
        """Quoted false must not fan out approval resolution across the queue."""
        app = _create_runs_app(adapter)
        run_id = "run_bool_parse"
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        adapter._run_approval_sessions[run_id] = "session-123"

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once", "all": "false"},
                )

        assert approval_resp.status == 200
        mock_resolve.assert_called_once_with(
            "session-123",
            "once",
            resolve_all=False,
        )

    @pytest.mark.asyncio
    async def test_approval_resolve_all_is_scoped_to_target_run(self, auth_adapter):
        """Same client session_id must not let one run approve another run's queue."""
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(auth_adapter, "_create_agent") as mock_create:
                victim_agent, victim_ready, victim_interrupted = _make_slow_agent()
                attacker_agent, attacker_ready, attacker_interrupted = _make_slow_agent()
                mock_create.side_effect = [victim_agent, attacker_agent]

                victim_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "victim", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                attacker_resp = await cli.post(
                    "/v1/runs",
                    json={"input": "attacker", "session_id": "shared-project"},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                assert victim_resp.status == 202
                assert attacker_resp.status == 202
                victim_run = (await victim_resp.json())["run_id"]
                attacker_run = (await attacker_resp.json())["run_id"]

                victim_ready.wait(timeout=3.0)
                attacker_ready.wait(timeout=3.0)
                assert auth_adapter._run_approval_sessions[victim_run] == victim_run
                assert auth_adapter._run_approval_sessions[attacker_run] == attacker_run
                assert auth_adapter._run_approval_sessions[victim_run] != auth_adapter._run_approval_sessions[attacker_run]

                victim_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c victim-danger",
                    "description": "victim approval",
                    "pattern_keys": ["shell-c"],
                })
                attacker_entry = approval_mod._ApprovalEntry({
                    "command": "bash -c attacker-danger",
                    "description": "attacker approval",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[victim_run] = [victim_entry]
                    approval_mod._gateway_queues[attacker_run] = [attacker_entry]

                approval_resp = await cli.post(
                    f"/v1/runs/{attacker_run}/approval",
                    json={"choice": "always", "resolve_all": True},
                    headers={"Authorization": "Bearer sk-secret"},
                )
                approval_data = await approval_resp.json()

                assert approval_resp.status == 200
                assert approval_data["resolved"] == 1
                assert attacker_entry.result == "always"
                assert attacker_entry.event.is_set()
                assert victim_entry.result is None
                assert not victim_entry.event.is_set()
                with approval_mod._lock:
                    assert approval_mod._gateway_queues[victim_run] == [victim_entry]
                    assert victim_run in approval_mod._gateway_queues
                    assert attacker_run not in approval_mod._gateway_queues

                # Clean up the synthetic pending victim approval and unblock the
                # slow test agents so their background run tasks can finish.
                with approval_mod._lock:
                    approval_mod._gateway_queues.pop(victim_run, None)
                victim_interrupted.set()
                attacker_interrupted.set()


    @pytest.mark.asyncio
    async def test_events_not_found_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_nonexistent/events")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_events_rejects_second_consumer(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_events_conflict"
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = 1.0
        adapter._run_event_consumers[run_id] = "events"
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(f"/v1/runs/{run_id}/events")
            data = await resp.json()
        assert resp.status == 409
        assert data["error"]["code"] == "run_events_consumer_conflict"
        adapter._run_streams.pop(run_id, None)
        adapter._run_streams_created.pop(run_id, None)
        adapter._run_event_consumers.pop(run_id, None)

    @pytest.mark.asyncio
    async def test_events_tool_executor_disabled_returns_409_and_does_not_leak_consumer(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_tool_executor_disabled"
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = 1.0
        adapter._run_tool_sessions[run_id] = "disabled-tool-session"
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(f"/v1/runs/{run_id}/events?tool_executor=1")
            data = await resp.json()
        assert resp.status == 409
        assert data["error"]["code"] == "split_runtime_disabled"
        assert run_id not in adapter._run_event_consumers
        adapter._run_streams.pop(run_id, None)
        adapter._run_streams_created.pop(run_id, None)
        adapter._run_tool_sessions.pop(run_id, None)

    @pytest.mark.asyncio
    async def test_events_tool_executor_round_trips_routed_read_file(self, split_adapter):
        """A fake local executor can answer a routed read_file over /v1/runs SSE."""
        app = _create_runs_app(split_adapter)
        attach_ready = threading.Event()

        class FakeAgent:
            session_prompt_tokens = 0
            session_completion_tokens = 0
            session_total_tokens = 0

            def run_conversation(self, user_message, conversation_history, task_id):
                import model_tools

                assert attach_ready.wait(timeout=3.0)
                result = model_tools.handle_function_call(
                    "read_file",
                    {"path": "README.md"},
                    task_id=task_id,
                    tool_call_id="call_http_e2e",
                    session_id="session-http-e2e",
                    skip_pre_tool_call_hook=True,
                    skip_tool_request_middleware=True,
                )
                return {"final_response": f"local executor returned: {result}"}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(split_adapter, "_create_agent", return_value=FakeAgent()):
                start_resp = await cli.post("/v1/runs", json={"input": "read local file"})
                assert start_resp.status == 202
                run_id = (await start_resp.json())["run_id"]

                events_resp = await cli.get(
                    f"/v1/runs/{run_id}/events?tool_executor=1",
                    headers={"X-Hermes-Tool-Executor-Token": "executor-token"},
                )
                assert events_resp.status == 200
                attach_ready.set()

                request_event = None
                for _ in range(10):
                    raw = await asyncio.wait_for(events_resp.content.readline(), timeout=3.0)
                    line = raw.decode().strip()
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        if event.get("event") == "tool.request":
                            request_event = event
                            break
                assert request_event is not None
                assert request_event["tool_call_id"] == "call_http_e2e"
                assert request_event["tool_name"] == "read_file"
                assert request_event["arguments"] == {"path": "README.md"}

                result_resp = await cli.post(
                    f"/v1/runs/{run_id}/tool_result",
                    json={"tool_call_id": "call_http_e2e", "result": "LOCAL README CONTENT"},
                    headers={"X-Hermes-Tool-Executor-Token": "executor-token"},
                )
                assert result_resp.status == 200

                body = await asyncio.wait_for(events_resp.text(), timeout=3.0)
                assert "tool.result" in body
                assert "run.completed" in body
                assert "LOCAL README CONTENT" in body

    @pytest.mark.asyncio
    async def test_events_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_any/events")
        assert resp.status == 401


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/tool_result — split-runtime local tool response
# ---------------------------------------------------------------------------


class TestRunToolResult:
    @pytest.mark.asyncio
    async def test_tool_result_returns_409_when_split_runtime_disabled(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_tool_disabled"
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        adapter._run_tool_sessions[run_id] = "disabled-session"
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/v1/runs/{run_id}/tool_result",
                json={"tool_call_id": "call_disabled", "result": "ignored"},
            )
            data = await resp.json()
        assert resp.status == 409
        assert data["error"]["code"] == "split_runtime_disabled"

    @pytest.mark.asyncio
    async def test_tool_result_resolves_pending_request_and_emits_event(self, split_adapter):
        app = _create_runs_app(split_adapter)
        run_id = "run_tool_result"
        session_key = "tool-session"
        q: asyncio.Queue = asyncio.Queue()
        split_adapter._run_statuses[run_id] = {"run_id": run_id, "status": "waiting_for_tool_result"}
        split_adapter._run_tool_sessions[run_id] = session_key
        split_adapter._run_streams[run_id] = q
        try:
            assert register_tool_notify(session_key, lambda request: None, "") is True
            entry = submit_tool_request(session_key, {
                "v": 1,
                "tool_call_id": "call_1",
                "tool_name": "read_file",
                "arguments": {"path": "README.md"},
            })
            assert entry is not None

            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    f"/v1/runs/{run_id}/tool_result",
                    json={"tool_call_id": "call_1", "result": "local output"},
                )
                assert resp.status == 200
                data = await resp.json()

            assert data == {
                "object": "hermes.run.tool_result_response",
                "run_id": run_id,
                "tool_call_id": "call_1",
                "status": "resolved",
            }
            assert entry.event.is_set()
            assert entry.result == "local output"
            assert split_adapter._run_statuses[run_id]["status"] == "running"
            event = await asyncio.wait_for(q.get(), timeout=1.0)
            assert event["event"] == "tool.result"
            assert event["tool_call_id"] == "call_1"
        finally:
            unregister_tool_notify(session_key)
            clear_tool_channel_state(session_key)

    @pytest.mark.asyncio
    async def test_tool_result_rejects_mismatched_executor_token(self, split_adapter):
        app = _create_runs_app(split_adapter)
        run_id = "run_tool_token_mismatch"
        session_key = "tool-session-token"
        split_adapter._run_statuses[run_id] = {"run_id": run_id, "status": "waiting_for_tool_result"}
        split_adapter._run_tool_sessions[run_id] = session_key
        try:
            assert register_tool_notify(session_key, lambda request: None, "client-a") is True
            entry = submit_tool_request(session_key, {"tool_call_id": "call_token"})
            assert entry is not None
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    f"/v1/runs/{run_id}/tool_result",
                    json={"tool_call_id": "call_token", "result": "wrong"},
                    headers={"X-Hermes-Tool-Executor-Token": "client-b"},
                )
                data = await resp.json()
        finally:
            unregister_tool_notify(session_key)
            clear_tool_channel_state(session_key)
        assert resp.status == 403
        assert data["error"]["code"] == "tool_result_executor_mismatch"
        assert entry.result is None

    @pytest.mark.asyncio
    async def test_tool_result_accepts_matching_executor_token(self, split_adapter):
        app = _create_runs_app(split_adapter)
        run_id = "run_tool_token_match"
        session_key = "tool-session-token-match"
        split_adapter._run_statuses[run_id] = {"run_id": run_id, "status": "waiting_for_tool_result"}
        split_adapter._run_tool_sessions[run_id] = session_key
        try:
            assert register_tool_notify(session_key, lambda request: None, "client-a") is True
            entry = submit_tool_request(session_key, {"tool_call_id": "call_token_ok"})
            assert entry is not None
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    f"/v1/runs/{run_id}/tool_result",
                    json={"tool_call_id": "call_token_ok", "result": "right"},
                    headers={"X-Hermes-Tool-Executor-Token": "client-a"},
                )
                data = await resp.json()
        finally:
            unregister_tool_notify(session_key)
            clear_tool_channel_state(session_key)
        assert resp.status == 200
        assert data["status"] == "resolved"
        assert entry.result == "right"

    @pytest.mark.asyncio
    async def test_tool_result_duplicate_is_idempotent(self, split_adapter):
        app = _create_runs_app(split_adapter)
        run_id = "run_tool_duplicate"
        session_key = "tool-session-duplicate"
        split_adapter._run_statuses[run_id] = {"run_id": run_id, "status": "waiting_for_tool_result"}
        split_adapter._run_tool_sessions[run_id] = session_key
        try:
            assert register_tool_notify(session_key, lambda request: None, "") is True
            entry = submit_tool_request(session_key, {"tool_call_id": "call_dup"})
            assert entry is not None
            async with TestClient(TestServer(app)) as cli:
                first = await cli.post(
                    f"/v1/runs/{run_id}/tool_result",
                    json={"tool_call_id": "call_dup", "result": "one"},
                )
                second = await cli.post(
                    f"/v1/runs/{run_id}/tool_result",
                    json={"tool_call_id": "call_dup", "result": "two"},
                )
                assert first.status == 200
                assert second.status == 200
                second_data = await second.json()
            assert second_data["status"] == "duplicate"
            assert entry.result == "one"
        finally:
            unregister_tool_notify(session_key)
            clear_tool_channel_state(session_key)

    @pytest.mark.asyncio
    async def test_tool_result_without_pending_request_returns_409(self, split_adapter):
        app = _create_runs_app(split_adapter)
        run_id = "run_tool_missing"
        split_adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        split_adapter._run_tool_sessions[run_id] = "missing-session"
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/v1/runs/{run_id}/tool_result",
                json={"tool_call_id": "call_missing", "result": "late"},
            )
            data = await resp.json()
        assert resp.status == 409
        assert data["error"]["code"] == "tool_result_not_pending"


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — interrupt a running agent
# ---------------------------------------------------------------------------


class TestStopRun:
    @pytest.mark.asyncio
    async def test_stop_running_agent(self, adapter):
        """Stop should interrupt the agent and cancel the task."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                # Wait for agent to start running in the thread
                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Verify agent ref is stored
                assert run_id in adapter._active_run_agents

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["run_id"] == run_id
                assert stop_data["status"] == "stopping"

                # Agent interrupt should have been called
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

                status_resp = await cli.get(f"/v1/runs/{run_id}")
                assert status_resp.status == 200
                status_data = await status_resp.json()
                assert status_data["status"] in {"stopping", "cancelled"}

                # Refs should be cleaned up
                await asyncio.sleep(0.5)
                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks

    @pytest.mark.asyncio
    async def test_stop_nonexistent_run_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_nonexistent/stop")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_stop_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_any/stop")
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_stop_already_completed_run_returns_404(self, adapter):
        """Stopping a run that already finished should return 404 (refs cleaned up)."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "done"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                # Start and wait for completion
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                await asyncio.sleep(0.3)

                # Run should be done, refs cleaned up
                assert run_id not in adapter._active_run_agents

                # Stop should return 404
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 404

    @pytest.mark.asyncio
    async def test_stop_interrupt_exception_does_not_crash(self, adapter):
        """If agent.interrupt() raises, stop should still succeed."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, interrupted = _make_slow_agent()

                # Override the interrupt side_effect to raise. Still trip
                # ``interrupted`` so the slow_run thread unblocks at teardown
                # — without this the agent thread blocks the full 10s
                # timeout and the test teardown waits the same amount.
                def _raising_interrupt(message=None):
                    interrupted.set()
                    raise RuntimeError("interrupt failed")

                mock_agent.interrupt = MagicMock(side_effect=_raising_interrupt)
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                stop_data = await stop_resp.json()
                assert stop_data["status"] == "stopping"

    @pytest.mark.asyncio
    async def test_stop_sends_sentinel_to_events_stream(self, adapter):
        """After stop, the events stream should close."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                # Start run
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                agent_ready.wait(timeout=3.0)
                await asyncio.sleep(0.1)

                # Subscribe to events in background
                events_task = asyncio.ensure_future(
                    cli.get(f"/v1/runs/{run_id}/events")
                )

                await asyncio.sleep(0.1)

                # Stop the run
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200

                # Events stream should close
                events_resp = await asyncio.wait_for(events_task, timeout=5.0)
                assert events_resp.status == 200
                body = await events_resp.text()
                # Stream should have received run.failed and closed
                assert "run.failed" in body or "stream closed" in body
