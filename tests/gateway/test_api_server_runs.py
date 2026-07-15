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
import time
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    _approval_event_choices,
    cors_middleware,
    security_headers_middleware,
)
from tools import approval as approval_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smart_denied", "allow_permanent", "expected"),
    [
        (False, True, ["once", "session", "always", "deny"]),
        (False, False, ["once", "session", "deny"]),
        (True, True, ["once", "deny"]),
        (True, False, ["once", "deny"]),
    ],
)
def test_approval_event_choices_follow_backend_capabilities(
    smart_denied, allow_permanent, expected
):
    assert _approval_event_choices(
        smart_denied=smart_denied,
        allow_permanent=allow_permanent,
    ) == expected


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    """Create an adapter with optional API key."""
    extra = {}
    if api_key:
        extra["key"] = api_key
    config = PlatformConfig(enabled=True, extra=extra)
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
    async def test_start_invalid_permission_mode_does_not_allocate_run(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/v1/runs",
                json={"input": "hello", "permission_mode": "unrestricted"},
            )
            data = await resp.json()

        assert resp.status == 400
        assert data["error"]["code"] == "invalid_permission_mode"
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}

    @pytest.mark.asyncio
    async def test_full_access_permission_is_scoped_to_one_run(self, adapter):
        app = _create_runs_app(adapter)
        observed = []
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()

                def run_conversation(**_kwargs):
                    observed.append(approval_mod.is_current_session_yolo_enabled())
                    return {"final_response": "done"}

                mock_agent.run_conversation.side_effect = run_conversation
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "permission_mode": "full-access"},
                )
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]

                for _ in range(20):
                    status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert observed == [True]
        assert approval_mod.is_session_yolo_enabled(run_id) is False

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

    @pytest.mark.asyncio
    async def test_idempotency_key_replays_same_run_without_duplicate_agent(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "ok"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent
                headers = {"Idempotency-Key": "mobile-submit-1"}
                body = {"input": "hello", "session_id": "session-1"}

                first = await cli.post("/v1/runs", json=body, headers=headers)
                second = await cli.post("/v1/runs", json=body, headers=headers)
                first_data = await first.json()
                second_data = await second.json()

                assert first.status == 202
                assert second.status == 202
                assert second_data["run_id"] == first_data["run_id"]
                assert second_data["idempotent_replay"] is True
                for _ in range(20):
                    if mock_agent.run_conversation.call_count == 1:
                        break
                    await asyncio.sleep(0.05)
                assert mock_agent.run_conversation.call_count == 1

    @pytest.mark.asyncio
    async def test_idempotency_key_rejects_different_run_payload(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {"final_response": "ok"}
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent
                headers = {"Idempotency-Key": "mobile-submit-1"}

                first = await cli.post("/v1/runs", json={"input": "hello"}, headers=headers)
                conflict = await cli.post("/v1/runs", json={"input": "different"}, headers=headers)
                conflict_data = await conflict.json()

                assert first.status == 202
                assert conflict.status == 409
                assert conflict_data["error"]["code"] == "idempotency_key_reused"

    @pytest.mark.asyncio
    async def test_installed_skill_slash_command_is_expanded_for_api_run(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(adapter, "_create_agent") as mock_create,
                patch("agent.skill_commands.resolve_skill_command_key", return_value="/plan"),
                patch("agent.skill_commands.build_skill_invocation_message", return_value="expanded plan skill"),
            ):
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {
                    "final_response": "planned",
                    "messages": [],
                }
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "/plan ship the mobile release", "session_id": "session-plan"},
                )
                run_id = (await resp.json())["run_id"]
                for _ in range(20):
                    status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        call = mock_agent.run_conversation.call_args.kwargs
        assert call["user_message"] == "expanded plan skill"
        assert call["persist_user_message"] == "/plan ship the mobile release"

    @pytest.mark.asyncio
    async def test_goal_slash_command_sets_goal_and_runs_the_judged_loop(self, adapter):
        app = _create_runs_app(adapter)
        contract = MagicMock()
        contract.is_empty.return_value = True
        manager = MagicMock()
        manager.set.return_value = MagicMock(goal="ship the mobile release")
        manager.evaluate_after_turn.return_value = {
            "should_continue": False,
            "message": "Goal achieved",
        }
        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(adapter, "_create_agent") as mock_create,
                patch("hermes_cli.goals.GoalManager", return_value=manager),
                patch("hermes_cli.goals.parse_contract", return_value=("ship the mobile release", contract)),
            ):
                mock_agent = MagicMock()
                mock_agent.run_conversation.return_value = {
                    "final_response": "shipped",
                    "messages": [{"role": "assistant", "content": "shipped"}],
                }
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "/goal ship the mobile release", "session_id": "session-goal"},
                )
                run_id = (await resp.json())["run_id"]
                for _ in range(20):
                    status = await (await cli.get(f"/v1/runs/{run_id}")).json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        manager.set.assert_called_once_with("ship the mobile release", contract=None)
        manager.evaluate_after_turn.assert_called_once_with("shipped", user_initiated=True)
        call = mock_agent.run_conversation.call_args.kwargs
        assert call["user_message"] == "ship the mobile release"
        assert call["persist_user_message"] == "/goal ship the mobile release"

    @pytest.mark.asyncio
    async def test_goal_runs_for_one_session_are_serialized(self, adapter):
        app = _create_runs_app(adapter)
        entered = threading.Event()
        release = threading.Event()
        contract = MagicMock()
        contract.is_empty.return_value = True
        manager = MagicMock()
        manager.set.return_value = MagicMock(goal="ship safely")
        manager.evaluate_after_turn.return_value = {"should_continue": False, "message": "done"}

        def run_conversation(**_kwargs):
            entered.set()
            release.wait(timeout=3)
            return {"final_response": "done", "messages": []}

        def make_agent():
            agent = MagicMock()
            agent.run_conversation.side_effect = run_conversation
            agent.session_prompt_tokens = 0
            agent.session_completion_tokens = 0
            agent.session_total_tokens = 0
            return agent

        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(adapter, "_create_agent", side_effect=lambda **_kwargs: make_agent()),
                patch("hermes_cli.goals.GoalManager", return_value=manager) as manager_factory,
                patch("hermes_cli.goals.parse_contract", return_value=("ship safely", contract)),
            ):
                first = await cli.post(
                    "/v1/runs",
                    json={"input": "/goal ship safely", "session_id": "shared-goal"},
                )
                second = await cli.post(
                    "/v1/runs",
                    json={"input": "/goal ship safely", "session_id": "shared-goal"},
                )
                run_ids = [(await first.json())["run_id"], (await second.json())["run_id"]]
                assert await asyncio.to_thread(entered.wait, 2)
                await asyncio.sleep(0.05)
                assert manager_factory.call_count == 1

                release.set()
                for _ in range(40):
                    statuses = [await (await cli.get(f"/v1/runs/{run_id}")).json() for run_id in run_ids]
                    if all(status["status"] == "completed" for status in statuses):
                        break
                    await asyncio.sleep(0.05)

        assert manager_factory.call_count == 2


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
    async def test_events_replay_after_last_event_id_for_reconnect(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_replay"
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = time.time()
        adapter._publish_run_event(run_id, {"event": "tool.started", "tool": "terminal"})
        adapter._publish_run_event(run_id, {"event": "tool.completed", "tool": "terminal"})
        adapter._close_run_event_stream(run_id)

        async with TestClient(TestServer(app)) as cli:
            response = await cli.get(
                f"/v1/runs/{run_id}/events",
                headers={"Last-Event-ID": "1"},
            )
            body = await response.text()

        assert response.status == 200
        assert "id: 1" not in body
        assert "id: 2" in body
        assert "tool.completed" in body
        assert "tool.started" not in body

    @pytest.mark.asyncio
    async def test_completed_event_stream_can_be_read_by_multiple_subscribers(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_multi_subscriber"
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = time.time()
        adapter._publish_run_event(run_id, {"event": "run.completed", "output": "done"})
        adapter._close_run_event_stream(run_id)

        async with TestClient(TestServer(app)) as cli:
            first = await (await cli.get(f"/v1/runs/{run_id}/events")).text()
            second = await (await cli.get(f"/v1/runs/{run_id}/events")).text()

        assert "run.completed" in first
        assert "run.completed" in second

    @pytest.mark.asyncio
    async def test_connected_slow_subscriber_gets_explicit_cursor_expiry(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_slow_subscriber"
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = time.time()
        adapter._publish_run_event(run_id, {"event": "run.started"})

        async with TestClient(TestServer(app)) as cli:
            response = await cli.get(
                f"/v1/runs/{run_id}/events",
                headers={"Last-Event-ID": "1"},
            )
            for index in range(adapter._RUN_EVENT_HISTORY_LIMIT + 1):
                adapter._publish_run_event(run_id, {"event": "message.delta", "delta": str(index)})
            adapter._close_run_event_stream(run_id)
            body = await response.text()

        assert "event: error" in body
        assert "event_cursor_expired" in body
        assert run_id not in adapter._run_stream_subscribers

    @pytest.mark.asyncio
    async def test_prepare_failure_does_not_register_subscriber(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_prepare_failure"
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = time.time()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(web.StreamResponse, "prepare", side_effect=RuntimeError("disconnected")):
                with pytest.raises(Exception):
                    await cli.get(f"/v1/runs/{run_id}/events")

        assert run_id not in adapter._run_stream_subscribers
        assert run_id not in adapter._run_stream_subscriber_counts

    @pytest.mark.asyncio
    async def test_events_forward_subagent_thinking_only_when_enabled(self, adapter, monkeypatch):
        loop = asyncio.get_running_loop()
        enabled_run_id = "run_thinking_enabled"
        disabled_run_id = "run_thinking_disabled"
        adapter._run_streams[enabled_run_id] = asyncio.Queue()
        adapter._run_streams[disabled_run_id] = asyncio.Queue()

        monkeypatch.setattr(adapter, "_thinking_progress_enabled", lambda: True)
        adapter._make_run_event_callback(enabled_run_id, loop)("_thinking", "Checking the delegated task")
        event = await asyncio.wait_for(adapter._run_streams[enabled_run_id].get(), timeout=1)

        assert event["event"] == "reasoning.available"
        assert event["text"] == "Checking the delegated task"

        monkeypatch.setattr(adapter, "_thinking_progress_enabled", lambda: False)
        adapter._make_run_event_callback(disabled_run_id, loop)("_thinking", "Do not forward this")
        await asyncio.sleep(0)
        assert adapter._run_streams[disabled_run_id].empty()

    @pytest.mark.asyncio
    async def test_events_forward_todo_updates(self, adapter):
        loop = asyncio.get_running_loop()
        run_id = "run_tasks"
        adapter._run_streams[run_id] = asyncio.Queue()

        adapter._make_run_event_callback(run_id, loop)(
            "tool.completed",
            "todo",
            result=json.dumps({
                "todos": [
                    {"id": "plan", "content": "Plan the release", "status": "completed"},
                    {"id": "ship", "content": "Ship the release", "status": "in_progress"},
                    {"id": "invalid", "content": "Ignore this", "status": "unknown"},
                ],
            }),
        )

        tool_event = await asyncio.wait_for(adapter._run_streams[run_id].get(), timeout=1)
        task_event = await asyncio.wait_for(adapter._run_streams[run_id].get(), timeout=1)

        assert tool_event["event"] == "tool.completed"
        assert task_event["event"] == "tasks.updated"
        assert task_event["run_id"] == run_id
        assert task_event["tasks"] == [
            {"id": "plan", "content": "Plan the release", "status": "completed"},
            {"id": "ship", "content": "Ship the release", "status": "in_progress"},
        ]

    @pytest.mark.asyncio
    async def test_events_forward_subagent_updates(self, adapter, monkeypatch):
        loop = asyncio.get_running_loop()
        run_id = "run_subagent"
        adapter._run_streams[run_id] = asyncio.Queue()
        monkeypatch.setattr(adapter, "_thinking_progress_enabled", lambda: True)
        callback = adapter._make_run_event_callback(run_id, loop)

        callback(
            "subagent.start",
            preview="Inspect the API",
            subagent_id="subagent-1",
            task_index=0,
            task_count=2,
            goal="Inspect the API",
        )
        callback(
            "subagent.thinking",
            preview="Tracing the event callback",
            subagent_id="subagent-1",
            task_index=0,
            task_count=2,
        )

        start_event = await asyncio.wait_for(adapter._run_streams[run_id].get(), timeout=1)
        thinking_event = await asyncio.wait_for(adapter._run_streams[run_id].get(), timeout=1)

        assert start_event["event"] == "subagent.updated"
        assert start_event["subagent"] == {
            "id": "subagent-1",
            "status": "running",
            "task_index": 0,
            "task_count": 2,
            "tool_count": 0,
            "goal": "Inspect the API",
        }
        assert thinking_event["subagent"]["status"] == "thinking"
        assert thinking_event["subagent"]["activity"] == "Tracing the event callback"

    def test_session_response_marks_recent_unfinished_session_active(self):
        now = time.time()

        active = APIServerAdapter._session_response({
            "id": "desktop-active",
            "started_at": now - 60,
            "last_active": now - 1,
            "ended_at": None,
        })
        stale = APIServerAdapter._session_response({
            "id": "desktop-stale",
            "started_at": now - 600,
            "last_active": now - 301,
            "ended_at": None,
        })
        finished = APIServerAdapter._session_response({
            "id": "desktop-finished",
            "started_at": now - 60,
            "last_active": now - 1,
            "ended_at": now,
        })

        assert active["is_active"] is True
        assert stale["is_active"] is False
        assert finished["is_active"] is False

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
    async def test_events_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/v1/runs/run_any/events")
        assert resp.status == 401


# ---------------------------------------------------------------------------
# Run lifecycle TTL sweeping
# ---------------------------------------------------------------------------


class TestRunLifecycleSweep:
    def test_sweep_keeps_transport_with_active_subscriber(self, adapter):
        run_id = "run_subscribed"
        queue = asyncio.Queue()
        adapter._run_streams[run_id] = queue
        adapter._run_streams_created[run_id] = 0
        adapter._run_stream_subscribers.add(run_id)

        adapter._sweep_orphaned_runs_once(time.time())

        assert adapter._run_streams[run_id] is queue
        assert run_id in adapter._run_streams_created

    @pytest.mark.asyncio
    async def test_expired_live_run_drops_transport_but_keeps_control_state(self, adapter):
        """Stream TTL bounds buffering without detaching a live run."""
        app = _create_runs_app(adapter)
        adapter._max_concurrent_runs = 1

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert start_resp.status == 202
                run_id = (await start_resp.json())["run_id"]
                assert agent_ready.wait(timeout=3.0)

                task = adapter._active_run_tasks[run_id]
                assert isinstance(task, asyncio.Task)
                assert not task.done()

                pending = approval_mod._ApprovalEntry({
                    "command": "bash -c long-running",
                    "description": "approval after stream TTL",
                    "pattern_keys": ["shell-c"],
                })
                with approval_mod._lock:
                    approval_mod._gateway_queues[run_id] = [pending]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                # Exercise one real sweeper iteration without waiting 60 seconds.
                with patch(
                    "gateway.platforms.api_server.asyncio.sleep",
                    side_effect=[None, asyncio.CancelledError()],
                ):
                    with pytest.raises(asyncio.CancelledError):
                        await adapter._sweep_orphaned_runs()

                assert adapter._active_run_tasks[run_id] is task
                assert adapter._active_run_agents[run_id] is mock_agent
                assert run_id not in adapter._run_streams
                assert run_id not in adapter._run_streams_created
                assert adapter._run_approval_sessions[run_id] == run_id

                limited = adapter._concurrency_limited_response()
                assert limited is not None
                assert limited.status == 429

                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )
                assert approval_resp.status == 200
                assert pending.event.is_set()
                assert pending.result == "once"

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                mock_agent.interrupt.assert_called_once_with("Stop requested via API")

    @pytest.mark.asyncio
    async def test_expired_transport_stops_buffering_new_deltas(self, adapter):
        """An unconsumed expired queue must not grow for the rest of a live run."""
        app = _create_runs_app(adapter)

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent, agent_ready, _ = _make_slow_agent()
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]
                assert agent_ready.wait(timeout=3.0)
                expired_queue = adapter._run_streams[run_id]
                stream_delta = mock_create.call_args.kwargs["stream_delta_callback"]

                adapter._run_streams_created[run_id] -= adapter._RUN_STREAM_TTL + 1
                adapter._sweep_orphaned_runs_once(time.time())
                before = expired_queue.qsize()
                stream_delta("must-not-buffer")
                mock_agent.interrupt("finish test")
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert expired_queue.qsize() == before

    @pytest.mark.asyncio
    async def test_expired_orphan_run_state_is_reaped(self, adapter):
        run_id = "run_expired_orphan"
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = 0
        adapter._run_approval_sessions[run_id] = run_id

        pending = approval_mod._ApprovalEntry({
            "command": "bash -c orphaned",
            "description": "orphaned approval",
            "pattern_keys": ["shell-c"],
        })
        with approval_mod._lock:
            approval_mod._gateway_queues[run_id] = [pending]

        with patch(
            "gateway.platforms.api_server.asyncio.sleep",
            side_effect=[None, asyncio.CancelledError()],
        ):
            with pytest.raises(asyncio.CancelledError):
                await adapter._sweep_orphaned_runs()

        assert run_id not in adapter._run_streams
        assert run_id not in adapter._run_streams_created
        assert run_id not in adapter._run_approval_sessions
        assert pending.event.is_set()
        with approval_mod._lock:
            assert run_id not in approval_mod._gateway_queues


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — interrupt a running agent
# ---------------------------------------------------------------------------


class TestStopRun:
    @pytest.mark.asyncio
    async def test_stop_before_agent_creation_prevents_run_start(self, adapter):
        """A stop accepted while queued must prevent agent construction."""
        app = _create_runs_app(adapter)
        original_create_task = asyncio.create_task
        task_started = asyncio.Event()
        allow_task = asyncio.Event()

        def _delayed_create_task(coro):
            async def _delayed():
                task_started.set()
                await allow_task.wait()
                return await coro

            return original_create_task(_delayed())

        with patch("gateway.platforms.api_server.asyncio.create_task", side_effect=_delayed_create_task), \
             patch.object(adapter, "_create_agent") as mock_create:
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                await task_started.wait()

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                allow_task.set()

                for _ in range(20):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                mock_create.assert_not_called()
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_keeps_uncooperative_executor_tracked_until_exit(self, adapter):
        """Cancelling an asyncio wrapper must not hide its live executor thread."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_finished = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                started = threading.Event()

                def _run_conversation(*_args, **_kwargs):
                    started.set()
                    run_can_finish.wait(timeout=5)
                    run_finished.set()
                    return {"final_response": "late result"}

                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await resp.json())["run_id"]
                assert started.wait(timeout=3)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                await asyncio.sleep(0.1)

                assert not run_finished.is_set()
                assert run_id in adapter._active_run_agents
                assert run_id in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "stopping"

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks
                assert adapter._run_statuses[run_id]["status"] == "cancelled"

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
