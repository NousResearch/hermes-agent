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
import subprocess
import sys
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
    close_tool_channel,
    has_attached_client,
    register_tool_notify,
    resolve_tool_result,
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
    def test_run_executor_workers_do_not_block_process_exit(self):
        code = """
import threading
from gateway.platforms.api_server import _DaemonThreadPoolExecutor

started = threading.Event()
never = threading.Event()
executor = _DaemonThreadPoolExecutor(1, "exit-test")
executor.submit(lambda: (started.set(), never.wait()))
assert started.wait(2)
executor.shutdown(wait=False)
"""
        completed = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr

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

    @pytest.mark.asyncio
    async def test_split_runtime_rejects_codex_app_server_before_conversation(self, split_adapter):
        app = _create_runs_app(split_adapter)
        mock_agent = MagicMock()
        mock_agent.api_mode = "codex_app_server"
        async with TestClient(TestServer(app)) as cli:
            with patch.object(split_adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post("/v1/runs", json={"input": "read local file"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]

                status = {}
                for _ in range(50):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "failed":
                        break
                    await asyncio.sleep(0.01)

        assert status["status"] == "failed"
        assert status["code"] == "split_runtime_incompatible_runtime"
        mock_agent.run_conversation.assert_not_called()

    @pytest.mark.asyncio
    async def test_disabled_api_file_toolset_disables_split_runtime_for_run(
        self,
        split_adapter,
        monkeypatch,
    ):
        monkeypatch.setattr(
            "hermes_cli.tools_config._get_platform_tools",
            lambda config, platform: {"safe"},
        )
        app = _create_runs_app(split_adapter)
        mock_agent = MagicMock()
        mock_agent.api_mode = "codex_app_server"
        mock_agent.run_conversation.return_value = {"final_response": "server runtime allowed"}
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        async with TestClient(TestServer(app)) as cli:
            with patch.object(split_adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post("/v1/runs", json={"input": "no local tools"})
                run_id = (await resp.json())["run_id"]
                status = {}
                for _ in range(50):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.01)

                executor_resp = await cli.get(f"/v1/runs/{run_id}/events?tool_executor=1")

        assert status["status"] == "completed"
        assert split_adapter._run_split_runtime_active[run_id] is False
        assert executor_resp.status == 409
        mock_agent.run_conversation.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_uses_one_toolset_snapshot_for_routing_and_agent(self, split_adapter):
        app = _create_runs_app(split_adapter)
        mock_agent = MagicMock()
        mock_agent.api_mode = "chat_completions"
        mock_agent.run_conversation.return_value = {"final_response": "done"}
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        with (
            patch(
                "hermes_cli.tools_config._get_platform_tools",
                return_value={"file"},
            ) as get_tools,
            patch.object(
                split_adapter,
                "_create_agent",
                return_value=mock_agent,
            ) as create_agent,
        ):
            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post("/v1/runs", json={"input": "snapshot tools"})
                assert resp.status == 202
                run_id = (await resp.json())["run_id"]
                for _ in range(50):
                    if split_adapter._run_statuses[run_id]["status"] == "completed":
                        break
                    await asyncio.sleep(0.01)

        assert split_adapter._run_split_runtime_active[run_id] is True
        assert create_agent.call_args.kwargs["enabled_toolsets"] == ["file"]
        get_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_enabled_split_runtime_rejects_run_when_toolset_resolution_fails(
        self,
        split_adapter,
        monkeypatch,
    ):
        def fail_toolset_resolution(*_args, **_kwargs):
            raise RuntimeError("bad config")

        monkeypatch.setattr(
            "hermes_cli.tools_config._get_platform_tools",
            fail_toolset_resolution,
        )
        app = _create_runs_app(split_adapter)

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs", json={"input": "must fail closed"})

        assert resp.status == 500
        assert split_adapter._active_run_tasks == {}

    @pytest.mark.asyncio
    async def test_tool_channel_closes_only_after_terminal_status(self, split_adapter):
        from gateway.tool_channel_state import close_tool_channel

        app = _create_runs_app(split_adapter)
        mock_agent = MagicMock()
        mock_agent.api_mode = "chat_completions"
        mock_agent.run_conversation.return_value = {"final_response": "done"}
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0
        closed_statuses = []

        def observed_close(session_key):
            closed_statuses.append(split_adapter._run_statuses[session_key]["status"])
            close_tool_channel(session_key)

        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(split_adapter, "_create_agent", return_value=mock_agent),
                patch("gateway.tool_channel_state.close_tool_channel", side_effect=observed_close),
            ):
                resp = await cli.post("/v1/runs", json={"input": "finish"})
                run_id = (await resp.json())["run_id"]
                for _ in range(50):
                    if run_id not in split_adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.01)

        assert closed_statuses == ["completed"]

    @pytest.mark.asyncio
    async def test_worker_submission_failure_releases_capacity(self, split_adapter):
        class BrokenExecutor:
            def submit(self, *_args, **_kwargs):
                raise RuntimeError("executor unavailable")

        split_adapter._max_concurrent_runs = 1
        app = _create_runs_app(split_adapter)
        mock_agent = MagicMock()
        mock_agent.api_mode = "chat_completions"

        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(split_adapter, "_create_agent", return_value=mock_agent),
                patch.object(split_adapter, "_get_run_executor", return_value=BrokenExecutor()),
            ):
                resp = await cli.post("/v1/runs", json={"input": "submission fails"})
                run_id = (await resp.json())["run_id"]
                for _ in range(50):
                    status = split_adapter._run_statuses.get(run_id, {})
                    if status.get("status") == "failed":
                        break
                    await asyncio.sleep(0.01)

        assert split_adapter._run_statuses[run_id]["status"] == "failed"
        assert split_adapter._run_worker_done[run_id].is_set()
        assert split_adapter._concurrency_limited_response() is None

    @pytest.mark.asyncio
    async def test_slow_request_body_reserves_capacity_before_await(self, adapter):
        adapter._max_concurrent_runs = 1
        body_started = asyncio.Event()
        release_body = asyncio.Event()
        first_request = MagicMock()
        first_request.headers = {}

        async def slow_json():
            body_started.set()
            await release_body.wait()
            return {}

        first_request.json = slow_json
        first_task = asyncio.create_task(adapter._handle_runs(first_request))
        await body_started.wait()

        second_request = MagicMock()
        second_request.headers = {}
        second_response = await adapter._handle_runs(second_request)

        assert second_response.status == 429
        assert adapter._pending_agent_admissions == 1

        release_body.set()
        first_response = await first_task
        assert first_response.status == 400
        assert adapter._pending_agent_admissions == 0

    @pytest.mark.asyncio
    async def test_draining_server_rejects_body_that_was_already_admitted(self, adapter):
        body_started = asyncio.Event()
        release_body = asyncio.Event()
        request = MagicMock()
        request.headers = {}

        async def slow_json():
            body_started.set()
            await release_body.wait()
            return {"input": "must not start"}

        request.json = slow_json
        request_task = asyncio.create_task(adapter._handle_runs(request))
        await body_started.wait()

        await adapter.disconnect()
        adapter._accepting_agent_requests = True  # simulate a quick reconnect
        release_body.set()
        response = await request_task

        assert response.status == 503
        assert adapter._active_run_tasks == {}
        assert adapter._run_worker_futures == {}


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
    async def test_late_progress_callback_cannot_overwrite_terminal_status(self, adapter):
        run_id = "run_late_progress"
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "cancelled"}
        adapter._run_streams[run_id] = asyncio.Queue()
        callback = adapter._make_run_event_callback(run_id, asyncio.get_running_loop())

        await asyncio.to_thread(callback, "tool.started", tool_name="read_file")
        await asyncio.sleep(0)

        assert adapter._run_statuses[run_id]["status"] == "cancelled"
        assert adapter._run_streams[run_id].empty()

    @pytest.mark.asyncio
    async def test_late_progress_callback_cannot_resurrect_swept_run(self, adapter):
        run_id = "run_already_swept"
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "completed"}
        adapter._run_streams[run_id] = asyncio.Queue()
        callback = adapter._make_run_event_callback(run_id, asyncio.get_running_loop())

        adapter._run_statuses.pop(run_id)
        adapter._run_streams.pop(run_id)
        await asyncio.to_thread(callback, "tool.completed", tool_name="read_file")
        await asyncio.sleep(0)

        assert run_id not in adapter._run_statuses
        assert run_id not in adapter._run_streams

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
    async def test_stopping_run_rejects_approval_without_resolving(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_stopping_approval"
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "stopping"}
        adapter._run_approval_sessions[run_id] = run_id

        async with TestClient(TestServer(app)) as cli:
            with patch("tools.approval.resolve_gateway_approval") as mock_resolve:
                approval_resp = await cli.post(
                    f"/v1/runs/{run_id}/approval",
                    json={"choice": "once"},
                )

        assert approval_resp.status == 409
        mock_resolve.assert_not_called()
        assert adapter._run_statuses[run_id]["status"] == "stopping"

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
    async def test_events_rejects_second_consumer_in_split_mode(self, split_adapter):
        app = _create_runs_app(split_adapter)
        run_id = "run_events_conflict"
        split_adapter._run_streams[run_id] = asyncio.Queue()
        split_adapter._run_streams_created[run_id] = 1.0
        split_adapter._run_event_consumers[run_id] = ("events", "lease-a")
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(f"/v1/runs/{run_id}/events")
            data = await resp.json()
        assert resp.status == 409
        assert data["error"]["code"] == "run_events_consumer_conflict"
        split_adapter._run_streams.pop(run_id, None)
        split_adapter._run_streams_created.pop(run_id, None)
        split_adapter._run_event_consumers.pop(run_id, None)

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
    async def test_terminal_run_allows_late_executor_stream_to_drain_events(self, split_adapter):
        app = _create_runs_app(split_adapter)
        run_id = "run_finished_before_executor_attach"
        q = asyncio.Queue()
        q.put_nowait({"event": "run.completed", "run_id": run_id})
        q.put_nowait(None)
        split_adapter._run_streams[run_id] = q
        split_adapter._run_streams_created[run_id] = 1.0
        split_adapter._run_statuses[run_id] = {"run_id": run_id, "status": "completed"}

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(f"/v1/runs/{run_id}/events?tool_executor=1")
            assert resp.status == 200
            body = await resp.text()

        assert "run.completed" in body

    @pytest.mark.asyncio
    async def test_executor_stream_discards_canceled_tool_request_events(self, split_adapter):
        app = _create_runs_app(split_adapter)
        run_id = "run_stale_request_event"
        q = asyncio.Queue()
        q.put_nowait({
            "event": "tool.request",
            "run_id": run_id,
            "request_id": "toolreq_canceled",
            "tool_call_id": "call_canceled",
        })
        q.put_nowait({"event": "run.completed", "run_id": run_id})
        q.put_nowait(None)
        split_adapter._run_streams[run_id] = q
        split_adapter._run_streams_created[run_id] = 1.0
        split_adapter._run_statuses[run_id] = {"run_id": run_id, "status": "completed"}

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get(f"/v1/runs/{run_id}/events?tool_executor=1")
            assert resp.status == 200
            body = await resp.text()

        assert "toolreq_canceled" not in body
        assert "run.completed" in body

    @pytest.mark.asyncio
    async def test_executor_prepare_failure_releases_attachment_and_lease(self, split_adapter):
        run_id = "run_prepare_failure"
        session_key = "prepare-failure-session"
        split_adapter._run_streams[run_id] = asyncio.Queue()
        split_adapter._run_streams_created[run_id] = 1.0
        split_adapter._run_tool_sessions[run_id] = session_key
        split_adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        request = MagicMock()
        request.match_info = {"run_id": run_id}
        request.query = {"tool_executor": "1"}
        request.headers = {"X-Hermes-Tool-Executor-Token": "prepare-client"}

        async def fail_prepare(_response, _request):
            raise ConnectionResetError("client disconnected during prepare")

        with patch.object(web.StreamResponse, "prepare", fail_prepare):
            await split_adapter._handle_run_events(request)

        assert has_attached_client(session_key) is False
        assert run_id not in split_adapter._run_event_consumers
        assert run_id in split_adapter._run_streams

    @pytest.mark.asyncio
    async def test_events_tool_executor_round_trips_routed_read_file(self, split_adapter):
        """A fake local executor can answer a routed read_file over /v1/runs SSE."""
        split_adapter._cors_origins = ["https://executor.local"]
        app = _create_runs_app(split_adapter)

        class FakeAgent:
            session_prompt_tokens = 0
            session_completion_tokens = 0
            session_total_tokens = 0

            def run_conversation(self, user_message, conversation_history, task_id):
                import model_tools

                result = model_tools.handle_function_call(
                    "read_file",
                    {"path": "README.md"},
                    task_id=task_id,
                    tool_call_id="call_http_e2e",
                    session_id="session-http-e2e",
                )
                return {"final_response": f"local executor returned: {result}"}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(split_adapter, "_create_agent", return_value=FakeAgent()):
                start_resp = await cli.post("/v1/runs", json={"input": "read local file"})
                assert start_resp.status == 202
                run_id = (await start_resp.json())["run_id"]

                events_resp = await cli.get(
                    f"/v1/runs/{run_id}/events?tool_executor=1",
                    headers={
                        "X-Hermes-Tool-Executor-Token": "executor-token",
                        "Origin": "https://executor.local",
                    },
                )
                assert events_resp.status == 200
                assert events_resp.headers["Access-Control-Allow-Origin"] == "https://executor.local"

                request_event = None
                for _ in range(10):
                    raw = await asyncio.wait_for(events_resp.content.readline(), timeout=10.0)
                    line = raw.decode().strip()
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        if event.get("event") == "tool.request":
                            request_event = event
                            break
                assert request_event is not None
                assert request_event["request_id"].startswith("toolreq_")
                assert request_event["tool_call_id"] == "call_http_e2e"
                assert request_event["tool_name"] == "read_file"
                assert request_event["arguments"] == {"path": "README.md"}

                result_resp = await cli.post(
                    f"/v1/runs/{run_id}/tool_result",
                    json={"request_id": request_event["request_id"], "result": "LOCAL README CONTENT"},
                    headers={"X-Hermes-Tool-Executor-Token": "executor-token"},
                )
                assert result_resp.status == 200

                body = await asyncio.wait_for(events_resp.text(), timeout=10.0)
                assert "tool.result" in body
                assert "run.completed" in body
                assert "LOCAL README CONTENT" in body

    @pytest.mark.asyncio
    async def test_tool_timeout_restores_pollable_running_status(self, split_adapter):
        split_adapter._split_runtime_request_timeout = 0.05
        app = _create_runs_app(split_adapter)
        tool_returned = threading.Event()
        release_agent = threading.Event()

        class FakeAgent:
            api_mode = "chat_completions"
            session_prompt_tokens = 0
            session_completion_tokens = 0
            session_total_tokens = 0

            def run_conversation(self, user_message, conversation_history, task_id):
                import model_tools

                result = model_tools.handle_function_call(
                    "read_file",
                    {"path": "README.md"},
                    task_id=task_id,
                    tool_call_id="call_timeout_status",
                )
                tool_returned.set()
                release_agent.wait(timeout=1.0)
                return {"final_response": result}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(split_adapter, "_create_agent", return_value=FakeAgent()):
                start_resp = await cli.post("/v1/runs", json={"input": "timeout locally"})
                run_id = (await start_resp.json())["run_id"]
                events_resp = await cli.get(f"/v1/runs/{run_id}/events?tool_executor=1")
                assert events_resp.status == 200

                for _ in range(10):
                    raw = await asyncio.wait_for(events_resp.content.readline(), timeout=1.0)
                    if b'"event": "tool.request"' in raw:
                        break
                assert await asyncio.to_thread(tool_returned.wait, 1.0)

                status_resp = await cli.get(f"/v1/runs/{run_id}")
                status = await status_resp.json()
                assert status["status"] == "running"
                assert status["last_event"] == "tool.timeout"

                release_agent.set()
                events_resp.close()

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
    @pytest.mark.parametrize("body", [None, [], "result", 42])
    async def test_tool_result_rejects_non_object_json(self, split_adapter, body):
        app = _create_runs_app(split_adapter)
        run_id = "run_tool_invalid_shape"
        split_adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        split_adapter._run_tool_sessions[run_id] = "invalid-shape-session"
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/v1/runs/{run_id}/tool_result",
                data=json.dumps(body),
                headers={"Content-Type": "application/json"},
            )
            data = await resp.json()
        assert resp.status == 400
        assert data["error"]["code"] == "invalid_tool_result"

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
                    json={"request_id": entry.request["request_id"], "result": "local output"},
                )
                assert resp.status == 200
                data = await resp.json()

            assert data == {
                "object": "hermes.run.tool_result_response",
                "run_id": run_id,
                "request_id": entry.request["request_id"],
                "status": "resolved",
            }
            assert entry.event.is_set()
            assert entry.result == "local output"
            assert split_adapter._run_statuses[run_id]["status"] == "running"
            event = await asyncio.wait_for(q.get(), timeout=1.0)
            assert event["event"] == "tool.result"
            assert event["request_id"] == entry.request["request_id"]
        finally:
            unregister_tool_notify(session_key)
            clear_tool_channel_state(session_key)

    @pytest.mark.asyncio
    async def test_tool_result_keeps_waiting_status_until_parallel_requests_finish(self, split_adapter):
        app = _create_runs_app(split_adapter)
        run_id = "run_parallel_results"
        session_key = "parallel-tool-session"
        split_adapter._run_statuses[run_id] = {
            "run_id": run_id,
            "status": "waiting_for_tool_result",
        }
        split_adapter._run_tool_sessions[run_id] = session_key
        try:
            assert register_tool_notify(session_key, lambda request: None, "") is True
            first = submit_tool_request(session_key, {"tool_call_id": "call_parallel_1"})
            second = submit_tool_request(session_key, {"tool_call_id": "call_parallel_2"})
            assert first is not None and second is not None

            async with TestClient(TestServer(app)) as cli:
                first_resp = await cli.post(
                    f"/v1/runs/{run_id}/tool_result",
                    json={"request_id": first.request["request_id"], "result": "one"},
                )
                assert first_resp.status == 200
                assert split_adapter._run_statuses[run_id]["status"] == "waiting_for_tool_result"

                second_resp = await cli.post(
                    f"/v1/runs/{run_id}/tool_result",
                    json={"request_id": second.request["request_id"], "result": "two"},
                )
                assert second_resp.status == 200
                assert split_adapter._run_statuses[run_id]["status"] == "running"
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
                    json={"request_id": entry.request["request_id"], "result": "wrong"},
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
                    json={"request_id": entry.request["request_id"], "result": "right"},
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
                    json={"request_id": entry.request["request_id"], "result": "one"},
                )
                second = await cli.post(
                    f"/v1/runs/{run_id}/tool_result",
                    json={"request_id": entry.request["request_id"], "result": "two"},
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
    async def test_duplicate_tool_result_is_idempotent_after_run_completion(self, split_adapter):
        app = _create_runs_app(split_adapter)
        run_id = "run_tool_duplicate_terminal"
        split_adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        split_adapter._run_tool_sessions[run_id] = run_id
        try:
            assert register_tool_notify(run_id, lambda request: None, "") is True
            entry = submit_tool_request(run_id, {"tool_call_id": "call_terminal_dup"})
            assert entry is not None
            assert resolve_tool_result(run_id, entry.request["request_id"], "one") == "resolved"
            close_tool_channel(run_id)
            split_adapter._run_tool_sessions.pop(run_id, None)
            split_adapter._run_statuses[run_id]["status"] = "completed"

            async with TestClient(TestServer(app)) as cli:
                retry = await cli.post(
                    f"/v1/runs/{run_id}/tool_result",
                    json={"request_id": entry.request["request_id"], "result": "two"},
                )
                retry_data = await retry.json()

            assert retry.status == 200
            assert retry_data["status"] == "duplicate"
        finally:
            clear_tool_channel_state(run_id)

    @pytest.mark.asyncio
    async def test_tool_result_without_pending_request_returns_409(self, split_adapter):
        app = _create_runs_app(split_adapter)
        run_id = "run_tool_missing"
        split_adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        split_adapter._run_tool_sessions[run_id] = "missing-session"
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/v1/runs/{run_id}/tool_result",
                json={"request_id": "toolreq_missing", "result": "late"},
            )
            data = await resp.json()
        assert resp.status == 409
        assert data["error"]["code"] == "tool_result_not_pending"

    @pytest.mark.asyncio
    async def test_tool_result_does_not_overwrite_stopping_status(self, split_adapter):
        app = _create_runs_app(split_adapter)
        run_id = "run_tool_stopping"
        split_adapter._run_statuses[run_id] = {"run_id": run_id, "status": "stopping"}
        split_adapter._run_tool_sessions[run_id] = run_id
        try:
            assert register_tool_notify(run_id, lambda request: None, "") is True
            entry = submit_tool_request(run_id, {"tool_call_id": "call_stopping"})
            assert entry is not None

            async with TestClient(TestServer(app)) as cli:
                resp = await cli.post(
                    f"/v1/runs/{run_id}/tool_result",
                    json={"request_id": entry.request["request_id"], "result": "done"},
                )

            assert resp.status == 200
            assert split_adapter._run_statuses[run_id]["status"] == "stopping"
        finally:
            clear_tool_channel_state(run_id)


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
    async def test_stop_wakes_split_tool_waiting_for_executor_attachment(self, split_adapter):
        app = _create_runs_app(split_adapter)
        agent_started = threading.Event()
        mock_agent = MagicMock()
        mock_agent.api_mode = "chat_completions"
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        def run_conversation(user_message, conversation_history, task_id):
            import model_tools

            agent_started.set()
            result = model_tools.handle_function_call(
                "read_file",
                {"path": "README.md"},
                task_id=task_id,
                tool_call_id="call_stop_before_attach",
            )
            return {"final_response": result}

        mock_agent.run_conversation.side_effect = run_conversation

        async with TestClient(TestServer(app)) as cli:
            with patch.object(split_adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post("/v1/runs", json={"input": "read before attach"})
                run_id = (await resp.json())["run_id"]
                assert agent_started.wait(timeout=1.0)

                started = asyncio.get_running_loop().time()
                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                for _ in range(50):
                    if run_id not in split_adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.01)

                assert run_id not in split_adapter._active_run_tasks
                assert asyncio.get_running_loop().time() - started < 1.0

    @pytest.mark.asyncio
    async def test_stop_keeps_capacity_while_uninterruptible_worker_is_running(self):
        adapter = _make_adapter()
        adapter._max_concurrent_runs = 1
        app = _create_runs_app(adapter)
        ready = threading.Event()
        release = threading.Event()
        mock_agent = MagicMock()
        mock_agent.api_mode = "chat_completions"
        mock_agent.interrupt = MagicMock()
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        def blocked_run(**kwargs):
            ready.set()
            release.wait(timeout=10)
            return {"final_response": "released"}

        mock_agent.run_conversation.side_effect = blocked_run
        run_id = ""
        worker_done = None
        try:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_create_agent", return_value=mock_agent):
                    start_resp = await cli.post("/v1/runs", json={"input": "block"})
                    run_id = (await start_resp.json())["run_id"]
                    assert await asyncio.to_thread(ready.wait, 1.0)

                    stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                    assert stop_resp.status == 200

                    second_resp = await cli.post("/v1/runs", json={"input": "second"})
                    assert second_resp.status == 429
        finally:
            release.set()

        for _ in range(100):
            worker_done = adapter._run_worker_done.get(run_id)
            if worker_done is None or worker_done.is_set():
                break
            await asyncio.sleep(0.01)
        assert worker_done is None or worker_done.is_set()

    @pytest.mark.asyncio
    async def test_disconnect_interrupts_and_drains_running_worker(self):
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        started = threading.Event()
        release = threading.Event()
        mock_agent = MagicMock()
        mock_agent.api_mode = "chat_completions"
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        def blocked_run(**_kwargs):
            started.set()
            assert release.wait(timeout=10.0)
            return {"final_response": "drained"}

        def interrupt(_reason):
            release.set()

        mock_agent.run_conversation.side_effect = blocked_run
        mock_agent.interrupt.side_effect = interrupt

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post("/v1/runs", json={"input": "block"})
                run_id = (await resp.json())["run_id"]
                assert await asyncio.to_thread(started.wait, 3.0)

                await adapter.disconnect()

        mock_agent.interrupt.assert_called_once_with("API server disconnecting")
        assert adapter._run_worker_done[run_id].is_set()
        assert adapter._run_executor is None

    @pytest.mark.asyncio
    async def test_disconnect_can_interrupt_worker_after_wrapper_was_cancelled(self):
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        started = threading.Event()
        release = threading.Event()
        mock_agent = MagicMock()
        mock_agent.api_mode = "chat_completions"
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        def blocked_run(**_kwargs):
            started.set()
            assert release.wait(timeout=10.0)
            return {"final_response": "drained"}

        mock_agent.run_conversation.side_effect = blocked_run
        mock_agent.interrupt.side_effect = lambda _reason: release.set()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", return_value=mock_agent):
                resp = await cli.post("/v1/runs", json={"input": "block"})
                run_id = (await resp.json())["run_id"]
                assert await asyncio.to_thread(started.wait, 3.0)

                adapter._active_run_tasks[run_id].cancel()
                await asyncio.sleep(0.05)
                assert run_id in adapter._active_run_agents
                assert not adapter._run_worker_done[run_id].is_set()

                await adapter.disconnect()

        mock_agent.interrupt.assert_called_once_with("API server disconnecting")
        assert adapter._run_worker_done[run_id].is_set()

    @pytest.mark.asyncio
    async def test_disconnect_stubborn_worker_leaves_time_for_gateway_cleanup(self):
        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        started = threading.Event()
        release = threading.Event()
        mock_agent = MagicMock()
        mock_agent.api_mode = "chat_completions"
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        def blocked_run(**_kwargs):
            started.set()
            assert release.wait(timeout=10.0)
            return {"final_response": "eventually done"}

        mock_agent.run_conversation.side_effect = blocked_run

        try:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_create_agent", return_value=mock_agent):
                    resp = await cli.post("/v1/runs", json={"input": "block"})
                    run_id = (await resp.json())["run_id"]
                    assert await asyncio.to_thread(started.wait, 3.0)

                    before = asyncio.get_running_loop().time()
                    await asyncio.wait_for(adapter.disconnect(), timeout=4.0)
                    elapsed = asyncio.get_running_loop().time() - before

                    assert elapsed < 2.0
                    assert adapter._run_executor is None
                    assert adapter._accepting_agent_requests is False
        finally:
            release.set()

        assert await asyncio.to_thread(adapter._run_worker_done[run_id].wait, 3.0)

    @pytest.mark.asyncio
    async def test_disconnect_cancels_queued_run_before_worker_starts(self):
        from concurrent.futures import ThreadPoolExecutor

        adapter = _make_adapter()
        adapter._max_concurrent_runs = 2
        adapter._run_executor = ThreadPoolExecutor(max_workers=1)
        app = _create_runs_app(adapter)
        first_started = threading.Event()
        first_release = threading.Event()

        first_agent = MagicMock()
        first_agent.api_mode = "chat_completions"
        first_agent.session_prompt_tokens = 0
        first_agent.session_completion_tokens = 0
        first_agent.session_total_tokens = 0

        def first_run(**_kwargs):
            first_started.set()
            assert first_release.wait(timeout=10.0)
            return {"final_response": "first drained"}

        first_agent.run_conversation.side_effect = first_run
        first_agent.interrupt.side_effect = lambda _reason: first_release.set()

        queued_agent = MagicMock()
        queued_agent.api_mode = "chat_completions"
        queued_agent.session_prompt_tokens = 0
        queued_agent.session_completion_tokens = 0
        queued_agent.session_total_tokens = 0

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_create_agent",
                side_effect=[first_agent, queued_agent],
            ):
                first_resp = await cli.post("/v1/runs", json={"input": "first"})
                first_run_id = (await first_resp.json())["run_id"]
                assert await asyncio.to_thread(first_started.wait, 3.0)

                queued_resp = await cli.post("/v1/runs", json={"input": "queued"})
                queued_run_id = (await queued_resp.json())["run_id"]
                for _ in range(50):
                    if queued_run_id in adapter._run_worker_futures:
                        break
                    await asyncio.sleep(0.01)

                await adapter.disconnect()

        assert adapter._run_worker_done[first_run_id].is_set()
        assert adapter._run_worker_done[queued_run_id].is_set()
        queued_agent.run_conversation.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_cancels_queued_run_before_worker_starts(self):
        from concurrent.futures import ThreadPoolExecutor

        adapter = _make_adapter()
        adapter._max_concurrent_runs = 2
        adapter._run_executor = ThreadPoolExecutor(max_workers=1)
        app = _create_runs_app(adapter)
        first_started = threading.Event()
        first_release = threading.Event()

        first_agent = MagicMock()
        first_agent.api_mode = "chat_completions"
        first_agent.session_prompt_tokens = 0
        first_agent.session_completion_tokens = 0
        first_agent.session_total_tokens = 0

        def first_run(**_kwargs):
            first_started.set()
            assert first_release.wait(timeout=10.0)
            return {"final_response": "first done"}

        first_agent.run_conversation.side_effect = first_run
        first_agent.interrupt.side_effect = lambda _reason: first_release.set()

        queued_agent = MagicMock()
        queued_agent.api_mode = "chat_completions"
        queued_agent.session_prompt_tokens = 0
        queued_agent.session_completion_tokens = 0
        queued_agent.session_total_tokens = 0

        async with TestClient(TestServer(app)) as cli:
            with patch.object(
                adapter,
                "_create_agent",
                side_effect=[first_agent, queued_agent],
            ):
                first_resp = await cli.post("/v1/runs", json={"input": "first"})
                assert await asyncio.to_thread(first_started.wait, 3.0)

                queued_resp = await cli.post("/v1/runs", json={"input": "queued"})
                queued_run_id = (await queued_resp.json())["run_id"]
                for _ in range(50):
                    if queued_run_id in adapter._run_worker_futures:
                        break
                    await asyncio.sleep(0.01)

                stop_resp = await cli.post(f"/v1/runs/{queued_run_id}/stop")
                assert stop_resp.status == 200
                assert adapter._run_worker_done[queued_run_id].is_set()
                queued_agent.run_conversation.assert_not_called()

                await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_late_approval_callback_cannot_overwrite_cancelled_status(self):
        from tools.approval import get_current_session_key

        adapter = _make_adapter()
        app = _create_runs_app(adapter)
        redaction_started = threading.Event()
        release_redaction = threading.Event()
        mock_agent = MagicMock()
        mock_agent.api_mode = "chat_completions"
        mock_agent.session_prompt_tokens = 0
        mock_agent.session_completion_tokens = 0
        mock_agent.session_total_tokens = 0

        def run_conversation(**_kwargs):
            session_key = get_current_session_key(default="")
            with approval_mod._lock:
                callback = approval_mod._gateway_notify_cbs[session_key]
            callback({"command": "echo harmless"})
            return {"final_response": "done"}

        def blocked_redaction(command):
            redaction_started.set()
            assert release_redaction.wait(timeout=10.0)
            return command

        mock_agent.run_conversation.side_effect = run_conversation

        async with TestClient(TestServer(app)) as cli:
            with (
                patch.object(adapter, "_create_agent", return_value=mock_agent),
                patch("gateway.run._redact_approval_command", side_effect=blocked_redaction),
            ):
                resp = await cli.post("/v1/runs", json={"input": "approval race"})
                run_id = (await resp.json())["run_id"]
                assert await asyncio.to_thread(redaction_started.wait, 3.0)

                stop_task = asyncio.create_task(cli.post(f"/v1/runs/{run_id}/stop"))
                for _ in range(50):
                    if adapter._run_statuses[run_id]["status"] in {"stopping", "cancelled"}:
                        break
                    await asyncio.sleep(0.01)
                release_redaction.set()
                stop_resp = await stop_task

        assert stop_resp.status == 200
        assert adapter._run_statuses[run_id]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_stop_nonexistent_run_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_nonexistent/stop")
        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_repeated_stop_cannot_reopen_cancelled_run(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_already_cancelled"
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "cancelled"}
        adapter._active_run_agents[run_id] = MagicMock()

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(f"/v1/runs/{run_id}/stop")

        assert resp.status == 409
        assert adapter._run_statuses[run_id]["status"] == "cancelled"
        adapter._active_run_agents[run_id].interrupt.assert_not_called()

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


class TestRunSweep:
    @pytest.mark.asyncio
    async def test_status_sweep_keeps_terminal_fence_until_worker_finishes(self, adapter):
        run_id = "run_terminal_worker_alive"
        worker_done = threading.Event()
        adapter._run_statuses[run_id] = {
            "run_id": run_id,
            "status": "cancelled",
            "updated_at": 0.0,
        }
        adapter._run_split_runtime_active[run_id] = True
        adapter._run_worker_done[run_id] = worker_done
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = 0.0
        close_tool_channel(run_id)

        sleep_calls = 0

        async def one_sweep(_seconds):
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls > 1:
                raise asyncio.CancelledError

        with patch("gateway.platforms.api_server.asyncio.sleep", one_sweep), patch(
            "gateway.platforms.api_server.time.time",
            return_value=adapter._RUN_STATUS_TTL + 1,
        ):
            with pytest.raises(asyncio.CancelledError):
                await adapter._sweep_orphaned_runs()

        assert adapter._run_statuses[run_id]["status"] == "cancelled"
        assert adapter._run_split_runtime_active[run_id] is True
        assert adapter._run_worker_done[run_id] is worker_done
        assert run_id in adapter._run_streams
        assert register_tool_notify(run_id, lambda request: None, "") is False

    @pytest.mark.asyncio
    async def test_sweep_does_not_destroy_active_run_after_stream_ttl(self, adapter):
        run_id = "run_active_past_ttl"
        task = MagicMock()
        task.done.return_value = False
        agent = MagicMock()
        adapter._run_streams[run_id] = asyncio.Queue()
        adapter._run_streams_created[run_id] = 0.0
        adapter._active_run_tasks[run_id] = task
        adapter._active_run_agents[run_id] = agent
        adapter._run_approval_sessions[run_id] = run_id
        adapter._run_tool_sessions[run_id] = run_id

        sleep_calls = 0

        async def one_sweep(_seconds):
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls > 1:
                raise asyncio.CancelledError

        with patch("gateway.platforms.api_server.asyncio.sleep", one_sweep), patch(
            "gateway.platforms.api_server.time.time",
            return_value=adapter._RUN_STREAM_TTL + 1,
        ):
            with pytest.raises(asyncio.CancelledError):
                await adapter._sweep_orphaned_runs()

        assert adapter._run_streams[run_id] is not None
        assert adapter._active_run_tasks[run_id] is task
        assert adapter._active_run_agents[run_id] is agent
        assert adapter._run_tool_sessions[run_id] == run_id
