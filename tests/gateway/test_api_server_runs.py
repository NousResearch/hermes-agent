"""Tests for /v1/runs endpoints: start, status, events, steer, and stop.

Covers:
- POST /v1/runs — start a run (202)
- GET /v1/runs/{run_id} — poll run status
- GET /v1/runs/{run_id}/events — SSE event stream
- POST /v1/runs/{run_id}/steer — inject guidance into a running agent
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
from tools import clarify_gateway as clarify_mod


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
    app.router.add_post(
        "/v1/runs/{run_id}/clarification",
        adapter._handle_run_clarification,
    )
    app.router.add_post("/v1/runs/{run_id}/steer", adapter._handle_steer_run)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


def _create_multiplex_runs_app(adapter: APIServerAdapter) -> web.Application:
    """Runs + clarification routes under /p/{profile}/ with profile middleware."""
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app["api_server_adapter"] = adapter
    app.router.add_post("/p/{profile}/v1/runs", adapter._handle_runs)
    app.router.add_post(
        "/p/{profile}/v1/runs/{run_id}/clarification",
        adapter._handle_run_clarification,
    )
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
    async def test_start_binds_chat_id_for_delegation_wake_target(self, adapter):
        """/v1/runs must bind the raw session id as the api_server chat_id
        (like every other agent-entry route does via _run_agent): the async
        delegation dispatch reads HERMES_SESSION_CHAT_ID to pick its wake
        self-post target, and an empty binding forces background delegations
        on this route back to synchronous execution."""
        app = _create_runs_app(adapter)
        captured = {}

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()

                def _capture_run(user_message=None, conversation_history=None, task_id=None):
                    from tools.async_delegation import _current_origin_session_id

                    captured["origin_session_id"] = _current_origin_session_id()
                    return {"final_response": "done"}

                mock_agent.run_conversation.side_effect = _capture_run
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_create.return_value = mock_agent

                resp = await cli.post(
                    "/v1/runs",
                    json={"input": "hello", "session_id": "runs-raw-sid"},
                )
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert captured.get("origin_session_id") == "runs-raw-sid", (
            "runs route must bind chat_id so delegation dispatch sees a wake target"
        )


    @pytest.mark.asyncio
    async def test_start_rejects_conflicting_route_and_request_provider(self):
        adapter = APIServerAdapter(
            PlatformConfig(
                enabled=True,
                extra={
                    "model_routes": {
                        "alias": {
                            "model": "route/model",
                            "provider": "openrouter",
                        }
                    }
                },
            )
        )
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                resp = await cli.post(
                    "/v1/runs",
                    json={
                        "input": "hello",
                        "model": "alias",
                        "provider": "minimax",
                    },
                )
                data = await resp.json()

        assert resp.status == 400
        assert "provider" in data["error"]["message"].lower()
        assert adapter._run_streams == {}
        assert adapter._run_statuses == {}
        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_passes_request_model_provider_options_to_create_agent(self, adapter):
        app = _create_runs_app(adapter)
        model_options = {"reasoning_effort": "medium", "service_tier": "priority"}
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
                    json={
                        "input": "hello",
                        "model": "MiniMax-M3",
                        "provider": "minimax",
                        "model_options": model_options,
                    },
                )
                assert resp.status == 202
                for _ in range(20):
                    if mock_create.call_args is not None:
                        break
                    await asyncio.sleep(0.05)

        kwargs = mock_create.call_args.kwargs
        assert kwargs["requested_model"] == "MiniMax-M3"
        assert kwargs["requested_provider"] == "minimax"
        assert kwargs["model_options"] == model_options


# ---------------------------------------------------------------------------
# GET /v1/runs/{run_id} — poll run status
# ---------------------------------------------------------------------------


class TestRunStatus:

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
    async def test_clarification_event_waits_for_exact_choice_response(self, adapter):
        app = _create_runs_app(adapter)
        callback_result = {}

        def make_agent(**kwargs):
            mock_agent = MagicMock()

            def run_conversation(**_run_kwargs):
                callback_result["answer"] = kwargs["clarify_callback"](
                    "Pick a color OPENAI_API_KEY=sk-secret-12345678901234567890",
                    ["Red", "Blue"],
                )
                return {"final_response": callback_result["answer"]}

            mock_agent.run_conversation.side_effect = run_conversation
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=make_agent) as create:
                started = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await started.json())["run_id"]
                event = await asyncio.wait_for(adapter._run_streams[run_id].get(), timeout=3)

                assert event["event"] == "clarify.request"
                assert event["run_id"] == run_id
                assert event["request_id"].startswith("clarify_")
                assert event["prompt"] == {
                    "version": 1,
                    "type": "choice",
                    "question": "Pick a color OPENAI_API_KEY=***",
                    "choices": [
                        {"id": "choice-1", "label": "Red"},
                        {"id": "choice-2", "label": "Blue"},
                    ],
                    "multi_select": False,
                }
                assert create.call_args.kwargs["enable_clarify"] is True

                for _ in range(40):
                    polled = adapter._run_statuses[run_id]
                    if polled.get("status") == "waiting_for_clarification":
                        break
                    await asyncio.sleep(0.05)
                assert polled["status"] == "waiting_for_clarification"
                assert polled["awaiting_user"] is True
                assert adapter._session_awaiting_user.get(polled["session_id"]) is True

                response = await cli.post(
                    f"/v1/runs/{run_id}/clarification",
                    json={
                        "request_id": event["request_id"],
                        "response": {"type": "choice", "choice_id": "choice-2"},
                    },
                )
                assert response.status == 200
                payload = await response.json()
                assert payload["request_id"] == event["request_id"]
                assert payload["choice_id"] == "choice-2"
                assert adapter._run_statuses[run_id]["awaiting_user"] is False
                assert polled["session_id"] not in adapter._session_awaiting_user

                for _ in range(40):
                    status = adapter._run_statuses[run_id]
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                assert callback_result["answer"] == "Blue"
                assert status["status"] == "completed"
                assert status["awaiting_user"] is False
                assert run_id not in adapter._run_clarify_sessions

    @pytest.mark.asyncio
    async def test_clarification_multi_select_returns_json_array(self, adapter):
        app = _create_runs_app(adapter)
        callback_result = {}

        def make_agent(**kwargs):
            mock_agent = MagicMock()

            def run_conversation(**_run_kwargs):
                callback_result["answer"] = kwargs["clarify_callback"](
                    "Pick environments",
                    ["Staging", "Production", "Canary"],
                    multi_select=True,
                )
                return {"final_response": callback_result["answer"]}

            mock_agent.run_conversation.side_effect = run_conversation
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=make_agent):
                started = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await started.json())["run_id"]
                event = await asyncio.wait_for(adapter._run_streams[run_id].get(), timeout=3)

                assert event["event"] == "clarify.request"
                assert event["prompt"]["multi_select"] is True
                assert event["prompt"]["type"] == "choice"

                wrong_shape = await cli.post(
                    f"/v1/runs/{run_id}/clarification",
                    json={
                        "request_id": event["request_id"],
                        "response": {"type": "choice", "choice_id": "choice-1"},
                    },
                )
                assert wrong_shape.status == 400

                response = await cli.post(
                    f"/v1/runs/{run_id}/clarification",
                    json={
                        "request_id": event["request_id"],
                        "response": {
                            "type": "choices",
                            "choice_ids": ["choice-1", "choice-3"],
                        },
                    },
                )
                assert response.status == 200
                payload = await response.json()
                assert payload["type"] == "choices"
                assert payload["choice_ids"] == ["choice-1", "choice-3"]

                for _ in range(40):
                    status = adapter._run_statuses[run_id]
                    if status["status"] == "completed":
                        break
                    await asyncio.sleep(0.05)

                assert json.loads(callback_result["answer"]) == ["Staging", "Canary"]
                assert status["status"] == "completed"

    @pytest.mark.asyncio
    async def test_clarification_multi_select_rejects_bad_choice_ids(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_multi"
        request_id = "clarify_" + "d" * 32
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "waiting_for_clarification"}
        adapter._run_clarify_sessions[run_id] = run_id
        clarify_mod.register(
            request_id,
            run_id,
            "Pick many?",
            ["One", "Two", "Three"],
            multi_select=True,
        )

        async with TestClient(TestServer(app)) as cli:
            duplicate = await cli.post(
                f"/v1/runs/{run_id}/clarification",
                json={
                    "request_id": request_id,
                    "response": {
                        "type": "choices",
                        "choice_ids": ["choice-1", "choice-1"],
                    },
                },
            )
            assert duplicate.status == 400

            empty = await cli.post(
                f"/v1/runs/{run_id}/clarification",
                json={
                    "request_id": request_id,
                    "response": {"type": "choices", "choice_ids": []},
                },
            )
            assert empty.status == 400

            # Single-select shape must not unlock a multi-select pending entry.
            single = await cli.post(
                f"/v1/runs/{run_id}/clarification",
                json={
                    "request_id": request_id,
                    "response": {"type": "choice", "choice_id": "choice-2"},
                },
            )
            assert single.status == 400
            assert clarify_mod.get_pending_by_id(request_id, session_key=run_id) is not None

        clarify_mod.clear_session(run_id)

    @pytest.mark.asyncio
    async def test_clarification_is_bound_to_run_and_single_use(self, adapter):
        app = _create_runs_app(adapter)
        first_run = "run_first"
        second_run = "run_second"
        request_id = "clarify_" + "a" * 32
        for run_id in (first_run, second_run):
            adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
            adapter._run_clarify_sessions[run_id] = run_id
        clarify_mod.register(request_id, second_run, "Question?", ["One", "Two"])

        async with TestClient(TestServer(app)) as cli:
            cross_run = await cli.post(
                f"/v1/runs/{first_run}/clarification",
                json={
                    "request_id": request_id,
                    "response": {"type": "choice", "choice_id": "choice-1"},
                },
            )
            assert cross_run.status == 409

            exact = await cli.post(
                f"/v1/runs/{second_run}/clarification",
                json={
                    "request_id": request_id,
                    "response": {"type": "choice", "choice_id": "choice-2"},
                },
            )
            assert exact.status == 200
            assert clarify_mod.wait_for_response(request_id, timeout=0.1) == "Two"

            duplicate = await cli.post(
                f"/v1/runs/{second_run}/clarification",
                json={
                    "request_id": request_id,
                    "response": {"type": "choice", "choice_id": "choice-2"},
                },
            )
            assert duplicate.status == 409

    @pytest.mark.asyncio
    async def test_clarification_rejects_other_profile(self, adapter, tmp_path, monkeypatch):
        """A valid other-profile key must not resolve another profile's pending clarify."""
        from agent import secret_scope as ss
        from gateway.config import GatewayConfig

        alpha_home = tmp_path / "profiles" / "alpha"
        beta_home = tmp_path / "profiles" / "beta"
        alpha_home.mkdir(parents=True)
        beta_home.mkdir(parents=True)
        alpha_key = "a" * 32
        beta_key = "b" * 32
        (alpha_home / ".env").write_text(f"API_SERVER_KEY={alpha_key}\n", encoding="utf-8")
        (beta_home / ".env").write_text(f"API_SERVER_KEY={beta_key}\n", encoding="utf-8")

        adapter._api_key = "c" * 32
        adapter.gateway_runner = type(
            "_Runner", (), {"config": GatewayConfig(multiplex_profiles=True)}
        )()
        monkeypatch.setattr(
            "hermes_cli.profiles.profiles_to_serve",
            lambda multiplex, profile_allowlist=None: [
                ("default", tmp_path),
                ("alpha", alpha_home),
                ("beta", beta_home),
            ],
        )
        monkeypatch.setattr(
            "hermes_cli.profiles.get_profile_dir",
            lambda name: {"alpha": alpha_home, "beta": beta_home}.get(name, tmp_path),
        )
        ss.set_multiplex_active(True)
        app = _create_multiplex_runs_app(adapter)
        callback_result = {}

        def make_agent(**kwargs):
            mock_agent = MagicMock()

            def run_conversation(**_run_kwargs):
                callback_result["answer"] = kwargs["clarify_callback"](
                    "Which env?",
                    ["Staging", "Production"],
                )
                return {"final_response": callback_result["answer"]}

            mock_agent.run_conversation.side_effect = run_conversation
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        try:
            async with TestClient(TestServer(app)) as cli:
                with patch.object(adapter, "_create_agent", side_effect=make_agent):
                    started = await cli.post(
                        "/p/alpha/v1/runs",
                        json={"input": "hello"},
                        headers={"Authorization": f"Bearer {alpha_key}"},
                    )
                    assert started.status == 202
                    run_id = (await started.json())["run_id"]
                    event = await asyncio.wait_for(
                        adapter._run_streams[run_id].get(), timeout=3
                    )
                    assert event["event"] == "clarify.request"
                    assert adapter._run_statuses[run_id]["request_profile"] == "alpha"
                    pending = clarify_mod.get_pending_by_id(
                        event["request_id"], session_key=run_id
                    )
                    assert pending is not None

                    body = {
                        "request_id": event["request_id"],
                        "response": {"type": "choice", "choice_id": "choice-1"},
                    }
                    cross = await cli.post(
                        f"/p/beta/v1/runs/{run_id}/clarification",
                        json=body,
                        headers={"Authorization": f"Bearer {beta_key}"},
                    )
                    assert cross.status == 404
                    assert clarify_mod.get_pending_by_id(
                        event["request_id"], session_key=run_id
                    ) is not None
                    assert "answer" not in callback_result

                    owned = await cli.post(
                        f"/p/alpha/v1/runs/{run_id}/clarification",
                        json=body,
                        headers={"Authorization": f"Bearer {alpha_key}"},
                    )
                    assert owned.status == 200

                    for _ in range(40):
                        if callback_result.get("answer") == "Staging":
                            break
                        await asyncio.sleep(0.05)
                    assert callback_result["answer"] == "Staging"
        finally:
            ss.set_multiplex_active(False)

    @pytest.mark.asyncio
    async def test_clarification_rejects_malformed_and_oversized_text(self, adapter):
        app = _create_runs_app(adapter)
        run_id = "run_validation"
        request_id = "clarify_" + "b" * 32
        adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        adapter._run_clarify_sessions[run_id] = run_id
        clarify_mod.register(request_id, run_id, "Question?", None)

        async with TestClient(TestServer(app)) as cli:
            malformed = await cli.post(
                f"/v1/runs/{run_id}/clarification",
                json={"request_id": "../not-safe", "response": {"type": "text", "text": "ok"}},
            )
            assert malformed.status == 400

            too_long = await cli.post(
                f"/v1/runs/{run_id}/clarification",
                json={
                    "request_id": request_id,
                    "response": {"type": "text", "text": "x" * 2001},
                },
            )
            assert too_long.status == 400
            assert clarify_mod.get_pending_by_id(request_id, session_key=run_id) is not None

            accepted = await cli.post(
                f"/v1/runs/{run_id}/clarification",
                json={
                    "request_id": request_id,
                    "response": {"type": "text", "text": "A bounded answer"},
                },
            )
            assert accepted.status == 200
            assert clarify_mod.wait_for_response(request_id, timeout=0.1) == "A bounded answer"

    @pytest.mark.asyncio
    async def test_clarification_requires_api_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        run_id = "run_auth"
        request_id = "clarify_" + "c" * 32
        auth_adapter._run_statuses[run_id] = {"run_id": run_id, "status": "running"}
        auth_adapter._run_clarify_sessions[run_id] = run_id
        clarify_mod.register(request_id, run_id, "Question?", None)

        async with TestClient(TestServer(app)) as cli:
            response = await cli.post(
                f"/v1/runs/{run_id}/clarification",
                json={
                    "request_id": request_id,
                    "response": {"type": "text", "text": "answer"},
                },
            )
            assert response.status == 401
            assert clarify_mod.get_pending_by_id(request_id, session_key=run_id) is not None

        clarify_mod.clear_session(run_id)

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


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/steer — steer a running agent
# ---------------------------------------------------------------------------


class TestSteerRun:
    @pytest.mark.asyncio
    async def test_steer_running_agent(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        queue = asyncio.Queue()
        adapter._active_run_agents["run_123"] = agent
        adapter._run_streams["run_123"] = queue
        adapter._set_run_status("run_123", "running")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_123/steer", json={"input": "tighten the ending"})
            payload = await resp.json()

        assert resp.status == 200
        assert payload == {
            "object": "hermes.run.steer",
            "run_id": "run_123",
            "accepted": True,
        }
        agent.steer.assert_called_once_with("tighten the ending")
        assert adapter._run_statuses["run_123"]["last_event"] == "run.steered"
        event = queue.get_nowait()
        assert event["event"] == "run.steered"
        assert event["run_id"] == "run_123"
        assert event["accepted"] is True

    @pytest.mark.asyncio
    async def test_steer_nonexistent_run_returns_404(self, adapter):
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_missing/steer", json={"input": "hello"})
            payload = await resp.json()

        assert resp.status == 404
        assert payload["error"]["code"] == "run_not_found"

    @pytest.mark.asyncio
    async def test_steer_inactive_run_returns_409(self, adapter):
        app = _create_runs_app(adapter)
        adapter._set_run_status("run_done", "completed")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_done/steer", json={"input": "hello"})
            payload = await resp.json()

        assert resp.status == 409
        assert payload["error"]["code"] == "run_not_accepting_steer"

    @pytest.mark.asyncio
    async def test_steer_missing_input_returns_400(self, adapter):
        app = _create_runs_app(adapter)
        agent = MagicMock()
        agent.steer.return_value = True
        adapter._active_run_agents["run_123"] = agent
        adapter._set_run_status("run_123", "running")

        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_123/steer", json={"input": ""})
            payload = await resp.json()

        assert resp.status == 400
        assert payload["error"]["code"] == "invalid_steer_input"
        agent.steer.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_then_steer_rejects_retained_agent_ref(self, adapter):
        """Steer must reject a stopping run even if the executor thread is still live."""
        app = _create_runs_app(adapter)
        run_can_finish = threading.Event()
        run_started = threading.Event()

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_agent.steer = MagicMock(return_value=True)

                def _interrupt(_message=None):
                    return None

                def _run_conversation(*_args, **_kwargs):
                    run_started.set()
                    run_can_finish.wait(timeout=5)
                    return {"final_response": "late result"}

                mock_agent.interrupt = MagicMock(side_effect=_interrupt)
                mock_agent.run_conversation.side_effect = _run_conversation
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]
                assert run_started.wait(timeout=3.0)

                stop_resp = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stop_resp.status == 200
                assert run_id in adapter._active_run_agents

                steer_resp = await cli.post(
                    f"/v1/runs/{run_id}/steer",
                    json={"input": "tighten the ending"},
                )
                steer_data = await steer_resp.json()

                assert steer_resp.status == 409
                assert steer_data["error"]["code"] == "run_not_accepting_steer"
                mock_agent.steer.assert_not_called()

                run_can_finish.set()
                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

    @pytest.mark.asyncio
    async def test_pending_steer_preserved_on_run_completed(self, adapter):
        """A steer drained by the turn finalizer (accepted after the final
        response) must surface as pending_steer on the terminal run status
        instead of being silently dropped."""
        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_agent = MagicMock()
                mock_agent.session_prompt_tokens = 0
                mock_agent.session_completion_tokens = 0
                mock_agent.session_total_tokens = 0
                mock_agent.run_conversation.return_value = {
                    "final_response": "done",
                    "pending_steer": "tighten the ending",
                }
                mock_create.return_value = mock_agent

                start_resp = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await start_resp.json())["run_id"]

                for _ in range(40):
                    status = adapter._run_statuses.get(run_id, {})
                    if status.get("status") == "completed":
                        break
                    await asyncio.sleep(0.05)

        assert adapter._run_statuses[run_id]["status"] == "completed"
        assert adapter._run_statuses[run_id]["pending_steer"] == "tighten the ending"

    @pytest.mark.asyncio
    async def test_steer_requires_auth(self, auth_adapter):
        app = _create_runs_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_any/steer", json={"input": "hello"})

        assert resp.status == 401


# ---------------------------------------------------------------------------
# Run lifecycle TTL sweeping
# ---------------------------------------------------------------------------


class TestRunLifecycleSweep:

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


# ---------------------------------------------------------------------------
# POST /v1/runs/{run_id}/stop — interrupt a running agent
# ---------------------------------------------------------------------------


class TestStopRun:
    @pytest.mark.asyncio
    async def test_stop_releases_pending_clarification(self, adapter):
        app = _create_runs_app(adapter)

        def make_agent(**kwargs):
            mock_agent = MagicMock()
            mock_agent.run_conversation.side_effect = lambda **_run_kwargs: {
                "final_response": kwargs["clarify_callback"](
                    "Q" * 2500,
                    ["X" * 600, "Y" * 600, "Z" * 600, "W" * 600, "extra"],
                )
            }
            mock_agent.session_prompt_tokens = 0
            mock_agent.session_completion_tokens = 0
            mock_agent.session_total_tokens = 0
            return mock_agent

        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent", side_effect=make_agent):
                started = await cli.post("/v1/runs", json={"input": "hello"})
                run_id = (await started.json())["run_id"]
                event = await asyncio.wait_for(
                    adapter._run_streams[run_id].get(), timeout=3
                )
                assert event["event"] == "clarify.request"
                assert len(event["prompt"]["question"]) == 2000
                assert len(event["prompt"]["choices"]) == 4
                assert all(
                    len(choice["label"]) == 500
                    for choice in event["prompt"]["choices"]
                )

                stopped = await cli.post(f"/v1/runs/{run_id}/stop")
                assert stopped.status == 200
                assert clarify_mod.get_pending_by_id(
                    event["request_id"], session_key=run_id
                ) is None
                assert adapter._run_statuses[run_id]["awaiting_user"] is False
                assert adapter._run_statuses[run_id]["session_id"] not in adapter._session_awaiting_user

                for _ in range(40):
                    if run_id not in adapter._active_run_tasks:
                        break
                    await asyncio.sleep(0.05)

                assert adapter._run_statuses[run_id]["status"] == "cancelled"
                assert run_id not in adapter._run_clarify_sessions

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
                await asyncio.sleep(0.2)
                assert run_id not in adapter._active_run_agents
                assert run_id not in adapter._active_run_tasks


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


class TestRunsProviderAuthFailure:
    @pytest.mark.asyncio
    async def test_status_reports_provider_auth_failure_distinctly(self, adapter):
        """/v1/runs builds its own agent via _create_agent() and does not
        route through _run_agent(), so the controlled "Provider
        authentication failed" message added there does not cover this
        endpoint. _handle_runs()'s own _ProviderAuthResolutionError branch
        must give the same distinguished message instead of the generic
        except-Exception "run failed" text."""
        from gateway.platforms.api_server import _ProviderAuthResolutionError

        app = _create_runs_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            with patch.object(adapter, "_create_agent") as mock_create:
                mock_create.side_effect = _ProviderAuthResolutionError(
                    "No credentials found for provider 'nous'"
                )

                resp = await cli.post("/v1/runs", json={"input": "hello"})
                assert resp.status == 202
                data = await resp.json()
                run_id = data["run_id"]

                for _ in range(40):
                    status_resp = await cli.get(f"/v1/runs/{run_id}")
                    status = await status_resp.json()
                    if status["status"] == "failed":
                        break
                    await asyncio.sleep(0.05)

                assert status["status"] == "failed"
                assert status["error"] == "⚠️ Provider authentication failed: No credentials found for provider 'nous'"
                assert status["last_event"] == "run.failed"
