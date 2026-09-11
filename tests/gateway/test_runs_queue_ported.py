"""Ported behavioral pins for POST /v1/runs/{run_id}/queue + CORS session header.

Adapted from the old-branch pins (``tests/gateway/test_runs_steer_queue.py`` @
c861dcd0e8) to the new tree's wire shapes:

- queue response: ``{object: hermes.run.queue, run_id, status: queued, prompt, depth}``
- queue errors: 404 ``run_not_found`` / 400 ``invalid_queue_prompt``
- ownership via ``_claim_run`` + live agent/task (``_load_owned_run``,
  ``active_fallback=True``)
- drain: ``_execute_run`` drains ``_run_queues`` FIFO as chained follow-up
  turns with ``run.queued_turn_started`` / ``run.queued_turn_completed`` SSE
  events (mirrors ACP ``queued_prompts``)
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server_runs
from gateway.platforms.api_server import (
    APIServerAdapter,
    _CORS_HEADERS,
    cors_middleware,
    security_headers_middleware,
)


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _claim_run(adapter: APIServerAdapter, run_id: str) -> None:
    request = MagicMock()
    request.headers = {}
    adapter._run_owners[run_id] = adapter._run_idempotency_scope(request)


def _create_queue_app(adapter: APIServerAdapter) -> web.Application:
    mws = [mw for mw in (cors_middleware, security_headers_middleware) if mw is not None]
    app = web.Application(middlewares=mws)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs/{run_id}/queue", adapter._handle_queue_run)
    return app


@pytest.fixture
def adapter():
    return _make_adapter()


@pytest.fixture
def auth_adapter():
    return _make_adapter(api_key="sk-secret")


def _live_run(adapter: APIServerAdapter, run_id: str) -> None:
    """Stage a live run the queue handler admits: owned + agent + task."""
    adapter._active_run_agents[run_id] = MagicMock()
    adapter._active_run_tasks[run_id] = MagicMock()
    _claim_run(adapter, run_id)


class TestQueueRunEndpoint:
    @pytest.mark.asyncio
    async def test_queue_accepted(self, adapter):
        _live_run(adapter, "run_1")
        app = _create_queue_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_1/queue", json={"prompt": "follow up"})
            assert resp.status == 200
            body = await resp.json()
        assert body == {
            "object": "hermes.run.queue",
            "run_id": "run_1",
            "status": "queued",
            "prompt": "follow up",
            "depth": 1,
        }
        assert adapter._run_queues["run_1"] == ["follow up"]

    @pytest.mark.asyncio
    async def test_queue_fifo_depth(self, adapter):
        _live_run(adapter, "run_1")
        app = _create_queue_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            first = await cli.post("/v1/runs/run_1/queue", json={"prompt": "first"})
            assert (await first.json())["depth"] == 1
            second = await cli.post("/v1/runs/run_1/queue", json={"prompt": "second"})
            body = await second.json()
        assert body["depth"] == 2
        assert adapter._run_queues["run_1"] == ["first", "second"]

    @pytest.mark.asyncio
    async def test_queue_unknown_run_404(self, adapter):
        app = _create_queue_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/nope/queue", json={"prompt": "x"})
            assert resp.status == 404
            body = await resp.json()
        assert body["error"]["code"] == "run_not_found"

    @pytest.mark.asyncio
    async def test_queue_empty_prompt_400(self, adapter):
        _live_run(adapter, "run_1")
        app = _create_queue_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            for bad in ({}, {"prompt": ""}, {"prompt": "   "}, {"prompt": 123}):
                resp = await cli.post("/v1/runs/run_1/queue", json=bad)
                assert resp.status == 400
                assert (await resp.json())["error"]["code"] == "invalid_queue_prompt"
        assert adapter._run_queues.get("run_1", []) == []

    @pytest.mark.asyncio
    async def test_queue_requires_auth(self, auth_adapter):
        app = _create_queue_app(auth_adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/v1/runs/run_1/queue", json={"prompt": "x"})
            assert resp.status == 401


class TestQueueFifoDrain:
    @pytest.mark.asyncio
    async def test_drain_runs_queued_prompts_as_chained_turns_fifo(self, adapter):
        """Queued prompts drain FIFO as chained turns: each runs with prior
        turns folded into history, usage accumulates, and progress surfaces
        as run.queued_turn_started/completed SSE events."""
        run_id = "run_q"
        q: asyncio.Queue = asyncio.Queue()
        adapter._run_streams[run_id] = q
        adapter._run_streams_created[run_id] = time.time()
        adapter._run_queues[run_id] = ["second", "third"]

        launch = api_server_runs._RunLaunch(
            owner=adapter,
            run_id=run_id,
            queue=q,
            session_id="sess-1",
            gateway_session_key=None,
            declared_selected=False,
            user_message="first",
            conversation_history=[],
            agent_kwargs={},
            request_profile=None,
            browser_control_principal=None,
            browser_control_transport_family=None,
        )

        seen_messages = []
        seen_histories = []

        def _fake_sync(self, run, agent, approval_notify, *, _api_server):
            seen_messages.append(run.user_message)
            seen_histories.append(list(run.conversation_history))
            n = len(seen_messages)
            return (
                {"final_response": f"r{n}"},
                {"input_tokens": n, "output_tokens": 2 * n, "total_tokens": 3 * n},
            )

        mock_agent = MagicMock()
        import gateway.platforms.api_server as api_server_mod

        with (
            patch.object(adapter, "_create_agent", return_value=mock_agent),
            patch.object(api_server_runs, "_run_agent_sync", _fake_sync),
        ):
            await api_server_runs._execute_run(adapter, launch, _api_server=api_server_mod)

        # FIFO: initial turn first, then queued prompts in order.
        assert seen_messages == ["first", "second", "third"]
        # Each follow-up folds prior turns into history (user/assistant pairs).
        assert seen_histories[1] == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "r1"},
        ]
        assert seen_histories[2] == seen_histories[1] + [
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "r2"},
        ]

        events = []
        while not q.empty():
            events.append(q.get_nowait())
        names = [e["event"] for e in events if e is not None]
        assert names == [
            "run.queued_turn_started",
            "run.queued_turn_completed",
            "run.queued_turn_started",
            "run.queued_turn_completed",
            "run.completed",
        ]
        started = [e for e in events if e is not None and e["event"] == "run.queued_turn_started"]
        assert [e["depth_remaining"] for e in started] == [1, 0]
        completed = [e for e in events if e is not None and e["event"] == "run.queued_turn_completed"]
        assert [e["output"] for e in completed] == ["r2", "r3"]

        # Usage accumulates across all turns; queue state is dropped with the run.
        assert adapter._run_statuses[run_id]["usage"] == {
            "input_tokens": 6,
            "output_tokens": 12,
            "total_tokens": 18,
        }
        assert run_id not in adapter._run_queues


class TestCorsSessionHeader:
    def test_allow_headers_include_session_id(self):
        """Browser clients must be allowed to send X-Hermes-Session-Id."""
        assert "X-Hermes-Session-Id" in _CORS_HEADERS["Access-Control-Allow-Headers"]

    @pytest.mark.asyncio
    async def test_preflight_echoes_session_id_header(self):
        adapter = APIServerAdapter(
            PlatformConfig(enabled=True, extra={"cors_origins": ["http://localhost:3000"]})
        )
        app = _create_queue_app(adapter)
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.options(
                "/v1/runs/run_1/queue",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "X-Hermes-Session-Id",
                },
            )
            assert resp.status == 200
            assert (
                resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"
            )
            assert "X-Hermes-Session-Id" in resp.headers.get(
                "Access-Control-Allow-Headers", ""
            )
