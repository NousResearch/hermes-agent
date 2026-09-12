"""Boundary tests for route-scoped approval on the API server.

The durable ``/v1/runs`` route owns an authenticated approval resolver, while
the OpenAI-compatible routes do not.  These tests call the real adapter
lifecycle with a small agent double whose only work is the harmless
``execute_code`` guard; no code is spawned.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import (
    APIServerAdapter,
    cors_middleware,
    security_headers_middleware,
)
from tools import approval as approval_module
from tools import approval_context


class _GuardAgent:
    """Agent double that exposes the real execute-code approval result."""

    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_total_tokens = 0

    def __init__(self, seen: list[dict], *, calls: int = 1, hardline: bool = False):
        self.seen = seen
        self.calls = calls
        self.hardline = hardline

    def run_conversation(self, **_kwargs):
        for _ in range(self.calls):
            if self.hardline:
                result = approval_module.check_all_command_guards("rm -rf /", "local")
            else:
                result = approval_module.check_execute_code_guard(
                    "print('approval-probe')", "local"
                )
            self.seen.append(result)
            if not result.get("approved"):
                return {
                    "final_response": result.get("message", "blocked"),
                    "messages": [],
                }
        return {"final_response": "approval-probe continued", "messages": []}


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    middlewares = [
        middleware
        for middleware in (cors_middleware, security_headers_middleware)
        if middleware is not None
    ]
    app = web.Application(middlewares=middlewares)
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_post("/v1/runs/{run_id}/approval", adapter._handle_run_approval)
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    return app


async def _run_status(client: TestClient, run_id: str) -> dict:
    response = await client.get(f"/v1/runs/{run_id}")
    assert response.status == 200
    return await response.json()


async def _wait_for_status(
    client: TestClient, run_id: str, expected: str, *, timeout: float = 4.0
) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = await _run_status(client, run_id)
        if last.get("status") == expected:
            return last
        await asyncio.sleep(0.02)
    raise AssertionError(f"run {run_id} did not reach {expected!r}: {last!r}")


def _isolated_approval(monkeypatch):
    """Keep tests independent from the live host policy and approval state."""
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_context, "_get_approval_mode", lambda: "manual")
    monkeypatch.setattr(
        approval_context, "_get_unattended_approval_mode", lambda: "deny"
    )


def _clear_run_approval(run_id: str) -> None:
    approval_module.clear_session(run_id)
    with approval_module._lock:
        approval_module._gateway_queues.pop(run_id, None)
        approval_module._gateway_notify_cbs.pop(run_id, None)


@pytest.mark.asyncio
async def test_runs_execute_code_approval_round_trip_once(monkeypatch):
    """A real /v1/runs resolver must emit and resolve a one-shot approval."""
    _isolated_approval(monkeypatch)
    adapter = _make_adapter()
    seen: list[dict] = []
    agent = _GuardAgent(seen)

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            response = await client.post(
                "/v1/runs", json={"input": "safe approval control-flow probe"}
            )
        assert response.status == 202
        run_id = (await response.json())["run_id"]

        waiting = await _wait_for_status(client, run_id, "waiting_for_approval")
        assert waiting["approval"]["event"] == "approval.request"
        assert waiting["approval"]["run_id"] == run_id
        assert seen == []
        assert getattr(adapter, "_run_approval_sessions")[run_id] == run_id

        approval_response = await client.post(
            f"/v1/runs/{run_id}/approval", json={"choice": "once"}
        )
        assert approval_response.status == 200
        assert (await approval_response.json())["resolved"] == 1

        completed = await _wait_for_status(client, run_id, "completed")
        assert completed["output"] == "approval-probe continued"
        assert seen[0]["approved"] is True
        assert seen[0].get("user_approved") is True

    _clear_run_approval(run_id)


@pytest.mark.asyncio
async def test_runs_execute_code_explicit_deny_stops_before_continuation(monkeypatch):
    """The same run-scoped transport must honor an explicit deny."""
    _isolated_approval(monkeypatch)
    adapter = _make_adapter()
    seen: list[dict] = []
    agent = _GuardAgent(seen)

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            response = await client.post("/v1/runs", json={"input": "deny probe"})
        run_id = (await response.json())["run_id"]
        await _wait_for_status(client, run_id, "waiting_for_approval")

        approval_response = await client.post(
            f"/v1/runs/{run_id}/approval", json={"choice": "deny"}
        )
        assert approval_response.status == 200

        completed = await _wait_for_status(client, run_id, "completed")
        assert "BLOCKED" in completed["output"]
        assert seen[0]["approved"] is False
        assert seen[0]["outcome"] == "denied"

    _clear_run_approval(run_id)


@pytest.mark.asyncio
async def test_runs_approval_isolated_between_simultaneous_runs(monkeypatch):
    """Approving one run must not resolve another run's pending guard."""
    _isolated_approval(monkeypatch)
    adapter = _make_adapter()
    seen_a: list[dict] = []
    seen_b: list[dict] = []
    agents = [_GuardAgent(seen_a), _GuardAgent(seen_b)]

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", side_effect=agents):
            first = await client.post("/v1/runs", json={"input": "run A"})
            second = await client.post("/v1/runs", json={"input": "run B"})
        run_a = (await first.json())["run_id"]
        run_b = (await second.json())["run_id"]

        await _wait_for_status(client, run_a, "waiting_for_approval")
        await _wait_for_status(client, run_b, "waiting_for_approval")

        approval_response = await client.post(
            f"/v1/runs/{run_a}/approval", json={"choice": "once"}
        )
        assert approval_response.status == 200
        await _wait_for_status(client, run_a, "completed")
        still_waiting = await _run_status(client, run_b)
        assert still_waiting["status"] == "waiting_for_approval"
        assert seen_b == []

        denial_response = await client.post(
            f"/v1/runs/{run_b}/approval", json={"choice": "deny"}
        )
        assert denial_response.status == 200
        await _wait_for_status(client, run_b, "completed")
        assert seen_a[0]["approved"] is True
        assert seen_b[0]["approved"] is False

    _clear_run_approval(run_a)
    _clear_run_approval(run_b)


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["session", "always"])
async def test_runs_persistent_choice_applies_only_inside_that_run(monkeypatch, choice):
    """Persistent approval choices stay local to the run's exact approval key."""
    _isolated_approval(monkeypatch)
    adapter = _make_adapter()
    seen: list[dict] = []
    agent = _GuardAgent(seen, calls=2)

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=agent):
            response = await client.post("/v1/runs", json={"input": "session probe"})
        run_id = (await response.json())["run_id"]
        await _wait_for_status(client, run_id, "waiting_for_approval")

        approval_response = await client.post(
            f"/v1/runs/{run_id}/approval", json={"choice": choice}
        )
        assert approval_response.status == 200
        completed = await _wait_for_status(client, run_id, "completed")

        assert completed["output"] == "approval-probe continued"
        assert len(seen) == 2
        assert all(result["approved"] is True for result in seen)

    _clear_run_approval(run_id)
    with approval_module._lock:
        approval_module._permanent_approved.discard("execute_code")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [
        (
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "chat probe"}]},
        ),
        ("/v1/responses", {"input": "responses probe", "store": False}),
    ],
)
async def test_listenerless_openai_routes_keep_execute_code_fail_closed(
    monkeypatch, endpoint, payload
):
    """OpenAI routes have no approval responder and must not create a waiter."""
    _isolated_approval(monkeypatch)
    adapter = _make_adapter()
    seen: list[dict] = []

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_create_agent", return_value=_GuardAgent(seen)):
            response = await client.post(endpoint, json=payload)
        assert response.status == 200
        assert seen[0]["approved"] is False
        assert seen[0]["outcome"] == "blocked"
        assert not approval_module._gateway_queues
        assert not approval_module._pending


@pytest.mark.asyncio
async def test_runs_resolver_does_not_bypass_hardline_floor(monkeypatch):
    """A resolver-backed run still stops before ordinary human approval."""
    _isolated_approval(monkeypatch)
    adapter = _make_adapter()
    seen: list[dict] = []

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(
            adapter, "_create_agent", return_value=_GuardAgent(seen, hardline=True)
        ):
            response = await client.post("/v1/runs", json={"input": "floor probe"})
        run_id = (await response.json())["run_id"]
        completed = await _wait_for_status(client, run_id, "completed")

        assert completed["output"]
        assert seen[0]["approved"] is False
        assert seen[0]["hardline"] is True
        assert run_id not in approval_module._gateway_queues

    _clear_run_approval(run_id)
