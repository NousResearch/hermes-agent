"""OpenAI request ``metadata`` must reach Relay ``hermes.turn`` scope metadata (#107693)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from agent import relay_runtime
from agent.relay_runtime import RelayRuntime, RelaySessionCoordinator
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _make_adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/chat/completions", adapter._handle_chat_completions)
    app.router.add_post("/v1/responses", adapter._handle_responses)
    return app


def _agent_ok():
    return (
        {"final_response": "ok", "messages": [], "api_calls": 1},
        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )


@pytest.fixture
def adapter():
    return _make_adapter()


class _ScopeHandle:
    def __init__(self, name: str, seq: int) -> None:
        self.name = name
        self.seq = seq


class _FakeScopeModule:
    def __init__(self) -> None:
        self._seq = 0
        self.pushes: list[dict[str, Any]] = []

    def push(self, name: str, scope_type: Any, **kwargs: Any) -> _ScopeHandle:
        self._seq += 1
        self.pushes.append(
            {
                "name": name,
                "metadata": dict(kwargs.get("metadata") or {}),
                "parent": kwargs.get("handle"),
                "seq": self._seq,
            }
        )
        return _ScopeHandle(name, self._seq)

    def pop(self, handle: _ScopeHandle, **kwargs: Any) -> None:
        return None

    def event(self, *args: Any, **kwargs: Any) -> None:
        return None


class _FakeSubscribers:
    def flush(self) -> None:
        return None


class _FakeScopeType:
    Function = "function"
    Agent = "agent"


class _FakeRelay:
    def __init__(self) -> None:
        self.scope = _FakeScopeModule()
        self.subscribers = _FakeSubscribers()
        self.ScopeType = _FakeScopeType()

    def get_scope_stack(self) -> None:
        return None


def _turn_pushes(fake: _FakeRelay) -> list[dict[str, Any]]:
    return [p for p in fake.scope.pushes if p["name"] == relay_runtime.TURN_SCOPE]


def _acquire(coordinator: RelaySessionCoordinator, runtime: RelayRuntime, session_id: str = "sess-1"):
    class _Registry:
        def for_profile(self, key):
            return runtime

    coordinator.registry = _Registry()
    coordinator._prepare_session = lambda host, ctx: None
    return coordinator.acquire_conversation(
        profile_key=runtime.profile_key,
        session_id=session_id,
        platform="api_server",
    )


@pytest.mark.asyncio
async def test_responses_forwards_request_metadata_to_run_agent(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = _agent_ok()
            resp = await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": "hello",
                    "metadata": {"request_id": "request-123"},
                },
            )
            assert resp.status == 200
    kwargs = mock_run.call_args.kwargs
    assert kwargs.get("request_metadata") is not None
    assert kwargs["request_metadata"]["request_id"] == "request-123"


@pytest.mark.asyncio
async def test_chat_completions_forwards_request_metadata_to_run_agent(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = _agent_ok()
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "hello"}],
                    "metadata": {"request_id": "request-123"},
                },
            )
            assert resp.status == 200
    kwargs = mock_run.call_args.kwargs
    assert kwargs.get("request_metadata") is not None
    assert kwargs["request_metadata"]["request_id"] == "request-123"


@pytest.mark.asyncio
async def test_non_dict_metadata_is_fail_open(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = _agent_ok()
            resp = await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": "hello",
                    "metadata": "not-a-dict",
                },
            )
            assert resp.status == 200
    kwargs = mock_run.call_args.kwargs
    assert kwargs.get("request_metadata") in (None, {})


@pytest.mark.asyncio
async def test_omitted_metadata_is_fail_open(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = _agent_ok()
            resp = await cli.post(
                "/v1/chat/completions",
                json={
                    "model": "hermes-agent",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            assert resp.status == 200
    kwargs = mock_run.call_args.kwargs
    assert kwargs.get("request_metadata") in (None, {})


def test_begin_turn_merges_request_metadata_into_hermes_turn_push():
    fake = _FakeRelay()
    runtime = RelayRuntime(relay=fake, profile_key="/tmp/test-profile-107693")
    coordinator = RelaySessionCoordinator()
    try:
        lease = _acquire(coordinator, runtime)
        turn = coordinator.begin_turn(
            lease,
            turn_id="turn-1",
            task_id="task-1",
            request_metadata={"request_id": "request-123"},
        )
        coordinator.end_turn(turn, outcome="success")
        pushes = _turn_pushes(fake)
        assert pushes, "expected a hermes.turn scope.push"
        meta = pushes[0]["metadata"]
        assert meta["request_id"] == "request-123"
        assert meta.get("hermes.execution_surface")
        assert meta.get(relay_runtime.RUNTIME_SCHEMA_KEY) == relay_runtime.RUNTIME_SCHEMA_VERSION
        assert relay_runtime.RUNTIME_INSTANCE_KEY in meta
    finally:
        runtime.shutdown()


def test_begin_turn_drops_client_hermes_execution_surface():
    fake = _FakeRelay()
    runtime = RelayRuntime(relay=fake, profile_key="/tmp/test-profile-107693-stamp")
    coordinator = RelaySessionCoordinator()
    try:
        lease = _acquire(coordinator, runtime)
        turn = coordinator.begin_turn(
            lease,
            turn_id="turn-2",
            task_id="task-2",
            request_metadata={
                "request_id": "request-123",
                "hermes.execution_surface": "attacker",
            },
        )
        coordinator.end_turn(turn, outcome="success")
        meta = _turn_pushes(fake)[0]["metadata"]
        assert meta["request_id"] == "request-123"
        assert meta["hermes.execution_surface"] != "attacker"
        assert meta["hermes.execution_surface"] == "api_server"
    finally:
        runtime.shutdown()


def test_begin_turn_accepts_client_runtime_id_key_without_crashing():
    """A legal OpenAI metadata key ``runtime_id`` must not collide with runtime_metadata()."""
    fake = _FakeRelay()
    runtime = RelayRuntime(relay=fake, profile_key="/tmp/test-profile-107693-runtime-id")
    coordinator = RelaySessionCoordinator()
    try:
        lease = _acquire(coordinator, runtime)
        turn = coordinator.begin_turn(
            lease,
            turn_id="turn-3",
            task_id="task-3",
            request_metadata={
                "runtime_id": "client-runtime",
                "request_id": "request-123",
            },
        )
        coordinator.end_turn(turn, outcome="success")
        meta = _turn_pushes(fake)[0]["metadata"]
        assert meta["runtime_id"] == "client-runtime"
        assert meta["request_id"] == "request-123"
        assert meta[relay_runtime.RUNTIME_SCHEMA_KEY] == relay_runtime.RUNTIME_SCHEMA_VERSION
        assert meta[relay_runtime.RUNTIME_INSTANCE_KEY] == runtime.runtime_id
        assert meta[relay_runtime.RUNTIME_INSTANCE_KEY] != "client-runtime"
        assert meta["hermes.execution_surface"] == "api_server"
    finally:
        runtime.shutdown()


@pytest.mark.asyncio
async def test_client_hermes_stamp_is_dropped_from_run_agent_kwargs(adapter):
    app = _create_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        with patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = _agent_ok()
            resp = await cli.post(
                "/v1/responses",
                json={
                    "model": "hermes-agent",
                    "input": "hello",
                    "metadata": {
                        "request_id": "request-123",
                        "hermes.execution_surface": "attacker",
                    },
                },
            )
            assert resp.status == 200
    meta = mock_run.call_args.kwargs.get("request_metadata") or {}
    assert meta.get("request_id") == "request-123"
    assert "hermes.execution_surface" not in meta
