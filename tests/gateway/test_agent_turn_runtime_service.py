"""Unit tests for gateway sync-turn runtime helpers."""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.agent_turn_runtime_service import (
    build_gateway_background_review_callback,
    configure_gateway_agent_for_turn,
    reuse_or_create_gateway_agent,
)


def test_reuse_or_create_gateway_agent_reuses_cached_instance():
    cached_agent = object()
    cache = {"session-1": (cached_agent, "sig-1")}

    agent, created = reuse_or_create_gateway_agent(
        session_key="session-1",
        signature="sig-1",
        cache=cache,
        cache_lock=threading.Lock(),
        create_agent=lambda: (_ for _ in ()).throw(AssertionError("should not create")),
        logger=MagicMock(),
    )

    assert agent is cached_agent
    assert created is False


def test_reuse_or_create_gateway_agent_creates_and_caches_on_miss():
    created_agent = object()
    cache = {}

    agent, created = reuse_or_create_gateway_agent(
        session_key="session-1",
        signature="sig-1",
        cache=cache,
        cache_lock=threading.Lock(),
        create_agent=lambda: created_agent,
        logger=MagicMock(),
    )

    assert agent is created_agent
    assert created is True
    assert cache["session-1"] == (created_agent, "sig-1")


def test_configure_gateway_agent_for_turn_sets_callbacks():
    agent = SimpleNamespace(
        tool_progress_callback=None,
        step_callback=None,
        stream_delta_callback=None,
        status_callback=None,
        reasoning_config=None,
        background_review_callback=None,
    )
    progress_runtime = SimpleNamespace(
        progress_callback="progress-cb",
        step_callback="step-cb",
        status_callback="status-cb",
    )

    configure_gateway_agent_for_turn(
        agent=agent,
        progress_runtime=progress_runtime,
        stream_delta_callback="stream-cb",
        reasoning_config={"effort": "high"},
        background_review_callback="bg-cb",
    )

    assert agent.tool_progress_callback == "progress-cb"
    assert agent.step_callback == "step-cb"
    assert agent.stream_delta_callback == "stream-cb"
    assert agent.status_callback == "status-cb"
    assert agent.reasoning_config == {"effort": "high"}
    assert agent.background_review_callback == "bg-cb"


def test_build_gateway_background_review_callback_noops_without_adapter():
    callback = build_gateway_background_review_callback(
        status_adapter=None,
        status_chat_id="12345",
        status_thread_metadata=None,
        loop_for_step=None,
        logger=MagicMock(),
    )

    callback("ignored")


def test_build_gateway_background_review_callback_schedules_send(monkeypatch):
    adapter = SimpleNamespace(send=MagicMock(return_value="awaitable"))
    run_coroutine = MagicMock()
    future = MagicMock()
    future.result.return_value = None
    run_coroutine.return_value = future
    monkeypatch.setattr(
        "gateway.agent_turn_runtime_service.asyncio.run_coroutine_threadsafe",
        run_coroutine,
    )

    callback = build_gateway_background_review_callback(
        status_adapter=adapter,
        status_chat_id="12345",
        status_thread_metadata={"thread_id": "7"},
        loop_for_step="loop",
        logger=MagicMock(),
    )

    callback("memory updated")

    adapter.send.assert_called_once_with(
        "12345",
        "memory updated",
        metadata={"thread_id": "7"},
    )
    run_coroutine.assert_called_once()
