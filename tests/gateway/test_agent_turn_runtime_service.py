"""DEAD path: not imported by gateway/run.py — contract-only unit tests.

Unit tests for gateway sync-turn runtime helpers.
"""
import pytest

pytestmark = pytest.mark.dead_runtime_service

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

from gateway.agent_turn_runtime_service import (
    build_gateway_background_review_callback,
    configure_gateway_agent_for_turn,
    prepare_gateway_cached_turn_agent,
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


def test_prepare_gateway_cached_turn_agent_sets_up_streaming_cache_and_callbacks(
    monkeypatch,
):
    agent = object()
    runtime_spec = SimpleNamespace(
        turn_route={"model": "gpt-test", "runtime": {"api_key": "secret"}},
        enabled_toolsets=["hermes-qq"],
        combined_ephemeral="ctx",
    )
    progress_runtime = SimpleNamespace(
        progress_callback="progress-cb",
        step_callback="step-cb",
        status_callback="status-cb",
    )
    source = SimpleNamespace(chat_id="chat-1")
    setup_calls = {}
    reuse_calls = {}
    configure_calls = {}

    monkeypatch.setattr(
        "gateway.agent_turn_runtime_service.setup_gateway_stream_consumer",
        lambda **kwargs: setup_calls.update(kwargs) or ("stream-consumer", "stream-delta"),
    )
    monkeypatch.setattr(
        "gateway.agent_turn_runtime_service.agent_config_signature",
        lambda *args: "sig-1",
    )
    monkeypatch.setattr(
        "gateway.agent_turn_runtime_service.reuse_or_create_gateway_agent",
        lambda **kwargs: reuse_calls.update(kwargs) or (agent, True),
    )
    monkeypatch.setattr(
        "gateway.agent_turn_runtime_service.build_gateway_background_review_callback",
        lambda **kwargs: "background-review",
    )
    monkeypatch.setattr(
        "gateway.agent_turn_runtime_service.configure_gateway_agent_for_turn",
        lambda **kwargs: configure_calls.update(kwargs),
    )

    prepared = prepare_gateway_cached_turn_agent(
        runtime_spec=runtime_spec,
        session_key="session-1",
        session_id="turn-1",
        source=source,
        progress_runtime=progress_runtime,
        reasoning_config={"effort": "high"},
        streaming_config="streaming-config",
        adapter="adapter",
        thread_metadata={"thread_id": "7"},
        stream_consumer_holder=[None],
        cache={"session-1": ("old", "old-sig")},
        cache_lock=threading.Lock(),
        create_agent=lambda: agent,
        status_adapter="status-adapter",
        status_chat_id="chat-1",
        status_thread_metadata={"thread_id": "99"},
        loop_for_step="loop",
        logger=MagicMock(),
    )

    assert prepared.agent is agent
    assert prepared.stream_consumer == "stream-consumer"
    assert setup_calls["chat_id"] == "chat-1"
    assert setup_calls["thread_metadata"] == {"thread_id": "7"}
    assert reuse_calls["session_key"] == "session-1"
    assert reuse_calls["signature"] == "sig-1"
    assert callable(reuse_calls["create_agent"])
    assert configure_calls == {
        "agent": agent,
        "progress_runtime": progress_runtime,
        "stream_delta_callback": "stream-delta",
        "reasoning_config": {"effort": "high"},
        "background_review_callback": "background-review",
    }
