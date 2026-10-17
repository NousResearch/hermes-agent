"""Detached authority cannot fall back to a still-live dispatching worker."""
from contextvars import copy_context

import pytest

from agent.chat_execution_context import (
    bind_chat_execution, bind_concurrent_tool_resource,
    concurrent_tool_resource_for_submit, observe_tool_before_execution,
)
from gateway.chat_execution_recovery import ChatExecutionStore


def _started(home):
    store = ChatExecutionStore(home)
    assert store.reserve("key", 0, 0) == 1
    assert store.bind_worker_started("key", 1, "session")
    return store


def _parent_state(store, state):
    if state != "pending":
        assert store.worker_finished("key", 1)
    if state == "superseded":
        assert store.reserve("key", 0, 0, predecessor=(1, "session")) == 2


@pytest.mark.parametrize("parent_state", ["pending", "completed", "superseded"])
def test_detached_child_cannot_mint_detached_descendant(tmp_path, parent_state):
    store = _started(tmp_path)
    with bind_chat_execution(tmp_path, "key", 1):
        child = concurrent_tool_resource_for_submit("child", kind="async_delegation_thread")
        with bind_concurrent_tool_resource(child) as admitted:
            assert admitted
            _parent_state(store, parent_state)
            with pytest.raises(RuntimeError, match="reservation refused"):
                concurrent_tool_resource_for_submit("grandchild", kind="async_delegation_thread")


@pytest.mark.parametrize("parent_state", ["pending", "completed", "superseded"])
def test_fenced_detached_child_cannot_observe(tmp_path, parent_state):
    store = _started(tmp_path)
    with bind_chat_execution(tmp_path, "key", 1):
        child = concurrent_tool_resource_for_submit("child", kind="async_delegation_thread")
        with bind_concurrent_tool_resource(child) as admitted:
            assert admitted
            _parent_state(store, parent_state)
            if parent_state == "superseded":
                assert store.bind_worker_started("key", 2, "successor")
            assert store.begin_recovery("key", 2 if parent_state == "superseded" else 1, "fence")
            with pytest.raises(RuntimeError, match="Durable tool observation refused"):
                observe_tool_before_execution()


@pytest.mark.parametrize("parent_state", ["pending", "completed", "superseded"])
def test_finished_detached_context_cannot_regain_worker_authority(tmp_path, parent_state):
    store = _started(tmp_path)
    with bind_chat_execution(tmp_path, "key", 1):
        child = concurrent_tool_resource_for_submit("child", kind="async_delegation_thread")
        with bind_concurrent_tool_resource(child) as admitted:
            assert admitted
            descendant = concurrent_tool_resource_for_submit("descendant")
            copied = copy_context()
        _parent_state(store, parent_state)
        with pytest.raises(RuntimeError, match="reservation refused"):
            copied.run(concurrent_tool_resource_for_submit, "straggler")
        with pytest.raises(RuntimeError, match="Durable tool observation refused"):
            copied.run(observe_tool_before_execution)
        def start_descendant():
            with bind_concurrent_tool_resource(descendant) as admitted:
                return admitted
        assert copied.run(start_descendant) is False


@pytest.mark.parametrize("parent_state", ["pending", "completed", "superseded"])
def test_started_detached_child_retains_ordinary_descendant_authority(tmp_path, parent_state):
    store = _started(tmp_path)
    with bind_chat_execution(tmp_path, "key", 1):
        child = concurrent_tool_resource_for_submit("child", kind="async_delegation_thread")
        with bind_concurrent_tool_resource(child) as admitted:
            assert admitted
            _parent_state(store, parent_state)
            assert observe_tool_before_execution()
            descendant = concurrent_tool_resource_for_submit("descendant")
            with bind_concurrent_tool_resource(descendant) as admitted:
                assert admitted
                assert observe_tool_before_execution()


def test_ordinary_worker_preserves_fenced_observation_accounting(tmp_path):
    store = _started(tmp_path)
    with bind_chat_execution(tmp_path, "key", 1):
        assert store.begin_recovery("key", 1, "fence")
        assert observe_tool_before_execution()
        with pytest.raises(RuntimeError, match="reservation refused"):
            concurrent_tool_resource_for_submit("new-work")
