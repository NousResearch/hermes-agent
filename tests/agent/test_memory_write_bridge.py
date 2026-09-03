"""Behavior tests for the built-in memory → external provider bridge.

The bridge lives behind the MemoryManager interface
(``MemoryManager.notify_memory_tool_write``): the agent loop hands over the raw
built-in memory tool result + args, and the manager decides whether/what to
mirror to external providers. These tests drive that method with a fake
external provider and assert which ``on_memory_write`` calls land.
"""

import json
from types import SimpleNamespace

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _RecordingProvider(MemoryProvider):
    """Minimal external provider that records on_memory_write calls."""

    def __init__(self) -> None:
        self.calls = []

    @property
    def name(self) -> str:
        return "recording"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def get_tool_schemas(self):
        return []

    def shutdown(self) -> None:
        pass

    def on_memory_write(self, action, target, content, metadata=None):
        self.calls.append({
            "action": action,
            "target": target,
            "content": content,
            "metadata": dict(metadata or {}),
        })


def _manager_with_provider():
    mgr = MemoryManager()
    provider = _RecordingProvider()
    mgr.add_provider(provider)
    return mgr, provider


def test_notifies_remove_with_old_text_after_success():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": True}),
        {"action": "remove", "target": "memory", "old_text": "stale preference entry"},
    )
    assert provider.calls == [
        {
            "action": "remove",
            "target": "memory",
            "content": "",
            "metadata": {"old_text": "stale preference entry"},
        }
    ]






@pytest.mark.parametrize("tool_result", [None, [], object(), "not-json"])
def test_skips_unrecognized_tool_result_shape(tool_result):
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        tool_result,
        {"action": "add", "target": "memory", "content": "new fact"},
    )
    assert provider.calls == []






def test_build_metadata_callback_is_merged_per_op():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": True}),
        {"action": "add", "target": "memory", "content": "fact"},
        build_metadata=lambda: {"session_id": "s1", "tool_name": "memory"},
    )
    assert provider.calls == [
        {
            "action": "add",
            "target": "memory",
            "content": "fact",
            "metadata": {"session_id": "s1", "tool_name": "memory"},
        }
    ]


def test_plugin_handled_write_is_not_mirrored_again():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": True, "native_write": False}),
        {"action": "add", "target": "memory", "content": "routed fact"},
    )
    assert provider.calls == []


def test_batch_mirror_derives_one_idempotency_key_per_operation():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": True}),
        {
            "target": "memory",
            "operations": [
                {"action": "add", "content": "first"},
                {"action": "add", "content": "second"},
            ],
        },
        build_metadata=lambda: {"operation_id": "batch-1"},
    )

    assert [call["metadata"] for call in provider.calls] == [
        {
            "operation_id": "batch-1:0",
            "operation_index": 0,
            "parent_operation_id": "batch-1",
        },
        {
            "operation_id": "batch-1:1",
            "operation_index": 1,
            "parent_operation_id": "batch-1",
        },
    ]


def test_write_metadata_identifies_source_window_and_stable_operation():
    from agent.background_review import build_memory_write_metadata
    from gateway.session_context import clear_session_vars, set_session_vars

    tokens = set_session_vars(
        platform="",
        source="desktop",
        session_id="conversation-1",
        ui_session_id="window-2",
        profile="shared",
    )
    agent = SimpleNamespace(
        _memory_write_context="foreground",
        _memory_write_origin="assistant_tool",
        _parent_session_id="parent-1",
        platform="desktop",
        session_id="conversation-1",
    )
    try:
        metadata = build_memory_write_metadata(
            agent,
            task_id="task-3",
            tool_call_id="call-4",
        )
    finally:
        clear_session_vars(tokens)

    assert metadata == {
        "operation_id": "conversation-1:call-4",
        "write_origin": "assistant_tool",
        "execution_context": "foreground",
        "session_id": "conversation-1",
        "parent_session_id": "parent-1",
        "ui_session_id": "window-2",
        "platform": "desktop",
        "source": "desktop",
        "profile_name": "shared",
        "tool_name": "memory",
        "task_id": "task-3",
        "tool_call_id": "call-4",
    }
