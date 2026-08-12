"""Behavior tests for the built-in memory → external provider bridge.

The bridge lives behind the MemoryManager interface
(``MemoryManager.notify_memory_tool_write``): the agent loop hands over the raw
built-in memory tool result + args, and the manager decides whether/what to
mirror to external providers. These tests drive that method with a fake
external provider and assert which ``on_memory_write`` calls land.
"""

import json

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






def test_mirrors_patch_as_replace_with_full_rewritten_entry():
    """``patch`` reaches providers as a ``replace`` carrying the whole entry.

    Providers filter on the documented action names, so mirroring the literal
    "patch" would be silently dropped by every one of them — a successful patch
    would never reach an external backend at all.
    """
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({
            "success": True,
            "patched_entry": "Deploy target is prod.example.com (rebuild nightly)",
        }),
        {
            "action": "patch",
            "target": "memory",
            "pattern": r"staging\.example\.com",
            "content": "prod.example.com",
        },
    )
    assert provider.calls == [
        {
            "action": "replace",
            "target": "memory",
            "content": "Deploy target is prod.example.com (rebuild nightly)",
            "metadata": {
                "source_action": "patch",
                "pattern": r"staging\.example\.com",
            },
        }
    ]


def test_patch_without_patched_entry_falls_back_to_replacement_span():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": True}),
        {"action": "patch", "target": "user", "pattern": "dark mode", "content": "light mode"},
    )
    assert provider.calls[0]["action"] == "replace"
    assert provider.calls[0]["content"] == "light mode"
    assert provider.calls[0]["metadata"]["source_action"] == "patch"


def test_staged_patch_is_not_mirrored():
    mgr, provider = _manager_with_provider()
    mgr.notify_memory_tool_write(
        json.dumps({"success": True, "staged": True, "pending_id": "abc123"}),
        {"action": "patch", "target": "memory", "pattern": "x", "content": "y"},
    )
    assert provider.calls == []


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
