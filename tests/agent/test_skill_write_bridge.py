"""Behavior tests for the built-in skill_manage → external provider bridge.

The bridge lives behind the MemoryManager interface
(``MemoryManager.notify_skill_tool_write``): the agent loop (and the
approval-replay path) hands over the raw skill_manage tool result + args, and
the manager decides whether/what to mirror to external providers. These tests
drive that method with a fake external provider and assert which
``on_skill_write`` calls land — in particular the action-aware content mapping
(create/edit → ``content``, patch → ``new_string``, write_file →
``file_content``) and the supporting-file ``file_path`` metadata.
"""

import json

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider


class _RecordingProvider(MemoryProvider):
    """Minimal external provider that records on_skill_write calls."""

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

    def on_skill_write(self, action, name, content, metadata=None):
        self.calls.append({
            "action": action,
            "name": name,
            "content": content,
            "metadata": dict(metadata or {}),
        })


def _manager_with_provider():
    mgr = MemoryManager()
    provider = _RecordingProvider()
    mgr.add_provider(provider)
    return mgr, provider


_SKILL_MD = "---\nname: my-skill\ndescription: d\n---\n\nBody.\n"


# ---------------------------------------------------------------------------
# Action-aware content mapping
# ---------------------------------------------------------------------------

def test_create_mirrors_full_skill_content():
    mgr, provider = _manager_with_provider()
    mgr.notify_skill_tool_write(
        json.dumps({"success": True}),
        {"action": "create", "name": "my-skill", "content": _SKILL_MD},
    )
    assert provider.calls == [
        {"action": "create", "name": "my-skill", "content": _SKILL_MD,
         "metadata": {}},
    ]


def test_edit_mirrors_full_skill_content():
    mgr, provider = _manager_with_provider()
    mgr.notify_skill_tool_write(
        json.dumps({"success": True}),
        {"action": "edit", "name": "my-skill", "content": _SKILL_MD},
    )
    assert provider.calls == [
        {"action": "edit", "name": "my-skill", "content": _SKILL_MD,
         "metadata": {}},
    ]


def test_patch_mirrors_new_string_and_old_string_metadata():
    # patch sends its payload as new_string, NOT content — the bridge must
    # not forward an empty payload.
    mgr, provider = _manager_with_provider()
    mgr.notify_skill_tool_write(
        json.dumps({"success": True}),
        {"action": "patch", "name": "my-skill",
         "old_string": "old text", "new_string": "new text"},
    )
    assert provider.calls == [
        {"action": "patch", "name": "my-skill", "content": "new text",
         "metadata": {"old_string": "old text"}},
    ]


def test_patch_of_supporting_file_preserves_file_path():
    mgr, provider = _manager_with_provider()
    mgr.notify_skill_tool_write(
        json.dumps({"success": True}),
        {"action": "patch", "name": "my-skill", "file_path": "references/api.md",
         "old_string": "v1", "new_string": "v2"},
    )
    assert provider.calls == [
        {"action": "patch", "name": "my-skill", "content": "v2",
         "metadata": {"file_path": "references/api.md", "old_string": "v1"}},
    ]


def test_write_file_mirrors_file_content_and_file_path():
    # write_file sends its payload as file_content, NOT content.
    mgr, provider = _manager_with_provider()
    mgr.notify_skill_tool_write(
        json.dumps({"success": True}),
        {"action": "write_file", "name": "my-skill",
         "file_path": "references/guide.md", "file_content": "# Guide\n"},
    )
    assert provider.calls == [
        {"action": "write_file", "name": "my-skill", "content": "# Guide\n",
         "metadata": {"file_path": "references/guide.md"}},
    ]


def test_remove_file_has_empty_content_but_keeps_file_path():
    mgr, provider = _manager_with_provider()
    mgr.notify_skill_tool_write(
        json.dumps({"success": True}),
        {"action": "remove_file", "name": "my-skill",
         "file_path": "references/stale.md"},
    )
    assert provider.calls == [
        {"action": "remove_file", "name": "my-skill", "content": "",
         "metadata": {"file_path": "references/stale.md"}},
    ]


def test_delete_has_empty_content():
    mgr, provider = _manager_with_provider()
    mgr.notify_skill_tool_write(
        json.dumps({"success": True}),
        {"action": "delete", "name": "my-skill"},
    )
    assert provider.calls == [
        {"action": "delete", "name": "my-skill", "content": "", "metadata": {}},
    ]


# ---------------------------------------------------------------------------
# Committed-write gating
# ---------------------------------------------------------------------------

def test_skips_failed_skill_write():
    mgr, provider = _manager_with_provider()
    mgr.notify_skill_tool_write(
        json.dumps({"success": False, "error": "Skill not found"}),
        {"action": "edit", "name": "my-skill", "content": _SKILL_MD},
    )
    assert provider.calls == []


def test_skips_staged_skill_write():
    # A write staged for approval has not committed — providers must not be
    # told about it. (The approval-replay path re-notifies once it commits.)
    mgr, provider = _manager_with_provider()
    mgr.notify_skill_tool_write(
        json.dumps({"success": True, "staged": True, "pending_id": "abc123"}),
        {"action": "create", "name": "my-skill", "content": _SKILL_MD},
    )
    assert provider.calls == []


@pytest.mark.parametrize("tool_result", [None, [], object(), "not-json"])
def test_skips_unrecognized_tool_result_shape(tool_result):
    mgr, provider = _manager_with_provider()
    mgr.notify_skill_tool_write(
        tool_result,
        {"action": "create", "name": "my-skill", "content": _SKILL_MD},
    )
    assert provider.calls == []


def test_non_mutating_actions_are_not_mirrored():
    mgr, provider = _manager_with_provider()
    for action in ("read", "list", "search", ""):
        mgr.notify_skill_tool_write(
            json.dumps({"success": True}),
            {"action": action, "name": "my-skill"},
        )
    assert provider.calls == []


# ---------------------------------------------------------------------------
# Metadata plumbing
# ---------------------------------------------------------------------------

def test_build_metadata_callback_is_merged_with_bridge_metadata():
    mgr, provider = _manager_with_provider()
    mgr.notify_skill_tool_write(
        json.dumps({"success": True}),
        {"action": "write_file", "name": "my-skill",
         "file_path": "scripts/run.py", "file_content": "print('hi')\n"},
        build_metadata=lambda: {"session_id": "s1", "tool_name": "skill_manage"},
    )
    assert provider.calls == [
        {
            "action": "write_file",
            "name": "my-skill",
            "content": "print('hi')\n",
            "metadata": {
                "session_id": "s1",
                "tool_name": "skill_manage",
                "file_path": "scripts/run.py",
            },
        }
    ]


def test_provider_exception_does_not_raise():
    mgr, _ = _manager_with_provider()

    class _Exploding(_RecordingProvider):
        @property
        def name(self):
            return "exploding"

        def on_skill_write(self, action, name, content, metadata=None):
            raise RuntimeError("backend down")

    mgr2 = MemoryManager()
    mgr2.add_provider(_Exploding())
    # Must not raise — provider failures never break the tool loop.
    mgr2.notify_skill_tool_write(
        json.dumps({"success": True}),
        {"action": "create", "name": "my-skill", "content": _SKILL_MD},
    )
