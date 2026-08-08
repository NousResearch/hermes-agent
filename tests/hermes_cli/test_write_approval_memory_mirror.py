"""Behavior tests: approved memory writes must mirror to external providers.

Bug: with memory write-approval enabled, writes staged for approval landed in
MEMORY.md/USER.md when approved via /memory approve — but the external-provider
mirror (``MemoryManager.notify_memory_tool_write``) never fired, so providers
like Open Brain silently missed every approved write. The fix threads a
MemoryManager through the approval path. These tests pin that contract:
approving with a manager mirrors, approving without one still applies locally.
"""

import json

from tools import write_approval as wa
from tools.memory_tool import apply_memory_pending, load_on_disk_store
from hermes_cli.write_approval_commands import _apply_one, _approve


class _FakeManager:
    """Stands in for MemoryManager; records notify_memory_tool_write calls."""

    def __init__(self) -> None:
        self.calls = []

    def notify_memory_tool_write(self, result_json, args, build_metadata=None):
        self.calls.append({"result": json.loads(result_json), "args": dict(args)})


def _stage_add(content: str) -> str:
    rec = wa.stage_write(
        wa.MEMORY,
        {"action": "add", "target": "memory", "content": content},
        summary="test staged add",
        origin="test",
    )
    return rec["id"]


def test_apply_one_mirrors_when_manager_present():
    store = load_on_disk_store()
    pid = _stage_add("mirror-contract test fact alpha")
    rec = wa.get_pending(wa.MEMORY, pid)
    manager = _FakeManager()

    ok, _ = _apply_one(wa.MEMORY, rec, store, memory_manager=manager)

    assert ok
    assert len(manager.calls) == 1
    assert manager.calls[0]["result"]["success"] is True
    assert manager.calls[0]["args"]["content"] == "mirror-contract test fact alpha"


def test_apply_one_without_manager_still_applies():
    store = load_on_disk_store()
    pid = _stage_add("no-manager test fact beta")
    rec = wa.get_pending(wa.MEMORY, pid)

    ok, _ = _apply_one(wa.MEMORY, rec, store)

    assert ok


def test_mirror_failure_does_not_fail_the_approve():
    store = load_on_disk_store()
    pid = _stage_add("fragile-mirror test fact gamma")
    rec = wa.get_pending(wa.MEMORY, pid)

    class _ExplodingManager:
        def notify_memory_tool_write(self, *a, **k):
            raise RuntimeError("provider down")

    ok, _ = _apply_one(wa.MEMORY, rec, store, memory_manager=_ExplodingManager())

    assert ok  # local write must survive a provider-side mirror failure


def test_approve_full_path_mirrors_and_clears_pending():
    store = load_on_disk_store()
    pid = _stage_add("full-path test fact delta")
    manager = _FakeManager()

    out = _approve(wa.MEMORY, [pid], store, memory_manager=manager)

    assert len(manager.calls) == 1
    assert wa.get_pending(wa.MEMORY, pid) is None
    assert "delta" in out or "Applied" in out or "approved" in out.lower()
