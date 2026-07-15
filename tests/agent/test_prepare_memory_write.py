"""Tests for the ``prepare_memory_write`` pre-commit routing hook.

These tests verify the wiring contract that the tool dispatch sites
(``tool_executor.py`` and ``agent_runtime_helpers.py``) rely on:

  1. When a provider's ``prepare_memory_write`` returns ``{handled: True}``,
     the native memory store is NOT written to and the routed result is
     returned to the caller.
  2. When it returns ``None`` (the default), the native write proceeds as
     normal.
  3. When ``prepare_memory_write`` raises, the native write proceeds
     (provider bugs never block a memory write).
  4. The default ``MemoryProvider.prepare_memory_write`` returns ``None``
     so existing providers are unaffected.

The tests exercise a faithful replica of the dispatch decision logic
(rather than the full dispatch function, which requires extensive
agent/middleware construction). The replica lives in ``_run_memory_dispatch``
below and mirrors the exact control flow of both call sites.
"""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from agent.memory_provider import MemoryProvider


# ---------------------------------------------------------------------------
# Dispatch replica — mirrors the control flow in tool_executor.py and
# agent_runtime_helpers.py memory branches. Keeping it here (rather than
# importing the real function) avoids the heavy agent/middleware scaffolding
# while exercising the exact routing decisions the wiring must make.
# ---------------------------------------------------------------------------

def _run_memory_dispatch(
    next_args: dict,
    *,
    memory_manager,
    memory_store,
    build_metadata,
) -> str:
    """Faithful replica of the memory ``_execute`` routing logic."""
    target = next_args.get("target", "memory")
    operations = next_args.get("operations")

    # Pre-commit routing (mirrors tool_executor.py / agent_runtime_helpers.py)
    if memory_manager:
        for provider in memory_manager.providers:
            try:
                reroute = provider.prepare_memory_write(
                    action=next_args.get("action"),
                    target=target,
                    content=next_args.get("content") or "",
                    metadata=build_metadata(),
                    old_text=next_args.get("old_text"),
                )
            except Exception:
                continue
            if reroute and reroute.get("handled"):
                result = reroute.get("result")
                if isinstance(result, str):
                    return result
                if isinstance(result, dict):
                    return json.dumps(result, ensure_ascii=False)
                return "OK"

    # Native write
    from tools.memory_tool import memory_tool as _memory_tool
    result = _memory_tool(
        action=next_args.get("action"),
        target=target,
        content=next_args.get("content"),
        old_text=next_args.get("old_text"),
        operations=operations,
        store=memory_store,
    )
    if memory_manager:
        memory_manager.notify_memory_tool_write(
            result, next_args, build_metadata=lambda: build_metadata()
        )
    return result


# ---------------------------------------------------------------------------
# Test providers
# ---------------------------------------------------------------------------


class _RoutingProvider(MemoryProvider):
    """Provider that intercepts writes via prepare_memory_write."""

    def __init__(self, name="router"):
        self._name = name
        self.calls: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def prepare_memory_write(self, action, target, content, metadata=None, old_text=None):
        self.calls.append(
            {"action": action, "target": target, "content": content,
             "metadata": metadata, "old_text": old_text}
        )
        return self._next_response

    # test harness setter
    _next_response: Optional[Dict[str, Any]] = None


class _PassThroughProvider(MemoryProvider):
    """Provider that does NOT override prepare_memory_write (uses default)."""

    @property
    def name(self) -> str:
        return "passthrough"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []


class _RaisingProvider(MemoryProvider):
    """Provider whose prepare_memory_write raises."""

    @property
    def name(self) -> str:
        return "raiser"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id, **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def prepare_memory_write(self, *args, **kwargs):
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Default ABC behavior
# ---------------------------------------------------------------------------


class TestDefaultNoOp:
    """The default MemoryProvider.prepare_memory_write must return None."""

    def test_default_returns_none(self):
        provider = _PassThroughProvider()
        assert provider.prepare_memory_write("add", "memory", "x") is None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_store(tmp_path):
    """Build a real MemoryStore backed by a temp HERMES_HOME."""
    from tools.memory_tool import MemoryStore
    import os
    os.environ["HERMES_HOME"] = str(tmp_path / ".hermes")
    (tmp_path / ".hermes" / "memories").mkdir(parents=True, exist_ok=True)
    return MemoryStore()


@pytest.fixture
def store(tmp_path):
    return _make_store(tmp_path)


@pytest.fixture
def empty_metadata():
    return {"write_origin": "memory_tool"}


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


class TestRoutingIntercepts:
    """prepare_memory_write {handled: True} skips the native store."""

    def test_handled_true_skips_native_write(self, store, empty_metadata):
        provider = _RoutingProvider()
        provider._next_response = {"handled": True, "result": "🧠 Saved to CCA"}
        mgr = MagicMock()
        mgr.providers = [provider]

        result = _run_memory_dispatch(
            {"action": "add", "target": "memory", "content": "doctrine entry"},
            memory_manager=mgr,
            memory_store=store,
            build_metadata=lambda: empty_metadata,
        )

        # The routed result is returned to the caller
        assert result == "🧠 Saved to CCA"
        # The provider was consulted
        assert len(provider.calls) == 1
        assert provider.calls[0]["content"] == "doctrine entry"
        # The native store was NOT written to
        entries = store._entries_for("memory")
        assert all("doctrine entry" not in e for e in entries), \
            "native store should be empty — provider intercepted"
        # notify_memory_tool_write was NOT called (no native write happened)
        mgr.notify_memory_tool_write.assert_not_called()

    def test_handled_true_with_dict_result(self, store, empty_metadata):
        provider = _RoutingProvider()
        provider._next_response = {
            "handled": True,
            "result": {"success": True, "rerouted": "mempalace"},
        }
        mgr = MagicMock()
        mgr.providers = [provider]

        result = _run_memory_dispatch(
            {"action": "add", "target": "memory", "content": "x"},
            memory_manager=mgr,
            memory_store=store,
            build_metadata=lambda: empty_metadata,
        )

        assert json.loads(result) == {"success": True, "rerouted": "mempalace"}

    def test_handled_true_with_none_result_returns_ok(self, store, empty_metadata):
        provider = _RoutingProvider()
        provider._next_response = {"handled": True}  # no "result" key
        mgr = MagicMock()
        mgr.providers = [provider]

        result = _run_memory_dispatch(
            {"action": "add", "target": "memory", "content": "x"},
            memory_manager=mgr,
            memory_store=store,
            build_metadata=lambda: empty_metadata,
        )
        assert result == "OK"


class TestRoutingPassesThrough:
    """prepare_memory_write None proceeds with the native write."""

    def test_none_proceeds_to_native(self, store, empty_metadata):
        provider = _RoutingProvider()
        provider._next_response = None  # allow native write
        mgr = MagicMock()
        mgr.providers = [provider]

        result = _run_memory_dispatch(
            {"action": "add", "target": "memory", "content": "native entry"},
            memory_manager=mgr,
            memory_store=store,
            build_metadata=lambda: empty_metadata,
        )

        # A native write happened — result is JSON with success
        parsed = json.loads(result)
        assert parsed.get("success") is True
        # The native store now contains the entry
        entries = store._entries_for("memory")
        assert any("native entry" in e for e in entries)
        # notify_memory_tool_write WAS called (mirroring)
        mgr.notify_memory_tool_write.assert_called_once()

    def test_default_provider_does_not_intercept(self, store, empty_metadata):
        """A provider using the default (non-overridden) hook lets writes through."""
        provider = _PassThroughProvider()
        mgr = MagicMock()
        mgr.providers = [provider]

        result = _run_memory_dispatch(
            {"action": "add", "target": "memory", "content": "untouched"},
            memory_manager=mgr,
            memory_store=store,
            build_metadata=lambda: empty_metadata,
        )
        parsed = json.loads(result)
        assert parsed.get("success") is True
        entries = store._entries_for("memory")
        assert any("untouched" in e for e in entries)


class TestRoutingErrorHandling:
    """Provider exceptions must NOT block the native write."""

    def test_raising_provider_falls_through(self, store, empty_metadata):
        provider = _RaisingProvider()
        mgr = MagicMock()
        mgr.providers = [provider]

        result = _run_memory_dispatch(
            {"action": "add", "target": "memory", "content": "after error"},
            memory_manager=mgr,
            memory_store=store,
            build_metadata=lambda: empty_metadata,
        )
        # Despite the provider raising, the native write happened
        parsed = json.loads(result)
        assert parsed.get("success") is True
        entries = store._entries_for("memory")
        assert any("after error" in e for e in entries)


class TestRoutingOrderAndMetadata:
    """The hook receives correct metadata and action/target/content."""

    def test_metadata_passed_to_hook(self, store):
        meta = {"write_origin": "memory_tool", "session_id": "abc", "platform": "cli"}
        provider = _RoutingProvider()
        provider._next_response = None
        mgr = MagicMock()
        mgr.providers = [provider]

        _run_memory_dispatch(
            {"action": "replace", "target": "user", "content": "name: Bob",
             "old_text": "name: Alice"},
            memory_manager=mgr,
            memory_store=store,
            build_metadata=lambda: meta,
        )

        call = provider.calls[0]
        assert call["action"] == "replace"
        assert call["target"] == "user"
        assert call["content"] == "name: Bob"
        assert call["old_text"] == "name: Alice"
        assert call["metadata"] == meta

    def test_first_provider_that_handles_wins(self, store, empty_metadata):
        """If multiple providers exist, the first {handled: True} wins."""
        p1 = _RoutingProvider(name="p1")
        p1._next_response = None  # p1 passes
        p2 = _RoutingProvider(name="p2")
        p2._next_response = {"handled": True, "result": "p2 handled"}
        p3 = _RoutingProvider(name="p3")
        p3._next_response = {"handled": True, "result": "p3 handled"}
        mgr = MagicMock()
        mgr.providers = [p1, p2, p3]

        result = _run_memory_dispatch(
            {"action": "add", "target": "memory", "content": "x"},
            memory_manager=mgr,
            memory_store=store,
            build_metadata=lambda: empty_metadata,
        )
        assert result == "p2 handled"
        assert len(p1.calls) == 1
        assert len(p2.calls) == 1
        # p3 was never consulted (p2 handled it first)
        assert len(p3.calls) == 0


class TestNoMemoryManager:
    """When there's no memory manager, the native write proceeds directly."""

    def test_no_manager_proceeds_to_native(self, store, empty_metadata):
        result = _run_memory_dispatch(
            {"action": "add", "target": "memory", "content": "no mgr"},
            memory_manager=None,
            memory_store=store,
            build_metadata=lambda: empty_metadata,
        )
        parsed = json.loads(result)
        assert parsed.get("success") is True
        entries = store._entries_for("memory")
        assert any("no mgr" in e for e in entries)
