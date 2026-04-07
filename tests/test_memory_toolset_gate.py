"""Tests for memory tool injection respecting platform_toolsets (issue #5544).

When a platform is configured with an explicit toolset list that omits
"memory" (e.g. ``platform_toolsets: telegram: []``), memory provider tool
schemas must NOT be injected into the agent's tool surface.

Without the fix, the injection in AIAgent.__init__ runs unconditionally,
overriding the platform toolset configuration and adding the full
fact_store tool surface to every session on every platform.  On local
models this causes a 10x latency penalty and tool-call loops.

The gate condition:
  enabled_toolsets is None   → no filter active, inject (backward-compat)
  "memory" in enabled_toolsets → memory explicitly enabled, inject
  "memory" not in enabled_toolsets (including []) → do NOT inject
"""

import sys
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_memory_manager(schema_names=("fact_store",)):
    """Return a minimal MemoryManager mock with get_all_tool_schemas()."""
    schemas = [{"name": n, "description": f"{n} tool"} for n in schema_names]
    mm = MagicMock()
    mm.get_all_tool_schemas.return_value = schemas
    return mm


def _run_injection(enabled_toolsets, memory_manager):
    """Simulate the memory-tool injection block from AIAgent.__init__."""
    tools = []
    valid_tool_names = set()

    if memory_manager and tools is not None:
        if enabled_toolsets is None or "memory" in enabled_toolsets:
            for _schema in memory_manager.get_all_tool_schemas():
                _wrapped = {"type": "function", "function": _schema}
                tools.append(_wrapped)
                _tname = _schema.get("name", "")
                if _tname:
                    valid_tool_names.add(_tname)

    return tools, valid_tool_names


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMemoryToolInjectionGate:
    def test_no_toolset_filter_injects_memory_tools(self):
        """enabled_toolsets=None means no filtering — memory tools must be added."""
        mm = _make_memory_manager()
        tools, names = _run_injection(enabled_toolsets=None, memory_manager=mm)
        assert len(tools) == 1
        assert "fact_store" in names

    def test_memory_in_toolsets_injects_tools(self):
        """Memory tools are injected when 'memory' is in the enabled list."""
        mm = _make_memory_manager()
        tools, names = _run_injection(enabled_toolsets=["terminal", "memory", "web"], memory_manager=mm)
        assert len(tools) == 1
        assert "fact_store" in names

    def test_empty_toolsets_blocks_injection(self):
        """platform_toolsets: telegram: [] — no tools including memory."""
        mm = _make_memory_manager()
        tools, names = _run_injection(enabled_toolsets=[], memory_manager=mm)
        assert tools == []
        assert names == set()

    def test_toolsets_without_memory_blocks_injection(self):
        """Toolset list that doesn't include 'memory' must suppress injection."""
        mm = _make_memory_manager()
        tools, names = _run_injection(enabled_toolsets=["terminal", "web"], memory_manager=mm)
        assert tools == []
        assert names == set()

    def test_no_memory_manager_no_injection(self):
        """Injection is skipped when no memory manager is present."""
        tools, names = _run_injection(enabled_toolsets=None, memory_manager=None)
        assert tools == []
        assert names == set()

    def test_multiple_schemas_all_injected_when_enabled(self):
        """All schemas from the memory manager are injected when enabled."""
        mm = _make_memory_manager(("fact_store", "memory_search", "memory_add"))
        tools, names = _run_injection(enabled_toolsets=None, memory_manager=mm)
        assert len(tools) == 3
        assert names == {"fact_store", "memory_search", "memory_add"}

    def test_multiple_schemas_all_blocked_when_excluded(self):
        """All schemas are blocked together — partial injection is not allowed."""
        mm = _make_memory_manager(("fact_store", "memory_search", "memory_add"))
        tools, names = _run_injection(enabled_toolsets=["terminal"], memory_manager=mm)
        assert tools == []
        assert names == set()
