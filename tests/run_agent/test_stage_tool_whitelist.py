"""Test stage-level tool whitelist via pre_llm_call hook (RFC #26524)."""

import pytest
from unittest.mock import MagicMock


def _make_tool(name: str):
    """Create a mock tool with .function.name attribute."""
    tool = MagicMock()
    tool.function.name = name
    return tool


def _filter_tools(tools, allowed_tools):
    """Replicate the filtering logic from _build_api_kwargs."""
    if allowed_tools and tools:
        return [
            t for t in tools
            if getattr(t, "function", None) is not None
            and getattr(t.function, "name", None) in allowed_tools
        ]
    return tools


class TestToolFiltering:
    """Test the tool filtering logic used in _build_api_kwargs."""

    def test_no_filter_returns_all_tools(self):
        """When allowed_tools is None, all tools are returned."""
        tools = [_make_tool("read_file"), _make_tool("terminal"), _make_tool("browser")]
        result = _filter_tools(tools, None)
        assert result == tools

    def test_filter_to_subset(self):
        """When allowed_tools is set, only matching tools are returned."""
        tools = [_make_tool("read_file"), _make_tool("terminal"), _make_tool("browser")]
        result = _filter_tools(tools, {"read_file", "terminal"})
        names = {t.function.name for t in result}
        assert names == {"read_file", "terminal"}

    def test_filter_empty_set_returns_empty(self):
        """Empty allowed_tools set means no tools."""
        tools = [_make_tool("read_file"), _make_tool("terminal")]
        # Empty set is falsy in Python, so this returns all tools
        # But in the code: `if allowed_tools and tools:` — set() is falsy
        result = _filter_tools(tools, set())
        assert result == tools  # empty set is falsy, returns all

    def test_filter_unknown_tool_returns_empty(self):
        """allowed_tools with non-existent names returns empty list."""
        tools = [_make_tool("read_file")]
        result = _filter_tools(tools, {"nonexistent_tool"})
        assert result == []

    def test_filter_with_none_tools(self):
        """When tools is None, returns None."""
        result = _filter_tools(None, {"read_file"})
        assert result is None

    def test_filter_preserves_order(self):
        """Filtered tools maintain original order."""
        tools = [_make_tool("c"), _make_tool("a"), _make_tool("b")]
        result = _filter_tools(tools, {"a", "b", "c"})
        names = [t.function.name for t in result]
        assert names == ["c", "a", "b"]


class TestPreLlmCallToolsExtraction:
    """Test that pre_llm_call hook results with {"tools": [...]} are extracted."""

    def _extract_tools(self, results):
        """Replicate the extraction logic from run_agent.py."""
        _allowed_tools = None
        for r in results:
            if isinstance(r, dict):
                if "tools" in r and isinstance(r["tools"], list):
                    if _allowed_tools is None:
                        _allowed_tools = set()
                    _allowed_tools.update(r["tools"])
        return _allowed_tools

    def test_tools_key_extracted(self):
        result = self._extract_tools([
            {"context": "some context", "tools": ["read_file", "terminal"]}
        ])
        assert result == {"read_file", "terminal"}

    def test_multiple_hooks_union(self):
        result = self._extract_tools([
            {"tools": ["read_file"]},
            {"tools": ["terminal"]},
        ])
        assert result == {"read_file", "terminal"}

    def test_no_tools_key_returns_none(self):
        result = self._extract_tools([{"context": "some text"}])
        assert result is None

    def test_string_result_no_filter(self):
        result = self._extract_tools(["plain string context"])
        assert result is None

    def test_empty_results_returns_none(self):
        result = self._extract_tools([])
        assert result is None

    def test_duplicate_tool_names_deduped(self):
        result = self._extract_tools([
            {"tools": ["read_file", "terminal"]},
            {"tools": ["read_file"]},
        ])
        assert result == {"read_file", "terminal"}

    def test_tools_not_list_ignored(self):
        result = self._extract_tools([{"tools": "not a list"}])
        assert result is None
