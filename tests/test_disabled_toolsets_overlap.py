"""Tests for _compute_tool_definitions disabled_toolsets overlap protection.

Regression tests for issue #58281: disabling a composite toolset (e.g.
'coding') must not strip tools belonging to explicitly-enabled toolsets
(e.g. 'terminal', 'file').
"""

import pytest

import model_tools


class TestDisabledToolsetsOverlapProtection:
    """Prevent disabled_toolsets from stripping tools belonging to enabled toolsets."""

    @pytest.fixture(autouse=True)
    def clear_cache(self):
        """Clear the tool definitions cache before each test."""
        model_tools._tool_defs_cache.clear()
        yield
        model_tools._tool_defs_cache.clear()

    # ---- The core bug: disabling a superset that overlaps enabled toolsets ----

    def test_disable_coding_does_not_strip_enabled_terminal(self):
        """Disabling 'coding' must not remove 'terminal' tools from terminal+file."""
        tools = model_tools._compute_tool_definitions(
            enabled_toolsets=["terminal", "file"],
            disabled_toolsets=["coding"],
            quiet_mode=True,
        )
        tool_names = {t["function"]["name"] for t in tools}
        # These tools come from the terminal toolset
        assert "terminal" in tool_names
        assert "process" in tool_names

    def test_disable_coding_does_not_strip_enabled_file(self):
        """Disabling 'coding' must not remove 'file' tools from terminal+file."""
        tools = model_tools._compute_tool_definitions(
            enabled_toolsets=["terminal", "file"],
            disabled_toolsets=["coding"],
            quiet_mode=True,
        )
        tool_names = {t["function"]["name"] for t in tools}
        # These tools come from the file toolset
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "patch" in tool_names
        assert "search_files" in tool_names

    def test_disable_coding_strips_coding_only_tools(self):
        """Disabling 'coding' should still remove tools unique to coding."""
        tools = model_tools._compute_tool_definitions(
            enabled_toolsets=["terminal", "file"],
            disabled_toolsets=["coding"],
            quiet_mode=True,
        )
        tool_names = {t["function"]["name"] for t in tools}
        # These are only in 'coding', not in terminal/file
        assert "web_search" not in tool_names
        assert "browser_navigate" not in tool_names
        assert "vision_analyze" not in tool_names

    def test_disable_coding_non_empty_result(self):
        """The most critical assertion: the model must not receive zero tools."""
        tools = model_tools._compute_tool_definitions(
            enabled_toolsets=["terminal", "file"],
            disabled_toolsets=["coding"],
            quiet_mode=True,
        )
        assert len(tools) > 0, (
            "Model receives 0 tool definitions when terminal+file is enabled "
            "and coding is disabled — #58281"
        )

    # ---- Other composite toolsets must also be protected ----

    def test_disable_safe_does_not_strip_terminal(self):
        """Disabling 'safe' must not remove 'terminal' tools."""
        tools = model_tools._compute_tool_definitions(
            enabled_toolsets=["terminal"],
            disabled_toolsets=["safe"],
            quiet_mode=True,
        )
        tool_names = {t["function"]["name"] for t in tools}
        assert "terminal" in tool_names
        assert "process" in tool_names

    def test_disable_debugging_does_not_strip_terminal(self):
        """Disabling 'debugging' must not remove 'terminal' tools."""
        tools = model_tools._compute_tool_definitions(
            enabled_toolsets=["terminal"],
            disabled_toolsets=["debugging"],
            quiet_mode=True,
        )
        tool_names = {t["function"]["name"] for t in tools}
        assert "terminal" in tool_names

    # ---- Normal disabled_toolsets behavior still works ----

    def test_disable_non_overlapping_still_removes_tools(self):
        """Disabling a toolset should still remove it when it overlaps."""
        tools = model_tools._compute_tool_definitions(
            enabled_toolsets=["browser", "terminal", "file"],
            disabled_toolsets=["web_search"],
            quiet_mode=True,
        )
        tool_names = {t["function"]["name"] for t in tools}
        # web_search is in the browser toolset but should be removed
        assert "web_search" not in tool_names
        # browser tools should still be present (not stripped)
        assert "browser_navigate" in tool_names

    # ---- Legacy toolset protection ----

    def test_legacy_toolset_protection(self):
        """Legacy disabled toolsets should also respect enabled toolsets."""
        from model_tools import _LEGACY_TOOLSET_MAP

        if not _LEGACY_TOOLSET_MAP:
            pytest.skip("No legacy toolset maps to test against")

        # Just ensure no exception is raised
        tools = model_tools._compute_tool_definitions(
            enabled_toolsets=["terminal", "file"],
            disabled_toolsets=list(_LEGACY_TOOLSET_MAP.keys())[:1],
            quiet_mode=True,
        )
        assert len(tools) >= 2  # At least terminal+file tools

    # ---- When no enabled_toolsets are specified, all tools remain ----

    def test_disable_with_default_enabled_toolsets_still_works(self):
        """When enabled_toolsets is None (all enabled), disabling should work normally."""
        tools = model_tools._compute_tool_definitions(
            disabled_toolsets=["tts"],
            quiet_mode=True,
        )
        tool_names = {t["function"]["name"] for t in tools}
        assert "text_to_speech" not in tool_names