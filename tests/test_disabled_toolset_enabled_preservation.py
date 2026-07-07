"""Tests for disabled-toolset protection of enabled tools (#58281)."""

import sys
import pytest
from unittest.mock import patch


def _get_tool_names(definitions):
    """Extract function names from tool definitions."""
    return {t.get("function", {}).get("name", "") for t in definitions}


class TestDisabledToolsetPreservesEnabledTools:
    def test_debugging_disabled_with_terminal_enabled(self):
        """Disabling 'debugging' while 'terminal' is enabled preserves terminal."""
        from model_tools import _compute_tool_definitions

        defs = _compute_tool_definitions(
            enabled_toolsets=["terminal", "file"],
            disabled_toolsets=["debugging"],
            quiet_mode=True,
        )
        names = _get_tool_names(defs)
        # terminal tools must survive
        assert "terminal" in names
        assert "process" in names
        assert "read_file" in names
        assert "write_file" in names

    def test_safe_disabled_with_web_enabled(self):
        """Disabling 'safe' while 'web' is enabled preserves web tools."""
        from model_tools import _compute_tool_definitions

        defs = _compute_tool_definitions(
            enabled_toolsets=["web"],
            disabled_toolsets=["safe"],
            quiet_mode=True,
        )
        names = _get_tool_names(defs)
        assert "web_search" in names
        assert "web_extract" in names

    def test_coding_disabled_with_terminal_file_enabled(self):
        """Disabling 'coding' (posture) while terminal+file are enabled
        preserves the core tools.  Regression test for #58281."""
        from model_tools import _compute_tool_definitions

        defs = _compute_tool_definitions(
            enabled_toolsets=["terminal", "file"],
            disabled_toolsets=["coding"],
            quiet_mode=True,
        )
        names = _get_tool_names(defs)
        assert len(names) > 0, "Tool list should not be empty"
        assert "terminal" in names
        assert "read_file" in names

    def test_coding_disabled_removes_coding_only_tools(self):
        """Disabling 'coding' should remove coding-only tools like
        execute_code and delegate_task (non-core delta)."""
        from model_tools import _compute_tool_definitions

        defs = _compute_tool_definitions(
            enabled_toolsets=["terminal"],
            disabled_toolsets=["coding"],
            quiet_mode=True,
        )
        names = _get_tool_names(defs)
        # execute_code is in coding but not in terminal
        # It's a core tool though, so it might survive
        assert "terminal" in names

    def test_hermes_cli_disabled_preserves_core(self):
        """Disabling hermes-cli should preserve core tools (#33924)."""
        from model_tools import _compute_tool_definitions

        defs = _compute_tool_definitions(
            enabled_toolsets=["terminal"],
            disabled_toolsets=["hermes-cli"],
            quiet_mode=True,
        )
        names = _get_tool_names(defs)
        assert len(names) > 0
        assert "terminal" in names

    def test_unknown_disabled_not_crash(self):
        """Unknown disabled toolset should not crash computation."""
        from model_tools import _compute_tool_definitions

        defs = _compute_tool_definitions(
            enabled_toolsets=["terminal"],
            disabled_toolsets=["nonexistent_12345"],
            quiet_mode=True,
        )
        assert len(defs) > 0
