"""Tests for pre_tool_call hook blocking behavior.

Verifies that pre_tool_call hooks can return {"block": True, "reason": "..."}
to abort tool execution before it happens.
"""

import json
import pytest
from unittest.mock import patch, MagicMock


def _make_blocking_hook(tool_to_block: str, reason: str = "Blocked by test"):
    """Create a hook that blocks a specific tool."""
    def hook(tool_name: str, args: str, task_id: str, **kwargs):
        if tool_name == tool_to_block:
            return {"block": True, "reason": reason}
        return None
    return hook


def _make_passthrough_hook():
    """Create a hook that always allows."""
    def hook(**kwargs):
        return None
    return hook


class TestPreToolCallBlocking:
    """Test that pre_tool_call hooks can block tool execution."""

    def test_blocking_hook_returns_error(self):
        """A hook returning {block: True} should prevent tool execution."""
        from hermes_cli.plugins import PluginManager

        pm = PluginManager()
        hook = _make_blocking_hook("dangerous_tool", "Policy violation")
        pm._hooks["pre_tool_call"] = [hook]

        results = pm.invoke_hook(
            "pre_tool_call",
            tool_name="dangerous_tool",
            args="{}",
            task_id="test",
        )

        assert len(results) == 1
        assert results[0]["block"] is True
        assert "Policy violation" in results[0]["reason"]

    def test_non_blocking_hook_returns_empty(self):
        """A hook returning None should not block."""
        from hermes_cli.plugins import PluginManager

        pm = PluginManager()
        hook = _make_passthrough_hook()
        pm._hooks["pre_tool_call"] = [hook]

        results = pm.invoke_hook(
            "pre_tool_call",
            tool_name="safe_tool",
            args="{}",
            task_id="test",
        )

        assert len(results) == 0  # None returns are filtered

    def test_blocking_hook_only_blocks_target(self):
        """A hook should only block the specific tool it targets."""
        from hermes_cli.plugins import PluginManager

        pm = PluginManager()
        hook = _make_blocking_hook("dangerous_tool")
        pm._hooks["pre_tool_call"] = [hook]

        # Should block
        results_blocked = pm.invoke_hook(
            "pre_tool_call",
            tool_name="dangerous_tool",
            args="{}",
            task_id="test",
        )
        assert len(results_blocked) == 1
        assert results_blocked[0]["block"] is True

        # Should allow
        results_allowed = pm.invoke_hook(
            "pre_tool_call",
            tool_name="safe_tool",
            args="{}",
            task_id="test",
        )
        assert len(results_allowed) == 0

    def test_multiple_hooks_first_block_wins(self):
        """If multiple hooks exist, any block should prevent execution."""
        from hermes_cli.plugins import PluginManager

        pm = PluginManager()
        pm._hooks["pre_tool_call"] = [
            _make_passthrough_hook(),
            _make_blocking_hook("target", "Hook 2 blocked"),
            _make_passthrough_hook(),
        ]

        results = pm.invoke_hook(
            "pre_tool_call",
            tool_name="target",
            args="{}",
            task_id="test",
        )

        # Should have one block result
        block_results = [r for r in results if isinstance(r, dict) and r.get("block")]
        assert len(block_results) == 1
        assert "Hook 2 blocked" in block_results[0]["reason"]

    def test_hook_exception_does_not_block(self):
        """A hook that raises should not block (fail-open for non-policy hooks)."""
        from hermes_cli.plugins import PluginManager

        pm = PluginManager()

        def bad_hook(**kwargs):
            raise RuntimeError("Hook crashed")

        pm._hooks["pre_tool_call"] = [bad_hook]

        results = pm.invoke_hook(
            "pre_tool_call",
            tool_name="any_tool",
            args="{}",
            task_id="test",
        )

        # Exception is caught, no block result
        assert len(results) == 0
