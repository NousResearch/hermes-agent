"""Tests for get_pre_tool_call_directives — single-fire block + rewrite."""

from unittest.mock import patch

import pytest

from hermes_cli.plugins import get_pre_tool_call_directives


def _make_invoke(results, expected_hook_name="pre_tool_call"):
    """Return a mock invoke_hook that yields *results*."""
    def _invoke(hook_name, **kwargs):
        assert hook_name == expected_hook_name
        return results
    return _invoke


class TestGetPreToolCallDirectives:
    def test_no_directives_returns_none_none(self):
        with patch("hermes_cli.plugins.invoke_hook", _make_invoke([])):
            block, rewrite = get_pre_tool_call_directives("terminal", {"command": "ls"})
        assert block is None
        assert rewrite is None

    def test_block_directive_returned(self):
        result = {"action": "block", "message": "rate limited"}
        with patch("hermes_cli.plugins.invoke_hook", _make_invoke([result])):
            block, rewrite = get_pre_tool_call_directives("terminal", {"command": "ls"})
        assert block == "rate limited"
        assert rewrite is None

    def test_rewrite_directive_returned(self):
        result = {"action": "rewrite", "args": {"command": "rtk ls"}}
        with patch("hermes_cli.plugins.invoke_hook", _make_invoke([result])):
            block, rewrite = get_pre_tool_call_directives("terminal", {"command": "ls"})
        assert block is None
        assert rewrite == {"command": "rtk ls"}

    def test_block_takes_precedence_over_rewrite(self):
        results = [
            {"action": "block", "message": "blocked"},
            {"action": "rewrite", "args": {"command": "rtk ls"}},
        ]
        with patch("hermes_cli.plugins.invoke_hook", _make_invoke(results)):
            block, rewrite = get_pre_tool_call_directives("terminal", {"command": "ls"})
        assert block == "blocked"
        assert rewrite == {"command": "rtk ls"}

    def test_later_rewrites_override_earlier_keys(self):
        results = [
            {"action": "rewrite", "args": {"command": "first", "cwd": "/tmp/a"}},
            {"action": "rewrite", "args": {"command": "second"}},
        ]
        with patch("hermes_cli.plugins.invoke_hook", _make_invoke(results)):
            _, rewrite = get_pre_tool_call_directives("terminal", {"command": "ls"})
        assert rewrite == {"command": "second", "cwd": "/tmp/a"}

    def test_observer_only_hook_ignored(self):
        results = [None, "string", 42, {"action": "observe"}]
        with patch("hermes_cli.plugins.invoke_hook", _make_invoke(results)):
            block, rewrite = get_pre_tool_call_directives("terminal", {"command": "ls"})
        assert block is None
        assert rewrite is None

    def test_block_with_empty_message_ignored(self):
        result = {"action": "block", "message": ""}
        with patch("hermes_cli.plugins.invoke_hook", _make_invoke([result])):
            block, rewrite = get_pre_tool_call_directives("terminal", {"command": "ls"})
        assert block is None

    def test_rewrite_with_non_dict_args_ignored(self):
        result = {"action": "rewrite", "args": "not-a-dict"}
        with patch("hermes_cli.plugins.invoke_hook", _make_invoke([result])):
            _, rewrite = get_pre_tool_call_directives("terminal", {"command": "ls"})
        assert rewrite is None

    def test_passes_kwargs_to_invoke_hook(self):
        captured = {}
        def _capture(hook_name, **kwargs):
            captured.update(kwargs)
            return []
        with patch("hermes_cli.plugins.invoke_hook", _capture):
            get_pre_tool_call_directives(
                "terminal", {"command": "ls"},
                task_id="t1", session_id="s1", tool_call_id="tc1",
            )
        assert captured["tool_name"] == "terminal"
        assert captured["task_id"] == "t1"
        assert captured["session_id"] == "s1"
        assert captured["tool_call_id"] == "tc1"
