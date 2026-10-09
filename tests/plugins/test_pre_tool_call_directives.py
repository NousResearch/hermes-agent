"""Policy hooks and request rewrites use separate, ordered extension points."""

from unittest.mock import patch

import pytest

from hermes_cli.middleware import apply_tool_request_middleware
from hermes_cli.plugins import resolve_pre_tool_block


@pytest.mark.parametrize("results, expected", [
    ([], None),
    ([{"action": "block", "message": "rate limited"}], "rate limited"),
    ([None, "string", 42, {"action": "observe"}], None),
    ([{"action": "block", "message": ""}], None),
])
def test_policy_hook_ignores_observers_and_preserves_blocks(results, expected):
    with patch("hermes_cli.plugins.invoke_hook", return_value=results) as invoke:
        assert resolve_pre_tool_block("terminal", {"command": "ls"}) == expected
    invoke.assert_called_once()


@pytest.mark.parametrize("results, expected", [
    ([{"args": {"command": "rtk ls"}}], {"command": "rtk ls"}),
    ([None, "string", {"args": "not-a-dict"}], {"command": "ls"}),
    ([{"args": {"command": "first", "cwd": "/tmp/a"}},
      {"args": {"command": "second", "cwd": "/tmp/a"}}],
     {"command": "second", "cwd": "/tmp/a"}),
])
def test_request_middleware_updates_arguments(results, expected):
    with patch("hermes_cli.plugins.has_middleware", return_value=True), patch(
        "hermes_cli.plugins.invoke_middleware", return_value=results
    ):
        result = apply_tool_request_middleware("terminal", {"command": "ls"})
    assert result.payload == expected


def test_policy_context_reaches_hook_once():
    with patch("hermes_cli.plugins.invoke_hook", return_value=[]) as invoke:
        assert resolve_pre_tool_block("terminal", {"command": "ls"},
            task_id="t1", session_id="s1", tool_call_id="tc1") is None
    invoke.assert_called_once()
    assert invoke.call_args.kwargs["tool_name"] == "terminal"
    assert invoke.call_args.kwargs["task_id"] == "t1"
    assert invoke.call_args.kwargs["session_id"] == "s1"
    assert invoke.call_args.kwargs["tool_call_id"] == "tc1"
