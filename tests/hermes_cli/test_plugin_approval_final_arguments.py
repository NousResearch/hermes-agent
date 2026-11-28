"""Plugin approvals must not authorize arguments changed after policy evaluation."""
from unittest.mock import patch

import pytest

from hermes_cli.plugins import PluginManager, _dispatch_pre_tool_call_hooks


APPROVE = {"action": "approve", "message": "reviewed command", "rule_key": "test"}


def dispatch(hooks, args=None):
    with patch("hermes_cli.lifecycle.invoke_hook", return_value=hooks), patch(
        "tools.approval.request_tool_approval", return_value={"approved": True}
    ) as gate:
        blocked, final = _dispatch_pre_tool_call_hooks(
            "terminal", args if args is not None else {"command": "git status"}
        )
    return blocked, final, gate.call_count


@pytest.mark.parametrize("reverse", [False, True])
def test_modified_approval_never_asks_in_either_order(reverse):
    hooks = [APPROVE, {"action": "modify", "args": {"command": "different action"}}]
    blocked, _, calls = dispatch(list(reversed(hooks)) if reverse else hooks)
    assert "BLOCKED:" in blocked
    assert calls == 0


def test_unchanged_approval_and_hard_veto_keep_their_contract():
    assert dispatch([APPROVE])[2] == 1
    blocked, _, calls = dispatch([APPROVE, {"action": "block", "message": "veto"}])
    assert blocked == "veto"
    assert calls == 0


def test_noop_modifier_keeps_approval_but_boolean_is_not_a_number():
    blocked, _, calls = dispatch([
        APPROVE, {"action": "modify", "args": {"command": "git status"}}
    ])
    assert blocked is None
    assert calls == 1
    blocked, _, calls = dispatch([
        APPROVE, {"action": "modify", "args": {"count": True}}
    ], {"count": 1})
    assert "BLOCKED:" in blocked
    assert calls == 0


def test_one_callback_cannot_mutate_another_callbacks_inputs(tmp_path):
    manager = PluginManager(scope_key=str(tmp_path))
    original = {"command": "git status", "nested": {"value": "reviewed"}}
    seen = []

    def mutate(**kwargs):
        kwargs["args"]["command"] = "different action"
        kwargs["args"]["nested"]["value"] = "different value"

    manager._hooks["pre_tool_call"] = [mutate, lambda **kw: seen.append(kw["args"])]
    with patch("hermes_cli.plugins._resolve_hook_callback_timeout", return_value=0):
        manager.invoke_hook("pre_tool_call", tool_name="terminal", args=original)
    assert seen == [{"command": "git status", "nested": {"value": "reviewed"}}]
    assert original == seen[0]


def test_modifier_without_approval_still_returns_modified_arguments():
    blocked, final, calls = dispatch([{"action": "modify", "args": {"command": "git diff"}}])
    assert blocked is None
    assert final == {"command": "git diff"}
    assert calls == 0
