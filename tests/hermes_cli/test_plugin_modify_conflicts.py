"""Regression for #147587: conflicting argument rewrites must be visible without logging values."""

import logging

import pytest

from hermes_cli import plugins


def _run_hooks(monkeypatch, callbacks, args):
    manager = plugins.PluginManager()
    manager._discovered = True
    manager._hooks["pre_tool_call"] = callbacks
    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
    return plugins._get_pre_tool_call_directive_details("terminal", args)


def test_conflicting_rewrites_warn_without_values_or_changing_original_payload(monkeypatch, caplog):
    original = {"command": "private-original", "cwd": "/work"}
    seen = []

    def wrapper(args, **_):
        seen.append(dict(args))
        return {"action": "modify", "args": {"command": "private-wrapper"}}

    def guard(args, **_):
        seen.append(dict(args))
        return {"action": "modify", "args": {"command": "private-guard"}}

    with caplog.at_level(logging.WARNING):
        result = _run_hooks(monkeypatch, [wrapper, guard], original)
    assert result.modified_args == {"command": "private-guard", "cwd": "/work"}
    assert seen == [original, original]
    assert original["command"] == "private-original"
    conflicts = [r.getMessage() for r in caplog.records if "modify" in r.getMessage() and "conflict" in r.getMessage()]
    assert len(conflicts) == 1
    assert "terminal" in conflicts[0] and "command" in conflicts[0]
    assert "private-" not in conflicts[0]


@pytest.mark.parametrize("second", [{"command": "rewritten"}, {"cwd": "/safe"}])
def test_identical_and_disjoint_rewrites_do_not_warn(monkeypatch, caplog, second):
    callbacks = [lambda **_: {"action": "modify", "args": {"command": "rewritten"}},
                 lambda **_: {"action": "modify", "args": second}]
    with caplog.at_level(logging.WARNING):
        result = _run_hooks(monkeypatch, callbacks, {"command": "original", "cwd": "/work"})
    assert result.modified_args == {"command": "rewritten", "cwd": second.get("cwd", "/work")}
    assert not [r for r in caplog.records if "modify" in r.getMessage() and "conflict" in r.getMessage()]
