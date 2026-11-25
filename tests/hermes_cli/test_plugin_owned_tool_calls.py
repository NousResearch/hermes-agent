"""A plugin runs a tool call through Hermes's full pipeline under its own task id, releases only its
own tasks, and stops only the worker thread it opened an interrupt scope on."""

import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _ctx(name="demo"):
    return PluginContext(PluginManifest(name=name), PluginManager())


def _registry(captured):
    registry = MagicMock()

    def dispatch(name, args, **kwargs):
        captured.append((name, args, kwargs))
        return json.dumps({"ok": True})

    registry.dispatch.side_effect = dispatch
    return registry


def test_run_tool_call_gets_the_pre_tool_call_block():
    ran = []
    with patch("model_tools.registry", _registry(ran)), \
            patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=("blocked by a rule", None)):
        result = _ctx().run_tool_call("read_file", {"path": "/etc/hosts"}, task_key="s1")
    assert ran == []
    assert "blocked by a rule" in result


def test_run_tool_call_runs_under_the_plugins_own_task_id():
    ran = []
    with patch("model_tools.registry", _registry(ran)), \
            patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=(None, None)):
        _ctx().run_tool_call("read_file", {"path": "/etc/hosts"}, task_key="s1")
    assert ran[0][2]["task_id"] == "plugin:demo:s1"


def test_run_tool_call_uses_a_modify_hooks_args():
    ran = []
    with patch("model_tools.registry", _registry(ran)), \
            patch("hermes_cli.plugins._dispatch_pre_tool_call_hooks", return_value=(None, {"path": "/tmp/x"})):
        _ctx().run_tool_call("read_file", {"path": "/etc/hosts"}, task_key="s1")
    assert ran[0][1] == {"path": "/tmp/x"}


@pytest.mark.parametrize("key", ["", "a:b", None, 7])
def test_task_key_must_be_a_plain_string(key):
    with pytest.raises(ValueError):
        _ctx().run_tool_call("read_file", {}, task_key=key)


def test_release_tool_task_releases_only_the_plugins_own_task():
    from tools.terminal_tool import _active_environments, _env_lock
    mine, other = MagicMock(), MagicMock()
    with _env_lock:
        _active_environments["plugin:demo:s1"] = mine
        _active_environments["s1"] = other
        _active_environments["plugin:other:s1"] = other
    try:
        _ctx().release_tool_task("s1")
        with _env_lock:
            assert "plugin:demo:s1" not in _active_environments
            assert "s1" in _active_environments and "plugin:other:s1" in _active_environments
        mine.cleanup.assert_called_once()
        other.cleanup.assert_not_called()
    finally:
        with _env_lock:
            for key in ("plugin:demo:s1", "s1", "plugin:other:s1"):
                _active_environments.pop(key, None)


def test_interrupt_scope_stops_only_its_own_thread_while_open():
    from tools.interrupt import InterruptScope, is_interrupted
    seen, entered, go = {}, threading.Event(), threading.Event()
    scope = InterruptScope()

    def worker():
        with scope:
            entered.set()
            go.wait(5)
            seen["inside"] = is_interrupted()
        seen["after"] = is_interrupted()

    t = threading.Thread(target=worker)
    t.start()
    entered.wait(5)
    scope.interrupt()
    assert not is_interrupted()          # the caller's own thread is never touched
    go.set()
    t.join(5)
    assert seen == {"inside": True, "after": False}


def test_interrupt_scope_after_exit_does_nothing():
    from tools.interrupt import InterruptScope, is_interrupted
    scope = InterruptScope()
    with scope:
        pass
    scope.interrupt()
    assert not is_interrupted()


def test_interrupt_scope_keeps_a_stop_it_did_not_set():
    from tools.interrupt import InterruptScope, is_interrupted, set_interrupt
    try:
        with InterruptScope():
            set_interrupt(True)          # a /stop aimed at this thread lands inside the scope
        assert is_interrupted()
    finally:
        set_interrupt(False)


def test_interrupt_scope_cannot_be_reentered():
    from tools.interrupt import InterruptScope
    scope = InterruptScope()
    with scope:
        with pytest.raises(RuntimeError):
            scope.__enter__()
