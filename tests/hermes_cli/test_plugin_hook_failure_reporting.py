"""A plugin hook that fails identically on every call is reported once, not once per call.

A callback whose signature names a parameter the hook never sends (``tool_data`` instead of
``tool_name``/``args``) raises the same ``TypeError`` on every tool call; before the fix core
logged a WARNING each time — ~1700 lines an hour in the report that motivated this (#111922).
"""

import logging

import pytest

from hermes_cli.plugins import PluginManager


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    return PluginManager()


def test_identical_hook_failure_warns_once_then_debug(manager, caplog):
    def on_pre_tool(tool_data):  # core sends tool_name/args, never tool_data
        return None

    manager._hooks.setdefault("pre_tool_call", []).append(on_pre_tool)
    with caplog.at_level(logging.DEBUG, logger="hermes_cli.plugins"):
        for i in range(5):
            manager.invoke_hook("pre_tool_call", tool_name="read_file", args={"path": f"/p{i}"})

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "on_pre_tool" in r.getMessage()]
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG and "on_pre_tool" in r.getMessage()]
    assert len(warnings) == 1
    assert len(debugs) == 4
    # The one warning tells the author what the hook actually provides.
    assert "tool_data" in warnings[0].getMessage()
    assert "tool_name" in warnings[0].getMessage()


def test_distinct_hook_failures_each_warn(manager, caplog):
    """Deduplication is per distinct error: a callback failing in a new way still warns."""
    calls = []

    def flaky(**kwargs):
        calls.append(1)
        raise RuntimeError(f"failure #{len(calls)}")

    manager._hooks.setdefault("post_tool_call", []).append(flaky)
    with caplog.at_level(logging.DEBUG, logger="hermes_cli.plugins"):
        for _ in range(3):
            manager.invoke_hook("post_tool_call", tool_name="x", args={}, result="ok")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "flaky" in r.getMessage()]
    assert len(warnings) == 3


def test_middleware_failure_warns_once_and_unload_forgets_it(manager, caplog):
    """Middleware runs once per tool call like a hook, so it dedupes the same way; a plugin
    reload (unload-all) forgets the reported failures so the reloaded callback's first failure
    warns again."""
    def on_exec(tool_data):  # core sends tool_name/args, never tool_data
        return None

    manager._middleware.setdefault("agent_tool_execution", []).append(on_exec)
    with caplog.at_level(logging.DEBUG, logger="hermes_cli.plugins"):
        for i in range(3):
            manager.invoke_middleware("agent_tool_execution", tool_name="x", args={"path": f"/p{i}"})
        manager._reset_after_unload_all([])
        assert not manager._hook_failures_reported
        manager._middleware.setdefault("agent_tool_execution", []).append(on_exec)
        manager.invoke_middleware("agent_tool_execution", tool_name="x", args={})

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "on_exec" in r.getMessage()]
    assert len(warnings) == 2
    assert "Middleware 'agent_tool_execution'" in warnings[0].getMessage()
