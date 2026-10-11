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


def test_execution_chain_middleware_failure_warns_once(manager, caplog, monkeypatch):
    """The execution chain (``tool_execution``/``llm_execution``, one frame per tool or LLM call)
    reports a mis-declared callback through the same warn-once path as hooks — and still skips the
    frame and runs the tool."""
    from hermes_cli import middleware as mw

    monkeypatch.setattr("hermes_cli.plugins._plugin_manager", manager)

    def on_exec(tool_data, next_call):  # core sends tool_name/args, never tool_data
        return next_call()

    manager._middleware.setdefault(mw.TOOL_EXECUTION_MIDDLEWARE, []).append(on_exec)
    with caplog.at_level(logging.DEBUG):
        results = [
            mw.run_tool_execution_middleware("read_file", {"path": f"/p{i}"}, lambda args: "ran")
            for i in range(4)
        ]

    assert results == ["ran"] * 4
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "on_exec" in r.getMessage()]
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG and "on_exec" in r.getMessage()]
    assert len(warnings) == 1
    assert len(debugs) == 3
    assert "Middleware 'tool_execution'" in warnings[0].getMessage()


def test_stream_observer_hook_failure_warns_once(manager, caplog, monkeypatch):
    """Stream observer hooks fire once per streaming delta (far more often than per tool call);
    the per-consumer worker reports a mis-declared callback through the same warn-once path."""
    from agent import plugin_stream_hooks as psh

    monkeypatch.setattr("hermes_cli.plugins._plugin_manager", manager)

    def on_stream_delta(tool_data, **kwargs):  # core sends delta, never tool_data
        return None

    monkeypatch.setattr(psh, "_registered_callbacks", lambda name: (on_stream_delta,))
    psh.shutdown_plugin_stream_hook_dispatcher()
    try:
        with caplog.at_level(logging.DEBUG):
            for i in range(20):
                assert psh.enqueue_plugin_stream_hook("on_stream_delta", delta=f"d{i}")
            for dispatcher in psh._dispatchers_for("on_stream_delta"):
                dispatcher.events.join()
    finally:
        psh.shutdown_plugin_stream_hook_dispatcher()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "on_stream_delta" in r.getMessage()]
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG and "on_stream_delta" in r.getMessage()]
    assert len(warnings) == 1
    assert len(debugs) == 19
    assert "Hook 'on_stream_delta'" in warnings[0].getMessage()
    assert "delta" in warnings[0].getMessage()


def test_event_subscriber_failure_warns_once(manager, caplog):
    """Plugin event subscribers deliver on the host worker; one that raises identically on every
    emit is reported once with the ``Event`` surface label."""
    manager._discovered = True

    def on_event(tool_data, **kwargs):  # the emitter sends its own payload, never tool_data
        return None

    manager._subscribe_event("listener", "emitter:tick", on_event)
    with caplog.at_level(logging.DEBUG, logger="hermes_cli.plugins"):
        for i in range(4):
            assert manager._dispatch_event("emitter:tick", {"n": i}) == 1
        assert manager._wait_for_event_dispatch(timeout=2.0)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING and "on_event" in r.getMessage()]
    debugs = [r for r in caplog.records if r.levelno == logging.DEBUG and "on_event" in r.getMessage()]
    assert len(warnings) == 1
    assert len(debugs) == 3
    assert "Event 'emitter:tick'" in warnings[0].getMessage()
