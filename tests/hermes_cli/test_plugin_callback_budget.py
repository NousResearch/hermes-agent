"""A plugin callback may declare its own hook budget (``__hook_timeout__``).

``plugins.hook_callback_timeout`` is one value for every bounded hook. A ``pre_tool_call`` gate
that asks a model for a go/no-go needs more than the default 30 s, while a result transform must
keep it. The declaration extends that one callback; the threaded path itself still follows the
global value, so ``hook_callback_timeout: 0`` keeps the synchronous call and no plugin can
re-enable the worker on its own.
"""

from __future__ import annotations

import threading
import time

from hermes_cli.plugins import PluginManager


def test_a_callback_without_a_declaration_keeps_the_global_budget(monkeypatch):
    """The same callback, no declaration: the global 0.2 s still abandons it."""
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.2)
    hold = threading.Event()

    def slow(**_kwargs):
        hold.wait(1.0)
        return {"late": True}

    manager = PluginManager()
    manager._hooks["post_tool_call"] = [slow]

    started = time.monotonic()
    try:
        assert manager.invoke_hook("post_tool_call") == []
        assert time.monotonic() - started < 1.0
    finally:
        hold.set()


def test_a_declared_budget_overrides_the_global_value(monkeypatch):
    """1 s of work survives a 0.2 s global budget because the callback declared 3 s."""
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.2)
    hold = threading.Event()

    def slow(**_kwargs):
        hold.wait(1.0)
        return {"ok": True}

    slow.__hook_timeout__ = 3
    manager = PluginManager()
    manager._hooks["post_tool_call"] = [slow]

    try:
        assert manager.invoke_hook("post_tool_call") == [{"ok": True}]
    finally:
        hold.set()


def test_an_undeclared_hook_still_fails_closed_under_the_global_budget(monkeypatch):
    """``pre_tool_call`` keeps its veto: a declared budget cannot turn a timeout into a pass."""
    monkeypatch.setattr("hermes_cli.plugins._resolve_hook_callback_timeout", lambda: 0.2)
    hold = threading.Event()

    def slow(**_kwargs):
        hold.wait(1.0)

    manager = PluginManager()
    manager._hooks["pre_tool_call"] = [slow]

    try:
        results = manager.invoke_hook("pre_tool_call", tool_name="write_file", args={})
    finally:
        hold.set()
    assert results and results[0]["action"] == "block"


def test_the_budget_is_clamped_and_malformed_values_are_ignored():
    from hermes_cli import plugins_dispatch

    def callback(**_kwargs):
        return None

    assert plugins_dispatch._callback_budget(callback, 30.0) == 30.0
    callback.__hook_timeout__ = 120
    assert plugins_dispatch._callback_budget(callback, 30.0) == 120.0
    callback.__hook_timeout__ = 10_000
    assert (plugins_dispatch._callback_budget(callback, 30.0)
            == plugins_dispatch._MAX_HOOK_CALLBACK_TIMEOUT_SECS)
    for bad in (0, -1, "soon", None, object()):
        callback.__hook_timeout__ = bad
        assert plugins_dispatch._callback_budget(callback, 30.0) == 30.0
