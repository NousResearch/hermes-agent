"""Regression for #157321: resetting a parked turn worker must not trust a module that is
still importing.

``sys.modules`` publishes a module object before its body has run, and ``tools.terminal_tool_sudo``
is first imported lazily from a turn worker thread (it defines its lock only after pulling in
``utils``). When a second worker parked during that import, ``reset_reused_thread_state`` read
``_sudo_password_cache_lock`` off the half-built module, raised ``AttributeError``, and the executor
retired a healthy worker. ``tools.interrupt`` has the same shape: ``set_interrupt`` is defined after
its own ``utils`` import.

A module that is present but still executing holds no state yet, exactly like one that was never
imported, so the reset skips it and carries on with the rest.
"""

from __future__ import annotations

import sys
import threading
import types

from gateway.turn_thread_reset import reset_reused_thread_state


def _half_imported(monkeypatch, name: str) -> None:
    """Publish an empty module under ``name``, as ``sys.modules`` does while a body is still running."""
    monkeypatch.setitem(sys.modules, name, types.ModuleType(name))


def test_a_sudo_module_that_is_still_importing_is_skipped(monkeypatch):
    _half_imported(monkeypatch, "tools.terminal_tool_sudo")

    reset_reused_thread_state()  # raised AttributeError before the fix


def test_an_interrupt_module_that_is_still_importing_is_skipped(monkeypatch):
    _half_imported(monkeypatch, "tools.interrupt")

    reset_reused_thread_state()


def test_the_rest_of_the_reset_still_runs_around_a_half_imported_module(monkeypatch):
    # A skipped module must not end the reset early: the per-turn callbacks of the previous turn
    # still have to be gone before the worker takes another session's turn.
    _half_imported(monkeypatch, "tools.terminal_tool_sudo")
    terminal = types.ModuleType("tools.terminal_tool")
    terminal._callback_tls = threading.local()
    terminal._callback_tls.approval = "turn 1's approval callback"
    monkeypatch.setitem(sys.modules, "tools.terminal_tool", terminal)

    reset_reused_thread_state()

    assert not hasattr(terminal._callback_tls, "approval")


def test_a_finished_sudo_module_still_drops_this_threads_cached_password(monkeypatch):
    import tools.terminal_tool_sudo as sudo

    tid = threading.get_ident()
    monkeypatch.setitem(sudo._sudo_password_cache, f"thread:{tid}", "turn 1's password")
    monkeypatch.setitem(sudo._sudo_password_cache, "thread:another-worker", "kept")

    reset_reused_thread_state()

    assert f"thread:{tid}" not in sudo._sudo_password_cache
    assert sudo._sudo_password_cache["thread:another-worker"] == "kept"
