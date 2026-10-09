"""Phase 4.5 Step 6 dispatch-runtime state ownership gates."""

from __future__ import annotations

import ast
import queue
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANAGER = ROOT / "plugin_runtime" / "manager.py"

DISPATCH_STATE = {
    "_subscriptions",
    "_event_lock",
    "_event_idle",
    "_event_generation",
    "_event_pending_by_generation",
    "_event_queue",
    "_event_worker",
    "_emit_depth",
    "_hook_running_callbacks",
    "_hook_abandoned",
    "_hook_timeout_suppressed_until",
    "_hook_timeout_lock",
    "_hook_timeout_suppression_seconds",
    "_hook_failures_reported",
}


def test_runtime_dispatch_initializer_owns_internal_state():
    import plugin_runtime.dispatch as dispatch

    assert "_init_dispatch_runtime_state" in dispatch.PluginDispatchMixin.__dict__


def test_plugin_manager_composes_dispatch_state_without_redeclaring_it():
    tree = ast.parse(MANAGER.read_text(encoding="utf-8"), filename=str(MANAGER))
    manager = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PluginManager"
    )
    init = next(
        node for node in manager.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    calls = [
        node for node in ast.walk(init)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "self"
        and node.func.attr == "_init_dispatch_runtime_state"
    ]
    assigned = {
        target.attr
        for node in ast.walk(init)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            [*node.targets] if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    }

    assert len(calls) == 1
    assert DISPATCH_STATE.isdisjoint(assigned)


def test_runtime_initializer_preserves_dispatch_state_defaults(tmp_path):
    import plugin_runtime.dispatch as dispatch
    from plugin_runtime.manager import PluginManager

    manager = PluginManager(scope_key=str(tmp_path))

    assert manager._subscriptions == {}
    assert isinstance(manager._event_lock, type(threading.RLock()))
    assert manager._event_idle._lock is manager._event_lock
    assert manager._event_generation == 0
    assert manager._event_pending_by_generation == {0: 0}
    assert isinstance(manager._event_queue, queue.Queue)
    assert manager._event_queue.maxsize == dispatch._EVENT_PENDING_CAP
    assert manager._event_worker is None
    assert isinstance(manager._emit_depth, threading.local)
    assert manager._hook_running_callbacks == {}
    assert manager._hook_abandoned == {}
    assert manager._hook_timeout_suppressed_until == {}
    assert isinstance(manager._hook_timeout_lock, type(threading.Lock()))
    assert (
        manager._hook_timeout_suppression_seconds
        == dispatch._HOOK_TIMEOUT_SUPPRESSION_SECONDS
    )
    assert manager._hook_failures_reported == set()


def test_dispatch_runtime_state_is_manager_local(tmp_path):
    from plugin_runtime.manager import PluginManager

    first = PluginManager(scope_key=str(tmp_path / "one"))
    second = PluginManager(scope_key=str(tmp_path / "two"))

    for name in (
        "_subscriptions",
        "_event_lock",
        "_event_idle",
        "_event_pending_by_generation",
        "_event_queue",
        "_emit_depth",
        "_hook_running_callbacks",
        "_hook_abandoned",
        "_hook_timeout_suppressed_until",
        "_hook_timeout_lock",
        "_hook_failures_reported",
    ):
        assert getattr(first, name) is not getattr(second, name)
