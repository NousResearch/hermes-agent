"""Phase 4.5 Step 4 event-dispatch ownership gates."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DISPATCH = ROOT / "plugin_runtime" / "dispatch.py"

EVENT_METHODS = {
    "_subscribe_event",
    "_remove_plugin_subscriptions",
    "_ensure_event_worker_locked",
    "_event_worker_loop",
    "_mark_event_done",
    "_deliver_event",
    "_wait_for_event_dispatch",
    "_dispatch_event",
}


def _imports(path: Path) -> set[str]:
    modules: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_event_execution_methods_are_runtime_owned():
    import plugin_runtime.dispatch as dispatch

    assert EVENT_METHODS <= set(dispatch.PluginEventDispatchMixin.__dict__)


def test_runtime_event_owner_declares_host_state_and_ownership_seams():
    import plugin_runtime.dispatch as dispatch

    expected_state = {
        "_subscriptions",
        "_event_lock",
        "_event_idle",
        "_event_generation",
        "_event_pending_by_generation",
        "_event_queue",
        "_event_worker",
        "_emit_depth",
        "_ownership_ledger",
    }
    expected_methods = {
        "_track_owner_registration",
        "_dispose_registrations",
        "_forget_registrations",
    }

    assert expected_state <= set(dispatch.PluginEventDispatchHost.__annotations__)
    assert expected_methods <= set(dispatch.PluginEventDispatchHost.__dict__)


def test_runtime_event_execution_has_no_cli_or_agent_back_edges():
    imports = _imports(RUNTIME_DISPATCH)

    assert not any(
        module == "hermes_cli"
        or module.startswith("hermes_cli.")
        or module == "agent"
        or module.startswith("agent.")
        for module in imports
    )


def test_event_result_resolution_uses_existing_runtime_host_seam(monkeypatch, tmp_path):
    from plugin_runtime.manager import PluginManager

    manager = PluginManager(scope_key=str(tmp_path))
    observed = []
    resolved = []

    def callback(**payload):
        observed.append(payload["value"])
        return "subscriber-result"

    def resolve(result):
        resolved.append(result)
        return result

    monkeypatch.setattr(manager, "_plugin_dispatch_resolve_result", resolve)
    registration = manager._subscribe_event("listener", "source:tick", callback)

    assert registration.active
    assert registration in manager._ownership_ledger["listener"]
    assert manager._dispatch_event("source:tick", {"value": 7}) == 1
    assert manager._wait_for_event_dispatch(timeout=2.0)
    assert observed == [7]
    assert resolved == ["subscriber-result"]
