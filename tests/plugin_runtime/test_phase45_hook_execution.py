"""Phase 4.5 Step 3 hook-execution ownership gates."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DISPATCH = ROOT / "plugin_runtime" / "dispatch.py"
PLUGIN_TESTS = ROOT / "tests" / "hermes_cli" / "test_plugins.py"

HOOK_METHODS = {
    "_hook_callback_kwargs",
    "_invoke_hook_callback",
    "invoke_hook",
    "_report_hook_failure",
    "_run_hook_callback_bounded",
    "has_hook",
    "ainvoke_hook",
    "iter_hook_callbacks",
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


def test_hook_execution_methods_are_runtime_owned():
    import plugin_runtime.dispatch as dispatch

    assert HOOK_METHODS <= set(dispatch.PluginHookDispatchMixin.__dict__)


def test_runtime_hook_owner_declares_narrow_host_contract():
    import plugin_runtime.dispatch as dispatch

    expected_state = {
        "_hooks",
        "_hook_running_callbacks",
        "_hook_abandoned",
        "_hook_timeout_suppressed_until",
        "_hook_timeout_lock",
        "_hook_timeout_suppression_seconds",
        "_hook_failures_reported",
        "_plugin_dispatch_safe_worker_enabled",
        "_plugin_dispatch_resolve_result",
    }
    declared_state = set(dispatch.PluginHookDispatchHost.__annotations__)
    declared_methods = set(dispatch.PluginHookDispatchHost.__dict__)
    assert expected_state <= declared_state | declared_methods


def test_plugin_manager_supplies_hook_host_dependencies():
    from plugin_runtime.manager import PluginManager

    assert "_plugin_dispatch_safe_worker_enabled" in PluginManager.__dict__
    assert "_plugin_dispatch_resolve_result" in PluginManager.__dict__


def test_runtime_hook_execution_has_no_cli_or_agent_back_edges():
    imports = _imports(RUNTIME_DISPATCH)

    assert not any(
        module == "hermes_cli"
        or module.startswith("hermes_cli.")
        or module == "agent"
        or module.startswith("agent.")
        for module in imports
    )


def test_hook_timeout_monkeypatches_follow_runtime_owner():
    source = PLUGIN_TESTS.read_text(encoding="utf-8")

    assert "hermes_cli.plugins._resolve_hook_callback_timeout" not in source
    assert "plugin_runtime.dispatch._resolve_hook_callback_timeout" in source


def test_safe_worker_policy_is_host_injected(monkeypatch, tmp_path):
    from plugin_runtime.manager import PluginManager

    manager = PluginManager(scope_key=str(tmp_path))
    manager._hooks["pre_tool_call"] = [lambda **_kwargs: {"action": "allow"}]
    monkeypatch.setattr(manager, "_plugin_dispatch_safe_worker_enabled", lambda: True)

    assert manager.has_hook("pre_tool_call") is False
    assert manager.iter_hook_callbacks("pre_tool_call") == ()
    assert manager.invoke_hook("pre_tool_call", tool_name="read_file", args={}) == []


def test_hook_result_resolution_uses_host_seam(monkeypatch, tmp_path):
    from plugin_runtime.manager import PluginManager

    manager = PluginManager(scope_key=str(tmp_path))
    seen = []

    def resolve(result):
        seen.append(result)
        return "resolved"

    monkeypatch.setattr(manager, "_plugin_dispatch_resolve_result", resolve)
    manager._hooks["post_tool_call"] = [lambda **_kwargs: "raw"]
    monkeypatch.setattr("plugin_runtime.dispatch._resolve_hook_callback_timeout", lambda: 0.0)

    assert manager.invoke_hook("post_tool_call", tool_name="x", args={}, result="ok") == ["resolved"]
    assert seen == ["raw"]
