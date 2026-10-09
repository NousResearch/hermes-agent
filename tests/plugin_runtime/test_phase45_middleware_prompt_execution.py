"""Phase 4.5 Step 5 middleware/prompt execution ownership gates."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DISPATCH = ROOT / "plugin_runtime" / "dispatch.py"

EXECUTION_METHODS = {
    "render_system_prompt_sections",
    "_render_prompt_section_text",
    "has_middleware",
    "invoke_middleware",
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


def test_middleware_and_prompt_execution_are_runtime_owned():
    import plugin_runtime.dispatch as dispatch

    assert EXECUTION_METHODS <= set(dispatch.PluginDispatchMixin.__dict__)


def test_complete_dispatch_host_declares_remaining_execution_state():
    import plugin_runtime.dispatch as dispatch

    assert {"_middleware", "_system_prompt_sections"} <= set(
        dispatch.PluginDispatchHost.__annotations__
    )


def test_runtime_dispatch_still_has_no_cli_or_agent_back_edges():
    imports = _imports(RUNTIME_DISPATCH)

    assert not any(
        module == "hermes_cli"
        or module.startswith("hermes_cli.")
        or module == "agent"
        or module.startswith("agent.")
        for module in imports
    )


def test_safe_worker_policy_uses_host_seam_for_remaining_surfaces(monkeypatch, tmp_path):
    from hermes_cli.plugins import PluginManifest
    from plugin_runtime.manager import PluginManager
    from plugin_runtime.context import PluginContext

    manager = PluginManager(scope_key=str(tmp_path))
    context = PluginContext(PluginManifest(name="example", key="example"), manager)
    context.register_middleware("probe", lambda **_kwargs: "middleware")
    context.register_system_prompt_section("example.prompt", "prompt")
    monkeypatch.setattr(manager, "_plugin_dispatch_safe_worker_enabled", lambda: True)

    assert manager.has_middleware("probe") is False
    assert manager.invoke_middleware("probe") == []
    assert manager.render_system_prompt_sections({}) == []

def test_prompt_execution_preserves_section_count_budget(caplog, tmp_path):
    import logging

    import plugin_runtime.dispatch as dispatch
    from hermes_cli.plugins import PluginManifest
    from plugin_runtime.manager import PluginManager
    from plugin_runtime.context import PluginContext

    manager = PluginManager(scope_key=str(tmp_path))
    context = PluginContext(PluginManifest(name="example", key="example"), manager)
    for index in range(dispatch.MAX_SYSTEM_PROMPT_SECTIONS + 1):
        context.register_system_prompt_section(f"example.section-{index:02d}", "x")

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        rendered = manager.render_system_prompt_sections({})

    assert len(rendered) == dispatch.MAX_SYSTEM_PROMPT_SECTIONS
    assert "section-count budget" in caplog.text


def test_prompt_execution_rejects_reserved_persistence_markers(caplog, tmp_path):
    import logging

    import plugin_runtime.dispatch as dispatch
    from hermes_cli.plugins import PluginManifest
    from plugin_runtime.manager import PluginManager
    from plugin_runtime.context import PluginContext

    manager = PluginManager(scope_key=str(tmp_path))
    context = PluginContext(PluginManifest(name="example", key="example"), manager)
    context.register_system_prompt_section(
        "example.reserved",
        f"before {dispatch.PLUGIN_SECTIONS_START} after",
    )

    with caplog.at_level(logging.WARNING, logger="hermes_cli.plugins"):
        rendered = manager.render_system_prompt_sections({})

    assert rendered == []
    assert "reserved persistence marker" in caplog.text
