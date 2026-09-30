"""Ownership contracts for Phase 4.6.4 manager lifecycle extraction."""

from __future__ import annotations

import ast
from pathlib import Path


RUNTIME_STATE = {
    "_plugin_manager",
    "_plugin_managers_by_home",
    "_plugin_managers_lock",
    "_published_tui_message_injector",
    "_published_tui_host_lock",
    "_background_discovery_thread",
    "_background_discovery_lock",
}


def _assigned_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    assigned: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            assigned.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigned.add(node.target.id)
    return assigned


def test_runtime_lifecycle_is_sole_state_owner() -> None:
    import hermes_cli.plugins as plugin_api
    import plugin_runtime.lifecycle as lifecycle

    assert RUNTIME_STATE <= _assigned_names(Path(lifecycle.__file__))
    assert RUNTIME_STATE.isdisjoint(_assigned_names(Path(plugin_api.__file__)))


def test_cli_lifecycle_exports_are_canonical_identities() -> None:
    import hermes_cli.plugins as plugin_api
    import plugin_runtime.lifecycle as lifecycle

    assert plugin_api.get_plugin_manager is lifecycle.get_plugin_manager
    assert plugin_api.discover_plugins is lifecycle.discover_plugins
    assert (
        plugin_api.start_background_plugin_discovery
        is lifecycle.start_background_plugin_discovery
    )
    assert plugin_api.get_plugin_toolset_keys_nowait is lifecycle.get_plugin_toolset_keys_nowait
    assert (
        plugin_api.get_portable_mcp_server_names_nowait
        is lifecycle.get_portable_mcp_server_names_nowait
    )
    assert plugin_api.publish_tui_message_host is lifecycle.publish_tui_message_host
    assert plugin_api.clear_published_tui_message_host is lifecycle.clear_published_tui_message_host
    assert plugin_api._ensure_plugins_discovered is lifecycle.ensure_plugins_discovered
    assert plugin_api.has_enabled_agent_plugin_mcp is lifecycle.has_enabled_agent_plugin_mcp


def test_runtime_lifecycle_has_no_cli_plugin_monolith_backedge() -> None:
    import plugin_runtime.lifecycle as lifecycle

    tree = ast.parse(Path(lifecycle.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.module != "hermes_cli.plugins"
        elif isinstance(node, ast.Import):
            assert all(alias.name != "hermes_cli.plugins" for alias in node.names)


def test_private_runtime_lifecycle_helpers_are_not_cli_exports() -> None:
    import hermes_cli.plugins as plugin_api

    for name in (
        "_plugin_manager",
        "_plugin_managers_by_home",
        "_plugin_managers_lock",
        "_background_discovery_thread",
        "_background_discovery_lock",
        "_delivery_manager",
        "_join_background_discovery",
        "_reset_plugin_managers_for_tests",
    ):
        assert not hasattr(plugin_api, name)


def test_cli_private_consumers_use_runtime_lifecycle_owner() -> None:
    root = Path(__file__).resolve().parents[2]

    middleware = (root / "hermes_cli" / "middleware.py").read_text(encoding="utf-8")
    activation = (root / "plugin_runtime" / "activation.py").read_text(encoding="utf-8")

    assert "from plugin_runtime.lifecycle import delivery_manager" in middleware
    assert "from hermes_cli.plugins import _delivery_manager" not in middleware
    assert "from plugin_runtime.lifecycle import (" in activation
    assert "get_plugin_manager," in activation
    assert "join_background_discovery," in activation
    assert "refresh_tui_plugin_sessions," in activation
    assert "_join_background_discovery" not in activation
