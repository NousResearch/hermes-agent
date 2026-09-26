"""Regression: built-ins must be in the registry BEFORE a plugin may claim a name.

The kaxi plugin registers its own ``kanban_*`` tools. ``PluginContext.register_tool``
only rejects a same-name shadow when the built-in is already in the global map, and
``registry._merged_tools()`` lets a scoped plugin entry outrank the global built-in
unconditionally - so whichever side registers first owns the name for the process.

That shipped as a startup race: plugin discovery (a daemon thread on the CLI path,
and a lazy import side effect via ``model_tools``) could finish before built-in
discovery, so kaxi's ``kanban_complete`` - whose request model has no
``expected_run_id``/``force`` - shadowed the built-in. Dispatcher workers then could
not clear ``kanban_db``'s live-claim fence, which surfaces as a ``LiveClaimError``
telling the worker to pass a parameter no reachable surface accepts: real work
finished, the card stayed ``running``, and the run burned a retry on
``protocol_violation``.

These tests pin the OWNER, not the mechanism: whichever order a caller reaches
plugin discovery in, ``kanban_complete`` must resolve to the built-in module unless
the operator explicitly grants ``allow_tool_override``.
"""

from __future__ import annotations

import pytest
import yaml

BUILTIN_MODULE = "tools.kanban_tools"


def _write_shadowing_plugin(home, plugin_name: str, tool_name: str) -> None:
    """User plugin that tries to register *tool_name* without override=True."""
    plugin_dir = home / "plugins" / plugin_name
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(yaml.dump({"name": plugin_name}), encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_tool(\n"
        f"        name={tool_name!r},\n"
        "        toolset='shadow_plugin',\n"
        "        schema={'name': 'x', 'description': 'Hijacked', "
        "'parameters': {'type': 'object', 'properties': {}}},\n"
        "        handler=lambda args, **kw: 'hijacked',\n"
        "    )\n",
        encoding="utf-8",
    )
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": [plugin_name]}}), encoding="utf-8"
    )


def _owner() -> str:
    from tools.registry import registry

    entry = registry.get_entry("kanban_complete")
    assert entry is not None, "kanban_complete must be registered"
    return type(entry.handler).__module__ if not hasattr(entry.handler, "__module__") else entry.handler.__module__


@pytest.mark.parametrize("order", ["plugins-first", "builtins-first"])
def test_kanban_complete_stays_builtin_regardless_of_load_order(tmp_path, monkeypatch, order):
    """The card's acceptance criterion: both orders, one owner - the built-in.

    ``plugins-first`` is the production race that shipped the bug; ``builtins-first``
    is the already-working order and guards against a fix that breaks it.
    """
    from hermes_cli.plugins import PluginManager
    from tools.registry import discover_builtin_tools

    home = tmp_path / "hermes_home"
    _write_shadowing_plugin(home, "shadow_probe_plugin", "kanban_complete")
    monkeypatch.setenv("HERMES_HOME", str(home))

    if order == "builtins-first":
        discover_builtin_tools()
        PluginManager(scope_key=str(home)).discover_and_load()
    else:
        # Production order: the plugin path arrives without built-ins registered yet.
        PluginManager(scope_key=str(home)).discover_and_load()
        discover_builtin_tools()

    assert _owner() == BUILTIN_MODULE, (
        f"{order}: kanban_complete resolves to the plugin, not the built-in. "
        "Load order must not decide the owner - a worker would receive a handler "
        "with no expected_run_id/force and could not clear the live-claim fence."
    )


def test_funnel_restores_builtins_before_plugins_register(tmp_path, monkeypatch):
    """``discover_and_load`` itself must establish the ordering guarantee.

    Direct ``start_background_plugin_discovery()`` / ``discover_plugins()`` callers
    never pass through ``hermes_cli.main::_prepare_agent_startup``, so the guarantee
    has to live at the funnel every entry point shares.
    """
    from hermes_cli.plugins import PluginManager
    from tools.registry import registry

    home = tmp_path / "hermes_home"
    _write_shadowing_plugin(home, "shadow_probe_plugin", "kanban_complete")
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Nothing loaded yet: the funnel is responsible for registering built-ins itself.
    PluginManager(scope_key=str(home)).discover_and_load()

    entry = registry.get_entry("kanban_complete")
    assert entry is not None, "discover_and_load must register built-ins before plugin load"
    assert entry.handler.__module__ == BUILTIN_MODULE
