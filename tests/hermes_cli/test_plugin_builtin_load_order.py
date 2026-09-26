"""Regression: a plugin must never shadow a built-in tool by winning the load-order race.

The kaxi plugin registers ``kanban_complete``/``kanban_block``/``kanban_comment``/
``kanban_create``/``kanban_link``/``kanban_show``/``kanban_list`` under the same names as
the built-in tools. ``PluginContext.register_tool`` only rejects a same-name shadow when
the built-in is ALREADY in ``registry._tools`` (``hermes_cli/plugins.py``), and
``ToolRegistry._merged_tools()`` overlays the profile-scoped plugin map on top of the
global built-in map unconditionally. So whichever side registers first keeps the name.

That race shipped a real incident: workers got kaxi's ``kanban_complete`` handler, whose
request schema has no ``expected_run_id``/``force``, so the built-in live-claim fence
(``hermes_cli/kanban_db.py::complete_task`` -> ``LiveClaimError``) could not be cleared from
any surface the worker could reach. Workers finished real work, failed to close their
cards, exited rc=0, and were scored as ``protocol_violation``, burning a retry each time.

The fix is an ordering guarantee, not a new precedence rule: ``PluginManager.discover_and_load``
runs built-in tool discovery before any plugin's ``register(ctx)`` executes. These tests pin
that guarantee from both directions so a future reorder cannot silently reopen the race.

The registry-precedence rule itself (``_merged_tools``) is deliberately NOT changed here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from hermes_cli.plugins import PluginManager
from tools.registry import registry

# A unique name so these tests never collide with a real built-in or plugin.
TOOL_NAME = "load_order_fixture_tool"
BUILTIN_TOOLSET = "load_order_fixture"

BUILTIN_MARKER = "builtin-handler"


def _register_builtin(name: str) -> None:
    """Register a stand-in built-in (the real modules self-register on import)."""
    registry.register(
        name=name,
        toolset=BUILTIN_TOOLSET,
        schema={
            "name": name,
            "description": "builtin",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda args, **kw: BUILTIN_MARKER,
    )


def _winner() -> str:
    """``BUILTIN_MARKER`` or ``"plugin-handler"`` — which handler owns the contested name."""
    entry = registry.get_entry(TOOL_NAME)
    assert entry is not None, f"{TOOL_NAME} vanished from the registry"
    return entry.handler({})


def _make_plugin_dir(base: Path, name: str, tool_name: str) -> Path:
    """An enabled directory plugin whose register() claims *tool_name*.

    Deliberately WITHOUT ``override=True``: this is the shape of an accidental name
    collision (kaxi), not an operator-consented replacement.
    """
    plugin_dir = base / name
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": name, "version": "0.1.0"}), encoding="utf-8"
    )
    (plugin_dir / "__init__.py").write_text(
        "def register(ctx):\n"
        "    ctx.register_tool(\n"
        f'        name="{tool_name}",\n'
        '        toolset="load_order_fixture_plugin",\n'
        f'        schema={{"name": "{tool_name}", "description": "plugin", '
        '"parameters": {"type": "object", "properties": {}}},\n'
        '        handler=lambda args, **kw: "plugin-handler",\n'
        "    )\n",
        encoding="utf-8",
    )
    return plugin_dir


@pytest.fixture
def fixture_home(tmp_path, monkeypatch):
    """Isolated Hermes home + one enabled plugin claiming TOOL_NAME.

    ``tools.registry.discover_builtin_tools`` is stubbed by default so tests do not
    import the repo's ~44 real tool modules: these tests are about ORDER, and the
    stand-in built-in below is the contested name. Tests that need discovery to do
    real work patch it themselves.
    """
    home = tmp_path / "hermes_test"
    (home / "plugins").mkdir(parents=True)
    _make_plugin_dir(home / "plugins", "load_order_plugin", TOOL_NAME)
    (home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": {"enabled": ["load_order_plugin"]}}), encoding="utf-8"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))

    # Point discovery at an empty bundled dir: the real repo bundle would load 50+
    # unrelated plugins and make failures hard to attribute.
    empty_bundled = tmp_path / "bundled"
    empty_bundled.mkdir()
    monkeypatch.setattr(
        "hermes_cli.plugins.get_bundled_plugins_dir", lambda: empty_bundled
    )

    discover_calls: list[str] = []

    def _stub_discover_builtin_tools(*args, **kwargs):
        discover_calls.append("builtin-discovery")
        return []

    monkeypatch.setattr("tools.registry.discover_builtin_tools", _stub_discover_builtin_tools)

    # TOOL_NAME is unique to this module, so nothing may own it on entry.
    scope = registry.current_scope_key()
    assert registry.snapshot_registration(TOOL_NAME, scope=scope) is None, (
        f"fixture precondition: {TOOL_NAME} already registered in scope {scope!r}"
    )
    assert registry.get_entry(TOOL_NAME) is None, (
        f"fixture precondition: {TOOL_NAME} already registered globally"
    )

    try:
        yield discover_calls
    finally:
        for target_scope in (None, scope):
            entry = (
                registry.snapshot_registration(TOOL_NAME, scope=target_scope)
                if target_scope is not None
                else registry._tools.get(TOOL_NAME)
            )
            if entry is not None:
                registry.deregister(TOOL_NAME, scope=target_scope)


class TestBuiltinWinsRegardlessOfLoadOrder:
    """Both orders must land on the built-in; only the mechanism differs by design."""

    def test_builtin_already_registered_is_protected(self, fixture_home):
        """Builtin-first: the existing guard in ``register_tool`` rejects the shadow."""
        _register_builtin(TOOL_NAME)
        assert _winner() == BUILTIN_MARKER

        manager = PluginManager()
        manager.discover_and_load()

        assert _winner() == BUILTIN_MARKER, (
            "a plugin shadowed an already-registered built-in tool"
        )
        assert TOOL_NAME not in manager._plugin_tool_names, (
            "the rejected shadow must not be tracked as a plugin-owned tool"
        )

    def test_plugin_cannot_win_when_discovery_registers_builtins_first(
        self, fixture_home, monkeypatch
    ):
        """Plugins-first input: discover_and_load must still hand the name to the built-in.

        This is the kaxi incident shape - nothing pre-registered the built-in, so the
        plugin registered into the scoped overlay against an empty global map and
        ``_merged_tools()`` handed it the name permanently.
        """

        def _discover_and_register(*args, **kwargs):
            fixture_home.append("builtin-discovery")
            _register_builtin(TOOL_NAME)
            return []

        monkeypatch.setattr("tools.registry.discover_builtin_tools", _discover_and_register)

        PluginManager().discover_and_load()

        assert _winner() == BUILTIN_MARKER, (
            "a plugin won a name because built-ins were registered after it loaded; "
            "discover_and_load must populate the global tool map before the plugin sweep"
        )


class TestDiscoveryOrdering:
    """The ordering guarantee itself, independent of which tool name is contested."""

    def test_builtin_discovery_runs_before_the_plugin_sweep(self, fixture_home, monkeypatch):
        calls: list[str] = []

        def _fake_discover_builtin_tools(*args, **kwargs):
            calls.append("builtin")
            return []

        real_load_plugin = PluginManager._load_plugin

        def _spy_load_plugin(self, manifest):
            calls.append("plugin")
            return real_load_plugin(self, manifest)

        monkeypatch.setattr(
            "tools.registry.discover_builtin_tools", _fake_discover_builtin_tools
        )
        monkeypatch.setattr(PluginManager, "_load_plugin", _spy_load_plugin)

        PluginManager().discover_and_load()

        assert "builtin" in calls, (
            "discover_and_load never registered built-in tools; a plugin registering "
            "first would permanently shadow a built-in name"
        )
        assert "plugin" in calls, "fixture plugin did not load; the assertion below is vacuous"
        assert calls.index("builtin") < calls.index("plugin"), (
            f"built-in discovery must precede the plugin sweep, got {calls}"
        )

    def test_force_rediscovery_keeps_the_ordering(self, fixture_home, monkeypatch):
        """force=True unloads first, so it needs the same guarantee on every pass."""
        calls: list[str] = []

        def _fake_discover_builtin_tools(*args, **kwargs):
            calls.append("builtin")
            return []

        real_load_plugin = PluginManager._load_plugin

        def _spy_load_plugin(self, manifest):
            calls.append("plugin")
            return real_load_plugin(self, manifest)

        monkeypatch.setattr(
            "tools.registry.discover_builtin_tools", _fake_discover_builtin_tools
        )
        monkeypatch.setattr(PluginManager, "_load_plugin", _spy_load_plugin)

        manager = PluginManager()
        manager.discover_and_load()
        calls.clear()
        manager.discover_and_load(force=True)

        assert calls[:2] == ["builtin", "plugin"], (
            f"force rediscovery must register built-ins before reloading plugins, got {calls}"
        )
