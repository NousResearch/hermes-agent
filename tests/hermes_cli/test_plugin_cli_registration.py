"""Tests for plugin CLI registration system.

Covers:
  - PluginContext.register_cli_command()
  - PluginManager._cli_commands storage
  - get_plugin_cli_commands() convenience function
  - Memory plugin CLI discovery (discover_plugin_cli_commands)
  - Honcho register_cli() builds correct argparse tree
"""

import sys
from unittest.mock import MagicMock


from hermes_cli.plugins import (
    PluginContext,
    PluginManager,
    PluginManifest,
)


# ── PluginContext.register_cli_command ─────────────────────────────────────


class TestRegisterCliCommand:
    def _make_ctx(self):
        mgr = PluginManager()
        manifest = PluginManifest(name="test-plugin")
        return PluginContext(manifest, mgr), mgr

    def test_registers_command(self):
        ctx, mgr = self._make_ctx()
        setup = MagicMock()
        handler = MagicMock()
        ctx.register_cli_command(
            name="mycmd",
            help="Do something",
            setup_fn=setup,
            handler_fn=handler,
            description="Full description",
        )
        assert "mycmd" in mgr._cli_commands
        entry = mgr._cli_commands["mycmd"]
        assert entry["name"] == "mycmd"
        assert entry["help"] == "Do something"
        assert entry["setup_fn"] is setup
        assert entry["handler_fn"] is handler
        assert entry["plugin"] == "test-plugin"

    def test_overwrites_on_duplicate(self):
        ctx, mgr = self._make_ctx()
        ctx.register_cli_command("x", "first", MagicMock())
        ctx.register_cli_command("x", "second", MagicMock())
        assert mgr._cli_commands["x"]["help"] == "second"

    def test_handler_optional(self):
        ctx, mgr = self._make_ctx()
        ctx.register_cli_command("nocb", "test", MagicMock())
        assert mgr._cli_commands["nocb"]["handler_fn"] is None


# ── Memory plugin CLI discovery ───────────────────────────────────────────


class TestMemoryPluginCliDiscovery:
    def test_discovers_active_plugin_with_register_cli(self, tmp_path, monkeypatch):
        """Only the active memory provider's CLI commands are discovered."""
        plugin_dir = tmp_path / "testplugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("pass\n")
        (plugin_dir / "cli.py").write_text(
            "def register_cli(subparser):\n"
            "    subparser.add_argument('--test')\n"
            "\n"
            "def testplugin_command(args):\n"
            "    pass\n"
        )
        (plugin_dir / "plugin.yaml").write_text(
            "name: testplugin\ndescription: A test plugin\n"
        )

        # Also create a second plugin that should NOT be discovered
        other_dir = tmp_path / "otherplugin"
        other_dir.mkdir()
        (other_dir / "__init__.py").write_text("pass\n")
        (other_dir / "cli.py").write_text(
            "def register_cli(subparser):\n"
            "    subparser.add_argument('--other')\n"
        )

        import plugins.memory as pm
        original_dir = pm._MEMORY_PLUGINS_DIR
        mod_key = "plugins.memory.testplugin.cli"
        sys.modules.pop(mod_key, None)

        monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", tmp_path)
        # Set testplugin as the active provider
        monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: "testplugin")
        try:
            cmds = pm.discover_plugin_cli_commands()
        finally:
            monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", original_dir)
            sys.modules.pop(mod_key, None)

        # Only testplugin should be discovered, not otherplugin
        assert len(cmds) == 1
        assert cmds[0]["name"] == "testplugin"
        assert cmds[0]["help"] == "A test plugin"
        assert callable(cmds[0]["setup_fn"])
        assert cmds[0]["handler_fn"].__name__ == "testplugin_command"

    def test_returns_nothing_when_no_active_provider(self, tmp_path, monkeypatch):
        """No commands when memory.provider is not set in config."""
        plugin_dir = tmp_path / "testplugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("pass\n")
        (plugin_dir / "cli.py").write_text(
            "def register_cli(subparser):\n    pass\n"
        )

        import plugins.memory as pm
        original_dir = pm._MEMORY_PLUGINS_DIR
        monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", tmp_path)
        monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: None)
        try:
            cmds = pm.discover_plugin_cli_commands()
        finally:
            monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", original_dir)

        assert len(cmds) == 0

    def test_skips_plugin_without_register_cli(self, tmp_path, monkeypatch):
        """An active plugin with cli.py but no register_cli returns nothing."""
        plugin_dir = tmp_path / "noplugin"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("pass\n")
        (plugin_dir / "cli.py").write_text("def some_other_fn():\n    pass\n")

        import plugins.memory as pm
        original_dir = pm._MEMORY_PLUGINS_DIR
        monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", tmp_path)
        monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: "noplugin")
        try:
            cmds = pm.discover_plugin_cli_commands()
        finally:
            monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", original_dir)
            sys.modules.pop("plugins.memory.noplugin.cli", None)

        assert len(cmds) == 0

    def test_skips_plugin_without_cli_py(self, tmp_path, monkeypatch):
        """An active provider without cli.py returns nothing."""
        plugin_dir = tmp_path / "nocli"
        plugin_dir.mkdir()
        (plugin_dir / "__init__.py").write_text("pass\n")

        import plugins.memory as pm
        original_dir = pm._MEMORY_PLUGINS_DIR
        monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", tmp_path)
        monkeypatch.setattr(pm, "_get_active_memory_provider", lambda: "nocli")
        try:
            cmds = pm.discover_plugin_cli_commands()
        finally:
            monkeypatch.setattr(pm, "_MEMORY_PLUGINS_DIR", original_dir)

        assert len(cmds) == 0


# ── Honcho register_cli ──────────────────────────────────────────────────


# ── ProviderCollector no-op ──────────────────────────────────────────────


class TestProviderCollectorCliNoop:
    def test_register_cli_command_is_noop(self):
        """_ProviderCollector.register_cli_command is a no-op (doesn't crash)."""
        from plugins.memory import _ProviderCollector

        collector = _ProviderCollector()
        collector.register_cli_command(
            name="test", help="test", setup_fn=lambda s: None
        )
        # Should not store anything — CLI is discovered via file convention
        assert not hasattr(collector, "_cli_commands")


# ── Deferred platform plugin CLI resolution ───────────────────────────────


class TestDeferredPlatformCliResolution:
    """A bundled platform plugin registers its ``hermes <platform>`` CLI
    command inside ``register()``, but platform plugins load lazily (deferred):
    the module — and thus the ``register_cli_command()`` call — is only imported
    when the platform_registry is first asked for that platform.

    Regression: a standalone ``hermes <platform> ...`` invocation parses argv
    before the registry is touched, so the subcommand was never wired up and
    argparse rejected it as an invalid choice. The fix force-resolves the one
    deferred platform whose name matches the first positional token. These
    tests assert the underlying contract that fix relies on: resolving a
    deferred platform runs its loader, which populates ``_cli_commands``.
    """

    def _make_deferred_platform(self, platform_name):
        """Wire a deferred loader whose module registers a CLI command.

        Returns ``(registry, manager, loader_calls)`` where ``loader_calls`` is
        a single-element list used as a call counter so tests can assert the
        loader ran at most once.
        """
        from gateway.platform_registry import PlatformRegistry

        registry = PlatformRegistry()
        manager = PluginManager()
        loader_calls = []

        def _loader():
            # Mirrors what a real platform plugin's register() does: register
            # the platform entry AND its `hermes <platform>` CLI command.
            loader_calls.append(1)
            ctx = PluginContext(PluginManifest(name=f"{platform_name}-platform"), manager)
            ctx.register_cli_command(
                name=platform_name,
                help=f"Manage the {platform_name} integration",
                setup_fn=MagicMock(),
                handler_fn=MagicMock(),
            )

        registry.register_deferred(platform_name, _loader)
        return registry, manager, loader_calls

    def test_cli_command_absent_until_resolved(self):
        """Before the platform is resolved, its CLI command is not registered."""
        registry, manager, loader_calls = self._make_deferred_platform("acmechat")
        # Deferred loader is pending but has NOT run yet.
        assert "acmechat" in registry._deferred
        assert loader_calls == []
        assert "acmechat" not in manager._cli_commands

    def test_resolving_platform_registers_cli_command(self):
        """Resolving the deferred platform runs its loader and wires the CLI."""
        registry, manager, loader_calls = self._make_deferred_platform("acmechat")

        entry = registry.get("acmechat")

        assert entry is None or entry.name == "acmechat"  # loader may skip register()
        assert loader_calls == [1]
        # The CLI command is now available on the manager.
        assert "acmechat" in manager._cli_commands
        assert manager._cli_commands["acmechat"]["help"] == (
            "Manage the acmechat integration"
        )

    def test_loader_runs_at_most_once(self):
        """A second lookup does not re-run the loader (deferred entry popped)."""
        registry, manager, loader_calls = self._make_deferred_platform("acmechat")
        registry.get("acmechat")
        registry.get("acmechat")
        assert loader_calls == [1]

    def test_unrelated_token_does_not_resolve_platform(self):
        """A non-platform token (e.g. a chat prompt) leaves loaders untouched.

        This is the fast-path guarantee: ``hermes <chat prompt>`` must not
        import any platform SDK just because discovery ran.
        """
        registry, manager, loader_calls = self._make_deferred_platform("acmechat")

        # Simulate the guard in main(): only resolve when the token names a
        # deferred platform. "hello" is not a platform, so nothing resolves.
        token = "hello"
        if token in registry._deferred:
            registry.get(token)

        assert loader_calls == []
        assert "acmechat" not in manager._cli_commands

