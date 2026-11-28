"""Guards for CLI startup performance regression.

``hermes_cli.main`` skips eager plugin discovery at argparse-setup time
when the invocation is clearly targeting a known built-in subcommand.
This saves 500-650ms on ``hermes --help``, ``hermes --version``,
``hermes logs``, etc., by not importing ``google.cloud.pubsub_v1``,
``aiohttp``, ``grpc``, and friends.

Invariants: ``_plugin_cli_discovery_needed()`` classifies flag/positional argv
correctly, and deferred platforms register their CLI command before argparse
builds the plugin subparsers (issue #54678).
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from hermes_cli.main import (
    _resolve_deferred_platform_cli_command,
)


# ── _resolve_deferred_platform_cli_command (issue #54678) ──────────────────




def test_deferred_platform_loader_registers_cli_command_before_parser_table():
    """Fake deferred loader must resolve through real registry + register CLI
    before argparse builds plugin command subparsers (issue #54678 review).

    Existing unit tests mock ``platform_registry.get`` and only assert the
    helper calls it. This regression exercises the full chain hermes-sweeper
    asked for:

    1. a deferred platform loader is pending on a real ``PlatformRegistry``
    2. that loader materializes a ``register_cli_command`` side effect on the
       plugin manager (no Photon SDK import)
    3. ``_resolve_deferred_platform_cli_command`` runs the pending loader
    4. the registered command is present on the manager *and* visible as an
       argparse subparser/choice after the same construction path ``main()``
       uses for plugin CLI commands
    """
    import argparse

    from gateway.platform_registry import PlatformRegistry
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    mgr = PluginManager()
    manifest = PluginManifest(name="fake-photon-platform")
    command_name = "fakephoton"

    def _fake_loader():
        # Mirrors what a real platform adapter does on import: register its
        # top-level hermes <name> CLI command via PluginContext.
        ctx = PluginContext(manifest, mgr)
        ctx.register_cli_command(
            name=command_name,
            help="Fake deferred platform CLI (test-only)",
            setup_fn=lambda p: None,
            description="Hermetic stand-in for deferred platform CLI registration",
        )

    registry = PlatformRegistry()
    registry.register_deferred(command_name, _fake_loader)

    # Precondition: deferred is known, but CLI command is not yet registered.
    assert registry.is_registered(command_name)
    assert command_name not in mgr._cli_commands
    assert command_name in registry._deferred

    fake_module = type(sys)("gateway.platform_registry")
    fake_module.platform_registry = registry
    with patch.dict(sys.modules, {"gateway.platform_registry": fake_module}):
        _resolve_deferred_platform_cli_command(command_name)

    # Loader resolution must promote deferred -> concrete and fire CLI register.
    assert command_name not in registry._deferred
    assert command_name in mgr._cli_commands
    cmd_info = mgr._cli_commands[command_name]
    assert cmd_info["name"] == command_name
    assert cmd_info["plugin"] == "fake-photon-platform"

    # Same parser-table assembly main() uses after reading _cli_commands.
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    for info in mgr._cli_commands.values():
        plugin_parser = subparsers.add_parser(
            info["name"],
            help=info["help"],
            description=info.get("description", ""),
        )
        info["setup_fn"](plugin_parser)
        if info.get("handler_fn") is not None:
            plugin_parser.set_defaults(func=info["handler_fn"])

    # argparse surface: choices + parse success.
    choices = getattr(subparsers, "choices", None) or {}
    assert command_name in choices
    parsed = parser.parse_args([command_name])
    assert parsed.command == command_name


# ── `hermes completion` enumerates plugin commands (issue #145347) ─────────


@pytest.mark.parametrize("argv, needed", [
    (["hermes", "completion", "bash"], True),
    (["hermes", "completion"], True),
    (["hermes", "logs"], False),
    # Top-level --help skipping plugin commands is the documented trade-off.
    (["hermes", "--help"], False),
])
def test_completion_alone_among_builtins_triggers_plugin_discovery(monkeypatch, argv, needed):
    from hermes_cli.main import _plugin_cli_discovery_needed

    monkeypatch.setattr("sys.argv", argv)
    assert _plugin_cli_discovery_needed() is needed


def test_completion_scripts_list_plugin_and_deferred_platform_commands(monkeypatch):
    """A completion script walks the live parser, so it must be built with every plugin command
    attached — including one a deferred platform only registers when its loader runs."""
    import plugins.memory as memory_plugins
    from hermes_cli import left_core_migration
    from hermes_cli import plugins as plugins_mod
    from gateway.platform_registry import PlatformRegistry
    from hermes_cli.completion import generate_bash, generate_fish, generate_zsh
    from hermes_cli.main import _build_cli_parser
    from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest

    mgr = PluginManager()
    PluginContext(PluginManifest(name="fake-a2a"), mgr).register_cli_command(
        name="fakea2a", help="Fake plugin CLI", setup_fn=lambda p: None)

    def _deferred_loader():
        PluginContext(PluginManifest(name="fake-photon-platform"), mgr).register_cli_command(
            name="fakephoton", help="Fake deferred platform CLI", setup_fn=lambda p: None)

    registry = PlatformRegistry()
    registry.register_deferred("fakephoton", _deferred_loader)
    fake_registry_module = type(sys)("gateway.platform_registry")
    fake_registry_module.platform_registry = registry

    monkeypatch.setattr(plugins_mod, "discover_plugins", lambda *a, **k: None)
    monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: mgr)
    monkeypatch.setattr(memory_plugins, "discover_plugin_cli_commands", list)
    monkeypatch.setattr(left_core_migration, "recover_for_command", lambda name: None)
    monkeypatch.setattr("sys.argv", ["hermes", "completion", "bash"])
    with patch.dict(sys.modules, {"gateway.platform_registry": fake_registry_module}):
        parser, _subparsers = _build_cli_parser()

    for generate in (generate_bash, generate_zsh, generate_fish):
        script = generate(parser)
        assert "fakea2a" in script, generate.__name__
        assert "fakephoton" in script, generate.__name__
