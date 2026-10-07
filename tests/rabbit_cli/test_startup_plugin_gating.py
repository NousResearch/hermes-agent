"""Guards for CLI startup performance regression.

``rabbit_cli.main`` skips eager plugin discovery at argparse-setup time
when the invocation is clearly targeting a known built-in subcommand.
This saves 500-650ms on ``rabbit --help``, ``rabbit --version``,
``rabbit logs``, etc., by not importing ``google.cloud.pubsub_v1``,
``aiohttp``, ``grpc``, and friends.

Invariants: ``_plugin_cli_discovery_needed()`` classifies flag/positional argv
correctly, and deferred platforms register their CLI command before argparse
builds the plugin subparsers (issue #54678).
"""

from __future__ import annotations

import sys
from unittest.mock import patch

from rabbit_cli.main import (
    _resolve_deferred_platform_cli_command,
)


# ── _resolve_deferred_platform_cli_command (issue #54678) ──────────────────




def test_deferred_platform_loader_registers_cli_command_before_parser_table():
    """Fake deferred loader must resolve through real registry + register CLI
    before argparse builds plugin command subparsers (issue #54678 review).

    Existing unit tests mock ``platform_registry.get`` and only assert the
    helper calls it. This regression exercises the full chain rabbit-sweeper
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
    from rabbit_cli.plugins import PluginContext, PluginManager, PluginManifest

    mgr = PluginManager()
    manifest = PluginManifest(name="fake-photon-platform")
    command_name = "fakephoton"

    def _fake_loader():
        # Mirrors what a real platform adapter does on import: register its
        # top-level rabbit <name> CLI command via PluginContext.
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
    parser = argparse.ArgumentParser(prog="rabbit")
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
