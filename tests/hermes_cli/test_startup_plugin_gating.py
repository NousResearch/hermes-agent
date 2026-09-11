"""Guards for CLI startup performance regression.

``hermes_cli.main`` skips eager plugin discovery at argparse-setup time
when the invocation is clearly targeting a known built-in subcommand.
This saves 500-650ms on ``hermes --help``, ``hermes --version``,
``hermes logs``, etc., by not importing ``google.cloud.pubsub_v1``,
``aiohttp``, ``grpc``, and friends.

Two invariants:

1. ``_BUILTIN_SUBCOMMANDS`` must contain every subcommand that is actually
   registered by ``main()``.  If an entry is missing, plugin discovery
   runs unnecessarily for that command (correctness-safe, just slow).
   If an entry is PRESENT but the subcommand doesn't exist, a plugin
   could shadow the name — also bad.

2. ``_plugin_cli_discovery_needed()`` returns the right answer for the
   flag/positional parsing cases it's meant to handle.
"""

from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stdout
from unittest.mock import patch

import pytest

from hermes_cli._parser import build_top_level_parser, top_level_value_flag_sets
from hermes_cli.main import (
    _BUILTIN_SUBCOMMANDS,
    _first_positional_argv,
    _plugin_cli_discovery_needed,
    _resolve_deferred_platform_cli_command,
)


# ── helper: grab the live set of top-level subcommands from argparse ───────


def _live_subcommand_names() -> set[str]:
    """Run ``hermes --help`` in-process and parse the subcommand block.

    We patch ``_plugin_cli_discovery_needed`` to always return False so
    plugin-registered commands aren't included — we're validating the
    built-in-only set.
    """
    from hermes_cli import main as _main

    argv_backup = sys.argv[:]
    sys.argv = ["hermes", "--help"]
    buf = io.StringIO()
    try:
        with patch.object(_main, "_plugin_cli_discovery_needed", return_value=False):
            with redirect_stdout(buf):
                with pytest.raises(SystemExit):
                    _main.main()
    finally:
        sys.argv = argv_backup

    text = buf.getvalue()
    # argparse prints "{chat,model,...}" somewhere in the help output
    m = re.search(r"\{([a-zA-Z0-9_,\-]+)\}", text)
    assert m, f"Could not find subcommand group in --help output:\n{text[:500]}"
    return set(m.group(1).split(","))


# ── _first_positional_argv ─────────────────────────────────────────────────


def test_value_flag_sets_match_top_level_parser():
    required, optional = top_level_value_flag_sets()

    for action in build_top_level_parser()[0]._actions:
        if not action.option_strings or action.nargs == 0:
            continue
        expected = optional if action.nargs == "?" else required
        assert set(action.option_strings) <= expected


def test_reasoning_value_is_not_misclassified_as_subcommand(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["hermes", "--reasoning", "high", "chat", "hello"],
    )

    assert _first_positional_argv() == "chat"
    assert _plugin_cli_discovery_needed() is False




# ── _plugin_cli_discovery_needed ───────────────────────────────────────────


# ── _BUILTIN_SUBCOMMANDS ↔ argparse registration parity ────────────────────




# ── _resolve_deferred_platform_cli_command (issue #54678) ──────────────────


def test_deferred_platform_cli_resolution_targets_matching_platform():
    """The slow path must import the deferred platform whose name matches the
    invoked command, so its register_cli_command side effect fires.

    Photon registers ``hermes photon`` only when its adapter module is
    imported; on the unknown-command slow path the platform is still a
    deferred entry, so without this resolution step the CLI command stays
    absent and argparse rejects ``photon`` (issue #54678).
    """
    from hermes_cli import main as _main

    class _FakeRegistry:
        def __init__(self):
            self.resolved: list[str] = []

        def get(self, name):
            self.resolved.append(name)
            return None

    fake = _FakeRegistry()
    fake_module = type(sys)("gateway.platform_registry")
    fake_module.platform_registry = fake
    with patch.dict(sys.modules, {"gateway.platform_registry": fake_module}):
        _resolve_deferred_platform_cli_command("photon")

    assert fake.resolved == ["photon"]


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
