"""Plugin CLI commands can never break the ``hermes`` parser or each other.

A plugin command named like a built-in used to either crash every non-built-in
invocation (``ValueError: conflicting subparser`` from the built-in's own
``add_parser``) or abort the discovery loop and drop every later plugin's
commands, as did a plugin whose ``setup_fn`` raised.
"""

from __future__ import annotations

import logging
import sys

import pytest

import hermes_cli.main as main
import hermes_cli.plugins as plugins_mod
import plugins.memory as memory_plugins
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


def _ran(args):
    return f"ran:{args.x}"


def _setup(parser):
    parser.add_argument("--x")


def _boom(parser):
    parser.add_argument("--half-built")
    raise RuntimeError("setup exploded")


@pytest.fixture
def builtin_names(monkeypatch):
    """Built-in command names in registration order, from a build with no plugin discovery."""
    monkeypatch.setattr(main, "_plugin_cli_discovery_needed", lambda: False)
    _parser, subparsers = main._build_cli_parser()
    monkeypatch.undo()
    return list(subparsers.choices)


@pytest.fixture
def build_with_plugins(monkeypatch):
    """Build the real parser for ``hermes plugcmd`` with the given plugin registrations."""

    def build(registrations):
        mgr = PluginManager()
        for plugin, name, setup_fn in registrations:
            PluginContext(PluginManifest(name=plugin), mgr).register_cli_command(
                name, help=name, setup_fn=setup_fn, handler_fn=_ran,
            )
        monkeypatch.setattr(sys, "argv", ["hermes", "plugcmd"])
        monkeypatch.setattr(memory_plugins, "discover_plugin_cli_commands", lambda: [])
        monkeypatch.setattr(plugins_mod, "discover_plugins", lambda *a, **k: None)
        monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: mgr)
        monkeypatch.setattr(main, "_resolve_deferred_platform_cli_command", lambda _name: None)
        return main._build_cli_parser()

    return build


def test_plugin_named_like_any_builtin_never_crashes_or_shadows_it(builtin_names, build_with_plugins):
    # The first and last built-ins bracket wherever plugin registration sits in the build.
    first, last = builtin_names[0], builtin_names[-1]
    parser, subparsers = build_with_plugins([
        ("clash", last, _setup), ("clash", first, _setup), ("clash", "plugcmd", _setup),
    ])

    assert subparsers.choices[first].get_default("func") is not _ran
    assert subparsers.choices[last].get_default("func") is not _ran
    args = parser.parse_args(["plugcmd", "--x", "1"])
    assert args.func(args) == "ran:1"


def test_failing_or_colliding_plugin_does_not_drop_another_plugins_command(
    builtin_names, build_with_plugins, caplog,
):
    with caplog.at_level(logging.WARNING, logger="hermes_cli.main"):
        parser, subparsers = build_with_plugins([
            ("aaa-clash", builtin_names[0], _setup),
            ("aab-broken", "brokencmd", _boom),
            ("zzz-good", "plugcmd", _setup),
        ])

    args = parser.parse_args(["plugcmd", "--x", "2"])
    assert args.func(args) == "ran:2"
    # No half-built parser left behind to parse ``hermes brokencmd`` or list it in --help.
    assert "brokencmd" not in subparsers.choices
    assert "brokencmd" not in parser.format_help()
    assert "aaa-clash" in caplog.text and "aab-broken" in caplog.text
