"""Unit tests for the ``hermes plugins`` parser builder."""

from __future__ import annotations

import argparse

from hermes_cli.subcommands.plugins import build_plugins_parser


def _sentinel_handler(args):  # pragma: no cover - only identity is asserted
    return "plugins-handler"


def _build():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_plugins_parser(subparsers, cmd_plugins=_sentinel_handler)
    return parser


def test_plugin_singular_alias_dispatches_to_plugins_handler():
    parser = _build()

    plural = parser.parse_args(["plugins", "list"])
    singular = parser.parse_args(["plugin", "list"])

    assert plural.plugins_action == "list"
    assert singular.plugins_action == "list"
    assert plural.func is _sentinel_handler
    assert singular.func is _sentinel_handler
