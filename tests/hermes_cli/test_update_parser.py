"""Unit tests for the extracted ``hermes update`` parser builder."""

from __future__ import annotations

import argparse

from hermes_cli.subcommands.update import build_update_parser


def _sentinel_handler(args):  # pragma: no cover - only identity is asserted
    return "update-handler"


def _build():
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_update_parser(subparsers, cmd_update=_sentinel_handler)
    return parser


def test_upgrade_alias_routes_to_update_handler():
    parser = _build()
    ns = parser.parse_args(["upgrade", "--check"])
    assert ns.command == "upgrade"
    assert ns.check is True
    assert ns.func is _sentinel_handler


def test_update_still_routes_to_update_handler():
    parser = _build()
    ns = parser.parse_args(["update", "--check"])
    assert ns.command == "update"
    assert ns.check is True
    assert ns.func is _sentinel_handler
