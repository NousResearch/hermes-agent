"""Smoke tests for the batch-extracted subcommand parser builders.

Each ``build_<group>_parser`` should attach its subcommand to a subparsers
group and wire ``func`` to the injected handler. These are intentionally
light — the byte-identical ``--help`` verification done at extraction time is
the real behavioral guarantee; this just guards against a module failing to
import or a builder raising.
"""

from __future__ import annotations

import argparse


from hermes_cli.subcommands.config import build_config_parser
from hermes_cli.subcommands.login import build_login_parser



def _h(name):
    def handler(args):  # pragma: no cover - identity only
        return name
    handler.__name__ = f"cmd_{name}"
    return handler






def test_config_get_unset_subcommands_parse():
    """`hermes config get/unset` parse key args (and --json for get)."""
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    handler = _h("config")
    build_config_parser(sub, cmd_config=handler)

    ns = parser.parse_args(["config", "get", "terminal.backend", "--json"])
    assert ns.func is handler
    assert ns.config_command == "get"
    assert ns.key == "terminal.backend"
    assert ns.json is True

    ns = parser.parse_args(["config", "unset", "terminal.backend"])
    assert ns.func is handler
    assert ns.config_command == "unset"
    assert ns.key == "terminal.backend"




# ── deprecated `hermes login` fails gracefully, not with argparse error ────
#
# `hermes login` is a removed command; its handler (`login_command` in
# `hermes_cli/auth.py`) prints a deprecation notice pointing at `hermes auth` /
# `hermes model` and exits 0.  Two behavior contracts guard the UX:
#   1. ANY `--provider <value>` (including ones the user actually wants, like
#      `anthropic`) must parse and reach the handler — never crash in argparse
#      with `invalid choice` before the friendly redirect is printed (#24756).
#   2. The subcommand must not advertise itself in the parser help row.






def test_login_subparser_help_is_suppressed():
    """The deprecated `login` row must not appear in `hermes --help`.

    Must hold without leaking argparse's literal `==SUPPRESS==` placeholder,
    which `help=argparse.SUPPRESS` emits for a top-level subparser on 3.12+.
    The fix omits the `help=` kwarg entirely instead.
    """
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_login_parser(sub, cmd_login=_h("login"))
    help_text = parser.format_help()
    # No leaked SUPPRESS placeholder row.
    assert "==SUPPRESS==" not in help_text
