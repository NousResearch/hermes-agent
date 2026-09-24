"""The desktop subcommand's --local launch flag.

The flag remains accepted for compatibility and is passed through to Electron.
Desktop local-model surfaces are available by default on supported platforms;
the Electron launch-policy tests cover that visibility contract.
"""

import argparse

from hermes_cli.subcommands.gui import build_gui_parser


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command")
    build_gui_parser(subparsers, cmd_gui=lambda args: None)

    return parser


def test_local_flag_parses():
    args = _parser().parse_args(["desktop", "--local"])

    assert args.local is True


def test_local_flag_defaults_off():
    args = _parser().parse_args(["desktop"])

    assert args.local is False

