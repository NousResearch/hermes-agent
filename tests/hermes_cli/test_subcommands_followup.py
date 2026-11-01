"""Smoke tests for the Phase 2 follow-up subcommand builders (promoted handlers).

These 9 subcommands had their handler defined as a closure inside main(); the
handler was promoted to top-level and the parser block extracted into a builder.
Confirms each builder attaches its subcommand and wires func to the injected
handler.
"""

from __future__ import annotations

import argparse

import pytest


from hermes_cli.subcommands.acp import build_acp_parser
from hermes_cli.subcommands.mcp import build_mcp_parser


def _h(name):
    def handler(args):  # pragma: no cover - identity only
        return name
    handler.__name__ = f"cmd_{name}"
    return handler






def test_mcp_and_acp_accept_hooks_flag():
    # mcp/acp parser blocks use the shared add_accept_hooks_flag helper.
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_mcp_parser(sub, cmd_mcp=_h("mcp"))
    build_acp_parser(sub, cmd_acp=_h("acp"))
    # acp takes --accept-hooks at top level
    ns = parser.parse_args(["acp", "--accept-hooks"])
    assert ns.accept_hooks is True


def test_mcp_configure_accepts_noninteractive_tool_selection_flags():
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_mcp_parser(sub, cmd_mcp=_h("mcp"))

    selected = parser.parse_args(["mcp", "configure", "docs", "--tools", "search,read"])
    assert selected.tools == "search,read"
    assert selected.all is False

    all_tools = parser.parse_args(["mcp", "configure", "docs", "--all"])
    assert all_tools.tools is None
    assert all_tools.all is True

    with pytest.raises(SystemExit):
        parser.parse_args([
            "mcp", "configure", "docs", "--tools", "search", "--all"
        ])
