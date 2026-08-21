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
from hermes_cli.subcommands.claw import build_claw_parser
from hermes_cli.subcommands.insights import build_insights_parser
from hermes_cli.subcommands.mcp import build_mcp_parser
from hermes_cli.subcommands.memory import build_memory_parser
from hermes_cli.subcommands.pairing import build_pairing_parser
from hermes_cli.subcommands.plugins import build_plugins_parser
from hermes_cli.subcommands.skills import build_skills_parser
from hermes_cli.subcommands.tools import build_tools_parser


def _h(name):
    def handler(args):  # pragma: no cover - identity only
        return name
    handler.__name__ = f"cmd_{name}"
    return handler


# (subcommand, builder, handler_kwarg, sample argv that should dispatch to func)
CASES = [
    ("memory", build_memory_parser, "cmd_memory", ["memory"]),
    ("acp", build_acp_parser, "cmd_acp", ["acp"]),
    ("tools", build_tools_parser, "cmd_tools", ["tools"]),
    ("insights", build_insights_parser, "cmd_insights", ["insights"]),
    ("skills", build_skills_parser, "cmd_skills", ["skills"]),
    ("pairing", build_pairing_parser, "cmd_pairing", ["pairing"]),
    ("plugins", build_plugins_parser, "cmd_plugins", ["plugins"]),
    ("mcp", build_mcp_parser, "cmd_mcp", ["mcp"]),
    ("claw", build_claw_parser, "cmd_claw", ["claw"]),
]


@pytest.mark.parametrize("name,builder,kw,argv", CASES, ids=[c[0] for c in CASES])
def test_followup_builders_dispatch(name, builder, kw, argv):
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    handler = _h(name)
    builder(sub, **{kw: handler})
    ns = parser.parse_args(argv)
    assert ns.command == name
    assert ns.func is handler


def test_mcp_and_acp_accept_hooks_flag():
    # mcp/acp parser blocks use the shared add_accept_hooks_flag helper.
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_mcp_parser(sub, cmd_mcp=_h("mcp"))
    build_acp_parser(sub, cmd_acp=_h("acp"))
    # acp takes --accept-hooks at top level
    ns = parser.parse_args(["acp", "--accept-hooks"])
    assert ns.accept_hooks is True


def test_skills_reset_help_distinguishes_plain_reset_from_restore(capsys):
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_skills_parser(sub, cmd_skills=_h("skills"))

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["skills", "reset", "--help"])

    assert exc.value.code == 0
    output = capsys.readouterr().out
    assert "byte-identical" in output
    assert "genuinely differs" in output
    assert "--restore" in output


def test_skills_list_modified_help_points_to_restore(capsys):
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_skills_parser(sub, cmd_skills=_h("skills"))

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["skills", "list-modified", "--help"])

    assert exc.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "hermes skills diff <name>" in output
    assert "hermes skills reset <name> --restore" in output
    assert "replace the modified copy" in output


def test_skills_diff_help_explains_modified_reset_contract(capsys):
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_skills_parser(sub, cmd_skills=_h("skills"))

    with pytest.raises(SystemExit) as exc:
        parser.parse_args(["skills", "diff", "--help"])

    assert exc.value.code == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "Plain reset only re-baselines" in output
    assert "genuinely modified copies stay preserved and skipped" in output
    assert "hermes skills reset <name> --restore" in output
    assert "replace the modified copy and resume stock updates" in output
