"""Smoke tests for the Phase 2 follow-up subcommand builders (promoted handlers).

These 9 subcommands had their handler defined as a closure inside main(); the
handler was promoted to top-level and the parser block extracted into a builder.
Confirms each builder attaches its subcommand and wires func to the injected
handler.
"""

from __future__ import annotations

import argparse
import re


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


def test_claw_migrate_no_backup_help_shows_active_hermes_home(monkeypatch, tmp_path, capsys):
    # The --no-backup help text must reflect the ACTIVE HERMES_HOME (per
    # AGENTS.md: display_hermes_home() for user-facing text), not a
    # hardcoded ~/.hermes/ literal — the archive itself is written under
    # the active home, so the help text describing it must match.
    from hermes_constants import display_hermes_home

    custom_home = tmp_path / "custom_profile_home"
    custom_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(custom_home))
    active_display = display_hermes_home()

    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_claw_parser(sub, cmd_claw=_h("claw"))

    with pytest.raises(SystemExit):
        parser.parse_args(["claw", "migrate", "--help"])
    help_text = capsys.readouterr().out
    # argparse's HelpFormatter wraps long help lines with textwrap, which can
    # break a long path mid-token at a hyphen with a bare "\n<indent>" (no
    # inserted space). Collapse those wrap points back out before checking
    # for the path so this test doesn't depend on terminal width / wrapping.
    flat_help_text = re.sub(r"\n\s*", "", help_text)

    assert "~/.hermes/backups/" not in flat_help_text
    assert "~/.hermes/" not in flat_help_text
    assert active_display in flat_help_text


def test_claw_migrate_no_backup_help_handles_percent_in_home(monkeypatch, tmp_path, capsys):
    # argparse's HelpFormatter applies %-templating to every action's help string
    # (_expand_help() does `self._get_help_string(action) % params`), so a literal
    # '%' anywhere in the interpolated HERMES_HOME path (e.g. an unusual profile
    # name) would otherwise blow up --help with "unsupported format character" /
    # "not enough arguments for format string" unless the '%' is escaped to '%%'
    # before interpolation. This pins that escape as a live behavior contract.
    from hermes_constants import display_hermes_home

    custom_home = tmp_path / "profile%name"
    custom_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(custom_home))
    active_display = display_hermes_home()
    assert "%" in active_display  # sanity: the fixture actually exercises a literal '%'

    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    build_claw_parser(sub, cmd_claw=_h("claw"))

    with pytest.raises(SystemExit):
        parser.parse_args(["claw", "migrate", "--help"])
    help_text = capsys.readouterr().out
    flat_help_text = re.sub(r"\n\s*", "", help_text)

    assert "~/.hermes/" not in flat_help_text
    assert active_display in flat_help_text
