"""Smoke tests for the final batch of subcommand builders extracted from main().

These groups either imported their handler from a sibling module inside the
parser block (moa, fallback, migrate, bundles, checkpoints, curator, pets,
journey, secrets, egress) or carried a closure handler that only closed over
its own parser (worktree, browser, computer-use, sessions, completion). The
closures moved verbatim into the builder; sessions/completion take the handler
by injection.
"""

from __future__ import annotations

import argparse

import pytest

from hermes_cli.subcommands.browser import build_browser_parser
from hermes_cli.subcommands.bundles import build_bundles_parser
from hermes_cli.subcommands.checkpoints import build_checkpoints_parser
from hermes_cli.subcommands.completion import build_completion_parser
from hermes_cli.subcommands.computer_use import build_computer_use_parser
from hermes_cli.subcommands.curator import build_curator_parser
from hermes_cli.subcommands.egress import build_egress_parser
from hermes_cli.subcommands.fallback import build_fallback_parser
from hermes_cli.subcommands.journey import build_journey_parser
from hermes_cli.subcommands.migrate import build_migrate_parser
from hermes_cli.subcommands.moa import build_moa_parser
from hermes_cli.subcommands.pets import build_pets_parser
from hermes_cli.subcommands.secrets import build_secrets_parser
from hermes_cli.subcommands.sessions import build_sessions_parser
from hermes_cli.subcommands.whatsapp import build_whatsapp_cloud_parser
from hermes_cli.subcommands.worktree import build_worktree_parser


def _tree():
    parser = argparse.ArgumentParser(prog="hermes")
    return parser, parser.add_subparsers(dest="command")


SELF_CONTAINED = [
    ("moa", build_moa_parser, ["moa", "list"]),
    ("fallback", build_fallback_parser, ["fallback", "add"]),
    ("worktree", build_worktree_parser, ["worktree", "prune", "--dry-run"]),
    ("browser", build_browser_parser, ["browser", "close-profile"]),
    ("secrets", build_secrets_parser, ["secrets"]),
    ("egress", build_egress_parser, ["egress"]),
    ("migrate", build_migrate_parser, ["migrate", "xai", "--apply"]),
    ("checkpoints", build_checkpoints_parser, ["checkpoints"]),
    ("bundles", build_bundles_parser, ["bundles"]),
    ("curator", build_curator_parser, ["curator"]),
    ("pets", build_pets_parser, ["pets"]),
    ("journey", build_journey_parser, ["journey"]),
    ("computer-use", build_computer_use_parser, ["computer-use", "status"]),
]


@pytest.mark.parametrize("name,builder,argv", SELF_CONTAINED, ids=[c[0] for c in SELF_CONTAINED])
def test_self_contained_builders_attach(name, builder, argv):
    parser, sub = _tree()
    builder(sub)
    ns = parser.parse_args(argv)
    assert ns.command == name
    assert callable(getattr(ns, "func", None)) or name in ("checkpoints", "curator", "pets", "journey")


def test_worktree_aliases_normalize_to_list(monkeypatch):
    parser, sub = _tree()
    build_worktree_parser(sub)
    seen = {}
    monkeypatch.setattr("hermes_cli.worktree_cmd.cmd_worktree", lambda a: seen.setdefault("action", a.worktree_action))
    ns = parser.parse_args(["worktree", "audit"])
    ns.func(ns)
    assert seen["action"] == "list"


def test_sessions_injects_handler_and_threads_parser():
    parser, sub = _tree()
    calls = []
    build_sessions_parser(sub, cmd_sessions=lambda a, **kw: calls.append((a, kw)) or 0)
    ns = parser.parse_args(["sessions", "prune", "--older-than", "2d", "--dry-run"])
    assert ns.sessions_action == "prune" and ns.older_than == "2d" and ns.dry_run is True
    assert ns.func(ns) == 0
    (a, kw), = calls
    assert a is ns
    assert kw["sessions_parser"].prog.endswith("sessions")


def test_completion_passes_top_level_parser():
    parser, sub = _tree()
    got = {}
    build_completion_parser(sub, cmd_completion=lambda a, p: got.update(a=a, p=p), parser=parser)
    ns = parser.parse_args(["completion", "zsh"])
    ns.func(ns)
    assert got["p"] is parser and got["a"].shell == "zsh"


def test_whatsapp_cloud_dispatch():
    parser, sub = _tree()
    h = lambda a: "wa"  # noqa: E731
    build_whatsapp_cloud_parser(sub, cmd_whatsapp_cloud=h)
    assert parser.parse_args(["whatsapp-cloud"]).func is h


def test_computer_use_no_action_prints_help(capsys):
    parser, sub = _tree()
    build_computer_use_parser(sub)
    ns = parser.parse_args(["computer-use"])
    ns.func(ns)
    assert "install" in capsys.readouterr().out
