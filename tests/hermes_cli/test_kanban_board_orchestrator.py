"""Per-board orchestrator_profile / default_assignee overrides (#34977).

A board's own ``board.json`` values shadow the global ``kanban.*`` keys, so one
project's orchestrator never adopts another board's decomposed roots or
unassigned cards. Unset values inherit the global config exactly as before.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli import kanban_boards
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_decompose as decomp

_PROFILES = ["default", "mt-observer", "lab-orchestrator", "lab-ops"]
_GLOBAL = {"kanban": {"orchestrator_profile": "mt-observer", "default_assignee": "mt-observer"}}


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / ".hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    kb.create_board("lab")
    fake = [SimpleNamespace(name=n, description=f"d {n}") for n in _PROFILES]
    with patch("hermes_cli.profiles.list_profiles", return_value=fake), \
            patch("hermes_cli.profiles.profile_exists", side_effect=lambda n: n in _PROFILES), \
            patch("hermes_cli.profiles.get_active_profile_name", return_value="default"), \
            patch("hermes_cli.config.load_config_readonly", return_value=_GLOBAL):
        yield h


def test_board_override_shadows_global_routing(home):
    kb.write_board_metadata("lab", orchestrator_profile="lab-orchestrator", default_assignee="lab-ops")
    lab = decomp._load_routing(board="lab")
    other = decomp._load_routing(board=kb.DEFAULT_BOARD)
    assert (lab.orchestrator, lab.default_assignee) == ("lab-orchestrator", "lab-ops")
    assert (other.orchestrator, other.default_assignee) == ("mt-observer", "mt-observer")


def test_routing_follows_the_current_board_when_not_passed(home):
    kb.write_board_metadata("lab", orchestrator_profile="lab-orchestrator")
    with kb.scoped_current_board("lab"):
        assert decomp._load_routing().orchestrator == "lab-orchestrator"
    assert decomp._load_routing().orchestrator == "mt-observer"


def test_unknown_board_orchestrator_falls_back_like_global(home):
    """A board naming a deleted profile degrades through the same chain as the global key."""
    kb.write_board_metadata("lab", orchestrator_profile="ghost")
    assert decomp._load_routing(board="lab", root_assignee="lab-ops").orchestrator == "lab-ops"


def test_dispatcher_applies_board_default_assignee(home):
    kb.write_board_metadata("lab", default_assignee="lab-ops")
    ids = {}
    for slug in ("lab", kb.DEFAULT_BOARD):
        with kbc.connect_closing(board=slug) as conn:
            ids[slug] = kb.create_task(conn, title=f"unassigned on {slug}")
    for slug in ids:
        with kbc.connect_closing(board=slug) as conn:
            kbd.dispatch_once(conn, board=slug, dry_run=False, default_assignee="mt-observer",
                              spawn_fn=lambda *a, **k: None)
    with kbc.connect_closing(board="lab") as conn:
        assert kb.get_task(conn, ids["lab"]).assignee == "lab-ops"
    with kbc.connect_closing(board=kb.DEFAULT_BOARD) as conn:
        assert kb.get_task(conn, ids[kb.DEFAULT_BOARD]).assignee == "mt-observer"


def test_cli_set_and_clear(home, capsys):
    ns = argparse.Namespace
    assert kanban_boards._cmd_boards_set_orchestrator(ns(slug="lab", profile="lab-orchestrator")) == 0
    assert kb.read_board_metadata("lab")["orchestrator_profile"] == "lab-orchestrator"
    assert kanban_boards._cmd_boards_set_orchestrator(ns(slug="lab", profile=None)) == 0
    assert kb.read_board_metadata("lab")["orchestrator_profile"] is None
    assert kanban_boards._cmd_boards_set_default_assignee(ns(slug="lab", profile="ghost")) != 0
    assert kb.read_board_metadata("lab")["default_assignee"] is None
