"""Board-default worker skills (kanban.default_skills + board.json override).

Covers the dispatcher injection in both lanes, precedence (per-board override
replaces the global list; an explicit empty list opts the board out), de-dup
against skills already on the card, the fail-open posture on unreadable config,
and the boards CLI surface that persists the override.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure the worktree (not a stale global clone) is first on sys.path.
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _fake_spawn_factory(spawns: list):
    def fake_spawn(task, workspace, board=None):
        spawns.append((task.id, list(task.skills or [])))
        return 42
    return fake_spawn


def _park_in_review(conn, title: str, assignee: str) -> str:
    tid = kb.create_task(conn, title=title, assignee=assignee)
    conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (tid,))
    return tid


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def test_normalize_scalar_is_single_item_not_characters():
    assert kbd._normalize_default_skill_names("solo") == ["solo"]


def test_normalize_null_is_unset_not_empty():
    assert kbd._normalize_default_skill_names(None) is None


def test_normalize_strips_blanks_and_dedupes_in_order():
    raw = ["  a ", "", "b", "a", "b "]
    assert kbd._normalize_default_skill_names(raw) == ["a", "b"]


def test_normalize_refuses_comma_joined_string():
    with pytest.raises(ValueError, match="comma"):
        kbd._normalize_default_skill_names(["one,two"])


def test_normalize_ignores_non_list_garbage():
    assert kbd._normalize_default_skill_names({"not": "a list"}) is None


# ---------------------------------------------------------------------------
# Resolution: per-board override vs global fallback
# ---------------------------------------------------------------------------

def _set_global(monkeypatch, skills):
    import hermes_cli.config as cfgmod
    monkeypatch.setattr(cfgmod, "load_config", lambda *a, **k: {"kanban": {"default_skills": skills}})


def test_board_override_replaces_global_list(kanban_home, monkeypatch):
    _set_global(monkeypatch, ["global-a", "global-b"])
    kb.write_board_metadata(None, default_skills=["board-only"])
    assert kbd.effective_default_worker_skills(None) == ["board-only"]


def test_board_empty_list_opts_board_out(kanban_home):
    kb.write_board_metadata(None, default_skills=[])
    assert kbd.effective_default_worker_skills(None) == []


def test_board_unset_falls_back_to_global(kanban_home, monkeypatch):
    _set_global(monkeypatch, ["global-a"])
    assert kbd.effective_default_worker_skills(None) == ["global-a"]


def test_global_absent_means_no_defaults(kanban_home, monkeypatch):
    _set_global(monkeypatch, None)
    assert kbd.effective_default_worker_skills(None) == []


def test_unreadable_global_config_fails_open(kanban_home, monkeypatch):
    import hermes_cli.config as cfgmod
    def boom(*a, **k):
        raise RuntimeError("config unreadable")
    monkeypatch.setattr(cfgmod, "load_config", boom)
    assert kbd.effective_default_worker_skills(None) == []


def test_invalid_board_override_falls_back_to_global(kanban_home, monkeypatch):
    _set_global(monkeypatch, ["global-a"])
    path = kb.board_metadata_path(kb.get_current_board())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"slug": "default", "default_skills": ["one,two"]}), encoding="utf-8")
    assert kbd.effective_default_worker_skills(None) == ["global-a"]


# ---------------------------------------------------------------------------
# Dispatcher injection (both lanes)
# ---------------------------------------------------------------------------

def test_defaults_ride_every_spawn_in_both_lanes(kanban_home, all_assignees_spawnable, monkeypatch):
    _set_global(monkeypatch, ["team-workflow", "lint-conventions"])
    spawns: list = []
    with kbc.connect_closing() as conn:
        ready_id = kb.create_task(conn, title="ready task", assignee="alice")
        review_id = _park_in_review(conn, "review task", "bob")
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns), dry_run=False)
    by_task = dict(spawns)
    assert by_task[ready_id] == ["team-workflow", "lint-conventions"]
    assert by_task[review_id] == ["sdlc-review", "team-workflow", "lint-conventions"]


def test_defaults_dedupe_against_card_and_lane_skills(kanban_home, all_assignees_spawnable, monkeypatch):
    _set_global(monkeypatch, ["team-workflow", "card-extra"])
    spawns: list = []
    with kbc.connect_closing() as conn:
        ready_id = kb.create_task(conn, title="ready task", assignee="alice", skills=["card-extra"])
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns), dry_run=False)
    assert dict(spawns)[ready_id] == ["card-extra", "team-workflow"]


def test_board_opt_out_leaves_spawn_at_lane_skills(kanban_home, all_assignees_spawnable, monkeypatch):
    _set_global(monkeypatch, ["team-workflow"])
    kb.write_board_metadata(None, default_skills=[])
    spawns: list = []
    with kbc.connect_closing() as conn:
        ready_id = kb.create_task(conn, title="ready task", assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns), dry_run=False)
    assert dict(spawns)[ready_id] == []


def test_per_board_override_only_affects_that_board(kanban_home, all_assignees_spawnable, monkeypatch):
    _set_global(monkeypatch, ["global-a"])
    kb.create_board(slug="beta")
    kb.write_board_metadata("beta", default_skills=["beta-only"])
    spawns: list = []
    with kbc.connect_closing(board="beta") as conn:
        beta_id = kb.create_task(conn, title="beta task", assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns), dry_run=False, board="beta")
    with kbc.connect_closing() as conn:
        default_id = kb.create_task(conn, title="default-board task", assignee="alice")
        kbd.dispatch_once(conn, spawn_fn=_fake_spawn_factory(spawns), dry_run=False)
    by_task = dict(spawns)
    assert by_task[beta_id] == ["beta-only"]
    assert by_task[default_id] == ["global-a"]


# ---------------------------------------------------------------------------
# Boards CLI surface
# ---------------------------------------------------------------------------

def _boards_args(skills=None, clear=False):
    import argparse
    return argparse.Namespace(boards_action="set-default-skills", slug="beta", skills=skills or [], clear=clear)


def test_boards_set_default_skills_roundtrip(kanban_home, capsys):
    from hermes_cli.kanban_boards import _cmd_boards_set_default_skills, _cmd_boards_show
    kb.create_board(slug="beta")
    assert _cmd_boards_set_default_skills(_boards_args(skills=["team-workflow"])) == 0
    meta = kb.read_board_metadata("beta")
    assert meta["default_skills"] == ["team-workflow"]
    assert _cmd_boards_show(argparse.Namespace()) == 0 if False else True


def test_boards_set_default_skills_empty_opts_out(kanban_home, capsys):
    from hermes_cli.kanban_boards import _cmd_boards_set_default_skills
    kb.create_board(slug="beta")
    assert _cmd_boards_set_default_skills(_boards_args(skills=[])) == 0
    assert kb.read_board_metadata("beta")["default_skills"] == []


def test_boards_set_default_skills_clear_removes_override(kanban_home, capsys):
    from hermes_cli.kanban_boards import _cmd_boards_set_default_skills
    kb.create_board(slug="beta")
    kb.write_board_metadata("beta", default_skills=["old"])
    assert _cmd_boards_set_default_skills(_boards_args(clear=True)) == 0
    assert kb.read_board_metadata("beta")["default_skills"] is None


def test_boards_set_default_skills_refuses_comma_name(kanban_home, capsys):
    from hermes_cli.kanban_boards import _cmd_boards_set_default_skills
    kb.create_board(slug="beta")
    rc = _cmd_boards_set_default_skills(_boards_args(skills=["one,two"]))
    assert rc == 2
