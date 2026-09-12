"""Per-board ``default_assignee`` (``board.json``).

A repo board can name the profile that owns its unassigned ready rows and the
decomposed children the LLM could not route, instead of the host-global
``kanban.default_assignee``. Covers the metadata round trip, the
``boards set-default-assignee`` CLI, the decomposer's routing + prompt, and
the dispatcher's unassigned-ready path. An absent field keeps the old
behaviour (global fallback) everywhere.
"""

from __future__ import annotations

import argparse
import json as jsonlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli import kanban_boards
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_decompose as decomp
from hermes_cli import kanban_parser

BOARD = "repo"


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_HOME"):
        monkeypatch.delenv(var, raising=False)
    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    kb.create_board(BOARD, name="Repo", description="The repo board", default_workdir="/src/repo")
    return home


def _patch_profiles(names: list[str]):
    fake_profiles = [
        SimpleNamespace(
            name=n, is_default=(i == 0), description=f"desc for {n}",
            description_auto=False, model="m", provider="p", skill_count=1,
        )
        for i, n in enumerate(names)
    ]
    return [
        patch("hermes_cli.profiles.list_profiles", return_value=fake_profiles),
        patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in names),
        patch("hermes_cli.profiles.get_active_profile_name", return_value=names[0]),
    ]


def _fake_aux_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    return resp


def _fake_spawn(*args, **kwargs):
    return 12345


# ---------------------------------------------------------------------------
# Metadata + CLI
# ---------------------------------------------------------------------------

def test_metadata_set_and_clear_default_assignee(kanban_home):
    assert kb.read_board_metadata(BOARD)["default_assignee"] is None
    meta = kb.write_board_metadata(BOARD, default_assignee="code-repo")
    assert meta["default_assignee"] == "code-repo"
    raw = jsonlib.loads(kb.board_metadata_path(BOARD).read_text(encoding="utf-8"))
    assert raw["default_assignee"] == "code-repo"
    # Unmentioned fields survive; "" clears.
    assert kb.write_board_metadata(BOARD, name="Repo 2")["default_assignee"] == "code-repo"
    assert kb.write_board_metadata(BOARD, default_assignee="")["default_assignee"] is None


def test_board_default_assignee_requires_existing_profile(kanban_home):
    kb.write_board_metadata(BOARD, default_assignee="ghost")
    with patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x == "code-repo"):
        assert kb.board_default_assignee(BOARD) is None
        kb.write_board_metadata(BOARD, default_assignee="code-repo")
        assert kb.board_default_assignee(BOARD) == "code-repo"
    assert kb.board_default_assignee("default") is None


def test_board_default_assignee_never_raises(kanban_home):
    # Hand-edited board.json: a non-string value reads as unset, and a malformed
    # slug falls back instead of raising into the dispatch tick.
    path = kb.board_metadata_path(BOARD)
    raw = jsonlib.loads(path.read_text(encoding="utf-8"))
    raw["default_assignee"] = 123
    path.write_text(jsonlib.dumps(raw), encoding="utf-8")
    assert kb.board_default_assignee(BOARD) is None
    assert kb.board_default_assignee("Not A Slug!") is None


def test_cli_set_default_assignee_validates_and_clears(kanban_home, capsys):
    ns = lambda profile: argparse.Namespace(slug=BOARD, profile=profile)  # noqa: E731
    with patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x == "code-repo"):
        assert kanban_boards._cmd_boards_set_default_assignee(ns("nope")) != 0
        assert kb.read_board_metadata(BOARD)["default_assignee"] is None
        assert kanban_boards._cmd_boards_set_default_assignee(ns("code-repo")) == 0
        assert kb.read_board_metadata(BOARD)["default_assignee"] == "code-repo"
        assert kanban_boards._cmd_boards_set_default_assignee(ns("none")) == 0
        assert kb.read_board_metadata(BOARD)["default_assignee"] is None
    assert kanban_boards._cmd_boards_set_default_assignee(
        argparse.Namespace(slug="missing-board", profile="code-repo")
    ) != 0
    out = capsys.readouterr().out
    assert "set to 'code-repo'" in out and "cleared" in out


def test_cli_show_prints_default_assignee(kanban_home, capsys):
    kb.write_board_metadata(BOARD, default_assignee="code-repo")
    kb.set_current_board(BOARD)
    assert kanban_boards._cmd_boards_show(argparse.Namespace()) == 0
    out = capsys.readouterr().out
    assert "Default assignee: code-repo" in out
    assert "Default workdir:  /src/repo" in out


def test_parser_registers_set_default_assignee():
    root = argparse.ArgumentParser()
    kanban_parser.build_parser(root.add_subparsers(dest="cmd"))
    args = root.parse_args(["kanban", "boards", "set-default-assignee", BOARD, "code-repo"])
    assert args.boards_action == "set-default-assignee"
    assert (args.slug, args.profile) == (BOARD, "code-repo")
    assert kanban_boards._BOARD_HANDLERS["set-default-assignee"] is kanban_boards._cmd_boards_set_default_assignee


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

_FANOUT = jsonlib.dumps({
    "fanout": True,
    "rationale": "split",
    "tasks": [
        {"title": "a", "body": "x", "assignee": "made_up", "parents": []},
        {"title": "b", "body": "y", "assignee": None, "parents": [0]},
    ],
})


def _decompose_on_board(monkeypatch, *, profiles: list[str]):
    """Run one fan-out decomposition pinned to BOARD; ``(outcome, user_prompt)``."""
    monkeypatch.setenv("HERMES_KANBAN_BOARD", BOARD)
    with kbc.connect_closing(board=BOARD) as conn:
        tid = kb.create_task(conn, title="ship it", triage=True)
    patches = _patch_profiles(profiles)
    for p in patches:
        p.start()
    try:
        with patch(
            "agent.auxiliary_client.call_llm", return_value=_fake_aux_response(_FANOUT),
        ) as call_llm, patch(
            "hermes_cli.kanban_decompose._load_config",
            return_value={"kanban": {"default_assignee": "fallback"}},
        ):
            outcome = decomp.decompose_task(tid, author="me")
    finally:
        for p in patches:
            p.stop()
    user_prompt = call_llm.call_args.kwargs["messages"][1]["content"]
    return outcome, user_prompt


def test_decompose_prefers_board_default_over_global(kanban_home, monkeypatch):
    kb.write_board_metadata(BOARD, default_assignee="code-repo")
    outcome, prompt = _decompose_on_board(
        monkeypatch, profiles=["orchestrator", "fallback", "code-repo"],
    )
    assert outcome.ok, outcome.reason
    with kbc.connect_closing(board=BOARD) as conn:
        assignees = [kb.get_task(conn, cid).assignee for cid in outcome.child_ids]
    assert assignees == ["code-repo", "code-repo"]
    assert "Board: Repo - The repo board (repo: /src/repo); Board default assignee: code-repo" in prompt
    assert "Default assignee (used when no profile fits a task): code-repo" in prompt


def test_decompose_absent_board_field_uses_global(kanban_home, monkeypatch):
    outcome, prompt = _decompose_on_board(monkeypatch, profiles=["orchestrator", "fallback"])
    assert outcome.ok, outcome.reason
    with kbc.connect_closing(board=BOARD) as conn:
        assignees = [kb.get_task(conn, cid).assignee for cid in outcome.child_ids]
    assert assignees == ["fallback", "fallback"]
    assert "Board: Repo - The repo board (repo: /src/repo)\n" in prompt
    assert "Default assignee (used when no profile fits a task): fallback" in prompt


def test_decompose_board_default_naming_missing_profile_uses_global(kanban_home, monkeypatch):
    kb.write_board_metadata(BOARD, default_assignee="ghost")
    outcome, _prompt = _decompose_on_board(monkeypatch, profiles=["orchestrator", "fallback"])
    assert outcome.ok, outcome.reason
    with kbc.connect_closing(board=BOARD) as conn:
        assert {kb.get_task(conn, cid).assignee for cid in outcome.child_ids} == {"fallback"}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _dispatch_unassigned(*, global_default: str = "engineer"):
    """One tick on BOARD with a fresh unassigned ready row; ``(task_id, result)``."""
    with kbc.connect_closing(board=BOARD) as conn:
        tid = kb.create_task(conn, title="t1", assignee=None)
    with patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in {"engineer", "code-repo"}):
        with kbc.connect_closing(board=BOARD) as conn:
            res = kbd.dispatch_once(
                conn, board=BOARD, spawn_fn=_fake_spawn, dry_run=False,
                default_assignee=global_default,
            )
    return tid, res


def _assigned_event(tid: str) -> dict:
    with kbc.connect_closing(board=BOARD) as conn:
        row = conn.execute("SELECT assignee FROM tasks WHERE id = ?", (tid,)).fetchone()
        evs = conn.execute(
            "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'assigned'", (tid,),
        ).fetchall()
    assert len(evs) == 1
    return {"assignee": row["assignee"], **jsonlib.loads(evs[0][0])}


def test_dispatch_board_default_overrides_global(kanban_home):
    kb.write_board_metadata(BOARD, default_assignee="code-repo")
    tid, res = _dispatch_unassigned()
    assert res.auto_assigned_default == [tid]
    assert not res.skipped_unassigned
    assert [(s[0], s[1]) for s in res.spawned] == [(tid, "code-repo")]
    ev = _assigned_event(tid)
    assert ev["assignee"] == "code-repo"
    assert ev["source"] == "board.default_assignee"


def test_dispatch_absent_board_field_uses_global(kanban_home):
    tid, res = _dispatch_unassigned()
    assert res.auto_assigned_default == [tid]
    assert [(s[0], s[1]) for s in res.spawned] == [(tid, "engineer")]
    ev = _assigned_event(tid)
    assert ev["assignee"] == "engineer"
    assert ev["source"] == "kanban.default_assignee"


def test_dispatch_board_default_naming_missing_profile_uses_global(kanban_home):
    kb.write_board_metadata(BOARD, default_assignee="ghost")
    tid, res = _dispatch_unassigned()
    assert [(s[0], s[1]) for s in res.spawned] == [(tid, "engineer")]


def test_dispatch_board_default_leaves_explicit_assignee(kanban_home):
    kb.write_board_metadata(BOARD, default_assignee="code-repo")
    with kbc.connect_closing(board=BOARD) as conn:
        tid = kb.create_task(conn, title="t1", assignee="engineer")
    with patch("hermes_cli.profiles.profile_exists", side_effect=lambda x: x in {"engineer", "code-repo"}):
        with kbc.connect_closing(board=BOARD) as conn:
            res = kbd.dispatch_once(conn, board=BOARD, spawn_fn=_fake_spawn, default_assignee="")
    assert [(s[0], s[1]) for s in res.spawned] == [(tid, "engineer")]
    assert not res.auto_assigned_default
    with kbc.connect_closing(board=BOARD) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id = ? AND kind = 'assigned'", (tid,),
        ).fetchone()[0] == 0


def test_dispatch_board_default_without_global(kanban_home):
    kb.write_board_metadata(BOARD, default_assignee="code-repo")
    tid, res = _dispatch_unassigned(global_default="")
    assert [(s[0], s[1]) for s in res.spawned] == [(tid, "code-repo")]
