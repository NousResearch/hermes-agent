"""Tests for hermes_cli.ideas_db."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import ideas_db as db


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def test_create_list_show_update_delete_roundtrip(fresh_home):
    idea = db.create_idea(title="Test idea", body="# Draft\n\nHello.", board="default")
    assert idea["id"].startswith("i_")
    assert idea["board"] == "default"
    assert idea["status"] == "draft"
    assert "Hello." in idea["body"]
    assert Path(idea["file_path"]).is_file()

    listed = db.list_ideas(board="default")
    assert len(listed["ideas"]) == 1

    shown = db.get_idea(idea["id"])
    assert shown["title"] == "Test idea"

    updated = db.update_idea(idea["id"], status="active", summary="short")
    assert updated["status"] == "active"
    assert updated["summary"] == "short"

    dup = db.duplicate_idea(idea["id"])
    assert dup["title"].endswith("copy")
    assert len(db.list_ideas(board="default")["ideas"]) == 2

    db.delete_idea(idea["id"])
    with pytest.raises(db.IdeaNotFoundError):
        db.get_idea(idea["id"])


def test_list_ideas_all_boards(fresh_home):
    db.create_idea(title="On default", board="default")
    db.create_idea(title="On other", board="roguelike-td")
    result = db.list_ideas_all_boards()
    assert result["count"] == 2
    assert set(result["boards"]) == {"default", "roguelike-td"}


def test_ideas_boards_without_json_flag(fresh_home, capsys):
    """Bare ``ideas boards`` must not require a --json attribute on the namespace."""
    import argparse

    from hermes_cli.ideas import ideas_command

    args = argparse.Namespace(ideas_action="boards", boards_action=None, board=None)
    assert ideas_command(args) == 0
    out = capsys.readouterr().out
    assert "default" in out
