"""Create routing must not triage assignees this home cannot judge, and must agree with the dispatcher."""

import argparse
import json
import shlex
from pathlib import Path

import pytest

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import profiles
from hermes_constants import mark_named_profile_deleted


def make_home(root: Path, name: str, *profile_names: str) -> Path:
    home = root / name
    for profile in profile_names:
        (home / "profiles" / profile).mkdir(parents=True)
        (home / "profiles" / profile / "config.yaml").write_text("{}\n")
    home.mkdir(exist_ok=True)
    return home


def create_status(command, capsys):
    parser = argparse.ArgumentParser()
    cli.build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["kanban", *shlex.split(command)])
    assert cli.kanban_command(args) == 0
    task = json.loads(capsys.readouterr().out)
    with kbc.connect_closing() as conn:
        return kb.get_task(conn, task["id"]).status


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
    return tmp_path


def test_shared_board_keeps_assignee_that_only_exists_in_another_home(isolated, monkeypatch, capsys):
    home_a = make_home(isolated, "homeA", "builder")
    home_b = make_home(isolated, "homeB", "sage")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(isolated / "shared" / "kanban.db"))
    monkeypatch.setenv("HERMES_HOME", str(home_a))
    kb.init_db()

    assert create_status("create card --assignee sage --json", capsys) == "ready"

    monkeypatch.setenv("HERMES_HOME", str(home_b))
    assert profiles.profile_exists("sage")


def test_home_local_board_still_triages_a_missing_assignee(isolated, monkeypatch, capsys):
    monkeypatch.setenv("HERMES_HOME", str(make_home(isolated, ".hermes", "builder")))
    kb.init_db()

    assert create_status("create card --assignee ghost --json", capsys) == "triage"


def test_identity_marker_only_profile_is_known_like_the_dispatcher_sees_it(isolated, monkeypatch, capsys):
    home = make_home(isolated, ".hermes", "builder")
    (home / "profiles" / "envonly").mkdir()
    (home / "profiles" / "envonly" / ".env").write_text("\n")
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.init_db()

    assert profiles.profile_exists("envonly")
    assert create_status("create card --assignee envonly --json", capsys) == "ready"


def test_tombstoned_profile_is_unknown_even_with_config(isolated, monkeypatch, capsys):
    home = make_home(isolated, ".hermes", "builder", "gone")
    monkeypatch.setenv("HERMES_HOME", str(home))
    mark_named_profile_deleted(home / "profiles" / "gone")
    kb.init_db()

    assert not profiles.profile_exists("gone")
    assert create_status("create card --assignee gone --json", capsys) == "triage"
