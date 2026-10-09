"""Create routing follows the profiles present in the isolated home."""

import argparse
import json
import shlex
from pathlib import Path

import pytest

from hermes_cli import kanban as cli
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


def create_from_cli(command, capsys):
    parser = argparse.ArgumentParser()
    cli.build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["kanban", *shlex.split(command)])
    assert cli.kanban_command(args) == 0
    captured = capsys.readouterr()
    return json.loads(captured.out), captured.err


@pytest.fixture
def board_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    profile = home / "profiles" / "builder"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text("{}\n")
    kb.init_db()
    return home


@pytest.mark.parametrize("assignee,expected", [
    ("missing", "triage"), ("builder", "blocked"), (None, "blocked"),
])
def test_cli_create_assignee_routes_and_warns_with_json(board_home, capsys, assignee, expected):
    with kbc.connect_closing() as conn:
        parent = kb.create_task(conn, title="open parent")
    command = f"create child --parent {parent} --initial-status blocked --json"
    if assignee:
        command += f" --assignee {assignee}"
    result, stderr = create_from_cli(command, capsys)
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, result["id"])
        comments = kb.list_comments(conn, task.id)
    assert task.status == expected
    if assignee == "missing":
        assert "missing" in stderr and "builder" in stderr
        assert any("missing" in comment.body and "triage" in comment.body for comment in comments)
    else:
        assert not stderr
        assert not comments


@pytest.mark.parametrize("unavailable", [False, True])
def test_cli_empty_profile_enumeration_keeps_assignee(board_home, monkeypatch, capsys, unavailable):
    def profiles():
        if unavailable:
            raise OSError("profile store unavailable")
        return []

    monkeypatch.setattr(kb, "list_profiles_on_disk", lambda **kwargs: profiles())
    result, stderr = create_from_cli("create card --assignee missing --json", capsys)
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, result["id"])
    assert task.status == "ready"
    assert not stderr


def test_cli_partial_profile_inventory_keeps_assignee(board_home, monkeypatch, capsys):
    original = Path.iterdir

    def failing_iterdir(path):
        if path == board_home / "profiles":
            raise OSError("profile directory unavailable")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", failing_iterdir)
    result, stderr = create_from_cli("create card --assignee builder --json", capsys)
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, result["id"])
        assert not kb.list_comments(conn, task.id)
    assert task.status == "ready"
    assert task.assignee == "builder"
    assert not stderr


@pytest.mark.parametrize("original_assignee,expected_status", [
    ("builder", "ready"), ("missing", "triage"),
])
def test_cli_idempotent_replay_has_no_new_warning(board_home, capsys, original_assignee, expected_status):
    key = f"replay-{original_assignee}"
    first, first_warning = create_from_cli(
        f"create original --assignee {original_assignee} --idempotency-key {key} --json", capsys)
    replay, replay_warning = create_from_cli(
        f"create replay --assignee missing --idempotency-key {key} --json", capsys)
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, first["id"])
        comments = kb.list_comments(conn, task.id)
    assert replay["id"] == first["id"]
    assert task.status == expected_status
    assert task.assignee == original_assignee
    assert len(comments) == (1 if original_assignee == "missing" else 0)
    assert bool(first_warning) == (original_assignee == "missing")
    assert not replay_warning
