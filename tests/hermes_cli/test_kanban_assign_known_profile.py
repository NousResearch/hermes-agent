"""CLI assign/reassign must refuse names that are not known assignees.

A typo like ``critic-typo`` used to be written onto the card with exit 0,
leaving review work permanently unclaimable. Known names are on-disk
profiles (``profile_exists`` / ``list_profiles_on_disk``) or names already
present as a board assignee. Unassign tokens stay allowed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _write_profile(home: Path, name: str) -> None:
    profile_dir = home / "profiles" / name
    profile_dir.mkdir(parents=True)
    (profile_dir / "config.yaml").write_text("{}\n", encoding="utf-8")


def _assign(task_id: str, profile: str) -> int:
    return kc._cmd_assign(argparse.Namespace(task_id=task_id, profile=profile))


def _reassign(task_id: str, profile: str) -> int:
    return kc._cmd_reassign(
        argparse.Namespace(
            task_id=task_id, profile=profile, reclaim=False, reason=None,
        )
    )


def _assignee(task_id: str) -> str | None:
    with kbc.connect_closing() as conn:
        task = kb.get_task(conn, task_id)
    assert task is not None
    return task.assignee


def test_assign_rejects_unknown_profile_typo(kanban_home, capsys):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="review card", assignee="default")

    rc = _assign(tid, "critic-typo")

    err = capsys.readouterr().err
    assert rc == 2
    assert "unknown profile" in err
    assert "critic-typo" in err
    assert _assignee(tid) == "default"


def test_reassign_rejects_unknown_profile_typo(kanban_home, capsys):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="review card", assignee="default")

    rc = _reassign(tid, "critic-typo")

    err = capsys.readouterr().err
    assert rc == 2
    assert "unknown profile" in err
    assert _assignee(tid) == "default"


def test_assign_accepts_on_disk_profile(kanban_home, capsys):
    _write_profile(kanban_home, "critic")
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="review card")

    rc = _assign(tid, "critic")

    out = capsys.readouterr().out
    assert rc == 0
    assert "Assigned" in out
    assert _assignee(tid) == "critic"


def test_assign_unassign_tokens_still_work(kanban_home):
    with kbc.connect_closing() as conn:
        tid = kb.create_task(conn, title="review card", assignee="default")

    for token in ("none", "-", "null", "NONE"):
        with kbc.connect_closing() as conn:
            assert kb.assign_task(conn, tid, "default") is True
        rc = _assign(tid, token)
        assert rc == 0, token
        assert _assignee(tid) is None


def test_assign_accepts_name_already_on_the_board(kanban_home, capsys):
    with kbc.connect_closing() as conn:
        kb.create_task(conn, title="already assigned", assignee="legacy-worker")
        tid = kb.create_task(conn, title="retarget me")

    rc = _assign(tid, "legacy-worker")

    assert rc == 0
    assert "Assigned" in capsys.readouterr().out
    assert _assignee(tid) == "legacy-worker"
    assert not (kanban_home / "profiles" / "legacy-worker").exists()


# ---------------------------------------------------------------------------
# Write-time gate on create (#99284): a card created for a profile that does
# not exist can never be dispatched, so the CLI refuses it before the write.
# ---------------------------------------------------------------------------

def _create(title: str, assignee: str | None) -> int:
    return kc._cmd_create(argparse.Namespace(
        title=title, body=None, workspace=None, assignee=assignee,
        created_by=None, tenant=None, priority=0, parent=[], json=False,
    ))


def test_create_rejects_unknown_profile(kanban_home, capsys):
    rc = _create("starving card", "arbiter")

    err = capsys.readouterr().err
    assert rc == 2
    assert "unknown profile" in err and "arbiter" in err
    with kbc.connect_closing() as conn:
        assert [t for t in kb.list_tasks(conn, limit=100) if t.title == "starving card"] == []


def test_create_accepts_on_disk_profile(kanban_home, capsys):
    _write_profile(kanban_home, "qa-worker")

    rc = _create("follow-up", "qa-worker")

    out = capsys.readouterr().out
    assert rc == 0
    assert "Created" in out
    with kbc.connect_closing() as conn:
        task = next(t for t in kb.list_tasks(conn, limit=100) if t.title == "follow-up")
        assert task.assignee == "qa-worker"


def test_create_without_assignee_still_works(kanban_home, capsys):
    rc = _create("unassigned seed", None)

    assert rc == 0
    assert "Created" in capsys.readouterr().out
