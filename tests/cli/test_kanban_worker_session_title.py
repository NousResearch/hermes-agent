"""A kanban worker names its session after the card it is working.

Contract: the title comes from the task row, collides are resolved through the
lineage helper (a re-dispatched card must not end up untitled), and a non-kanban
run is left alone.
"""

from __future__ import annotations

import types

import pytest

import cli as cli_mod


class _FakeDB:
    def __init__(self, taken=()):
        self._taken = set(taken)

    def get_session_by_title(self, title):
        return {"id": "other"} if title in self._taken else None

    def get_next_title_in_lineage(self, title):
        return f"{title} #2"


@pytest.fixture
def fake_task(monkeypatch):
    """Point the seeder at an in-memory task instead of a real board DB.

    Production does ``from hermes_cli import kanban_db`` inside the function, so the module
    object's attributes are the seam — patching ``sys.modules`` would be bypassed by the
    package-attribute lookup.
    """
    import contextlib

    from hermes_cli import kanban_db, kanban_db_connect

    monkeypatch.setattr(
        kanban_db_connect, "connect_closing", lambda **kw: contextlib.nullcontext(object()))

    def _install(title):
        monkeypatch.setattr(kanban_db, "get_task", lambda conn, tid: types.SimpleNamespace(title=title))

    return _install


def _cli(db=None):
    return types.SimpleNamespace(_pending_title=None, _session_db=db)


def test_title_taken_from_the_task(monkeypatch, fake_task):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc")
    fake_task("Fix the swap modal")
    cli = _cli(_FakeDB())
    cli_mod._seed_kanban_session_title(cli)
    assert cli._pending_title == "Fix the swap modal"


def test_second_run_of_the_same_card_is_not_left_untitled(monkeypatch, fake_task):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc")
    fake_task("Fix the swap modal")
    cli = _cli(_FakeDB(taken={"Fix the swap modal"}))
    cli_mod._seed_kanban_session_title(cli)
    assert cli._pending_title == "Fix the swap modal #2"


def test_non_kanban_run_is_untouched(monkeypatch, fake_task):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    fake_task("Fix the swap modal")
    cli = _cli(_FakeDB())
    cli_mod._seed_kanban_session_title(cli)
    assert cli._pending_title is None


def test_explicit_title_wins_over_the_card(monkeypatch, fake_task):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc")
    fake_task("Fix the swap modal")
    cli = _cli(_FakeDB())
    cli._pending_title = "chosen by the operator"
    cli_mod._seed_kanban_session_title(cli)
    assert cli._pending_title == "chosen by the operator"


def test_board_read_failure_does_not_block_the_worker(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_abc")
    from hermes_cli import kanban_db

    def _boom(conn, tid):
        raise RuntimeError("db gone")

    monkeypatch.setattr(kanban_db, "get_task", _boom)
    cli = _cli(_FakeDB())
    cli_mod._seed_kanban_session_title(cli)
    assert cli._pending_title is None
