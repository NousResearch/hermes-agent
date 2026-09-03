"""Regression tests: explicit ``--board`` outranks worker/env path pins (t_4eff74eb).

Scout finding (2026-09-03, probes t_433143bf): from a dispatched worker session
(``HERMES_KANBAN_DB`` + ``HERMES_KANBAN_BOARD`` pinned), a CLI call with an
explicit ``--board other`` flag was silently diverted back to the pinned
board's DB — ``create`` wrote a card to the wrong board,
``notify-subscribe`` wrote a subscription that could never observe its task's
events, and ``list`` displayed the wrong board's tasks. Root cause: the CLI
flag only engaged ``scoped_current_board`` (which outranks
``HERMES_KANBAN_BOARD``) while the path-pin env vars (``HERMES_KANBAN_DB`` /
``HERMES_KANBAN_WORKSPACES_ROOT`` / ``HERMES_KANBAN_ATTACHMENTS_ROOT``)
shortcut resolution unconditionally.

Fix: ``scoped_explicit_board`` (engaged by the CLI ``--board`` flag)
additionally flags the context as explicit; inside that scope the three path
pins are ignored for path resolution. Default resolution (no ``--board``
flag) is unchanged so worker env semantics are preserved.
"""

from __future__ import annotations

import argparse
import json

import pytest

from hermes_cli import kanban_db as kb


def _parse_kanban_args(tokens):
    parser = argparse.ArgumentParser(prog="hermes", add_help=False)
    sub = parser.add_subparsers(dest="command")
    from hermes_cli import kanban as kc

    kc.build_parser(sub)
    return parser.parse_args(tokens)


def _read_json(capsys):
    captured = capsys.readouterr()
    return json.loads(captured.out)


@pytest.fixture
def multi_board_home(tmp_path, monkeypatch):
    """Isolated hermes home simulating a pinned worker session.

    The env pins (HERMES_KANBAN_DB / _WORKSPACES_ROOT / _ATTACHMENTS_ROOT)
    all point at the DEFAULT board's files — exactly what the dispatcher
    injects into a default-board worker. ``alpha`` is the "other" board an
    operator targets with an explicit ``--board alpha`` flag.

    NOTE: alpha-side setup and reads in these tests MUST run inside
    ``kb.scoped_explicit_board("alpha")`` — the pins outrank a plain
    ``connect(board=...)`` outside the explicit scope (that is the bug).
    """
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(home / "kanban.db"))
    monkeypatch.setenv(
        "HERMES_KANBAN_WORKSPACES_ROOT", str(home / "kanban" / "workspaces")
    )
    monkeypatch.setenv(
        "HERMES_KANBAN_ATTACHMENTS_ROOT", str(home / "kanban" / "attachments")
    )
    with kb.scoped_explicit_board("alpha"):
        kb.create_board("alpha")
    with kb.connect() as conn:  # pinned → default board
        task_id = kb.create_task(conn, title="default-board task", assignee="a")
    return home, task_id


# ---------------------------------------------------------------------------
# scoped_explicit_board unit behavior
# ---------------------------------------------------------------------------


def test_scoped_explicit_board_overrides_path_pin_env(multi_board_home):
    home, _ = multi_board_home
    alpha_db = (home / "kanban" / "boards" / "alpha" / "kanban.db").resolve()
    with kb.scoped_explicit_board("alpha"):
        assert kb.kanban_db_path() == alpha_db
        assert kb.workspaces_root() == (
            home / "kanban" / "boards" / "alpha" / "workspaces"
        )
        assert kb.attachments_root() == (
            home / "kanban" / "boards" / "alpha" / "attachments"
        )
        assert kb.get_current_board() == "alpha"
    # Outside the scope the pins are back in force.
    assert kb.kanban_db_path() == (home / "kanban.db").expanduser()


def test_scoped_current_board_keeps_path_pin_precedence(multi_board_home):
    """The old scope must stay exactly as it was: pins outrank it."""
    home, _ = multi_board_home
    with kb.scoped_current_board("alpha"):
        assert kb.kanban_db_path() == (home / "kanban.db").expanduser()
        assert kb.get_current_board() == "alpha"


def test_explicit_scope_nesting_and_reset(multi_board_home):
    home, _ = multi_board_home
    alpha_db = (home / "kanban" / "boards" / "alpha" / "kanban.db").resolve()
    with kb.scoped_explicit_board("alpha"):
        with kb.scoped_current_board("default"):
            # Inner legacy scope clears the board override but NOT the
            # explicit flag (it never touches it) — pin still outranked.
            assert kb.kanban_db_path() == alpha_db
    assert kb.kanban_db_path() == (home / "kanban.db").expanduser()
    assert not kb._explicit_board_scope_engaged()


# ---------------------------------------------------------------------------
# Worker-env parity: pins win whenever no explicit board is engaged
# ---------------------------------------------------------------------------


def test_worker_env_without_flag_still_pinned(multi_board_home, monkeypatch):
    """No --board flag → worker resolution unchanged (env pin wins)."""
    home, _ = multi_board_home
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "alpha")  # even a board pin
    assert kb.kanban_db_path() == (home / "kanban.db").expanduser()
    assert kb.get_current_board() == "alpha"


def test_scoped_explicit_board_overrides_board_env_pin_too(
    multi_board_home, monkeypatch
):
    home, _ = multi_board_home
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "alpha")
    with kb.scoped_explicit_board("alpha"):
        assert kb.kanban_db_path() == (
            home / "kanban" / "boards" / "alpha" / "kanban.db"
        ).resolve()


# ---------------------------------------------------------------------------
# End-to-end through kanban_command — the entry both `hermes kanban` and
# /kanban share; reproduces the scout's live findings
# ---------------------------------------------------------------------------


def test_cli_list_with_board_flag_shows_flagged_board(multi_board_home, capsys):
    from hermes_cli import kanban as kc

    home, _ = multi_board_home
    with kb.scoped_explicit_board("alpha"):
        with kb.connect() as conn:
            kb.create_task(conn, title="alpha-board task", assignee="a")
    # Worker session shape: pins point at default, flag targets alpha.
    args = _parse_kanban_args(["kanban", "--board", "alpha", "list", "--json"])
    assert kc.kanban_command(args) == 0
    rows = _read_json(capsys)
    assert [r["title"] for r in rows] == ["alpha-board task"]


def test_cli_create_with_board_flag_lands_on_flagged_board(
    multi_board_home, capsys
):
    from hermes_cli import kanban as kc

    home, _ = multi_board_home
    args = _parse_kanban_args(
        [
            "kanban", "--board", "alpha", "create",
            "cross-board card", "--assignee", "critic", "--json",
        ]
    )
    assert kc.kanban_command(args) == 0
    new_id = _read_json(capsys)["id"]
    # Card exists on alpha's DB...
    with kb.scoped_explicit_board("alpha"):
        with kb.connect() as conn:
            titles = [
                r["title"]
                for r in conn.execute(
                    "SELECT title FROM tasks ORDER BY created_at"
                ).fetchall()
            ]
    # ...and NOT on the pinned default DB (fixture's card only).
    with kb.connect() as conn:
        default_titles = [
            r["title"]
            for r in conn.execute(
                "SELECT title FROM tasks ORDER BY created_at"
            ).fetchall()
        ]
    assert "cross-board card" in titles
    assert default_titles == ["default-board task"]


def test_cli_notify_subscribe_with_board_flag_writes_flagged_board(
    multi_board_home, capsys
):
    from hermes_cli import kanban as kc

    home, task_id = multi_board_home
    # The task lives on DEFAULT (the pinned board); an operator (or fleet
    # automation) subscribing to it from another board's worker session
    # passes --board default explicitly — the flag must win.
    args = _parse_kanban_args(
        [
            "kanban", "--board", "default", "notify-subscribe",
            task_id, "--platform", "telegram", "--chat-id", "123",
        ]
    )
    assert kc.kanban_command(args) == 0
    with kb.connect() as conn:  # pinned → default board DB
        subs = conn.execute(
            "SELECT task_id, platform, chat_id FROM kanban_notify_subs"
        ).fetchall()
    assert [(s["task_id"], s["platform"], int(s["chat_id"])) for s in subs] == [
        (task_id, "telegram", 123)
    ]
    # No stray sub landed on alpha's DB.
    with kb.scoped_explicit_board("alpha"):
        with kb.connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM kanban_notify_subs"
            ).fetchone()[0]
    assert count == 0


def test_cli_board_flag_engages_explicit_scope_even_when_guard_denies(
    multi_board_home, monkeypatch
):
    """The delegated-child mutation guard denies the WRITE before the scope
    is engaged — resolution must not leak the pinned DB either way; here we
    assert the denied call errors instead of silently writing the pin."""
    from hermes_cli import kanban as kc

    home, _ = multi_board_home
    monkeypatch.setenv("HERMES_DELEGATED_CHILD_CONTEXT", "1")
    args = _parse_kanban_args(
        [
            "kanban", "--board", "alpha", "create",
            "child-mutation card", "--assignee", "critic",
        ]
    )
    assert kc.kanban_command(args) == 1  # denied — no silent wrong-board write
    with kb.connect() as conn:  # pinned → default board DB
        titles = [
            r["title"]
            for r in conn.execute("SELECT title FROM tasks").fetchall()
        ]
    assert "child-mutation card" not in titles
