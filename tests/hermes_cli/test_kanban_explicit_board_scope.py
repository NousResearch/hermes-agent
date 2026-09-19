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
    # Hermetic vs ambient operator env: the simulated worker has path pins
    # only (HERMES_KANBAN_BOARD is set per-test where the scenario needs it;
    # HERMES_KANBAN_HOME would re-anchor kanban_home() outside tmp_path).
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_HOME", raising=False)
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


# ---------------------------------------------------------------------------
# boards surface under the explicit scope (critic round 1, t_4eff74eb):
# `boards list --json` enrichment (per-board db_path + task totals) must
# resolve through the same explicit scope as every other subcommand.
# ---------------------------------------------------------------------------


def test_cli_boards_list_json_with_board_flag_reports_truth(
    multi_board_home, capsys
):
    """Pinned worker + `--board alpha boards list --json`: alpha's entry must
    show alpha's OWN db_path and its own task total — not the pinned DB's."""
    from hermes_cli import kanban as kc

    home, default_task_id = multi_board_home
    with kb.scoped_explicit_board("alpha"):
        with kb.connect() as conn:
            alpha_task = kb.create_task(
                conn, title="alpha-board task", assignee="a"
            )
    alpha_db = str(
        (home / "kanban" / "boards" / "alpha" / "kanban.db").resolve()
    )

    args = _parse_kanban_args(
        ["kanban", "--board", "alpha", "boards", "list", "--json"]
    )
    assert kc.kanban_command(args) == 0
    boards = {b["slug"]: b for b in _read_json(capsys)}

    assert boards["alpha"]["db_path"] == alpha_db
    assert boards["alpha"]["total"] == 1  # alpha's own task, not the pin's
    assert boards["alpha"]["is_current"] is True  # inside the --board scope
    assert boards["default"]["total"] == 1
    assert boards["default"]["is_current"] is False

    # Negative control: with NO flag, the pinned worker view is unchanged —
    # pins win, so the default board reads the pinned DB and stays current.
    args = _parse_kanban_args(["kanban", "boards", "list", "--json"])
    assert kc.kanban_command(args) == 0
    boards = {b["slug"]: b for b in _read_json(capsys)}
    assert boards["default"]["is_current"] is True
    assert boards["default"]["db_path"] == str(
        (home / "kanban.db").expanduser()
    )
    assert boards["alpha"]["total"] == 1
    assert boards["alpha"]["is_current"] is False
    assert default_task_id and alpha_task  # keep both ids alive


def test_cli_boards_show_with_board_flag_shows_flagged_board(
    multi_board_home, capsys
):
    """`--board alpha boards show` must describe alpha (its db_path and its
    own counts), not the pinned board the worker env points at."""
    from hermes_cli import kanban as kc

    home, _ = multi_board_home
    with kb.scoped_explicit_board("alpha"):
        with kb.connect() as conn:
            kb.create_task(conn, title="alpha-board task", assignee="a")

    args = _parse_kanban_args(
        ["kanban", "--board", "alpha", "boards", "show"]
    )
    assert kc.kanban_command(args) == 0
    out = capsys.readouterr().out
    alpha_db = str(
        (home / "kanban" / "boards" / "alpha" / "kanban.db").resolve()
    )
    assert "Current board: alpha" in out
    assert alpha_db in out
    assert "Tasks:        1 total" in out


def test_cli_boards_list_json_no_flag_worker_parity(multi_board_home, capsys):
    """No-flag `boards list --json` from a pinned worker is byte-identical to
    the pre-fix resolution: pinned DB answers, default stays current."""
    from hermes_cli import kanban as kc

    home, _ = multi_board_home
    args = _parse_kanban_args(["kanban", "boards", "list", "--json"])
    assert kc.kanban_command(args) == 0
    boards = {b["slug"]: b for b in _read_json(capsys)}
    assert boards["default"]["db_path"] == str(
        (home / "kanban.db").expanduser()
    )
    assert boards["default"]["total"] == 1
    assert boards["default"]["is_current"] is True


# ---------------------------------------------------------------------------
# Round 3 (critic r2 RED): bootstrap-by-flag must stay legal
# ---------------------------------------------------------------------------


def test_cli_board_flag_bootstrap_boards_create_on_missing_board(
    multi_board_home, capsys
):
    """`--board ghost boards create ghost` must SUCCEED even though 'ghost'
    does not exist when the command starts.

    The explicit scope's board-exists pre-check (a typo guard for task-level
    subcommands) must NOT gate `boards ...` subcommands: bootstrap-by-flag
    was legal before the explicit scope existed (base 6051590677 dispatched
    `boards` before that check) and fleet automation uses exactly this shape
    — create the board, then seed it, all under one --board prefix. The r2
    dispatch reorder accidentally put `boards` AFTER the check, making the
    command fail with the self-contradictory message telling you to run the
    very command just executed (t_4eff74eb critic round 2).
    """
    from hermes_cli import kanban as kc

    home, _ = multi_board_home
    assert not kb.board_exists("ghost")

    args = _parse_kanban_args(
        ["kanban", "--board", "ghost", "boards", "create", "ghost"]
    )
    assert kc.kanban_command(args) == 0
    out = capsys.readouterr().out
    assert "Board 'ghost' created." in out
    assert "does not exist" not in out

    # The board now exists at its OWN directory (board.json + own DB)...
    ghost_dir = home / "kanban" / "boards" / "ghost"
    assert (ghost_dir / "board.json").exists()
    assert (ghost_dir / "kanban.db").exists()

    # ...and the pinned default board was NOT touched by the bootstrap.
    with kb.connect() as conn:  # pins → default board
        titles = [
            r["title"]
            for r in conn.execute("SELECT title FROM tasks").fetchall()
        ]
    assert titles == ["default-board task"]


def test_cli_board_flag_bootstrap_boards_import_on_missing_board(
    multi_board_home, tmp_path, capsys
):
    """`--board fresh boards import <archive> --as fresh` must bootstrap the
    same way: `boards import` creates its own board by slug, so the missing-
    flag-target pre-check must not gate it either."""
    import tarfile

    from hermes_cli import kanban as kc

    home, _ = multi_board_home
    # Build a minimal importable archive via the real exporter: export the
    # existing `alpha` board, then import it under a DIFFERENT slug that
    # does not exist yet — with an explicit --board pinned to that slug.
    archive = home / "alpha.tar.gz"
    export_args = _parse_kanban_args(
        [
            "kanban", "--board", "alpha", "boards", "export", "alpha",
            "-o", str(archive),
        ]
    )
    assert kc.kanban_command(export_args) == 0
    assert archive.exists()
    capsys.readouterr()  # discard export output; isolate the import's JSON
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert names, "export produced an empty archive"

    assert not kb.board_exists("fresh")
    import_args = _parse_kanban_args(
        [
            "kanban", "--board", "fresh", "boards", "import",
            str(archive), "--as", "fresh", "--json",
        ]
    )
    assert kc.kanban_command(import_args) == 0
    res = _read_json(capsys)
    assert res["board"] == "fresh"
    assert res["renamed"] is False

    fresh_dir = home / "kanban" / "boards" / "fresh"
    assert (fresh_dir / "board.json").exists()
    assert (fresh_dir / "kanban.db").exists()
    assert kb.board_exists("fresh")
    # Sibling boards untouched by the bootstrap import.
    assert kb.board_exists("alpha")
    assert not kb.board_exists("ghost")


def test_cli_task_level_create_under_flag_still_requires_existing_board(
    multi_board_home, capsys
):
    """Negative control for the r3 exemption: the typo guard still gates
    TASK-level subcommands. `--board ghost create ...` must keep failing
    with the guidance message (typoed slugs must not silently create empty
    boards via the task surface) — only `boards ...` bootstraps."""
    from hermes_cli import kanban as kc

    home, task_id = multi_board_home
    args = _parse_kanban_args(
        [
            "kanban", "--board", "ghost", "create",
            "should not land anywhere", "--assignee", "critic", "--json",
        ]
    )
    assert kc.kanban_command(args) == 1  # pre-check still enforced
    captured = capsys.readouterr()
    assert "board 'ghost' does not exist" in captured.err
    assert "boards create ghost" in captured.err

    # Nothing was written anywhere: no ghost board materialized...
    assert not kb.board_exists("ghost")
    # ...and the card did not silently land on the pinned default DB.
    with kb.connect() as conn:  # pins → default board
        titles = [
            r["title"]
            for r in conn.execute("SELECT title FROM tasks").fetchall()
        ]
    assert titles == ["default-board task"]
    assert task_id  # keep the fixture id alive
