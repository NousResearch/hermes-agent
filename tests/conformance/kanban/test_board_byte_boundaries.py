"""The board's bytes refuse a hosted process at every arm that opens or copies them.

The shared board's hosted refusal cannot sit only in
``hermes_cli.kanban_db_connect.connect``: the two private arms it delegates to
(``_sqlite_connect`` / ``_open_readonly``) are importable, the export's staged-copy
helper opens a copy of the board, and the backup code's generic SQLite receiver
opens whatever database file it is handed — each would read the first agent's
private cards in a hosted process without crossing ``connect``. So the refusal
sits in the lowest primitives themselves: each connect arm refuses wherever the
process serves hosted tenants (whoever calls it, whatever path form it arrives
with), the staged-copy helper refuses likewise, and the backup receiver asks the
one board predicate (``hermes_cli.kanban_db.is_board_path``) whether the file it
is about to copy is a board, and refuses on its answer. The model's own
``read_file`` is not kanban's to govern: the hosted tool allowlist runs it in the
tenant's own sandbox (a lexical path policy, no host filesystem), so the board
path is unreachable through it — see
https://github.com/NousResearch/hermes-agent/pull/149924. On a single-user
machine nothing changes: the same arms open, export and back the shared root
exactly as today.
"""
import contextlib
import sqlite3
import sys
import types

import pytest

#: Both refusals say what they are; the board's own words for its boundary.
_REFUSAL = "refuses to open in a hosted process"


def _serve_hosted_tenants(monkeypatch):
    """This process reports itself hosted the way the boot profile does
    (``hermes_cli.hosted_mode.is_hosted`` — absent from a checkout without that module, which is
    why the shared-board boundary's guarded import answers False there)."""
    module = types.ModuleType("hermes_cli.hosted_mode")
    module.is_hosted = lambda: True
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.fixture
def seeded_shared_board(tmp_path, monkeypatch):
    """One agent's recognizable private card in the machine's one shared root,
    seeded while the process is not hosted. A second profile reading it is
    today's single-user handoff, and what a hosted process must never read
    through a low-level arm, a staged copy or a backup snapshot."""
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "shared"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    with kbc.connect_closing() as conn:
        kb.create_task(conn, title="TENANT-A-PRIVATE-BOARD-DATA", assignee="default")
    board_db = kb.kanban_db_path()
    return tmp_path, board_db


def _first_title(path) -> str:
    """The first task title as any raw reader would see it."""
    with contextlib.closing(sqlite3.connect(path)) as conn:
        return conn.execute("SELECT title FROM tasks").fetchone()[0]


@pytest.mark.parametrize("arm", ["read_write", "read_only"])
def test_hosted_the_private_connect_arms_refuse(seeded_shared_board, monkeypatch, arm):
    """``_sqlite_connect`` and ``_open_readonly`` are the lowest arms that open
    the board's database; imported and called directly — not through
    ``connect`` — each must refuse a hosted process before any board byte is
    read, and a write through the read-write arm never happens because the open
    itself refuses."""
    _root, board_db = seeded_shared_board
    from hermes_cli import kanban_db_connect as kbc
    _serve_hosted_tenants(monkeypatch)
    opener = kbc._sqlite_connect if arm == "read_write" else kbc._open_readonly
    with pytest.raises(LookupError, match=_REFUSAL):
        conn = opener(board_db)
        conn.close()  # unreachable; the refusal must precede the open
    if arm == "read_write":
        # The UPDATE the arm exists to serve: it never runs, because the arm
        # refuses to hand out a connection to a hosted process at all.
        with pytest.raises(LookupError, match=_REFUSAL):
            with contextlib.closing(kbc._sqlite_connect(board_db)) as conn:
                conn.execute("UPDATE tasks SET title = 'hosted write'")


def test_hosted_the_staged_export_copy_refuses(seeded_shared_board, monkeypatch):
    """The export's staged-copy helper opens a copy of the board's database; its
    normal target is a private file the export just staged, but a hosted process
    must not open board bytes through it whatever path it is handed."""
    _root, board_db = seeded_shared_board
    from hermes_cli.kanban_transfer import _open_staged_export_copy
    _serve_hosted_tenants(monkeypatch)
    with pytest.raises(LookupError, match=_REFUSAL):
        conn = _open_staged_export_copy(board_db)
        conn.close()  # unreachable; the refusal must precede the open


def test_hosted_the_backup_receiver_refuses_board_bytes(seeded_shared_board, monkeypatch):
    """``backup_sqlite._safe_copy_db`` receives board paths from the quick
    snapshot walker (``kanban.db``, ``kanban/boards/<slug>/kanban.db``) and opens
    them with a generic read-only connect; it must ask the one board predicate
    and refuse a hosted process before the board's bytes are opened or copied."""
    _root, board_db = seeded_shared_board
    from hermes_cli.backup_sqlite import _safe_copy_db
    _serve_hosted_tenants(monkeypatch)
    with pytest.raises(LookupError, match=_REFUSAL):
        assert not _safe_copy_db(board_db, _root / "copy.db")
    assert not (_root / "copy.db").exists() or \
        _first_title(_root / "copy.db") != "TENANT-A-PRIVATE-BOARD-DATA"


def test_hosted_the_quick_snapshot_refuses_the_board(seeded_shared_board, monkeypatch):
    """``create_quick_snapshot`` walked with the shared root as its home copies
    the default board's database into the snapshot; in a hosted process the
    snapshot must refuse the board rather than archive the first agent's
    private cards."""
    _root, board_db = seeded_shared_board
    from hermes_cli.backup import create_quick_snapshot
    _serve_hosted_tenants(monkeypatch)
    with pytest.raises(LookupError, match=_REFUSAL):
        create_quick_snapshot(hermes_home=board_db.parent)


def test_the_board_predicate_recognizes_every_board_location(tmp_path, monkeypatch):
    """``is_board_path`` is the one predicate the backup boundary asks: it
    recognizes the default board, an explicit ``HERMES_KANBAN_DB`` pin and any
    named board under the shared root's boards directory, and does not mistake a
    foreign SQLite file for a board."""
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "shared"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    with kbc.connect_closing() as conn:
        kb.create_task(conn, title="default card", assignee="default")
    kb.create_board("team")
    with kbc.connect_closing(board="team") as conn:
        kb.create_task(conn, title="team card", assignee="default")
    assert kb.is_board_path(kb.kanban_db_path())
    assert kb.is_board_path(kb.kanban_db_path(board="team"))
    pinned = tmp_path / "pinned.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(pinned))
    assert kb.is_board_path(pinned)
    monkeypatch.delenv("HERMES_KANBAN_DB")
    foreign = tmp_path / "state.db"
    foreign.write_bytes(b"SQLite format 3\x00")
    assert not kb.is_board_path(foreign)


def test_locally_the_same_arms_open_export_and_back_the_shared_root(seeded_shared_board):
    """The control: on a single-user machine — not hosted — the private arms
    open the shared board, the staged copy opens, the backup receiver copies
    the board's bytes and the quick snapshot archives it, exactly as today."""
    root, board_db = seeded_shared_board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.backup_sqlite import _safe_copy_db
    from hermes_cli.kanban_transfer import _open_staged_export_copy
    with contextlib.closing(kbc._sqlite_connect(board_db)) as conn:
        assert conn.execute("SELECT title FROM tasks").fetchone()[0]
    with contextlib.closing(kbc._open_readonly(board_db)) as conn:
        assert conn.execute("SELECT title FROM tasks").fetchone()[0]
    staged = root / "staged.db"
    with contextlib.closing(_open_staged_export_copy(staged)) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS tasks (title TEXT)")
    copied = root / "copy.db"
    assert _safe_copy_db(board_db, copied)
    assert _first_title(copied) == "TENANT-A-PRIVATE-BOARD-DATA"
