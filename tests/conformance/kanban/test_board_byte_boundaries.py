"""The board's bytes refuse a hosted process at every arm that opens or copies them.

The shared board's hosted refusal cannot sit only in
``hermes_cli.kanban_db_connect.connect``: the two private arms it delegates to
(``_sqlite_connect`` / ``_open_readonly``) are importable, and the export's
staged-copy helper opens a copy of the board — each would read the first
agent's private cards in a hosted process without crossing ``connect``. So the
refusal sits in the lowest primitives themselves: each connect arm refuses
wherever the process serves hosted tenants (whoever calls it, whatever path
form it arrives with), and the staged-copy helper refuses likewise. The
model's own ``read_file`` is not kanban's to govern: the hosted tool allowlist
runs it in the tenant's own sandbox (a lexical path policy, no host
filesystem), so the board path is unreachable through it — see
https://github.com/NousResearch/hermes-agent/pull/149924. Operator commands
that run outside the gateway process (backups, restores, the doctor) are not
part of this boundary: they never run inside a hosted gateway. On a
single-user machine nothing changes: the same arms open and export the shared
root exactly as today.
"""
import contextlib
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
    through a low-level arm or a staged copy."""
    from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "shared"))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    with kbc.connect_closing() as conn:
        kb.create_task(conn, title="TENANT-A-PRIVATE-BOARD-DATA", assignee="default")
    board_db = kb.kanban_db_path()
    return tmp_path, board_db


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


def test_locally_the_same_arms_open_and_export_the_shared_root(seeded_shared_board):
    """The control: on a single-user machine — not hosted — the private arms
    open the shared board and the staged copy opens, exactly as today."""
    root, board_db = seeded_shared_board
    from hermes_cli import kanban_db_connect as kbc
    from hermes_cli.kanban_transfer import _open_staged_export_copy
    with contextlib.closing(kbc._sqlite_connect(board_db)) as conn:
        assert conn.execute("SELECT title FROM tasks").fetchone()[0]
    with contextlib.closing(kbc._open_readonly(board_db)) as conn:
        assert conn.execute("SELECT title FROM tasks").fetchone()[0]
    staged = root / "staged.db"
    with contextlib.closing(_open_staged_export_copy(staged)) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS tasks (title TEXT)")
