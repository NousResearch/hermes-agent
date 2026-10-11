"""Legacy board directories without ``board.json`` (#135556).

Commit 63e5656409 made ``board.json`` the sole identity marker, but nothing
migrates board directories created by the pre-metadata era: they hold a real
``kanban.db`` (with tasks) and no ``board.json``, so ``board_exists()`` and
``list_boards()`` dropped them permanently — discovery must honour the
documented rule ("dirs holding a kanban.db **or** board.json").

Discriminator against the #43243 stub resurrection: a DB-only directory is a
real board only when its kanban.db is a live database carrying at least one
task row. Stub shapes (0-byte file, schema-only DB) and archived tombstones
stay excluded.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

# Ensure the worktree (not the stale global clone) is first on sys.path.
_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb


@pytest.fixture
def fresh_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with no prior kanban state."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in (
        "HERMES_KANBAN_DB",
        "HERMES_KANBAN_WORKSPACES_ROOT",
        "HERMES_KANBAN_HOME",
        "HERMES_KANBAN_BOARD",
    ):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants

        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    return home


def _seed_board_db(slug: str, *, tasks: int, schema_only: bool = False) -> Path:
    """Create ``boards/<slug>/`` holding only a kanban.db (no board.json).

    ``tasks > 0``            → pre-metadata-era board: schema + task rows.
    ``schema_only=True``     → #43243 stub: schema, zero tasks.
    default (``tasks == 0``) → true 0-byte stub: an empty file, never opened.
    """
    d = kb.board_dir(slug)
    d.mkdir(parents=True, exist_ok=True)
    db = d / "kanban.db"
    if schema_only:
        conn = sqlite3.connect(db)
        conn.executescript(kb.SCHEMA_SQL)
        conn.commit()
        conn.close()
    elif tasks > 0:
        conn = sqlite3.connect(db)
        conn.executescript(kb.SCHEMA_SQL)
        for i in range(tasks):
            conn.execute(
                "INSERT INTO tasks (id, title, status, created_at) VALUES (?, ?, ?, ?)",
                (f"t-legacy-{i}", f"Legacy task {i}", "todo", 1_700_000_000),
            )
        conn.commit()
        conn.close()
    else:
        db.touch()  # true 0-byte shape: stays byte-identical to an unopened stub
    return d


# ---------------------------------------------------------------------------
# Legacy board dirs must stay visible / parseable (#135556)
# ---------------------------------------------------------------------------


class TestLegacyBoardDirVisible:
    def test_board_with_tasks_is_discoverable_and_parseable(self, fresh_home):
        _seed_board_db("legacy", tasks=1)

        # Existence + discovery: the documented "kanban.db or board.json" rule.
        assert kb.board_exists("legacy") is True
        entries = {meta["slug"]: meta for meta in kb.list_boards()}
        assert "legacy" in entries
        # Parseable: discovery yields synthesized display metadata.
        assert entries["legacy"]["name"] == "Legacy"

        # The board's data opens and reads back (direct open always worked;
        # pinned so the fix never breaks the parse path it claims to heal).
        from hermes_cli import kanban_db_connect as kbc

        with kbc.connect_closing(board="legacy") as conn:
            count = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        assert count == 1

    def test_board_with_tasks_still_found_when_board_json_arrives_later(self, fresh_home):
        # A legacy dir that later gains board.json must not double-list or break.
        _seed_board_db("legacy", tasks=1)
        assert kb.board_exists("legacy") is True
        kb.write_board_metadata("legacy", name="Legacy Named")
        entries = [meta["slug"] for meta in kb.list_boards()]
        assert entries.count("legacy") == 1
        meta = {m["slug"]: m for m in kb.list_boards()}["legacy"]
        assert meta["name"] == "Legacy Named"


# ---------------------------------------------------------------------------
# Reverse invariants: stub resurrection (#43243) and tombstones stay dead
# ---------------------------------------------------------------------------


class TestStubShapesStayInvisible:
    @pytest.mark.parametrize("shape", ["zero-byte", "schema-only"])
    def test_db_only_stub_without_tasks_is_not_a_board(self, fresh_home, shape):
        _seed_board_db("ghost", tasks=0, schema_only=(shape == "schema-only"))

        assert kb.board_exists("ghost") is False
        assert "ghost" not in [meta["slug"] for meta in kb.list_boards()]

    def test_archived_tombstone_keeps_current_semantics(self, fresh_home):
        kb.create_board("gone", name="Gone")
        kb.remove_board("gone", archive=True)
        # Tombstone shape: board.json with archived=true at the original slug.
        assert kb.read_board_metadata("gone")["archived"] is True

        # Original semantics (must survive the #135556 fix bit-for-bit):
        # the tombstone's board.json satisfies board_exists / list_boards
        # (flagged ``archived``), only the un-archived views exclude it, and
        # a stale open falls through to default instead of resurrecting.
        assert kb.board_exists("gone") is True
        listed = {meta["slug"]: meta for meta in kb.list_boards()}
        assert listed["gone"]["archived"] is True
        live = [meta["slug"] for meta in kb.list_boards(include_archived=False)]
        assert "gone" not in live
        assert kb.get_current_board() != "gone"
