#!/usr/bin/env python3
"""Copy one stopped Kanban SQLite board into its PostgreSQL board schema."""

from __future__ import annotations

import argparse
import contextlib
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli.kanban_db_postgres import copy_sqlite_to_postgres


def _upgraded_snapshot(source: Path, target: Path) -> None:
    """Copy and upgrade SQLite without ever opening the source writable."""
    source_connection = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    target_connection = sqlite3.connect(target, isolation_level=None)
    target_connection.row_factory = sqlite3.Row
    try:
        if source_connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError(f"SQLite integrity check failed for {source}")
        source_connection.backup(target_connection)
        target_connection.executescript(kb.SCHEMA_SQL)
        kbc._migrate_add_optional_columns(target_connection)
    finally:
        target_connection.close()
        source_connection.close()


def migrate_kanban_sqlite_to_postgres(
    source: Path,
    *,
    board: str | None = None,
    batch_size: int = 1000,
) -> dict[str, int]:
    """Migrate a board atomically while leaving the SQLite source unchanged."""
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Kanban SQLite database not found: {source}")
    if not kbc.uses_postgres():
        raise ValueError("Target Kanban root must set database.backend: postgres")

    with tempfile.TemporaryDirectory(prefix="hermes-kanban-migration-") as directory:
        snapshot = Path(directory) / "kanban.db"
        _upgraded_snapshot(source, snapshot)
        kb.init_db(board=board)
        with kbc.connect_closing(board=board) as target:
            return copy_sqlite_to_postgres(snapshot, target, batch_size=batch_size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Source kanban.db")
    parser.add_argument("--board", default=None, help="Target board slug (default: active board)")
    parser.add_argument(
        "--kanban-home",
        type=Path,
        default=None,
        help="Hermes root configured for PostgreSQL",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    if args.kanban_home is not None:
        home = str(args.kanban_home.expanduser().resolve())
        os.environ["HERMES_HOME"] = home
        os.environ["HERMES_KANBAN_HOME"] = home

    counts = migrate_kanban_sqlite_to_postgres(
        args.source,
        board=args.board,
        batch_size=args.batch_size,
    )
    for table, count in counts.items():
        print(f"{table}: {count} rows copied")
    print("Migration committed. The SQLite source was not changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
