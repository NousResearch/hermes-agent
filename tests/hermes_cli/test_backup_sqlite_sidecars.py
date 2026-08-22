"""A SQLite store whose filename isn't ``.db`` must not be backed up raw.

The backup takes a consistent ``sqlite3.backup()`` snapshot of each database
and drops the live WAL / shared-memory / rollback-journal, because — as the
skip list's own comment puts it — shipping them together "would pair a fresh
snapshot with stale sidecar state and produce a torn restore on the next
open".

Both halves were keyed on the ``.db`` extension, so the observability metrics
store (``metrics.sqlite3``) fell through both: copied byte-for-byte while its
sidecars rode along.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_cli.backup as backup


@pytest.mark.parametrize(
    "name",
    [
        "state.db-wal",
        "state.db-shm",
        "state.db-journal",
        "metrics.sqlite3-wal",
        "metrics.sqlite3-shm",
        "metrics.sqlite3-journal",
        "board.sqlite-wal",
    ],
)
def test_sidecars_are_excluded_for_every_database_extension(name):
    assert backup._should_exclude(Path(name)) is True


@pytest.mark.parametrize(
    "name",
    ["state.db", "metrics.sqlite3", "board.sqlite", "config.yaml", "SOUL.md"],
)
def test_databases_and_ordinary_files_are_kept(name):
    assert backup._should_exclude(Path(name)) is False


@pytest.mark.parametrize(
    "name,expected",
    [
        ("state.db", True),
        ("metrics.sqlite3", True),
        ("board.sqlite", True),
        ("notes.md", False),
        ("archive.tar.gz", False),
    ],
)
def test_snapshot_covers_every_database_extension(name, expected):
    """The consistent-snapshot path must recognise the same set."""
    assert backup._is_sqlite_db_path(Path(name)) is expected


def test_sidecar_skip_does_not_swallow_unrelated_files():
    """The match is anchored on a database extension, not a bare ``-wal``."""
    for name in ("notes-wal", "report-shm", "daily-journal", "my-wal.txt"):
        assert backup._should_exclude(Path(name)) is False
