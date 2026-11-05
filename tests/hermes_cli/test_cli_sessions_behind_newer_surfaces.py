"""CLI session lookups must page by creating surface, not by a fixed local window.

Gateway rows for Desktop, TUI and the classic CLI are stored as ``source='local'``
with the real surface in ``created_source``. ``hermes -c``, in-chat ``/sessions``
and ``/resume`` read those rows through ``SurfaceLabelReader``. A fixed over-fetch
of every local row, filtered afterwards, dropped the CLI session once enough newer
Desktop/TUI rows filled the window.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from hermes_cli.gateway_session_store import SurfaceLabelReader
from hermes_state import SessionDB


def _populate(db: SessionDB, rows: list[tuple[str, str, str, float]]) -> None:
    for sid, source, _created, _started in rows:
        db.create_session(sid, source=source)
    with db._lock:
        db._conn.executemany(
            "UPDATE sessions SET created_source=?, started_at=?, message_count=1 WHERE id=?",
            [(created, started, sid) for sid, _source, created, started in rows],
        )
        db._conn.commit()


def _newer_locals(count: int, *, start: float, prefix: str) -> list[tuple[str, str, str, float]]:
    """``count`` gateway rows newer than a CLI session, alternating Desktop and TUI."""
    return [
        (f"{prefix}-{index:03d}", "local", ("desktop", "tui")[index % 2], start + index)
        for index in range(count)
    ]


def _resolve_cli(monkeypatch, home) -> str | None:
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.main._resolve_workspace_key", lambda: None)
    from hermes_cli.main import _resolve_last_session

    return _resolve_last_session("cli")


def _recent_cli_ids(db: SessionDB) -> tuple[list[str], list[str]]:
    """Ids from the shared listing helper and from ``/sessions`` / ``/resume`` (limit 10)."""
    from hermes_cli.cli_session_mixin import CLISessionMixin
    from hermes_cli.session_listing import query_session_listing
    from hermes_state_sessions import INTERNAL_LISTING_SOURCES

    reader = SurfaceLabelReader(db)
    listed = query_session_listing(
        reader, source="cli", current_session_id="other-session", include_all_sources=False,
        include_unnamed=True, limit=10, exclude_sources=list(INTERNAL_LISTING_SOURCES),
    )
    via_command = CLISessionMixin._list_recent_sessions(
        SimpleNamespace(_session_db=reader, session_id="other-session"), limit=10,
    )
    return [row["id"] for row in listed], [row["id"] for row in via_command]


def _ids(reader, method: str, **kwargs) -> list[str]:
    return [row["id"] for row in getattr(reader, method)(**kwargs)]


def test_continue_and_sessions_find_cli_session_behind_newer_desktop_rows(tmp_path, monkeypatch):
    """Bare ``hermes -c``, ``/sessions`` and ``/resume`` reach the CLI session behind 200 newer rows."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _populate(db, [("cli-session", "local", "cli", 1_000_000), *_newer_locals(200, start=2_000_000, prefix="d")])
    assert _recent_cli_ids(db) == (["cli-session"], ["cli-session"])
    db.close()

    assert _resolve_cli(monkeypatch, tmp_path) == "cli-session"


def test_surface_pages_are_exact_across_newer_local_rows(tmp_path):
    """LIMIT/OFFSET walk matching rows only.

    A non-local ``source='cli'`` row matches whatever its ``created_source``. ``unknown``,
    Desktop, TUI and an unrelated platform do not.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    _populate(db, [
        ("telegram-new", "telegram", "telegram", 2_100_000),
        ("unknown-new", "local", "unknown", 2_000_000),
        *_newer_locals(60, start=1_900_000, prefix="top"),
        ("legacy-cli", "cli", "desktop", 1_800_000),
        *_newer_locals(10, start=1_700_000, prefix="gap-a"),
        ("cli-a", "local", "cli", 1_600_000),
        *_newer_locals(10, start=1_500_000, prefix="gap-b"),
        ("cli-b", "local", "cli", 1_400_000),
        *_newer_locals(10, start=1_300_000, prefix="gap-c"),
        ("cli-c", "local", "cli", 1_200_000),
    ])
    reader = SurfaceLabelReader(db)
    for method in ("search_sessions", "list_sessions_rich"):
        assert _ids(reader, method, source="cli", limit=10) == ["legacy-cli", "cli-a", "cli-b", "cli-c"]
        assert _ids(reader, method, source="cli", limit=1, offset=1) == ["cli-a"]
        assert _ids(reader, method, source="cli", limit=2, offset=2) == ["cli-b", "cli-c"]
    # ``hermes -c`` asks for the CLI family (cli + oneshot), still paged in SQL.
    assert _ids(reader, "search_sessions", source=["cli", "oneshot"], limit=1) == ["legacy-cli"]
    db.close()


def test_startup_adds_the_local_created_source_index_to_an_older_store(tmp_path):
    """The partial index the surface filter seeks is created when a store that predates it
    reopens. Query plans are left to the planner; pages are pinned above."""
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db._conn.execute("DROP INDEX idx_sessions_local_created_started")
    db._conn.commit()
    db.close()

    db = SessionDB(db_path=path)
    sql = db._conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND name='idx_sessions_local_created_started'"
    ).fetchone()
    db.close()
    assert sql is not None and "WHERE source = 'local'" in sql[0]


def _pre_column_readonly(tmp_path) -> SessionDB:
    """A store last written before ``created_source``, reopened read-only (which never migrates)."""
    path = tmp_path / "state.db"
    writer = SessionDB(db_path=path)
    writer.create_session("legacy-cli", source="cli")
    with writer._lock:
        writer._conn.execute("UPDATE sessions SET message_count=1 WHERE id='legacy-cli'")
        writer._conn.execute("DROP INDEX IF EXISTS idx_sessions_local_created_started")
        writer._conn.execute("ALTER TABLE sessions DROP COLUMN created_source")
        writer._conn.commit()
    writer.close()
    return SessionDB(db_path=path, read_only=True)


def _columns(db: SessionDB) -> set[str]:
    return {row[1] for row in db._conn.execute("PRAGMA table_info(sessions)")}


def test_readonly_pre_column_store_finds_legacy_cli_session(tmp_path, monkeypatch):
    """A missing ``created_source`` column must not read as "no previous session", and the
    lookups must not migrate the file."""
    db = _pre_column_readonly(tmp_path)
    try:
        assert _recent_cli_ids(db) == (["legacy-cli"], ["legacy-cli"])
    finally:
        db.close()
    assert _resolve_cli(monkeypatch, tmp_path) == "legacy-cli"
    ro = SessionDB(db_path=tmp_path / "state.db", read_only=True)
    try:
        assert "created_source" not in _columns(ro)
    finally:
        ro.close()


@pytest.mark.parametrize(("error", "failures", "first_call_raises"), [
    ("disk I/O error", 1, False),         # transient: replayed like every other read
    ("disk I/O error", 99, True),         # persistent: propagates, not remembered
    ("database is locked", 1, True),      # a lock is not a missing column
])
def test_schema_probe_error_is_never_cached_as_absent(tmp_path, error, failures, first_call_raises):
    db = SessionDB(db_path=tmp_path / "state.db")
    _populate(db, [("legacy-cli", "cli", "cli", 1_000_000), ("local-cli", "local", "cli", 4_000_000_000)])
    left = {"n": failures}
    real_ctx = db._read_ctx

    @contextmanager
    def read_ctx():
        with real_ctx() as conn:
            real_execute = conn.execute

            def execute(sql, *args, **kwargs):
                if "table_info" in sql and left["n"] > 0:
                    left["n"] -= 1
                    raise sqlite3.OperationalError(error)
                return real_execute(sql, *args, **kwargs)

            conn.execute = execute
            try:
                yield conn
            finally:
                conn.execute = real_execute

    db._read_ctx = read_ctx
    reader = SurfaceLabelReader(db)
    if first_call_raises:
        with pytest.raises(sqlite3.OperationalError, match=error):
            reader.search_sessions(source=["cli", "oneshot"], limit=1)
        left["n"] = 0
    assert _ids(reader, "search_sessions", source=["cli", "oneshot"], limit=1) == ["local-cli"]
    assert left["n"] == 0  # the probe went through the retrying reader
    db.close()


def test_absent_schema_probe_sees_a_column_added_later(tmp_path):
    """In-chat ``/sessions`` keeps one reader; after another connection adds ``created_source``
    and a newer local CLI row, that same reader returns it."""
    db = _pre_column_readonly(tmp_path)
    reader = SurfaceLabelReader(db)
    try:
        assert _ids(reader, "search_sessions", source="cli", limit=5) == ["legacy-cli"]
        other = sqlite3.connect(tmp_path / "state.db")
        try:
            other.execute("ALTER TABLE sessions ADD COLUMN created_source TEXT")
            other.execute(
                "INSERT INTO sessions (id, source, created_source, started_at) VALUES (?, ?, ?, ?)",
                ("local-new", "local", "cli", 4_000_000_000),
            )
            other.commit()
        finally:
            other.close()
        assert _ids(reader, "search_sessions", source="cli", limit=5)[0] == "local-new"
        assert "local-new" in _ids(reader, "list_sessions_rich", source="cli", limit=5)
    finally:
        db.close()
