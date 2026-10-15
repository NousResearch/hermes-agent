"""CLI session lookups must page by creating surface, not by a fixed local window.

Gateway rows for Desktop, TUI and the classic CLI are stored as ``source='local'``
with the real surface in ``created_source``. ``hermes -c``, in-chat ``/sessions``
and ``/resume`` read those rows through ``SurfaceLabelReader``. A fixed over-fetch
of every local row, filtered afterwards, drops the CLI session once enough newer
Desktop/TUI rows fill the window. ``hermes sessions list`` does not use that reader.

``created_source='unknown'`` stays unmatched. #138838 and #139058 repair that
placeholder on upsert; this reader does not admit it.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from hermes_state import SessionDB


def _populate(db: SessionDB, rows: list[tuple[str, str, str, float]]) -> None:
    for sid, source, _created, _started in rows:
        db.create_session(sid, source=source)
    with db._lock:
        db._conn.executemany(
            "UPDATE sessions SET created_source=?, started_at=? WHERE id=?",
            [(created, started, sid) for sid, _source, created, started in rows],
        )
        db._conn.commit()


def _newer_locals(count: int, *, start: float, prefix: str) -> list[tuple[str, str, str, float]]:
    """``count`` gateway rows newer than a CLI session, alternating Desktop and TUI."""
    surfaces = ("desktop", "tui")
    return [
        (f"{prefix}-{index:03d}", "local", surfaces[index % 2], start + index)
        for index in range(count)
    ]


def _cli_row(sid: str, started: float, *, source: str = "local", created: str = "cli"):
    return (sid, source, created, started)


def _resolve_cli(monkeypatch, home) -> str | None:
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("hermes_cli.main._resolve_workspace_key", lambda: None)
    from hermes_cli.main import _resolve_last_session

    return _resolve_last_session("cli")


def _recent_cli_ids(db: SessionDB) -> tuple[list[str], list[str]]:
    """Ids from the shared listing helper and from ``/sessions`` / ``/resume`` (limit 10)."""
    from hermes_cli.cli_session_mixin import CLISessionMixin
    from hermes_cli.gateway_session_store import SurfaceLabelReader
    from hermes_cli.session_listing import query_session_listing
    from hermes_state_sessions import INTERNAL_LISTING_SOURCES

    reader = SurfaceLabelReader(db)
    listed = query_session_listing(
        reader,
        source="cli",
        current_session_id="other-session",
        include_all_sources=False,
        include_unnamed=True,
        limit=10,
        exclude_sources=list(INTERNAL_LISTING_SOURCES),
    )
    via_command = CLISessionMixin._list_recent_sessions(
        SimpleNamespace(_session_db=reader, session_id="other-session"),
        limit=10,
    )
    return [row["id"] for row in listed], [row["id"] for row in via_command]


def _reader_ids(db: SessionDB, method: str, **kwargs) -> list[str]:
    from hermes_cli.gateway_session_store import SurfaceLabelReader

    rows = getattr(SurfaceLabelReader(db), method)(**kwargs)
    return [row["id"] for row in rows]


def test_continue_finds_cli_session_behind_sixty_newer_desktop_sessions(tmp_path, monkeypatch):
    """Bare ``hermes -c`` resolves the CLI session even when 60 newer Desktop/TUI rows exist."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _populate(db, [
        _cli_row("cli-session", 1_000_000),
        *_newer_locals(60, start=2_000_000, prefix="desk"),
    ])
    db.close()

    assert _resolve_cli(monkeypatch, tmp_path) == "cli-session"


def test_sessions_and_resume_list_cli_session_behind_two_hundred_newer_rows(tmp_path):
    """In-chat ``/sessions`` and ``/resume`` still list the CLI session behind 200 newer rows."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _populate(db, [
        _cli_row("cli-session", 1_000_000),
        *_newer_locals(200, start=3_000_000, prefix="newer"),
    ])

    listed, via_command = _recent_cli_ids(db)
    assert listed == ["cli-session"]
    assert via_command == ["cli-session"]
    db.close()


def test_surface_pages_are_exact_across_newer_local_rows(tmp_path):
    """LIMIT/OFFSET walk matching rows, including across Desktop/TUI rows that sit between them.

    A non-local ``source='cli'`` row still matches when ``created_source`` names another
    surface. ``unknown``, Desktop, TUI and an unrelated platform do not.
    """
    rows = [
        ("telegram-new", "telegram", "telegram", 2_100_000),
        ("unknown-new", "local", "unknown", 2_000_000),
        *_newer_locals(60, start=1_900_000, prefix="top"),
        _cli_row("legacy-cli", 1_800_000, source="cli", created="desktop"),
        *_newer_locals(10, start=1_700_000, prefix="gap-a"),
        _cli_row("cli-a", 1_600_000),
        *_newer_locals(10, start=1_500_000, prefix="gap-b"),
        _cli_row("cli-b", 1_400_000),
        *_newer_locals(10, start=1_300_000, prefix="gap-c"),
        _cli_row("cli-c", 1_200_000),
    ]
    db = SessionDB(db_path=tmp_path / "state.db")
    _populate(db, rows)
    expected = ["legacy-cli", "cli-a", "cli-b", "cli-c"]

    for method in ("search_sessions", "list_sessions_rich"):
        kwargs = {"source": "cli"}
        assert _reader_ids(db, method, limit=10, **kwargs) == expected
        assert _reader_ids(db, method, limit=2, offset=0, **kwargs) == ["legacy-cli", "cli-a"]
        assert _reader_ids(db, method, limit=1, offset=1, **kwargs) == ["cli-a"]
        assert _reader_ids(db, method, limit=2, offset=2, **kwargs) == ["cli-b", "cli-c"]
    # ``hermes -c`` asks for the CLI family (cli + oneshot), still paged in SQL.
    assert _reader_ids(db, "search_sessions", source=["cli", "oneshot"], limit=1) == ["legacy-cli"]
    db.close()


def test_continue_and_sessions_find_a_lone_cli_session(tmp_path, monkeypatch):
    """Control: with no newer Desktop/TUI rows, both surfaces already find the CLI session."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _populate(db, [_cli_row("cli-session", 1_000_000)])
    listed, via_command = _recent_cli_ids(db)
    db.close()

    assert listed == ["cli-session"]
    assert via_command == ["cli-session"]
    assert _resolve_cli(monkeypatch, tmp_path) == "cli-session"


def test_unknown_placeholder_does_not_become_the_cli_mru(tmp_path, monkeypatch):
    """Control: a newer ``created_source='unknown'`` local row is not CLI history.

    #138838 and #139058 repair that placeholder on the creator upsert. Widening this
    reader to admit it would duplicate those fixes and hide the real CLI session.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    _populate(db, [
        _cli_row("cli-session", 1_000_000),
        ("unknown-new", "local", "unknown", 2_000_000),
    ])
    db.close()

    assert _resolve_cli(monkeypatch, tmp_path) == "cli-session"


def test_oneshot_remains_cli_family_mru(tmp_path, monkeypatch):
    """Control: a finite CLI run (``source='oneshot'``) is still the ``-c`` MRU when it is newest."""
    db = SessionDB(db_path=tmp_path / "state.db")
    _populate(db, [
        _cli_row("cli-session", 1_000_000),
        ("shot", "oneshot", "oneshot", 2_000_000),
    ])
    db.close()

    assert _resolve_cli(monkeypatch, tmp_path) == "shot"


def _plans_for_surface_reads(db: SessionDB) -> list[str]:
    """EXPLAIN QUERY PLAN text for the surface-filtered SELECTs the reader actually runs."""
    from hermes_cli.gateway_session_store import SurfaceLabelReader

    captured: list[tuple[str, tuple]] = []
    original = db._read_all

    def _capture(sql, params=()):
        captured.append((sql, tuple(params)))
        return original(sql, params)

    db._read_all = _capture
    reader = SurfaceLabelReader(db)
    reader.search_sessions(source=["cli", "oneshot"], limit=1)
    reader.list_sessions_rich(source="cli", limit=10, offset=2)
    plans = []
    for sql, params in captured:
        if "created_source" not in sql:
            continue
        rows = db._conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
        plans.append("\n".join(row["detail"] if hasattr(row, "keys") else row[3] for row in rows))
    return plans


def test_surface_filter_seeks_the_local_created_source_index(tmp_path):
    """The local branch seeks ``created_source`` instead of scanning every local row.

    Startup recreates the partial index on a store that predates it (the index is
    declared with the other post-reconcile indexes).
    """
    path = tmp_path / "state.db"
    db = SessionDB(db_path=path)
    db._conn.execute("DROP INDEX IF EXISTS idx_sessions_local_created_started")
    db._conn.commit()
    db.close()

    db = SessionDB(db_path=path)
    present = db._conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?",
        ("idx_sessions_local_created_started",),
    ).fetchone()
    assert present is not None
    _populate(db, [
        _cli_row("cli-session", 1_000_000),
        *_newer_locals(30, start=2_000_000, prefix="desk"),
    ])
    plans = _plans_for_surface_reads(db)
    assert plans, "surface-filtered reads did not reach SQL"
    for plan in plans:
        assert "idx_sessions_local_created_started" in plan
        assert "SCAN s" not in plan
        assert "SCAN sessions" not in plan
    db.close()


def test_sessions_list_stays_a_recency_page_across_surfaces(tmp_path, capsys):
    """Control: ``hermes sessions list`` does not apply the CLI surface window.

    The default page is the newest rows of every source. A limit that reaches the older
    CLI row includes it together with the Desktop rows, the same as an unfiltered read.
    """
    from hermes_cli.sessions_cmd import _cmd_list

    db = SessionDB(db_path=tmp_path / "state.db")
    _populate(db, [
        _cli_row("cli-session", 1_000_000),
        *_newer_locals(60, start=2_000_000, prefix="desk"),
    ])

    _cmd_list(db, SimpleNamespace(limit=20, source=None, workspace=None))
    short = capsys.readouterr().out
    assert "cli-session" not in short
    assert "desk-059" in short

    _cmd_list(db, SimpleNamespace(limit=80, source=None, workspace=None))
    full = capsys.readouterr().out
    assert "cli-session" in full
    assert "desk-059" in full
    db.close()


def _column_names(db: SessionDB) -> set[str]:
    return {row[1] for row in db._conn.execute("PRAGMA table_info(sessions)")}


def _pre_column_readonly(tmp_path) -> SessionDB:
    """A store last written before ``created_source``, reopened read-only.

    Writable open adds the column. Bare ``-c``, ``--resume latest``, ``/sessions``
    and ``hermes sessions list`` never do that.
    """
    path = tmp_path / "state.db"
    writer = SessionDB(db_path=path)
    writer.create_session("legacy-cli", source="cli")
    writer.append_message("legacy-cli", role="user", content="hello from cli")
    with writer._lock:
        writer._conn.execute("DROP INDEX IF EXISTS idx_sessions_local_created_started")
        writer._conn.execute("ALTER TABLE sessions DROP COLUMN created_source")
        writer._conn.commit()
    writer.close()
    reader = SessionDB(db_path=path, read_only=True)
    assert "created_source" not in _column_names(reader)
    return reader


def test_readonly_pre_column_continue_finds_legacy_cli_session(tmp_path, monkeypatch):
    """Bare ``-c`` and ``--resume latest`` share ``_resolve_last_session`` and open read-only.

    A missing ``created_source`` column must not become "no previous session".
    The lookup must not migrate the file.
    """
    _pre_column_readonly(tmp_path).close()

    assert _resolve_cli(monkeypatch, tmp_path) == "legacy-cli"
    healed = SessionDB(db_path=tmp_path / "state.db", read_only=True)
    try:
        assert "created_source" not in _column_names(healed)
    finally:
        healed.close()


def test_readonly_pre_column_session_listing_finds_legacy_cli_session(tmp_path):
    """In-chat ``/sessions`` and ``/resume`` list the legacy CLI row on a pre-column store."""
    db = _pre_column_readonly(tmp_path)
    try:
        listed, via_command = _recent_cli_ids(db)
        assert listed == ["legacy-cli"]
        assert via_command == ["legacy-cli"]
        assert "created_source" not in _column_names(db)
    finally:
        db.close()


def test_readonly_sessions_list_shows_pre_column_cli_row_without_migrating(tmp_path, capsys):
    """Control: ``hermes sessions list`` is read-only and does not use the surface predicate.

    It still prints the legacy row, and it does not add ``created_source``.
    """
    from hermes_cli.sessions_cmd import _cmd_list

    db = _pre_column_readonly(tmp_path)
    try:
        _cmd_list(db, SimpleNamespace(limit=20, source=None, workspace=None))
        assert "legacy-cli" in capsys.readouterr().out
        assert "created_source" not in _column_names(db)
    finally:
        db.close()


def test_schema_probe_retries_one_disk_io_error(tmp_path, monkeypatch):
    """One ``disk I/O error`` on ``PRAGMA table_info`` still resolves the CLI session.

    The session SELECT already replays that blip (``_read_retrying_ioerr``). The
    schema probe is the same read. An unretried error is swallowed by bare ``-c``
    and in-chat ``/sessions``, which then look empty.
    """
    path = tmp_path / "state.db"
    writer = SessionDB(db_path=path)
    _populate(writer, [_cli_row("local-cli", 4_000_000_000)])
    writer.close()

    probes: list[dict[str, int]] = []

    def fail_first_table_info(conn):
        real = conn.execute
        seen = {"n": 0}

        def execute(sql, *args, **kwargs):
            if isinstance(sql, str) and "table_info(sessions)" in sql:
                seen["n"] += 1
                if seen["n"] == 1:
                    raise sqlite3.OperationalError("disk I/O error")
            return real(sql, *args, **kwargs)

        conn.execute = execute
        probes.append(seen)

    import hermes_state

    real_open = hermes_state.SessionDB._open_read_only

    def open_read_only(self):
        real_open(self)
        fail_first_table_info(self._conn)

    monkeypatch.setattr(hermes_state.SessionDB, "_open_read_only", open_read_only)
    assert _resolve_cli(monkeypatch, tmp_path) == "local-cli"
    assert probes[0]["n"] >= 2

    ro = SessionDB(db_path=path, read_only=True)
    try:
        listed, via_command = _recent_cli_ids(ro)
    finally:
        ro.close()
    assert listed == ["local-cli"]
    assert via_command == ["local-cli"]
    assert probes[-1]["n"] >= 2


def test_persistent_disk_io_on_schema_probe_is_not_cached(tmp_path):
    """A disk I/O error that survives the retry budget still propagates and is not stored."""
    path = tmp_path / "state.db"
    writer = SessionDB(db_path=path)
    _populate(writer, [_cli_row("local-cli", 4_000_000_000)])
    writer.close()
    ro = SessionDB(db_path=path, read_only=True)
    real = ro._conn.execute

    def execute(sql, *args, **kwargs):
        if isinstance(sql, str) and "table_info(sessions)" in sql:
            raise sqlite3.OperationalError("disk I/O error")
        return real(sql, *args, **kwargs)

    ro._conn.execute = execute
    from hermes_cli.gateway_session_store import SurfaceLabelReader

    reader = SurfaceLabelReader(ro)
    try:
        caught = None
        try:
            reader.search_sessions(source=["cli", "oneshot"], limit=1)
        except sqlite3.OperationalError as exc:
            caught = str(exc)
        assert caught and "disk I/O error" in caught
        assert reader._created_source_column is None
        ro._conn.execute = real
        found = [row["id"] for row in reader.search_sessions(source=["cli", "oneshot"], limit=1)]
        assert found == ["local-cli"]
    finally:
        ro.close()


def test_locked_schema_probe_is_not_remembered_as_a_missing_column(tmp_path):
    """A ``PRAGMA table_info`` failure is not proof the column is missing.

    Caching that failure makes the next lookup, after the lock is gone, match
    ``source`` only and drop the newer ``source='local'`` / ``created_source='cli'``
    row. The error has to surface, and the following call has to see the column.
    """
    db = SessionDB(db_path=tmp_path / "state.db")
    _populate(db, [
        _cli_row("legacy-cli", 1_000_000, source="cli", created="cli"),
        _cli_row("local-cli", 4_000_000_000),
    ])
    from hermes_cli.gateway_session_store import SurfaceLabelReader

    reader = SurfaceLabelReader(db)
    raised = {"done": False}
    real_ctx = db._read_ctx

    from contextlib import contextmanager

    @contextmanager
    def read_ctx():
        with real_ctx() as conn:
            real_execute = conn.execute

            def execute(sql, *args, **kwargs):
                if isinstance(sql, str) and "table_info" in sql and not raised["done"]:
                    raised["done"] = True
                    raise sqlite3.OperationalError("database is locked")
                return real_execute(sql, *args, **kwargs)

            conn.execute = execute
            try:
                yield conn
            finally:
                conn.execute = real_execute

    db._read_ctx = read_ctx
    caught = None
    try:
        first = [row["id"] for row in reader.search_sessions(source=["cli", "oneshot"], limit=1)]
    except sqlite3.OperationalError as exc:
        caught = str(exc)
        first = None
    assert caught and "database is locked" in caught, first
    assert reader._created_source_column is None
    assert [row["id"] for row in reader.search_sessions(source=["cli", "oneshot"], limit=1)] == ["local-cli"]
    db.close()


def test_absent_schema_probe_sees_a_column_added_later(tmp_path):
    """A successful "column absent" answer is not permanent.

    In-chat ``/sessions`` keeps one reader. After another connection adds
    ``created_source`` and a newer local CLI row, that same reader must return it.
    """
    db = _pre_column_readonly(tmp_path)
    from hermes_cli.gateway_session_store import SurfaceLabelReader

    reader = SurfaceLabelReader(db)
    try:
        assert [row["id"] for row in reader.search_sessions(source="cli", limit=5)] == ["legacy-cli"]
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
        found = [row["id"] for row in reader.search_sessions(source="cli", limit=5)]
        assert found[0] == "local-new"
        listed = [row["id"] for row in reader.list_sessions_rich(source="cli", limit=5)]
        assert "local-new" in listed
    finally:
        db.close()
