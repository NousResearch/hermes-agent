"""Tests for recovery-tooling gaps: issue #80205 (range-query budget can
omit a recoverable tail row) and the lost_and_found last-resort lane for
sources whose table schemas are unreadable.

The corrupted fixtures here are REAL physical SQLite page damage (flipped
b-tree/schema header bytes), not mocked cursor exceptions.
"""

from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_cli import session_recovery
from hermes_cli.session_lost_and_found import (
    classify_lost_and_found_row,
    map_lost_and_found_rows,
    rebuild_fts_indexes,
    stub_missing_parent_sessions,
)
from hermes_cli.session_recovery import (
    SessionRecoverySafetyError,
    SessionRecoverySourceError,
    _probe_populated_edge,
    recover_session_database,
)

from tests.hermes_cli.test_session_recovery import (
    _btree_leaf_pages,
    _make_page_spanning_source,
)


from hermes_cli.session_lost_and_found import find_sqlite3_cli

# .recover needs a sqlite3 shell built with sqlite_dbpage — PATH presence
# alone is not enough (Ubuntu CI ships a build without it).
HAVE_SQLITE3_CLI = find_sqlite3_cli() is not None


# ── physical corruption helpers ─────────────────────────────────────────────


def _page_size(data: bytes) -> int:
    size = int.from_bytes(data[16:18], "big")
    return 65_536 if size == 1 else size


def _leaf_cell_count(path: Path, page_number: int) -> int:
    data = path.read_bytes()
    page_size = _page_size(data)
    header = (page_number - 1) * page_size + (100 if page_number == 1 else 0)
    assert data[header] in {0x0A, 0x0D}
    return int.from_bytes(data[header + 3 : header + 5], "big")


def _corrupt_leaf(path: Path, page_number: int) -> None:
    data = bytearray(path.read_bytes())
    page_size = _page_size(bytes(data))
    header = (page_number - 1) * page_size + (100 if page_number == 1 else 0)
    assert data[header] in {0x0A, 0x0D}
    data[header + 3 : header + 5] = b"\xff\xff"
    path.write_bytes(data)


def _corrupt_schema_page(path: Path) -> None:
    """Damage the sqlite_master b-tree so no table schema is readable.

    Page 1 holds the schema table root. An impossible cell count in its
    header makes every ``PRAGMA table_info`` / schema read raise
    'database disk image is malformed' while the file still opens and the
    data pages of every table remain physically intact.
    """
    data = bytearray(path.read_bytes())
    assert data[:16] == b"SQLite format 3\x00"
    header = 100
    assert data[header] in {0x02, 0x05, 0x0A, 0x0D}
    data[header + 3 : header + 5] = b"\xff\xff"
    path.write_bytes(data)


def _make_schema_unreadable_source(path: Path) -> dict[str, int]:
    db = SessionDB(db_path=path)
    try:
        for session_number in range(3):
            session_id = f"20260812_1353{session_number:02d}_abc{session_number:03x}"
            db.create_session(session_id, "cli", cwd=f"/tmp/laf-{session_number}")
            db.set_session_title(session_id, f"LAF {session_number}")
            for message_number in range(9):
                db.append_message(
                    session_id,
                    "user" if message_number % 2 == 0 else "assistant",
                    f"lost-and-found payload {session_number} {message_number}",
                )
    finally:
        db.close()
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("VACUUM")
    finally:
        conn.close()
    _corrupt_schema_page(path)
    return {"sessions": 3, "messages": 27}


# ── issue #80205: recoverable tail row next to a damaged rowid edge ─────────


def test_exact_lookup_recovers_tail_row_next_to_damaged_high_edge(
    tmp_path: Path,
) -> None:
    """Regression for #80205: a readable boundary row must not be omitted.

    Damaging the RIGHTMOST messages leaf makes the ordered high-edge probe
    fail, so salvage falls back to the full rowid domain. The last readable
    row (the final cell of the last healthy leaf) can only be reached through
    a singleton range once bisection narrows down — and a singleton *range*
    scan must advance past the row into the damaged sibling page to prove the
    range is exhausted, discarding the already-produced row. The fix performs
    an exact ``rowid = ?`` lookup for singleton ranges, which stops at the
    hit and recovers the row exactly as SQLite's page-level ``.recover``
    does.
    """
    source = tmp_path / "tail-damaged.db"
    output = tmp_path / "tail-recovered.db"
    message_count = 320
    messages_root, count_index_root = _make_page_spanning_source(
        source, message_count
    )

    _, leaf_pages = _btree_leaf_pages(source, messages_root)
    assert len(leaf_pages) >= 3
    rightmost_leaf = leaf_pages[-1]
    lost_rows = _leaf_cell_count(source, rightmost_leaf)
    assert 0 < lost_rows < message_count
    boundary_rowid = message_count - lost_rows
    _corrupt_leaf(source, rightmost_leaf)
    if count_index_root is not None:
        _, index_leaves = _btree_leaf_pages(source, count_index_root)
        _corrupt_leaf(source, index_leaves[-1])

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        chunk_size=8,
        allow_partial=True,
    )

    copied = report["copy"]["messages"]
    bounds = copied["rowid_bounds"]
    # Premise check: the high edge probe really failed and fell back.
    assert any("high rowid" in error for error in bounds["errors"]), bounds
    assert "high" in bounds["fallback_edges"]

    conn = sqlite3.connect(str(output))
    try:
        recovered_ids = {
            int(row[0]) for row in conn.execute("SELECT id FROM messages")
        }
    finally:
        conn.close()

    assert 1 in recovered_ids
    # The headline regression: the last readable row before the damage.
    assert boundary_rowid in recovered_ids, (
        f"boundary row {boundary_rowid} was omitted; max recovered "
        f"{max(recovered_ids)}; exact_lookup_recovered="
        f"{copied.get('exact_lookup_recovered')}"
    )
    assert copied["exact_lookup_recovered"] >= 1
    assert recovered_ids == set(range(1, boundary_rowid + 1))
    assert report["verification"]["integrity_check"] == ["ok"]
    assert report["verified"] is True


def test_probe_populated_edge_caps_synthetic_domain(tmp_path: Path) -> None:
    """The gallop converges on a finite bound in O(log) probes when the
    region beyond the data is cleanly seekable."""
    db_path = tmp_path / "clean.db"
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v TEXT)")
        conn.executemany(
            "INSERT INTO t (id, v) VALUES (?, ?)",
            [(i, f"value {i}") for i in range(1, 101)],
        )
        probe = _probe_populated_edge(conn, "t", edge="high", anchor=1)
        assert probe["capped"] is True
        assert probe["bound"] >= 100
        assert probe["bound"] < 10_000
        assert probe["probes"] <= 64

        probe_low = _probe_populated_edge(conn, "t", edge="low", anchor=100)
        assert probe_low["capped"] is True
        assert probe_low["bound"] <= 1
    finally:
        conn.close()


# ── lost_and_found lane: unreadable table schemas ───────────────────────────


def test_unreadable_schema_without_cli_names_the_sqlite3_requirement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a sqlite3 CLI the refusal must say exactly what to install."""
    source = tmp_path / "schemaless.db"
    output = tmp_path / "schemaless-recovered.db"
    _make_schema_unreadable_source(source)

    import hermes_cli.session_lost_and_found as laf

    monkeypatch.setattr(laf, "find_sqlite3_cli", lambda: None)
    with pytest.raises(SessionRecoverySourceError) as excinfo:
        recover_session_database(
            source,
            output,
            work_dir=tmp_path,
            allow_partial=True,
        )
    message = str(excinfo.value)
    assert "sessions" in message and "messages" in message
    assert "sqlite3" in message
    assert ".recover" in message
    assert not output.exists()


@pytest.mark.skipif(
    not HAVE_SQLITE3_CLI,
    reason="sqlite3 CLI not on PATH; .recover is a shell-only feature",
)
def test_lost_and_found_lane_recovers_schema_unreadable_source(
    tmp_path: Path,
) -> None:
    """The last-resort lane must salvage rows SQL-level recovery cannot."""
    source = tmp_path / "schemaless.db"
    output = tmp_path / "schemaless-recovered.db"
    expected = _make_schema_unreadable_source(source)

    # Premise: the schema really is unreadable at the SQL level.
    probe = sqlite3.connect(str(source))
    try:
        with pytest.raises(sqlite3.DatabaseError):
            probe.execute("SELECT COUNT(*) FROM messages").fetchone()
    finally:
        probe.close()

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        allow_partial=True,
    )

    assert report["mode"] == "lost_and_found_salvage"
    assert report["best_effort"] is True
    assert report["partial"] is True
    assert report["complete"] is False
    assert report["installed"] is False
    assert report["unreadable_schemas"] == ["sessions", "messages"]
    assert any(
        "BEST-EFFORT" in warning
        for warning in report["verification"]["warnings"]
    )

    conn = sqlite3.connect(str(output))
    try:
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        orphans = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id NOT IN "
            "(SELECT id FROM sessions)"
        ).fetchone()[0]
        fts_matches = conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
            ("payload",),
        ).fetchone()[0]
    finally:
        conn.close()

    assert session_count == expected["sessions"]
    assert message_count == expected["messages"]
    assert orphans == 0
    assert fts_matches == expected["messages"]

    # The output must open as a regular current-schema session database.
    recovered_db = SessionDB(db_path=output)
    try:
        sessions = recovered_db.list_sessions_rich(limit=10)
        assert len(sessions) == expected["sessions"]
    finally:
        recovered_db.close()


# ── mapper unit tests (no sqlite3 CLI required) ─────────────────────────────


def _make_synthetic_lost_and_found(
    path: Path,
    dest_schema_db: Path,
) -> dict[str, int]:
    """Build a .recover-shaped lost_and_found DB directly, no CLI needed."""
    schema = sqlite3.connect(str(dest_schema_db))
    try:
        sessions_columns = [
            str(row[1]) for row in schema.execute("PRAGMA table_info(sessions)")
        ]
        messages_columns = [
            str(row[1]) for row in schema.execute("PRAGMA table_info(messages)")
        ]
        usage_columns = [
            str(row[1])
            for row in schema.execute("PRAGMA table_info(session_model_usage)")
        ]
    finally:
        schema.close()
    # Width is derived from the live schema so ordinary column additions
    # don't break this test (it pinned 54, then 55, then 56 in one week).
    # The floor guards against accidentally reading an empty/old schema.
    current_width = len(sessions_columns)
    assert current_width >= 55
    assert len(usage_columns) == 18

    max_fields = current_width
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        cells = ", ".join(f"c{i}" for i in range(max_fields))
        conn.execute(
            f"CREATE TABLE lost_and_found (rootpgno INTEGER, pgno INTEGER, "
            f"nfield INTEGER, id INTEGER, {cells})"
        )

        def insert(nfield: int, rowid, values: list) -> None:
            padded = list(values) + [None] * (max_fields - len(values))
            placeholders = ", ".join("?" for _ in range(4 + max_fields))
            conn.execute(
                f"INSERT INTO lost_and_found VALUES ({placeholders})",
                [2, 5, nfield, rowid, *padded],
            )

        def session_row(session_id: str, ncols: int) -> list:
            base = {
                "id": session_id,
                "source": "telegram",
                "started_at": 1_754_000_000.0,
                "message_count": 2,
                "title": f"synthetic {session_id}",
            }
            return [base.get(column) for column in sessions_columns[:ncols]]

        # Current layout (dynamic width) and historical 52-column layout.
        insert(max_fields, 1, session_row("20260101_010101_aaa001", max_fields))
        insert(52, 2, session_row("20260202_020202_bbb002", 52))
        # 14-column legacy layout: identity + a plausible epoch timestamp.
        legacy = ["20250303_030303_ccc003", "cli", 1_741_000_000.0] + [None] * 11
        insert(14, 3, legacy)

        # messages rows: NULL first cell (rowid alias), session id second,
        # role third.
        for index, (session_id, role, content) in enumerate(
            [
                ("20260101_010101_aaa001", "user", "hello from user"),
                ("20260101_010101_aaa001", "assistant", "hello from assistant"),
                ("20261111_111111_ddd004", "user", "orphaned message payload"),
                ("20261111_111111_ddd004", "tool", "orphaned tool payload"),
            ]
        ):
            row = {
                "id": None,
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": 1_754_000_100.0 + index,
            }
            insert(
                23,
                100 + index,
                [row.get(column) for column in messages_columns[:23]],
            )

        # session_model_usage: 18 columns, orphaned session id on purpose.
        usage = {
            "session_id": "20261212_121212_eee005",
            "model": "test/model",
            "billing_provider": "",
            "billing_base_url": "",
            "billing_mode": "",
            "task": "",
            "api_call_count": 4,
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
            "estimated_cost_usd": 0.01,
            "actual_cost_usd": 0.01,
            "first_seen": 1_754_000_000.0,
            "last_seen": 1_754_000_500.0,
        }
        insert(18, 200, [usage.get(column) for column in usage_columns])

        # Junk that must NOT be classified into canonical tables.
        insert(3, 300, ["random", "noise", 42])
        insert(max_fields, 301, ["not-a-session-id", "cli"] + [None] * (max_fields - 2))
        insert(23, 302, [None, "sess-x", "not-a-role", "junk"])
    finally:
        conn.close()
    return {
        "sessions": 3,
        "messages": 4,
        "session_model_usage": 1,
        "junk": 3,
    }


def test_classify_lost_and_found_row_sentinels() -> None:
    assert (
        classify_lost_and_found_row(
            23, (None, "20260101_010101_aaa001", "user", "hi")
        )
        == "messages"
    )
    assert (
        classify_lost_and_found_row(
            55, ("20260101_010101_aaa001", "cli") + (None,) * 53
        )
        == "sessions"
    )
    assert (
        classify_lost_and_found_row(
            52, ("20260101_010101_aaa001", "discord") + (None,) * 50
        )
        == "sessions"
    )
    assert (
        classify_lost_and_found_row(
            14, ("20250101_010101_zzz999", "cli") + (None,) * 12
        )
        == "sessions"
    )
    assert (
        classify_lost_and_found_row(
            18, ("20260101_010101_aaa001", "gpt-x") + (None,) * 16
        )
        == "session_model_usage"
    )
    # Junk shapes.
    assert classify_lost_and_found_row(3, ("random", "noise", 42)) is None
    assert (
        classify_lost_and_found_row(55, ("not-a-session-id", "cli") + (None,) * 53)
        is None
    )
    assert (
        classify_lost_and_found_row(23, (None, "sess", "not-a-role", "x")) is None
    )
    assert classify_lost_and_found_row(0, ()) is None


def test_mapper_rebuilds_sessiondb_from_synthetic_lost_and_found(
    tmp_path: Path,
) -> None:
    """Binary-independent: mapper + stubbing + FTS rebuild end to end."""
    schema_ref = tmp_path / "schema-ref.db"
    SessionDB(db_path=schema_ref).close()

    lf_path = tmp_path / "lost_and_found.db"
    expected = _make_synthetic_lost_and_found(lf_path, schema_ref)

    output = tmp_path / "mapped.db"
    SessionDB(db_path=output).close()

    lf_conn = sqlite3.connect(str(lf_path), isolation_level=None)
    dest = sqlite3.connect(str(output), isolation_level=None)
    try:
        dest.execute("PRAGMA foreign_keys=OFF")
        mapping = map_lost_and_found_rows(lf_conn, dest)
        stubbing = stub_missing_parent_sessions(dest)
        fts = rebuild_fts_indexes(dest)

        assert mapping["mapped"]["sessions"] == expected["sessions"]
        assert mapping["mapped"]["messages"] == expected["messages"]
        assert (
            mapping["mapped"]["session_model_usage"]
            == expected["session_model_usage"]
        )
        assert mapping["legacy_minimal_sessions"] == 1
        assert mapping["unmapped_rows"] == expected["junk"]

        # Orphaned children got stub parents — never deleted.
        assert stubbing["sessions_stubbed"] == 2  # ddd004 + eee005
        assert stubbing["messages_retained"] == 2
        message_count = dest.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        assert message_count == expected["messages"]
        usage_count = dest.execute(
            "SELECT COUNT(*) FROM session_model_usage"
        ).fetchone()[0]
        assert usage_count == expected["session_model_usage"]

        stub_titles = [
            str(row[0])
            for row in dest.execute(
                "SELECT title FROM sessions WHERE source = 'recovered'"
            )
        ]
        assert len(stub_titles) == 2
        assert all(title.startswith("[best-effort recovered") for title in stub_titles)

        # The 52-col row landed with its real metadata preserved.
        row = dest.execute(
            "SELECT source, title FROM sessions WHERE id = ?",
            ("20260202_020202_bbb002",),
        ).fetchone()
        assert row == ("telegram", "synthetic 20260202_020202_bbb002")

        assert fts.get("messages_fts") == "rebuilt"
        fts_hits = dest.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH ?",
            ("payload OR hello",),
        ).fetchone()[0]
        assert fts_hits == expected["messages"]

        assert dest.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert dest.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        lf_conn.close()
        dest.close()

    # And the mapped output opens through the normal SessionDB path.
    db = SessionDB(db_path=output)
    try:
        assert len(db.list_sessions_rich(limit=20)) == 5
    finally:
        db.close()


# ── issue #72291: source-fingerprint error must name the parent CLI ─────────


def test_fingerprint_error_enumerates_parent_cli_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "busy.db"
    SessionDB(db_path=source).close()

    fingerprints = iter([{"main": {"size": 1, "mtime_ns": 1}},
                         {"main": {"size": 2, "mtime_ns": 2}},
                         {"main": {"size": 3, "mtime_ns": 3}}])
    monkeypatch.setattr(
        session_recovery,
        "_source_fingerprint",
        lambda _source: next(fingerprints),
    )
    with pytest.raises(SessionRecoverySafetyError) as excinfo:
        session_recovery.inspect_session_database(source, work_dir=tmp_path)
    message = str(excinfo.value)
    assert "Stop every Hermes process" in message
    # The gap from #72291: the parent CLI session itself must be enumerated.
    assert "CLI session" in message
    assert "fresh shell" in message
    assert "snapshot" in message

# ── header-damaged source: 'file is not a database' ─────────────────────────
#
# A SIGKILL mid-write can damage page 1 itself, which makes SQLite refuse the
# file outright ('file is not a database') and takes the CLI .recover path
# down with it — .recover opens the file like any other database. The data
# pages below page 1 are usually intact, so a valid 100-byte header spliced
# onto a copy of the file is enough to make page-level salvage work again
# (validated on a real 335 MB incident file that gave back 136k rows).


def _corrupt_header(path: Path, *, keep_page_size_bytes: bool) -> dict[str, int]:
    """Scramble the first 100 bytes of a closed database file.

    Deterministic damage (rotated bytes, destroyed magic string). With
    ``keep_page_size_bytes=False`` the page-size field is also replaced with
    3250 — not a power of two — mirroring the real incident file.
    """
    data = bytearray(path.read_bytes())
    original_page_size_bytes = bytes(data[16:18])
    garbage = bytearray(((byte + 91) % 256) for byte in data[:100])
    garbage[0:16] = b"\x00" * 16          # 'SQLite format 3\x00' is gone
    garbage[16:18] = b"\x0c\xb2"          # 3250: valid-looking, not a page size
    if keep_page_size_bytes:
        garbage[16:18] = original_page_size_bytes
    data[:100] = garbage
    path.write_bytes(bytes(data))
    return {"page_size_bytes": original_page_size_bytes}


def _make_header_damaged_source(
    path: Path,
    *,
    keep_page_size_bytes: bool,
) -> dict[str, int]:
    """A fully header-damaged Hermes session DB with readable data pages."""
    db = SessionDB(db_path=path)
    try:
        for session_number in range(3):
            session_id = f"20260812_1353{session_number:02d}_abc{session_number:03x}"
            db.create_session(session_id, "cli", cwd=f"/tmp/hdr-{session_number}")
            db.set_session_title(session_id, f"Header {session_number}")
            for message_number in range(9):
                db.append_message(
                    session_id,
                    "user" if message_number % 2 == 0 else "assistant",
                    f"header salvage payload {session_number} {message_number}",
                )
    finally:
        db.close()
    conn = sqlite3.connect(str(path), isolation_level=None)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("VACUUM")
    finally:
        conn.close()
    _corrupt_header(path, keep_page_size_bytes=keep_page_size_bytes)
    return {"sessions": 3, "messages": 27}


def _salvage_helper(symbol: str):
    """Import one header-salvage helper lazily so RED runs fail per-test."""
    import hermes_cli.session_lost_and_found as laf

    helper = getattr(laf, symbol, None)
    if helper is None:
        pytest.fail(f"session_lost_and_found is missing header helper {symbol!r}")
    return helper


def test_header_damage_makes_sqlite_refuse_the_file(tmp_path: Path) -> None:
    """Premise guard: the fixture must reproduce the incident symptom."""
    source = tmp_path / "clean.db"
    SessionDB(db_path=source).close()
    _corrupt_header(source, keep_page_size_bytes=False)
    conn = sqlite3.connect(str(source), isolation_level=None)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not a database"):
            conn.execute("PRAGMA journal_mode").fetchone()
    finally:
        conn.close()


def test_open_error_probe_distinguishes_refused_from_malformed(
    tmp_path: Path,
) -> None:
    """'file is not a database' must be detected; malformed files must not."""
    probe = _salvage_helper("detect_database_open_error")

    refused = tmp_path / "refused.db"
    SessionDB(db_path=refused).close()
    _corrupt_header(refused, keep_page_size_bytes=True)
    assert probe(refused) is not None

    # Malformed but openable (schema page damaged): the plain lane must run.
    malformed = tmp_path / "malformed.db"
    _make_schema_unreadable_source(malformed)
    assert probe(malformed) is None

    healthy = tmp_path / "healthy.db"
    SessionDB(db_path=healthy).close()
    assert probe(healthy) is None


def test_donor_header_is_a_valid_sqlite_header(tmp_path: Path) -> None:
    """Header candidates: correct magic, page size, and sanitized ver bytes."""
    make_header = _salvage_helper("build_sqlite_header")

    header = make_header(8192, version_bytes=(1, 1))
    assert header[:16] == b"SQLite format 3\x00"
    assert int.from_bytes(header[16:18], "big") == 8192
    assert tuple(header[18:20]) == (1, 1)
    # A spliced header must be openable as a database again.
    fresh = tmp_path / "fresh.db"
    conn = sqlite3.connect(str(fresh))
    conn.execute("CREATE TABLE seed (x)")
    conn.commit()
    conn.close()
    data = bytearray(fresh.read_bytes())
    data[:100] = header
    repaired = tmp_path / "repaired.db"
    repaired.write_bytes(bytes(data))
    conn = sqlite3.connect(str(repaired), isolation_level=None)
    try:
        assert conn.execute("PRAGMA page_size").fetchone()[0] == 8192
    finally:
        conn.close()

    # 65536 is encoded as 0x0001 per the SQLite file format.
    header_big = make_header(65536, version_bytes=(2, 2))
    assert int.from_bytes(header_big[16:18], "big") == 1
    assert tuple(header_big[18:20]) == (2, 2)


@pytest.mark.skipif(
    not HAVE_SQLITE3_CLI,
    reason="sqlite3 CLI not on PATH; .recover is a shell-only feature",
)
@pytest.mark.parametrize("keep_page_size_bytes", [True, False])
def test_header_salvage_lane_recovers_file_not_a_database_source(
    tmp_path: Path,
    keep_page_size_bytes: bool,
) -> None:
    """End to end: header-damaged source still yields a live session DB."""
    source = tmp_path / "header-damaged.db"
    output = tmp_path / "header-recovered.db"
    expected = _make_header_damaged_source(
        source, keep_page_size_bytes=keep_page_size_bytes
    )

    # Premise: the lane's usual CLI entry cannot even open the source.
    probe = sqlite3.connect(str(source), isolation_level=None)
    try:
        with pytest.raises(sqlite3.DatabaseError, match="not a database"):
            probe.execute("PRAGMA journal_mode").fetchone()
    finally:
        probe.close()

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        allow_partial=True,
    )

    assert report["mode"] == "lost_and_found_salvage"
    assert report["best_effort"] is True
    salvage = report["sqlite3_cli"].get("header_salvage")
    assert salvage, (
        "the lost_and_found lane must record header salvage when the source "
        f"cannot be opened; report={report['sqlite3_cli']}"
    )
    assert salvage["triggered"] is True
    assert "not a database" in salvage["open_error"]
    assert any(
        "header" in warning.lower()
        for warning in report["verification"]["warnings"]
    )

    conn = sqlite3.connect(str(output))
    try:
        assert conn.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        payloads = conn.execute(
            "SELECT COUNT(*) FROM messages_fts WHERE messages_fts MATCH 'salvage'"
        ).fetchone()[0]
    finally:
        conn.close()

    assert session_count == expected["sessions"]
    assert message_count == expected["messages"]
    assert payloads == expected["messages"]

    recovered_db = SessionDB(db_path=output)
    try:
        assert len(recovered_db.list_sessions_rich(limit=10)) == expected["sessions"]
    finally:
        recovered_db.close()


@pytest.mark.skipif(
    not HAVE_SQLITE3_CLI,
    reason="sqlite3 CLI not on PATH; .recover is a shell-only feature",
)
def test_header_salvage_is_skipped_for_openable_malformed_source(
    tmp_path: Path,
) -> None:
    """Malformed-but-openable sources keep using the untouched plain lane."""
    source = tmp_path / "schemaless.db"
    output = tmp_path / "schemaless-recovered.db"
    _make_schema_unreadable_source(source)

    report = recover_session_database(
        source,
        output,
        work_dir=tmp_path,
        allow_partial=True,
    )
    assert report["mode"] == "lost_and_found_salvage"
    assert report["sqlite3_cli"].get("header_salvage") is None


@pytest.mark.skipif(
    not HAVE_SQLITE3_CLI,
    reason="sqlite3 CLI not on PATH; .recover is a shell-only feature",
)
def test_run_cli_recover_reports_header_salvage_result(tmp_path: Path) -> None:
    """Unit level: run_cli_lost_and_found_recover repairs and labels itself."""
    from hermes_cli.session_lost_and_found import (
        LostAndFoundError,
        run_cli_lost_and_found_recover,
    )

    source = tmp_path / "damaged.db"
    _make_header_damaged_source(source, keep_page_size_bytes=False)
    lf_path = tmp_path / "lost_and_found.db"

    cli_report = run_cli_lost_and_found_recover(
        source, lf_path, find_sqlite3_cli()
    )
    assert cli_report["header_salvage"]["triggered"] is True
    salvage_source = Path(cli_report["header_salvage"]["repaired_source"])
    assert salvage_source != source
    # The repaired copy differs from the source only in the first 100 bytes.
    original = source.read_bytes()
    repaired = salvage_source.read_bytes()
    assert len(repaired) == len(original)
    assert repaired[100:] == original[100:]

    conn = sqlite3.connect(str(lf_path))
    try:
        tables = {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master")
        }
    finally:
        conn.close()
    assert "lost_and_found" in tables or {"sessions", "messages"} <= tables

    # A source damaged beyond any header must still raise, not hang or lie.
    hopeless = tmp_path / "hopeless.db"
    hopeless.write_bytes(b"not a database at all, not even close" * 4)
    with pytest.raises(LostAndFoundError):
        run_cli_lost_and_found_recover(
            hopeless, tmp_path / "hopeless_lf.db", find_sqlite3_cli()
        )
