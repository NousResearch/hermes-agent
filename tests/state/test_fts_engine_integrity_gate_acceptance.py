"""Acceptance tests: engine-version-gated FTS5 integrity self-repair (#86027).

Red-team acceptance suite, written ONLY from the behavior spec (design doc),
never from the implementation.

Run (from the repo root)::

    PYTHONPATH=. python -m pytest tests/state/test_fts_engine_integrity_gate_acceptance.py -q

Spec under test (summary):
  1. state_meta ``fts_integrity_engine`` == sqlite3.sqlite_version → zero
     'integrity-check' executions on open; marker missing or different →
     exactly one 'integrity-check' per EXISTING fts table
     (messages_fts / messages_fts_trigram / messages_fts_cjk).
  2. A table whose 'integrity-check' raises sqlite3.DatabaseError gets
     exactly one 'rebuild' plus exactly one re-check; on success the marker
     is stamped with the current engine and MATCH hit counts are identical
     to pre-corruption.
  3. If the re-check still fails: logger.error containing the guidance
     string ``hermes sessions repair``, marker NOT stamped, open survives.
  4. read_only=True → no integrity-check, no marker write, open succeeds.
  5. Legacy inline DB (malformed index + one missing fts trigger) → open
     succeeds, indexes backfilled from ``messages``, all 6 legacy fts
     triggers present.
  6. The v23 external-content layout (ordinary SessionDB-created DB) is
     covered by the same gate.
  7. No fts tables → marker still written.
  8. No sqlite error raised inside the gate may fail the open.

Fabrication notes (verified against raw SQLite 3.50.4 by a one-off script;
see the red-team report):
  * CI has one SQLite engine, so a genuine "index written by an older
    engine" state cannot be produced. We corrupt the FTS5 shadow segment
    store directly — ``UPDATE <fts>_data SET block = X'DEADBEEF...'`` —
    which makes ``INSERT INTO <fts>(<fts>) VALUES('integrity-check')``
    raise sqlite3.DatabaseError("database disk image is malformed") on BOTH
    layouts, while 'rebuild' afterwards restores the exact pre-corruption
    MATCH hits.
  * The other candidate (DELETE a ``%_content`` shadow row, leaving orphan
    postings) also trips integrity-check, but its repair legitimately loses
    that row (4/5 hits), so it is unusable for the MATCH-preservation
    assertions and is not used.
  * "Engine changed" is fabricated by writing a stale marker value directly
    into state_meta (the gate keys on marker != sqlite3.sqlite_version).

Observability notes:
  * Counts of special commands come from a connection trace callback
    (``set_trace_callback``), which is independent of how the gate executes
    statements.
  * The behavioral stubs (re-corrupt after 'rebuild'; raise on
    'integrity-check') intercept ``cursor.execute`` via a connection
    factory. On Python 3.11 ``Connection.execute()`` bypasses a Python-level
    ``cursor()`` override, so the stubs rely on the contract signature
    ``_verify_fts_engine_integrity(self, cursor)`` — i.e. the special
    commands run through a cursor obtained from ``self._conn.cursor()``
    (which ``_init_schema`` demonstrably uses).
"""

import logging
import re
import sqlite3

import pytest

from hermes_state import (
    FTS_SQL,
    FTS_TRIGRAM_SQL,
    LEGACY_FTS_SQL,
    LEGACY_FTS_TRIGRAM_SQL,
    SCHEMA_SQL,
    SessionDB,
    _FTS_TRIGGERS,
)

# ── Contract literals (must match the design doc verbatim) ──────────────────
GATE_MARKER_KEY = "fts_integrity_engine"
REPAIR_HINT = "hermes sessions repair"
FTS_TABLES = ("messages_fts", "messages_fts_trigram", "messages_fts_cjk")

# A plausible OLD engine version; tests assert it differs from the current
# one before relying on it to fabricate an engine change.
STALE_ENGINE = "3.31.1"

DEADBEEF_BLOCK = "X'DEADBEEFDEADBEEFDEADBEEFDEADBEEF'"

_INTEGRITY_CHECK_RE = re.compile(r"values\s*\(\s*'integrity-check'\s*\)", re.I)
_REBUILD_RE = re.compile(r"values\s*\(\s*'rebuild'\s*\)", re.I)
_SPECIAL_TABLE_RE = re.compile(
    r"insert\s+into\s+(messages_fts_cjk|messages_fts_trigram|messages_fts)\s*\(",
    re.I,
)


def _fts5_available() -> bool:
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE _probe USING fts5(x)")
        return True
    except sqlite3.OperationalError:
        return False
    finally:
        conn.close()


if not _fts5_available():
    pytest.skip(
        "FTS5 unavailable in this build — engine-integrity gate not testable",
        allow_module_level=True,
    )


@pytest.fixture(autouse=True)
def _hermetic_hermes_home(tmp_path, monkeypatch):
    """Keep schema-parse caches and any home-relative writes off ~/.hermes."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))


# ── Spy plumbing ─────────────────────────────────────────────────────────────


class _GateSpyCursor(sqlite3.Cursor):
    """Cursor double for the engine-integrity gate.

    Counting is done by the trace callback (see _traced_open); this cursor
    only provides the two BEHAVIORAL stubs the trace callback cannot:

    * ``raise_on_integrity_check`` — raise sqlite3.OperationalError from
      every 'integrity-check' execution (spec 8: the open must survive).
    * ``recorrupt_table`` — after the real 'rebuild' for that table runs,
      re-apply the shadow-store corruption so the post-repair re-check
      fails (spec 3: repair-failure path).

    Relies on the contract that the gate executes through a cursor
    (``_verify_fts_engine_integrity(self, cursor)``).
    """

    raise_on_integrity_check = False
    recorrupt_table = None
    integrity_attempts = 0

    def execute(self, sql, params=()):
        if _INTEGRITY_CHECK_RE.search(sql or ""):
            type(self).integrity_attempts += 1
            if type(self).raise_on_integrity_check:
                raise sqlite3.OperationalError(
                    "disk I/O error (fabricated by the acceptance test)"
                )
        result = super().execute(sql, params)
        if type(self).recorrupt_table is not None:
            match = _SPECIAL_TABLE_RE.search(sql or "")
            if (
                match is not None
                and match.group(1) == type(self).recorrupt_table
                and _REBUILD_RE.search(sql or "")
            ):
                super().execute(
                    f"UPDATE {match.group(1)}_data SET block = {DEADBEEF_BLOCK}"
                )
        return result

    def executescript(self, script):
        result = super().executescript(script)
        if type(self).recorrupt_table is not None:
            match = _SPECIAL_TABLE_RE.search(script or "")
            if (
                match is not None
                and match.group(1) == type(self).recorrupt_table
                and _REBUILD_RE.search(script or "")
            ):
                super().execute(
                    f"UPDATE {match.group(1)}_data SET block = {DEADBEEF_BLOCK}"
                )
        return result


class _GateSpyConnection(sqlite3.Connection):
    def cursor(self, *args, **kwargs):
        kwargs.setdefault("factory", _GateSpyCursor)
        return super().cursor(*args, **kwargs)


def _traced_open(
    db_path,
    monkeypatch,
    statements,
    *,
    read_only=False,
    raise_on_integrity_check=False,
    recorrupt_table=None,
):
    """Open a SessionDB with connection-level statement tracing enabled.

    Every SQL statement executed on any connection opened during the
    constructor is appended to *statements* (scoped: the sqlite3.connect
    patch is undone as soon as the constructor returns).
    """
    real_connect = sqlite3.connect

    def spy_connect(*args, **kwargs):
        kwargs["factory"] = _GateSpyConnection
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    _GateSpyCursor.raise_on_integrity_check = raise_on_integrity_check
    _GateSpyCursor.recorrupt_table = recorrupt_table
    _GateSpyCursor.integrity_attempts = 0
    try:
        with monkeypatch.context() as mp:
            mp.setattr(sqlite3, "connect", spy_connect)
            return SessionDB(db_path=db_path, read_only=read_only)
    finally:
        _GateSpyCursor.raise_on_integrity_check = False
        _GateSpyCursor.recorrupt_table = None


def _count_integrity_checks(statements):
    return sum(1 for s in statements if _INTEGRITY_CHECK_RE.search(s))


def _count_rebuilds(statements):
    return sum(1 for s in statements if _REBUILD_RE.search(s))


# ── Raw-database helpers ─────────────────────────────────────────────────────


def _seed_raw_messages(conn, session_id="s1", *, count, start=0, prefix="needle message"):
    conn.execute(
        "INSERT OR IGNORE INTO sessions (id, source, started_at) "
        "VALUES (?, 'test', 1.0)",
        (session_id,),
    )
    for i in range(start, start + count):
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) "
            "VALUES (?, 'user', ?, ?)",
            (session_id, f"{prefix} {i}", float(i)),
        )


def _build_raw_v23_db(db_path, *, messages=4):
    """A v23-layout DB (external-content FTS) with no gate marker yet."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA_SQL + FTS_SQL + FTS_TRIGRAM_SQL)
        _seed_raw_messages(conn, count=messages)
        conn.commit()
    finally:
        conn.close()


def _build_raw_legacy_db(db_path, *, messages=4):
    """A v22-style legacy inline-FTS DB with no gate marker yet."""
    conn = sqlite3.connect(str(db_path))
    try:
        try:
            conn.executescript(SCHEMA_SQL + LEGACY_FTS_SQL + LEGACY_FTS_TRIGRAM_SQL)
        except sqlite3.OperationalError as exc:
            pytest.skip(f"required FTS tokenizer unavailable: {exc}")
        _seed_raw_messages(conn, count=messages)
        conn.commit()
    finally:
        conn.close()


def _marker_value(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT value FROM state_meta WHERE key = ?", (GATE_MARKER_KEY,)
        ).fetchone()
        return None if row is None else row[0]
    finally:
        conn.close()


def _set_marker(db_path, value):
    conn = sqlite3.connect(str(db_path))
    try:
        if value is None:
            conn.execute("DELETE FROM state_meta WHERE key = ?", (GATE_MARKER_KEY,))
        else:
            conn.execute(
                "INSERT OR REPLACE INTO state_meta (key, value) VALUES (?, ?)",
                (GATE_MARKER_KEY, value),
            )
        conn.commit()
    finally:
        conn.close()


def _corrupt_fts_index(db_path, table):
    """Fabricate the 'engine-upgrade malformed FTS index' state.

    Overwrites every shadow segment block with DEADBEEF. Verified by a
    one-off script: this makes the FTS5 special command 'integrity-check'
    raise sqlite3.DatabaseError on both layouts, while 'rebuild' repairs it
    and restores the exact pre-corruption MATCH hits.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(f"UPDATE {table}_data SET block = {DEADBEEF_BLOCK}")
        conn.commit()
    finally:
        conn.close()


def _match_count(db_path, table, term):
    conn = sqlite3.connect(str(db_path))
    try:
        return conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {table} MATCH ?", (term,)
        ).fetchone()[0]
    finally:
        conn.close()


def _existing_fts_tables(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        names = {row[0] for row in rows}
        return {t for t in FTS_TABLES if t in names}
    finally:
        conn.close()


def _legacy_fts_trigger_names(db_path):
    conn = sqlite3.connect(str(db_path))
    try:
        placeholders = ",".join("?" for _ in _FTS_TRIGGERS)
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            f"WHERE type = 'trigger' AND name IN ({placeholders})",
            _FTS_TRIGGERS,
        ).fetchall()
        return {row[0] for row in rows}
    finally:
        conn.close()


def _open_and_close(db_path, monkeypatch, statements, **kwargs):
    """Open + immediately close; statements accumulate from the open only."""
    db = None
    try:
        db = _traced_open(db_path, monkeypatch, statements, **kwargs)
    finally:
        if db is not None:
            db.close()


class TestFtsEngineIntegrityGate:
    # ── Contract interface ────────────────────────────────────────────────

    def test_contract_method_exists_with_cursor_signature(self):
        """The spec pins SessionSchemaMixin._verify_fts_engine_integrity(cursor)."""
        import inspect

        from hermes_state_schema import SessionSchemaMixin

        assert hasattr(SessionSchemaMixin, "_verify_fts_engine_integrity"), (
            "SessionSchemaMixin._verify_fts_engine_integrity is missing — "
            "the engine-version FTS integrity gate is not implemented"
        )
        signature = inspect.signature(
            SessionSchemaMixin._verify_fts_engine_integrity
        )
        positional = [
            p
            for p in signature.parameters.values()
            if p.kind
            in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert len(positional) == 2  # (self, cursor)

    # ── Spec 1: gating ────────────────────────────────────────────────────

    def test_missing_marker_verifies_each_fts_table_exactly_once(
        self, tmp_path, monkeypatch
    ):
        """Marker absent → exactly one integrity-check per existing fts table,
        no rebuilds, marker stamped with the current engine."""
        db_path = tmp_path / "state.db"
        _build_raw_v23_db(db_path, messages=4)
        assert _marker_value(db_path) is None
        pre_tables = _existing_fts_tables(db_path)
        assert len(pre_tables) >= 2  # messages_fts + messages_fts_trigram

        statements = []
        _open_and_close(db_path, monkeypatch, statements)

        post_tables = _existing_fts_tables(db_path)
        # CONTRACT_AMBIGUOUS: whether the gate runs before or after the
        # same-open FTS DDL (e.g. messages_fts_cjk creation) decides which
        # existence snapshot applies; both orderings are contract-conformant,
        # so accept either — but exactly once per table in that snapshot.
        assert _count_integrity_checks(statements) in {
            len(pre_tables),
            len(post_tables),
        }
        assert _count_rebuilds(statements) == 0
        assert _marker_value(db_path) == sqlite3.sqlite_version

    def test_stale_marker_triggers_reverification_and_restamp(
        self, tmp_path, monkeypatch
    ):
        """Marker != current engine → verification runs again and the marker
        is re-stamped with the current engine."""
        assert STALE_ENGINE != sqlite3.sqlite_version
        db_path = tmp_path / "state.db"
        _build_raw_v23_db(db_path, messages=3)
        _set_marker(db_path, STALE_ENGINE)
        pre_tables = _existing_fts_tables(db_path)

        statements = []
        _open_and_close(db_path, monkeypatch, statements)

        assert _count_integrity_checks(statements) in {
            len(pre_tables),
            len(_existing_fts_tables(db_path)),
        }
        assert _count_rebuilds(statements) == 0
        assert _marker_value(db_path) == sqlite3.sqlite_version

    def test_matching_marker_runs_zero_integrity_checks_on_reopen(
        self, tmp_path, monkeypatch
    ):
        """Marker == current engine → open performs zero 'integrity-check'
        executions (zero index cost) and leaves the marker intact."""
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            db.create_session("s1", source="test")
            db.append_message("s1", "user", "needle first")
            db.append_message("s1", "user", "needle second")
        finally:
            db.close()
        # First open must have stamped the marker (spec 1/7).
        assert _marker_value(db_path) == sqlite3.sqlite_version

        statements = []
        _open_and_close(db_path, monkeypatch, statements)

        assert _count_integrity_checks(statements) == 0
        assert _count_rebuilds(statements) == 0
        assert _marker_value(db_path) == sqlite3.sqlite_version

    # ── Spec 2 + 6: detection, repair, v23 external-content coverage ──────

    def test_engine_change_detects_and_repairs_external_content_index(
        self, tmp_path, monkeypatch, caplog
    ):
        """Spec 6: an ordinary SessionDB-created (v23 external-content) DB is
        covered by the gate. Spec 2: corrupted table → exactly one 'rebuild'
        + exactly one re-check; warning names the table; marker stamped;
        MATCH hits identical to pre-corruption."""
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            db.create_session("s1", source="test")
            for i in range(4):
                db.append_message("s1", "user", f"needle payload {i}")
        finally:
            db.close()
        baseline_base = _match_count(db_path, "messages_fts", "needle")
        baseline_trigram = _match_count(db_path, "messages_fts_trigram", "needle")
        assert baseline_base == 4
        assert baseline_trigram == 4

        # Fabricate the engine change + the carried-over malformed index.
        _set_marker(db_path, STALE_ENGINE)
        _corrupt_fts_index(db_path, "messages_fts")
        pre_tables = _existing_fts_tables(db_path)
        assert len(pre_tables) >= 2

        caplog.set_level(logging.DEBUG)
        statements = []
        _open_and_close(db_path, monkeypatch, statements)

        # Spec 2: exactly one rebuild for the corrupted table.
        assert _count_rebuilds(statements) == 1
        # Every existing table checked once + exactly one post-repair re-check
        # (CONTRACT_AMBIGUOUS: gate-vs-DDL ordering, see the marker test above).
        assert _count_integrity_checks(statements) in {
            len(pre_tables) + 1,
            len(_existing_fts_tables(db_path)) + 1,
        }
        # Spec 2: repair is logged as a warning naming the corrupted table
        # (word-bounded so messages_fts_trigram does not satisfy it).
        assert any(
            record.levelno == logging.WARNING
            and re.search(r"\bmessages_fts\b", record.getMessage())
            for record in caplog.records
        )
        assert _marker_value(db_path) == sqlite3.sqlite_version
        # Spec 2: MATCH hit counts identical to pre-corruption.
        assert _match_count(db_path, "messages_fts", "needle") == baseline_base
        assert (
            _match_count(db_path, "messages_fts_trigram", "needle")
            == baseline_trigram
        )

    def test_engine_change_repairs_legacy_inline_trigram_index(
        self, tmp_path, monkeypatch, caplog
    ):
        """Legacy inline (v11–v22) messages_fts_trigram: same gate, same
        repair contract, MATCH hits restored."""
        db_path = tmp_path / "legacy-state.db"
        _build_raw_legacy_db(db_path, messages=4)
        baseline_trigram = _match_count(db_path, "messages_fts_trigram", "needle")
        baseline_base = _match_count(db_path, "messages_fts", "needle")
        assert baseline_trigram == 4

        _set_marker(db_path, STALE_ENGINE)
        _corrupt_fts_index(db_path, "messages_fts_trigram")
        pre_tables = _existing_fts_tables(db_path)
        assert pre_tables == {"messages_fts", "messages_fts_trigram"}

        caplog.set_level(logging.DEBUG)
        statements = []
        _open_and_close(db_path, monkeypatch, statements)

        assert _count_rebuilds(statements) == 1
        # 2 tables verified once + 1 post-repair re-check (legacy has no cjk).
        assert _count_integrity_checks(statements) == len(pre_tables) + 1
        assert any(
            record.levelno == logging.WARNING
            and re.search(r"\bmessages_fts_trigram\b", record.getMessage())
            for record in caplog.records
        )
        assert _marker_value(db_path) == sqlite3.sqlite_version
        assert _match_count(db_path, "messages_fts_trigram", "needle") == (
            baseline_trigram
        )
        assert _match_count(db_path, "messages_fts", "needle") == baseline_base

    # ── Spec 3: repair-failure fallback ───────────────────────────────────

    def test_failed_repair_verification_logs_hint_and_keeps_db_open(
        self, tmp_path, monkeypatch, caplog
    ):
        """'rebuild' succeeds but the re-check still fails (re-corrupted by
        the cursor stub) → logger.error with 'hermes sessions repair', no
        marker stamp, exactly one repair attempt, open survives."""
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            db.create_session("s1", source="test")
            for i in range(3):
                db.append_message("s1", "user", f"needle payload {i}")
        finally:
            db.close()

        _set_marker(db_path, STALE_ENGINE)
        _corrupt_fts_index(db_path, "messages_fts")

        caplog.set_level(logging.DEBUG)
        statements = []
        db = None
        try:
            db = _traced_open(
                db_path, monkeypatch, statements, recorrupt_table="messages_fts"
            )
            # Spec 3: the open must NOT fail, and the DB stays usable.
            db.create_session("survivor", source="test")
        finally:
            if db is not None:
                db.close()

        assert _count_rebuilds(statements) == 1  # one attempt, no repair loop
        assert _GateSpyCursor.integrity_attempts >= 2  # detect + re-check
        assert any(
            record.levelno == logging.ERROR
            and REPAIR_HINT in record.getMessage()
            for record in caplog.records
        ), "repair-verification failure must log an error guiding the user " \
           "to 'hermes sessions repair'"
        # Spec 3: marker must NOT be stamped as verified for this engine.
        assert _marker_value(db_path) != sqlite3.sqlite_version

    # ── Spec 4: read-only opens ───────────────────────────────────────────

    def test_read_only_open_runs_no_gate_and_writes_no_marker(
        self, tmp_path, monkeypatch
    ):
        db_path = tmp_path / "state.db"
        _build_raw_v23_db(db_path, messages=3)
        assert _marker_value(db_path) is None

        statements = []
        ro = None
        try:
            ro = _traced_open(db_path, monkeypatch, statements, read_only=True)
            # Open succeeded and the FTS surfaces are genuinely present.
            assert ro._fts_enabled is True
        finally:
            if ro is not None:
                ro.close()

        assert _count_integrity_checks(statements) == 0
        assert _count_rebuilds(statements) == 0
        assert _marker_value(db_path) is None  # no marker write on ro open

        # Control: the SAME database opened writable DOES run the gate and
        # stamp the marker — proving the skip is the read-only gate policy,
        # not the gate being absent.
        writable_pre_tables = _existing_fts_tables(db_path)
        writable_statements = []
        _open_and_close(db_path, monkeypatch, writable_statements)
        assert _count_integrity_checks(writable_statements) in {
            len(writable_pre_tables),
            len(_existing_fts_tables(db_path)),
        }
        assert _marker_value(db_path) == sqlite3.sqlite_version

    # ── Spec 5: legacy DELETE-fallback (trigger-degraded) path ────────────

    def test_legacy_trigger_degraded_corruption_backfills_from_messages(
        self, tmp_path, monkeypatch
    ):
        """Legacy DB: malformed trigram index + one missing fts trigger →
        open succeeds, index content backfilled from ``messages`` (gap rows
        included), all 6 legacy fts triggers present."""
        db_path = tmp_path / "legacy-degraded.db"
        conn = sqlite3.connect(str(db_path))
        try:
            try:
                conn.executescript(
                    SCHEMA_SQL + LEGACY_FTS_SQL + LEGACY_FTS_TRIGRAM_SQL
                )
            except sqlite3.OperationalError as exc:
                pytest.skip(f"required FTS tokenizer unavailable: {exc}")
            _seed_raw_messages(conn, count=3)
            conn.commit()
            # Degrade one trigger, then write rows the missing trigger let
            # slip past the trigram index (the index-gap this path repairs).
            conn.execute("DROP TRIGGER messages_fts_trigram_insert")
            _seed_raw_messages(conn, count=2, start=100)
            conn.commit()
            conn.execute(
                f"UPDATE messages_fts_trigram_data SET block = {DEADBEEF_BLOCK}"
            )
            conn.commit()
        finally:
            conn.close()
        total_messages = 5
        assert _marker_value(db_path) is None

        statements = []
        db = None
        try:
            db = _traced_open(db_path, monkeypatch, statements)
        finally:
            if db is not None:
                db.close()

        # Spec 5: open succeeded (we got here) and all 6 triggers are back.
        assert _legacy_fts_trigger_names(db_path) == set(_FTS_TRIGGERS)
        # Spec 5: index content backfilled from messages — the 2 gap rows
        # written while the trigger was missing are searchable too.
        assert (
            _match_count(db_path, "messages_fts_trigram", "needle")
            == total_messages
        )
        assert _match_count(db_path, "messages_fts", "needle") == total_messages
        # CONTRACT_AMBIGUOUS: the spec pins the OUTCOME of this path, not the
        # mechanism (gate 'rebuild' vs _rebuild_legacy_fts_indexes DELETE
        # fallback), so require: every existing table verified at least once
        # and no repair loop.
        assert _count_integrity_checks(statements) >= 2
        assert _count_rebuilds(statements) <= 1
        # CONTRACT_AMBIGUOUS: assumes the gate stamps the marker once the
        # post-repair state verifies (strict spec-2 reading).
        assert _marker_value(db_path) == sqlite3.sqlite_version

    # ── Spec 7: no fts tables ─────────────────────────────────────────────

    def test_marker_stamped_when_db_transiently_has_no_fts_tables(
        self, tmp_path, monkeypatch
    ):
        """Spec 7 on an FTS5-capable runtime: a DB that (transiently) has no
        fts tables — e.g. an interrupted optimize-storage — still gets the
        marker stamped."""
        db_path = tmp_path / "no-fts.db"
        _build_raw_v23_db(db_path, messages=2)
        conn = sqlite3.connect(str(db_path))
        try:
            for trigger in _FTS_TRIGGERS:
                conn.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            conn.execute("DROP VIEW IF EXISTS messages_fts_trigram_src")
            conn.execute("DROP TABLE IF EXISTS messages_fts_trigram")
            conn.execute("DROP TABLE IF EXISTS messages_fts")
            conn.commit()
        finally:
            conn.close()
        assert _existing_fts_tables(db_path) == set()
        assert _marker_value(db_path) is None

        statements = []
        _open_and_close(db_path, monkeypatch, statements)

        # The DDL recreates the fts tables during this same open, so the
        # gate-time table set is order-dependent; the marker write is not.
        assert _marker_value(db_path) == sqlite3.sqlite_version
        assert _count_integrity_checks(statements) in {
            0,  # gate ran before the DDL recreated the tables
            len(_existing_fts_tables(db_path)),  # gate ran after
        }

    def test_marker_not_stamped_on_fts5less_runtime(
        self, tmp_path, monkeypatch
    ):
        """Spec 7 (AMENDED 2026-08-14, adjudicated): an FTS5-less runtime
        must NOT stamp the engine marker. A host with no verification
        capability must not vouch for the indexes on behalf of a capable
        host running the same engine version (same conservative principle
        as the probe-None skip), and skipping the gate entirely keeps
        FTS5-less opens zero-cost — the next capable host re-verifies.

        Original pin (strict pre-amendment reading: marker still written)
        was adjudicated against the conservative reading; this test now
        guards the amended contract."""
        db_path = tmp_path / "no-fts5.db"
        conn = sqlite3.connect(str(db_path))
        try:
            conn.executescript(SCHEMA_SQL)
            _seed_raw_messages(conn, count=2)
            conn.commit()
        finally:
            conn.close()

        monkeypatch.setattr(
            SessionDB, "_sqlite_supports_fts5", lambda self, cursor: False
        )

        statements = []
        _open_and_close(db_path, monkeypatch, statements)

        assert _existing_fts_tables(db_path) == set()
        assert _count_integrity_checks(statements) == 0
        assert _marker_value(db_path) is None

    # ── Spec 8: the gate never fails the open ─────────────────────────────

    def test_gate_sqlite_errors_never_fail_the_open(self, tmp_path, monkeypatch):
        """Every 'integrity-check' execution raises sqlite3.OperationalError
        (a sqlite3.DatabaseError subclass) — the open must still succeed and
        the DB stay usable."""
        db_path = tmp_path / "state.db"
        db = SessionDB(db_path=db_path)
        try:
            db.create_session("s1", source="test")
            db.append_message("s1", "user", "needle payload")
        finally:
            db.close()

        _set_marker(db_path, STALE_ENGINE)

        statements = []
        db = None
        try:
            db = _traced_open(
                db_path, monkeypatch, statements, raise_on_integrity_check=True
            )
            # The open survived; the store is still usable for writes.
            db.create_session("survivor", source="test")
        finally:
            if db is not None:
                db.close()

        # The gate really did attempt integrity checks (and hit the raised
        # error) rather than being skipped entirely.
        assert _GateSpyCursor.integrity_attempts >= 1
        # CONTRACT_AMBIGUITY: marker semantics for this path (probe raised
        # OperationalError — repair-vs-swallow is not pinned by the spec)
        # are deliberately left unasserted; only open survival is spec 8.
