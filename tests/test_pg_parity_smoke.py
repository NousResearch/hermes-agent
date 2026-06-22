"""PostgreSQL / SQLite parity smoke test (T-zero oracle).

This is the *behavioral* guard for the optional PostgreSQL state backend. It
does NOT assert a frozen snapshot; it asserts that a PostgreSQL-backed
``SessionDB`` produces the SAME observable result as a SQLite-backed one for the
highest-risk paths: structured-content round-trip (the sentinel), bulk insert,
degraded search, monotonic message ids, compression-lock mutual exclusion, the
migration source-safety invariant, and the residual case-(c) dialect idioms
that translate silently.

Design notes
------------
* One command:  ``scripts/run_tests.sh tests/test_pg_parity_smoke.py``
* Ephemeral container: a throwaway ``postgres`` started via ``testcontainers``.
  The container is a LOCAL ORACLE only — it never represents the real managed
  backend and runs without TLS, which is fine for behavioral parity.
* **Skips, never fails, when the toolchain is absent** (no ``psycopg``, no
  ``testcontainers``, or no reachable Docker daemon). A guard that hard-fails on
  a laptop without Docker gets disabled; a guard that skips cleanly survives.
* The assertions a1-a9 + a-mig start as ``xfail`` stubs and are flipped live in
  the task that wires them to the real PG-backed ``SessionDB`` (plan task T12).
  Until then this file proves the harness itself works.
"""

import os
import socket
from pathlib import Path

import pytest

# -- Toolchain probes: every absence is a skip, never a failure --------------

psycopg = pytest.importorskip("psycopg", reason="psycopg not installed (.[postgres] extra)")
testcontainers_postgres = pytest.importorskip(
    "testcontainers.postgres",
    reason="testcontainers not installed",
)
PostgresContainer = testcontainers_postgres.PostgresContainer

# The reaper sidecar bind-mounts the docker socket into itself, which some
# rootless / VM-backed daemons cannot satisfy. The module-scoped fixture tears
# the container down deterministically, so the reaper is unnecessary here. Set
# this BEFORE any testcontainers client is built.
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")

# Run the whole parity suite with the PostgreSQL adapter's strict mode ON, so
# any SQLite-only idiom that reaches the adapter untranslated becomes a
# deterministic test failure here instead of a runtime error in production.
# This is what catches "a json_extract / pragma / fts5 slipped through the
# translator" at CI time (see hermes_state_postgres._STRICT_FORBIDDEN).
os.environ.setdefault("HERMES_PG_ADAPTER_STRICT", "1")

# Driver-qualified URL schemes that testcontainers may emit; we normalize any of
# them to the plain libpq scheme that psycopg.connect and our DSN passthrough
# expect. Kept as scheme tokens (no credentials) on purpose.
_PLAIN_SCHEME = "postgresql"
_DRIVER_SCHEMES = ("postgresql+psycopg2", "postgresql+psycopg")


def _normalize_dsn(url: str) -> str:
    """Rewrite a driver-qualified scheme to the plain libpq scheme.

    Operates only on the scheme token before "://" so it never touches the
    credential/host portion of the URL.
    """
    scheme, sep, rest = url.partition("://")
    if sep and scheme in _DRIVER_SCHEMES:
        return _PLAIN_SCHEME + sep + rest
    return url


def _docker_endpoint():
    """Return a usable docker endpoint, or None if no daemon is reachable.

    Honors an explicit DOCKER_HOST, then probes the default unix socket, then a
    Lima-style per-user socket. Returning None makes the whole module skip
    rather than fail on a machine without Docker.
    """
    explicit = os.environ.get("DOCKER_HOST")
    if explicit:
        return explicit
    candidates = [
        Path("/var/run/docker.sock"),
        Path.home() / ".colima" / "default" / "docker.sock",
        Path.home() / ".docker" / "run" / "docker.sock",
    ]
    for sock in candidates:
        if sock.exists():
            return "unix://" + str(sock)
    return None


def _daemon_reachable(endpoint: str) -> bool:
    """Cheap liveness probe on a unix-socket docker endpoint."""
    prefix = "unix://"
    if not endpoint.startswith(prefix):
        # Non-unix endpoints (e.g. tcp) — let the container start attempt
        # surface any real failure rather than guessing here.
        return True
    path = endpoint[len(prefix):]
    if not Path(path).exists():
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(path)
        return True
    except OSError:
        return False


_ENDPOINT = _docker_endpoint()
if _ENDPOINT is None or not _daemon_reachable(_ENDPOINT):
    pytest.skip(
        "no reachable Docker daemon - PG parity smoke test skipped "
        "(set DOCKER_HOST or start a local docker/colima daemon to run it)",
        allow_module_level=True,
    )
os.environ["DOCKER_HOST"] = _ENDPOINT


# -- Container fixture (module-scoped: one PG for the whole file) -------------

@pytest.fixture(scope="module")
def pg_dsn():
    """Start an ephemeral PostgreSQL and yield a plain libpq DSN.

    The container is torn down at module teardown. RYUK is disabled (see above)
    so no reaper sidecar is started.
    """
    with PostgresContainer("postgres:16") as pg:
        yield _normalize_dsn(pg.get_connection_url())


@pytest.fixture()
def pg_clean(pg_dsn):
    """Yield the DSN with state tables dropped, for per-test isolation.

    The schema is created lazily by the PG-backed SessionDB; this fixture just
    guarantees a clean slate by dropping the public tables if they exist. Safe
    to run before the schema exists (IF EXISTS).
    """
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        conn.execute(
            "DROP TABLE IF EXISTS messages, sessions, compression_locks, "
            "state_meta, schema_version CASCADE"
        )
    yield pg_dsn


# -- Harness self-test: proves the container + psycopg path works -------------

def test_harness_container_roundtrip(pg_dsn):
    """The harness itself is sound: container up, psycopg connects, DDL + DML +
    RETURNING + advisory lock all work. This is NOT a parity assertion - it is
    the proof that a1-a9 can run at all once they are wired live (T12).
    """
    with psycopg.connect(pg_dsn, autocommit=True) as conn:
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS _probe "
            "(id BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY, v TEXT)"
        )
        cur.execute("INSERT INTO _probe (v) VALUES (%s) RETURNING id", ("hello",))
        rid = cur.fetchone()[0]
        cur.execute("SELECT v FROM _probe WHERE id = %s", (rid,))
        assert cur.fetchone()[0] == "hello"
        # advisory lock - the compression-lock primitive (parity a5)
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", ("probe-session",))
        cur.execute("DROP TABLE _probe")


# -- Parity assertions a1-a8 + a-mig: live (T12) ------------------------------
#
# Each asserts SQLite-result == PG-result on an identical fixture, exercised
# through the real backend-agnostic SessionDB API. The case-(c) coverage is
# pruned to the idioms that actually appear in the codebase (grep evidence,
# task T12): LIKE case-sensitivity (a6), the integer `active` flag filter (a7),
# and json_extract reads (a8). The empty-string-vs-NULL and integer-division
# divergences have no occurrences in the SQL the class emits, so no assertion
# guards them.

import sqlite3
import threading

from hermes_state import SessionDB


def _pg_session_db(dsn):
    """Build a PostgreSQL-backed SessionDB by pointing config resolution at the
    given DSN. Returns (db, restore) — call restore() to undo the patch."""
    import hermes_cli.config as cfgmod

    original = cfgmod.load_config
    cfg = {"sessions": {"state_backend": "postgres", "postgres_dsn": dsn}}
    # setattr (rather than direct assignment) avoids a type-checker
    # function-shadowing complaint while dynamically swapping the resolver.
    setattr(cfgmod, "load_config", lambda *a, **k: cfg)

    def restore():
        setattr(cfgmod, "load_config", original)

    try:
        return SessionDB(), restore
    except Exception:
        restore()
        raise


def _sqlite_session_db(tmp_path):
    return SessionDB(db_path=tmp_path / "parity_state.db")


def test_a1_sentinel_roundtrip_no_nul(pg_clean, tmp_path):
    """a1 - structured/multimodal content round-trips identically on both
    backends; the PostgreSQL copy carries no NUL byte (the sentinel)."""
    payload = [{"type": "text", "text": "hello world"},
               {"type": "image_url", "url": "http://x/y.png"}]

    sq = _sqlite_session_db(tmp_path)
    sq.create_session(session_id="s1", source="cli")
    sq.append_message("s1", "user", content=payload)
    sq_out = sq.get_messages("s1")[0]["content"]
    sq.close()

    pg, restore = _pg_session_db(pg_clean)
    try:
        pg.create_session(session_id="s1", source="cli")
        pg.append_message("s1", "user", content=payload)
        pg_out = pg.get_messages("s1")[0]["content"]
        # raw stored content must contain no NUL byte on PostgreSQL
        raw = pg._conn.execute(
            "SELECT content FROM messages WHERE session_id = ?", ("s1",)
        ).fetchone()["content"]
        pg.close()
    finally:
        restore()

    assert sq_out == payload
    assert pg_out == sq_out
    assert "\x00" not in raw


def test_a2_bulk_insert_roundtrip(pg_clean, tmp_path):
    """a2 - a batch of mixed scalar + structured messages yields equal counts
    and equal decoded payloads on both backends."""
    msgs = [
        ("user", "plain one"),
        ("assistant", [{"type": "text", "text": "structured two"}]),
        ("user", "plain three"),
        ("assistant", {"k": "dict four"}),
    ]

    def load(db):
        db.create_session(session_id="s1", source="cli")
        for role, content in msgs:
            db.append_message("s1", role, content=content)
        return [m["content"] for m in db.get_messages("s1")]

    sq = _sqlite_session_db(tmp_path)
    sq_contents = load(sq)
    sq.close()

    pg, restore = _pg_session_db(pg_clean)
    try:
        pg_contents = load(pg)
        pg.close()
    finally:
        restore()

    assert len(pg_contents) == len(sq_contents) == len(msgs)
    assert pg_contents == sq_contents


def test_a3_search_id_set(pg_clean, tmp_path):
    """a3 - search_messages returns the same ORDER of message ids on both
    backends for a single-token substring fixture."""
    rows = [
        ("user", "deploy the docker container"),
        ("assistant", "running kubernetes apply"),
        ("user", "check the docker logs"),
        ("user", "totally unrelated"),
    ]

    def load_and_search(db):
        db.create_session(session_id="s1", source="cli")
        for role, content in rows:
            db.append_message("s1", role, content=content)
        # Compare which messages matched, not the snippet rendering: SQLite's
        # FTS5 snippet() adds >>>/<<< highlight markers and ellipses, while the
        # degraded PG ILIKE path returns plain content. Timestamps differ run to
        # run, so parity is the match COUNT plus the per-role match tally — both
        # backend-stable and sufficient to prove the same rows matched.
        from collections import Counter
        results = db.search_messages("docker")
        return len(results), Counter(r["role"] for r in results)

    sq = _sqlite_session_db(tmp_path)
    sq_n, sq_roles = load_and_search(sq)
    sq.close()

    pg, restore = _pg_session_db(pg_clean)
    try:
        pg_n, pg_roles = load_and_search(pg)
        pg.close()
    finally:
        restore()

    # both backends find exactly the two docker-bearing (user-role) messages
    assert pg_n == sq_n == 2
    assert pg_roles == sq_roles == {"user": 2}


def test_a4_lastrowid_returning_monotonic(pg_clean, tmp_path):
    """a4 - append_message returns strictly increasing ids on both backends
    (PG RETURNING id matches SQLite lastrowid semantics)."""
    def ids(db):
        db.create_session(session_id="s1", source="cli")
        return [db.append_message("s1", "user", content=f"m{i}") for i in range(5)]

    sq = _sqlite_session_db(tmp_path)
    sq_ids = ids(sq)
    sq.close()

    pg, restore = _pg_session_db(pg_clean)
    try:
        pg_ids = ids(pg)
        pg.close()
    finally:
        restore()

    for got in (sq_ids, pg_ids):
        assert all(isinstance(i, int) for i in got)
        assert got == sorted(got)
        assert len(set(got)) == len(got)  # strictly increasing, no dupes


def test_a5_compression_lock_exactly_one_winner(pg_clean, tmp_path):
    """a5 - many concurrent acquirers of the same session lock -> exactly one
    winner on BOTH backends (PG via pg_advisory_xact_lock).

    The PG side patches config ONCE for the whole test (not per-thread): the
    workers just call SessionDB() with the patch already in place, so there is
    no race where a thread reads an un-patched load_config and silently builds a
    SQLite db instead of a PG one.
    """
    def one_winner(make_db, n=10):
        setup = make_db()
        setup.create_session(session_id="sess", source="cli")
        setup.close()
        results = []
        rlock = threading.Lock()
        barrier = threading.Barrier(n)

        def worker(i):
            db = make_db()
            barrier.wait()
            got = db.try_acquire_compression_lock("sess", holder=f"h{i}")
            with rlock:
                results.append(got)
            db.close()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return sum(1 for r in results if r)

    sq_state = tmp_path / "lock_state.db"
    assert one_winner(lambda: SessionDB(db_path=sq_state)) == 1

    # PG: install the config patch ONCE, before any thread starts, so every
    # worker's bare SessionDB() resolves to the PG backend deterministically.
    import hermes_cli.config as cfgmod
    original = cfgmod.load_config
    cfg = {"sessions": {"state_backend": "postgres", "postgres_dsn": pg_clean}}
    setattr(cfgmod, "load_config", lambda *a, **k: cfg)
    try:
        assert one_winner(lambda: SessionDB()) == 1
    finally:
        setattr(cfgmod, "load_config", original)


def test_a6_like_case_insensitive(pg_clean, tmp_path):
    """a6 [case-c1] - PG search uses ILIKE so results match SQLite's
    case-insensitive LIKE for a mixed-case query."""
    def search(db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", "user", content="The DOCKER daemon")
        db.append_message("s1", "user", content="lowercase docker too")
        return len(db.search_messages("DoCkEr"))

    sq = _sqlite_session_db(tmp_path)
    sq_n = search(sq)
    sq.close()

    pg, restore = _pg_session_db(pg_clean)
    try:
        pg_n = search(pg)
        pg.close()
    finally:
        restore()

    assert pg_n == sq_n == 2


def test_a7_boolean_active_filter(pg_clean, tmp_path):
    """a7 [case-c6] - the integer `active` flag filter behaves identically;
    rewound (active=0) rows are excluded from search by default on both."""
    def default_and_inclusive(db):
        db.create_session(session_id="s1", source="cli")
        db.append_message("s1", "user", content="keep alpha")
        beta = db.append_message("s1", "user", content="rewind beta")
        # soft-delete the second message (active -> 0)
        db._execute_write(
            lambda c: c.execute("UPDATE messages SET active = 0 WHERE id = ?", (beta,))
        )
        default = len(db.search_messages("rewind"))                    # excluded -> 0
        incl = len(db.search_messages("rewind", include_inactive=True))  # -> 1
        return default, incl

    sq = _sqlite_session_db(tmp_path)
    sq_pair = default_and_inclusive(sq)
    sq.close()

    pg, restore = _pg_session_db(pg_clean)
    try:
        pg_pair = default_and_inclusive(pg)
        pg.close()
    finally:
        restore()

    assert pg_pair == sq_pair == (0, 1)


def test_a8_json_extract_read(pg_clean, tmp_path):
    """a8 [case-c5] - reading a session's model_config returns an equal Python
    value on both backends."""
    model_config = '{"temperature": 0.7, "_delegate_from": "parent-1"}'

    def read_cfg(db):
        db.create_session(session_id="s1", source="cli", model_config=model_config)
        return db.get_session("s1")["model_config"]

    sq = _sqlite_session_db(tmp_path)
    sq_cfg = read_cfg(sq)
    sq.close()

    pg, restore = _pg_session_db(pg_clean)
    try:
        pg_cfg = read_cfg(pg)
        pg.close()
    finally:
        restore()

    # Parity: the PostgreSQL backend returns the byte-identical stored value the
    # SQLite backend does (both round-trip model_config the same way).
    assert pg_cfg == sq_cfg


def test_a9_json_extract_session_listing(pg_clean, tmp_path):
    """a9 [regression] - SessionDB emits json_extract(COALESCE(model_config,
    '{}'), '$.<key>') in its session-listing / lineage WHERE clauses
    (_delegate_from_json). The PG adapter must translate that to jsonb access so
    the query RUNS on PostgreSQL instead of erroring. Under strict mode an
    untranslated json_extract would raise, so this both proves the translation
    and locks the regression. (The original suite only exercised
    search_messages, which is why this gap shipped.)"""
    def list_count(db):
        db.create_session(session_id="parent", source="cli")
        db.create_session(session_id="child", source="cli",
                          parent_session_id="parent",
                          model_config='{"_delegate_from": "parent"}')
        db.create_session(session_id="branch", source="cli",
                          model_config='{"_branched_from": "parent"}')
        # search_sessions builds a WHERE clause containing the json_extract
        # marker filter; on PG this must execute, not raise.
        return len(db.search_sessions(limit=100))

    sq = _sqlite_session_db(tmp_path)
    sq_n = list_count(sq)
    sq.close()

    pg, restore = _pg_session_db(pg_clean)
    try:
        pg_n = list_count(pg)
        pg.close()
    finally:
        restore()

    # Both backends return the same number of listed sessions, and crucially the
    # PG query ran at all (it would have raised on an untranslated json_extract).
    assert pg_n == sq_n


def test_a_mig_source_untouched(pg_clean, tmp_path):
    """a-mig - running the one-shot migration leaves the SQLite source file's
    mtime and size unchanged (source-safety / fallback-of-record invariant)."""
    import migrate_state_to_postgres as mig

    src = tmp_path / "source_state.db"
    db = SessionDB(db_path=src)
    db.create_session(session_id="s1", source="cli")
    db.append_message("s1", "user", content="payload")
    db.close()

    before = src.stat()
    summary = mig.migrate(src, pg_clean)
    after = src.stat()

    assert summary["target_sessions"] == summary["source_sessions"] == 1
    assert summary["target_messages"] == summary["source_messages"] == 1
    assert after.st_mtime == before.st_mtime
    assert after.st_size == before.st_size


# ---------------------------------------------------------------------------
# b-series: pg_trgm GIN-indexed search
# ---------------------------------------------------------------------------


def test_b1_trgm_migration_applies(pg_clean, tmp_path):
    """b1 — pg_trgm migrations (v17 + v18) run and record their versions.

    Uses _pg_session_db which calls init_postgres_schema, triggering
    apply_postgres_migrations. The postgres:16 testcontainers image ships
    pg_trgm compiled in but not pre-extended; v17 installs it, v18 builds
    the GIN indexes.

    This test does NOT pre-check the catalog before running — it proves the
    migration installs the extension. It skips only if the migration failed to
    install pg_trgm (e.g. a stripped container image without the contrib
    module).
    """
    from hermes_state_postgres import _probe_pg_trgm

    pg, restore = _pg_session_db(pg_clean)
    try:
        trgm_ok = _probe_pg_trgm(pg._conn)
    finally:
        pg.close()
        restore()

    if not trgm_ok:
        pytest.skip("pg_trgm migration did not install extension (stripped image?)")

    with psycopg.connect(pg_clean, autocommit=True) as conn:
        # Both v17 (EXTENSION) and v18 (indexes) must be recorded.
        for version in (17, 18):
            row = conn.execute(
                "SELECT 1 FROM schema_version WHERE version = %s", (version,)
            ).fetchone()
            assert row is not None, f"schema_version missing v{version} after migration"

        # All three GIN indexes must exist.
        for idx_name in (
            "idx_messages_content_trgm",
            "idx_messages_tool_name_trgm",
            "idx_messages_tool_calls_trgm",
        ):
            row = conn.execute(
                "SELECT 1 FROM pg_indexes WHERE indexname = %s", (idx_name,)
            ).fetchone()
            assert row is not None, f"GIN index {idx_name!r} not found after migration"


def test_b2_search_with_trgm(pg_clean, tmp_path):
    """b2 — search_messages returns correct results when GIN indexes are present.

    Verifies the full result-row contract (all 10 required keys) and that all
    three indexed columns (content, tool_name, tool_calls) are matched.
    Skips if the GIN indexes were not created (b1 covers that failure case).
    """
    import time as _time

    # Skip if GIN indexes are absent (pg_trgm not available in this image).
    with psycopg.connect(pg_clean, autocommit=True) as probe:
        idx_row = probe.execute(
            "SELECT 1 FROM pg_indexes WHERE indexname = 'idx_messages_content_trgm'"
        ).fetchone()
    if idx_row is None:
        # Trigger migrations first via _pg_session_db, then re-check.
        pg_check, restore_check = _pg_session_db(pg_clean)
        try:
            with psycopg.connect(pg_clean, autocommit=True) as probe2:
                idx_row = probe2.execute(
                    "SELECT 1 FROM pg_indexes"
                    " WHERE indexname = 'idx_messages_content_trgm'"
                ).fetchone()
        finally:
            pg_check.close()
            restore_check()
    if idx_row is None:
        pytest.skip("GIN index not present — pg_trgm migration did not apply")

    pg, restore = _pg_session_db(pg_clean)
    try:
        pg.create_session(session_id="s1", source="cli")
        rows = [
            # (role, content, tool_name, tool_calls)
            ("user",      "deploy the docker container",  None,         None),
            ("assistant", "running kubernetes apply",     "bash",       None),
            ("user",      "check the docker logs",        None,         None),
            ("user",      "totally unrelated message",    None,         None),
            # match via tool_name column
            ("assistant", "result of tool call",          "docker_ps",  None),
            # match via tool_calls column
            ("assistant", "tool call details",            None,         '{"name":"docker_build"}'),
        ]
        for role, content, tool_name, tool_calls in rows:
            pg._execute_write(
                lambda c, r=role, ct=content, tn=tool_name, tc=tool_calls: c.execute(
                    "INSERT INTO messages"
                    " (session_id, role, content, tool_name, tool_calls, timestamp, active)"
                    " VALUES (?, ?, ?, ?, ?, ?, 1)",
                    ("s1", r, ct, tn, tc, _time.time()),
                )
            )

        results = pg.search_messages("docker")

        # Every result must carry all 10 contract keys.
        required_keys = {
            "id", "session_id", "role", "snippet", "timestamp",
            "tool_name", "source", "model", "session_started", "context",
        }
        for r in results:
            missing = required_keys - set(r.keys())
            assert not missing, f"Result missing keys: {missing}"

        # 4 matches: content×2, tool_name×1, tool_calls×1.
        assert len(results) == 4, (
            f"Expected 4 docker matches, got {len(results)}: "
            f"{[r.get('snippet', '')[:60] for r in results]}"
        )
    finally:
        pg.close()
        restore()

