"""Optional PostgreSQL state backend for :class:`hermes_state.SessionDB`.

This module owns PostgreSQL connections and backend dispatch. Schema, SQL
translation, and migration helpers live in neighboring ``hermes_state_pg_*``
modules. SQLite remains the default backend; the driver is loaded only when
the operator selects ``sessions.state_backend = "postgres"``.

Design contract (kept deliberately narrow):

* A thin cursor adapter (:class:`_PostgresConnection` / :class:`_PostgresCursor`)
  translates only the *closed* set of SQLite idioms the ``SessionDB`` class
  emits at its write/read sites: ``?`` -> ``%s``, ``BEGIN IMMEDIATE`` ->
  ``BEGIN ISOLATION LEVEL SERIALIZABLE``, ``INSERT OR IGNORE`` -> ``INSERT ... ON CONFLICT DO NOTHING``, and
  ``RETURNING id`` synthesis so ``cursor.lastrowid`` keeps working on message
  inserts. Everything else passes through untouched.
* Queries that cannot be mechanically translated (full-text search, the
  compression-lock acquisition, migration imports) are purpose-built methods
  here, dispatched from explicit seams in ``SessionDB``.
* The connection string (DSN) is passed through to the driver unchanged, so
  TLS mode, host, port, and credentials all come from the operator's DSN.

The driver dependency (``psycopg``) is optional and imported lazily; a base
install without the ``postgres`` extra never imports the driver.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import hermes_state_pg_schema as pg_schema
from hermes_state_pg_sql import _needs_returning_id, _translate_sql

logger = logging.getLogger(__name__)


def _fts_column_available(conn: Any) -> bool:
    """Return True if the fts_content column exists on the messages table.

    Cached on the connection object after the first probe so subsequent
    search calls pay zero extra round-trips. Uses pg_catalog (faster than
    information_schema, no access-check view overhead).
    """
    cached = getattr(conn, "_fts_col_available", None)
    if cached is not None:
        return cached
    row = conn.raw.execute(
        "SELECT 1 FROM pg_catalog.pg_attribute a"
        " JOIN pg_catalog.pg_class c ON c.oid = a.attrelid"
        " WHERE c.oid = to_regclass('messages')"
        " AND a.attname = 'fts_content' AND a.attnum > 0 AND NOT a.attisdropped"
    ).fetchone()
    result = row is not None
    conn._fts_col_available = result
    return result


def _fts_backfill_complete(conn: Any) -> bool:
    """Return True if every messages row has a non-NULL fts_content.

    While legacy rows remain un-backfilled (fts_content IS NULL), an FTS-only
    query would silently miss them, so search_messages_postgres stays on the
    ILIKE path (which covers all rows) until backfill finishes.

    Check each search: another writer or a bulk import can add unindexed rows
    after this reader observed a complete backfill. The partial NULL index
    keeps the all-indexed case cheap.
    """
    if not _fts_column_available(conn):
        return False
    try:
        row = conn._conn.execute(
            "SELECT 1 FROM messages WHERE fts_content IS NULL LIMIT 1"
        ).fetchone()
        complete = row is None
    except Exception:
        complete = False
    return complete


def _build_tsquery(conn: Any, query: str) -> Optional[str]:
    """Parse *query* into a PostgreSQL tsquery text, or return None.

    Uses ``websearch_to_tsquery('simple', ...)`` which handles implicit AND,
    OR, quoted phrases, and ``-`` exclusion. Falls back to
    ``plainto_tsquery('simple', ...)`` on parse error (e.g. a bare ``-``).
    Returns None when:
    - the ``fts_content`` column doesn't exist yet (v19 migration pending), or
    - the tsquery is empty (stopword-only or unrecognised input).
    The caller treats None as "use ILIKE instead".

    Uses 'simple' dictionary (lowercase only, no stemming) so code identifiers,
    proper nouns, and non-English terms are indexed and matched verbatim.
    """
    if not _fts_column_available(conn):
        return None

    # Strip FTS5-specific prefix wildcard (*) — websearch_to_tsquery handles
    # quoted phrases and OR/- natively but not the FTS5 "term*" prefix form.
    # Only strip a * that terminates a word token (followed by whitespace or
    # end-of-string), i.e. the FTS5 prefix marker "deploy*". Asterisks embedded
    # mid-token (e.g. "a*b") are left intact so non-FTS queries keep meaning.
    cleaned = re.sub(r"(\w)\*+(?=\s|$)", r"\1", query).strip()
    if not cleaned:
        return None

    try:
        row = conn._conn.execute(
            "SELECT websearch_to_tsquery('simple', %s)::text",
            (cleaned,),
        ).fetchone()
        tsq = row[0] if row else None
    except Exception:
        tsq = None

    if not tsq:
        try:
            row = conn._conn.execute(
                "SELECT plainto_tsquery('simple', %s)::text",
                (cleaned,),
            ).fetchone()
            tsq = row[0] if row else None
        except Exception:
            return None

    return tsq or None  # empty string → None


def _update_fts_content(conn: Any, msg_id: int) -> None:
    """Update the fts_content column for a freshly inserted message row.

    Called by ``_PostgresCursor.execute`` immediately after each message INSERT
    so new rows are searchable without a separate backfill. Silently skipped
    when the v19 column doesn't exist yet. Uses 'simple' dictionary to match
    _build_tsquery.
    """
    raw = conn.raw
    in_transaction = conn._in_transaction
    if in_transaction:
        raw.execute("SAVEPOINT hermes_fts_content")
    try:
        if _fts_column_available(conn):
            # Read the stored fields: INSERT ... SELECT clones, literal SQL
            # values, and named parameters need not match the input column order.
            raw.execute(
                "UPDATE messages SET fts_content = to_tsvector('simple', "
                "COALESCE(content, '') || ' ' || COALESCE(tool_name, '') || ' ' || "
                "COALESCE(tool_calls, '')) WHERE id = %s",
                (msg_id,))
    except Exception:
        # A caught PostgreSQL error still aborts its transaction. Roll back
        # only the derived index update so COMMIT cannot silently discard the
        # canonical message INSERT. Outside a transaction autocommit does this.
        if in_transaction:
            raw.execute("ROLLBACK TO SAVEPOINT hermes_fts_content")
        logger.debug("fts_content update failed for message %s", msg_id, exc_info=True)
    finally:
        if in_transaction:
            raw.execute("RELEASE SAVEPOINT hermes_fts_content")


class _PostgresCursor:
    """Translate the small ``sqlite3.Cursor`` surface SessionDB relies on."""

    def __init__(self, cursor: Any, conn: Any = None):
        self._cursor = cursor
        self._conn = conn  # _PostgresConnection reference, for fts_content hook
        self.lastrowid: Optional[int] = None

    def execute(self, sql: str, params: Any = ()):
        translated = _translate_sql(sql)
        want_id = _needs_returning_id(translated) and " RETURNING " not in translated.upper()
        if want_id:
            translated = translated.rstrip().rstrip(";") + " RETURNING id"
        # NO reconnect-and-replay here, deliberately.
        #
        # A dropped connection is usually discovered on this round-trip rather
        # than at cursor() time, so replaying the statement on a fresh
        # connection looks like the obvious repair. It is not safe:
        #
        #   * ``_reconnect()`` builds the replacement with ``autocommit=True``,
        #     so the replayed statement does NOT belong to the ``BEGIN`` the
        #     enclosing ``_execute_write`` closure opened. It commits on its
        #     own while the rest of the closure believes it is still in a
        #     transaction, and the closure can then report success over a
        #     partially-applied write.
        #   * The new connection does not hold the transaction-scoped advisory
        #     locks the original acquired, so the serialization those locks
        #     provide is silently lost mid-closure.
        #   * The first execute's outcome is UNKNOWN when the response is lost:
        #     the server may already have applied it. Replaying can therefore
        #     double-apply a non-idempotent statement.
        #
        # Connection loss during a write must fail closed and propagate. Retry
        # belongs at the whole-closure level in ``_execute_write``, and only
        # for known-clean aborts (40001 serialization failure / 40P01 deadlock)
        # where the server guarantees the transaction applied nothing.
        self._cursor.execute(translated, params or ())
        if want_id:
            row = self._cursor.fetchone()
            self.lastrowid = row[0] if row else None
            # New VALUES inserts need an index entry. INSERT ... SELECT may
            # already copy fts_content; rebuilding from the stored row remains
            # correct for both forms without interpreting input parameters.
            if self.lastrowid is not None and self._conn is not None:
                _update_fts_content(self._conn, self.lastrowid)
        return self

    def executescript(self, sql_script: str):
        """Run a multi-statement DDL/script. PostgreSQL's driver executes one
        statement per ``execute``; split on ``;`` and run the non-empty pieces.
        """
        for statement in sql_script.split(";"):
            statement = statement.strip()
            if statement:
                self.execute(statement)
        return self

    def executemany(self, sql: str, seq_of_params):
        """Execute *sql* once per row in *seq_of_params*.

        Translates the same closed set of SQLite idioms as ``execute`` and
        delegates to the underlying psycopg cursor's ``executemany``.
        ``lastrowid`` is not set (matches sqlite3 behaviour for executemany).
        """
        translated = _translate_sql(sql)
        params_list = list(seq_of_params)
        if params_list:
            self._cursor.executemany(translated, params_list)
        return self

    def fetchone(self):
        row = self._cursor.fetchone()
        if row is None:
            return None
        return _PostgresRow(self._columns(), tuple(row))

    def fetchall(self):
        cols = self._columns()
        return [_PostgresRow(cols, tuple(r)) for r in self._cursor.fetchall()]

    def __iter__(self):
        return iter(self.fetchall())

    def _columns(self) -> List[str]:
        return [desc[0] for desc in (self._cursor.description or [])]

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount


class _PostgresRow:
    """A row that supports both ``row["col"]`` and ``row[0]`` plus ``keys()`` and
    ``dict(row)`` — the access patterns SessionDB uses on ``sqlite3.Row``."""

    def __init__(self, columns: List[str], values: Tuple[Any, ...]):
        self._columns = columns
        self._values = values
        self._map = dict(zip(columns, values))

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return self._map[key]
        return self._values[key]

    def __contains__(self, key: Any) -> bool:
        return key in self._map

    def keys(self) -> List[str]:
        return list(self._columns)

    def get(self, key: str, default: Any = None) -> Any:
        return self._map.get(key, default)


def _is_connection_broken_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like a dropped/broken PostgreSQL connection.

    psycopg raises ``OperationalError`` / ``InterfaceError`` for a server-side
    close, network drop, or admin terminate (SQLSTATE class 08, plus the
    generic "connection is closed" / "consuming input failed" messages we see
    in the wild when the server restarts under a long-lived gateway process).
    We classify by SQLSTATE first (class 08, ``57P01`` admin shutdown,
    ``57P02`` crash shutdown, ``57P03`` cannot connect now) and fall back to
    a tight message-substring check so the heuristic still fires when the
    driver does not expose ``sqlstate`` on a low-level transport failure.
    """
    sqlstate = getattr(exc, "sqlstate", None) or ""
    if sqlstate.startswith("08") or sqlstate in ("57P01", "57P02", "57P03"):
        return True
    msg = str(exc).lower()
    return (
        "connection is closed" in msg
        or "connection already closed" in msg
        or "server closed the connection" in msg
        or "consuming input failed" in msg
        or "connection is bad" in msg
        or "ssl connection has been closed" in msg
        or "terminating connection" in msg
        # Raw TCP teardown: psycopg surfaces this when the server is killed
        # mid-query and no SQLSTATE is available.
        or "connection reset by peer" in msg
        or "broken pipe" in msg
    )


class _PostgresConnection:
    """Adapter exposing the ``sqlite3.Connection`` surface SessionDB calls.

    The adapter holds the DSN so it can transparently reconnect when the
    underlying psycopg connection has been closed by the server (idle
    timeout, server restart, admin terminate, network drop). Without this,
    a single broken connection cascades into every subsequent state write
    failing for the lifetime of the host process, because SessionDB caches
    one ``_PostgresConnection`` for its whole lifetime.

    Reconnect strategy:
      * Before yielding a cursor / committing / rolling back, check the
        psycopg connection's ``closed`` flag and reopen if it is non-zero.
      * On an ``OperationalError`` / ``InterfaceError`` that looks like a
        dropped connection (see ``_is_connection_broken_error``), reopen
        once and retry the call. Surfaces a single layered ``RuntimeError``
        if the reconnect itself fails, so callers see the real cause.

    The DSN may be None (legacy callers that wrap a pre-built connection
    directly); in that case reconnect is disabled and the original behaviour
    is preserved.
    """

    def __init__(self, conn: Any, dsn: Optional[str] = None):
        self._conn = conn
        self._dsn = dsn
        self._read_only = False
        # True between a successful BEGIN and the commit/rollback that ends it.
        # While set, transparent reconnect is forbidden: a replacement
        # connection cannot carry the open transaction or its
        # transaction-scoped advisory locks, so silently swapping one in lets a
        # partially applied write be reported as a success.
        self._in_transaction = False

    # ------------------------------------------------------------------
    # Reconnect helpers
    # ------------------------------------------------------------------

    def _ensure_live(self) -> None:
        """Reopen ``self._conn`` if it has been closed under us.

        No-op when the connection is healthy or when no DSN is available
        (legacy code-path).
        """
        if self._dsn is None:
            return
        if getattr(self._conn, "closed", 0):
            self._reconnect()

    def _reconnect(self) -> None:
        """Replace ``self._conn`` with a fresh psycopg connection.

        Raises ``RuntimeError`` (chained from the underlying psycopg error)
        if the reconnect fails, so callers get a clear signal rather than a
        bare driver exception bubbling out of an adapter method.
        """
        if self._dsn is None:
            raise RuntimeError(
                "PostgreSQL adapter has no DSN; cannot reconnect a closed "
                "connection"
            )
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PostgreSQL state backend requires psycopg"
            ) from exc
        # Best-effort close of the stale handle so we don't leak server-side
        # resources if it is half-open.
        try:
            self._conn.close()
        except Exception:
            pass
        try:
            replacement = psycopg.connect(self._dsn, autocommit=True)
            try:
                _require_supported_server(replacement)
                if self._read_only:
                    replacement.execute("SET default_transaction_read_only = on")
            except BaseException:
                replacement.close()
                raise
            self._conn = replacement
        except Exception as exc:
            raise RuntimeError(
                f"PostgreSQL reconnect failed: {exc}"
            ) from exc

    def _call_with_retry(self, op_name: str, fn):
        """Run ``fn()`` with one transparent reconnect-and-retry.

        ``fn`` is a zero-arg callable that performs the actual psycopg
        operation. If it raises and the exception looks like a dropped
        connection, we reconnect once and retry. Any other exception (or a
        second drop in a row) propagates.

        **Never reconnects while a transaction is open.** ``_reconnect()``
        builds the replacement with ``autocommit=True``, so a statement issued
        on it does not belong to the ``BEGIN`` the caller opened: it
        self-commits, without the transaction-scoped advisory locks the
        original connection held, and a later ``commit()`` on the replacement
        succeeds vacuously — letting ``_execute_write`` report success over a
        partially applied transaction. Once ``BEGIN`` has succeeded, connection
        loss must fail closed so the caller can abort the whole write.
        """
        if self._in_transaction:
            # Surface the drop rather than papering over it. The caller's
            # transaction is already unrecoverable on this connection.
            self._ensure_live_in_transaction()
            return fn()
        self._ensure_live()
        try:
            return fn()
        except Exception as exc:
            if not _is_connection_broken_error(exc):
                raise
            # Connection looked open but the server had already gone away,
            # or the call itself surfaced the drop. Reconnect and retry once.
            self._reconnect()
            return fn()

    def _ensure_live_in_transaction(self) -> None:
        """Assert the connection is still usable inside an open transaction.

        Mirrors ``_ensure_live`` but refuses to repair: a replacement
        connection cannot carry the open transaction, so a dead handle here is
        a terminal condition for the caller's write.
        """
        if getattr(self._conn, "closed", False):
            raise RuntimeError(
                "PostgreSQL connection was lost inside an open transaction; "
                "NOT reconnecting, because a fresh connection cannot carry the "
                "in-flight BEGIN or its transaction-scoped advisory locks. "
                "Failing closed so the write is not reported as successful."
            )

    # ------------------------------------------------------------------
    # sqlite3.Connection-shaped surface
    # ------------------------------------------------------------------

    def cursor(self):
        return _PostgresCursor(
            self._call_with_retry("cursor", lambda: self._conn.cursor()),
            conn=self,
        )

    def execute(self, sql: str, params: Any = ()) -> _PostgresCursor:
        cur = self.cursor().execute(sql, params)
        # Track transaction lifetime so _call_with_retry can refuse to swap the
        # connection out from under an open BEGIN. _translate_sql rewrites
        # SQLite's "BEGIN IMMEDIATE" to a serializable BEGIN, so match on the translated
        # form's leading keyword rather than the caller's dialect.
        head = sql.strip().upper()
        if head.startswith("BEGIN") or head.startswith("START TRANSACTION"):
            self._in_transaction = True
        elif head.startswith("COMMIT") or head.startswith("ROLLBACK"):
            self._in_transaction = False
        return cur

    def executemany(self, sql: str, seq_of_params) -> _PostgresCursor:
        return self.cursor().executemany(sql, seq_of_params)

    def executescript(self, sql_script: str) -> _PostgresCursor:
        return self.cursor().executescript(sql_script)

    def commit(self):
        # SAFETY: retrying commit() on a new connection is UNSAFE. If the
        # connection drops during or after commit, the original transaction's
        # outcome is UNKNOWN — the data may or may not have landed. A fresh
        # empty commit on a new connection succeeds vacuously, which would
        # report durable writes that never happened (synthetic success over an
        # unknown outcome). Fail closed instead: surface the connection error so
        # _execute_write sees a genuine failure and does NOT treat it as success.
        try:
            info = getattr(self._conn, "info", None)
            if info is not None:
                from psycopg.pq import TransactionStatus

                if info.transaction_status == TransactionStatus.INERROR:
                    raise RuntimeError("PostgreSQL transaction is aborted; refusing to report a successful commit")
            return self._conn.commit()
        except Exception as exc:
            if _is_connection_broken_error(exc):
                raise RuntimeError(
                    "PostgreSQL commit: connection lost during or after COMMIT — "
                    "transaction outcome is UNKNOWN; NOT retrying to avoid "
                    "reporting synthetic success over an unknown write"
                ) from exc
            raise
        finally:
            # The transaction is over either way: on success it committed, on
            # failure the caller must abort. Clearing here re-enables
            # transparent reconnect for the next independent operation.
            self._in_transaction = False

    def rollback(self):
        # Rollback on a dead connection is a no-op (the server already rolled
        # back on disconnect). Best-effort: swallow connection errors so the
        # caller's except/finally path is not blocked by a cascade error.
        try:
            return self._conn.rollback()
        except Exception as exc:
            if _is_connection_broken_error(exc):
                return None
            raise
        finally:
            self._in_transaction = False

    def close(self):
        # Close is best-effort and never reconnects — the caller is shutting
        # down. Swallow OperationalError on an already-dead handle.
        self._in_transaction = False
        try:
            return self._conn.close()
        except Exception:
            return None

    @property
    def raw(self) -> Any:
        """The underlying psycopg connection (for advisory-lock SQL etc.).

        Calls ``_ensure_live`` first so callers that bypass the adapter
        (advisory-lock SQL, migration paths) also get a live connection —
        EXCEPT inside an open transaction, where a replacement connection
        cannot carry the in-flight ``BEGIN`` or its transaction-scoped
        advisory locks. Handing one out there would let a caller issue
        statements that self-commit on an ``autocommit=True`` connection,
        outside the transaction the caller believes it is in. Fail closed
        instead, matching ``_call_with_retry``.
        """
        if self._in_transaction:
            self._ensure_live_in_transaction()
            return self._conn
        self._ensure_live()
        return self._conn


def _require_supported_server(raw) -> None:
    # The shared query translator uses SQL/JSON predicates and subqueries
    # without aliases, both introduced in PostgreSQL 16. Fail before any DDL.
    version = raw.info.server_version
    if version < 160000:
        raise RuntimeError(
            "PostgreSQL state backend requires PostgreSQL 16 or newer; "
            f"connected server is PostgreSQL {version // 10000}."
        )


def connect_postgres(database_url: str) -> _PostgresConnection:
    """Open a PostgreSQL connection wrapped in the SQLite-compatible adapter.

    The DSN is passed through unchanged, so the operator's connection string
    fully determines TLS mode, host, port, and credentials. The DSN is also
    retained on the adapter so it can transparently reconnect when the
    underlying connection drops (server restart, idle timeout, admin
    terminate).
    """
    try:
        import psycopg
    except ImportError:
        # Not installed yet: try the lazy-install path (same mechanism the
        # other opt-in backends use) before giving up with instructions.
        try:
            from tools.lazy_deps import ensure as _lazy_ensure

            _lazy_ensure("state.postgres", prompt=False)
            import psycopg
        except Exception as exc:  # pragma: no cover - exercised via install state
            raise RuntimeError(
                "PostgreSQL state backend requires psycopg; install the "
                "'postgres' extra (pip install 'hermes-agent[postgres]')"
            ) from exc
    raw = psycopg.connect(database_url, autocommit=True)
    try:
        _require_supported_server(raw)
    except BaseException:
        raw.close()
        raise
    return _PostgresConnection(raw, dsn=database_url)


# ---------------------------------------------------------------------------
# Backend resolution + open entrypoint
# ---------------------------------------------------------------------------

_ENV_DSN_KEYS = ("HERMES_STATE_DATABASE_URL", "HERMES_STATE_POSTGRES_DSN")
_ENV_BACKEND_KEYS = ("HERMES_STATE_BACKEND",)


def resolve_postgres_dsn(config: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Return the configured PostgreSQL DSN, or None when not selected.

    Resolution order, first non-empty wins:
      1. ``HERMES_STATE_DATABASE_URL`` / ``HERMES_STATE_POSTGRES_DSN`` env vars
      2. ``sessions.postgres_dsn`` in config.yaml

    Backend selection (must equal "postgres" to engage this module):
      1. ``HERMES_STATE_BACKEND`` env var
      2. ``sessions.state_backend`` in config.yaml

    Returns None when ``sessions.state_backend`` is not "postgres" (the default
    "sqlite" path), so callers can cheaply decide whether to engage this module.

    Fail-loud invariant: ``None`` means "Postgres was NOT selected."  It never
    means "selection could not be evaluated."  Once the operator has expressed an
    explicit selection via an env var, any failure in config loading or elsewhere
    MUST propagate as a targeted error rather than silently returning None (which
    the caller interprets as "use SQLite instead").
    """
    # Read the env backend selector FIRST — before any config I/O.  A config
    # load failure must not mask an explicit env-var selection.
    env_backend = ""
    for key in _ENV_BACKEND_KEYS:
        env_val = (os.environ.get(key) or "").strip().lower()
        if env_val:
            env_backend = env_val
            break
    if env_backend in ("postgresql", "pg"):
        env_backend = "postgres"

    if config is None:
        # The backend selector is load-bearing: if this file is the only place
        # Postgres is selected, silently reading defaults out of a malformed
        # file routes the process to SQLite and splits history.
        #
        # ``load_config()`` deliberately degrades a broken config.yaml to
        # DEFAULT_CONFIG (or last-known-good) rather than raising — correct for
        # its own callers, wrong here, because the degraded value is
        # indistinguishable from a genuine "sqlite" selection. So validate the
        # file with a strict raw parse FIRST, then use the normal loader for
        # the merged values.
        authority = _assert_active_config_parseable()
        try:
            from hermes_cli.config import load_config

            config = load_config()
        except Exception as _cfg_exc:
            selected = str((authority.get("sessions") or {}).get("state_backend") or "").strip().lower()
            if env_backend == "postgres" or (not env_backend and selected in {"postgres", "postgresql", "pg"}):
                # The operator explicitly selected Postgres via env var.
                # A config-loading failure must NOT silently degrade to SQLite.
                raise RuntimeError(
                    f"PostgreSQL state is selected but config loading "
                    f"failed; cannot evaluate backend selection: {_cfg_exc}"
                ) from _cfg_exc
            # No explicit env selection — genuinely not configured for Postgres.
            return None

    if not isinstance(config, dict):
        raise RuntimeError("State backend configuration must be a mapping")
    sessions_cfg = config.get("sessions") or {}
    if not isinstance(sessions_cfg, dict):
        raise RuntimeError("sessions must be a mapping for backend selection")

    # Resolve backend: env var (already read) takes precedence over config.yaml.
    backend = env_backend
    if not backend:
        backend = str(sessions_cfg.get("state_backend") or "sqlite").strip().lower()
        if backend in ("postgresql", "pg"):
            backend = "postgres"

    if backend not in {"sqlite", "postgres"}:
        raise RuntimeError(f"Unknown sessions.state_backend: {backend!r}")
    if backend == "sqlite":
        return None

    for key in _ENV_DSN_KEYS:
        env_val = (os.environ.get(key) or "").strip()
        if env_val:
            return env_val
    dsn = (sessions_cfg.get("postgres_dsn") or "").strip()
    if not dsn:
        raise RuntimeError(
            "sessions.state_backend is 'postgres' but no DSN was provided; set "
            "sessions.postgres_dsn, HERMES_STATE_DATABASE_URL, or "
            "HERMES_STATE_POSTGRES_DSN"
        )
    return dsn


def _dsn_from_profile_env(profile_dir: Any) -> str:
    """Read a Postgres DSN out of a specific profile's own ``.env`` file.

    Parses the file directly rather than loading it into ``os.environ``: the
    caller wants the TARGET profile's credential, and mutating process-global
    state would let a concurrent ``SessionDB()`` on another thread observe it
    and open the wrong physical store.

    Only the two documented DSN keys are read, in the same precedence
    ``resolve_postgres_dsn`` uses. Returns "" when the file is absent or holds
    neither key. A malformed line is skipped rather than raising: unlike the
    backend *selector*, an unparseable line here cannot silently redirect the
    store — a missing DSN raises at the call site.
    """
    from pathlib import Path

    env_path = Path(profile_dir) / ".env"
    if not env_path.is_file():
        return ""

    from dotenv import dotenv_values

    try:
        # Follow the normal .env quoting/comment rules without borrowing values
        # from the process's active profile during interpolation.
        found = dotenv_values(env_path, interpolate=False)
    except OSError:
        return ""

    for key in _ENV_DSN_KEYS:
        if found.get(key):
            return found[key]
    return ""


def _assert_active_config_parseable() -> Dict[str, Any]:
    """Raise if the ACTIVE profile's config.yaml exists but cannot be parsed.

    ``hermes_cli.config.load_config()`` deliberately degrades a malformed
    config.yaml to DEFAULT_CONFIG (or the last-known-good value) instead of
    raising, so that a mid-edit file cannot wipe out security-critical
    overrides in a long-running process. That is right for its own callers and
    wrong for backend selection: the degraded config reports
    ``state_backend: sqlite``, which is indistinguishable from an operator who
    genuinely chose SQLite. If the file was the only place Postgres was
    selected, the process silently opens the wrong physical store.

    An ABSENT or EMPTY file legitimately means "no selection" -> SQLite.
    An EXISTING file that cannot be read or parsed cannot safely mean anything,
    so fail closed and let the operator fix it.
    """
    from hermes_cli.config_readers import read_user_config_for_authority
    from hermes_constants import get_hermes_home

    config_path = get_hermes_home() / "config.yaml"
    if not config_path.is_file():
        return {}  # absent is a legitimate "no selection"

    # read_user_config_for_authority() is the strict reader: it preserves root
    # shape, so an existing-but-structurally-unusable file is distinguishable
    # from a genuinely empty one. load_config() flattens both to defaults, and
    # read_user_config_raw() flattens both to {} — either would let a file the
    # operator clearly wrote something into be read as "chose SQLite".
    try:
        authority = read_user_config_for_authority(config_path) or {}
        sessions = authority.get("sessions")
        if sessions is not None and not isinstance(sessions, dict):
            raise ValueError("sessions must be a mapping")
        return authority
    except Exception as exc:
        raise RuntimeError(
            f"config.yaml at {config_path} exists but is not usable as a "
            f"backend-selection source ({exc}); refusing to resolve the "
            f"session state backend from degraded defaults, because this file "
            f"may be the only source selecting PostgreSQL. Fix or remove it."
        ) from exc


def _prepare_readonly_postgres(
    conn: Any, schema_version: int, dsn: str
) -> None:
    """Validate and lock down a connection opened for read-only use.

    Read-only opens serve the dashboard's status/session listing, cron
    history, usage analytics, and resume lookup. They must never be able to
    change the store they are reporting on, so this does the opposite of
    :func:`init_postgres_schema`: it verifies, and it refuses.

    Three things happen here, in order:

    1. **Engine-level write prohibition.** ``default_transaction_read_only``
       is set on the session, so PostgreSQL itself rejects INSERT/UPDATE/
       DELETE/DDL with ``read_only_sql_transaction`` (25006). This is chosen
       over an adapter-side statement inspector deliberately: a parser that
       tries to classify SQL as read or write is a permanent source of
       both false negatives (a write shape it fails to recognise) and false
       positives, and it protects nothing against code that reaches the raw
       connection. The server has no such gap.

    2. **Schema must already exist.** A read-only caller is not the owner of
       this store. If the base tables are absent the correct answer is an
       error telling the operator to provision it, not silent creation of a
       schema through a path that claims to read.

    3. **Both schema ledgers must meet this build's versions.** The shared
       schema version and the PostgreSQL migration version must each be at
       least as new as this build expects. A store behind this build may be missing
       columns, tables, or indexes that the reader's own queries reference,
       so serving from it produces errors or silently wrong results. The
       reverse — a store AHEAD of this build, written by a newer Hermes — is
       deliberately allowed: the schema only ever grows, so a newer store
       still satisfies an older reader's queries, and refusing it would break
       every mixed-version deployment during a rollout.
    """
    # 1. Engine-enforced read-only. Do this FIRST so nothing below can write
    #    even if a later check is added carelessly.
    conn._read_only = True
    conn.execute("SET default_transaction_read_only = on")
    conn.commit()

    # 2. Base schema present?
    try:
        cur = conn.execute(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = current_schema() AND table_name = 'sessions'"
        )
        has_sessions = cur.fetchone() is not None
    except Exception as exc:
        raise RuntimeError(
            f"read-only PostgreSQL open could not inspect the schema at "
            f"{_redact_dsn(dsn)}: {exc}"
        ) from exc

    if not has_sessions:
        raise RuntimeError(
            f"read-only PostgreSQL open refused: no Hermes schema found at "
            f"{_redact_dsn(dsn)} (the 'sessions' table is absent). A "
            f"read-only open never provisions schema — that is the job of a "
            f"writable open. Start Hermes normally against this database "
            f"once, or run 'hermes migrate state-to-postgres', to create it."
        )

    # 3. Behind this build?
    expected = max((m.version for m in pg_schema._PG_ONLY_MIGRATIONS), default=0)
    try:
        recorded = pg_schema.postgres_migration_version(conn)
    except Exception as exc:
        raise RuntimeError(
            f"read-only PostgreSQL open could not read the migration ledger "
            f"at {_redact_dsn(dsn)}: {exc}"
        ) from exc

    if recorded < expected:
        raise RuntimeError(
            f"read-only PostgreSQL open refused: the store at "
            f"{_redact_dsn(dsn)} is at PostgreSQL migration version "
            f"{recorded}, but this build expects {expected}. A read-only "
            f"open will not migrate it. Start Hermes normally against this "
            f"database once to apply the pending migrations, then retry."
        )

    try:
        shared_recorded = pg_schema.postgres_schema_version(conn)
    except Exception as exc:
        raise RuntimeError(
            f"read-only PostgreSQL open could not read the shared schema version "
            f"at {_redact_dsn(dsn)}: {exc}"
        ) from exc

    if shared_recorded < schema_version:
        raise RuntimeError(
            f"read-only PostgreSQL open refused: the store at "
            f"{_redact_dsn(dsn)} is at shared schema version "
            f"{shared_recorded}, but this build expects {schema_version}. "
            f"A read-only open will not migrate it. Start Hermes normally "
            f"against this database once to apply the pending schema changes, "
            f"then retry."
        )


def _redact_dsn(dsn: str) -> str:
    """A DSN safe to put in an error message: no password, no query string."""
    try:
        from urllib.parse import urlsplit

        parts = urlsplit(dsn)
        if parts.scheme not in {"postgres", "postgresql"} or not parts.hostname:
            return "the configured PostgreSQL server"
        host = parts.hostname
        port = f":{parts.port}" if parts.port else ""
        db = (parts.path or "/?").lstrip("/") or "?"
        return f"{host}{port}/{db}"
    except Exception:
        return "the configured PostgreSQL server"


def maybe_open_postgres(
    read_only: bool,
    schema_version: int,
    config: Optional[Dict[str, Any]] = None,
    dsn_override: Optional[str] = None,
) -> Optional[_PostgresConnection]:
    """Open the PostgreSQL backend if it is configured, else return None.

    Returns None (so the caller proceeds with SQLite) when the configured
    backend is not "postgres".

    ``read_only`` does NOT select the backend — it describes how the caller
    intends to use the store, not which physical store owns the data. Gating
    the backend on it sent every dashboard/helper reader to the local
    ``state.db`` while the live write path was on PostgreSQL, which is the
    dual-truth split this backend exists to prevent.

    It DOES, however, change what this function is permitted to do:

      * **No DDL.** A read-only open never provisions or reconciles schema.
        Schema is owned by writable opens. A status/resume/analytics reader
        must not be able to create tables or apply migrations through a path
        presented as read-only.
      * **Fail closed on an unusable store.** If the schema is absent, or its
        recorded migration version is older than this build expects, the open
        raises instead of mutating it or silently serving a store it cannot
        correctly read.
      * **Enforced write prohibition.** The returned handle has
        ``default_transaction_read_only`` set on the session, so the SERVER
        rejects writes. This is an engine-level guarantee, not a convention
        the caller is trusted to honour.

    ``dsn_override`` pins the target explicitly, bypassing env/config
    resolution entirely. The backend-aware profile seam uses it so opening
    another profile's store never mutates process-global state that a
    concurrent ``SessionDB()`` on another thread could observe.

    Raises if the DSN is missing or psycopg is absent.
    """
    dsn = dsn_override or resolve_postgres_dsn(config)
    if not dsn:
        return None
    conn = connect_postgres(dsn)
    try:
        if read_only:
            _prepare_readonly_postgres(conn, schema_version, dsn)
        else:
            pg_schema.init_postgres_schema(conn, schema_version)
        return conn
    except BaseException:
        conn.close()
        raise


def _home_sessions_config(profile_home: Any) -> Dict[str, Any]:
    """Read a target home's selector without inheriting the active profile's defaults."""
    from pathlib import Path
    from hermes_cli.config_readers import read_user_config_for_authority

    config_path = Path(profile_home) / "config.yaml"
    try:
        config = read_user_config_for_authority(config_path) or {}
        sessions = config.get("sessions")
        if sessions is None:
            return {}
        if not isinstance(sessions, dict):
            raise ValueError("sessions must be a mapping")
        return sessions
    except Exception as exc:
        raise RuntimeError(
            f"config.yaml at {config_path} is not usable as a backend-selection source; "
            "refusing to assume the SQLite backend. Fix or remove it."
        ) from exc


def home_selects_postgres(profile_home: Any) -> bool:
    """Resolve the backend of an explicit home, independent of process profile context."""
    sessions = _home_sessions_config(profile_home)
    backend = str(sessions.get("state_backend") or "sqlite").strip().lower()
    if backend not in {"sqlite", "postgres", "postgresql", "pg"}:
        raise RuntimeError(f"Unknown sessions.state_backend: {backend!r}")
    return backend in {"postgres", "postgresql", "pg"}


def open_store_for_home(profile_home: Any, read_only: bool = False) -> Any:
    """Open an explicit home's configured store without changing process-wide state."""
    from pathlib import Path
    from hermes_state import SessionDB

    home = Path(profile_home)
    sessions = _home_sessions_config(home)
    backend = str(sessions.get("state_backend") or "sqlite").strip().lower()
    if backend == "sqlite":
        return SessionDB(db_path=home / "state.db", read_only=read_only)
    if backend not in {"postgres", "postgresql", "pg"}:
        raise RuntimeError(f"Unknown sessions.state_backend: {backend!r}")
    dsn = _dsn_from_profile_env(home) or (sessions.get("postgres_dsn") or "").strip()
    if not dsn:
        raise RuntimeError(
            f"profile at {home} has sessions.state_backend = 'postgres' but no DSN was found "
            "in its .env (HERMES_STATE_DATABASE_URL / HERMES_STATE_POSTGRES_DSN) "
            "or sessions.postgres_dsn; cannot open the store for that profile"
        )
    db = SessionDB(db_path=home / "state.db", read_only=read_only, postgres_dsn=dsn)
    if not db._is_postgres:
        db.close()
        raise RuntimeError("Postgres was selected but the store opened on SQLite")
    return db


def open_store_for_profile(profile_name: str, read_only: bool = False) -> Any:
    """Open a named profile's own store, with the requested access mode enforced."""
    from hermes_cli import profiles as profiles_mod

    canon = profiles_mod.normalize_profile_name(profile_name)
    profiles_mod.validate_profile_name(canon)
    if not profiles_mod.profile_exists(canon):
        raise ValueError(f"profile '{canon}' does not exist")
    return open_store_for_home(profiles_mod.get_profile_dir(canon), read_only=read_only)


def profile_selects_postgres(profile_name: str) -> bool:
    """Resolve a named profile's backend before callers choose their SQLite open path."""
    from hermes_cli import profiles as profiles_mod

    canon = profiles_mod.normalize_profile_name(profile_name)
    if not profiles_mod.profile_exists(canon):
        return False
    return home_selects_postgres(profiles_mod.get_profile_dir(canon))


def is_postgres_retryable(exc: BaseException) -> bool:
    """True if a PostgreSQL exception is a transient serialization/deadlock that
    warrants the jittered write retry (the PG analogue of SQLite "locked").

    Identified by SQLSTATE class 40 (transaction rollback: 40001 serialization
    failure, 40P01 deadlock detected) when psycopg exposes ``sqlstate``, with a
    message-substring fallback. Returns False for any non-PostgreSQL exception
    so the caller re-raises unchanged.
    """
    sqlstate = getattr(exc, "sqlstate", None)
    if sqlstate in ("40001", "40P01"):
        return True
    msg = str(exc).lower()
    return "serialization failure" in msg or "deadlock detected" in msg


def acquire_compression_lock_sql(
    conn: Any, session_id: str, holder: str, now: float, expires_at: float
) -> Tuple[bool, Optional[str]]:
    """Acquire the per-session compression lock under a PostgreSQL transaction.

    ``conn`` is the adapter connection inside an active transaction (the caller's
    ``_execute_write`` has already issued BEGIN). Because PostgreSQL's default
    isolation does not take a whole-database write lock the way SQLite's
    ``BEGIN IMMEDIATE`` does, we first take a transaction-scoped advisory lock
    keyed on the session id. That advisory lock serializes the
    delete-expired / insert / confirm sequence per session and auto-releases at
    transaction end, so a crashed acquirer cannot leak it.

    Returns ``(acquired, reclaimed_holder)`` — the same 2-tuple the SQLite
    branch of :meth:`SessionDB.try_acquire_compression_lock` returns, because
    the caller unpacks ``_execute_write(_do)`` into two names for either
    backend. Returning a bare ``bool`` here raised ``TypeError`` in that
    unpack *after* the INSERT below had committed, and the caller's fail-open
    arm swallowed it — so every acquire leaked a lock row and Postgres-backed
    sessions could never compress.

    ``reclaimed_holder`` is the holder of a lock this call cleared, or
    ``None``; it drives the caller's "Reclaimed stale compression lock"
    warning, the operator's only signal that a holder died without releasing.

    Deliberate asymmetry with SQLite — reclaims on TTL expiry ONLY, never via
    ``_compression_lock_holder_process_is_dead()``. That probe asks the LOCAL
    kernel about ``pid=<n>`` from the holder string: sound for SQLite (one
    host, one PID namespace), unsound here. A Postgres state database is
    shared across hosts, holder ids carry no host identifier
    (``pid:tid:agent:nonce`` — see
    ``agent/conversation_compression.py::_compression_lock_holder``), and
    rolling deploys routinely run two containers with low, colliding PIDs
    against one database. A local probe would call a live peer's lock dead and
    let two compressors rotate one session, splitting its lineage — precisely
    what this lock prevents. Cost: a holder killed mid-compression stalls that
    session for at most ``ttl_seconds`` instead of being reclaimed at once.
    """
    conn.execute("SELECT pg_advisory_xact_lock(hashtext(?))", (session_id,))
    # Read the expired holder before deleting it so the caller can report what
    # it reclaimed. DELETE ... RETURNING is avoided: the SQL is translated for
    # both drivers and the adapter cursor only guarantees fetchone/fetchall.
    expired = conn.execute(
        "SELECT holder FROM compression_locks "
        "WHERE session_id = ? AND expires_at < ?",
        (session_id, now),
    ).fetchone()
    reclaimed_holder = expired["holder"] if expired is not None else None
    conn.execute(
        "DELETE FROM compression_locks WHERE session_id = ? AND expires_at < ?",
        (session_id, now),
    )
    conn.execute(
        "INSERT OR IGNORE INTO compression_locks "
        "(session_id, holder, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
        (session_id, holder, now, expires_at),
    )
    row = conn.execute(
        "SELECT holder FROM compression_locks WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    acquired = row is not None and row["holder"] == holder
    # Claim the reclaim only if we also won the row: if another writer raced in
    # between our DELETE and INSERT, their lock is not ours to report.
    return acquired, (reclaimed_holder if acquired else None)


# ---------------------------------------------------------------------------
# ILIKE search for the PostgreSQL backend (GIN-accelerated)
# ---------------------------------------------------------------------------
#
# PostgreSQL has no SQLite FTS5 equivalent. Search uses case-insensitive
# ILIKE over the same text columns the FTS5 index covers (content, tool_name,
# tool_calls). When pg_trgm is installed and the GIN indexes created by
# migrations v17/v18 are present, PostgreSQL uses them automatically —
# no query-syntax change needed. Without the indexes the query falls back
# to a sequential scan (slower but functionally identical).
# Queries shorter than 3 characters bypass the trigram index (fundamental
# trigram minimum); those fall back to seq scan as well.
#
# The result-row contract matches the SQLite search_messages
# path exactly (id/session_id/role/snippet/timestamp/tool_name/source/model/
# session_started + a bounded before/anchor/after ``context`` window, with the
# full ``content`` removed), so callers are backend-agnostic.

_SEARCH_CONTEXT_SQL = """
    WITH target AS (
        SELECT session_id, timestamp, id FROM messages WHERE id = ?
    )
    SELECT role, content FROM (
        SELECT m.id, m.timestamp, m.role, m.content
        FROM messages m JOIN target t ON t.session_id = m.session_id
        WHERE (m.timestamp < t.timestamp)
           OR (m.timestamp = t.timestamp AND m.id < t.id)
        ORDER BY m.timestamp DESC, m.id DESC LIMIT 1
    ) AS before_row
    UNION ALL
    SELECT role, content FROM messages WHERE id = ?
    UNION ALL
    SELECT role, content FROM (
        SELECT m.id, m.timestamp, m.role, m.content
        FROM messages m JOIN target t ON t.session_id = m.session_id
        WHERE (m.timestamp > t.timestamp)
           OR (m.timestamp = t.timestamp AND m.id > t.id)
        ORDER BY m.timestamp ASC, m.id ASC LIMIT 1
    ) AS after_row
"""


def _content_preview(decode_content, raw: Any) -> str:
    decoded = decode_content(raw)
    if isinstance(decoded, list):
        parts = [
            p.get("text", "") for p in decoded
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return " ".join(t for t in parts if t).strip() or "[multimodal content]"
    if isinstance(decoded, str):
        return decoded
    return ""


def search_messages_postgres(
    conn: Any,
    decode_content,
    query: str,
    source_filter: Optional[List[str]] = None,
    exclude_sources: Optional[List[str]] = None,
    role_filter: Optional[List[str]] = None,
    limit: int = 20,
    offset: int = 0,
    sort: Optional[str] = None,
    include_inactive: bool = False,
    include_context: bool = True,
) -> List[Dict[str, Any]]:
    """Tokenized FTS search for the PostgreSQL backend (SQLite FTS5 parity).

    When the v19 migration has run (``fts_content tsvector`` column present),
    uses ``fts_content @@ tsquery`` with ``'simple'`` dictionary — tokenized
    AND-search, word-order-independent, matching code identifiers and proper
    nouns without stemming. Multi-word queries behave like SQLite FTS5:
    "docker deployment" matches any message containing both words anywhere.

    Falls back to ILIKE substring search when:
    - the v19 column is absent (migration pending), or
    - the query produces an empty tsquery (pure punctuation / no tokens).

    The result-row contract is identical to the SQLite ``search_messages``
    path: id/session_id/role/snippet/timestamp/tool_name/source/model/
    session_started + a bounded before/anchor/after ``context`` window.
    """
    if not query or not query.strip():
        return []

    sort_norm = sort.strip().lower() if isinstance(sort, str) else None

    # ── FTS path (v19 column present AND backfill complete) ───────────────────
    # While the column exists but legacy rows are still fts_content IS NULL,
    # an FTS-only query would silently miss every un-backfilled row. To honor
    # the "old rows fall through to ILIKE until backfill" contract, we only take
    # the FTS path once no NULL rows remain. ILIKE covers all rows regardless,
    # so it is the correct choice during the backfill window.
    tsq = _build_tsquery(conn, query)
    if tsq is not None and _fts_backfill_complete(conn):
        try:
            matches = _search_messages_fts(
                conn, decode_content, tsq,
                source_filter=source_filter,
                exclude_sources=exclude_sources,
                role_filter=role_filter,
                limit=limit,
                offset=offset,
                sort_norm=sort_norm,
                include_inactive=include_inactive,
            )
            if include_context:
                _attach_context(conn, decode_content, matches)
            return matches
        except Exception:
            logger.warning(
                "FTS search failed, falling back to ILIKE", exc_info=True
            )

    # ── ILIKE fallback (column absent, backfill incomplete, empty tsquery) ────
    return _search_messages_ilike(
        conn, decode_content, query,
        source_filter=source_filter,
        exclude_sources=exclude_sources,
        role_filter=role_filter,
        limit=limit,
        offset=offset,
        sort_norm=sort_norm,
        include_inactive=include_inactive,
        include_context=include_context,
    )


def _build_where(source_filter, exclude_sources, role_filter, include_inactive, params):
    """Use the common visibility contract for both PostgreSQL search paths."""
    from hermes_state_search import _search_filter_clauses

    where = []
    _search_filter_clauses(
        where, params, include_inactive=include_inactive, source_filter=source_filter,
        exclude_sources=exclude_sources, role_filter=role_filter)
    return where


def _search_messages_fts(
    conn: Any,
    decode_content,
    tsq: str,
    source_filter, exclude_sources, role_filter,
    limit, offset, sort_norm, include_inactive,
) -> List[Dict[str, Any]]:
    """Execute the tsvector FTS search path. Returns matches without context."""
    params: list = [tsq]  # for fts_content @@ %s::tsquery
    where = ["m.fts_content @@ %s::tsquery"]
    where += _build_where(
        source_filter, exclude_sources, role_filter, include_inactive, params
    )

    if sort_norm == "oldest":
        order_by = "ORDER BY m.timestamp ASC, ts_rank(m.fts_content, %s::tsquery) DESC"
    elif sort_norm == "newest":
        order_by = "ORDER BY m.timestamp DESC, ts_rank(m.fts_content, %s::tsquery) DESC"
    else:
        order_by = "ORDER BY ts_rank(m.fts_content, %s::tsquery) DESC, m.timestamp DESC"
    params.append(tsq)  # for ts_rank
    params.extend([limit, offset])

    rows = conn.execute(
        f"""
        SELECT m.id, m.session_id, m.role, m.content, m.timestamp, m.tool_name,
               s.source, s.model, s.started_at AS session_started
        FROM messages m
        JOIN sessions s ON s.id = m.session_id
        WHERE {' AND '.join(where)}
        {order_by}
        LIMIT ? OFFSET ?
        """,
        params,
    ).fetchall()

    matches = []
    for r in rows:
        preview = _content_preview(decode_content, r["content"])
        matches.append({
            "id": r["id"],
            "session_id": r["session_id"],
            "role": r["role"],
            "snippet": preview[:200],
            "timestamp": r["timestamp"],
            "tool_name": r["tool_name"],
            "source": r["source"],
            "model": r["model"],
            "session_started": r["session_started"],
        })
    return matches


def _search_messages_ilike(
    conn: Any,
    decode_content,
    query: str,
    source_filter, exclude_sources, role_filter,
    limit, offset, sort_norm, include_inactive,
    include_context=True,
) -> List[Dict[str, Any]]:
    """Execute the ILIKE fallback search path. Returns matches with context."""
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    like = f"%{escaped}%"

    params: list = [like, like, like]
    where = [
        "(m.content ILIKE ? ESCAPE '\\' OR m.tool_name ILIKE ? ESCAPE '\\' "
        "OR m.tool_calls ILIKE ? ESCAPE '\\')"
    ]
    where += _build_where(
        source_filter, exclude_sources, role_filter, include_inactive, params
    )

    order_by = "ORDER BY m.timestamp ASC" if sort_norm == "oldest" else "ORDER BY m.timestamp DESC"
    params.extend([limit, offset])

    rows = conn.execute(
        f"""
        SELECT m.id, m.session_id, m.role, m.content, m.timestamp, m.tool_name,
               s.source, s.model, s.started_at AS session_started
        FROM messages m
        JOIN sessions s ON s.id = m.session_id
        WHERE {' AND '.join(where)}
        {order_by}
        LIMIT ? OFFSET ?
        """,
        params,
    ).fetchall()

    matches = []
    for r in rows:
        preview = _content_preview(decode_content, r["content"])
        matches.append({
            "id": r["id"],
            "session_id": r["session_id"],
            "role": r["role"],
            "snippet": preview[:200],
            "timestamp": r["timestamp"],
            "tool_name": r["tool_name"],
            "source": r["source"],
            "model": r["model"],
            "session_started": r["session_started"],
        })

    if include_context:
        _attach_context(conn, decode_content, matches)
    return matches


def _attach_context(conn: Any, decode_content, matches: List[Dict[str, Any]]) -> None:
    """Add bounded before/anchor/after context window to each match in-place."""
    for match in matches:
        try:
            ctx_rows = conn.execute(
                _SEARCH_CONTEXT_SQL, (match["id"], match["id"])
            ).fetchall()
            match["context"] = [
                {"role": r["role"],
                 "content": _content_preview(decode_content, r["content"])[:200]}
                for r in ctx_rows
            ]
        except Exception:
            match["context"] = []
