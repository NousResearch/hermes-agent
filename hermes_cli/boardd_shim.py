#!/usr/bin/env python3
"""boardd_shim — routes hermes_cli.kanban_db through the boardd broker when
HERMES_KANBAN_BROKER=1.

This is the REAL mechanism the runbook's flag refers to (the earlier draft
claimed a flag that did not exist — BLOCKER-1). It is wired in by a minimal
frozen patch that appends, at the END of kanban_db.py, a block that rebinds the
module's public functions to the delegates below (see
`kanban_db.broker-shim.patch.diff`). Rebinding at end-of-module (not a runtime
monkeypatch of module attributes) is what makes `from hermes_cli.kanban_db
import add_comment` callers ALSO get the broker path: the end block runs during
kanban_db's own import, before any dependent module finishes `from … import …`.

Coverage (delegated to boardd, exactly-once via op_id):
  connect / connect_closing (→ BrokerConnection) and the core op functions
  create_task, add_comment, heartbeat_worker, set_workspace_path, set_branch_name.
  claim_task deliberately remains the real kanban_db implementation over a
  BrokerConnection: continuation grants and operator overrides require its full
  policy/CAS transaction and must never be reduced to boardd's legacy native
  claim payload. Lifecycle hooks fire CLIENT-SIDE (off the broker DB thread).

Anything NOT yet covered fails LOUDLY (NotImplementedError) instead of silently
corrupting: a raw multi-statement write_txn on a BrokerConnection, create_task
with parent links, etc. The CI gate + this loudness are what let cutover proceed
only when the necessary condition is actually met.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sqlite3

try:  # production: kb_client is installed as a hermes_cli submodule
    from hermes_cli import kb_client
except Exception:  # test/standalone: kb_client on sys.path
    import kb_client  # type: ignore


def enabled() -> bool:
    return os.environ.get("HERMES_KANBAN_BROKER") == "1"


# --------------------------------------------------------------------------- #
# Exception-type translation across the shim boundary (rev6 #1).
# BrokerConnection impersonates a sqlite3.Connection, so it MUST raise the SAME
# sqlite3.* exception TYPES the real connection would — otherwise the unchanged
# `except sqlite3.IntegrityError` / `except sqlite3.OperationalError` sites in
# kanban_db.py (e.g. create_task's id-collision retry) silently stop matching.
# The broker tags every SQL error with etype = type(exc).__name__, so we map it
# straight back to the sqlite3 class. Broker-internal control errors (TxnStale
# from a 2s-cap rollback, TxnBusy, Forbidden, DiskGuardError, BadRequest) become
# sqlite3.OperationalError so they PROPAGATE through write_txn's generic
# except/rollback and the non-busy boundary-retry (never swallowed-as-success).
# --------------------------------------------------------------------------- #
_SQLITE_EXC = {
    "IntegrityError": sqlite3.IntegrityError,
    "OperationalError": sqlite3.OperationalError,
    "DatabaseError": sqlite3.DatabaseError,
    "ProgrammingError": sqlite3.ProgrammingError,
    "InterfaceError": sqlite3.InterfaceError,
    "DataError": sqlite3.DataError,
    "NotSupportedError": sqlite3.NotSupportedError,
    "InternalError": sqlite3.InternalError,
}


def _to_sqlite_exc(err):
    """Translate a kb_client.BoarddError into the sqlite3 exception the real
    connection would raise for the same failure."""
    etype = getattr(err, "etype", None)
    cls = _SQLITE_EXC.get(etype)
    if cls is not None:
        return cls(str(err))
    # DiskGuardError -> a disk-full-shaped OperationalError so callers fail
    # closed (and it reads clearly in logs); other control errors -> generic
    # OperationalError (non-busy -> propagates, not retried).
    if etype == "DiskGuardError":
        return sqlite3.OperationalError(f"database or disk is full: {err}")
    return sqlite3.OperationalError(f"{etype or 'BrokerError'}: {err}")


class BoarddUnavailableError(sqlite3.OperationalError):
    """Raised by the shim when the boardd broker is unreachable.

    Inherits from :class:`sqlite3.OperationalError` so existing
    ``except sqlite3.OperationalError`` sites react to broker loss the same
    way they react to a local SQLite lock.  Callers that need to distinguish
    broker unavailability can catch this subclass directly.
    """
    pass


def _x(fn, *args, **kwargs):
    """Call a kb_client method, translating broker errors into sqlite3.* so
    every ``except sqlite3.*`` site behaves identically shim-vs-native."""
    try:
        return fn(*args, **kwargs)
    except kb_client.BoarddError as e:
        raise _to_sqlite_exc(e) from None
    except kb_client.BoarddUnavailable as e:
        raise BoarddUnavailableError(f"boardd unavailable: {e}") from None


# --------------------------------------------------------------------------- #
# Coverage instrumentation (rev6 #6). When BOARDD_SHIM_COVERAGE_LOG is set, every
# proxied op records "<op> <caller-file>:<line> <func>" so a maintenance-window
# smoke can prove it exercised 100% of the gateway's write call sites (compare
# the log against a static enumeration; abort if any site was never proxied).
# Zero overhead when the env var is unset.
# --------------------------------------------------------------------------- #
_COV_PATH = os.environ.get("BOARDD_SHIM_COVERAGE_LOG")


def _cov(op):
    if not _COV_PATH:
        return
    import traceback
    site = "?"
    for fr in reversed(traceback.extract_stack()[:-1]):
        if "boardd_shim" in fr.filename or "kb_client" in fr.filename:
            continue
        site = f"{os.path.basename(fr.filename)}:{fr.lineno}:{fr.name}"
        break
    try:
        with open(_COV_PATH, "a", encoding="utf-8") as fh:
            fh.write(f"{op}\t{site}\n")
    except Exception:
        pass


def _require_original(op, original):
    """Return a captured local implementation or fail closed on bad wiring."""
    if original is None:
        raise RuntimeError(
            f"boardd_shim.{op} pass-through: original function was not captured "
            "(install_rebind / end-block rebind did not run)."
        )
    return original


def noop_flen(conn):
    """Connection-gated replacement for ``_check_file_length_invariant``.

    The original runs at the END of write_txn and reads the raw db FILE
    (os.path.getsize + header). On a client-via-broker connection that file read
    is DECOUPLED from the broker's connection view: under WAL the main file can
    legitimately lag the (WAL-aware) header page_count, yielding a false
    "torn-extend". The authoritative check therefore runs on the broker's own
    connection. A pass-through sqlite connection still owns its local file and
    MUST retain the original invariant check.
    """
    if isinstance(conn, BrokerConnection):
        return None
    return _require_original(
        "noop_flen", _ORIG_CHECK_FILE_LENGTH_INVARIANT
    )(conn)


def _c():
    return kb_client.get_client()  # thread-local (M-9)


# --------------------------------------------------------------------------- #
# FLAG-AND-FLEET routing gate (rev7 re-scope — council GO-FLEET-ONLY).
#
# The broker exists to serialize writers to the ONE live+corrupting board: fleet
# (~/.hermes/kanban/boards/fleet/kanban.db, the board `current` points at, which
# max-parallel's 14 workers write). Per-seat boards are archived legacy and the
# `default` board is dead; a raw sqlite open of EITHER cannot corrupt fleet, so
# routing them through the single writer is pure downside (couples unrelated
# boards to boardd's availability). connect()/connect_closing() therefore route
# to the broker ONLY when the requested open RESOLVES TO THE FLEET DB, and pass
# every other board straight through to the ORIGINAL local sqlite connect.
#
# Correctness rules (a drifting copy of the resolver = split-brain routing):
#   * The requested path is resolved with kanban_db's OWN resolver, exactly as
#     the real connect() does — explicit db_path used as-is, else
#     kanban_db_path(board=…) honoring HERMES_KANBAN_DB → HERMES_KANBAN_BOARD →
#     <root>/kanban/current → default. Never a hardcoded string, never a reimpl.
#   * The fleet IDENTITY is anchored on board_dir("fleet")/kanban.db (kanban_db's
#     own board_dir → boards_root → kanban_home), NOT kanban_db_path("fleet"):
#     a stray HERMES_KANBAN_DB override must never be able to REDEFINE what fleet
#     is (it may only redirect a REQUEST, which is still compared to this fixed
#     identity).
#   * Both sides are realpath'd (symlinks are live on this box) with a
#     not-yet-created leaf tolerated, so the first connect() that creates the
#     file still routes correctly.
# --------------------------------------------------------------------------- #

# Captured BEFORE kanban_db's public names are pointed at this shim, so every
# non-fleet pass-through uses the GENUINE implementation, never a
# reimplementation. Set by install_rebind() (tests/tools) and the end-of-module
# rebind block in kanban_db.py (production). _KDB is the exact module whose
# functions were captured (guards against two live copies of kanban_db resolving
# differently).
_KDB = None
_ORIG_CONNECT = None
_ORIG_CONNECT_CLOSING = None
_ORIG_ADD_COMMENT = None
_ORIG_HEARTBEAT_WORKER = None
_ORIG_SET_WORKSPACE_PATH = None
_ORIG_SET_BRANCH_NAME = None
_ORIG_CHECK_FILE_LENGTH_INVARIANT = None

_route_log = logging.getLogger("boardd_shim")


def _capture_original(kdb):
    """Capture one module's genuine functions exactly once.

    Re-installing the shim on the same module is intentionally idempotent: its
    public names already point at this module, so capturing them again would
    make each pass-through delegate recurse into itself. A different module is
    rejected rather than replacing the originals used by the active resolver.
    """
    global _KDB, _ORIG_CONNECT, _ORIG_CONNECT_CLOSING
    global _ORIG_ADD_COMMENT, _ORIG_HEARTBEAT_WORKER
    global _ORIG_SET_WORKSPACE_PATH, _ORIG_SET_BRANCH_NAME
    global _ORIG_CHECK_FILE_LENGTH_INVARIANT
    if _KDB is kdb:
        return
    if _KDB is not None:
        raise RuntimeError(
            "boardd_shim originals are already captured from a different "
            "kanban_db module"
        )

    # Resolve every attribute before publishing _KDB so a malformed module
    # cannot leave a partially initialized capture that later looks complete.
    originals = (
        kdb.connect,
        kdb.connect_closing,
        kdb.add_comment,
        kdb.heartbeat_worker,
        kdb.set_workspace_path,
        kdb.set_branch_name,
        kdb._check_file_length_invariant,
    )
    (
        _ORIG_CONNECT,
        _ORIG_CONNECT_CLOSING,
        _ORIG_ADD_COMMENT,
        _ORIG_HEARTBEAT_WORKER,
        _ORIG_SET_WORKSPACE_PATH,
        _ORIG_SET_BRANCH_NAME,
        _ORIG_CHECK_FILE_LENGTH_INVARIANT,
    ) = originals
    _KDB = kdb


def _resolver():
    """The kanban_db module to resolve paths with — the captured one if present,
    else a best-effort import (production always captures; this is the standalone
    fallback)."""
    if _KDB is not None:
        return _KDB
    try:
        import hermes_cli.kanban_db as kdb  # production layout
    except Exception:  # pragma: no cover - test/standalone layout
        import kanban_db as kdb  # type: ignore
    return kdb


def _rp(p) -> str:
    """realpath that tolerates a not-yet-created leaf. os.path.realpath resolves
    every EXISTING symlink component and leaves a nonexistent final component
    as-is — exactly right for a DB file connect() is about to create."""
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(p))))


def _fleet_db_realpath(kdb) -> str:
    """Canonical fleet-board DB realpath via kanban_db's own board resolver."""
    return _rp(kdb.board_dir("fleet") / "kanban.db")


def _requested_db_realpath(kdb, db_path, board) -> str:
    """Realpath of the file THIS connect() would open, resolved via kanban_db's
    own resolver exactly as the real connect() does (explicit db_path as-is, else
    kanban_db_path honoring the env → current-file chain)."""
    req = db_path if db_path is not None else kdb.kanban_db_path(board=board)
    return _rp(req)


def routes_to_fleet(db_path=None, board=None):
    """Return (is_fleet, requested_realpath, fleet_realpath). Pure/side-effect
    free so unit tests can assert the decision directly."""
    kdb = _resolver()
    req = _requested_db_realpath(kdb, db_path, board)
    fleet = _fleet_db_realpath(kdb)
    return (req == fleet, req, fleet)


class _Row(dict):
    """dict row that ALSO supports positional access (row[0]) + .keys(), like
    sqlite3.Row — the broker returns column-ordered dicts (dict(sqlite3.Row)),
    so positional access via values() matches column order. kanban_db mixes
    named (row["status"]) and positional (PRAGMA database_list row[1]) access."""
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return dict.__getitem__(self, key)


class _Cursor:
    """Minimal DB-API cursor over broker results (dict/positional rows)."""
    def __init__(self, rows=None, rowcount=-1, lastrowid=None):
        self._rows = [r if isinstance(r, _Row) else _Row(r) for r in (rows or [])]
        self.rowcount = rowcount
        self.lastrowid = lastrowid

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def fetchmany(self, n=1):
        out, self._rows = self._rows[:n], self._rows[n:]
        return out

    def __iter__(self):
        return iter(self._rows)


class BrokerConnection:
    """Drop-in for the sqlite3.Connection returned by kanban_db.connect(),
    routing ALL SQL to boardd. Supports INTERACTIVE TRANSACTIONS so the REAL
    kanban_db `write_txn` functions (complete_task, block_task, promote_task,
    unblock_task, reclaim_task, recompute_ready, decompose, …) run UNCHANGED and
    atomically through the single writer — mid-transaction reads branch normally.

    - BEGIN [IMMEDIATE] → broker txn_begin (one txn at a time; the broker holds
      the sole DB thread and defers other clients until COMMIT/ROLLBACK).
    - conn.execute(read)  inside a txn → txn_exec, returns rows (so
      `if cur.rowcount != 1: return False` and `SELECT … fetchone()` work).
    - conn.execute(write) inside a txn → txn_exec, returns rowcount + real
      lastrowid.
    - COMMIT/ROLLBACK → txn_commit/txn_rollback.
    - autocommit reads → query proxy; autocommit writes → exec (real lastrowid).

    Exactly-once for the interactive path comes from the lifecycle transitions'
    own status-guard CAS (a re-run finds the new status and no-ops), NOT
    applied_ops; the exactly-once-sensitive INSERT/claim ops are delegated
    NATIVELY (add_comment/claim/… → kb_client, applied_ops). SAVEPOINT is the
    one unsupported form (unused by write_txn) and raises loudly."""
    row_factory = None

    def __init__(self):
        self._txn = None  # open broker txn token, or None (autocommit)

    def execute(self, sql, params=()):
        s = (sql or "").strip()
        first = s.split(None, 1)[0].lower() if s else ""
        _cov("sql:" + first)
        if first == "begin":                 # BEGIN / BEGIN IMMEDIATE / DEFERRED
            if self._txn is not None:
                raise sqlite3.OperationalError(
                    "cannot start a transaction within a transaction")
            self._txn = _x(_c().txn_begin)
            return _Cursor()
        if first in ("commit", "end"):
            if self._txn is not None:
                tok, self._txn = self._txn, None
                _x(_c().txn_commit, tok)
            return _Cursor()
        if first == "rollback":
            if self._txn is not None:
                tok, self._txn = self._txn, None
                _x(_c().txn_rollback, tok)
            return _Cursor()
        if first in ("savepoint", "release"):
            raise NotImplementedError(
                "BrokerConnection: SAVEPOINT/RELEASE unsupported (unused by "
                "write_txn). Route this call site through an op function.")
        if self._txn is not None:            # statement inside an open txn
            r = _x(_c().txn_exec, self._txn, sql, list(params))
            return _Cursor(rows=r.get("rows") or [], rowcount=r.get("rowcount", -1),
                           lastrowid=r.get("lastrowid"))
        # autocommit
        if first in ("select", "with", "explain", "values", "pragma"):
            return _Cursor(rows=_x(_c().query, sql, list(params)))
        if first in ("update", "insert", "delete"):
            r = _x(_c().exec_write, sql, list(params))
            return _Cursor(rowcount=r.get("rowcount", -1),
                           lastrowid=r.get("lastrowid"))
        raise NotImplementedError(f"BrokerConnection: unsupported SQL {first!r}")

    def executemany(self, sql, seq):
        last = _Cursor()
        for p in seq:
            last = self.execute(sql, p)
        return last

    def cursor(self):
        return self

    def commit(self):
        if self._txn is not None:
            tok, self._txn = self._txn, None
            _x(_c().txn_commit, tok)

    def rollback(self):
        if self._txn is not None:
            tok, self._txn = self._txn, None
            _x(_c().txn_rollback, tok)

    def close(self):
        # a dangling open txn (caller leaked the connection) — roll back.
        if self._txn is not None:
            try:
                _c().txn_rollback(self._txn)
            except Exception:
                pass
            self._txn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *a):
        # sqlite3 Connection context manager commits on success, rolls back on
        # exception. Mirror that for callers using `with conn:`.
        if self._txn is not None:
            tok, self._txn = self._txn, None
            if exc_type is None:
                _x(_c().txn_commit, tok)
            else:
                try:
                    _x(_c().txn_rollback, tok)
                except Exception:
                    pass
        return False


# --------------------------------------------------------------------------- #
# Delegates the patched kanban_db functions call. Signatures mirror the live
# module so callers are unchanged.
# --------------------------------------------------------------------------- #
def connect(db_path=None, *, board=None):
    """FLAG-AND-FLEET-GATED connect.

    Routes to boardd only when process custody is enabled and the requested open
    resolves to the fleet DB. Every other request passes through to the original
    SQLite connect. The signature mirrors the real connect() surface.
    """
    is_fleet, req, fleet = routes_to_fleet(db_path=db_path, board=board)
    if enabled() and is_fleet:
        _route_log.info("boardd route=BROKER db=%s (fleet=%s)", req, fleet)
        return BrokerConnection()
    orig = _ORIG_CONNECT
    if orig is None:
        # Never silently reimplement connect(): a missing original means the
        # rebind wiring is broken. Fail loud rather than split-brain.
        raise RuntimeError(
            "boardd_shim.connect pass-through: original kanban_db.connect was "
            "not captured (install_rebind / end-block rebind did not run).")
    _route_log.info("boardd route=PASSTHROUGH db=%s (fleet=%s)", req, fleet)
    return orig(db_path=db_path, board=board)


@contextlib.contextmanager
def connect_closing(db_path=None, *, board=None):
    """FLAG-AND-FLEET-GATED connect_closing. Delegates to the gated connect()
    (broker for fleet, real sqlite for everything else) and guarantees close on
    exit — identical close semantics to the real connect_closing (BrokerConnection
    and sqlite3.Connection both honor .close())."""
    conn = connect(db_path=db_path, board=board)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


def add_comment(conn, task_id, author, body):
    if not isinstance(conn, BrokerConnection):
        return _require_original("add_comment", _ORIG_ADD_COMMENT)(
            conn, task_id, author, body
        )
    _cov("add_comment")
    return _c().add_comment(task_id, author, body)["comment_id"]


def set_workspace_path(conn, task_id, path):
    if not isinstance(conn, BrokerConnection):
        return _require_original(
            "set_workspace_path", _ORIG_SET_WORKSPACE_PATH
        )(conn, task_id, path)
    _cov("set_workspace_path")
    _c().set_workspace_path(task_id, str(path))
    return None


def set_branch_name(conn, task_id, branch_name):
    if not isinstance(conn, BrokerConnection):
        return _require_original(
            "set_branch_name", _ORIG_SET_BRANCH_NAME
        )(conn, task_id, branch_name)
    _cov("set_branch_name")
    _c().set_branch_name(task_id, str(branch_name))
    return None


def heartbeat_worker(conn, task_id, *, note=None, expected_run_id=None):
    if not isinstance(conn, BrokerConnection):
        return _require_original(
            "heartbeat_worker", _ORIG_HEARTBEAT_WORKER
        )(
            conn,
            task_id,
            note=note,
            expected_run_id=expected_run_id,
        )
    _cov("heartbeat_worker")
    return bool(_c().heartbeat(task_id, note=note).get("ok", False))


# NOTE: create_task and claim_task are deliberately NOT rebound. Their REAL
# implementations run through the interactive-transaction path on a
# BrokerConnection. create_task needs its full signature (parents/task_links,
# idempotency_key, tenant, skills, project linking, goal_mode, …). claim_task
# needs continuation_authorization_id, operator_override_reason, the post-lock
# active-PR recheck, one-shot consume CAS, and model-aware audit payload. The
# broker's legacy native claim endpoint cannot express those contracts; routing
# through it would silently bypass guarded continuation. Both real functions
# retain their policy and atomicity because write_txn composes over the broker's
# tokenized interactive transaction.


def claim_task(
    conn,
    task_id,
    *,
    ttl_seconds=None,
    claimer=None,
    operator_override_reason=None,
    continuation_authorization_id=None,
):
    """Fail closed if a caller tries the obsolete native claim delegate.

    Production intentionally leaves ``kanban_db.claim_task`` unrebound so its
    full continuation policy executes over ``BrokerConnection``. Keeping this
    named guard makes stale direct imports fail loudly instead of silently
    dropping authorization or override arguments through boardd's legacy
    native ``claim`` payload.
    """
    _cov("claim_task_rejected")
    raise RuntimeError(
        "boardd_shim.claim_task is disabled: use hermes_cli.kanban_db.claim_task "
        "over BrokerConnection so continuation policy is preserved"
    )


def _row_to_task(kdb, row):
    if row is None:
        return None
    T = getattr(kdb, "Task", None)
    if T is not None:
        try:
            import dataclasses
            fields = {f.name for f in dataclasses.fields(T)}
            return T(**{k: v for k, v in row.items() if k in fields})
        except Exception:
            pass
    import types
    return types.SimpleNamespace(**row)


# Names the kanban_db end-block rebinds to (kept in one place for the patch).
# create_task and claim_task are intentionally absent (both run as the real
# functions via the interactive txn with their complete policy signatures).
# Everything else not listed here (complete/block/unblock/promote/reclaim/
# schedule/decompose/…) also runs as the real function via the interactive txn
# on the BrokerConnection.
REBIND_NAMES = (
    "connect", "connect_closing", "add_comment",
    "heartbeat_worker", "set_workspace_path", "set_branch_name",
)


def install_rebind(kdb):
    """Apply the EXACT rebinds the frozen kanban_db.broker-shim.patch.diff does,
    for tests/tools that import kanban_db directly instead of relying on the
    end-of-module patch. Keeping tests on this single helper means they exercise
    the identical production surface — including
    `_check_file_length_invariant = noop_flen` (whose omission caused a WAL-lag
    torn-extend FALSE POSITIVE under concurrent load in an earlier harness)."""
    # Capture every GENUINE rebound function BEFORE repointing it, so the
    # fleet-gate's non-fleet pass-through uses the real implementation.
    _capture_original(kdb)
    kdb.connect = connect
    kdb.connect_closing = connect_closing
    kdb.add_comment = add_comment
    kdb.heartbeat_worker = heartbeat_worker
    kdb.set_workspace_path = set_workspace_path
    kdb.set_branch_name = set_branch_name
    kdb._check_file_length_invariant = noop_flen
