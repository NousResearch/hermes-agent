#!/usr/bin/env python3
"""boardd — single-writer broker daemon for the Hermes kanban board.

WHY THIS EXISTS
---------------
The fleet MIRROR board `~/.hermes/kanban/boards/fleet/kanban.db` reliably
corrupts ("database disk image is malformed", "2nd reference to page N",
"invalid page number", "btreeInitPage", "Rowid out of order"); all 360
quarantine files are this board. The DEFAULT board `~/.hermes/kanban.db`
(one uniform write path) stays CLEAN. A cross-process advisory flock that
serialized writes did NOT stop it (probe.log shows the flock strictly
serializing A/B yet corruption still occurred) — because the flock excludes
nobody: the swarm that writes the fleet board is heterogeneous and
uncoordinated.

ROOT CAUSE (2026-07-20 forensics):
  1. An UNCOORDINATED, HETEROGENEOUS multi-writer swarm on the fleet board:
     the gateway + kanban-dispatch + 14 workers (via kanban_db, venv), a 60s
     kanban-sync.py (system python), a 2min fleet-board-sync shelling many
     `hermes kanban comment/heartbeat/claim/promote` CLI writes, board-steward
     + kanban_bridge_state (which DO take .dispatch.lock — a lock the others
     ignore). Multiple independent CONNECTIONS, no shared serialization.
  2. SQLITE VERSION SKEW: system python is SQLite 3.45.1 (kanban-sync, the
     integrity-guard, some CLI) while the venv is 3.50.4 — two different SQLite
     builds writing the same WAL concurrently.
  3. A SELF-CORRUPTING RECOVERY LOOP: the integrity-guard REINDEXes and
     atomically swaps the file while the long-lived gateway keeps a STALE
     handle to the swapped-out inode -> immediate re-"malformed".

REFUTED: symlink / divergent-`-shm`. `namei` shows no symlink; the board is on
/dev/sda1 root ext4 (not the Hetzner volume); a two-arm repro (7 writers via a
symlink alias + 7 via realpath, same inode) stayed CLEAN — modern SQLite keys
the wal-index by device+inode. So canonicalizing to a realpath is good HYGIENE
but is NOT the fix. The fix is: ONE process, ONE connection, ONE SQLite build,
and the broker OWNS recovery (no external REINDEX/inode-swap). Single-writer is
clean by construction; no Postgres needed (the environment is healthy).

THE FIX (this daemon)
---------------------
boardd owns the ONE and ONLY connection to the board. Every access — reads AND
writes — routes through it over a Unix domain socket. That collapses SQLite to
its bulletproof single-process case: with a single connection there is no
cross-connection wal-index to corrupt, and the whole WAL-checkpoint race
evaporates.

INVARIANTS (converged from a 4-model council):
  * Standalone daemon (NOT embedded in the gateway) so a gateway wedge/deploy
    never stalls board writes.
  * The single sqlite3.Connection is opened INSIDE the DB thread, AFTER any
    daemonize/fork — a handle is never inherited across fork.
  * Open by a CANONICALIZED ABSOLUTE realpath (HYGIENE — ensures every path
    spelling maps to the same file; NOT the fix, since SQLite already keys the
    wal-index by device+inode). The fix is the single connection itself.
  * The broker OWNS recovery: REINDEX runs IN-PLACE on the owned connection
    (op `_reindex`); no external process REINDEXes or inode-swaps the file
    under the live handle. A file-swap restore requires a broker RESTART so the
    sole writer reopens the NEW inode (never keeps a stale handle).
  * One dedicated DB thread + an in-process FIFO queue == total ordering by
    construction. All client sockets funnel into this one thread.
  * Pragmas: WAL, synchronous=FULL, busy_timeout, foreign_keys=ON. Default
    wal_autocheckpoint (safe with one connection) + an idle
    wal_checkpoint(TRUNCATE) timer to bound the WAL.
  * Exactly-once + durability: every mutation runs BEGIN IMMEDIATE .. COMMIT
    and records applied_ops(op_id) INSIDE the same transaction; a retried
    op_id returns the STORED result instead of re-applying (no double-claim).
    The reply is sent ONLY after COMMIT returns.
  * Claims are atomic CAS: UPDATE .. WHERE id=? AND status='ready' AND
    claim_lock IS NULL — at most one writer wins.
  * HARD RULE: NO external calls (MCP / network / subprocess) on the DB
    handler path. Handlers are pure SQL on the owned connection. This is the
    discipline that stops the broker from ever becoming a 0-byte wedge.
  * Broker owns backups (VACUUM INTO on its own connection) + integrity_check
    of the COPY before rotating. Never cp/rsync a live WAL db.
  * Free-space guard refuses writes + alerts below a floor (disk-full history).

Sequence B->C: the SQL lives behind a single driver surface; a future Postgres
migration is a one-file driver swap, not a rewrite. The business schema is
byte-identical to hermes_cli.kanban_db (in production boardd obtains its
connection FROM kanban_db.connect(), guaranteeing identity). The ONLY
broker-added table is `applied_ops` (idempotency ledger); it never touches the
business tables.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import secrets
import signal
import socket
import socketserver
import sqlite3
import struct
import sys
import threading
import time

_log = logging.getLogger("boardd")

# --------------------------------------------------------------------------- #
# Defaults / configuration
# --------------------------------------------------------------------------- #
HERMES_HOME = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
DEFAULT_DB = os.environ.get(
    "BOARDD_DB",
    os.environ.get(
        "HERMES_KANBAN_DB",
        os.path.join(HERMES_HOME, "kanban", "boards", "fleet", "kanban.db"),
    ),
)
DEFAULT_SOCK = os.environ.get(
    # Dedicated-user design: the socket lives in its OWN boardd-owned dir
    # (SOCKDIR=~/.hermes/kanban/boardd-run, 0710 group boardd), NOT the 0700 DB
    # dir. See ROLLBACK-RUNBOOK.md §4.
    "BOARDD_SOCK", os.path.join(HERMES_HOME, "kanban", "boardd-run", "boardd.sock")
)
DEFAULT_BUSY_TIMEOUT_MS = int(os.environ.get("BOARDD_BUSY_TIMEOUT_MS", "5000"))
# Bound the WAL: with ONE connection there is no reader holding it open, so a
# TRUNCATE checkpoint on an idle timer keeps kanban.db-wal from growing.
CHECKPOINT_INTERVAL_S = float(os.environ.get("BOARDD_CHECKPOINT_INTERVAL_S", "60"))
BACKUP_INTERVAL_S = float(os.environ.get("BOARDD_BACKUP_INTERVAL_S", "900"))
BACKUP_KEEP = int(os.environ.get("BOARDD_BACKUP_KEEP", "6"))
DISKGUARD_INTERVAL_S = float(os.environ.get("BOARDD_DISKGUARD_INTERVAL_S", "30"))
DISKGUARD_MIN_FREE_BYTES = int(
    os.environ.get("BOARDD_DISKGUARD_MIN_FREE_BYTES", str(2 * 1024 * 1024 * 1024))
)
# How long a client request may sit in the queue before we give up (safety).
HANDLER_TIMEOUT_S = float(os.environ.get("BOARDD_HANDLER_TIMEOUT_S", "30"))
# Retain applied_ops long enough that no in-flight client retry can miss its
# stored result, then prune (checkpoint timer). Must exceed KB_CLIENT_RETRY_DEADLINE.
APPLIED_OPS_TTL_S = int(os.environ.get("BOARDD_APPLIED_OPS_TTL_S", "3600"))
# Bound read-proxy results + runtime so no single query can monopolize the sole
# DB thread (BLOCKER-2 / DoS). A runaway scan is interrupted, keeping every op
# short enough that the heartbeat watchdog can tell busy from wedged.
MAX_QUERY_ROWS = int(os.environ.get("BOARDD_MAX_QUERY_ROWS", "20000"))
QUERY_DEADLINE_S = float(os.environ.get("BOARDD_QUERY_DEADLINE_S", "5"))
# INTERACTIVE TRANSACTIONS: a client (a real kanban_db write_txn) may hold ONE
# broker transaction open across round-trips so mid-txn reads can branch. Bound
# it: if the client sends no txn activity for this long, the broker ROLLBACKs
# and frees the sole DB thread (a dead client can't wedge the board). The body
# of a write_txn is pure local Python between statements (no external I/O), so
# round-trips are sub-ms and this ceiling is never hit in healthy operation.
TXN_DEADLINE_S = float(os.environ.get("BOARDD_TXN_DEADLINE_S", "5"))
# ABSOLUTE ceiling on how long ANY interactive transaction may hold the sole DB
# thread — enforced even while the client keeps sending statements (the idle
# deadline above only catches TOTAL silence). A legit write_txn is a few sub-ms
# round-trips; anything past this is rolled back so a slow-but-progressing (or
# FS-I/O-blocked) client cannot head-of-line-block every other board op
# (HIGH #1). Bounds the worst-case block a concurrent claim/heartbeat can see.
TXN_MAX_S = float(os.environ.get("BOARDD_TXN_MAX_S", "2.0"))
MAX_LINE_BYTES = int(os.environ.get("BOARDD_MAX_LINE_BYTES", str(4 * 1024 * 1024)))


def _parse_host_limits(raw: str) -> dict[str, int]:
    limits: dict[str, int] = {}
    for item in raw.split(","):
        name, separator, value = item.strip().partition(":")
        if not separator or not name:
            continue
        try:
            limit = int(value)
        except ValueError:
            continue
        if limit >= 0:
            limits[name] = limit
    return limits


BOARDD_HOST_IDENTITIES = frozenset(
    item.strip()
    for item in os.environ.get(
        "BOARDD_HOST_IDENTITIES", "blitz,blitz-vps-2"
    ).split(",")
    if item.strip()
)
BOARDD_HOST_LIMITS = _parse_host_limits(
    os.environ.get("BOARDD_HOST_LIMITS", "blitz:12,blitz-vps-2:8")
)
BOARDD_GLOBAL_MAX = int(os.environ.get("BOARDD_GLOBAL_MAX", "20"))

APPLIED_OPS_DDL = (
    "CREATE TABLE IF NOT EXISTS applied_ops ("
    "  op_id     TEXT PRIMARY KEY,"
    "  worker_id TEXT,"
    "  seq       INTEGER,"
    "  op        TEXT,"
    "  result    TEXT,"
    "  ts        INTEGER NOT NULL"
    ")"
)
# Durable ledger of COMMITTED interactive-txn tokens. The token is inserted
# INSIDE the client's transaction just before COMMIT, so it is atomic with the
# writes. If a txn_commit ACK is lost (broker restart in the commit->ack window)
# the client's retry lands on a fresh broker with no open txn; instead of a
# spurious TxnStale (which would skip the write_txn caller's side-effect tail —
# leaked worktree / unfired hook, MEDIUM #2) the broker finds the token here and
# reports the commit as SUCCEEDED, so complete_task/etc. run their tail.
COMMITTED_TXNS_DDL = (
    "CREATE TABLE IF NOT EXISTS committed_txns ("
    "  token TEXT PRIMARY KEY,"
    "  ts    INTEGER NOT NULL"
    ")"
)

# Read pragmas allowed through the read-only proxy. Anything that could write
# (journal_mode=, wal_checkpoint, vacuum, ...) is rejected.
_RO_PRAGMA_WHITELIST = {
    "integrity_check", "quick_check", "table_info", "index_list",
    "index_info", "database_list", "page_count", "freelist_count",
    "page_size", "wal_autocheckpoint", "foreign_key_check", "table_list",
    "schema_version", "user_version",
}
_RO_FIRST_TOKEN = {"select", "with", "explain", "values"}
# Statements permitted inside an interactive transaction (txn_exec). Reads +
# UPDATE/INSERT/DELETE only; read-PRAGMAs handled via _RO_PRAGMA_WHITELIST.
# DDL / ATTACH / DETACH / VACUUM / REINDEX / write-PRAGMA are rejected (LOW #4).
_TXN_EXEC_WHITELIST = {"select", "with", "explain", "values",
                       "update", "insert", "delete"}


# Markers that identify an ENOSPC / disk-full failure so it is ALARMED and
# reported DISTINCTLY from logical corruption (rev6 #4) — a full disk must never
# masquerade as a broker/torn-extend defect (which would trigger a wrong REINDEX
# / rollback). SQLite surfaces these as OperationalError text.
_DISK_FULL_MARKERS = (
    "disk i/o error", "database or disk is full", "disk full",
    "no space left", "enospc", "out of memory",  # mmap ENOSPC can read as OOM
)


def _is_disk_full(exc) -> bool:
    return any(m in str(exc).lower() for m in _DISK_FULL_MARKERS)


class DiskGuardError(RuntimeError):
    pass


class _Holder:
    """One-shot result slot the DB thread fills and the caller waits on."""
    __slots__ = ("_ev", "value")

    def __init__(self) -> None:
        self._ev = threading.Event()
        self.value = None

    def set(self, v) -> None:
        self.value = v
        self._ev.set()

    def wait(self, timeout: float):
        # Returns None on TIMEOUT (a distinct sentinel, NOT a fake ok:false).
        # The request handler drops the socket on None so the client sees a
        # transport failure and resends the SAME op_id -> applied_ops makes it
        # exactly-once even though the original op may still be queued and
        # commit later (BLOCKER-2). Never tell a client "failed" for an op that
        # can still apply.
        if not self._ev.wait(timeout):
            return None
        return self.value


# Sentinel to stop the DB thread.
_STOP = object()


def _now() -> int:
    return int(time.time())


def _new_task_id() -> str:
    return "t_" + secrets.token_hex(4)


def _canon_assignee(a):
    # Simplified, dependency-free normalization (HARD RULE: no imports/subprocess
    # on the handler path). The live layer lowercases via
    # profiles.normalize_profile_name; for broker-native rows this parity match
    # (strip+lower) is sufficient and pure.
    if a is None:
        return None
    a = str(a).strip()
    return a.lower() or None


def _pid_alive(pid) -> bool:
    """Return whether *pid* is a live local process without signalling it."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok — boardd is a POSIX UDS daemon
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


# --------------------------------------------------------------------------- #
# The broker
# --------------------------------------------------------------------------- #
class Broker:
    def __init__(self, db_path: str, sock_path: str, *,
                 schema_sql_file: str | None = None,
                 import_schema: bool = False):
        # Canonical absolute realpath (hygiene; the single connection is the fix).
        self.db_realpath = os.path.realpath(db_path)
        self.sock_path = sock_path
        self.schema_sql_file = schema_sql_file
        self.import_schema = import_schema
        self.q: "queue.Queue" = queue.Queue()
        self.conn: sqlite3.Connection | None = None
        self._stop = threading.Event()
        self._db_thread_obj: threading.Thread | None = None
        self._server: "BrokerServer | None" = None
        # health / stats
        self._started_at = _now()
        self._last_commit_ts = 0
        self._writes_applied = 0
        self._replays = 0
        self._reads = 0
        self._errors = 0
        self._disk_ok = True
        self._disk_free = -1
        self._lock = threading.Lock()  # guards counters (writes from db thread only)
        # DB-thread liveness heartbeat. Bumped at the top of every queue-loop
        # iteration AND from long-op progress callbacks (backup). The watchdog
        # pets systemd only when this is fresh -> a WEDGE (no bump) restarts us,
        # a legit BUSY op (bumps via progress) does NOT (HIGH-3).
        self._db_heartbeat = time.monotonic()
        self._db_loop_started = False   # watchdog startup grace (HIGH-3)
        # interactive-transaction state (DB thread only)
        self._txn_token = None
        self._txn_deadline = 0.0      # idle deadline (refreshed per statement)
        self._txn_started = 0.0       # wall-clock start (absolute cap, NOT refreshed)
        self._txns = 0
        self._txn_capped = 0          # count of txns rolled back by the absolute cap
        self._integrity_alarm = None  # set to the failing check result on integrity fail

    # ---- schema / connection (RUNS ONLY IN THE DB THREAD) ------------------ #
    def _load_schema_sql(self) -> str:
        if self.import_schema:
            # Production identity: pull the exact DDL the live module ships.
            import hermes_cli.kanban_db as k  # noqa: WPS433
            return k.SCHEMA_SQL
        if self.schema_sql_file:
            with open(self.schema_sql_file, "r", encoding="utf-8") as fh:
                return fh.read()
        raise RuntimeError(
            "no schema source: pass --schema-sql-file or --import-schema"
        )

    def _open_conn(self) -> sqlite3.Connection:
        """Open THE single connection. Called only from the DB thread, after any
        daemonize/fork, so the handle is never inherited across a fork."""
        os.makedirs(os.path.dirname(self.db_realpath), exist_ok=True)
        if self.import_schema:
            # Obtain the connection FROM the live layer so schema + pragmas are
            # byte-identical to production. kanban_db.connect() runs
            # SCHEMA_SQL + additive migrations and sets WAL/synchronous=FULL/
            # foreign_keys/secure_delete/cell_size_check.
            import hermes_cli.kanban_db as k  # noqa: WPS433
            from pathlib import Path
            conn = k.connect(db_path=Path(self.db_realpath))
        else:
            conn = sqlite3.connect(
                self.db_realpath,
                isolation_level=None,           # autocommit; we drive BEGIN/COMMIT
                timeout=DEFAULT_BUSY_TIMEOUT_MS / 1000.0,
                check_same_thread=True,         # hard guard: only the DB thread
            )
            conn.row_factory = sqlite3.Row
            conn.execute(f"PRAGMA busy_timeout={DEFAULT_BUSY_TIMEOUT_MS}")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            # wal_autocheckpoint left at the SQLite default (safe with one
            # connection); the idle TRUNCATE timer bounds the WAL.
            conn.executescript(self._load_schema_sql())
        # broker-added ledgers: applied_ops (native-op idempotency) +
        # committed_txns (interactive-txn ack-loss recovery). Both additive.
        conn.execute(APPLIED_OPS_DDL)
        conn.execute(COMMITTED_TXNS_DDL)
        return conn

    def bump(self) -> None:
        """Mark the DB thread alive (called each loop iteration + from long-op
        progress callbacks so the watchdog can tell busy from wedged)."""
        self._db_heartbeat = time.monotonic()

    def _open_conn_with_recovery(self) -> sqlite3.Connection:
        """Open the single connection; if the board is index-corrupt at open,
        attempt IN-PLACE recovery (REINDEX) on a raw handle before giving up —
        so a damaged board does NOT become a 1 Hz crash loop that never gets to
        run recovery (HIGH-3)."""
        try:
            conn = self._open_conn()
            self._ensure_custody_columns(conn)
            return conn
        except Exception as exc:
            msg = str(exc).lower()
            if not ("malformed" in msg or "corrupt" in msg or "integrity" in msg
                    or "not a database" in msg):
                raise
            _log.error("boardd: board damaged during connection setup (%s) — "
                       "attempting in-place "
                       "REINDEX recovery on a raw handle", exc)
            raw = sqlite3.connect(self.db_realpath, isolation_level=None,
                                  timeout=DEFAULT_BUSY_TIMEOUT_MS / 1000.0)
            raw.row_factory = sqlite3.Row
            raw.execute("PRAGMA busy_timeout=%d" % DEFAULT_BUSY_TIMEOUT_MS)
            raw.execute("REINDEX")
            chk = raw.execute("PRAGMA integrity_check").fetchone()[0]
            if chk != "ok":
                raw.close()
                _log.error("boardd: REINDEX did not recover (integrity=%s); a "
                           "file-level restore is required (see runbook)", chk)
                raise
            _log.warning("boardd: in-place REINDEX recovered the board")
            raw.execute("PRAGMA journal_mode=WAL")
            raw.execute("PRAGMA synchronous=FULL")
            raw.execute("PRAGMA foreign_keys=ON")
            raw.executescript(self._load_schema_sql())
            raw.execute(APPLIED_OPS_DDL)
            raw.execute(COMMITTED_TXNS_DDL)
            self._ensure_custody_columns(raw)
            return raw

    @staticmethod
    def _ensure_custody_columns(conn: sqlite3.Connection) -> None:
        """Add broker-owned custody metadata to fresh and legacy task tables."""
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)")}
        additions = {
            "host_identity": "TEXT",
            "claim_pid": "INTEGER",
            "claimed_at": "INTEGER",
        }
        for name, column_type in additions.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE tasks ADD COLUMN {name} {column_type}")

    # ---- DB thread --------------------------------------------------------- #
    def _db_thread(self) -> None:
        try:
            self.conn = self._open_conn_with_recovery()
        except Exception as exc:  # unrecoverable: exit; systemd StartLimit gates the loop
            _log.error("boardd: failed to open board connection: %s", exc)
            self._stop.set()
            self._sd_notify("STOPPING=1")
            os._exit(75)
        _log.info("boardd: owning single connection to %s", self.db_realpath)
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        self._db_loop_started = True  # end watchdog startup grace
        while True:
            self.bump()  # DB thread is alive and about to take the next item
            try:
                # Poll with a timeout SHORTER than the watchdog 'fresh' window so
                # an IDLE broker still loops back to bump() the heartbeat. A bare
                # blocking get() parked the thread here whenever no client was
                # connected, letting _db_heartbeat age past the fresh window ->
                # systemd watchdog-killed an otherwise-healthy idle broker every
                # ~30s (crash-loop between dispatch bursts / during quiesce).
                item = self.q.get(timeout=5.0)
            except queue.Empty:
                continue  # idle: re-loop; bump() keeps the heartbeat fresh
            if item is _STOP:
                break
            req, holder = item
            try:
                resp = self._handle(req)
            except Exception as exc:  # never let the DB thread die
                self._errors += 1
                self._classify_and_alert(exc)   # ENOSPC -> disk-full alarm (#4)
                resp = {"ok": False, "error": str(exc), "etype": type(exc).__name__}
            if holder is not None:
                holder.set(resp)
            # If that op opened an interactive transaction, take over the loop
            # until it commits/rolls back/deadlines — processing ONLY this txn's
            # statements and DEFERRING every other client's op (a second
            # BEGIN IMMEDIATE would error / break isolation).
            if self._txn_token is not None:
                self._run_txn_loop()
        # graceful close
        try:
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass
        _log.info("boardd: DB thread stopped, connection closed")

    def _alert(self, kind: str, detail: str) -> None:
        """Raise a real, out-of-band ALERT (not just a log line). Writes an
        alert file next to the board so an operator / the integrity-guard
        watchdog picks it up, and flips the in-memory alarm surfaced by ping."""
        _log.error("boardd: INTEGRITY ALERT %s: %s", kind, detail)
        self._integrity_alarm = {"kind": kind, "detail": detail, "ts": _now()}
        try:
            alert_path = os.path.join(
                os.path.dirname(self.db_realpath), "boardd-INTEGRITY-ALERT"
            )
            with open(alert_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(self._integrity_alarm) + "\n")
        except Exception:
            pass

    def _classify_and_alert(self, exc) -> bool:
        """If exc is an ENOSPC/disk-full failure, raise a DISTINCT 'disk-full'
        alarm and fail closed (refuse further writes) — never let it be reported
        as corruption (rev6 #4). Returns True if it was disk-full."""
        if _is_disk_full(exc):
            self._disk_ok = False
            self._alert("disk-full", f"ENOSPC/disk-full write failure: {exc}")
            return True
        return False

    def _broker_flen_check(self) -> None:
        """Torn-extend tripwire, run RIGHT AFTER wal_checkpoint(TRUNCATE) so the
        -wal is empty and page_count reflects the fully-checkpointed main file —
        i.e. it ACTUALLY EVALUATES (the earlier version bailed whenever a -wal
        existed, ~always, and never ran; MEDIUM #3). A real shortfall ALERTS.
        (The client-side _check_file_length_invariant is a no-op under the flag
        because post-cutover the board is chmod 0600 and the client cannot open
        the file — verified; the WAL false-positive theory did not reproduce.)"""
        try:
            wal = self.db_realpath + "-wal"
            wal_sz = os.path.getsize(wal) if os.path.exists(wal) else 0
            if wal_sz > 32:
                return  # WAL holds frames -> main file legitimately lags
            # Read the RAW header (NOT PRAGMA page_count — SQLite reconciles that
            # to the truncated file size on open, hiding the shortfall). Header:
            # offset 16 = page size (2B BE, 1==65536); offset 28 = db size in
            # pages (4B BE). This is exactly what a torn extend corrupts.
            with open(self.db_realpath, "rb") as fh:
                fh.seek(16); ps_b = fh.read(2)
                fh.seek(28); pc_b = fh.read(4)
            if len(pc_b) < 4 or len(ps_b) < 2:
                return
            ps = int.from_bytes(ps_b, "big") or 1
            if ps == 1:
                ps = 65536
            pc = int.from_bytes(pc_b, "big")
            if pc == 0:
                return  # new/empty db
            fs = os.path.getsize(self.db_realpath)
            if (fs // ps) < pc:
                self._alert("torn-extend",
                            f"header claims {pc} pages, main file has {fs // ps} "
                            f"(page_size={ps}, file_size={fs})")
        except Exception:
            pass

    def _periodic_quick_check(self) -> None:
        """Periodic PRAGMA quick_check on the owned connection — the real
        corruption tripwire (data-page level). ALERTS on any non-'ok' (#3)."""
        try:
            res = self.conn.execute("PRAGMA quick_check").fetchone()
            val = res[0] if res else "?"
            if val != "ok":
                self._alert("quick_check", str(val))
        except Exception as exc:
            if not self._classify_and_alert(exc):   # ENOSPC vs real corruption (#4)
                self._alert("quick_check_error", str(exc))

    def _run_txn_loop(self) -> None:
        """Own the DB thread while ONE interactive transaction is open: process
        only that txn's statements, DEFER every other client's op, and ROLLBACK
        if the client stalls past the deadline (a dead client cannot wedge the
        board)."""
        import queue as _queue
        deferred = []
        while self._txn_token is not None:
            self.bump()
            now = time.monotonic()
            idle_left = self._txn_deadline - now
            abs_left = (self._txn_started + TXN_MAX_S) - now
            # Roll back on EITHER: total idle silence (client died) OR the
            # ABSOLUTE wall-clock cap (a slow-but-progressing / FS-I/O-blocked
            # client that would otherwise head-of-line-block the whole board).
            if idle_left <= 0 or abs_left <= 0:
                why = ("idle %.1fs" % TXN_DEADLINE_S) if idle_left <= 0 \
                    else ("absolute cap %.1fs (slow holder)" % TXN_MAX_S)
                _log.error("boardd: interactive txn %s exceeded %s — ROLLBACK; "
                           "deferring clients released", self._txn_token, why)
                try:
                    self.conn.execute("ROLLBACK")
                except Exception:
                    pass
                self._txn_token = None
                if abs_left <= 0 and idle_left > 0:
                    self._txn_capped += 1
                break
            remaining = min(idle_left, abs_left)
            try:
                item = self.q.get(timeout=min(remaining, 0.5))
            except _queue.Empty:
                continue
            if item is _STOP:
                try:
                    self.conn.execute("ROLLBACK")
                except Exception:
                    pass
                self._txn_token = None
                self.q.put(_STOP)   # re-inject for the outer loop
                break
            req, holder = item
            op = req.get("op")
            if op in ("txn_exec", "txn_commit", "txn_rollback") \
               and (req.get("args") or {}).get("txn") == self._txn_token:
                try:
                    resp = self._handle(req)
                except Exception as exc:
                    self._errors += 1
                    self._classify_and_alert(exc)   # ENOSPC -> disk-full alarm (#4)
                    resp = {"ok": False, "error": str(exc),
                            "etype": type(exc).__name__}
                if holder is not None:
                    holder.set(resp)
            else:
                deferred.append(item)   # other clients wait for the txn to close
        for it in deferred:            # re-queue deferred ops (order preserved)
            self.q.put(it)

    def submit(self, req: dict, holder: "_Holder | None") -> None:
        self.q.put((req, holder))

    def call_sync(self, req: dict, timeout: float = HANDLER_TIMEOUT_S) -> dict:
        h = _Holder()
        self.submit(req, h)
        return h.wait(timeout)

    # ---- dispatch (RUNS IN DB THREAD) ------------------------------------- #
    def _handle(self, req: dict) -> dict:
        op = req.get("op")
        if op is None:
            return {"ok": False, "error": "missing op", "etype": "BadRequest"}
        # control / internal
        if op == "ping":
            return {"ok": True, "result": self._health()}
        if op == "stats":
            return {"ok": True, "result": self._health()}
        if op == "_stall" and os.environ.get("BOARDD_ALLOW_STALL") == "1":
            # TEST-ONLY (gated): occupy the DB thread to force a handler timeout,
            # so the BLOCKER-2 close-socket->client-resend->exactly-once path can
            # be demonstrated. Never enabled in production.
            time.sleep(float((req.get("args") or {}).get("seconds", 1)))
            return {"ok": True, "result": "stalled"}
        # ---- interactive transactions (the real kanban_db write_txn surface) --
        if op == "txn_begin":
            if self._txn_token is not None:
                return {"ok": False, "error": "a transaction is already open",
                        "etype": "TxnBusy"}
            if not self._disk_ok:
                return {"ok": False, "error": "disk guard: refusing writes",
                        "etype": "DiskGuardError"}
            self.conn.execute("BEGIN IMMEDIATE")
            self._txn_token = secrets.token_hex(8)
            now = time.monotonic()
            self._txn_deadline = now + TXN_DEADLINE_S
            self._txn_started = now       # absolute cap anchor (NOT refreshed)
            self._txns += 1
            return {"ok": True, "result": {"txn": self._txn_token}}
        if op == "txn_exec":
            a = req.get("args") or {}
            if a.get("txn") != self._txn_token or self._txn_token is None:
                return {"ok": False, "error": "no such open transaction",
                        "etype": "TxnStale"}
            sql = a["sql"]
            first = sql.lstrip().split(None, 1)[0].lower() if sql.strip() else ""
            # Defense-in-depth (LOW #4): a txn statement may only read or do
            # UPDATE/INSERT/DELETE. Reject DDL / ATTACH / DETACH / write-PRAGMA /
            # VACUUM / REINDEX etc. so a compromised client can't run arbitrary
            # SQL on the owned connection. Read-PRAGMAs are limited to the RO set.
            if first == "pragma":
                m = re.match(r"pragma\s+([a-z_]+)", sql.strip(), re.I)
                if not m or m.group(1).lower() not in _RO_PRAGMA_WHITELIST:
                    return {"ok": False, "error": f"txn_exec: pragma not allowed",
                            "etype": "Forbidden"}
            elif first not in _TXN_EXEC_WHITELIST:
                return {"ok": False,
                        "error": f"txn_exec rejects {first!r} (DDL/ATTACH/etc.)",
                        "etype": "Forbidden"}
            self._txn_deadline = time.monotonic() + TXN_DEADLINE_S
            cur = self.conn.execute(sql, a.get("params", []))
            rows = ([dict(r) for r in cur.fetchall()]
                    if first in ("select", "with", "values", "explain", "pragma")
                    else None)
            return {"ok": True, "result": {"rows": rows, "rowcount": cur.rowcount,
                                           "lastrowid": cur.lastrowid}}
        if op == "txn_commit":
            tok = (req.get("args") or {}).get("txn")
            if tok == self._txn_token and self._txn_token is not None:
                # Record the token INSIDE the still-open txn so it is atomic with
                # the writes, THEN commit. On crash after COMMIT the token is
                # durable; on crash before, it rolls back with everything (#2).
                self.conn.execute(
                    "INSERT OR IGNORE INTO committed_txns(token, ts) VALUES(?,?)",
                    (tok, _now()))
                self.conn.execute("COMMIT")
                self._txn_token = None
                self._writes_applied += 1
                self._last_commit_ts = _now()
                return {"ok": True, "result": "committed"}
            # Not the open txn. If it's a retry of a commit whose ACK was lost
            # (broker restarted in the commit->ack window), the token is durably
            # recorded -> report SUCCESS so the client's write_txn completes and
            # the caller runs its side-effect tail (no leaked worktree/hook #2).
            if self.conn.execute("SELECT 1 FROM committed_txns WHERE token=?",
                                 (tok,)).fetchone():
                return {"ok": True, "result": "committed", "replayed": True}
            return {"ok": False, "error": "no such open transaction",
                    "etype": "TxnStale"}
        if op == "txn_rollback":
            if (req.get("args") or {}).get("txn") == self._txn_token \
               and self._txn_token is not None:
                try:
                    self.conn.execute("ROLLBACK")
                except Exception:
                    pass
                self._txn_token = None
            return {"ok": True, "result": "rolledback"}
        if op == "_checkpoint":
            self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            # Bound the ledgers: prune rows far older than any client could still
            # retry (retry deadline ~90s; 1h is safe).
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                self.conn.execute("DELETE FROM applied_ops WHERE ts < ?",
                                  (_now() - APPLIED_OPS_TTL_S,))
                self.conn.execute("DELETE FROM committed_txns WHERE ts < ?",
                                  (_now() - APPLIED_OPS_TTL_S,))
                self.conn.execute("COMMIT")
            except Exception:
                try:
                    self.conn.execute("ROLLBACK")
                except sqlite3.OperationalError:
                    pass
            # Integrity tripwire (#3): runs RIGHT AFTER the TRUNCATE checkpoint,
            # when the -wal is empty, so the file-length check actually evaluates
            # (and is not a WAL-lag false positive). Plus a periodic quick_check.
            self._broker_flen_check()
            self._periodic_quick_check()
            return {"ok": True, "result": "checkpointed"}
        if op == "_reindex":
            # Broker-OWNED recovery: REINDEX in place on the single owned
            # connection. This SUBSUMES the old integrity-guard, which
            # REINDEXed/inode-swapped the file from an OUTSIDE process while the
            # gateway held a stale handle -> immediate re-corruption. Here there
            # is exactly one handle and no file swap.
            self.conn.execute("REINDEX")
            rows = self.conn.execute("PRAGMA integrity_check").fetchall()
            return {"ok": True, "result": [r[0] for r in rows]}
        if op == "_backup_to":
            path = req["args"]["path"]
            # Online backup API on the OWNED connection (single-writer, so the
            # copy is consistent). The progress callback bumps the DB-thread
            # heartbeat so a multi-second backup keeps petting the watchdog
            # instead of tripping WatchdogSec (HIGH-3). Never opens a 2nd
            # connection to the SOURCE.
            dest = sqlite3.connect(path)
            try:
                def _progress(_status, _remaining, _total):
                    self.bump()
                self.conn.backup(dest, pages=256, progress=_progress, sleep=0)
            finally:
                dest.close()
            return {"ok": True, "result": path}
        # reads
        if op in _READ_HANDLERS:
            self._reads += 1
            return _READ_HANDLERS[op](self, req.get("args") or {})
        # mutations
        if op in _MUTATION_HANDLERS:
            return self._apply_mutation(req)
        return {"ok": False, "error": f"unknown op {op!r}", "etype": "UnknownOp"}

    def _apply_mutation(self, req: dict) -> dict:
        op = req["op"]
        args = dict(req.get("args") or {})
        # Socket-derived custody always wins over spoofable JSON arguments.
        args["_peer_host_identity"] = req.get("_peer_host_identity")
        args["_peer_pid"] = req.get("_peer_pid")
        op_id = req.get("op_id")
        if not op_id:
            return {"ok": False, "error": "mutation requires op_id",
                    "etype": "BadRequest"}
        if not self._disk_ok:
            return {"ok": False,
                    "error": f"disk guard: <{DISKGUARD_MIN_FREE_BYTES} bytes free, "
                             f"refusing writes (free={self._disk_free})",
                    "etype": "DiskGuardError"}
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute(
                "SELECT result FROM applied_ops WHERE op_id=?", (op_id,)
            ).fetchone()
            if row is not None:
                conn.execute("COMMIT")
                self._replays += 1
                stored = json.loads(row["result"]) if row["result"] is not None else None
                return {"ok": True, "result": stored, "replayed": True}
            result = _MUTATION_HANDLERS[op](self, conn, args)
            conn.execute(
                "INSERT INTO applied_ops(op_id, worker_id, seq, op, result, ts) "
                "VALUES (?,?,?,?,?,?)",
                (op_id, req.get("worker_id"), req.get("seq"), op,
                 json.dumps(result), _now()),
            )
            conn.execute("COMMIT")
        except Exception as exc:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            self._errors += 1
            self._classify_and_alert(exc)   # ENOSPC -> distinct disk-full alarm (#4)
            return {"ok": False, "error": str(exc), "etype": type(exc).__name__}
        # Reply/ACK only AFTER COMMIT returns.
        self._writes_applied += 1
        self._last_commit_ts = _now()
        return {"ok": True, "result": result}

    def capped_query(self, sql: str, params, max_rows: int | None = None) -> list:
        """Run a read with a row cap + wall-clock interrupt (BLOCKER-2). The
        progress handler bumps the heartbeat AND aborts a runaway scan so a
        single query can never wedge the sole DB thread. A caller with a KNOWN
        large-but-bounded read (e.g. a dashboard listing) may pass an explicit
        higher ``max_rows`` (LOW); the wall-clock interrupt still bounds cost.
        Prefer LIMIT/OFFSET paging via kb_client.query for truly unbounded sets."""
        cap = int(max_rows) if max_rows else MAX_QUERY_ROWS
        deadline = time.monotonic() + QUERY_DEADLINE_S
        def _ph():
            self.bump()
            return 1 if time.monotonic() > deadline else 0
        self.conn.set_progress_handler(_ph, 4000)
        try:
            cur = self.conn.execute(sql, params)
            rows = cur.fetchmany(cap + 1)
        finally:
            self.conn.set_progress_handler(None, 0)
        if len(rows) > cap:
            raise ValueError(f"query exceeded row cap {cap}; pass max_rows or "
                             f"page with LIMIT/OFFSET")
        return [dict(r) for r in rows]

    def _health(self) -> dict:
        return {
            "db": self.db_realpath,
            "pid": os.getpid(),
            "uptime_s": _now() - self._started_at,
            "last_commit_ts": self._last_commit_ts,
            "writes_applied": self._writes_applied,
            "replays": self._replays,
            "reads": self._reads,
            "errors": self._errors,
            "queue_depth": self.q.qsize(),
            "disk_ok": self._disk_ok,
            "disk_free": self._disk_free,
            "txns": self._txns,
            "txn_capped": self._txn_capped,
            "txn_open": self._txn_token is not None,
            "integrity_alarm": self._integrity_alarm,
        }

    # ---- maintenance loops (enqueue onto the DB thread) ------------------- #
    def _checkpoint_loop(self) -> None:
        while not self._stop.wait(CHECKPOINT_INTERVAL_S):
            try:
                r = self.call_sync({"op": "_checkpoint"}, timeout=HANDLER_TIMEOUT_S)
                if r is None:
                    _log.warning("boardd: checkpoint timed out")
            except Exception as exc:
                _log.warning("boardd: checkpoint failed: %s", exc)

    def _backup_loop(self) -> None:
        import glob
        backups_dir = os.path.join(os.path.dirname(self.db_realpath), "boardd-backups")
        os.makedirs(backups_dir, exist_ok=True)
        # Sweep stale partials from a prior kill -9 mid-VACUUM.
        for stale in glob.glob(os.path.join(backups_dir, ".kanban.*.partial")):
            try:
                os.unlink(stale)
            except OSError:
                pass
        while not self._stop.wait(BACKUP_INTERVAL_S):
            if not self._disk_ok:
                continue
            ts = time.strftime("%Y%m%d-%H%M%S")
            tmp = os.path.join(backups_dir, f".kanban.{ts}.partial")
            final = os.path.join(backups_dir, f"kanban.{ts}.db")
            try:
                # Online backup runs on the DB thread (owned connection).
                r = self.call_sync({"op": "_backup_to", "args": {"path": tmp}},
                                   timeout=300)
                if not r or not r.get("ok"):
                    _log.warning("boardd: backup failed/timed out: %s", r)
                    try:
                        if os.path.exists(tmp):
                            os.unlink(tmp)
                    except OSError:
                        pass
                    continue
                # integrity_check of the COPY on a THROWAWAY connection (a
                # different file — never the live board, off the DB thread).
                vc = sqlite3.connect(tmp)
                ok = vc.execute("PRAGMA integrity_check").fetchone()[0]
                vc.close()
                if ok != "ok":
                    _log.error("boardd: backup integrity_check FAILED: %s", ok)
                    os.unlink(tmp)
                    continue
                os.replace(tmp, final)
                # rotate
                keep = sorted(glob.glob(os.path.join(backups_dir, "kanban.*.db")))
                for old in keep[:-BACKUP_KEEP]:
                    try:
                        os.unlink(old)
                    except OSError:
                        pass
                _log.info("boardd: backup ok -> %s", final)
            except Exception as exc:
                _log.warning("boardd: backup error: %s", exc)
                try:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                except OSError:
                    pass

    def _diskguard_loop(self) -> None:
        while not self._stop.wait(0 if self._disk_free < 0 else DISKGUARD_INTERVAL_S):
            try:
                st = os.statvfs(os.path.dirname(self.db_realpath))
                free = st.f_bavail * st.f_frsize
                self._disk_free = free
                was_ok = self._disk_ok
                self._disk_ok = free >= DISKGUARD_MIN_FREE_BYTES
                if was_ok and not self._disk_ok:
                    _log.error("boardd: DISK GUARD TRIPPED free=%d < %d — "
                               "REFUSING WRITES", free, DISKGUARD_MIN_FREE_BYTES)
                elif not was_ok and self._disk_ok:
                    _log.warning("boardd: disk recovered free=%d — writes resumed",
                                 free)
            except Exception as exc:
                _log.warning("boardd: diskguard error: %s", exc)
            if self._stop.is_set():
                break

    # ---- systemd sd_notify watchdog (dependency-free) --------------------- #
    def _sd_notify(self, state: str) -> None:
        addr = os.environ.get("NOTIFY_SOCKET")
        if not addr:
            return
        if addr[0] == "@":            # abstract namespace
            addr = "\0" + addr[1:]
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            s.connect(addr)
            s.sendall(state.encode("utf-8"))
            s.close()
        except Exception:
            pass

    def _watchdog_loop(self) -> None:
        usec = os.environ.get("WATCHDOG_USEC")
        if not usec:
            return
        watchdog_s = int(usec) / 1_000_000.0
        interval = max(1.0, watchdog_s / 3.0)
        # Pet only when the DB thread bumped its heartbeat recently. A genuine
        # WEDGE (deadlock / infinite loop) stops bumping -> no pet -> systemd
        # restarts us. A legit BUSY op (backup pumps the heartbeat via its
        # progress callback; quick ops bump between items) keeps petting. This
        # distinguishes busy from wedged (HIGH-3). All ops are bounded (queries
        # capped, checkpoint/reindex sub-second, backup pumps) so a healthy
        # thread never lets the heartbeat age past the fresh window.
        fresh = watchdog_s * 0.8
        while not self._stop.wait(interval):
            # Startup grace (HIGH-3): until the DB loop starts, the thread may be
            # in a long at-open REINDEX recovery on a damaged board — pet
            # unconditionally so the watchdog doesn't SIGKILL mid-recovery. Once
            # the loop starts, switch to heartbeat-based wedge detection.
            if not self._db_loop_started:
                self._sd_notify("WATCHDOG=1")
            elif (time.monotonic() - self._db_heartbeat) < fresh:
                self._sd_notify("WATCHDOG=1")
            else:
                _log.error("boardd: DB thread heartbeat stale (%.1fs) — "
                           "withholding watchdog pet; systemd will restart",
                           time.monotonic() - self._db_heartbeat)

    # ---- lifecycle --------------------------------------------------------- #
    def start(self) -> None:
        self._db_thread_obj = threading.Thread(
            target=self._db_thread, name="boardd-db", daemon=True)
        self._db_thread_obj.start()
        # prime disk guard once synchronously
        threading.Thread(target=self._diskguard_loop, name="boardd-disk",
                         daemon=True).start()
        threading.Thread(target=self._checkpoint_loop, name="boardd-ckpt",
                         daemon=True).start()
        threading.Thread(target=self._backup_loop, name="boardd-backup",
                         daemon=True).start()
        # UDS server. The socket lives in a SEPARATE boardd-owned dir (SOCKDIR,
        # 0710 group boardd) — NOT the 0700 DB dir — so odai-uid clients in the
        # boardd group can traverse to + connect the socket while having NO path
        # to the 0600 DB file. See ROLLBACK-RUNBOOK.md §4.
        sockdir = os.path.dirname(self.sock_path)
        os.makedirs(sockdir, exist_ok=True)
        # Best-effort self-heal of the socket-dir mode (group-traversable 0710).
        # The §4 `install -d -o boardd -g boardd -m 0710 <SOCKDIR>` is the source
        # of truth; this only fixes a dir boardd itself just created (idempotent,
        # and safely skipped via OSError when boardd is not the dir owner).
        try:
            os.chmod(sockdir, 0o710)
        except OSError:
            pass
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        self._server = BrokerServer(self.sock_path, _RequestHandler, self)
        # 0660 GROUP boardd — the dedicated-user fail-closed design. boardd runs
        # as User=boardd/Group=boardd, so the socket is created group `boardd`;
        # 0660 lets group members (odai, added to `boardd`) connect, while the
        # 0600 boardd-owned DB stays unreachable to them. NOTE: 0600 here would
        # lock the odai-uid gateway/dispatch/kb_client OUT of the broker.
        os.chmod(self.sock_path, 0o660)
        _log.info("boardd: listening on %s (socket 0660 group)", self.sock_path)
        # systemd readiness + watchdog (no-ops when not under systemd notify).
        self._sd_notify("READY=1")
        threading.Thread(target=self._watchdog_loop, name="boardd-watchdog",
                         daemon=True).start()

    def serve_forever(self) -> None:
        assert self._server is not None
        self._server.serve_forever(poll_interval=0.5)

    def shutdown(self, *_a) -> None:
        _log.info("boardd: shutdown requested")
        self._stop.set()
        if self._server is not None:
            self._server.shutdown()
        self.q.put(_STOP)
        if self._db_thread_obj is not None:
            self._db_thread_obj.join(timeout=10)
        try:
            if os.path.exists(self.sock_path):
                os.unlink(self.sock_path)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Mutation handlers (PURE SQL — run inside BEGIN IMMEDIATE opened by
# _apply_mutation; they must NOT issue BEGIN/COMMIT). Faithful to
# hermes_cli.kanban_db op semantics.
# --------------------------------------------------------------------------- #
def _ev(conn, task_id, kind, payload=None, run_id=None):
    conn.execute(
        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
        "VALUES (?,?,?,?,?)",
        (task_id, run_id, kind,
         json.dumps(payload, ensure_ascii=False) if payload else None, _now()),
    )


def _h_create_task(broker, conn, a):
    title = a.get("title")
    if not title or not str(title).strip():
        raise ValueError("title is required")
    tid = a.get("id") or _new_task_id()
    status = a.get("status", "running")
    conn.execute(
        "INSERT INTO tasks (id, title, body, assignee, status, priority, "
        "created_by, created_at, workspace_kind) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (tid, str(title), a.get("body"), _canon_assignee(a.get("assignee")),
         status, int(a.get("priority", 0)), _canon_assignee(a.get("created_by")),
         _now(), a.get("workspace_kind", "scratch")),
    )
    _ev(conn, tid, "created", {"title": str(title)})
    return {"id": tid, "status": status}


def _h_add_comment(broker, conn, a):
    task_id, author, body = a["task_id"], a["author"], a["body"]
    if not body or not str(body).strip():
        raise ValueError("comment body is required")
    if not author or not str(author).strip():
        raise ValueError("comment author is required")
    if not conn.execute("SELECT 1 FROM tasks WHERE id=?", (task_id,)).fetchone():
        raise ValueError(f"unknown task {task_id}")
    cur = conn.execute(
        "INSERT INTO task_comments (task_id, author, body, created_at) "
        "VALUES (?,?,?,?)",
        (task_id, str(author).strip(), str(body).strip(), _now()),
    )
    _ev(conn, task_id, "commented", {"author": author, "len": len(str(body))})
    return {"comment_id": int(cur.lastrowid or 0)}


def _h_set_workspace_path(broker, conn, a):
    conn.execute("UPDATE tasks SET workspace_path=? WHERE id=?",
                 (str(a["path"]), a["task_id"]))
    return {"ok": True}


def _h_set_branch_name(broker, conn, a):
    conn.execute("UPDATE tasks SET branch_name=? WHERE id=?",
                 (str(a["branch_name"]), a["task_id"]))
    return {"ok": True}


def _h_set_body(broker, conn, a):
    cur = conn.execute("UPDATE tasks SET body=? WHERE id=?",
                       (a.get("body"), a["task_id"]))
    if cur.rowcount == 0:
        raise ValueError(f"card {a['task_id']} not found")
    return {"rowcount": cur.rowcount}


def _h_set_status(broker, conn, a):
    """Mirror of the `kb` CLI setstatus (per-seat direct-opener reroute)."""
    task_id, st = a["task_id"], a["status"]
    now = _now()
    if st == "running":
        cur = conn.execute(
            "UPDATE tasks SET status='running', started_at=COALESCE(started_at,?), "
            "last_heartbeat_at=?, claim_lock='seat-sticky', claim_expires=? "
            "WHERE id=?",
            (now, now, now + 315360000, task_id),
        )
    else:
        cur = conn.execute(
            "UPDATE tasks SET status=?, started_at=COALESCE(started_at,?) WHERE id=?",
            (st, now, task_id),
        )
    if cur.rowcount == 0:
        raise ValueError(f"card {task_id} not found (status NOT changed to {st})")
    return {"rowcount": cur.rowcount, "status": st}


def _h_heartbeat(broker, conn, a):
    task_id = a["task_id"]
    note = a.get("note")
    now = _now()
    cur = conn.execute(
        "UPDATE tasks SET last_heartbeat_at=? WHERE id=? AND status='running'",
        (now, task_id),
    )
    if cur.rowcount != 1:
        return {"ok": False, "reason": "not_running"}
    r = conn.execute("SELECT current_run_id FROM tasks WHERE id=?",
                     (task_id,)).fetchone()
    run_id = r["current_run_id"] if r else None
    if run_id is not None:
        conn.execute("UPDATE task_runs SET last_heartbeat_at=? WHERE id=?",
                     (now, run_id))
    _ev(conn, task_id, "heartbeat", {"note": note} if note else None, run_id=run_id)
    return {"ok": True}


def _h_claim(broker, conn, a):
    """Atomic ready->running CAS. At most one writer wins the same task."""
    task_id = a["task_id"]
    now = _now()
    lock = a.get("claimer") or f"{socket.gethostname()}:{a.get('worker_id', '?')}"
    ttl = int(a.get("ttl_seconds", 7200))
    expires = now + ttl
    cur = conn.execute(
        "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
        "started_at=COALESCE(started_at,?) "
        "WHERE id=? AND status='ready' AND claim_lock IS NULL",
        (lock, expires, now, task_id),
    )
    if cur.rowcount != 1:
        return {"won": False, "task_id": task_id}
    trow = conn.execute(
        "SELECT assignee, max_runtime_seconds, current_step_key FROM tasks WHERE id=?",
        (task_id,),
    ).fetchone()
    run_cur = conn.execute(
        "INSERT INTO task_runs (task_id, profile, step_key, status, claim_lock, "
        "claim_expires, max_runtime_seconds, started_at) "
        "VALUES (?,?,?,'running',?,?,?,?)",
        (task_id, trow["assignee"] if trow else None,
         trow["current_step_key"] if trow else None, lock, expires,
         trow["max_runtime_seconds"] if trow else None, now),
    )
    run_id = run_cur.lastrowid
    conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, task_id))
    _ev(conn, task_id, "claimed", {"lock": lock, "expires": expires, "run_id": run_id},
        run_id=run_id)
    return {"won": True, "task_id": task_id, "claim_lock": lock, "run_id": run_id}


def _continuation_claim_guard(conn, task_id, assignee, now):
    """Apply the SQL-verifiable part of the active-PR continuation guard.

    The broker handler path must not perform network or subprocess work. The
    canonical normalized PR-ownership and continuation-authorization tables
    therefore provide the fail-closed local decision here.
    """
    active_prs = conn.execute(
        "SELECT canonical_url FROM task_pr_ownership "
        "WHERE task_id=? AND declared=1 AND last_seen_at>=? "
        "ORDER BY last_seen_at DESC",
        (task_id, now - 86400),
    ).fetchall()
    authorization = conn.execute(
        "SELECT id, pr_tuples, authorized_profile FROM continuation_authorizations "
        "WHERE task_id=? AND consumed_at IS NULL AND revoked_at IS NULL "
        "AND expires_at>? ORDER BY created_at DESC, id DESC LIMIT 1",
        (task_id, now),
    ).fetchone()

    if authorization is not None:
        try:
            pr_tuples = json.loads(authorization["pr_tuples"])
        except (TypeError, ValueError):
            return "continuation_denied"
        if (
            _canon_assignee(authorization["authorized_profile"]) != assignee
            or not isinstance(pr_tuples, list)
            or not pr_tuples
        ):
            return "continuation_denied"

    # A locally active PR remains guarded. A syntactically valid one-shot grant
    # still requires the canonical remote head/state verifier before it can be
    # consumed. boardd deliberately performs no network work in its DB thread,
    # so the native endpoint fails closed instead of bypassing PR custody.
    if active_prs:
        return "active_pr"
    return None


def _h_claim_with_custody(broker, conn, a):
    """Atomically claim one lane while enforcing DAG, PR, and host custody."""
    host_identity = a.get("_peer_host_identity")
    if host_identity not in BOARDD_HOST_IDENTITIES:
        return {"won": False, "reason": "bad_host_identity"}

    assignee = _canon_assignee(a.get("assignee"))
    if assignee is None:
        return {"won": False, "reason": "bad_assignee"}

    now = _now()
    try:
        ttl = int(a.get("ttl_seconds", 7200))
    except (TypeError, ValueError):
        ttl = 7200
    expires = now + ttl
    try:
        claim_pid = int(a.get("_peer_pid"))
    except (TypeError, ValueError):
        claim_pid = None
    worker_id = str(a.get("worker_id") or "?").strip() or "?"
    claim_lock = f"{host_identity}:{worker_id}:{now}"

    task_id = a.get("task_id")
    select_columns = (
        "id, assignee, status, claim_lock, claim_expires, claim_pid, "
        "host_identity, worker_pid, current_run_id, max_runtime_seconds, "
        "current_step_key"
    )
    if task_id:
        task = conn.execute(
            f"SELECT {select_columns} FROM tasks WHERE id=?",
            (task_id,),
        ).fetchone()
    else:
        task = conn.execute(
            f"SELECT {select_columns} FROM tasks "
            "WHERE assignee=? AND status='ready' AND claim_lock IS NULL "
            "ORDER BY priority DESC, created_at ASC, id ASC LIMIT 1",
            (assignee,),
        ).fetchone()
    if task is None:
        return {"won": False, "reason": "no_ready_task"}

    task_id = task["id"]
    if _canon_assignee(task["assignee"]) != assignee:
        return {"won": False, "reason": "assignee_mismatch"}
    if (
        task["status"] == "ready"
        and task["host_identity"] is not None
        and task["host_identity"] != host_identity
    ):
        return {"won": False, "reason": "wrong_host_identity"}

    stale_takeover = False
    if task["status"] == "running":
        if task["host_identity"] != host_identity:
            return {"won": False, "reason": "already_claimed"}
        stale_takeover = (
            task["claim_pid"] is not None
            and task["claim_expires"] is not None
            and int(task["claim_expires"]) <= now
            and not _pid_alive(task["claim_pid"])
            and not _pid_alive(task["worker_pid"])
        )
        if not stale_takeover:
            return {"won": False, "reason": "already_claimed"}
    elif task["status"] != "ready" or task["claim_lock"] is not None:
        return {"won": False, "reason": "already_claimed"}

    undone_parent = conn.execute(
        "SELECT 1 FROM task_links l JOIN tasks p ON p.id=l.parent_id "
        "WHERE l.child_id=? AND p.status NOT IN ('done','archived') LIMIT 1",
        (task_id,),
    ).fetchone()
    if undone_parent is not None:
        return {"won": False, "reason": "parent_not_done"}

    continuation_reason = _continuation_claim_guard(conn, task_id, assignee, now)
    if continuation_reason is not None:
        return {"won": False, "reason": continuation_reason}

    lane_owner = conn.execute(
        "SELECT id FROM tasks WHERE assignee=? AND status='running' AND id<>? LIMIT 1",
        (assignee, task_id),
    ).fetchone()
    if lane_owner is not None:
        return {"won": False, "reason": "lane_in_use"}

    host_limit = BOARDD_HOST_LIMITS.get(host_identity, 0)
    excluded_task = task_id if stale_takeover else ""
    host_running = conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE status='running' "
        "AND host_identity=? AND id<>?",
        (host_identity, excluded_task),
    ).fetchone()["n"]
    if int(host_running) >= host_limit:
        return {"won": False, "reason": "host_capacity"}

    global_running = conn.execute(
        "SELECT COUNT(*) AS n FROM tasks WHERE status='running' AND id<>?",
        (excluded_task,),
    ).fetchone()["n"]
    if int(global_running) >= BOARDD_GLOBAL_MAX:
        return {"won": False, "reason": "global_capacity"}

    if stale_takeover:
        cur = conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "started_at=COALESCE(started_at,?), host_identity=?, claim_pid=?, "
            "claimed_at=?, current_run_id=NULL, worker_pid=NULL "
            "WHERE id=? AND status='running' AND claim_pid=? AND claim_expires=? "
            "AND claim_expires<=?",
            (
                claim_lock,
                expires,
                now,
                host_identity,
                claim_pid,
                now,
                task_id,
                task["claim_pid"],
                task["claim_expires"],
                now,
            ),
        )
    else:
        cur = conn.execute(
            "UPDATE tasks SET status='running', claim_lock=?, claim_expires=?, "
            "started_at=COALESCE(started_at,?), host_identity=?, claim_pid=?, "
            "claimed_at=?, current_run_id=NULL, worker_pid=NULL "
            "WHERE id=? AND status='ready' AND claim_lock IS NULL",
            (claim_lock, expires, now, host_identity, claim_pid, now, task_id),
        )
    if cur.rowcount != 1:
        return {"won": False, "reason": "claim_failed"}

    prior_run_id = task["current_run_id"]
    if stale_takeover and prior_run_id is not None:
        conn.execute(
            "UPDATE task_runs SET status='reclaimed', outcome='reclaimed', "
            "ended_at=?, claim_lock=NULL, claim_expires=NULL, worker_pid=NULL "
            "WHERE id=? AND ended_at IS NULL",
            (now, prior_run_id),
        )

    run_cur = conn.execute(
        "INSERT INTO task_runs (task_id, profile, step_key, status, claim_lock, "
        "claim_expires, max_runtime_seconds, started_at) "
        "VALUES (?,?,?,'running',?,?,?,?)",
        (
            task_id,
            assignee,
            task["current_step_key"],
            claim_lock,
            expires,
            task["max_runtime_seconds"],
            now,
        ),
    )
    run_id = run_cur.lastrowid
    conn.execute("UPDATE tasks SET current_run_id=? WHERE id=?", (run_id, task_id))
    _ev(
        conn,
        task_id,
        "claimed",
        {
            "lock": claim_lock,
            "expires": expires,
            "host_identity": host_identity,
            "claim_pid": claim_pid,
            "run_id": run_id,
        },
        run_id=run_id,
    )
    return {
        "won": True,
        "task_id": task_id,
        "run_id": run_id,
        "claim_lock": claim_lock,
    }


def _h_record_worker_pid(broker, conn, a):
    task_id = a["task_id"]
    run_id = int(a["run_id"])
    worker_pid = int(a["worker_pid"])
    claim_lock = a["claim_lock"]
    if worker_pid <= 0:
        raise ValueError("worker_pid must be positive")
    task = conn.execute(
        "SELECT 1 FROM tasks WHERE id=? AND claim_lock=? "
        "AND current_run_id=? AND status='running'",
        (task_id, claim_lock, run_id),
    ).fetchone()
    if task is None:
        return {"ok": False, "reason": "stale_claim"}
    conn.execute("UPDATE tasks SET worker_pid=? WHERE id=?", (worker_pid, task_id))
    conn.execute("UPDATE task_runs SET worker_pid=? WHERE id=?", (worker_pid, run_id))
    return {"ok": True}


_EXEC_WHITELIST = {"update", "insert", "delete"}


def _h_exec(broker, conn, a):
    """Generic parametrized write (reroute target for predicate updates like
    board-steward's archive-debris). Single statement, params bound.

    WHITELIST (not blacklist): only UPDATE/INSERT/DELETE. This deliberately
    rejects ATTACH/DETACH (a client could ATTACH a second path-alias to the
    board and defeat the single-handle invariant), plus PRAGMA/ALTER/DROP/
    CREATE/VACUUM/REINDEX and anything else (M-6)."""
    sql = a["sql"]
    params = a.get("params", [])
    if _looks_multi_statement(sql):
        raise ValueError("exec: single statement only")
    first = sql.lstrip().split(None, 1)[0].lower() if sql.strip() else ""
    if first not in _EXEC_WHITELIST:
        raise ValueError(f"exec rejects {first!r}; only {sorted(_EXEC_WHITELIST)}")
    cur = conn.execute(sql, params)
    # round-trip the REAL lastrowid so a raw INSERT never silently returns None
    # (MEDIUM). BrokerConnection surfaces this on its cursor.
    return {"rowcount": cur.rowcount, "lastrowid": cur.lastrowid}


_MUTATION_HANDLERS = {
    "create_task": _h_create_task,
    "add_comment": _h_add_comment,
    "set_workspace_path": _h_set_workspace_path,
    "set_branch_name": _h_set_branch_name,
    "set_body": _h_set_body,
    "set_status": _h_set_status,
    "heartbeat": _h_heartbeat,
    "claim": _h_claim,
    "claim_with_custody": _h_claim_with_custody,
    "record_worker_pid": _h_record_worker_pid,
    "exec": _h_exec,
}


# --------------------------------------------------------------------------- #
# Read handlers (no applied_ops; single connection => consistent committed view)
# --------------------------------------------------------------------------- #
def _looks_multi_statement(sql: str) -> bool:
    s = sql.strip().rstrip(";")
    return ";" in s


def _rows(cur):
    return [dict(r) for r in cur.fetchall()]


def _h_query(broker, a):
    """Read-only SELECT/CTE/EXPLAIN proxy — powers `kb sql --read-only`,
    kanban-sync seat reads, the disk-guard read, doctor/diagnostics."""
    sql = a["sql"]
    params = a.get("params", [])
    stripped = sql.strip()
    if _looks_multi_statement(stripped):
        return {"ok": False, "error": "read proxy: single statement only",
                "etype": "BadRequest"}
    first = stripped.split(None, 1)[0].lower() if stripped else ""
    if first == "pragma":
        m = re.match(r"pragma\s+([a-z_]+)", stripped, re.I)
        name = (m.group(1).lower() if m else "")
        if name not in _RO_PRAGMA_WHITELIST:
            return {"ok": False, "error": f"pragma {name!r} not allowed read-only",
                    "etype": "Forbidden"}
    elif first not in _RO_FIRST_TOKEN:
        return {"ok": False, "error": f"read proxy rejects {first!r}",
                "etype": "Forbidden"}
    return {"ok": True, "result": broker.capped_query(sql, params, a.get("max_rows"))}


def _h_integrity_check(broker, a):
    rows = broker.conn.execute("PRAGMA integrity_check").fetchall()
    return {"ok": True, "result": [r[0] for r in rows]}


def _h_quick_check(broker, a):
    rows = broker.conn.execute("PRAGMA quick_check").fetchall()
    return {"ok": True, "result": [r[0] for r in rows]}


def _h_get_task(broker, a):
    r = broker.conn.execute("SELECT * FROM tasks WHERE id=?",
                            (a["task_id"],)).fetchone()
    return {"ok": True, "result": (dict(r) if r else None)}


def _h_list_tasks(broker, a):
    where, params = "", []
    if a.get("assignee"):
        where, params = "WHERE assignee=?", [_canon_assignee(a["assignee"])]
    cur = broker.conn.execute(
        f"SELECT * FROM tasks {where} ORDER BY created_at DESC LIMIT ?",
        params + [int(a.get("limit", 500))],
    )
    return {"ok": True, "result": _rows(cur)}


_READ_HANDLERS = {
    "query": _h_query,
    "integrity_check": _h_integrity_check,
    "quick_check": _h_quick_check,
    "get_task": _h_get_task,
    "list_tasks": _h_list_tasks,
}


# --------------------------------------------------------------------------- #
# UDS server
# --------------------------------------------------------------------------- #
def _peer_custody(peer_socket) -> tuple[int | None, str | None]:
    """Read a UDS peer PID and its process-start host identity token."""
    try:
        size = struct.calcsize("3i")
        credentials = peer_socket.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, size
        )
        peer_pid, _uid, _gid = struct.unpack("3i", credentials)
    except (AttributeError, OSError, struct.error):
        return None, None
    try:
        with open(f"/proc/{peer_pid}/environ", "rb") as environ_file:
            entries = environ_file.read().split(b"\0")
    except OSError:
        return peer_pid, None
    prefix = b"FLEET_HOST_IDENTITY="
    for entry in entries:
        if entry.startswith(prefix):
            try:
                identity = entry[len(prefix):].decode("utf-8").strip()
            except UnicodeDecodeError:
                return peer_pid, None
            return peer_pid, identity or None
    return peer_pid, None


class BrokerServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, sock_path, handler, broker):
        self.broker = broker
        super().__init__(sock_path, handler)


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        broker: Broker = self.server.broker
        for raw in self.rfile:
            if not raw:
                continue
            if len(raw) > MAX_LINE_BYTES:
                self._send({"ok": False, "error": "request too large",
                            "etype": "BadRequest"})
                continue
            try:
                req = json.loads(raw)
            except Exception as exc:
                self._send({"ok": False, "error": f"bad json: {exc}",
                            "etype": "BadRequest"})
                continue
            peer_pid, peer_host_identity = _peer_custody(self.request)
            req["_peer_pid"] = peer_pid
            req["_peer_host_identity"] = peer_host_identity
            resp = broker.call_sync(req)
            if resp is None:
                # DB thread did not answer within HANDLER_TIMEOUT_S. The op may
                # still be queued and commit later, so we must NOT tell the
                # client "failed". Drop the socket: the client sees a transport
                # failure and resends the SAME op_id; applied_ops dedups it ->
                # exactly-once preserved (BLOCKER-2).
                return
            self._send(resp)

    def _send(self, obj) -> None:
        try:
            self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
            self.wfile.flush()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="boardd — kanban single-writer broker")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--sock", default=DEFAULT_SOCK)
    ap.add_argument("--schema-sql-file", default=None,
                    help="DDL file (byte-identical extract of kanban_db.SCHEMA_SQL)")
    ap.add_argument("--import-schema", action="store_true",
                    help="obtain connection+schema from hermes_cli.kanban_db "
                         "(production: guarantees byte-identical schema/pragmas)")
    ap.add_argument("--log-level", default=os.environ.get("BOARDD_LOG_LEVEL", "INFO"))
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if not args.schema_sql_file and not args.import_schema:
        # default to importing the live schema in production
        args.import_schema = True

    broker = Broker(args.db, args.sock,
                    schema_sql_file=args.schema_sql_file,
                    import_schema=args.import_schema)

    signal.signal(signal.SIGTERM, broker.shutdown)
    signal.signal(signal.SIGINT, broker.shutdown)

    broker.start()
    try:
        broker.serve_forever()
    finally:
        broker.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
