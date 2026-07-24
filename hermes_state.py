#!/usr/bin/env python3
"""
SQLite State Store for Hermes Agent.

Provides persistent session storage with FTS5 full-text search, replacing
the per-session JSONL file approach. Stores session metadata, full message
history, and model configuration for CLI and gateway sessions.

Key design decisions:
- WAL mode for concurrent readers + one writer (gateway multi-platform)
- FTS5 virtual table for fast text search across all session messages
- Compression-triggered session splitting via parent_session_id chains
- Batch runner and RL trajectories are NOT stored here (separate systems)
- Session source tagging ('cli', 'telegram', 'discord', etc.) for filtering
"""

import asyncio
import json
import logging
import os
import random
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path

from agent.memory_manager import sanitize_context
from agent.message_sanitization import _sanitize_surrogates
from hermes_constants import get_hermes_home
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

try:  # Hard dependency, but tolerate scaffold-phase imports before pip install.
    import psutil
except ImportError:  # pragma: no cover - stripped/scaffold installs only
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_COMPRESSION_LOCK_HOLDER_PID_RE = re.compile(r"(?:^|:)pid=(\d+)(?::|$)")


def _compression_lock_holder_process_is_dead(holder: str) -> bool:
    """Return True only when a structured lock holder's local PID is gone.

    Compression locks are stored in a host-local SQLite database and holder
    IDs created by ``conversation_compression`` start with ``pid=<n>``. A
    process killed during gateway shutdown cannot release its lease, so waiting
    for the full TTL makes every new turn repeatedly attempt compaction. Reclaim
    only when the kernel proves that PID no longer exists; legacy/unstructured
    holders, same-process holders, permission errors, and any probe doubt
    remain protected until normal TTL expiry (conservative: PID reuse must
    never steal a live lease, and a wrongly-kept lease self-heals via TTL).
    """
    # Windows stays TTL-only: stdlib os.kill(pid, 0) is NOT a no-op probe
    # there (bpo-14484 — sig=0 maps to CTRL_C_EVENT and can kill the target's
    # console group), and PID recycling semantics make liveness a weaker
    # deadness signal. The 300s lease TTL remains the recovery path.
    if os.name == "nt":
        return False
    match = _COMPRESSION_LOCK_HOLDER_PID_RE.search(holder or "")
    if match is None:
        return False
    try:
        pid = int(match.group(1))
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if pid == os.getpid():
        # Same-process holder (e.g. another thread's live lease): never
        # self-reclaim — the lease refresher and release path own it.
        return False
    if psutil is not None:
        try:
            # psutil is the canonical cross-platform liveness answer
            # (CONTRIBUTING.md "Critical rules" #1). pid_exists() reports
            # recycled PIDs as alive — conservative, the TTL still applies.
            return not psutil.pid_exists(pid)
        except Exception:
            return False  # any doubt → keep the lease until TTL expiry
    # Scaffold-phase fallback only (psutil missing). POSIX-only by the
    # os.name gate above.
    try:
        os.kill(pid, 0)  # windows-footgun: ok — function early-returns on nt above
    except ProcessLookupError:
        return True
    except (PermissionError, OSError, OverflowError):
        return False
    return False


def _scrub_surrogates(value: Any) -> Any:
    """Replace lone surrogates when *value* is text; pass anything else through.

    sqlite3 encodes bound ``str`` parameters as UTF-8 and raises
    ``UnicodeEncodeError`` on lone surrogates (U+D800..U+DFFF), so a single
    such code point anywhere in a message aborts the whole write. No-op for
    well-formed text.
    """
    return _sanitize_surrogates(value) if isinstance(value, str) else value


def workspace_key(row: Dict[str, Any]) -> Optional[str]:
    """A session's workspace grouping key: its git repo root when known, else
    its cwd.

    Branch is deliberately excluded so checking out a new branch doesn't
    fragment a workspace's session history. Returns None for cwd-less (unbound)
    sessions. Both fields are already recorded on ``sessions`` — this just picks
    the coarser identity for grouping/filtering.
    """
    root = (row.get("git_repo_root") or "").strip()
    if root:
        return root

    cwd = (row.get("cwd") or "").strip()
    return cwd or None


def _delegate_from_json(col: str = "model_config") -> str:
    return f"json_extract(COALESCE({col}, '{{}}'), '$._delegate_from')"


def _cwd_prefix_clause(cwd_prefix: str) -> Tuple[str, List[str]]:
    prefix = cwd_prefix.rstrip("/\\") or cwd_prefix
    return "(s.cwd = ? OR s.cwd LIKE ? OR s.cwd LIKE ?)", [prefix, f"{prefix}/%", f"{prefix}\\%"]


# A child session counts as a /branch (kept visible, never cascade-deleted) if
# it carries the stable marker OR the legacy end_reason heuristic holds.
_BRANCH_CHILD_SQL = (
    "json_extract(COALESCE({a}.model_config, '{{}}'), '$._branched_from') IS NOT NULL"
    " OR EXISTS (SELECT 1 FROM sessions p"
    "            WHERE p.id = {a}.parent_session_id"
    "            AND p.end_reason = 'branched'"
    "            AND {a}.started_at >= p.ended_at)"
)

_COMPRESSION_CHILD_SQL = (
    "EXISTS (SELECT 1 FROM sessions p"
    "        WHERE p.id = {a}.parent_session_id"
    "        AND p.end_reason = 'compression')"
)

# Rows that surface in pickers: roots + branch children (subagent runs and
# compression continuations stay hidden).
_LISTABLE_CHILD_SQL = f"(s.parent_session_id IS NULL OR {_BRANCH_CHILD_SQL.format(a='s')})"


def _ephemeral_child_sql(alias: str = "s") -> str:
    """Subagent runs (cascade-delete targets), not branches or compression tips."""
    branch = _BRANCH_CHILD_SQL.format(a=alias)
    compression = _COMPRESSION_CHILD_SQL.format(a=alias)
    return (
        f"({alias}.parent_session_id IS NOT NULL"
        f" AND NOT ({branch})"
        f" AND NOT ({compression}))"
    )


def _collect_delegate_child_ids(conn, parent_ids: List[str]) -> List[str]:
    """Delegate-subagent ids to cascade-delete with *parent_ids*.

    Only rows carrying the ``_delegate_from`` marker (set at creation, and
    backfilled by the v16 migration) — generic untagged children keep the
    orphan-don't-delete contract. Walks marker chains recursively so an
    orchestrator subagent's own delegate children go too (FK safety).
    """
    df = _delegate_from_json()
    seeds = {sid for sid in parent_ids if sid}
    # Seed the visited set with the parents themselves. A delegation marker
    # chain can loop back onto a parent — a cycle, or a parent that is also
    # another parent's delegate child when several ids are deleted at once —
    # and without this guard that parent would be collected as one of its own
    # descendants and cascade-deleted along with all of its messages. Callers
    # delete the parents separately, so parents must never appear in the
    # returned child set. (#49148)
    found: set[str] = set(seeds)
    frontier = list(seeds)
    while frontier:
        ph = ",".join("?" * len(frontier))
        cursor = conn.execute(
            f"SELECT id FROM sessions WHERE {df} IN ({ph}) "
            f"OR (parent_session_id IN ({ph}) AND {df} IS NOT NULL)",
            frontier + frontier,
        )
        frontier = [row["id"] for row in cursor.fetchall() if row["id"] not in found]
        found.update(frontier)
    # Return only the discovered children — never the parents themselves.
    return [sid for sid in found if sid not in seeds]


def _delete_delegate_children(conn, parent_ids: List[str]) -> List[str]:
    ids = _collect_delegate_child_ids(conn, parent_ids)
    if ids:
        ph = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM messages WHERE session_id IN ({ph})", ids)
        # FK safety: orphan any untagged stragglers pointing at a doomed row.
        conn.execute(
            f"UPDATE sessions SET parent_session_id = NULL "
            f"WHERE parent_session_id IN ({ph})",
            ids,
        )
        conn.execute(f"DELETE FROM sessions WHERE id IN ({ph})", ids)
    return ids

T = TypeVar("T")

DEFAULT_DB_PATH = get_hermes_home() / "state.db"

SCHEMA_VERSION = 23

# FTS storage-layout version, tracked INDEPENDENTLY of SCHEMA_VERSION in the
# state_meta key ``fts_storage_version``. The main schema version advances
# freely on open (so future migrations always land); the FTS *layout* only
# reaches the current version when a DB is either born fresh or explicitly
# optimized via ``hermes sessions optimize-storage``. A legacy DB sits at
# layout 0 (marker absent) with a working inline index until the user opts in.
#   1 = v23 external-content layout (content/tool_name/tool_calls,
#       tool-row-excluded trigram)
FTS_STORAGE_VERSION = 1

# Cap on user-controlled FTS5 query input before regex/sanitizer processing.
# Search queries do not need to be arbitrarily large, and bounding them keeps
# sanitizer/runtime behavior predictable under adversarial input.
MAX_FTS5_QUERY_CHARS = 2_048

# ---------------------------------------------------------------------------
# WAL-compatibility fallback
# ---------------------------------------------------------------------------
# SQLite's WAL mode requires shared-memory (mmap) coordination and fcntl
# byte-range locks that don't reliably work on network filesystems (NFS,
# SMB/CIFS, some FUSE mounts, WSL1).  Upstream documents this explicitly:
# https://www.sqlite.org/wal.html#sometimes_queries_return_sqlite_busy_in_wal_mode
#
# On those filesystems ``PRAGMA journal_mode=WAL`` raises
# ``sqlite3.OperationalError: locking protocol`` (SQLITE_PROTOCOL).  If we
# propagate that, every feature backed by state.db / kanban.db breaks
# silently — /resume, /title, /history, /branch, kanban dispatcher, etc.
#
# Instead, fall back to ``journal_mode=DELETE`` (the pre-WAL default) which
# works on NFS.  Concurrency drops — concurrent readers are blocked during
# a write — but the feature works.
#
# Separately, SQLite's WAL-reset bug can corrupt multi-process WAL databases
# on unfixed library builds (issue #69784).  See:
# https://sqlite.org/wal.html#walresetbug
# Fixed in 3.51.3+ with backports 3.50.7 and 3.44.6.  On vulnerable builds we
# refuse to *enable* WAL for fresh / non-WAL databases (prefer DELETE).  We do
# NOT live-downgrade an on-disk WAL database — other gateway/cron/worker
# connections may still hold it open, and flipping journal_mode under them is
# unsafe (same invariant as the NFS path below).
_WAL_INCOMPAT_MARKERS = (
    "locking protocol",       # SQLITE_PROTOCOL on NFS/SMB
    "not authorized",         # Some FUSE mounts block WAL pragma outright
)

# Last SessionDB() init error, per-process.  Surfaced in /resume and
# related slash-command error strings so users know WHY the DB is
# unavailable instead of getting a bare "Session database not available."
# Only SessionDB.__init__ writes to this; kanban_db.connect() failures
# do not update it (by design — kanban failures are reported via their
# own caller's error handling, not via /resume-style slash commands).
_last_init_error: Optional[str] = None
_last_init_error_lock = threading.Lock()

# Paths for which we've already logged a WAL-fallback WARNING.  Without
# this, kanban_db.connect() (called on every kanban operation — see
# hermes_cli/kanban_db.py for ~30 call sites) would re-log the same
# filesystem-incompat warning on every connection, filling errors.log.
_wal_fallback_warned_paths: set[str] = set()
_wal_fallback_warned_lock = threading.Lock()

# Dedup WARNING for the WAL-reset vulnerability fallback (issue #69784).
_wal_reset_bug_warned_paths: set[str] = set()
_wal_reset_bug_warned_lock = threading.Lock()

_FTS_TRIGGERS = (
    "messages_fts_insert",
    "messages_fts_delete",
    "messages_fts_update",
    "messages_fts_trigram_insert",
    "messages_fts_trigram_delete",
    "messages_fts_trigram_update",
)


def _set_last_init_error(msg: Optional[str]) -> None:
    """Record (or clear) the most recent state.db init failure.

    Thread-safe via _last_init_error_lock.  Callers pass a message to
    record a failure or None to clear.  SessionDB.__init__ only calls
    this to SET on failure — it deliberately does NOT clear on success,
    because in a multi-threaded caller (e.g. gateway / web_server per-
    request SessionDB() instantiation), a concurrent successful open
    racing past a different thread's failure would erase the cause
    string that thread's /resume handler is about to format.  Explicit
    clears (e.g. test fixtures) are still supported by passing None.
    """
    global _last_init_error
    with _last_init_error_lock:
        _last_init_error = msg


def get_last_init_error() -> Optional[str]:
    """Return the most recent state.db init failure, if any.

    Slash-command handlers (``/resume``, ``/title``, ``/history``, ``/branch``)
    call this to surface the underlying cause in their error messages when
    ``_session_db is None``.  Returns ``None`` if SessionDB initialized
    successfully (or hasn't been attempted).
    """
    return _last_init_error


# Distinctive opening shared by both background-review harness prompts
# (_SKILL_REVIEW_PROMPT and _MEMORY_REVIEW_PROMPT in agent/background_review.py).
# Matched case-sensitively against the leading content of a user/system message.
_REVIEW_HARNESS_PREFIXES = (
    "Review the conversation above and update the skill library",
    "Review the conversation above and consider saving to memory",
)


def _is_background_review_harness_message(msg: Dict[str, Any]) -> bool:
    """True when ``msg`` is a persisted background-review harness prompt.

    These are user/system turns the forked skill/memory review agent wrote into
    a real session in older builds (before the ``_persist_disabled`` isolation
    fix). They instruct the agent to act as the curator under a hard tool
    restriction, so replaying them as live history hijacks the session.
    """
    if not isinstance(msg, dict):
        return False
    if msg.get("role") not in {"user", "system"}:
        return False
    content = msg.get("content")
    if not isinstance(content, str):
        return False
    head = content.lstrip()
    return any(head.startswith(p) for p in _REVIEW_HARNESS_PREFIXES)


def _strip_background_review_harness(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Drop background-review harness messages and the curator-mode assistant
    reply that immediately followed each one.

    Walk the list once; when a harness user/system message is found, skip it and
    also skip the next message if it is the assistant turn that answered it.
    Everything else passes through untouched and in order.
    """
    if not messages:
        return messages
    out: List[Dict[str, Any]] = []
    skip_next_assistant = False
    for msg in messages:
        if _is_background_review_harness_message(msg):
            skip_next_assistant = True
            continue
        if skip_next_assistant:
            skip_next_assistant = False
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                # The curator-mode reply to the harness prompt — drop it.
                continue
        out.append(msg)
    return out


def format_session_db_unavailable(prefix: str = "Session database not available") -> str:
    """Format a user-facing 'session DB unavailable' message with cause.

    When ``SessionDB()`` init fails, callers set ``_session_db = None`` and
    several slash commands (/resume, /title, /history, /branch) previously
    responded with a bare ``"Session database not available."`` — no
    indication of WHY.  This helper includes the captured cause (typically
    ``"locking protocol"`` from NFS/SMB) and points users at the known
    culprit so they can fix it themselves.

    Example output:
        Session database not available: locking protocol (state.db may be
        on NFS/SMB — see https://www.sqlite.org/wal.html).
    """
    cause = get_last_init_error()
    if not cause:
        return f"{prefix}."
    hint = ""
    if any(marker in cause.lower() for marker in _WAL_INCOMPAT_MARKERS):
        hint = " (state.db may be on NFS/SMB/FUSE — see https://www.sqlite.org/wal.html)"
    return f"{prefix}: {cause}{hint}."


def _on_disk_journal_mode(conn: sqlite3.Connection) -> Optional[str]:
    """Read the journal mode from the SQLite DB header on disk.

    Returns the mode string (e.g. ``"wal"``, ``"delete"``), or ``None``
    if the value cannot be determined (new DB, or PRAGMA read failed).
    """
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
    except sqlite3.OperationalError:
        return None
    if row is None:
        return None
    mode = row[0]
    if isinstance(mode, bytes):  # defensive: sqlite3 occasionally returns bytes
        try:
            mode = mode.decode("ascii")
        except UnicodeDecodeError:
            return None
    return str(mode).strip().lower() if mode is not None else None


def _apply_macos_checkpoint_barrier(conn: sqlite3.Connection) -> None:
    """Enable ``PRAGMA checkpoint_fullfsync`` on macOS (no-op elsewhere).

    On Darwin, ``synchronous=FULL`` (the WAL default) issues a plain
    ``fsync()``, which Apple documents does *not* guarantee that data
    has reached stable storage or that writes are not reordered — see
    the ``fsync(2)`` man page.  SQLite's WAL corruption-safety guarantee
    assumes the OS honors the fsync write barrier; macOS does not unless
    the app uses ``F_FULLFSYNC``.

    During a launchd *system* shutdown/reboot the OS page cache is
    dropped (effectively a power-loss event for in-flight pages), so a
    WAL checkpoint whose ``fsync()`` "reported" durable may never have
    hit the platter — corrupting ``state.db`` with a malformed image.
    This is the trigger in issue #30636 ("SIGTERM during launchd
    shutdown under high load"), distinct from a plain in-session kill
    (which the page cache survives and SQLite recovers from).

    ``checkpoint_fullfsync=1`` forces an ``F_FULLFSYNC`` barrier only at
    checkpoint boundaries — where WAL frames land in the main DB — so the
    cost amortizes to roughly +0.1 ms/commit (vs ~+4 ms for the broader
    ``fullfsync=1`` that flushes on every commit's WAL sync).  Guarded by
    ``sys.platform == "darwin"`` because ``F_FULLFSYNC`` is macOS-only;
    on other platforms the PRAGMA is a no-op, so we skip it entirely.

    Best-effort: never raises.
    """
    if sys.platform != "darwin":
        return
    try:
        conn.execute("PRAGMA checkpoint_fullfsync=1")
    except sqlite3.OperationalError:
        pass


def _enforce_macos_synchronous_full(conn: sqlite3.Connection) -> None:
    """Enforce ``PRAGMA synchronous=FULL`` on macOS to prevent btree corruption.

    On Darwin, the default ``synchronous=NORMAL`` only calls ``fsync()``,
    which Apple's fsync(2) man page explicitly states does *not* guarantee
    data-on-platter or write-ordering. During a WAL checkpoint race with
    process termination (e.g., launchd shutdown), this can leave the main
    DB with half-written btree pages → ``btreeInitPage error 11``.

    WAL mode's durability guarantee assumes the OS honors fsync barriers;
    macOS does not unless we explicitly set ``synchronous=FULL``, which issues
    a real ``fsync()`` on every transaction commit.  The ``F_FULLFSYNC``
    barrier at checkpoint boundaries is handled separately by
    :func:`_apply_macos_checkpoint_barrier`.

    This function is called after any successful WAL activation (either
    from ``apply_wal_with_fallback()`` setting a fresh WAL or when probing
    an existing WAL mode). It ensures macOS connections always use FULL
    synchronous mode, even if a prior connection set ``synchronous=NORMAL``.

    Best-effort: never raises.
    """
    if sys.platform != "darwin":
        return
    try:
        conn.execute("PRAGMA synchronous=FULL")
    except sqlite3.OperationalError:
        pass


def is_sqlite_wal_reset_vulnerable(
    version_info: Optional[tuple] = None,
) -> bool:
    """Return True when the linked SQLite library has the WAL-reset bug.

    Upstream documents the bug in versions 3.7.0 through 3.51.2, fixed in
    3.51.3+, with backports 3.50.7 and 3.44.6:
    https://sqlite.org/wal.html#walresetbug

    Pre-WAL libraries (< 3.7.0) cannot hit the race and are treated as safe.
    """
    info = version_info if version_info is not None else sqlite3.sqlite_version_info
    if info < (3, 7, 0):
        return False
    if info >= (3, 51, 3):
        return False
    # Backports of the same fix on older release lines.
    if (3, 50, 7) <= info < (3, 51, 0):
        return False
    if (3, 44, 6) <= info < (3, 45, 0):
        return False
    return True


def sqlite_source_id() -> str:
    """Return ``sqlite_source_id()``, or an empty string when unavailable."""
    try:
        conn = sqlite3.connect(":memory:")
        try:
            row = conn.execute("SELECT sqlite_source_id()").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return ""
    if not row or row[0] is None:
        return ""
    return str(row[0])


def apply_wal_with_fallback(
    conn: sqlite3.Connection,
    *,
    db_label: str = "state.db",
) -> str:
    """Set ``journal_mode=WAL`` on ``conn``, falling back to DELETE on failure.

    Returns the journal mode actually set (``"wal"`` or ``"delete"``).

    On WAL-incompatible filesystems (NFS, SMB, some FUSE), SQLite raises
    ``OperationalError("locking protocol")`` when setting WAL.  We fall
    back to DELETE mode — the pre-WAL default, which works on NFS — and
    log one WARNING explaining why.

    On SQLite builds that still contain the WAL-reset corruption bug
    (issue #69784), refuse to enable WAL on fresh / non-WAL databases
    (prefer DELETE).  If the on-disk DB is already WAL, keep WAL and warn
    — never live-downgrade under possible concurrent openers.

    The WARNING is deduplicated per ``db_label``: repeated connections
    to the same underlying DB (e.g. kanban_db.connect() which is called
    on every kanban operation) log once per process, not once per call.
    Different db_labels log independently, so state.db and kanban.db
    each get one warning on the same NFS mount.

    Shared by :class:`SessionDB` and ``hermes_cli.kanban_db.connect`` so
    both databases get identical fallback behavior.

    Never downgrades to DELETE if the on-disk DB header reports WAL — see
    _on_disk_journal_mode.  That holds for both the NFS path and the
    WAL-reset vulnerability path.
    """
    # Vulnerable SQLite: do not enable WAL on new/non-WAL files.
    if is_sqlite_wal_reset_vulnerable():
        return _apply_delete_for_wal_reset_bug(conn, db_label=db_label)

    # Read-only probe — no flock, no checkpoint, no WAL/SHM unlink.
    # Skipping the set-pragma prevents WAL-init from unlinking files other connections hold open.
    try:
        current_mode = conn.execute("PRAGMA journal_mode").fetchone()
        if current_mode and current_mode[0] == "wal":
            _apply_macos_checkpoint_barrier(conn)
            _enforce_macos_synchronous_full(conn)
            return "wal"
    except sqlite3.OperationalError:
        pass

    try:
        conn.execute("PRAGMA journal_mode=WAL")
        _apply_macos_checkpoint_barrier(conn)
        _enforce_macos_synchronous_full(conn)
        return "wal"
    except sqlite3.OperationalError as exc:
        msg = str(exc).lower()
        if not any(marker in msg for marker in _WAL_INCOMPAT_MARKERS):
            # Unrelated OperationalError — don't silently swallow.
            raise
        # Don't downgrade if another process already set WAL on disk.
        existing = _on_disk_journal_mode(conn)
        if existing == "wal":
            raise
        _log_wal_fallback_once(db_label, exc)
        conn.execute("PRAGMA journal_mode=DELETE")
        return "delete"


def _apply_delete_for_wal_reset_bug(
    conn: sqlite3.Connection,
    *,
    db_label: str,
) -> str:
    """Avoid enabling WAL when the linked SQLite has the WAL-reset bug.

    - Already-WAL on disk: leave WAL alone (no live downgrade) and warn.
    - Otherwise: set DELETE and warn.
    """
    current = ""
    try:
        row = conn.execute("PRAGMA journal_mode").fetchone()
        if row and row[0] is not None:
            current = str(row[0]).strip().lower()
    except sqlite3.OperationalError:
        current = ""

    if current == "wal":
        # Do not TRUNCATE / journal_mode=DELETE while other processes may
        # still hold this WAL DB open — same safety rule as the NFS path.
        _log_wal_reset_bug_once(db_label, kept_wal=True)
        _apply_macos_checkpoint_barrier(conn)
        _enforce_macos_synchronous_full(conn)
        return "wal"

    try:
        conn.execute("PRAGMA journal_mode=DELETE")
    except sqlite3.OperationalError:
        # Best-effort: DELETE is usually already the default for new files.
        pass
    _log_wal_reset_bug_once(db_label, kept_wal=False)
    return "delete"


def _log_wal_reset_bug_once(
    db_label: str,
    *,
    kept_wal: bool,
) -> None:
    """Log once per (process, db_label) about the WAL-reset vulnerability path."""
    with _wal_reset_bug_warned_lock:
        if db_label in _wal_reset_bug_warned_paths:
            return
        _wal_reset_bug_warned_paths.add(db_label)
    action = (
        "is already in WAL mode — leaving WAL in place (no live "
        "downgrade under concurrent openers)"
        if kept_wal
        else "using journal_mode=DELETE instead of enabling WAL"
    )
    logger.warning(
        "%s: linked SQLite %s is vulnerable to the WAL-reset corruption "
        "bug (https://sqlite.org/wal.html#walresetbug) — %s. "
        "Upgrade to SQLite 3.51.3+ (or backports 3.50.7 / 3.44.6); "
        "`hermes update` alone may not change python-build-standalone's "
        "embedded SQLite. See `hermes doctor`. This warning fires once "
        "per process per database.",
        db_label,
        sqlite3.sqlite_version,
        action,
    )


def _log_wal_fallback_once(db_label: str, exc: Exception) -> None:
    """Log a single WARNING per (process, db_label) about WAL fallback.

    Without this dedup, NFS users running kanban (which opens a fresh
    connection on every operation — see hermes_cli/kanban_db.py) would
    fill errors.log with hundreds of identical warnings per hour.
    """
    with _wal_fallback_warned_lock:
        if db_label in _wal_fallback_warned_paths:
            return
        _wal_fallback_warned_paths.add(db_label)
    logger.warning(
        "%s: WAL journal_mode unsupported on this filesystem (%s) — "
        "falling back to journal_mode=DELETE (slower rollback-journal "
        "mode; reduces concurrency but works on NFS/SMB/FUSE). See "
        "https://www.sqlite.org/wal.html for details. This warning "
        "fires once per process per database.",
        db_label,
        exc,
    )

# ---------------------------------------------------------------------------
# Malformed-schema recovery
# ---------------------------------------------------------------------------
# A distinct, nastier failure class than a malformed FTS *inverted index*:
# the ``sqlite_master`` schema table itself becomes inconsistent — most
# commonly a DUPLICATE object definition, e.g. two ``CREATE VIRTUAL TABLE
# messages_fts`` rows.  SQLite parses the entire schema while preparing the
# FIRST statement on a connection, so on this class *every* statement raises
# before it runs — including ``PRAGMA journal_mode`` (which is why this trips
# in ``apply_wal_with_fallback`` during ``SessionDB.__init__``, long before
# ``_init_schema`` is reached) and even ``PRAGMA integrity_check`` and a plain
# ``DROP TABLE``.  The only operations that still work are
# ``PRAGMA writable_schema=ON`` plus direct ``sqlite_master`` surgery.
#
# Symptom users hit (Desktop/Dashboard show "no sessions" while 200+ JSON
# files sit on disk):
#   sqlite3.DatabaseError: malformed database schema (messages_fts) -
#   table messages_fts already exists
#
# The canonical ``sessions`` / ``messages`` data is intact in these cases —
# only the derived schema is broken — so recovery preserves all transcripts
# and merely rebuilds the FTS layer.
_MALFORMED_SCHEMA_MARKERS = (
    "malformed database schema",
    "database disk image is malformed",
)

# Process-global guard so auto-repair is attempted at most once per DB path
# per process (prevents repair loops and serialises concurrent web_server /
# gateway opens against the same malformed file).
_repair_attempted_paths: set[str] = set()
_repair_attempt_lock = threading.Lock()


def is_malformed_db_error(exc: BaseException) -> bool:
    """True if *exc* is a SQLite 'malformed schema / disk image' error.

    These are the corruption classes where the schema fails to parse, so
    targeted ``sqlite_master`` surgery (not an ordinary FTS rebuild) is the
    only recovery path.
    """
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    return any(marker in str(exc).lower() for marker in _MALFORMED_SCHEMA_MARKERS)


def _claim_repair_attempt(db_path: Path) -> bool:
    """Claim the one-shot repair attempt for *db_path* in this process.

    Returns True for the first caller, False afterwards. Keeps a malformed
    DB from triggering an unbounded repair/reopen loop and stops concurrent
    callers from racing surgery on the same file.
    """
    key = str(db_path)
    with _repair_attempt_lock:
        if key in _repair_attempted_paths:
            return False
        _repair_attempted_paths.add(key)
        return True


def _backup_db_file(db_path: Path) -> Optional[Path]:
    """Copy a (possibly malformed) DB file to a timestamped backup beside it.

    Raw file copy on purpose: the DB won't open cleanly, so we preserve the
    bytes exactly for forensics / manual restore. WAL and SHM sidecars are
    copied too when present. Returns the backup path, or None on failure.
    """
    import datetime
    import shutil

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.malformed-backup-{stamp}")
    try:
        shutil.copy2(db_path, backup_path)
        for suffix in ("-wal", "-shm"):
            sidecar = db_path.with_name(db_path.name + suffix)
            if sidecar.exists():
                shutil.copy2(sidecar, backup_path.with_name(backup_path.name + suffix))
        return backup_path
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Could not back up malformed DB %s: %s", db_path, exc)
        return None


def _db_opens_cleanly(db_path: Path) -> Optional[str]:
    """Probe a DB on a fresh connection. Returns None if healthy, else a reason.

    Runs the same first-statement (``PRAGMA journal_mode``) that trips the
    malformed-schema parse, then ``PRAGMA integrity_check`` and a canonical
    ``sessions`` read, and finally a rolled-back ``messages`` write so that
    FTS5 index corruption — which leaves base-table reads and
    ``integrity_check`` passing while every ``INSERT INTO messages`` fails
    through the FTS triggers — is reported as unhealthy rather than slipping
    past as a false "ok" (#50502).
    """
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        # Best-effort tokenizer load: a DB carrying the messages_fts_cjk
        # index needs the cjk_unicode61 extension before any statement can
        # touch that table — including the trigger-driven write probe below.
        # Without it, this probe sees the DB exactly as a tokenizer-less
        # SessionDB open would (which drops the cjk triggers to keep writes
        # working), so tokenizer absence must never classify as corruption.
        load_fts5_cjk_extension(conn)
        conn.execute("PRAGMA journal_mode").fetchone()
        rows = conn.execute("PRAGMA integrity_check").fetchall()
        problems = [str(r[0]) for r in rows if r and str(r[0]).lower() != "ok"]
        if problems:
            return "; ".join(problems[:3])
        conn.execute("SELECT COUNT(*) FROM sessions").fetchone()

        # FTS5 read probe: run a representative MATCH query against the
        # messages_fts* virtual tables. The FTS *write* probe below catches
        # the corruption class where base tables read fine but writes fail
        # through the triggers (#50502). It does NOT catch partial FTS5
        # index corruption — bad shadow-table segments where reads still
        # parse but MATCH / snippet / rank queries error out with
        # "database disk image is malformed" (a `sqlite3.DatabaseError`,
        # not `OperationalError`). session_search, /resume title resolution,
        # and any feature relying on FTS5 discovery then break silently
        # because the official repair tool's check-only path reports the
        # DB as healthy. #66724.
        # Catch the full sqlite3 exception hierarchy (not just
        # OperationalError) so the malformed-shadow-table class is reported
        # rather than letting it crash the caller.
        for fts_table in ("messages_fts", "messages_fts_trigram", "messages_fts_cjk"):
            try:
                # No-op queries against the actual FTS5 APIs the search
                # tools use. The trigram table is included because it backs
                # the title-resolution path; either corruption mode would
                # break session recall without this probe. MATCH '""' is
                # the empty phrase-token probe — FTS5 rejects MATCH ''
                # outright ("fts5: syntax error"), but a quoted empty
                # phrase parses, scans zero rows, and exercises the same
                # shadow-table read path the search tools use.
                conn.execute(
                    f"SELECT 1 FROM {fts_table} WHERE {fts_table} MATCH '\"\"' LIMIT 1"
                ).fetchone()
            except sqlite3.OperationalError as exc:
                # Use the canonical capability classifier instead of a
                # hand-rolled substring check. On SQLite builds without the
                # fts5 module, the legacy messages_fts table may exist on
                # disk (from a prior build that had FTS5) and MATCH queries
                # against it raise OperationalError("no such module: fts5");
                # the substring check below would misclassify that as
                # corruption and send the DB into the repair path, whose
                # final fallback deletes the messages_fts% schema
                # (hermes_state.py:645-723). The supported degraded-runtime
                # path (SessionDB._is_fts5_unavailable_error + the
                # regression suite in tests/test_hermes_state.py:600-632)
                # treats both "no such module: fts5" and
                # "no such tokenizer: trigram" as the capability error.
                if SessionDB._is_fts5_unavailable_error(exc):
                    # Degraded runtime — not the corruption class we probe.
                    continue
                msg = str(exc).lower()
                if "no such table" in msg or "no such column" in msg:
                    # FTS5 not built yet (brand new file mid-init) — not the
                    # corruption class we probe.
                    continue
                return f"fts5 read probe failed on {fts_table}: {exc}"
            except sqlite3.DatabaseError as exc:
                # This is the corruption class #66724 actually wants caught:
                # partial shadow-table damage where MATCH / snippet / rank
                # queries raise DatabaseError("database disk image is malformed")
                # while reads of the FTS5 table itself parse fine.
                return f"fts5 read probe failed on {fts_table}: {exc}"

        # FTS write probe: drive a row through the messages_fts* triggers in a
        # transaction that is always rolled back, so a corrupt FTS index that
        # rejects writes is caught even though reads look healthy. The probe is
        # best-effort — if the messages/sessions tables don't exist yet (brand
        # new file mid-init) the OperationalError is treated as "not yet a
        # populated DB", not corruption.
        probe_session_id = f"_hermes_fts_health_probe_{time.time_ns()}"
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO sessions (id, source, started_at) VALUES (?, ?, ?)",
                (probe_session_id, "_health_probe", time.time()),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (probe_session_id, "user", "_fts_health_probe", time.time()),
            )
            conn.execute("ROLLBACK")
        except sqlite3.OperationalError as exc:
            # Missing tables / FTS disabled — not the corruption class we probe.
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            msg = str(exc).lower()
            if "no such table" in msg or "no such column" in msg:
                return None
            if "no such tokenizer: cjk_unicode61" in msg:
                # This probe process couldn't load the cjk extension while
                # the DB carries the cjk index — capability gap, not
                # corruption. A tokenizer-capable SessionDB serves it fine;
                # a tokenizer-less one self-heals by dropping the triggers.
                return None
            return str(exc)
        return None
    except sqlite3.DatabaseError as exc:
        return str(exc)
    finally:
        conn.close()


def repair_state_db_schema(db_path: Path, *, backup: bool = True) -> Dict[str, Any]:
    """Repair a state.db whose ``sqlite_master`` schema is malformed or whose
    FTS indexes reject writes.

    Handles two corruption classes: the "duplicate object definition" /
    malformed-schema class where even ``PRAGMA`` statements fail, and the FTS
    write-corruption class (#50502) where base tables read fine and
    ``integrity_check`` passes but writes fail through the ``messages_fts*``
    triggers. Tries least-destructive recovery first and escalates:

      1. **Rebuild FTS indexes in place** via the FTS5 ``'rebuild'`` command,
         which rewrites the internal b-tree segments from the canonical
         ``messages`` rows without dropping or recreating anything. Fixes the
         FTS write-corruption class while preserving the schema intact.
      2. **De-duplicate** ``sqlite_master`` (keep the lowest rowid per
         ``type``/``name``). Fixes the canonical "table X already exists"
         case and PRESERVES the existing FTS index intact.
      3. **Drop the FTS schema** (every ``messages_fts*`` object) + ``VACUUM``.
         The next ``SessionDB()`` open rebuilds the FTS indexes from the
         canonical ``messages`` table.

    Canonical ``sessions`` / ``messages`` rows are never modified. A
    timestamped raw backup is taken first unless ``backup=False``.

    Returns a report dict: ``{repaired: bool, strategy: str|None,
    backup_path: str|None, error: str|None}``.
    """
    report: Dict[str, Any] = {
        "repaired": False,
        "strategy": None,
        "backup_path": None,
        "error": None,
    }

    db_path = Path(db_path)
    if not db_path.exists():
        report["error"] = f"{db_path} does not exist"
        return report

    if _db_opens_cleanly(db_path) is None:
        report["repaired"] = True
        report["strategy"] = "already_healthy"
        return report

    if backup:
        bpath = _backup_db_file(db_path)
        report["backup_path"] = str(bpath) if bpath else None

    # ── Strategy 0: rebuild FTS indexes in place (FTS write-corruption) ──
    # The FTS5 'rebuild' command rewrites the internal index from the canonical
    # content table. This is the recommended, least-destructive recovery for a
    # corrupt FTS index that rejects message writes while reads still succeed.
    try:
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            # The cjk index can only be rebuilt with its tokenizer loaded;
            # best-effort (a tokenizer-less host skips it at the probe below).
            load_fts5_cjk_extension(conn)
            for table_name in (
                "messages_fts", "messages_fts_trigram", "messages_fts_cjk"
            ):
                try:
                    conn.execute(
                        f"INSERT INTO {table_name}({table_name}) VALUES('rebuild')"
                    )
                except sqlite3.OperationalError:
                    # Table absent (FTS disabled / trigram off / cjk not
                    # present or tokenizer unavailable) — skip it.
                    continue
        finally:
            conn.close()
        if _db_opens_cleanly(db_path) is None:
            report["repaired"] = True
            report["strategy"] = "rebuild_fts"
            logger.warning(
                "state.db FTS indexes rebuilt in place (schema preserved): %s",
                db_path,
            )
            return report
    except sqlite3.DatabaseError as exc:
        logger.warning("state.db FTS in-place rebuild pass failed: %s", exc)

    # ── Strategy 0.5: rebuild stale B-tree indexes (#63386) ──
    # PRAGMA integrity_check can report "wrong # of entries in index" when a
    # B-tree index (e.g. idx_sessions_handoff_state) falls out of sync with its
    # base table. REINDEX rewrites the index b-tree from the canonical table
    # rows using the existing index definition, fixing the mismatch without
    # touching data or FTS schema.
    try:
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            conn.execute("REINDEX")
            conn.commit()
        finally:
            conn.close()
        if _db_opens_cleanly(db_path) is None:
            report["repaired"] = True
            report["strategy"] = "reindex_btree"
            logger.warning(
                "state.db B-tree indexes rebuilt via REINDEX: %s", db_path
            )
            return report
    except sqlite3.DatabaseError as exc:
        logger.warning("state.db REINDEX pass failed: %s", exc)

    # ── Strategy 1: de-duplicate sqlite_master (keeps FTS index) ──
    try:
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            conn.execute("PRAGMA writable_schema=ON")
            dupes = conn.execute(
                "SELECT type, name, COUNT(*) AS c, MIN(rowid) AS keep "
                "FROM sqlite_master GROUP BY type, name HAVING c > 1"
            ).fetchall()
            for type_, name, _count, keep in dupes:
                conn.execute(
                    "DELETE FROM sqlite_master "
                    "WHERE type IS ? AND name IS ? AND rowid <> ?",
                    (type_, name, keep),
                )
            conn.execute("PRAGMA writable_schema=OFF")
            conn.commit()
        finally:
            conn.close()
        if _db_opens_cleanly(db_path) is None:
            report["repaired"] = True
            report["strategy"] = "dedup_schema"
            logger.warning(
                "state.db schema repaired by de-duplicating sqlite_master "
                "(FTS index preserved): %s", db_path
            )
            return report
    except sqlite3.DatabaseError as exc:
        logger.warning("state.db dedup repair pass failed: %s", exc)

    # ── Strategy 2: drop all FTS schema, VACUUM, rebuild on next open ──
    try:
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            conn.execute("PRAGMA writable_schema=ON")
            conn.execute("DELETE FROM sqlite_master WHERE name LIKE 'messages_fts%'")
            conn.execute("PRAGMA writable_schema=OFF")
            conn.commit()
            conn.execute("VACUUM")
        finally:
            conn.close()
        reason = _db_opens_cleanly(db_path)
        if reason is None:
            report["repaired"] = True
            report["strategy"] = "drop_fts_rebuild"
            logger.warning(
                "state.db schema repaired by dropping FTS schema; indexes "
                "will rebuild from messages on next open: %s", db_path
            )
            return report
        report["error"] = reason
    except sqlite3.DatabaseError as exc:
        report["error"] = str(exc)

    if not report["repaired"]:
        logger.error(
            "state.db schema repair could not recover %s automatically "
            "(backup: %s); manual restore from backup may be required.",
            db_path, report["backup_path"],
        )
    return report


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    session_key TEXT,
    chat_id TEXT,
    chat_type TEXT,
    thread_id TEXT,
    display_name TEXT,
    origin_json TEXT,
    expiry_finalized INTEGER DEFAULT 0,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    cwd TEXT,
    git_branch TEXT,
    git_repo_root TEXT,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT,
    api_call_count INTEGER DEFAULT 0,
    handoff_state TEXT,
    handoff_platform TEXT,
    handoff_error TEXT,
    compression_failure_cooldown_until REAL,
    compression_failure_error TEXT,
    compression_fallback_streak INTEGER NOT NULL DEFAULT 0,
    compression_ineffective_count INTEGER NOT NULL DEFAULT 0,
    profile_name TEXT,
    rewind_count INTEGER NOT NULL DEFAULT 0,
    archived INTEGER NOT NULL DEFAULT 0,
    pinned INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    effect_disposition TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_content TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT,
    codex_message_items TEXT,
    platform_message_id TEXT,
    observed INTEGER DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1,
    compacted INTEGER NOT NULL DEFAULT 0,
    api_content TEXT,
    display_kind TEXT,
    display_metadata TEXT
);

CREATE TABLE IF NOT EXISTS session_model_usage (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    billing_provider TEXT NOT NULL DEFAULT '',
    billing_base_url TEXT NOT NULL DEFAULT '',
    billing_mode TEXT NOT NULL DEFAULT '',
    task TEXT NOT NULL DEFAULT '',
    api_call_count INTEGER NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd REAL NOT NULL DEFAULT 0,
    actual_cost_usd REAL NOT NULL DEFAULT 0,
    cost_status TEXT,
    cost_source TEXT,
    first_seen REAL,
    last_seen REAL,
    PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task)
);

CREATE TABLE IF NOT EXISTS state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS gateway_routing (
    scope TEXT NOT NULL DEFAULT '',
    session_key TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (scope, session_key)
);

CREATE TABLE IF NOT EXISTS compression_locks (
    session_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS async_delegations (
    delegation_id TEXT PRIMARY KEY,
    origin_session TEXT NOT NULL,
    origin_ui_session_id TEXT NOT NULL DEFAULT '',
    parent_session_id TEXT,
    state TEXT NOT NULL,
    dispatched_at REAL NOT NULL,
    completed_at REAL,
    updated_at REAL NOT NULL,
    event_json TEXT,
    result_json TEXT,
    delivery_state TEXT NOT NULL DEFAULT 'pending',
    delivery_attempts INTEGER NOT NULL DEFAULT 0,
    delivered_at REAL,
    owner_pid INTEGER,
    owner_started_at INTEGER,
    task_json TEXT,
    delivery_claim TEXT,
    delivery_claimed_at REAL
);

CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source);
CREATE INDEX IF NOT EXISTS idx_sessions_source_id ON sessions(source, id);
CREATE INDEX IF NOT EXISTS idx_sessions_parent ON sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_compression_locks_expires ON compression_locks(expires_at);
CREATE INDEX IF NOT EXISTS idx_session_model_usage_session ON session_model_usage(session_id);
CREATE INDEX IF NOT EXISTS idx_session_model_usage_model ON session_model_usage(model);
CREATE INDEX IF NOT EXISTS idx_async_delegations_delivery
    ON async_delegations(delivery_state, completed_at);
"""

# Indexes that reference columns added in later schema versions must be
# created AFTER _reconcile_columns() has had a chance to ADD them on
# existing databases. SCHEMA_SQL above is run by sqlite executescript
# which would otherwise fail on legacy DBs ("no such column: active").
DEFERRED_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_messages_session_active
    ON messages(session_id, active, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_active_null
    ON messages(active) WHERE active IS NULL;
CREATE INDEX IF NOT EXISTS idx_sessions_session_key
    ON sessions(session_key, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_gateway_peer
    ON sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_handoff_state
    ON sessions(handoff_state, started_at);
"""

# ── Deferred FTS rebuild bookkeeping (schema v23) ──
# While a background index rebuild is pending, two state_meta keys define
# which message rows are currently IN the FTS indexes:
#
#   fts_rebuild_high_water  H — MAX(messages.id) at the moment the old
#                                indexes were dropped
#   fts_rebuild_progress    P — highest id the chunked backfill has indexed
#
# A row is indexed iff  id <= P  (backfilled)  OR  id > H  (inserted after
# the drop; ids are AUTOINCREMENT so new rows are always > H and the insert
# triggers index them live).  Rows in (P, H] are not yet indexed.
#
# Every trigger below gates on that same predicate: firing an FTS5
# external-content 'delete' for a row that is NOT in the index corrupts the
# index, and skipping it for a row that IS indexed leaves a stale entry.
# When no rebuild is pending both keys are absent and COALESCE turns the
# predicate into a tautology (id > -1 OR id <= -1), i.e. normal operation.
# The two state_meta PK probes per write are negligible next to the FTS
# insert itself.
FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages
WHEN (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                         WHERE key = 'fts_rebuild_high_water'), -1)
   OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                          WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages
WHEN (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                         WHERE key = 'fts_rebuild_high_water'), -1)
   OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                          WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages
WHEN (old.content IS NOT new.content
    OR old.tool_name IS NOT new.tool_name
    OR old.tool_calls IS NOT new.tool_calls)
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;
"""

# Trigram FTS5 table for CJK substring search.  The default unicode61
# tokenizer splits CJK characters into individual tokens, breaking phrase
# matching.  The trigram tokenizer creates overlapping 3-byte sequences so
# substring queries work natively for any script (CJK, Thai, etc.).
#
# The trigram index is the most expensive index in state.db (~2.6x the size
# of the text it covers), and ``role='tool'`` rows are ~90% of message bytes
# while being almost entirely machine noise (base64 payloads, file dumps,
# delegation transcripts).  The index therefore reads through
# ``messages_fts_trigram_src``, a view that excludes tool rows — they stay
# fully stored in ``messages`` and fully searchable via the standard
# ``messages_fts`` index; they just don't get trigram (CJK substring)
# treatment.  ``search_messages`` routes CJK queries that filter on
# ``role='tool'`` to the LIKE fallback for the same reason.
FTS_TRIGRAM_SQL = """
CREATE VIEW IF NOT EXISTS messages_fts_trigram_src AS
    SELECT id, role, content, tool_name, tool_calls
    FROM messages
    WHERE role <> 'tool';

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages_fts_trigram_src',
    content_rowid='id',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages
WHEN new.role <> 'tool'
   AND (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages
WHEN old.role <> 'tool'
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update AFTER UPDATE ON messages
WHEN (old.content IS NOT new.content
    OR old.tool_name IS NOT new.tool_name
    OR old.tool_calls IS NOT new.tool_calls
    OR old.role IS NOT new.role)
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name, tool_calls)
    SELECT 'delete', old.id, old.content, old.tool_name, old.tool_calls
    WHERE old.role <> 'tool';
    INSERT INTO messages_fts_trigram(rowid, content, tool_name, tool_calls)
    SELECT new.id, new.content, new.tool_name, new.tool_calls
    WHERE new.role <> 'tool';
END;
"""

# ── CJK-bigram FTS index (replaces the trigram index when available) ────
#
# The trigram tokenizer needs >=3 chars per query term, so 1-2 char CJK
# terms (ubiquitous in Korean/Chinese: 일본, 구글, 项目, ...) fall through
# to a LIKE full-table scan — measured 3-6s CPU per query on multi-GB
# installs and the dominant base cost of session_search on CJK workloads.
#
# ``cjk_unicode61`` (native/fts5_cjk/, a ~250-line loadable FTS5 tokenizer
# with no dependencies) wraps unicode61: maximal CJK runs are re-emitted as
# overlapping character bigrams (Lucene CJKAnalyzer semantics), everything
# else passes through unchanged. FTS5 phrase semantics turn a query term's
# consecutive bigrams into exact substring matching down to 2 chars at
# index speed. Contributed by Soju06 (PR #65544).
#
# Same v23 storage discipline as the trigram table it replaces:
# external-content over a tool-row-excluding view (zero inline text
# copies; tool rows stay searchable via ``messages_fts``), triggers gated
# on a DEDICATED marker pair (``fts_cjk_rebuild_high_water`` /
# ``fts_cjk_rebuild_progress``) so a cjk-only backfill — e.g. the
# trigram→cjk upgrade on an already-optimized DB — never gates the
# complete ``messages_fts`` index's triggers.
#
# The table exists ONLY when the loadable tokenizer is available
# (``~/.hermes/lib/libfts5_cjk.so``, built by ``native/fts5_cjk/build.sh``).
# A process that cannot load it self-heals by dropping the cjk triggers
# (message writes keep working; the index goes stale and is rebuilt by the
# next ``hermes sessions optimize-storage`` on a capable host).
#
# Split DDL: the table/view part is safe to ensure any time; the triggers
# are created ONLY while the index is complete-or-marker-gated. A stale
# index (trigger gap of unknown extent) must keep its triggers DROPPED —
# an external-content 'delete' op for a rowid the index never held is the
# canonical FTS5 index-corruption hazard the v23 marker gating exists to
# prevent.
FTS_CJK_TABLE_SQL = """
CREATE VIEW IF NOT EXISTS messages_fts_cjk_src AS
    SELECT id, role, content, tool_name, tool_calls
    FROM messages
    WHERE role <> 'tool';

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_cjk USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages_fts_cjk_src',
    content_rowid='id',
    tokenize='cjk_unicode61'
);
"""

FTS_CJK_TRIGGER_SQL = """
CREATE TRIGGER IF NOT EXISTS messages_fts_cjk_insert AFTER INSERT ON messages
WHEN new.role <> 'tool'
   AND (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_cjk_rebuild_high_water'), -1)
     OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_cjk_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_cjk(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_cjk_delete AFTER DELETE ON messages
WHEN old.role <> 'tool'
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_cjk_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_cjk_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_cjk(messages_fts_cjk, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_cjk_update AFTER UPDATE ON messages
WHEN (old.content IS NOT new.content
    OR old.tool_name IS NOT new.tool_name
    OR old.tool_calls IS NOT new.tool_calls
    OR old.role IS NOT new.role)
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_cjk_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_cjk_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_cjk(messages_fts_cjk, rowid, content, tool_name, tool_calls)
    SELECT 'delete', old.id, old.content, old.tool_name, old.tool_calls
    WHERE old.role <> 'tool';
    INSERT INTO messages_fts_cjk(rowid, content, tool_name, tool_calls)
    SELECT new.id, new.content, new.tool_name, new.tool_calls
    WHERE new.role <> 'tool';
END;
"""

_FTS_CJK_TRIGGERS = (
    "messages_fts_cjk_insert",
    "messages_fts_cjk_delete",
    "messages_fts_cjk_update",
)

# state_meta breadcrumb set when a tokenizer-less process had to drop the
# cjk triggers to keep message writes alive: rows written from that moment
# on are missing from the cjk index, so it must not serve reads until
# `hermes sessions optimize-storage` rebuilds it on a capable host.
FTS_CJK_STALE_KEY = "fts_cjk_stale"


def fts5_cjk_so_path() -> Path:
    """Location of the cjk_unicode61 loadable extension."""
    env = os.getenv("HERMES_FTS5_CJK_SO")
    if env:
        return Path(env).expanduser()
    return get_hermes_home() / "lib" / "libfts5_cjk.so"


def _cjk_fts_config_enabled() -> bool:
    """config.yaml ``sessions.cjk_fts`` (default on), via its env bridge."""
    return os.getenv("HERMES_CJK_FTS", "1").strip().lower() not in (
        "0", "false", "off", "no",
    )


def load_fts5_cjk_extension(conn: sqlite3.Connection) -> bool:
    """Best-effort load of the cjk_unicode61 tokenizer into ``conn``.

    Returns False (never raises) when the .so is absent, the feature is
    disabled via ``sessions.cjk_fts``, or this Python build has extension
    loading compiled out — every caller treats False as "behave exactly as
    before the cjk index existed".
    """
    if not _cjk_fts_config_enabled():
        return False
    path = fts5_cjk_so_path()
    if not path.exists():
        return False
    try:
        conn.enable_load_extension(True)
        try:
            conn.load_extension(str(path))
        finally:
            conn.enable_load_extension(False)
        return True
    except Exception:
        logger.warning("fts5_cjk extension load failed (%s)", path, exc_info=True)
        return False



# ── Legacy (v22 / inline-content) FTS DDL ──────────────────────────────
# Used ONLY to keep an existing pre-v23 install's search working and its
# triggers repairable UNTIL the user opts into `hermes db optimize`. This is
# the exact inline shape v11..v22 shipped: each virtual table stores its own
# copy of ``content || tool_name || tool_calls`` and the trigram table indexes
# every row (including role='tool'). We never CREATE these on a fresh install —
# fresh installs are born on the v23 external-content schema above. These
# constants exist so a legacy DB is never accidentally handed the v23 DDL
# (which would create the external-content trigram source VIEW and leave the
# DB in a mixed, broken state). `optimize_fts_storage()` is what migrates a
# legacy DB to the v23 shape.
LEGACY_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content
);

CREATE TRIGGER IF NOT EXISTS messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.id;
    INSERT INTO messages_fts(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""

LEGACY_FTS_TRIGRAM_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts_trigram USING fts5(
    content,
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
END;

CREATE TRIGGER IF NOT EXISTS messages_fts_trigram_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts_trigram WHERE rowid = old.id;
    INSERT INTO messages_fts_trigram(rowid, content) VALUES (
        new.id,
        COALESCE(new.content, '') || ' ' || COALESCE(new.tool_name, '') || ' ' || COALESCE(new.tool_calls, '')
    );
END;
"""


class SessionDB:
    """
    SQLite-backed session storage with FTS5 search.

    Thread-safe for the common gateway pattern (multiple reader threads,
    single writer via WAL mode). Each method opens its own cursor.
    """

    # ── Write-contention tuning ──
    # With multiple hermes processes (gateway + CLI sessions + worktree agents)
    # all sharing one state.db, WAL write-lock contention causes visible TUI
    # freezes.  SQLite's built-in busy handler uses a deterministic sleep
    # schedule that causes convoy effects under high concurrency.
    #
    # Instead, we keep the SQLite timeout short (1s) and handle retries at the
    # application level with random jitter, which naturally staggers competing
    # writers and avoids the convoy.
    _WRITE_MAX_RETRIES = 15
    _WRITE_RETRY_MIN_S = 0.020   # 20ms
    _WRITE_RETRY_MAX_S = 0.150   # 150ms
    # Attempt a WAL checkpoint every N successful writes (PASSIVE mode).
    _CHECKPOINT_EVERY_N_WRITES = 50
    # Merge fragmented FTS5 segments every N successful writes. The message
    # triggers append one segment per insert; left unmaintained these grow
    # into tens of thousands of segments, so every MATCH must scan them all
    # and every insert pays a growing automerge cost — which lengthens the
    # write-lock hold time and starves competing writers (gateway + cron
    # processes share one state.db), surfacing as "database is locked".
    # 'optimize' is a no-op once the index is already merged, so an idle DB
    # pays almost nothing; the cadence is deliberately coarse so the one-off
    # merge cost is amortised far below the checkpoint cadence.
    _OPTIMIZE_EVERY_N_WRITES = 1000
    # Session imports intentionally use a lower cap than exports: import holds
    # one BEGIN IMMEDIATE transaction, so bounded batches avoid starving live
    # gateway/CLI writers. The dashboard accepts one exported JSON/JSONL file
    # at a time, so these still cover normal history restores.
    _IMPORT_MAX_SESSIONS = 500
    _IMPORT_MAX_MESSAGES_PER_SESSION = 10_000
    _IMPORT_MAX_TOTAL_MESSAGES = 50_000
    _IMPORT_MAX_SESSION_BYTES = 5 * 1024 * 1024
    _IMPORT_MAX_TOTAL_BYTES = 25 * 1024 * 1024

    def __init__(self, db_path: Path = None, read_only: bool = False):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.read_only = read_only

        self._lock = threading.Lock()
        self._write_count = 0
        # One-shot guard for the runtime FTS rebuild recovery on the write
        # path. A corrupt FTS shadow table makes EVERY message write raise
        # the malformed/corrupt error class via the sync triggers; we repair
        # in place at most once per SessionDB instance so a genuinely
        # unrecoverable database can't put writers into a rebuild loop.
        self._fts_runtime_rebuild_attempted = False
        self._fts_enabled = False
        self._trigram_available = False
        # CJK-bigram index (cjk_unicode61 loadable tokenizer). _fts_cjk_loaded:
        # extension present on the writer connection; _fts_cjk_available: the
        # messages_fts_cjk table is queryable AND not marked stale. Set during
        # _init_schema / _probe_fts_cjk.
        self._fts_cjk_loaded = False
        self._fts_cjk_available = False
        self._fts_unavailable_warned = False
        self._conn = None
        try:
            if read_only:
                # Read-only attach for cross-profile aggregation: SELECT-only,
                # so we skip schema init entirely (no DDL, no FTS probe, no
                # column reconcile). Crucially this takes NO write lock, so
                # polling another profile's live DB on every sidebar refresh
                # never contends with that profile's running backend. The DB
                # must already exist + be initialised (callers guard on
                # db_path.exists()); a SELECT against an empty file raises and
                # the caller degrades per-profile.
                self._conn = sqlite3.connect(
                    f"file:{self.db_path}?mode=ro",
                    uri=True,
                    check_same_thread=False,
                    timeout=1.0,
                    isolation_level=None,
                )
                self._conn.row_factory = sqlite3.Row
                return

            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            def _connect_and_init():
                self._conn = sqlite3.connect(
                    str(self.db_path),
                    check_same_thread=False,
                    # Short timeout — application-level retry with random
                    # jitter handles contention instead of sitting in
                    # SQLite's internal busy handler for up to 30s.
                    timeout=1.0,
                    # auto-starts transactions on DML, which conflicts with
                    # our explicit BEGIN IMMEDIATE.  None = we manage
                    # transactions ourselves.
                    isolation_level=None,
                )
                self._conn.row_factory = sqlite3.Row
                apply_wal_with_fallback(self._conn, db_label="state.db")
                self._conn.execute("PRAGMA foreign_keys=ON")
                self._fts_cjk_loaded = load_fts5_cjk_extension(self._conn)
                self._init_schema()

            try:
                _connect_and_init()
            except sqlite3.DatabaseError as exc:
                # The malformed-schema class (e.g. a duplicate sqlite_master
                # row for messages_fts) fails on the very first statement —
                # before _init_schema can run — so it can't be caught at the
                # FTS-rebuild layer. Recover by repairing sqlite_master in
                # place (backup first; canonical sessions/messages preserved),
                # then reopen once. This is what lets Desktop/Dashboard
                # self-heal instead of silently showing "no sessions".
                if not is_malformed_db_error(exc) or not _claim_repair_attempt(self.db_path):
                    raise
                logger.error(
                    "state.db schema is malformed (%s) — attempting automatic "
                    "repair (a backup copy is made first).", exc,
                )
                try:
                    if self._conn is not None:
                        self._conn.close()
                except Exception:
                    pass
                report = repair_state_db_schema(self.db_path)
                if not report.get("repaired"):
                    raise
                _connect_and_init()

            # NOTE: the v23 FTS optimization is OPT-IN (`hermes db optimize`),
            # never auto-started on open. Legacy installs keep their working
            # v22 inline FTS untouched here; only the explicit foreground
            # command demotes + rebuilds. This avoids a background worker
            # racing session lifecycle and the surprise disk/latency cost on
            # an unattended open. (An interrupted optimize resumes when the
            # user re-runs the command.)
        except Exception as exc:
            # Capture the cause so /resume and friends can surface WHY the
            # session DB is unavailable instead of a bare "Session database
            # not available."  Callers that catch this exception keep their
            # existing ``self._session_db = None`` degradation path.
            #
            # Note: we deliberately do NOT clear _last_init_error on the
            # success path (no else branch).  In multi-threaded callers
            # (gateway, web_server per-request SessionDB()), a concurrent
            # successful open racing past this failure would erase the
            # cause that another thread's /resume is about to format.
            # Tests that need to reset the state can call
            # ``hermes_state._set_last_init_error(None)`` explicitly.
            _set_last_init_error(f"{type(exc).__name__}: {exc}")
            raise

    # ── Core write helper ──

    @staticmethod
    def _is_fts5_unavailable_error(exc: sqlite3.OperationalError) -> bool:
        err = str(exc).lower()
        if "no such module" in err and "fts5" in err:
            return True
        # SQLite builds that have FTS5 but lack the optional trigram tokenizer
        # raise "no such tokenizer: trigram" instead of "no such module".
        # Scope to trigram specifically to avoid masking unrelated tokenizer errors.
        if "no such tokenizer: trigram" in err:
            return True
        # The cjk_unicode61 tokenizer is a loadable extension — a process
        # that couldn't load it sees the same capability-error shape.
        if "no such tokenizer: cjk_unicode61" in err:
            return True
        return False

    @staticmethod
    def _is_trigram_unavailable_error(exc: sqlite3.OperationalError) -> bool:
        """True when only an optional tokenizer is missing (FTS5 itself works).

        Covers the built-in trigram tokenizer (needs SQLite >= 3.34) and the
        loadable cjk_unicode61 tokenizer — both mean "this one index can't be
        served here", never "disable FTS".
        """
        err = str(exc).lower()
        return (
            "no such tokenizer: trigram" in err
            or "no such tokenizer: cjk_unicode61" in err
        )

    @staticmethod
    def _db_has_legacy_inline_fts(cursor: sqlite3.Cursor) -> bool:
        """True when messages_fts exists in ANY pre-v23 shape.

        v23's messages_fts is external-content over THREE real columns
        (content, tool_name, tool_calls). Every pre-v23 shape lacks the
        tool_name/tool_calls columns — whether the old inline single-column
        form (v11..v22) or the even older external-content single-column form
        (v10-era, pre-#16751). We therefore detect "needs optimize" as "the
        stored CREATE lacks the tool_name column", which is the precise v23
        marker and correctly catches BOTH legacy variants.

        Returns False when messages_fts doesn't exist yet (fresh DB mid-init):
        the post-migration FTS setup block will create it in the v23 shape.
        """
        row = cursor.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'messages_fts'"
        ).fetchone()
        if row is None:
            return False
        sql = (row[0] if not isinstance(row, sqlite3.Row) else row["sql"]) or ""
        # The v23 table declares tool_name/tool_calls columns. Their absence
        # means a legacy shape that doesn't index tool metadata → optimize.
        return "tool_name" not in sql

    def _warn_trigram_unavailable(self, exc: sqlite3.OperationalError) -> None:
        """Log once that the trigram tokenizer is missing; base FTS5 stays enabled."""
        if getattr(self, "_trigram_unavailable_warned", False):
            return
        self._trigram_unavailable_warned = True
        logger.info(
            "SQLite trigram tokenizer unavailable for %s "
            "(requires SQLite >= 3.34, this build is %s); "
            "CJK/substring search will fall back to LIKE: %s",
            self.db_path,
            sqlite3.sqlite_version,
            exc,
        )

    def _warn_fts5_unavailable(self, exc: sqlite3.OperationalError) -> None:
        self._fts_enabled = False
        if self._fts_unavailable_warned:
            return
        self._fts_unavailable_warned = True
        logger.warning(
            "SQLite FTS5 unavailable for %s; full-text session search "
            "disabled. Run `hermes update` to rebuild the venv with a "
            "current Python (managed uv guarantees FTS5). "
            "(underlying error: %s)",
            self.db_path,
            exc,
        )

    def _sqlite_supports_fts5(self, cursor: sqlite3.Cursor) -> bool:
        try:
            cursor.execute("CREATE VIRTUAL TABLE temp._hermes_fts5_probe USING fts5(x)")
            cursor.execute("DROP TABLE temp._hermes_fts5_probe")
            return True
        except sqlite3.OperationalError as exc:
            if not self._is_fts5_unavailable_error(exc):
                raise
            self._warn_fts5_unavailable(exc)
            return False

    def _ensure_fts_cjk_schema(self, cursor) -> None:
        """Create / repair / self-heal the CJK-bigram index surface.

        ``cursor`` may be a Cursor or a Connection (both expose execute /
        executescript). Called only for v23-shape DBs with the base FTS
        surface healthy. Sets ``self._fts_cjk_available``. Never raises;
        every failure mode degrades to "no cjk index" (trigram/LIKE routing
        keeps working).

        Cases:
          tokenizer loaded, table absent  → create. Empty DB: index is
              complete by construction (triggers cover everything). Populated
              DB: set the cjk backfill markers so the id-gated triggers stay
              correct and `optimize-storage` can backfill; the index is NOT
              served until the backfill completes.
          tokenizer loaded, table present → ensure triggers (recreates any
              dropped by a tokenizer-less process), honour the stale
              breadcrumb (serve only when absent and no backfill pending).
          tokenizer NOT loaded, table present with live triggers → drop the
              cjk triggers so message INSERTs don't fail at trigger time,
              and leave the stale breadcrumb (#self-heal). The table itself
              stays for a later capable open to rebuild.
        """
        cjk_present = bool(cursor.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name = 'messages_fts_cjk'"
        ).fetchone())

        if not self._fts_cjk_loaded:
            if cjk_present:
                live = [
                    r[0] for r in cursor.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'trigger' "
                        f"AND name IN ({','.join('?' for _ in _FTS_CJK_TRIGGERS)})",
                        _FTS_CJK_TRIGGERS,
                    ).fetchall()
                ]
                if live:
                    # Self-heal: this process cannot tokenize, so every
                    # message INSERT would die inside the cjk trigger.
                    # Breadcrumb FIRST (crash between the two statements is
                    # merely conservative), then drop.
                    logger.warning(
                        "messages_fts_cjk triggers present but the "
                        "cjk_unicode61 tokenizer is unavailable (%s) — "
                        "dropping the cjk triggers so message writes keep "
                        "working. CJK search falls back to trigram/LIKE; "
                        "run `hermes sessions optimize-storage` on a host "
                        "with the extension to rebuild.",
                        fts5_cjk_so_path(),
                    )
                    cursor.execute(
                        "INSERT INTO state_meta (key, value) VALUES (?, '1') "
                        "ON CONFLICT(key) DO UPDATE SET value = '1'",
                        (FTS_CJK_STALE_KEY,),
                    )
                    for trig in live:
                        cursor.execute(f"DROP TRIGGER IF EXISTS {trig}")
            self._fts_cjk_available = False
            return

        try:
            cursor.executescript(FTS_CJK_TABLE_SQL)
            if not cjk_present:
                # Freshly created. An empty DB's index is complete by
                # construction (triggers will cover every future row); a
                # populated DB (e.g. a v23 install predating the cjk index)
                # gets the dedicated marker pair so the id-gated triggers
                # keep NEW rows indexed while old rows await the
                # `optimize-storage` backfill. Either way any old stale
                # breadcrumb refers to a table that no longer exists.
                cursor.execute(
                    "DELETE FROM state_meta WHERE key = ?",
                    (FTS_CJK_STALE_KEY,),
                )
                n_msgs = cursor.execute(
                    "SELECT COUNT(*) FROM messages WHERE role <> 'tool'"
                ).fetchone()[0]
                if n_msgs > 0:
                    hw = cursor.execute(
                        "SELECT COALESCE(MAX(id), 0) FROM messages"
                    ).fetchone()[0]
                    for k, v in (
                        ("fts_cjk_rebuild_high_water", str(hw)),
                        ("fts_cjk_rebuild_progress", "0"),
                    ):
                        cursor.execute(
                            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                            (k, v),
                        )
            stale = cursor.execute(
                "SELECT 1 FROM state_meta WHERE key = ?",
                (FTS_CJK_STALE_KEY,),
            ).fetchone()
            if stale:
                # A tokenizer-less process dropped the triggers at some
                # unknown point — the index has a gap of unknown extent.
                # Do NOT reinstall triggers (an external-content 'delete'
                # for an unindexed rowid corrupts the index); the next
                # `optimize-storage` run rebuilds from scratch.
                self._fts_cjk_available = False
                return
            cursor.executescript(FTS_CJK_TRIGGER_SQL)
            backfill_pending = cursor.execute(
                "SELECT 1 FROM state_meta "
                "WHERE key = 'fts_cjk_rebuild_high_water' LIMIT 1"
            ).fetchone()
            self._fts_cjk_available = not backfill_pending
        except sqlite3.OperationalError:
            # Includes "no such tokenizer: cjk_unicode61" if the extension
            # loaded but registration failed — degrade to trigram/LIKE.
            logger.warning(
                "messages_fts_cjk ensure failed; CJK search stays on "
                "trigram/LIKE", exc_info=True,
            )
            self._fts_cjk_available = False

    @staticmethod
    def _drop_fts_triggers(cursor: sqlite3.Cursor) -> None:
        for trigger in _FTS_TRIGGERS:
            try:
                cursor.execute(f"DROP TRIGGER IF EXISTS {trigger}")
            except sqlite3.OperationalError:
                pass

    @staticmethod
    def _fts_trigger_count(cursor: sqlite3.Cursor) -> int:
        placeholders = ",".join("?" for _ in _FTS_TRIGGERS)
        row = cursor.execute(
            f"SELECT COUNT(*) FROM sqlite_master "
            f"WHERE type = 'trigger' AND name IN ({placeholders})",
            _FTS_TRIGGERS,
        ).fetchone()
        return int(row[0] if not isinstance(row, sqlite3.Row) else row[0])

    @staticmethod
    def _rebuild_fts_indexes(
        cursor: sqlite3.Cursor,
        *,
        include_trigram: bool = True,
    ) -> None:
        # Both FTS tables are external-content (v23+): the special 'rebuild'
        # command wipes the inverted index and repopulates it from the
        # content source (messages for the standard index, the tool-row-
        # excluding messages_fts_trigram_src view for the trigram index).
        cursor.execute("INSERT INTO messages_fts(messages_fts) VALUES('rebuild')")
        if include_trigram:
            cursor.execute(
                "INSERT INTO messages_fts_trigram(messages_fts_trigram) VALUES('rebuild')"
            )
        # 'rebuild' indexes EVERY row, so any deferred-backfill markers are
        # now satisfied — clear them, otherwise the background worker would
        # re-insert rows the rebuild already covered (duplicate entries).
        cursor.execute(
            "DELETE FROM state_meta WHERE key IN "
            "('fts_rebuild_high_water', 'fts_rebuild_progress')"
        )

    @staticmethod
    def _rebuild_legacy_fts_indexes(
        cursor: sqlite3.Cursor,
        *,
        include_trigram: bool = True,
    ) -> None:
        """Rebuild the LEGACY inline FTS indexes (pre-v23) from messages.

        Used only to repair a legacy DB whose triggers degraded under an
        earlier no-FTS5 runtime. Inline tables have no external-content
        'rebuild' source, so we DELETE + reinsert the concatenated content
        the legacy triggers produced. Never touches the v23 shape.
        """
        cursor.execute("DELETE FROM messages_fts")
        cursor.execute(
            "INSERT INTO messages_fts(rowid, content) "
            "SELECT id, "
            "COALESCE(content, '') || ' ' || "
            "COALESCE(tool_name, '') || ' ' || "
            "COALESCE(tool_calls, '') "
            "FROM messages"
        )
        if not include_trigram:
            return
        cursor.execute("DELETE FROM messages_fts_trigram")
        cursor.execute(
            "INSERT INTO messages_fts_trigram(rowid, content) "
            "SELECT id, "
            "COALESCE(content, '') || ' ' || "
            "COALESCE(tool_name, '') || ' ' || "
            "COALESCE(tool_calls, '') "
            "FROM messages"
        )

    def _fts_table_probe(self, cursor: sqlite3.Cursor, table_name: str) -> Optional[bool]:
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 0")
            return True
        except sqlite3.OperationalError as exc:
            if self._is_fts5_unavailable_error(exc):
                # Only disable FTS entirely when the whole module is missing.
                # A missing trigram tokenizer only affects trigram searches.
                if self._is_trigram_unavailable_error(exc):
                    self._warn_trigram_unavailable(exc)
                else:
                    self._warn_fts5_unavailable(exc)
                return None
            if "no such table" in str(exc).lower():
                return False
            raise

    def _ensure_fts_schema(
        self,
        cursor: sqlite3.Cursor,
        table_name: str,
        ddl: str,
    ) -> bool:
        status = self._fts_table_probe(cursor, table_name)
        if status is None:
            return False
        try:
            # Run even when the virtual table exists so any dropped or missing
            # triggers are recreated after a previous no-FTS5 runtime disabled
            # them to keep message writes working.
            cursor.executescript(ddl)
            return True
        except sqlite3.OperationalError as exc:
            if not self._is_fts5_unavailable_error(exc):
                raise
            # Only disable FTS entirely when the whole FTS5 module is missing.
            # A missing specific tokenizer (e.g. trigram) means only that
            # particular table cannot be created — the base FTS5 table is fine.
            if self._is_trigram_unavailable_error(exc):
                self._warn_trigram_unavailable(exc)
            else:
                self._warn_fts5_unavailable(exc)
            return False

    def _execute_write(
        self,
        fn: Callable[[sqlite3.Connection], T],
    ) -> T:
        """Execute a write transaction with BEGIN IMMEDIATE and jitter retry.

        *fn* receives the connection and should perform INSERT/UPDATE/DELETE
        statements.  The caller must NOT call ``commit()`` — that's handled
        here after *fn* returns.

        BEGIN IMMEDIATE acquires the WAL write lock at transaction start
        (not at commit time), so lock contention surfaces immediately.
        On ``database is locked``, we release the Python lock, sleep a
        random 20-150ms, and retry — breaking the convoy pattern that
        SQLite's built-in deterministic backoff creates.

        Returns whatever *fn* returns.
        """
        last_err: Optional[Exception] = None
        for attempt in range(self._WRITE_MAX_RETRIES):
            try:
                with self._lock:
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        result = fn(self._conn)
                        self._conn.commit()
                    except BaseException:
                        try:
                            self._conn.rollback()
                        except Exception:
                            pass
                        raise
                # Success — periodic best-effort checkpoint + FTS merge.
                self._write_count += 1
                if self._write_count % self._CHECKPOINT_EVERY_N_WRITES == 0:
                    self._try_wal_checkpoint()
                if self._write_count % self._OPTIMIZE_EVERY_N_WRITES == 0:
                    self._try_optimize_fts()
                return result
            except sqlite3.OperationalError as exc:
                err_msg = str(exc).lower()
                if "locked" in err_msg or "busy" in err_msg:
                    last_err = exc
                    if attempt < self._WRITE_MAX_RETRIES - 1:
                        jitter = random.uniform(
                            self._WRITE_RETRY_MIN_S,
                            self._WRITE_RETRY_MAX_S,
                        )
                        time.sleep(jitter)
                        continue
                # Non-lock error or retries exhausted — propagate.
                raise
            except sqlite3.DatabaseError as exc:
                # Corrupt FTS shadow tables make every write raise the
                # malformed/corrupt error class through the FTS sync triggers
                # while the canonical messages table is intact. The gateway
                # session store has its own retry queue for transcript
                # appends (#65637 salvage), but cron and CLI writers call
                # SessionDB directly — without this, their writes hard-fail
                # until the next process restart triggers the offline repair.
                # Rebuild the FTS index in place (once per instance) via
                # rebuild_fts() and retry the failed write immediately.
                if not self._try_runtime_fts_rebuild(exc):
                    raise
                continue
        # Retries exhausted (shouldn't normally reach here).
        raise last_err or sqlite3.OperationalError(
            "database is locked after max retries"
        )

    @staticmethod
    def _is_fts_write_corruption_error(exc: sqlite3.DatabaseError) -> bool:
        """True for the error class a corrupt FTS index raises on writes.

        The message varies by SQLite version: older builds raise the generic
        ``database disk image is malformed`` (covered by
        ``is_malformed_db_error``); newer builds (e.g. ubuntu-latest CI)
        raise the FTS5-specific ``fts5: corrupt structure record for table
        "messages_fts"``. Both mean the same thing for the write path: the
        canonical rows are fine, the FTS shadow tables are not.
        """
        if is_malformed_db_error(exc):
            return True
        msg = str(exc).lower()
        return "fts5" in msg and "corrupt" in msg

    def _try_runtime_fts_rebuild(self, exc: sqlite3.DatabaseError) -> bool:
        """One-shot in-place FTS rebuild after a corrupt-index write failure.

        Returns True when a rebuild was performed and the failed write should
        be retried; False when the error isn't the FTS-corruption class, FTS
        is disabled, or a rebuild was already attempted for this instance.

        Delegates to :meth:`rebuild_fts` (the FTS5 ``'rebuild'`` command —
        index rewritten from the canonical messages table, zero message-row
        mutation). Safe to call from ``_execute_write``'s except path: the
        failed transaction was rolled back and ``self._lock`` released before
        the exception propagated, and ``rebuild_fts`` re-acquires it.
        E2E-verified: a corrupted ``messages_fts_data`` shadow table rejects
        every append; after the in-place rebuild the same append succeeds and
        search works again.
        """
        if self._fts_runtime_rebuild_attempted:
            return False
        if not self._fts_enabled:
            return False
        if not self._is_fts_write_corruption_error(exc):
            return False
        self._fts_runtime_rebuild_attempted = True
        logger.warning(
            "state.db write failed with an FTS-corruption error (%s) — "
            "attempting one-shot in-place FTS rebuild; canonical message "
            "rows are preserved.", exc,
        )
        try:
            rebuilt = self.rebuild_fts()
        except Exception as rebuild_exc:
            logger.error(
                "In-place FTS rebuild failed (%s); the database needs the "
                "full offline repair path (repair_state_db_schema).",
                rebuild_exc,
            )
            return False
        if not rebuilt:
            logger.error(
                "In-place FTS rebuild made no progress; the database needs "
                "the full offline repair path (repair_state_db_schema)."
            )
            return False
        logger.warning(
            "state.db FTS indexes rebuilt in place (%d); retrying the failed write.",
            rebuilt,
        )
        return True

    def _try_wal_checkpoint(self) -> None:
        """Best-effort PASSIVE WAL checkpoint.  Never raises.

        Flushes committed WAL frames back into the main DB file without
        requiring an exclusive lock.  PASSIVE is safe for frequent
        periodic use because it does not block concurrent writers and
        cannot corrupt B-tree pages under I/O pressure.

        PASSIVE does not truncate the WAL file — it stays at its
        high-water mark.  WAL truncation happens in :meth:`close`
        (TRUNCATE) and pre-VACUUM checkpoints, which run infrequently
        under controlled conditions.

        Previous TRUNCATE strategy caused B-tree corruption on large
        databases (65K+ pages) due to the exclusive-lock I/O pressure
        from checkpointing thousands of frames at once (issue #45383).
        """
        try:
            with self._lock:
                result = self._conn.execute(
                    "PRAGMA wal_checkpoint(PASSIVE)"
                ).fetchone()
                if result and result[1] > 0:
                    logger.debug(
                        "WAL checkpoint: %d/%d pages checkpointed",
                        result[2], result[1],
                    )
        except Exception as exc:
            logger.warning("WAL checkpoint (PASSIVE) failed: %s", exc)

    def _try_optimize_fts(self) -> None:
        """Best-effort FTS5 segment merge. Never raises.

        Runs on the ``_OPTIMIZE_EVERY_N_WRITES`` cadence from the write hot
        path (off the lock — ``optimize_fts`` re-acquires ``self._lock``
        itself, mirroring ``_try_wal_checkpoint``). ``read_only`` connections
        never reach the write path, so this is implicitly skipped for them.
        Once the index is merged the 'optimize' command is close to free, so
        the steady-state cost is negligible; the expensive case is only the
        first merge of a long-neglected index.
        """
        try:
            self.optimize_fts()
        except Exception:
            pass  # Best effort — never fatal.

    def close(self):
        """Close the database connection.

        Attempts a TRUNCATE WAL checkpoint first so that exiting processes
        help shrink the WAL file.
        """
        with self._lock:
            if self._conn:
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                except Exception as exc:
                    logger.debug("WAL checkpoint (TRUNCATE) at close failed: %s", exc)
                self._conn.close()
                self._conn = None

    # ── Chunked FTS rebuild engine (v23 opt-in optimize) ──
    #
    # `optimize_fts_storage()` (the `hermes sessions optimize-storage`
    # command) drops the legacy inline FTS indexes and backfills the new
    # external-content ones. A single blocking rebuild measured ~16 minutes
    # of held write lock on a real 25 GB DB, so the backfill runs in small
    # chunks, each in its own short write transaction:
    #   - concurrent readers/writers are never starved (WAL stays small,
    #     each chunk checkpoints via the normal _execute_write cadence);
    #   - an interrupted run (Ctrl-C, crash) resumes from
    #     fts_rebuild_progress when the command is re-run;
    #   - multiple processes sharing the DB don't double-run it — each chunk
    #     claims work by compare-and-swap on fts_rebuild_progress, so even a
    #     concurrent second runner just interleaves chunks safely.
    #
    # THROTTLING (the part that keeps a live gateway sharing the DB
    # responsive): a greedy chunk loop re-acquires BEGIN IMMEDIATE nearly
    # back-to-back and can starve another process's writer into exhausting
    # its lock retries (an early 5000-row/50ms version owned the write lock
    # ~85% of the time and visibly froze concurrent CLI sessions on a large
    # install). Two layers prevent that:
    #   1. Small chunks (500 rows) — a foreground write queues behind a
    #      chunk for at most ~tens of ms.
    #   2. Inter-chunk pause — the loop sleeps max(_FTS_REBUILD_MIN_PAUSE,
    #      chunk cost x _FTS_REBUILD_DUTY_FACTOR) between chunks, capping
    #      this process's share of DB bandwidth so concurrent writers always
    #      find open windows. This works cross-process (unlike any
    #      same-process activity stamp) because it bounds our own duty
    #      cycle unconditionally.

    _FTS_REBUILD_CHUNK_ROWS = 500
    _FTS_REBUILD_DUTY_FACTOR = 4.0      # sleep >= 4x chunk cost (≤20% duty)
    _FTS_REBUILD_MIN_PAUSE = 0.2        # seconds — floor between chunks

    def fts_rebuild_status(self) -> Optional[Dict[str, Any]]:
        """Return deferred-rebuild progress, or None when no rebuild pending.

        Shape: {"pending": True, "total": <rows at drop time>,
        "indexed": <rows backfilled>, "percent": <0-100 int>}.
        Consumed by search_messages() notes and by status surfaces
        (dashboard/desktop can poll this to render a progress indicator).
        """
        high_water = self.get_meta("fts_rebuild_high_water")
        if high_water is None:
            return None
        progress = int(self.get_meta("fts_rebuild_progress") or 0)
        total = int(high_water)
        if total <= 0:
            return None
        pct = min(100, int(100 * progress / total))
        return {"pending": True, "total": total, "indexed": progress, "percent": pct}

    def _fts_rebuild_finish(self) -> None:
        """Finalize the deferred rebuild: boundary sweep + clear markers.

        The sweep is cheap insurance against any write that slipped through
        the migration-boundary instant (between high_water capture and
        trigger activation): re-index any row near the boundary that the
        index is missing. docsize has one row per indexed doc, so the
        anti-join is exact and runs on a narrow id range.
        """
        def _do(conn):
            hw_row = conn.execute(
                "SELECT value FROM state_meta WHERE key = 'fts_rebuild_high_water'"
            ).fetchone()
            if hw_row is not None:
                hw = int(hw_row[0])
                # Sweep a generous window around the boundary.
                lo, hi = hw - 1000, hw + 1000
                conn.execute(
                    "INSERT INTO messages_fts(rowid, content, tool_name, tool_calls) "
                    "SELECT m.id, m.content, m.tool_name, m.tool_calls "
                    "FROM messages m "
                    "WHERE m.id > ? AND m.id <= ? "
                    "AND NOT EXISTS (SELECT 1 FROM messages_fts_docsize d WHERE d.id = m.id)",
                    (lo, hi),
                )
                conn.execute(
                    "INSERT INTO messages_fts_trigram(rowid, content, tool_name, tool_calls) "
                    "SELECT m.id, m.content, m.tool_name, m.tool_calls "
                    "FROM messages m "
                    "WHERE m.id > ? AND m.id <= ? AND m.role <> 'tool' "
                    "AND NOT EXISTS (SELECT 1 FROM messages_fts_trigram_docsize d WHERE d.id = m.id)",
                    (lo, hi),
                )
            conn.execute(
                "DELETE FROM state_meta WHERE key IN "
                "('fts_rebuild_high_water', 'fts_rebuild_progress')"
            )
        self._execute_write(_do)
        logger.info("Deferred FTS rebuild complete — all messages indexed.")

    # Demoted v22 FTS shadow tables awaiting teardown (see the v23 migration:
    # DROP of a multi-GB FTS vtable blocks for minutes, so the migration
    # demotes the vtable definitions out of sqlite_master and renames the
    # orphaned shadow tables — now plain tables — to fts_v22_trash_*; the
    # worker empties them in bounded chunks, then drops them cheaply).
    _FTS_TRASH_PREFIX = "fts_v22_trash_"

    def _fts_teardown_trash_step(self) -> bool:
        """Tear down one chunk of a demoted v22 FTS shadow table.

        The trash tables are PLAIN tables (their vtable parent was demoted
        away during the migration), so chunked DELETE + final DROP involve
        no FTS5 machinery at all. Returns True while teardown work remains.
        """
        with self._lock:
            trash = [
                r[0] for r in self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name LIKE ? ESCAPE '\\'",
                    (self._FTS_TRASH_PREFIX.replace("_", "\\_") + "%",),
                ).fetchall()
            ]
        if not trash:
            return False

        tbl = trash[0]

        def _do(conn):
            pk_cols = [
                r[1] for r in conn.execute(f"PRAGMA table_info({tbl})")
                if r[5] > 0
            ]
            key = ", ".join(pk_cols) if pk_cols else "rowid"
            cur = conn.execute(
                f"DELETE FROM {tbl} WHERE ({key}) IN "
                f"(SELECT {key} FROM {tbl} LIMIT {self._FTS_REBUILD_CHUNK_ROWS})"
            )
            if cur.rowcount == 0:
                # Empty — the DROP is cheap now.
                conn.execute(f"DROP TABLE IF EXISTS {tbl}")
                logger.info("Old FTS shadow table %s torn down.", tbl)
            return True  # re-check: more trash tables / chunks may remain

        try:
            return bool(self._execute_write(_do))
        except sqlite3.OperationalError as exc:
            logger.debug("FTS trash teardown chunk failed (will retry): %s", exc)
            return True

    def fts_rebuild_step(self) -> bool:
        """Backfill one chunk of the deferred FTS rebuild.

        Returns True when more work remains, False when the rebuild is
        complete (or none is pending). Safe to call from any process at any
        time; chunks are claimed atomically inside the write transaction, so
        concurrent callers interleave instead of duplicating rows.
        """
        if not self._fts_enabled:
            return False
        high_water_raw = self.get_meta("fts_rebuild_high_water")
        if high_water_raw is None:
            return False
        high_water = int(high_water_raw)
        include_trigram = self._trigram_available
        chunk = self._FTS_REBUILD_CHUNK_ROWS

        def _do(conn):
            # Re-read progress inside the write transaction (BEGIN IMMEDIATE
            # is already held by _execute_write) — this is the claim: two
            # workers can't read the same progress value concurrently.
            row = conn.execute(
                "SELECT value FROM state_meta WHERE key = 'fts_rebuild_progress'"
            ).fetchone()
            if row is None:
                return False  # finished (or cleared) by another process
            progress = int(row[0])
            if progress >= high_water:
                return False

            # The chunk upper bound is an id, not a row count, so gaps from
            # deleted rows don't shrink chunks below the claimed range.
            upper = min(progress + chunk, high_water)
            conn.execute(
                "INSERT INTO messages_fts(rowid, content, tool_name, tool_calls) "
                "SELECT id, content, tool_name, tool_calls FROM messages "
                "WHERE id > ? AND id <= ?",
                (progress, upper),
            )
            if include_trigram:
                conn.execute(
                    "INSERT INTO messages_fts_trigram"
                    "(rowid, content, tool_name, tool_calls) "
                    "SELECT id, content, tool_name, tool_calls FROM messages "
                    "WHERE id > ? AND id <= ? AND role <> 'tool'",
                    (progress, upper),
                )
            # Publish progress in the same transaction as the rows it
            # covers — crash-atomic: either both land or neither does.
            conn.execute(
                "UPDATE state_meta SET value = ? "
                "WHERE key = 'fts_rebuild_progress'",
                (str(upper),),
            )
            return upper < high_water

        try:
            more = self._execute_write(_do)
        except sqlite3.OperationalError as exc:
            logger.debug("FTS rebuild chunk failed (will retry): %s", exc)
            return True  # transient (lock contention) — caller retries
        if more is False:
            status = self.fts_rebuild_status()
            if status is not None and status["indexed"] >= status["total"]:
                self._fts_rebuild_finish()
            return False
        return bool(more)

    # ── CJK-bigram index backfill (dedicated marker pair) ──
    #
    # Same chunk engine as the main deferred rebuild, but on the
    # ``fts_cjk_rebuild_*`` markers so a cjk-only backfill (the common case:
    # an already-optimized v23 DB gaining the cjk index) never gates the
    # complete ``messages_fts`` / trigram triggers.

    def fts_cjk_rebuild_status(self) -> Optional[Dict[str, Any]]:
        """CJK-index backfill progress, or None when none is pending."""
        high_water = self.get_meta("fts_cjk_rebuild_high_water")
        if high_water is None:
            return None
        progress = int(self.get_meta("fts_cjk_rebuild_progress") or 0)
        total = int(high_water)
        if total <= 0:
            return None
        pct = min(100, int(100 * progress / total))
        return {"pending": True, "total": total, "indexed": progress, "percent": pct}

    def fts_cjk_rebuild_step(self) -> bool:
        """Backfill one chunk of the CJK index. True while work remains."""
        if not self._fts_enabled or not self._fts_cjk_loaded:
            return False
        high_water_raw = self.get_meta("fts_cjk_rebuild_high_water")
        if high_water_raw is None:
            return False
        high_water = int(high_water_raw)
        chunk = self._FTS_REBUILD_CHUNK_ROWS

        def _do(conn):
            row = conn.execute(
                "SELECT value FROM state_meta "
                "WHERE key = 'fts_cjk_rebuild_progress'"
            ).fetchone()
            if row is None:
                return False  # finished (or cleared) by another process
            progress = int(row[0])
            if progress >= high_water:
                return False
            upper = min(progress + chunk, high_water)
            conn.execute(
                "INSERT INTO messages_fts_cjk(rowid, content, tool_name, tool_calls) "
                "SELECT id, content, tool_name, tool_calls FROM messages "
                "WHERE id > ? AND id <= ? AND role <> 'tool'",
                (progress, upper),
            )
            conn.execute(
                "UPDATE state_meta SET value = ? "
                "WHERE key = 'fts_cjk_rebuild_progress'",
                (str(upper),),
            )
            return upper < high_water

        try:
            more = self._execute_write(_do)
        except sqlite3.OperationalError as exc:
            logger.debug("CJK FTS rebuild chunk failed (will retry): %s", exc)
            return True
        if more is False:
            status = self.fts_cjk_rebuild_status()
            if status is not None and status["indexed"] >= status["total"]:
                self._fts_cjk_rebuild_finish()
            return False
        return bool(more)

    def _fts_cjk_rebuild_finish(self) -> None:
        """Boundary sweep + clear the cjk markers; index becomes servable."""
        def _do(conn):
            hw_row = conn.execute(
                "SELECT value FROM state_meta "
                "WHERE key = 'fts_cjk_rebuild_high_water'"
            ).fetchone()
            if hw_row is not None:
                hw = int(hw_row[0])
                lo, hi = hw - 1000, hw + 1000
                conn.execute(
                    "INSERT INTO messages_fts_cjk(rowid, content, tool_name, tool_calls) "
                    "SELECT m.id, m.content, m.tool_name, m.tool_calls "
                    "FROM messages m "
                    "WHERE m.id > ? AND m.id <= ? AND m.role <> 'tool' "
                    "AND NOT EXISTS (SELECT 1 FROM messages_fts_cjk_docsize d WHERE d.id = m.id)",
                    (lo, hi),
                )
            conn.execute(
                "DELETE FROM state_meta WHERE key IN "
                "('fts_cjk_rebuild_high_water', 'fts_cjk_rebuild_progress')"
            )
        self._execute_write(_do)
        self._fts_cjk_available = True
        logger.info("CJK FTS index backfill complete — serving CJK search.")

    def _fts_cjk_reset_if_stale(self) -> None:
        """Rebuild path for a stale cjk index (triggers were dropped).

        The gap's extent is unknown, so the only safe recovery is a from-
        scratch rebuild: drop the table + triggers, clear the breadcrumb,
        recreate via ``_ensure_fts_cjk_schema`` (which sets fresh backfill
        markers on a populated DB). Called from ``optimize_fts_storage`` on
        a tokenizer-capable host; no-op when not stale.
        """
        if not self._fts_cjk_loaded:
            return

        def _do(conn):
            stale = conn.execute(
                "SELECT 1 FROM state_meta WHERE key = ?",
                (FTS_CJK_STALE_KEY,),
            ).fetchone()
            if not stale:
                return False
            for trig in _FTS_CJK_TRIGGERS:
                conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
            conn.execute("DROP TABLE IF EXISTS messages_fts_cjk")
            conn.execute("DROP VIEW IF EXISTS messages_fts_cjk_src")
            conn.execute(
                "DELETE FROM state_meta WHERE key IN "
                f"('{FTS_CJK_STALE_KEY}', 'fts_cjk_rebuild_high_water', "
                "'fts_cjk_rebuild_progress')"
            )
            return True
        was_stale = self._execute_write(_do)
        if was_stale:
            # Recreate outside the write transaction — _ensure_fts_cjk_schema
            # uses executescript(), which implicitly commits any pending
            # transaction and must not run inside _execute_write's BEGIN
            # IMMEDIATE. Sets fresh backfill markers on a populated DB.
            with self._lock:
                self._ensure_fts_cjk_schema(self._conn)
                self._conn.commit()

    # ── Opt-in v23 FTS storage optimization (`hermes sessions optimize-storage`) ──
    #
    # This is the ONLY path that migrates an existing legacy (v22 inline) DB
    # to the v23 external-content schema. It is deliberately foreground and
    # user-invoked, never automatic, because it is disk-heavy and long. It
    # runs the throttled/resumable chunk engine above to completion
    # synchronously — demote → new schema → chunked backfill → chunked
    # teardown — with progress callbacks, a disk preflight in the CLI
    # wrapper, a VACUUM at the end, and a defensive schema_version bump.

    def fts_optimize_available(self) -> bool:
        """True when `optimize_fts_storage()` has work to do: either this DB
        is a legacy inline-FTS install that can be optimized to the v23
        external-content schema, or a previous optimize run was interrupted
        (legacy vtables already demoted, but backfill markers and/or trash
        tables remain) and re-running would resume it, or the CJK-bigram
        index needs a backfill/rebuild on this tokenizer-capable host.
        False for fresh and fully-optimized installs (and when FTS5 is
        unavailable)."""
        if not self._fts_enabled or self.read_only:
            return False
        with self._lock:
            if self._db_has_legacy_inline_fts(self._conn):
                return True
            # Interrupted optimize: demotion already removed the legacy
            # vtables (so the check above is False), but the transition is
            # unfinished until the backfill markers are cleared and the
            # demoted trash tables are torn down. Search stays complete
            # through the gap supplement meanwhile; re-running resumes.
            if self._conn.execute(
                "SELECT 1 FROM state_meta "
                "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
            ).fetchone():
                return True
            # CJK-bigram index work — only offerable when THIS process can
            # tokenize: a pending backfill (markers set at creation on a
            # populated DB) or a stale index awaiting a from-scratch rebuild.
            if self._fts_cjk_loaded and self._conn.execute(
                "SELECT 1 FROM state_meta WHERE key IN "
                f"('fts_cjk_rebuild_high_water', '{FTS_CJK_STALE_KEY}') LIMIT 1"
            ).fetchone():
                return True
            return self._has_fts_trash(self._conn)

    def _has_fts_trash(self, conn) -> bool:
        """True when demoted v22 shadow tables are still awaiting teardown.
        Caller must hold ``self._lock`` (or pass a migration-time cursor)."""
        return bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name LIKE ? ESCAPE '\\' LIMIT 1",
            (self._FTS_TRASH_PREFIX.replace("_", "\\_") + "%",),
        ).fetchone())

    def _demote_legacy_fts_to_trash(self) -> int:
        """Demote the legacy inline FTS vtables and stage their shadow tables
        for chunked teardown. Returns MAX(messages.id) as the rebuild high
        water. O(1) schema surgery — the heavy delete is deferred to the
        chunked teardown, exactly as the validated auto path did."""
        def _do(conn):
            self._drop_fts_triggers(conn)
            conn.execute("DROP VIEW IF EXISTS messages_fts_trigram_src")
            had = bool(conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name IN ('messages_fts', 'messages_fts_trigram') "
                "AND sql LIKE 'CREATE VIRTUAL TABLE%' LIMIT 1"
            ).fetchone())
            if had:
                conn.execute("PRAGMA writable_schema=ON")
                conn.execute(
                    "DELETE FROM sqlite_master WHERE type = 'table' "
                    "AND name IN ('messages_fts', 'messages_fts_trigram') "
                    "AND sql LIKE 'CREATE VIRTUAL TABLE%'"
                )
                conn.execute("PRAGMA writable_schema=RESET")
                shadows = [
                    r[0] for r in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table' "
                        "AND (name LIKE 'messages_fts_%' ESCAPE '\\' "
                        "OR name LIKE 'messages_fts_trigram_%' ESCAPE '\\')"
                    ).fetchall()
                ]
                for sh in shadows:
                    conn.execute(f"ALTER TABLE {sh} RENAME TO fts_v22_trash_{sh}")
            # Create the new v23 empty schema + set the backfill markers.
            self._ensure_fts_schema(conn, "messages_fts", FTS_SQL)
            self._ensure_fts_schema(conn, "messages_fts_trigram", FTS_TRIGRAM_SQL)
            hw = conn.execute("SELECT COALESCE(MAX(id), 0) FROM messages").fetchone()[0]
            for k, v in (
                ("fts_rebuild_high_water", str(hw)),
                ("fts_rebuild_progress", "0"),
            ):
                conn.execute(
                    "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (k, v),
                )
            conn.execute("DELETE FROM state_meta WHERE key = 'fts_optimize_available'")
            return hw
        return int(self._execute_write(_do))

    def optimize_fts_storage(
        self,
        *,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
        vacuum: bool = True,
    ) -> Dict[str, Any]:
        """Migrate a legacy v22 inline-FTS DB to the v23 external-content
        schema, foreground and to completion. Safe to re-run: if a previous
        attempt was interrupted it resumes from the progress marker.

        ``progress_cb`` receives {"phase", "percent", "indexed", "total"}
        dicts for a CLI progress bar. Returns a summary dict.

        The trigram tokenizer being unavailable is not fatal — the base index
        is still rebuilt (CJK falls back to LIKE), mirroring normal startup.
        """
        if not self._fts_enabled:
            return {"ok": False, "reason": "fts5_unavailable"}
        if self.read_only:
            return {"ok": False, "reason": "read_only"}

        # Only demote if we're actually still on the legacy shape. If a prior
        # run already demoted (markers/trash present), skip straight to
        # finishing the backfill + teardown — this is what makes re-running
        # after an interruption safe.
        with self._lock:
            legacy = self._db_has_legacy_inline_fts(self._conn)
        pending = self.get_meta("fts_rebuild_high_water") is not None
        if legacy and not pending:
            self._demote_legacy_fts_to_trash()

        # A stale CJK index (triggers dropped by a tokenizer-less process)
        # can only be recovered from scratch — reset it now so the cjk
        # backfill phase below rebuilds it. No-op without the tokenizer.
        self._fts_cjk_reset_if_stale()
        # An optimized v23 DB gaining the cjk index for the first time (no
        # legacy work left, tokenizer newly installed): ensure the table +
        # markers exist so the backfill phase has work to claim.
        if self._fts_cjk_loaded:
            with self._lock:
                self._ensure_fts_cjk_schema(self._conn)
                self._conn.commit()

        def _emit(phase: str) -> None:
            if progress_cb is None:
                return
            st = self.fts_rebuild_status()
            if st is None:
                st = self.fts_cjk_rebuild_status()
            progress_cb({
                "phase": phase,
                "percent": st["percent"] if st else 100,
                "indexed": st["indexed"] if st else 0,
                "total": st["total"] if st else 0,
            })

        def _pause(chunk_seconds: float) -> None:
            """Inter-chunk throttle (see the chunk-engine note above).

            The chunk methods themselves never sleep, so this loop is the
            single place the duty cycle is enforced: without it, back-to-back
            BEGIN IMMEDIATE chunks starve any live gateway/CLI process
            sharing the DB out of its lock retries (the measured ~85%
            write-lock ownership that froze concurrent sessions).
            """
            time.sleep(max(
                self._FTS_REBUILD_MIN_PAUSE,
                chunk_seconds * self._FTS_REBUILD_DUTY_FACTOR,
            ))

        # Phase 1: backfill (foreground, throttled between chunks so a live
        # gateway sharing the DB stays responsive).
        _emit("backfill")
        while True:
            _t0 = time.monotonic()
            if not self.fts_rebuild_step():
                break
            _emit("backfill")
            _pause(time.monotonic() - _t0)
        _emit("backfill")

        # Phase 1b: backfill the CJK-bigram index (its own marker pair; a
        # no-op when the tokenizer isn't loadable or nothing is pending).
        while True:
            _t0 = time.monotonic()
            if not self.fts_cjk_rebuild_step():
                break
            _emit("backfill")
            _pause(time.monotonic() - _t0)

        # Phase 2: tear down the demoted legacy shadow tables in chunks.
        _emit("teardown")
        while True:
            _t0 = time.monotonic()
            if not self._fts_teardown_trash_step():
                break
            _emit("teardown")
            _pause(time.monotonic() - _t0)

        # Phase 3: reclaim freed pages to the OS.
        vacuum_ok = None
        if vacuum:
            _emit("vacuum")
            try:
                with self._lock:
                    self._conn.execute("VACUUM")
                vacuum_ok = True
            except sqlite3.OperationalError as exc:
                # Most common cause: not enough free disk for VACUUM's temp
                # copy. The optimization still succeeded; space just isn't
                # reclaimed until a later VACUUM. Non-fatal.
                logger.warning("VACUUM after FTS optimize failed: %s", exc)
                vacuum_ok = False

        # Phase 4: stamp the FTS storage layout as current, clear the "available"
        # flag, and advance schema_version if it was somehow still behind (the
        # main version normally advances on open now, but bump defensively so a
        # DB opened only by pre-decoupling code still settles). The FTS-layout
        # marker is the source of truth for "is this DB optimized".
        def _settle(conn):
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES ('fts_storage_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(FTS_STORAGE_VERSION),),
            )
            conn.execute("DELETE FROM state_meta WHERE key = 'fts_optimize_available'")
            conn.execute(
                "UPDATE schema_version SET version = ? WHERE version < ?",
                (SCHEMA_VERSION, SCHEMA_VERSION),
            )
        self._execute_write(_settle)
        _emit("done")
        logger.info(
            "FTS storage optimization complete (layout v%d).", FTS_STORAGE_VERSION
        )
        return {"ok": True, "vacuumed": vacuum_ok}

    @staticmethod
    def _parse_schema_columns(schema_sql: str) -> Dict[str, Dict[str, str]]:
        """Extract expected columns per table from SCHEMA_SQL.

        Uses an in-memory SQLite database to parse the SQL — SQLite itself
        handles all syntax (DEFAULT expressions with commas, inline
        REFERENCES, CHECK constraints, etc.) so there are zero regex
        edge cases.  The in-memory DB is opened, the schema DDL is
        executed, and PRAGMA table_info extracts the column metadata.

        Adding a column to SCHEMA_SQL is all that's needed; the
        reconciliation loop picks it up automatically.
        """
        ref = sqlite3.connect(":memory:")
        try:
            ref.executescript(schema_sql)
            table_columns: Dict[str, Dict[str, str]] = {}
            for (tbl,) in ref.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            ).fetchall():
                cols: Dict[str, str] = {}
                for row in ref.execute(
                    f'PRAGMA table_info("{tbl}")'
                ).fetchall():
                    # row: (cid, name, type, notnull, dflt_value, pk)
                    col_name = row[1]
                    col_type = row[2] or ""
                    notnull = row[3]
                    default = row[4]
                    pk = row[5]
                    # Reconstruct the type expression for ALTER TABLE ADD COLUMN
                    parts = [col_type] if col_type else []
                    if notnull and not pk:
                        parts.append("NOT NULL")
                    if default is not None:
                        parts.append(f"DEFAULT {default}")
                    cols[col_name] = " ".join(parts)
                table_columns[tbl] = cols
            return table_columns
        finally:
            ref.close()

    def _reconcile_columns(self, cursor: sqlite3.Cursor) -> None:
        """Ensure live tables have every column declared in SCHEMA_SQL.

        Follows the Beets/sqlite-utils pattern: the CREATE TABLE definition
        in SCHEMA_SQL is the single source of truth for the desired schema.
        On every startup this method diffs the live columns (via PRAGMA
        table_info) against the declared columns, and ADDs any that are
        missing.

        This makes column additions a declarative operation — just add
        the column to SCHEMA_SQL and it appears on the next startup.
        Version-gated migration blocks are no longer needed for ADD COLUMN.
        """
        expected = self._parse_schema_columns(SCHEMA_SQL)
        for table_name, declared_cols in expected.items():
            # Get current columns from the live table
            try:
                rows = cursor.execute(
                    f'PRAGMA table_info("{table_name}")'
                ).fetchall()
            except sqlite3.OperationalError:
                continue  # Table doesn't exist yet (shouldn't happen after executescript)
            live_cols = set()
            for row in rows:
                # PRAGMA table_info returns (cid, name, type, notnull, dflt_value, pk)
                name = row[1] if isinstance(row, (tuple, list)) else row["name"]
                live_cols.add(name)

            for col_name, col_type in declared_cols.items():
                if col_name not in live_cols:
                    safe_name = col_name.replace('"', '""')
                    try:
                        cursor.execute(
                            f'ALTER TABLE "{table_name}" ADD COLUMN "{safe_name}" {col_type}'
                        )
                    except sqlite3.OperationalError as exc:
                        # Expected: "duplicate column name" from a race or
                        # re-run.  Unexpected: "Cannot add a NOT NULL column
                        # with default value NULL" from a schema mistake.
                        # Log at DEBUG so it's visible in agent.log.
                        logger.debug(
                            "reconcile %s.%s: %s", table_name, col_name, exc,
                        )

    def _init_schema(self):
        """Create tables and FTS if they don't exist, reconcile columns.

        Schema management follows the declarative reconciliation pattern
        (Beets, sqlite-utils): SCHEMA_SQL is the single source of truth.
        On existing databases, _reconcile_columns() diffs live columns
        against SCHEMA_SQL and ADDs any missing ones.  This eliminates
        the version-gated migration chain for column additions, making
        it impossible for reordered or inserted migrations to skip columns.

        The schema_version table is retained for future data migrations
        (transforming existing rows) which cannot be handled declaratively.
        """
        cursor = self._conn.cursor()

        cursor.executescript(SCHEMA_SQL)

        # ── Declarative column reconciliation ──────────────────────────
        # Diff live tables against SCHEMA_SQL and ADD any missing columns.
        # This is idempotent and self-healing: even if a version-gated
        # migration was skipped (e.g. due to version renumbering), the
        # column gets created here.
        self._reconcile_columns(cursor)

        # Indexes that reference reconciler-added columns must be created
        # AFTER _reconcile_columns runs — declaring them in SCHEMA_SQL
        # makes the initial executescript fail on legacy DBs (the index's
        # WHERE clause references a column that doesn't exist yet).
        try:
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_platform_msg_id "
                "ON messages(session_id, platform_message_id) "
                "WHERE platform_message_id IS NOT NULL"
            )
        except sqlite3.OperationalError as exc:
            logger.debug("idx_messages_platform_msg_id create skipped: %s", exc)

        # Deferred indexes that reference the reconciler-added ``active``
        # column (idx_messages_session_active) — same ordering constraint.
        cursor.executescript(DEFERRED_INDEX_SQL)

        # Heal NULL ``active`` rows unconditionally on every startup.
        # On real-world DBs the reconciler-added ``active`` column can lack
        # its NOT NULL DEFAULT 1 (older reconciler builds reconstructed the
        # type without the default — see #51646: PRAGMA shows
        # (17,'active','INTEGER',0,None,0) in the wild), so INSERTs that
        # omitted the column wrote NULL and the ``WHERE active = 1``
        # transcript loaders hid the whole history.  The INSERTs now set
        # active=1 explicitly; this idempotent repair un-hides rows written
        # before the fix.  It was previously gated at ``current_version <
        # 12`` which never re-ran for already-v12+ databases.
        try:
            cursor.execute(
                "UPDATE messages SET active = 1 WHERE active IS NULL"
            )
        except sqlite3.OperationalError:
            pass

        fts5_available = self._sqlite_supports_fts5(cursor)
        fts_migrations_complete = True
        if not fts5_available:
            # Existing FTS triggers can still fire on messages INSERT/UPDATE
            # even though the current sqlite runtime cannot read the virtual
            # tables they target. Drop only the triggers so core persistence
            # continues; if a future runtime has FTS5, _ensure_fts_schema()
            # recreates them.
            self._drop_fts_triggers(cursor)

        # ── Schema version bookkeeping ─────────────────────────────────
        # Bump to current so future data migrations (if any) can gate on
        # version.  No version-gated column additions remain.
        cursor.execute("SELECT version FROM schema_version LIMIT 1")
        row = cursor.fetchone()
        if row is None:
            cursor.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (SCHEMA_VERSION,),
            )
        else:
            current_version = row["version"] if isinstance(row, sqlite3.Row) else row[0]
            # Data migrations that can't be expressed declaratively (row
            # backfills, index changes tied to a specific version step) stay
            # in a version-gated chain. Column additions are handled by
            # _reconcile_columns() above and no longer need entries here.
            if current_version < 10 and SCHEMA_VERSION == 10:
                # v10: trigram FTS5 table for CJK/substring search. The
                # virtual table + triggers are created unconditionally via
                # FTS_TRIGRAM_SQL below, but existing rows need a one-time
                # backfill into the FTS index.
                #
                # Only run this when v10 itself is the target schema. Current
                # v11+ code drops and rebuilds both FTS tables below, so doing
                # the v10-only trigram backfill first only burns startup time
                # and WAL space before v11 throws the work away.
                if fts5_available:
                    _fts_trigram_exists = self._fts_table_probe(
                        cursor, "messages_fts_trigram"
                    )
                    if _fts_trigram_exists is False:
                        if self._ensure_fts_schema(
                            cursor, "messages_fts_trigram", FTS_TRIGRAM_SQL
                        ):
                            cursor.execute(
                                "INSERT INTO messages_fts_trigram(rowid, content) "
                                "SELECT id, content FROM messages WHERE content IS NOT NULL"
                            )
                        else:
                            fts_migrations_complete = False
                    elif _fts_trigram_exists is None:
                        fts_migrations_complete = False
                else:
                    fts_migrations_complete = False
            if current_version < 11 and SCHEMA_VERSION < 23:
                # v11 (SUPERSEDED by v23): re-index FTS5 tables to cover
                # tool_name + tool_calls in inline mode (#16751). v23 drops
                # and rebuilds both FTS tables in external-content form, so
                # running the v11 inline backfill first would only burn
                # startup time and WAL space before v23 throws the work
                # away — and its inline INSERT shape no longer matches the
                # current external-content FTS_SQL anyway. Kept only for
                # source archaeology; unreachable while SCHEMA_VERSION >= 23.
                pass
            if current_version < 16:
                # v16: tag delegate subagent rows so pickers stay clean after
                # parent deletes that used to orphan them (parent_session_id → NULL).
                try:
                    cursor.execute(
                        "UPDATE sessions SET model_config = json_set("
                        "COALESCE(model_config, '{}'), '$._delegate_from', parent_session_id) "
                        f"WHERE parent_session_id IS NOT NULL "
                        "AND json_extract(COALESCE(model_config, '{}'), '$._delegate_from') IS NULL "
                        f"AND {_ephemeral_child_sql('sessions')}"
                    )
                    cursor.execute(
                        "UPDATE sessions SET model_config = json_set("
                        "COALESCE(model_config, '{}'), '$._delegate_from', '__orphaned__') "
                        "WHERE parent_session_id IS NULL "
                        "AND json_extract(COALESCE(model_config, '{}'), '$._delegate_from') IS NULL "
                        "AND json_extract(COALESCE(model_config, '{}'), '$._branched_from') IS NULL "
                        "AND title IS NULL "
                        "AND message_count <= 25 "
                        "AND EXISTS (SELECT 1 FROM messages m "
                        "            WHERE m.session_id = sessions.id AND m.role = 'tool') "
                        "AND NOT EXISTS (SELECT 1 FROM sessions ch "
                        "                WHERE ch.parent_session_id = sessions.id)"
                    )
                except sqlite3.OperationalError:
                    pass
            if current_version < 18:
                # v18: gateway metadata consolidation (#9006). Backfill
                # display_name / origin_json / expiry_finalized from
                # sessions.json so pre-migration gateway sessions are
                # discoverable from state.db without the JSON index.
                try:
                    self._backfill_gateway_metadata_from_sessions_json(cursor)
                except Exception as exc:
                    # Backfill is best-effort: sessions.json may be absent,
                    # corrupted, or partially stale. Missing metadata simply
                    # means consumers fall back to sessions.json for those
                    # rows until the gateway rewrites them.
                    logger.debug("v18 gateway metadata backfill skipped: %s", exc)
            if current_version < 20:
                # v20: per-model usage attribution (issue #51607). Going
                # forward update_token_counts() records each API call into
                # session_model_usage keyed by the live model, but existing
                # sessions only have their aggregate totals on the sessions
                # row. Seed one usage row per historical session from those
                # aggregates so insights reads uniformly from the new table.
                # INSERT OR IGNORE keeps it idempotent: if newer code already
                # wrote a (session_id, model, provider) row for a session, the
                # PK conflict skips the stale aggregate rather than doubling it.
                try:
                    cursor.execute(
                        """INSERT OR IGNORE INTO session_model_usage (
                               session_id, model, billing_provider,
                               billing_base_url, billing_mode,
                               api_call_count, input_tokens,
                               output_tokens, cache_read_tokens,
                               cache_write_tokens, reasoning_tokens,
                               estimated_cost_usd, actual_cost_usd,
                               cost_status, cost_source, first_seen, last_seen
                           )
                           SELECT id, COALESCE(model, 'unknown'),
                                  COALESCE(billing_provider, ''),
                                  COALESCE(billing_base_url, ''),
                                  COALESCE(billing_mode, ''),
                                  COALESCE(api_call_count, 0),
                                  COALESCE(input_tokens, 0),
                                  COALESCE(output_tokens, 0),
                                  COALESCE(cache_read_tokens, 0),
                                  COALESCE(cache_write_tokens, 0),
                                  COALESCE(reasoning_tokens, 0),
                                  COALESCE(estimated_cost_usd, 0),
                                  COALESCE(actual_cost_usd, 0),
                                  cost_status, cost_source,
                                  started_at, COALESCE(ended_at, started_at)
                           FROM sessions
                           WHERE COALESCE(input_tokens, 0)
                                 + COALESCE(output_tokens, 0)
                                 + COALESCE(cache_read_tokens, 0)
                                 + COALESCE(cache_write_tokens, 0)
                                 + COALESCE(reasoning_tokens, 0) > 0"""
                    )
                except sqlite3.OperationalError:
                    pass
            if current_version < 22:
                # v22: task-dimension usage attribution (issue #23270).
                # session_model_usage gains a ``task`` column ('' = main agent
                # loop; 'vision'/'compression'/'title_generation'/... =
                # auxiliary calls) so aux model spend is visible in analytics.
                # The column participates in the PRIMARY KEY and SQLite cannot
                # ALTER a PK, so rebuild the table. The reconciler will have
                # already ADDed the plain column on legacy DBs (harmless);
                # the rebuild bakes it into the PK properly. Existing rows are
                # main-loop accounting by definition → task=''.
                try:
                    legacy_pk = cursor.execute(
                        "SELECT COUNT(*) FROM pragma_table_info('session_model_usage') "
                        "WHERE name = 'task' AND pk > 0"
                    ).fetchone()[0]
                    if not legacy_pk:
                        cursor.execute("ALTER TABLE session_model_usage RENAME TO session_model_usage_v21")
                        cursor.execute(
                            """CREATE TABLE session_model_usage (
                                   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                                   model TEXT NOT NULL,
                                   billing_provider TEXT NOT NULL DEFAULT '',
                                   billing_base_url TEXT NOT NULL DEFAULT '',
                                   billing_mode TEXT NOT NULL DEFAULT '',
                                   task TEXT NOT NULL DEFAULT '',
                                   api_call_count INTEGER NOT NULL DEFAULT 0,
                                   input_tokens INTEGER NOT NULL DEFAULT 0,
                                   output_tokens INTEGER NOT NULL DEFAULT 0,
                                   cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                                   cache_write_tokens INTEGER NOT NULL DEFAULT 0,
                                   reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                                   estimated_cost_usd REAL NOT NULL DEFAULT 0,
                                   actual_cost_usd REAL NOT NULL DEFAULT 0,
                                   cost_status TEXT,
                                   cost_source TEXT,
                                   first_seen REAL,
                                   last_seen REAL,
                                   PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task)
                               )"""
                        )
                        cursor.execute(
                            """INSERT INTO session_model_usage (
                                   session_id, model, billing_provider, billing_base_url,
                                   billing_mode, task, api_call_count, input_tokens,
                                   output_tokens, cache_read_tokens, cache_write_tokens,
                                   reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                                   cost_status, cost_source, first_seen, last_seen
                               )
                               SELECT session_id, model, billing_provider, billing_base_url,
                                      billing_mode, '', api_call_count, input_tokens,
                                      output_tokens, cache_read_tokens, cache_write_tokens,
                                      reasoning_tokens, estimated_cost_usd, actual_cost_usd,
                                      cost_status, cost_source, first_seen, last_seen
                               FROM session_model_usage_v21"""
                        )
                        cursor.execute("DROP TABLE session_model_usage_v21")
                        cursor.execute(
                            "CREATE INDEX IF NOT EXISTS idx_session_model_usage_session "
                            "ON session_model_usage(session_id)"
                        )
                        cursor.execute(
                            "CREATE INDEX IF NOT EXISTS idx_session_model_usage_model "
                            "ON session_model_usage(model)"
                        )
                except sqlite3.OperationalError as exc:
                    logger.debug("v22 session_model_usage rebuild skipped: %s", exc)
            if current_version < 23:
                # v23: FTS storage redesign (issues #22478, #43690, #55233).
                # The v11 inline-mode FTS tables each store a full private
                # copy of every message (content || tool_name || tool_calls),
                # and the trigram index additionally covers role='tool' rows
                # (~90% of message bytes: base64 payloads, file dumps) at
                # ~2.6x amplification — together ~75% of state.db on heavy
                # installs (observed: 18.9 GB of a 25 GB DB).
                #
                # OPT-IN, NOT AUTOMATIC. The transition (demote old vtables →
                # new external-content schema → backfill → teardown → VACUUM)
                # is disk-heavy (transient ~2x file size to fully reclaim via
                # VACUUM) and long (~1-2h background on a 25 GB DB). Doing it
                # silently on every big user's next open — with a completeness
                # guarantee that depends on the process staying alive long
                # enough — is the wrong default. So on an EXISTING install we
                # touch nothing here: the v22 inline FTS keeps working exactly
                # as before, and we only record a flag advertising that the
                # optimization is available. `hermes sessions optimize-storage`
                # performs the whole transition as one deliberate, disk-checked,
                # progress-reported foreground operation.
                #
                # DECOUPLED VERSIONING. Crucially, this does NOT hold back the
                # main schema_version. The FTS storage LAYOUT is tracked by an
                # independent `fts_storage_version` marker (see
                # _fts_storage_version / SETTLE below), so schema_version
                # advances to SCHEMA_VERSION here like every other migration —
                # future v24+ migrations land automatically for legacy-FTS
                # users too. Only the FTS *layout* waits for opt-in.
                if fts5_available and self._db_has_legacy_inline_fts(cursor):
                    self.set_meta("fts_optimize_available", "1", cursor=cursor)

            # The FTS storage layout is versioned independently of the main
            # schema (see the v23 note above). Stamp the current layout so the
            # main version can always advance: a fresh/optimized DB is at
            # FTS_STORAGE_VERSION; a legacy DB is left at whatever it had
            # (absent/0) until `optimize-storage` runs. An INTERRUPTED
            # optimize (legacy vtables already demoted, but rebuild markers
            # or demoted trash tables still present) is NOT stamped either —
            # the marker is the source of truth for "fully optimized", and
            # `fts_optimize_available()` keeps offering the resume until the
            # transition actually completes.
            if (
                fts5_available
                and not self._db_has_legacy_inline_fts(cursor)
                and cursor.execute(
                    "SELECT 1 FROM state_meta "
                    "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
                ).fetchone() is None
                and not self._has_fts_trash(cursor)
            ):
                self.set_meta(
                    "fts_storage_version", str(FTS_STORAGE_VERSION), cursor=cursor
                )

            # Advance schema_version to current for ALL non-FTS-layout
            # migrations. This is deliberately NOT gated on the FTS opt-in —
            # holding the whole version back would block every future schema
            # migration for a user who never optimizes. FTS5 being unavailable
            # is the one case we skip (we can't have created the current FTS
            # objects, so claiming the current schema would be a lie).
            if (
                current_version < SCHEMA_VERSION
                and fts_migrations_complete
                and fts5_available
            ):
                cursor.execute(
                    "UPDATE schema_version SET version = ?",
                    (SCHEMA_VERSION,),
                )

        # Unique title index — always ensure it exists. Older databases may
        # contain duplicate aliases from before the constraint was enforced;
        # preserve every session while letting the newest one retain the alias.
        title_index_sql = (
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_title_unique "
            "ON sessions(title) WHERE title IS NOT NULL"
        )
        try:
            cursor.execute(title_index_sql)
        except sqlite3.IntegrityError:
            # The index is an optimization — its creation must never abort
            # opening the database, so the repair itself is also guarded.
            try:
                cursor.execute(
                    """UPDATE sessions AS older
                       SET title = NULL
                       WHERE title IS NOT NULL
                         AND EXISTS (
                             SELECT 1 FROM sessions AS newer
                             WHERE newer.title = older.title
                               AND newer.rowid > older.rowid
                         )"""
                )
                logger.warning(
                    "Cleared %d duplicate session title(s) while restoring the unique index",
                    cursor.rowcount,
                )
                cursor.execute(title_index_sql)
            except sqlite3.Error:
                logger.exception(
                    "Could not repair duplicate session titles; "
                    "unique title index not created"
                )
        except sqlite3.OperationalError:
            pass  # Index already exists

        if fts5_available:
            # FTS5 setup. Run the DDL even when the virtual table exists so
            # CREATE TRIGGER IF NOT EXISTS repairs trigger-only degradation from
            # an earlier no-FTS5 runtime.
            #
            # OPT-IN v23 boundary: a legacy v22 install (inline-content FTS,
            # not yet opted into `hermes db optimize`) must keep its EXISTING
            # inline schema + triggers. Running the v23 external-content DDL
            # here would create the trigram source VIEW and leave the DB in a
            # mixed inline/external state. So for a legacy DB we only ensure
            # its inline triggers exist (via the legacy DDL), and skip the
            # v23 view/external tables entirely. Fresh installs and opted-in
            # DBs have no legacy inline FTS, so they get the v23 DDL.
            if self._db_has_legacy_inline_fts(cursor):
                triggers_need_repair = (
                    self._fts_trigger_count(cursor) < len(_FTS_TRIGGERS)
                )
                self._fts_enabled = self._ensure_fts_schema(
                    cursor, "messages_fts", LEGACY_FTS_SQL
                )
                if self._fts_enabled:
                    trigram_enabled = self._ensure_fts_schema(
                        cursor, "messages_fts_trigram", LEGACY_FTS_TRIGRAM_SQL
                    )
                    self._trigram_available = trigram_enabled
                    if triggers_need_repair:
                        self._rebuild_legacy_fts_indexes(
                            cursor, include_trigram=trigram_enabled
                        )
            else:
                triggers_need_repair = (
                    self._fts_trigger_count(cursor) < len(_FTS_TRIGGERS)
                )
                self._fts_enabled = self._ensure_fts_schema(
                    cursor, "messages_fts", FTS_SQL
                )

                # Trigram FTS5 for CJK/substring search. This is optional
                # relative to the main FTS table; if it cannot be created,
                # CJK search falls back to LIKE.
                if self._fts_enabled:
                    trigram_enabled = self._ensure_fts_schema(
                        cursor, "messages_fts_trigram", FTS_TRIGRAM_SQL
                    )
                    self._trigram_available = trigram_enabled
                    if triggers_need_repair:
                        self._rebuild_fts_indexes(
                            cursor,
                            include_trigram=trigram_enabled,
                        )
                    # CJK-bigram index (cjk_unicode61). Strictly additive to
                    # the surfaces above and gated on the loadable tokenizer:
                    self._ensure_fts_cjk_schema(cursor)

        self._conn.commit()

    # =========================================================================
    # Session lifecycle
    # =========================================================================

    def _insert_session_row(
        self,
        session_id: str,
        source: str,
        model: str = None,
        model_config: Dict[str, Any] = None,
        system_prompt: str = None,
        user_id: str = None,
        session_key: str = None,
        chat_id: str = None,
        chat_type: str = None,
        thread_id: str = None,
        parent_session_id: str = None,
        cwd: str = None,
        profile_name: str = None,
        git_repo_root: str = None,
    ) -> None:
        """Insert a session row, enriching NULL metadata on conflict.

        The gateway's ``get_or_create_session`` creates a bare row (source +
        user_id) *before* the agent exists; the agent's later
        ``create_session`` then carries the real ``model`` / ``model_config`` /
        ``system_prompt``. A plain ``INSERT OR IGNORE`` silently dropped that
        enrichment, leaving gateway sessions with NULL model/billing metadata.
        The ``ON CONFLICT`` upsert backfills those fields via ``COALESCE`` —
        only filling columns that are still NULL, never overwriting values an
        earlier writer already set (so a later bare call with source="unknown"
        can't clobber a real source/model).

        ``chat_id``/``thread_id`` record the messaging origin (the chat/room and
        thread the session was started in) so that gateway ``/resume`` can prove
        a persisted, now-inactive row belongs to the caller's chat/thread before
        switching to it (IDOR scoping — without them the ``sessions`` table has
        no chat/thread to compare).

        When ``parent_session_id`` is set (compression fork, delegate/subagent
        spawn, branch continuation) and this row's own ``cwd``/``git_repo_root``/
        ``git_branch``/``profile_name`` are still NULL after the insert, they are
        backfilled from the parent row. Callers of ``create_session`` for a child
        session historically didn't propagate these fields themselves (e.g. the
        compression-fork path), so a lineage could silently lose its working
        directory and drop out of the project sidebar every time it forked
        (#64709), or lose its owning profile and be aggregated as "default" every
        time it rotated or branched (the cross-profile session-jump bug). This
        only fills NULLs — an explicit value on the child is never overwritten.
        For compression forks specifically
        (parent ended with ``end_reason='compression'``), the gateway origin
        columns (``user_id``/``session_key``/``chat_id``/``chat_type``/
        ``thread_id``/``display_name``/``origin_json``) are inherited too, so a
        crash before the gateway re-records the peer can't strand the child
        without a recoverable routing mapping (#59527).
        """
        def _do(conn):
            conn.execute(
                """INSERT INTO sessions (
                   id, source, user_id, session_key, chat_id, chat_type, thread_id,
                   model, model_config, system_prompt, parent_session_id, cwd,
                   profile_name, git_repo_root, started_at
                )
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       model = COALESCE(sessions.model, excluded.model),
                       model_config = COALESCE(sessions.model_config, excluded.model_config),
                       system_prompt = COALESCE(sessions.system_prompt, excluded.system_prompt),
                       session_key = COALESCE(sessions.session_key, excluded.session_key),
                       chat_id = COALESCE(sessions.chat_id, excluded.chat_id),
                       chat_type = COALESCE(sessions.chat_type, excluded.chat_type),
                       thread_id = COALESCE(sessions.thread_id, excluded.thread_id),
                       parent_session_id = COALESCE(sessions.parent_session_id, excluded.parent_session_id),
                       cwd = COALESCE(sessions.cwd, excluded.cwd),
                       profile_name = COALESCE(sessions.profile_name, excluded.profile_name),
                       git_repo_root = COALESCE(sessions.git_repo_root, excluded.git_repo_root)""",
                (
                    session_id,
                    source,
                    user_id,
                    session_key,
                    chat_id,
                    chat_type,
                    thread_id,
                    model,
                    json.dumps(model_config) if model_config else None,
                    system_prompt,
                    parent_session_id,
                    cwd,
                    profile_name,
                    git_repo_root,
                    time.time(),
                ),
            )
            if parent_session_id:
                conn.execute(
                    """UPDATE sessions
                       SET cwd = COALESCE(sessions.cwd,
                                 (SELECT p.cwd FROM sessions p
                                   WHERE p.id = sessions.parent_session_id)),
                           git_repo_root = COALESCE(sessions.git_repo_root,
                                           (SELECT p.git_repo_root FROM sessions p
                                             WHERE p.id = sessions.parent_session_id)),
                           git_branch = COALESCE(sessions.git_branch,
                                        (SELECT p.git_branch FROM sessions p
                                          WHERE p.id = sessions.parent_session_id)),
                           profile_name = COALESCE(sessions.profile_name,
                                          (SELECT p.profile_name FROM sessions p
                                            WHERE p.id = sessions.parent_session_id))
                     WHERE id = ? AND parent_session_id IS NOT NULL""",
                    (session_id,),
                )
                # Belt-and-suspenders for gateway routing metadata (#59527):
                # the gateway re-records the peer on the child after rotation
                # (d5b4879d4), but a hard crash between child creation and that
                # write leaves the child row without origin columns, so
                # ``find_latest_gateway_session_for_peer`` can't recover the
                # mapping on restart. Inherit them from the parent at creation
                # time — but ONLY for compression forks (parent already ended
                # with end_reason='compression'). Delegate/subagent children
                # are spawned while the parent is still live and must NOT
                # inherit routing keys, or peer recovery could repoint gateway
                # traffic into a subagent's session.
                conn.execute(
                    """UPDATE sessions
                       SET user_id = COALESCE(sessions.user_id,
                                     (SELECT p.user_id FROM sessions p
                                       WHERE p.id = sessions.parent_session_id)),
                           session_key = COALESCE(sessions.session_key,
                                         (SELECT p.session_key FROM sessions p
                                           WHERE p.id = sessions.parent_session_id)),
                           chat_id = COALESCE(sessions.chat_id,
                                     (SELECT p.chat_id FROM sessions p
                                       WHERE p.id = sessions.parent_session_id)),
                           chat_type = COALESCE(sessions.chat_type,
                                       (SELECT p.chat_type FROM sessions p
                                         WHERE p.id = sessions.parent_session_id)),
                           thread_id = COALESCE(sessions.thread_id,
                                       (SELECT p.thread_id FROM sessions p
                                         WHERE p.id = sessions.parent_session_id)),
                           display_name = COALESCE(sessions.display_name,
                                          (SELECT p.display_name FROM sessions p
                                            WHERE p.id = sessions.parent_session_id)),
                           origin_json = COALESCE(sessions.origin_json,
                                         (SELECT p.origin_json FROM sessions p
                                           WHERE p.id = sessions.parent_session_id))
                     WHERE id = ? AND parent_session_id IS NOT NULL
                       AND EXISTS (
                           SELECT 1 FROM sessions p
                           WHERE p.id = sessions.parent_session_id
                             AND p.end_reason = 'compression'
                       )""",
                    (session_id,),
                )
        self._execute_write(_do)

    def create_session(self, session_id: str, source: str, **kwargs) -> str:
        """Create a new session record. Returns the session_id."""
        self._insert_session_row(session_id, source, **kwargs)
        return session_id

    def record_gateway_session_peer(
        self,
        session_id: str,
        *,
        source: str,
        user_id: str = None,
        session_key: str = None,
        chat_id: str = None,
        chat_type: str = None,
        thread_id: str = None,
        display_name: str = None,
        origin_json: str = None,
    ) -> None:
        """Persist the gateway routing peer for an existing session row.

        ``display_name`` / ``origin_json`` carry the gateway's presentation
        and full origin metadata (#9006) so consumers (mcp_serve, mirror,
        channel directory) can read routing data from state.db instead of
        sessions.json.  They are COALESCE'd only in the sense that ``None``
        leaves the existing value untouched.
        """
        if not session_id or not session_key:
            return

        def _do(conn):
            conn.execute(
                """UPDATE sessions
                   SET session_key = ?, source = ?, user_id = ?, chat_id = ?,
                       chat_type = ?, thread_id = ?,
                       display_name = COALESCE(?, display_name),
                       origin_json = COALESCE(?, origin_json)
                   WHERE id = ?""",
                (
                    session_key,
                    source,
                    user_id,
                    chat_id,
                    chat_type,
                    thread_id,
                    display_name,
                    origin_json,
                    session_id,
                ),
            )

        self._execute_write(_do)

    def set_expiry_finalized(self, session_id: str, finalized: bool = True) -> None:
        """Mark a gateway session's expiry-finalization flag in state.db.

        Mirrors ``SessionEntry.expiry_finalized`` (sessions.json) so the flag
        survives even if the JSON index is pruned or lost (#9006).
        """
        if not session_id:
            return

        def _do(conn):
            conn.execute(
                "UPDATE sessions SET expiry_finalized = ? WHERE id = ?",
                (1 if finalized else 0, session_id),
            )

        self._execute_write(_do)

    # ── Gateway routing index (replaces sessions.json, #9006 follow-up) ────

    def save_gateway_routing_entry(
        self, session_key: str, entry_json: str, *, scope: str = ""
    ) -> None:
        """Upsert one gateway routing entry (session_key -> SessionEntry JSON).

        The gateway_routing table is the durable replacement for
        sessions.json: one row per routing key, holding the full serialized
        ``SessionEntry`` so the gateway can rehydrate exactly what it wrote.

        ``scope`` namespaces the index the way separate sessions.json files
        did (one per sessions_dir) — callers pass their sessions_dir path so
        two stores with different directories never share routing state.
        """
        if not session_key or not entry_json:
            return

        def _do(conn):
            conn.execute(
                """INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(scope, session_key) DO UPDATE SET
                       entry_json = excluded.entry_json,
                       updated_at = excluded.updated_at""",
                (scope, session_key, entry_json, time.time()),
            )

        self._execute_write(_do)

    def replace_gateway_routing_entries(
        self, entries: Dict[str, str], *, scope: str = ""
    ) -> None:
        """Atomically replace the routing index for *scope* with *entries*.

        Mirrors the sessions.json full-rewrite semantics: keys absent from
        *entries* are removed (pruned/reset sessions disappear from the
        index).  Runs as a single write transaction.  Other scopes are
        untouched.
        """
        now = time.time()

        def _do(conn):
            conn.execute("DELETE FROM gateway_routing WHERE scope = ?", (scope,))
            if entries:
                conn.executemany(
                    "INSERT INTO gateway_routing (scope, session_key, entry_json, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    [(scope, k, v, now) for k, v in entries.items() if k and v],
                )

        self._execute_write(_do)

    def load_gateway_routing_entries(self, *, scope: str = "") -> Dict[str, str]:
        """Load routing entries for *scope* as {session_key: entry_json}."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_key, entry_json FROM gateway_routing WHERE scope = ?",
                (scope,),
            ).fetchall()
        return {r["session_key"]: r["entry_json"] for r in rows}

    def delete_gateway_routing_entries(
        self, session_keys: List[str], *, scope: str = ""
    ) -> None:
        """Remove routing entries for the given session keys in *scope*."""
        if not session_keys:
            return

        def _do(conn):
            conn.executemany(
                "DELETE FROM gateway_routing WHERE scope = ? AND session_key = ?",
                [(scope, k) for k in session_keys],
            )

        self._execute_write(_do)

    def list_gateway_sessions(
        self,
        *,
        platform: Optional[str] = None,
        active_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """List gateway sessions (rows with a session_key) from state.db.

        Returns the newest row per session_key — the same shape consumers got
        from sessions.json: one live mapping per routing key.  ``platform``
        filters on ``source``; ``active_only`` restricts to sessions that
        have not ended.
        """
        query = """
            SELECT sessions.*,
                   COALESCE(
                       (SELECT MAX(m.timestamp) FROM messages m
                        WHERE m.session_id = sessions.id),
                       sessions.started_at
                   ) AS last_active
            FROM sessions
            WHERE session_key IS NOT NULL
              AND started_at = (
                  SELECT MAX(s2.started_at) FROM sessions s2
                  WHERE s2.session_key = sessions.session_key
              )
        """
        params: list = []
        if platform:
            query += " AND LOWER(source) = LOWER(?)"
            params.append(platform)
        if active_only:
            query += " AND ended_at IS NULL"
        query += " ORDER BY last_active DESC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def find_session_by_origin(
        self,
        *,
        platform: str,
        chat_id: str,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Find the most recent live session_id for a platform + chat origin.

        Equivalent of gateway/mirror's sessions.json scan: matches on
        source + chat_id (+ thread_id when provided).  When ``user_id`` is
        provided, exact sender matches are preferred; if multiple distinct
        users share the chat and none matches, returns None rather than
        contaminating another participant's session.
        """
        if not platform or chat_id in (None, ""):
            return None
        query = """
            SELECT id, user_id, started_at FROM sessions
            WHERE LOWER(source) = LOWER(?)
              AND session_key IS NOT NULL
              AND chat_id = ?
              AND ended_at IS NULL
        """
        params: list = [platform, str(chat_id)]
        if thread_id is not None:
            query += " AND COALESCE(thread_id, '') = ?"
            params.append(str(thread_id))
        query += " ORDER BY started_at DESC"
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(query, params).fetchall()]
        if not rows:
            return None
        if user_id:
            exact = [r for r in rows if str(r.get("user_id") or "") == str(user_id)]
            if exact:
                return str(exact[0]["id"])
            if len(rows) > 1:
                return None
        elif len(rows) > 1:
            distinct_users = {
                str(r.get("user_id") or "").strip()
                for r in rows
                if str(r.get("user_id") or "").strip()
            }
            if len(distinct_users) > 1:
                return None
        return str(rows[0]["id"])

    def _backfill_gateway_metadata_from_sessions_json(
        self, cursor: sqlite3.Cursor
    ) -> None:
        """One-time v18 backfill of gateway metadata from sessions.json.

        Existing gateway sessions predate the display_name / origin_json /
        expiry_finalized columns; copy what sessions.json knows so consumers
        can switch to state.db without losing pre-migration sessions.
        Only fills NULL columns — never overwrites data written by newer code.
        """
        sessions_file = get_hermes_home() / "sessions" / "sessions.json"
        if not sessions_file.exists():
            return
        with open(sessions_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        for key, entry in data.items():
            if str(key).startswith("_") or not isinstance(entry, dict):
                continue
            session_id = entry.get("session_id")
            if not session_id:
                continue
            origin = entry.get("origin")
            cursor.execute(
                """UPDATE sessions
                   SET session_key = COALESCE(session_key, ?),
                       chat_id = COALESCE(chat_id, ?),
                       chat_type = COALESCE(chat_type, ?),
                       thread_id = COALESCE(thread_id, ?),
                       display_name = COALESCE(display_name, ?),
                       origin_json = COALESCE(origin_json, ?),
                       expiry_finalized = CASE
                           WHEN COALESCE(expiry_finalized, 0) = 0 AND ? = 1 THEN 1
                           ELSE expiry_finalized
                       END
                   WHERE id = ?""",
                (
                    entry.get("session_key") or key,
                    (origin or {}).get("chat_id") if isinstance(origin, dict) else None,
                    entry.get("chat_type"),
                    (origin or {}).get("thread_id") if isinstance(origin, dict) else None,
                    entry.get("display_name"),
                    json.dumps(origin) if isinstance(origin, dict) else None,
                    1 if entry.get("expiry_finalized") or entry.get("memory_flushed") else 0,
                    str(session_id),
                ),
            )

    def find_latest_gateway_session_for_peer(
        self,
        *,
        source: str,
        user_id: Optional[str] = None,
        session_key: Optional[str] = None,
        chat_id: Optional[str] = None,
        chat_type: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Find the latest recoverable gateway session for a routing peer.

        ``sessions.json`` is the fast routing index, but it can be missing or
        pruned after process-level restart bugs.  New gateway sessions persist
        the deterministic ``session_key`` on the durable session row so the
        mapping can be rebuilt exactly.  Rows ended only by older gateway
        cleanup's ``agent_close`` bug or a mistaken TUI ``ws_orphan_reap``
        (dashboard viewer disconnect before #60609) are treated as recoverable;
        explicit conversation boundaries such as /new, /resume switches, and
        compression splits are not.
        """
        if not session_key:
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT * FROM sessions
                WHERE session_key = ?
                  AND source = ?
                  AND (ended_at IS NULL OR end_reason IN ('agent_close', 'ws_orphan_reap'))
                  AND (COALESCE(message_count, 0) > 0 OR EXISTS (
                      SELECT 1 FROM messages WHERE messages.session_id = sessions.id LIMIT 1
                  ))
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (session_key, source),
            ).fetchone()
            if row is not None:
                return dict(row)

            # Conservative fallback for rows created by current code but with a
            # temporarily-missing exact key: still require the complete peer
            # tuple so we never cross chats/threads/users.
            if chat_id is None or chat_type is None:
                return None
            row = self._conn.execute(
                """
                SELECT * FROM sessions
                WHERE source = ?
                  AND COALESCE(user_id, '') = COALESCE(?, '')
                  AND COALESCE(chat_id, '') = COALESCE(?, '')
                  AND COALESCE(chat_type, '') = COALESCE(?, '')
                  AND COALESCE(thread_id, '') = COALESCE(?, '')
                  AND (ended_at IS NULL OR end_reason IN ('agent_close', 'ws_orphan_reap'))
                  AND (COALESCE(message_count, 0) > 0 OR EXISTS (
                      SELECT 1 FROM messages WHERE messages.session_id = sessions.id LIMIT 1
                  ))
                ORDER BY started_at DESC
                LIMIT 1
                """,
                (source, user_id, chat_id, chat_type, thread_id),
            ).fetchone()
        return dict(row) if row else None

    def end_session(self, session_id: str, end_reason: str) -> None:
        """Mark a session as ended.

        No-ops when the session is already ended. The first end_reason wins:
        compression-split sessions must keep their ``end_reason = 'compression'``
        record even if a later stale ``end_session()`` call (e.g. from a
        desynced CLI session_id after ``/resume`` or ``/branch``) targets them
        with a different reason. Use ``reopen_session()`` first if you
        intentionally need to re-end a closed session with a new reason.
        """
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? "
                "WHERE id = ? AND ended_at IS NULL",
                (time.time(), end_reason, session_id),
            )
        self._execute_write(_do)

    def reopen_session(self, session_id: str) -> None:
        """Clear ended_at/end_reason so a session can be resumed."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    def promote_to_session_reset(
        self, session_id: str, reason: str = "session_reset"
    ) -> bool:
        """Durably mark a session as ended by an intentional reset boundary.

        Promotes *only* live rows (``ended_at IS NULL``) or rows carrying an
        accidental end_reason that the recovery query
        (``find_latest_gateway_session_for_peer``) treats as recoverable:
        ``agent_close`` (older gateway cleanup bug) and ``ws_orphan_reap``
        (mistaken TUI reaper).  Explicit conversation boundaries such as
        ``compression``, ``session_reset``, ``session_switch``, etc. are
        preserved — the first writer wins for those, and a later expiry
        finalization must not silently overwrite them.

        Plain ``end_session()`` is NOT sufficient for reset boundaries: it
        no-ops on an already-ended row, so a row that agent cleanup already
        closed as ``agent_close`` would stay recoverable and stale-route
        recovery would resurrect the reset session with its full history
        (#61220, #61993, #63539).

        Keep this promotion set in sync with the recoverable set in
        ``find_latest_gateway_session_for_peer`` — any reason recovery would
        reopen must be promotable here.

        ``reason`` lets reset paths keep their auditable specific reasons
        (``idle``, ``daily``, ``suspended``, ``resume_pending_expired``).

        Returns ``True`` when the row was promoted, ``False`` when skipped
        (already has a different explicit end_reason, or row not found).
        """
        if not session_id:
            return False
        now = time.time()

        def _do(conn):
            cursor = conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? "
                "WHERE id = ? AND (ended_at IS NULL "
                "OR end_reason IN ('agent_close', 'ws_orphan_reap'))",
                (now, reason, session_id),
            )
            return cursor.rowcount

        try:
            rows = self._execute_write(_do)
            return bool(rows)
        except Exception:
            return False

    def update_session_cwd(
        self, session_id: str, cwd: str, git_branch: str = None, git_repo_root: str = None
    ) -> None:
        """Persist the session working directory when a frontend knows it.

        ``git_branch`` records the git branch checked out in ``cwd`` at the time
        the session started/resumed. The sidebar groups main-checkout sessions
        by this so feature-branch work doesn't pile under a single "main" row
        (the main checkout's *current* branch is transient and would
        misattribute past sessions).

        ``git_repo_root`` records the git repo this cwd belongs to — the
        authoritative project key. Resolving it here, at the lowest level, means
        every surface reads the same membership instead of re-probing git in the
        GUI over a partial page. Each field is only written when non-empty so a
        probe failure never clobbers a previously-captured value.
        """
        if not session_id or not cwd:
            return

        branch = (git_branch or "").strip()
        repo_root = (git_repo_root or "").strip()

        sets = ["cwd = ?"]
        params: List[Any] = [cwd]
        if branch:
            sets.append("git_branch = ?")
            params.append(branch)
        if repo_root:
            sets.append("git_repo_root = ?")
            params.append(repo_root)
        params.append(session_id)

        def _do(conn):
            conn.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", params)

        self._execute_write(_do)

    def backfill_repo_roots(self, cwd_to_root: Dict[str, str]) -> None:
        """Persist resolved git repo roots for cwds that don't have one yet.

        Backfills history so projects light up for sessions created before the
        column existed, without clobbering an already-recorded root. Only
        non-empty roots are written (a non-git cwd stays NULL).
        """
        pairs = [(root, cwd) for cwd, root in cwd_to_root.items() if root and cwd]
        if not pairs:
            return

        def _do(conn):
            for root, cwd in pairs:
                conn.execute(
                    "UPDATE sessions SET git_repo_root = ? "
                    "WHERE cwd = ? AND COALESCE(git_repo_root, '') = ''",
                    (root, cwd),
                )

        self._execute_write(_do)

    def record_compression_failure_cooldown(
        self,
        session_id: str,
        cooldown_until: float,
        error: Optional[str] = None,
    ) -> None:
        """Persist the active compression-failure cooldown for a session."""
        if not session_id:
            return

        def _do(conn):
            conn.execute(
                "UPDATE sessions SET compression_failure_cooldown_until = ?, "
                "compression_failure_error = ? WHERE id = ?",
                (cooldown_until, error, session_id),
            )

        try:
            self._execute_write(_do)
        except sqlite3.Error as exc:
            logger.warning(
                "record_compression_failure_cooldown(%s) failed: %s",
                session_id, exc,
            )

    def get_compression_failure_cooldown(
        self,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the active compression-failure cooldown for ``session_id``."""
        if not session_id:
            return None
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT compression_failure_cooldown_until, compression_failure_error "
                "FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        cooldown_until = (
            row["compression_failure_cooldown_until"]
            if isinstance(row, sqlite3.Row)
            else row[0]
        )
        if cooldown_until is None:
            return None
        cooldown_until = float(cooldown_until)
        if cooldown_until <= now:
            return None
        error = (
            row["compression_failure_error"]
            if isinstance(row, sqlite3.Row)
            else row[1]
        )
        return {
            "cooldown_until": cooldown_until,
            "remaining_seconds": cooldown_until - now,
            "error": error,
        }

    def clear_compression_failure_cooldown(self, session_id: str) -> None:
        """Clear any persisted compression-failure cooldown for a session."""
        if not session_id:
            return

        def _do(conn):
            conn.execute(
                "UPDATE sessions SET compression_failure_cooldown_until = NULL, "
                "compression_failure_error = NULL WHERE id = ?",
                (session_id,),
            )

        try:
            self._execute_write(_do)
        except sqlite3.Error as exc:
            logger.warning(
                "clear_compression_failure_cooldown(%s) failed: %s",
                session_id, exc,
            )

    def get_compression_fallback_streak(self, session_id: str) -> int:
        """Return the persisted deterministic-fallback streak."""
        if not session_id:
            return 0
        with self._lock:
            conn = self._conn
            if conn is None:
                return 0
            row = conn.execute(
                "SELECT compression_fallback_streak FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return 0
        value = (
            row["compression_fallback_streak"]
            if isinstance(row, sqlite3.Row)
            else row[0]
        )
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def set_compression_fallback_streak(self, session_id: str, streak: int) -> None:
        """Persist the deterministic-fallback streak for one session."""
        if not session_id:
            return
        normalized = max(0, int(streak))

        def _do(conn):
            conn.execute(
                "UPDATE sessions SET compression_fallback_streak = ? WHERE id = ?",
                (normalized, session_id),
            )

        self._execute_write(_do)

    def get_compression_ineffective_count(self, session_id: str) -> int:
        """Return the persisted ineffective-compaction strike count.

        Mirrors ``get_compression_fallback_streak``: this is the durable half
        of the anti-thrash guard (``_ineffective_compression_count`` on the
        built-in compressor), persisted so that a fresh compressor bound to a
        resumed session inherits an armed/tripped guard instead of starting
        from zero across process restarts (#54923).
        """
        if not session_id:
            return 0
        with self._lock:
            conn = self._conn
            if conn is None:
                return 0
            row = conn.execute(
                "SELECT compression_ineffective_count FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return 0
        value = (
            row["compression_ineffective_count"]
            if isinstance(row, sqlite3.Row)
            else row[0]
        )
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    def set_compression_ineffective_count(self, session_id: str, count: int) -> None:
        """Persist the ineffective-compaction strike count for one session."""
        if not session_id:
            return
        normalized = max(0, int(count))

        def _do(conn):
            conn.execute(
                "UPDATE sessions SET compression_ineffective_count = ? WHERE id = ?",
                (normalized, session_id),
            )

        self._execute_write(_do)

    # ──────────────────────────────────────────────────────────────────────
    # Compression locks
    # ──────────────────────────────────────────────────────────────────────
    # Atomic per-session locks that prevent two compression paths from
    # racing on the same session_id and producing orphan child sessions.
    #
    # The race: ``conversation_compression.py`` rotates ``agent.session_id``
    # as a side effect of a successful compression (end old session, create
    # new). That mutation is local to the AIAgent instance — but ``state.db``
    # is shared across all instances. Two AIAgents that share the same
    # ``session_id`` at the moment they both decide to compress (most
    # commonly the parent turn's agent + a background-review fork started
    # right after the turn ended) each end the parent and create their own
    # NEW session, parented to the same old id. The gateway SessionEntry
    # only catches one rotation; the other child silently accumulates
    # writes — Damien's "parent → two orphan children" repro shape.
    #
    # The lock is keyed by ``session_id`` and is held for the duration of
    # the compress() call plus the rotation. ``holder`` identifies the
    # current owner (pid:tid:nonce) for diagnostics; the lock is recovered
    # via ``expires_at`` if the holder process crashed without releasing.
    def refresh_compression_lock(
        self,
        session_id: str,
        holder: str,
        ttl_seconds: float = 300.0,
    ) -> bool:
        """Extend the compression lock lease if ``holder`` still owns it."""
        if not session_id or not holder:
            return False
        now = time.time()
        expires_at = now + ttl_seconds

        def _do(conn):
            cur = conn.execute(
                "UPDATE compression_locks SET expires_at = ? "
                "WHERE session_id = ? AND holder = ? AND expires_at >= ?",
                (expires_at, session_id, holder, now),
            )
            return cur.rowcount > 0

        try:
            return bool(self._execute_write(_do))
        except sqlite3.Error as exc:
            logger.warning(
                "refresh_compression_lock(%s) failed: %s",
                session_id, exc,
            )
            return False

    def try_acquire_compression_lock(
        self,
        session_id: str,
        holder: str,
        ttl_seconds: float = 300.0,
    ) -> bool:
        """Try to atomically acquire the compression lock for ``session_id``.

        Returns ``True`` on success (caller now owns the lock and must
        release via :meth:`release_compression_lock`).  Returns ``False``
        if another holder already owns a non-expired lock — the caller
        MUST NOT proceed with compression in that case (its rotation would
        race against the holder's, splitting the session lineage).

        Expired locks (``expires_at < now``) are reclaimed transparently.
        Structured holders whose local ``pid=`` no longer exists are reclaimed
        immediately, so a gateway killed during compression does not stall the
        replacement process for the full lease TTL.

        Implementation: single-transaction DELETE-expired + INSERT-or-IGNORE,
        followed by a SELECT to confirm we got the row. SQLite serialises
        writes, so the whole sequence is atomic against other writers.
        """
        if not session_id:
            return False
        now = time.time()
        expires_at = now + ttl_seconds

        def _do(conn):
            reclaimed_holder = None
            row = conn.execute(
                "SELECT holder, expires_at FROM compression_locks "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is not None:
                current_holder = (
                    row["holder"] if isinstance(row, sqlite3.Row) else row[0]
                )
                current_expires_at = (
                    row["expires_at"] if isinstance(row, sqlite3.Row) else row[1]
                )
                if (
                    current_expires_at < now
                    or _compression_lock_holder_process_is_dead(current_holder)
                ):
                    conn.execute(
                        "DELETE FROM compression_locks "
                        "WHERE session_id = ? AND holder = ?",
                        (session_id, current_holder),
                    )
                    reclaimed_holder = current_holder
            # Then: try to insert. INSERT OR IGNORE returns no rowcount
            # difference — verify ownership via SELECT.
            conn.execute(
                "INSERT OR IGNORE INTO compression_locks "
                "(session_id, holder, acquired_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, holder, now, expires_at),
            )
            row = conn.execute(
                "SELECT holder FROM compression_locks WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            acquired = row is not None and (
                row["holder"] if isinstance(row, sqlite3.Row) else row[0]
            ) == holder
            return acquired, reclaimed_holder

        try:
            acquired, reclaimed_holder = self._execute_write(_do)
            if reclaimed_holder:
                logger.warning(
                    "Reclaimed stale compression lock for session=%s "
                    "(holder=%s)",
                    session_id,
                    reclaimed_holder,
                )
            return bool(acquired)
        except sqlite3.Error as exc:
            logger.warning(
                "try_acquire_compression_lock(%s) failed: %s",
                session_id, exc,
            )
            # Fail open: returning False makes the caller skip compression,
            # which is the safe behaviour when the lock subsystem is broken.
            return False

    def release_compression_lock(self, session_id: str, holder: str) -> None:
        """Release the compression lock for ``session_id`` iff we own it.

        Idempotent: no-op when the lock has already expired and been
        reclaimed by a different holder, or when no lock exists. The
        ``holder`` check prevents a late-returning compressor from
        clobbering a fresh lock held by someone else.
        """
        if not session_id:
            return

        def _do(conn):
            conn.execute(
                "DELETE FROM compression_locks "
                "WHERE session_id = ? AND holder = ?",
                (session_id, holder),
            )

        try:
            self._execute_write(_do)
        except sqlite3.Error as exc:
            logger.warning(
                "release_compression_lock(%s) failed: %s",
                session_id, exc,
            )

    def get_compression_lock_holder(self, session_id: str) -> Optional[str]:
        """Return the current (non-expired) holder for ``session_id``, or None.

        Diagnostic helper — not used by the locking protocol itself.
        """
        if not session_id:
            return None
        now = time.time()
        row = self._conn.execute(
            "SELECT holder FROM compression_locks "
            "WHERE session_id = ? AND expires_at >= ?",
            (session_id, now),
        ).fetchone()
        if row is None:
            return None
        return row["holder"] if isinstance(row, sqlite3.Row) else row[0]

    def update_session_meta(
        self,
        session_id: str,
        model_config_json: str,
        model: Optional[str] = None,
    ) -> None:
        """Update model_config and optionally model for an existing session.

        Uses COALESCE so that passing model=None leaves the stored model
        column unchanged.  Routes through _execute_write for the standard
        BEGIN IMMEDIATE + jitter-retry + lock guarantee.
        """
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET model_config = ?, model = COALESCE(?, model) WHERE id = ?",
                (model_config_json, model, session_id),
            )
        self._execute_write(_do)

    def update_system_prompt(self, session_id: str, system_prompt: str) -> None:
        """Store the full assembled system prompt snapshot."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET system_prompt = ? WHERE id = ?",
                (system_prompt, session_id),
            )
        self._execute_write(_do)

    def update_session_model(self, session_id: str, model: str) -> None:
        """Update the model for a session after a mid-session switch.

        Unlike ``update_token_counts`` which uses ``COALESCE(model, ?)``
        (only filling in NULL), this unconditionally sets the model column
        so that the dashboard reflects the user's latest /model choice.
        """
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET model = ? WHERE id = ?",
                (model, session_id),
            )
        self._execute_write(_do)

    def update_session_billing_route(
        self,
        session_id: str,
        *,
        provider: str,
        base_url: str,
        billing_mode: Optional[str] = None,
    ) -> None:
        """Unconditionally update the billing provider/base_url for a session.

        Unlike ``update_token_counts`` which uses ``COALESCE(billing_provider, ?)``
        (only filling in NULL), this unconditionally sets the billing fields so
        that the dashboard reflects the user's latest /model switch.

        Also nulls ``system_prompt`` so the cached snapshot (which embeds a
        stale ``Model:`` / ``Provider:`` header) is rebuilt — matching the
        behavior of ``update_session_model`` (see #48173, #48248).
        """
        def _do(conn):
            conn.execute(
                """UPDATE sessions SET
                   billing_provider = ?,
                   billing_base_url = ?,
                   billing_mode = COALESCE(?, billing_mode),
                   system_prompt = NULL
                   WHERE id = ?""",
                (provider, base_url, billing_mode, session_id),
            )
        self._execute_write(_do)

    def update_token_counts(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
        actual_cost_usd: Optional[float] = None,
        cost_status: Optional[str] = None,
        cost_source: Optional[str] = None,
        pricing_version: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        billing_mode: Optional[str] = None,
        api_call_count: int = 0,
        absolute: bool = False,
    ) -> None:
        """Update token counters and backfill model if not already set.

        When *absolute* is False (default), values are **incremented** — use
        this for per-API-call deltas (CLI path).

        When *absolute* is True, values are **set directly** — use this when
        the caller already holds cumulative totals (gateway path, where the
        cached agent accumulates across messages).
        """
        # Ensure the session row exists so the UPDATE doesn't silently affect
        # 0 rows.  Under concurrent load (cron + kanban + delegate_task) the
        # initial create_session() may have failed due to SQLite locking.
        # INSERT OR IGNORE is cheap and idempotent.
        self._insert_session_row(session_id, "unknown", model=model)
        if absolute:
            sql = """UPDATE sessions SET
                   input_tokens = ?,
                   output_tokens = ?,
                   cache_read_tokens = ?,
                   cache_write_tokens = ?,
                   reasoning_tokens = ?,
                   estimated_cost_usd = COALESCE(?, 0),
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?),
                   api_call_count = ?
                   WHERE id = ?"""
        else:
            sql = """UPDATE sessions SET
                   input_tokens = input_tokens + ?,
                   output_tokens = output_tokens + ?,
                   cache_read_tokens = cache_read_tokens + ?,
                   cache_write_tokens = cache_write_tokens + ?,
                   reasoning_tokens = reasoning_tokens + ?,
                   estimated_cost_usd = COALESCE(estimated_cost_usd, 0) + COALESCE(?, 0),
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE COALESCE(actual_cost_usd, 0) + ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?),
                   api_call_count = COALESCE(api_call_count, 0) + ?
                   WHERE id = ?"""
        has_accounted_usage = bool(
            input_tokens or output_tokens or cache_read_tokens
            or cache_write_tokens or reasoning_tokens or api_call_count
            or estimated_cost_usd or actual_cost_usd
        )
        params = (
            input_tokens,
            output_tokens,
            cache_read_tokens,
            cache_write_tokens,
            reasoning_tokens,
            estimated_cost_usd,
            actual_cost_usd,
            actual_cost_usd,
            cost_status,
            cost_source,
            pricing_version,
            billing_provider if has_accounted_usage else None,
            billing_base_url if has_accounted_usage else None,
            billing_mode if has_accounted_usage else None,
            model if has_accounted_usage else None,
            api_call_count,
            session_id,
        )
        # Per-model usage attribution.  ``update_token_counts`` is the single
        # chokepoint every per-API-call delta flows through (CLI, gateway, cron,
        # delegated runs — see conversation_loop / codex_runtime), and each call
        # carries the model/provider *active at the time of that call*.  The
        # ``sessions`` row only keeps one (model, billing_provider) pair, so a
        # mid-session ``/model`` switch otherwise attributes every token to the
        # initial model (issue #51607).  Recording the per-call delta into
        # session_model_usage keyed by the live model preserves an accurate
        # per-model breakdown regardless of how many times the user switches.
        #
        # Only the incremental path records here. Absolute cumulative updates
        # cannot be split back into routes; Insights reconciles any positive
        # residual against the aggregate session row instead.
        record_model_usage = (not absolute) and (
            input_tokens or output_tokens or cache_read_tokens
            or cache_write_tokens or reasoning_tokens or api_call_count
            or estimated_cost_usd
        )

        def _do(conn):
            row = conn.execute(
                "SELECT model, billing_provider, api_call_count FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            existing_model = row["model"] if row is not None else None
            existing_provider = row["billing_provider"] if row is not None else None
            existing_api_calls = int((row["api_call_count"] if row is not None else 0) or 0)

            # Session creation records the requested primary route before any API
            # call. If it fails and fallback succeeds, the first accounted usage
            # event is the first authoritative route. After that, preserve the
            # legacy row: one row cannot represent mixed-provider usage.
            first_accounted_route = (
                existing_api_calls == 0
                and has_accounted_usage
                and bool(model)
                and bool(billing_provider)
                and (existing_model != model or existing_provider != billing_provider)
            )
            if first_accounted_route:
                conn.execute(
                    """UPDATE sessions
                       SET model = ?, billing_provider = ?,
                       billing_base_url = ?, billing_mode = ?
                       WHERE id = ?""",
                    (model, billing_provider, billing_base_url, billing_mode, session_id),
                )
            conn.execute(sql, params)
            if record_model_usage:
                self._record_model_usage(
                    conn,
                    session_id,
                    model=model,
                    billing_provider=billing_provider,
                    billing_base_url=billing_base_url,
                    billing_mode=billing_mode,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens,
                    cache_write_tokens=cache_write_tokens,
                    reasoning_tokens=reasoning_tokens,
                    estimated_cost_usd=estimated_cost_usd,
                    actual_cost_usd=actual_cost_usd,
                    cost_status=cost_status,
                    cost_source=cost_source,
                    api_call_count=api_call_count,
                )
        self._execute_write(_do)

    def _record_model_usage(
        self,
        conn,
        session_id: str,
        *,
        model: Optional[str],
        billing_provider: Optional[str],
        billing_base_url: Optional[str],
        billing_mode: Optional[str],
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        reasoning_tokens: int,
        estimated_cost_usd: Optional[float],
        actual_cost_usd: Optional[float],
        cost_status: Optional[str],
        cost_source: Optional[str],
        api_call_count: int,
        task: str = "",
    ) -> None:
        """Accumulate a per-API-call usage delta into session_model_usage.

        Runs inside the caller's write transaction (after the ``sessions``
        UPDATE) so the per-model rows stay consistent with the summary row.
        When the caller omits the model/provider (some paths only pass token
        deltas), fall back to the values already recorded on the session row —
        the same COALESCE-from-session behaviour the summary update uses.

        ``task`` distinguishes what kind of work consumed the tokens:
        ``''`` (empty) is the main agent loop; auxiliary calls record their
        task name (``vision``, ``compression``, ``title_generation``, ...)
        via :meth:`record_auxiliary_usage` (issue #23270).
        """
        row = conn.execute(
            "SELECT model, billing_provider, billing_base_url, billing_mode "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        sess_model = row["model"] if row is not None else None
        sess_provider = row["billing_provider"] if row is not None else None
        sess_base_url = row["billing_base_url"] if row is not None else None
        sess_billing_mode = row["billing_mode"] if row is not None else None

        # Aux-task rows (task != '') must NOT inherit the session's main-loop
        # route: an aux call may use a completely different provider/model
        # (vision on gemini while the main loop runs anthropic). Missing info
        # stays 'unknown'/empty rather than borrowing a misleading route.
        if task:
            eff_model = model or "unknown"
            eff_provider = billing_provider or ""
            eff_base_url = billing_base_url or ""
            eff_billing_mode = billing_mode or ""
        else:
            eff_model = model or sess_model or "unknown"
            eff_provider = billing_provider or sess_provider or ""
            eff_base_url = billing_base_url or sess_base_url or ""
            eff_billing_mode = billing_mode or sess_billing_mode or ""
        now = time.time()
        conn.execute(
            """INSERT INTO session_model_usage (
                   session_id, model, billing_provider, billing_base_url, billing_mode,
                   task, api_call_count, input_tokens, output_tokens,
                   cache_read_tokens, cache_write_tokens, reasoning_tokens,
                   estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
                   first_seen, last_seen
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, model, billing_provider, billing_base_url, billing_mode, task)
               DO UPDATE SET
                   api_call_count = api_call_count + excluded.api_call_count,
                   input_tokens = input_tokens + excluded.input_tokens,
                   output_tokens = output_tokens + excluded.output_tokens,
                   cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
                   cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens,
                   reasoning_tokens = reasoning_tokens + excluded.reasoning_tokens,
                   estimated_cost_usd = estimated_cost_usd + excluded.estimated_cost_usd,
                   actual_cost_usd = actual_cost_usd + excluded.actual_cost_usd,
                   cost_status = COALESCE(excluded.cost_status, cost_status),
                   cost_source = COALESCE(excluded.cost_source, cost_source),
                   last_seen = excluded.last_seen""",
            (
                session_id,
                eff_model,
                eff_provider,
                eff_base_url,
                eff_billing_mode,
                task or "",
                api_call_count or 0,
                input_tokens or 0,
                output_tokens or 0,
                cache_read_tokens or 0,
                cache_write_tokens or 0,
                reasoning_tokens or 0,
                float(estimated_cost_usd or 0.0),
                float(actual_cost_usd or 0.0),
                cost_status,
                cost_source,
                now,
                now,
            ),
        )

    def ensure_session(
        self,
        session_id: str,
        source: str = "unknown",
        model: str = None,
        **kwargs,
    ) -> str:
        """Ensure a session row exists (INSERT OR IGNORE). Accepts optional kwargs."""
        self._insert_session_row(session_id, source, model=model, **kwargs)
        return session_id

    def record_auxiliary_usage(
        self,
        session_id: str,
        task: str,
        *,
        model: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
    ) -> None:
        """Record an auxiliary LLM call's usage against *session_id* (issue #23270).

        Auxiliary calls (vision, compression, title_generation, web_extract,
        session_search, ...) historically discarded their usage, leaving the
        dashboard's per-model analytics blind to aux model spend. This writes
        a per-(model, provider, task) delta into ``session_model_usage`` —
        the same table the main loop's ``update_token_counts`` feeds — WITHOUT
        touching the ``sessions`` summary row. That separation is deliberate:
        the gateway overwrites session counters with absolute main-loop totals,
        so folding aux tokens into the summary row would either be clobbered
        or double-counted. Insights/analytics read the union of both.

        Best-effort by contract: callers must never fail an aux call because
        accounting failed.
        """
        if not session_id or not task:
            return
        # FK on session_model_usage.session_id → sessions.id: ensure the row
        # exists (same INSERT OR IGNORE guard update_token_counts uses — the
        # initial create_session() can fail under concurrent SQLite locking).
        self._insert_session_row(session_id, "unknown")

        def _do(conn):
            self._record_model_usage(
                conn,
                session_id,
                model=model,
                billing_provider=billing_provider,
                billing_base_url=billing_base_url,
                billing_mode=None,
                input_tokens=input_tokens or 0,
                output_tokens=output_tokens or 0,
                cache_read_tokens=cache_read_tokens or 0,
                cache_write_tokens=cache_write_tokens or 0,
                reasoning_tokens=reasoning_tokens or 0,
                estimated_cost_usd=estimated_cost_usd,
                actual_cost_usd=None,
                cost_status=None,
                cost_source=None,
                api_call_count=1,
                task=task,
            )
        self._execute_write(_do)

    def prune_empty_ghost_sessions(self, sessions_dir: "Optional[Path]" = None) -> int:
        """Remove empty TUI ghost sessions (no messages, no title, >24hr old)."""
        cutoff = time.time() - 86400  # Only sessions older than 24 hours

        def _do(conn):
            rows = conn.execute("""
                SELECT id FROM sessions
                WHERE source = 'tui'
                  AND title IS NULL
                  AND ended_at IS NOT NULL
                  AND started_at < ?
                  AND NOT EXISTS (
                      SELECT 1 FROM messages WHERE messages.session_id = sessions.id
                  )
            """, (cutoff,)).fetchall()
            ids = [r[0] if isinstance(r, (tuple, list)) else r["id"] for r in rows]
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"DELETE FROM sessions WHERE id IN ({placeholders})", ids
                )
            return ids

        removed_ids = self._execute_write(_do) or []
        # Clean up any on-disk session files (belt-and-suspenders)
        if sessions_dir and removed_ids:
            for sid in removed_ids:
                self._remove_session_files(sessions_dir, sid)
        return len(removed_ids)

    def finalize_orphaned_compression_sessions(self) -> int:
        """Mark orphaned compression continuation sessions as ended.

        Targets child sessions that were never finalized: parent is ended
        with reason='compression', child has messages but no end_reason/ended_at
        and api_call_count=0.  Non-destructive: preserves all messages and sets
        end_reason='orphaned_compression'.  Fix for #20001.
        """
        cutoff = time.time() - 604800  # 7 days

        def _do(conn):
            now = time.time()
            result = conn.execute(
                """
                UPDATE sessions
                SET ended_at = ?,
                    end_reason = 'orphaned_compression'
                WHERE api_call_count = 0
                  AND end_reason IS NULL
                  AND ended_at IS NULL
                  AND started_at < ?
                  AND parent_session_id IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM sessions p
                      WHERE p.id = sessions.parent_session_id
                        AND p.end_reason = 'compression'
                        AND p.ended_at IS NOT NULL
                  )
                  AND EXISTS (
                      SELECT 1 FROM messages m
                      WHERE m.session_id = sessions.id
                  )
                """,
                (now, cutoff),
            )
            return result.rowcount

        return self._execute_write(_do) or 0

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def resolve_session_id(self, session_id_or_prefix: str) -> Optional[str]:
        """Resolve an exact or uniquely prefixed session ID to the full ID.

        Returns the exact ID when it exists. Otherwise treats the input as a
        prefix and returns the single matching session ID if the prefix is
        unambiguous. Returns None for no matches or ambiguous prefixes.
        """
        exact = self.get_session(session_id_or_prefix)
        if exact:
            return exact["id"]

        escaped = (
            session_id_or_prefix
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' ORDER BY started_at DESC LIMIT 2",
                (f"{escaped}%",),
            )
            matches = [row["id"] for row in cursor.fetchall()]
        if len(matches) == 1:
            return matches[0]
        return None

    # Maximum length for session titles
    MAX_TITLE_LENGTH = 100

    @staticmethod
    def sanitize_title(title: Optional[str]) -> Optional[str]:
        """Validate and sanitize a session title.

        - Strips leading/trailing whitespace
        - Removes ASCII control characters (0x00-0x1F, 0x7F) and problematic
          Unicode control chars (zero-width, RTL/LTR overrides, etc.)
        - Collapses internal whitespace runs to single spaces
        - Normalizes empty/whitespace-only strings to None
        - Enforces MAX_TITLE_LENGTH

        Returns the cleaned title string or None.
        Raises ValueError if the title exceeds MAX_TITLE_LENGTH after cleaning.
        """
        if not title:
            return None

        # Lone surrogates cannot be bound by sqlite3 (UnicodeEncodeError at
        # UTF-8 encode time) — scrub them like every other write path here.
        title = _sanitize_surrogates(title)

        # Remove ASCII control characters (0x00-0x1F, 0x7F) but keep
        # whitespace chars (\t=0x09, \n=0x0A, \r=0x0D) so they can be
        # normalized to spaces by the whitespace collapsing step below
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', title)

        # Remove problematic Unicode control characters:
        # - Zero-width chars (U+200B-U+200F, U+FEFF)
        # - Directional overrides (U+202A-U+202E, U+2066-U+2069)
        # - Object replacement (U+FFFC), interlinear annotation (U+FFF9-U+FFFB)
        cleaned = re.sub(
            r'[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]',
            '', cleaned,
        )

        # Collapse internal whitespace runs and strip
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        if not cleaned:
            return None

        if len(cleaned) > SessionDB.MAX_TITLE_LENGTH:
            raise ValueError(
                f"Title too long ({len(cleaned)} chars, max {SessionDB.MAX_TITLE_LENGTH})"
            )

        return cleaned

    def _is_compression_ancestor(
        self, conn, *, ancestor_id: str, descendant_id: str
    ) -> bool:
        """Return True if *ancestor_id* is a compression predecessor of
        *descendant_id* (walking parent links up the continuation chain).

        The continuation edge is the canonical one shared with
        :func:`_ephemeral_child_sql` / :meth:`set_session_archived`
        (``_COMPRESSION_CHILD_SQL``): a parent → child edge counts only when the
        parent ended with ``end_reason = 'compression'`` and the child started
        at or after the parent's ``ended_at``, which distinguishes continuations
        from delegate subagents / branch children that also carry a
        ``parent_session_id``. Expressed as a single recursive CTE rather than a
        per-hop Python walk so the edge definition lives in exactly one place.
        """
        if not ancestor_id or not descendant_id or ancestor_id == descendant_id:
            return False
        # Walk parent links up from the descendant, following only compression
        # continuation edges, and check whether ancestor_id is reached.
        edge = _COMPRESSION_CHILD_SQL.format(a="child")
        row = conn.execute(
            f"""
            WITH RECURSIVE ancestors(id) AS (
                SELECT ?
                UNION
                SELECT parent.id
                FROM ancestors a
                JOIN sessions child ON child.id = a.id
                JOIN sessions parent ON parent.id = child.parent_session_id
                WHERE {edge}
            )
            SELECT 1 FROM ancestors WHERE id = ? AND id != ? LIMIT 1
            """,
            (descendant_id, ancestor_id, descendant_id),
        ).fetchone()
        return row is not None

    def _set_session_title(
        self,
        session_id: str,
        title: str,
        *,
        only_if_empty: bool,
    ) -> bool:
        title = self.sanitize_title(title)

        def _do(conn):
            if only_if_empty:
                current = conn.execute(
                    "SELECT title FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if current is None or current["title"] is not None:
                    return 0

            if title:
                # Check uniqueness (allow the same session to keep its own title)
                cursor = conn.execute(
                    "SELECT id FROM sessions WHERE title = ? AND id != ?",
                    (title, session_id),
                )
                conflict = cursor.fetchone()
                if conflict:
                    conflict_id = conflict["id"]
                    # A compression continuation is the live, projected-forward
                    # head of its conversation; its compressed predecessors are
                    # ended and hidden from the session list (list_sessions_rich
                    # projects roots → tip). When the title that "conflicts" is
                    # held by such a hidden ancestor, the user has no way to free
                    # it — renaming the visible tip back to the base name would
                    # dead-end with "already in use by <session they can't see>".
                    # Treat this as a transfer: move the title off the ancestor
                    # onto the continuation. Uniqueness is preserved (still only
                    # one session carries the exact title) and the parent-link
                    # lineage is untouched.
                    if self._is_compression_ancestor(
                        conn, ancestor_id=conflict_id, descendant_id=session_id
                    ):
                        conn.execute(
                            "UPDATE sessions SET title = NULL WHERE id = ?",
                            (conflict_id,),
                        )
                    else:
                        raise ValueError(
                            f"Title '{title}' is already in use by session {conflict_id}"
                        )
            predicate = " AND title IS NULL" if only_if_empty else ""
            cursor = conn.execute(
                f"UPDATE sessions SET title = ? WHERE id = ?{predicate}",
                (title, session_id),
            )
            return cursor.rowcount

        rowcount = self._execute_write(_do)
        return rowcount > 0

    def set_session_title(self, session_id: str, title: str) -> bool:
        """Set or update a session's title.

        Returns True if session was found and title was set.
        Raises ValueError if title is already in use by another session,
        or if the title fails validation (too long, invalid characters).
        Empty/whitespace-only strings are normalized to None (clearing the title).
        """
        return self._set_session_title(session_id, title, only_if_empty=False)

    def set_auto_title_if_empty(self, session_id: str, title: str) -> bool:
        """Set an auto-generated title only when the current title is NULL.

        The predicate and write run in one transaction so a concurrent manual
        rename cannot be overwritten. Validation and uniqueness behavior match
        :meth:`set_session_title`.
        """
        return self._set_session_title(session_id, title, only_if_empty=True)

    def get_session_title(self, session_id: str) -> Optional[str]:
        """Get the title for a session, or None."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
        return row["title"] if row else None

    def set_session_archived(self, session_id: str, archived: bool) -> bool:
        """Archive or unarchive a session.

        Archived sessions are hidden from the default session list but keep all
        their messages — this is a soft hide, not a delete. For compression
        chains, archive the whole logical conversation. Desktop lists compression
        roots projected forward to their latest continuation; updating only the
        displayed tip lets the still-unarchived root resurrect it on refresh.
        Returns True when at least one row was updated.
        """
        def _do(conn):
            cursor = conn.execute(
                """
                WITH RECURSIVE
                  ancestors(id) AS (
                    SELECT ?
                    UNION
                    SELECT parent.id
                    FROM ancestors a
                    JOIN sessions child ON child.id = a.id
                    JOIN sessions parent ON parent.id = child.parent_session_id
                    WHERE parent.end_reason = 'compression'
                  ),
                  descendants(id) AS (
                    SELECT ?
                    UNION
                    SELECT child.id
                    FROM descendants d
                    JOIN sessions parent ON parent.id = d.id
                    JOIN sessions child ON child.parent_session_id = parent.id
                    WHERE parent.end_reason = 'compression'
                  ),
                  lineage(id) AS (
                    SELECT id FROM ancestors
                    UNION
                    SELECT id FROM descendants
                  )
                UPDATE sessions
                SET archived = ?
                WHERE id IN (SELECT id FROM lineage)
                """,
                (session_id, session_id, 1 if archived else 0),
            )
            rowcount = cursor.rowcount
            if rowcount is None or rowcount < 0:
                rowcount = conn.execute("SELECT changes()").fetchone()[0]
            return rowcount
        rowcount = self._execute_write(_do)
        return rowcount > 0

    def set_session_pinned(self, session_id: str, pinned: bool) -> bool:
        """Pin or unpin a session (and its whole compression lineage).

        ``pinned`` is a durable "keep" flag: pinned sessions are exempt from
        the ``sessions.auto_archive`` stale sweep (see
        :meth:`archive_stale_sessions`). Desktop is the current writer — its
        sidebar pins mirror here so a backend/other-surface sweep honours
        them. Like :meth:`set_session_archived` the whole compression chain is
        flipped as a unit, so pinning the surfaced tip protects the root (and
        vice-versa) no matter which id the caller holds. Returns True when at
        least one row changed.
        """
        def _do(conn):
            cursor = conn.execute(
                """
                WITH RECURSIVE
                  ancestors(id) AS (
                    SELECT ?
                    UNION
                    SELECT parent.id
                    FROM ancestors a
                    JOIN sessions child ON child.id = a.id
                    JOIN sessions parent ON parent.id = child.parent_session_id
                    WHERE parent.end_reason = 'compression'
                  ),
                  descendants(id) AS (
                    SELECT ?
                    UNION
                    SELECT child.id
                    FROM descendants d
                    JOIN sessions parent ON parent.id = d.id
                    JOIN sessions child ON child.parent_session_id = parent.id
                    WHERE parent.end_reason = 'compression'
                  ),
                  lineage(id) AS (
                    SELECT id FROM ancestors
                    UNION
                    SELECT id FROM descendants
                  )
                UPDATE sessions
                SET pinned = ?
                WHERE id IN (SELECT id FROM lineage)
                """,
                (session_id, session_id, 1 if pinned else 0),
            )
            rowcount = cursor.rowcount
            if rowcount is None or rowcount < 0:
                rowcount = conn.execute("SELECT changes()").fetchone()[0]
            return rowcount
        rowcount = self._execute_write(_do)
        return rowcount > 0

    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Look up a session by exact title. Returns session dict or None."""
        with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE title = ?", (title,)
            )
            row = cursor.fetchone()
        return dict(row) if row else None

    def resolve_session_by_title(self, title: str) -> Optional[str]:
        """Resolve a title to a session ID, preferring the latest in a lineage.

        If the exact title exists, returns that session's ID.
        If not, searches for "title #N" variants and returns the latest one.
        If the exact title exists AND numbered variants exist, returns the
        latest numbered variant (the most recent continuation).
        """
        # First try exact match
        exact = self.get_session_by_title(title)

        # Also search for numbered variants: "title #2", "title #3", etc.
        # Escape SQL LIKE wildcards (%, _) in the title to prevent false matches
        escaped = title.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, title, started_at FROM sessions "
                "WHERE title LIKE ? ESCAPE '\\' ORDER BY started_at DESC",
                (f"{escaped} #%",),
            )
            numbered = cursor.fetchall()

        if numbered:
            # Return the most recent numbered variant
            return numbered[0]["id"]
        elif exact:
            return exact["id"]
        return None

    def get_next_title_in_lineage(self, base_title: str) -> str:
        """Generate the next title in a lineage (e.g., "my session" → "my session #2").

        Strips any existing " #N" suffix to find the base name, then finds
        the highest existing number and increments.
        """
        # Strip existing #N suffix to find the true base
        match = re.match(r'^(.*?) #(\d+)$', base_title)
        if match:
            base = match.group(1)
        else:
            base = base_title

        # Find all existing numbered variants
        # Escape SQL LIKE wildcards (%, _) in the base to prevent false matches
        escaped = base.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        with self._lock:
            cursor = self._conn.execute(
                "SELECT title FROM sessions WHERE title = ? OR title LIKE ? ESCAPE '\\'",
                (base, f"{escaped} #%"),
            )
            existing = [row["title"] for row in cursor.fetchall()]

        if not existing:
            return base  # No conflict, use the base name as-is

        # Find the highest number
        max_num = 1  # The unnumbered original counts as #1
        for t in existing:
            m = re.match(r'^.* #(\d+)$', t)
            if m:
                max_num = max(max_num, int(m.group(1)))

        return f"{base} #{max_num + 1}"

    def get_compression_tip(self, session_id: str) -> Optional[str]:
        """Walk the compression-continuation chain forward and return the tip.

        A compression continuation is a child of a session whose
        ``end_reason = 'compression'``.  Older builds tried to distinguish
        continuations from branches/subagents by requiring
        ``child.started_at >= parent.ended_at``.  That ordering is too brittle:
        gateway + compression races can insert the real continuation row before
        the parent row's ``ended_at`` is written, while a stale websocket later
        creates/reuses a sibling that *does* satisfy the timestamp test.  The
        visible symptom is brutal: desktop resume follows the stale sibling and
        the user's latest messages look "lost" even though they are persisted in
        the real continuation chain.

        Instead, only follow children of compression-ended parents, exclude
        explicit branch/delegate/tool children, and prefer children that are
        themselves continuing the compression chain (``end_reason='compression'``)
        or still live over stale closed siblings such as ``ws_orphan_reap``.
        Returns the latest continuation tip, or the input id when no
        continuation exists.
        """
        current = session_id
        seen = {current} if current else set()
        # Bound the walk defensively — compression chains this deep are
        # pathological and shouldn't happen in practice. 100 = plenty.
        for _ in range(100):
            with self._lock:
                cursor = self._conn.execute(
                    """
                    SELECT child.id
                    FROM sessions parent
                    JOIN sessions child ON child.parent_session_id = parent.id
                    WHERE parent.id = ?
                      AND parent.end_reason = 'compression'
                      AND json_extract(COALESCE(child.model_config, '{}'), '$._branched_from') IS NULL
                      AND json_extract(COALESCE(child.model_config, '{}'), '$._delegate_from') IS NULL
                      AND COALESCE(child.source, '') != 'tool'
                    ORDER BY
                      CASE
                        WHEN child.end_reason = 'compression' THEN 0
                        WHEN child.ended_at IS NULL THEN 1
                        ELSE 2
                      END,
                      COALESCE(
                        (SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = child.id),
                        child.started_at
                      ) DESC,
                      child.started_at DESC,
                      child.id DESC
                    LIMIT 1
                    """,
                    (current,),
                )
                row = cursor.fetchone()
            if row is None:
                return current
            child_id = row["id"]
            if not child_id or child_id in seen:
                return current
            seen.add(child_id)
            current = child_id
        return current

    # Columns excluded from compact_rows projections: only the payload-heavy
    # blob no list consumer renders. Everything else — including gateway
    # routing fields and desktop sidebar fields like git_branch — stays, and
    # the projection is derived from SCHEMA_SQL so columns added later via
    # declarative reconciliation are included automatically instead of
    # silently dropping out of list rows.
    _SESSION_COMPACT_EXCLUDED = frozenset({"system_prompt"})
    _session_compact_cols_sql: Optional[str] = None

    @classmethod
    def _compact_session_cols(cls) -> str:
        """SELECT list for compact_rows: every ``sessions`` column declared in
        SCHEMA_SQL except the ``system_prompt`` blob, aliased with the ``s``
        prefix used by list_sessions_rich/_get_session_rich_row queries."""
        if cls._session_compact_cols_sql is None:
            declared = cls._parse_schema_columns(SCHEMA_SQL)["sessions"]
            cls._session_compact_cols_sql = ", ".join(
                f"s.{name}" for name in declared
                if name not in cls._SESSION_COMPACT_EXCLUDED
            )
        return cls._session_compact_cols_sql

    def distinct_session_cwds(self, include_archived: bool = False) -> List[Dict[str, Any]]:
        """Distinct non-empty session cwds with usage stats, for repo discovery.

        Aggregates across ALL session history (not a single page), so the desktop
        can surface every git repo the user has worked in — not just the repos
        that happen to be in the currently-loaded recents. Children/branches
        count: a worktree session is still a real workspace signal.
        """
        where = "cwd IS NOT NULL AND TRIM(cwd) != ''"
        if not include_archived:
            where += " AND archived = 0"
        with self._lock:
            rows = self._conn.execute(
                "SELECT cwd AS cwd, COUNT(*) AS sessions, "
                "MAX(COALESCE(ended_at, started_at, 0)) AS last_active "
                f"FROM sessions WHERE {where} GROUP BY cwd"
            ).fetchall()
        return [
            {
                "cwd": r["cwd"],
                "sessions": int(r["sessions"] or 0),
                "last_active": float(r["last_active"] or 0),
            }
            for r in rows
        ]

    def list_sessions_rich(
        self,
        source: str = None,
        exclude_sources: List[str] = None,
        cwd_prefix: str = None,
        limit: int = 20,
        offset: int = 0,
        include_children: bool = False,
        min_message_count: int = 0,
        project_compression_tips: bool = True,
        order_by_last_active: bool = False,
        include_archived: bool = False,
        archived_only: bool = False,
        id_query: str = None,
        search_query: str = None,
        compact_rows: bool = False,
    ) -> List[Dict[str, Any]]:
        """List sessions with preview (first user message) and last active timestamp.

        Returns dicts with keys: id, source, model, title, started_at, ended_at,
        message_count, preview (first 60 chars of first user message),
        last_active (timestamp of last message).

        Uses a single query with correlated subqueries instead of N+2 queries.

        By default, child sessions (subagent runs, compression continuations)
        are excluded.  Pass ``include_children=True`` to include them.

        With ``project_compression_tips=True`` (default), sessions that are
        roots of compression chains are projected forward to their latest
        continuation — one logical conversation = one list entry, showing the
        live continuation's id/message_count/title/last_active. This prevents
        compressed continuations from being invisible to users while keeping
        delegate subagents and branches hidden. Pass ``False`` to return the
        raw root rows (useful for admin/debug UIs).

        Pass ``order_by_last_active=True`` to sort by most-recent activity
        instead of original conversation start time. For compression chains,
        the "most-recent activity" is taken from the live tip (not the root),
        so an old conversation that was compressed and continued recently
        surfaces in the correct slot. Ordering is computed at SQL level via
        a recursive CTE that walks compression-continuation edges, so LIMIT
        and OFFSET still apply efficiently.

        ``search_query`` matches case-insensitive substrings against each
        surfaced row's title and id (and, like ``id_query``, every title/id in
        its forward compression chain). A punctuation-stripped variant is also
        matched so e.g. ``an94`` finds ``AN-94``. Only honored in the
        ``order_by_last_active`` path.

        Pass ``compact_rows=True`` for dashboard and picker callers that only
        need lightweight metadata. This omits the ``system_prompt`` blob from
        the SELECT so SQLite never copies it out of the B-tree page — a
        significant I/O saving on large databases where the blob routinely
        runs to tens of kilobytes per row.
        """
        where_clauses = []
        params = []

        if not include_children:
            # Show root sessions and branch sessions, while still hiding
            # sub-agent runs and compression continuations (which also carry a
            # parent_session_id but were spawned while the parent was still
            # live — i.e., started_at < parent.ended_at).
            #
            # Branch sessions are identified two ways, OR'd for robustness:
            #   1. A stable ``_branched_from`` marker in model_config, written
            #      by /branch at creation time. This survives the parent being
            #      reopened and re-ended with a different end_reason (e.g.
            #      tui_shutdown overwriting 'branched'), which otherwise hides
            #      the branch — see issue #20856.
            #   2. The legacy heuristic (parent ended with 'branched' before the
            #      child started), covering branch sessions created before the
            #      marker existed.
            where_clauses.append(_LISTABLE_CHILD_SQL)
            where_clauses.append(f"{_delegate_from_json('s.model_config')} IS NULL")

        if source:
            where_clauses.append("s.source = ?")
            params.append(source)
        if exclude_sources:
            placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({placeholders})")
            params.extend(exclude_sources)
        if cwd_prefix:
            clause, clause_params = _cwd_prefix_clause(cwd_prefix)
            where_clauses.append(clause)
            params.extend(clause_params)
        if min_message_count > 0:
            where_clauses.append("s.message_count >= ?")
            params.append(min_message_count)
        if archived_only:
            where_clauses.append("s.archived = 1")
        elif not include_archived:
            where_clauses.append("s.archived = 0")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Optional session-id filter, pushed into SQL so callers (Desktop
        # session-id search) don't have to fetch every row and filter in
        # Python. ``id_query`` is matched as a case-insensitive substring
        # against each surfaced row's id AND every id in its forward
        # compression chain — so searching a compression *root* id or a *tip*
        # id both resolve to the same projected conversation. Only used in the
        # order_by_last_active path (which builds the chain CTE); other callers
        # pass id_query=None.
        id_needle = (id_query or "").strip().lower()
        search_needle = (search_query or "").strip().lower()
        if order_by_last_active:
            # Compute effective_last_active by walking each surfaced session's
            # compression-continuation chain forward in SQL and taking the MAX
            # timestamp across the chain. This lets us ORDER BY + LIMIT at SQL
            # level instead of fetching every row and sorting in Python, while
            # still surfacing old compression roots whose live tip is fresh.
            #
            # The CTE seeds from rows the outer WHERE admits (roots + branch
            # children), then recursively joins forward through robust
            # compression-continuation edges. Do NOT require
            # child.started_at >= parent.ended_at here: real desktop/gateway
            # races can insert the continuation row before the parent's
            # ended_at is written, while stale websocket siblings may satisfy
            # the timestamp test and hijack resume/list projection.
            outer_where = where_sql
            id_params: List[Any] = []
            filter_clauses: List[str] = []

            def _like_pattern(needle: str) -> str:
                escaped = (
                    needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                )
                return f"%{escaped}%"

            if id_needle:
                # Admit a surfaced row if its own id or any id in its forward
                # compression chain matches the needle. LIKE with a leading
                # wildcard can't use an index, but the chain membership and
                # the small result set keep this bounded — far cheaper than
                # fetching every session and scanning in Python.
                filter_clauses.append(
                    "EXISTS (SELECT 1 FROM chain cq"
                    "        WHERE cq.root_id = s.id"
                    "          AND LOWER(cq.cur_id) LIKE ? ESCAPE '\\')"
                )
                id_params.append(_like_pattern(id_needle))
            if search_needle:
                # Same chain-membership trick as id_query, but matching either
                # the title or the id of any session in the chain. The compact
                # (punctuation-stripped) variant lets `an94` match `AN-94`.
                compact_needle = re.sub(r"[\W_]+", "", search_needle)
                compact_sql = (
                    "REPLACE(REPLACE(REPLACE(REPLACE(LOWER(COALESCE({0}, '')),"
                    " '-', ''), '_', ''), '.', ''), ' ', '')"
                )
                search_clause = (
                    "EXISTS (SELECT 1 FROM chain cq"
                    " JOIN sessions cs ON cs.id = cq.cur_id"
                    " WHERE cq.root_id = s.id"
                    " AND (LOWER(COALESCE(cs.title, '')) LIKE ? ESCAPE '\\'"
                    " OR LOWER(cq.cur_id) LIKE ? ESCAPE '\\'"
                )
                id_params.extend([_like_pattern(search_needle)] * 2)
                if compact_needle:
                    search_clause += (
                        f" OR {compact_sql.format('cs.title')} LIKE ? ESCAPE '\\'"
                    )
                    id_params.append(_like_pattern(compact_needle))
                filter_clauses.append(search_clause + "))")
            if filter_clauses:
                combined = " AND ".join(filter_clauses)
                outer_where = (
                    f"{where_sql} AND {combined}" if where_sql else f"WHERE {combined}"
                )
            _sel = self._compact_session_cols() if compact_rows else "s.*"
            query = f"""
                WITH RECURSIVE chain(root_id, cur_id) AS (
                    SELECT s.id, s.id FROM sessions s {where_sql}
                    UNION ALL
                    SELECT c.root_id, child.id
                    FROM chain c
                    JOIN sessions parent ON parent.id = c.cur_id
                    JOIN sessions child ON child.parent_session_id = c.cur_id
                    WHERE parent.end_reason = 'compression'
                      AND json_extract(COALESCE(child.model_config, '{{}}'), '$._branched_from') IS NULL
                      AND json_extract(COALESCE(child.model_config, '{{}}'), '$._delegate_from') IS NULL
                      AND COALESCE(child.source, '') != 'tool'
                ),
                chain_max AS (
                    SELECT
                        root_id,
                        MAX(COALESCE(
                            (SELECT MAX(m.timestamp) FROM messages m WHERE m.session_id = cur_id),
                            (SELECT started_at FROM sessions ss WHERE ss.id = cur_id)
                        )) AS effective_last_active
                    FROM chain
                    GROUP BY root_id
                )
                SELECT {_sel},
                    COALESCE(
                        (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw,
                    COALESCE(
                        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                        s.started_at
                    ) AS last_active,
                    COALESCE(cm.effective_last_active, s.started_at) AS _effective_last_active
                FROM sessions s
                LEFT JOIN chain_max cm ON cm.root_id = s.id
                {outer_where}
                ORDER BY _effective_last_active DESC, s.started_at DESC, s.id DESC
                LIMIT ? OFFSET ?
            """
            # WHERE params apply twice (CTE seed + outer select); the id filter
            # only applies to the outer select.
            params = params + params + id_params + [limit, offset]
        else:
            _sel = self._compact_session_cols() if compact_rows else "s.*"
            query = f"""
                SELECT {_sel},
                    COALESCE(
                        (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw,
                    COALESCE(
                        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                        s.started_at
                    ) AS last_active
                FROM sessions s
                {where_sql}
                ORDER BY s.started_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
        with self._lock:
            cursor = self._conn.execute(query, params)
            rows = cursor.fetchall()
        sessions = []
        for row in rows:
            s = dict(row)
            # Build the preview from the raw substring
            raw = s.pop("_preview_raw", "").strip()
            if raw:
                text = raw[:60]
                s["preview"] = text + ("..." if len(raw) > 60 else "")
            else:
                s["preview"] = ""
            # Drop the internal ordering column so callers see a clean dict.
            s.pop("_effective_last_active", None)
            sessions.append(s)

        # Project compression roots forward to their tips. Each row whose
        # end_reason is 'compression' has a continuation child; replace the
        # surfaced fields (id, message_count, title, last_active, ended_at,
        # end_reason, preview) with the tip's values so the list entry acts
        # as the live conversation. Keep the root's started_at to preserve
        # chronological ordering by original conversation start.
        if project_compression_tips and not include_children:
            projected = []
            for s in sessions:
                if s.get("end_reason") != "compression":
                    projected.append(s)
                    continue
                tip_id = self.get_compression_tip(s["id"])
                if tip_id == s["id"]:
                    projected.append(s)
                    continue
                tip_row = self._get_session_rich_row(tip_id, compact_rows=compact_rows)
                if not tip_row:
                    projected.append(s)
                    continue
                # Preserve the root's started_at for stable sort order, but
                # surface the tip's identity and activity data.
                merged = dict(s)
                for key in (
                    "id", "ended_at", "end_reason", "message_count",
                    "tool_call_count", "title", "last_active", "preview",
                    "model", "system_prompt", "cwd", "git_branch", "git_repo_root",
                ):
                    if key in tip_row:
                        merged[key] = tip_row[key]
                merged["_lineage_root_id"] = s["id"]
                projected.append(merged)
            sessions = projected

        return sessions

    def list_cron_job_runs(
        self,
        job_id: str,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List the run sessions produced by a single cron job, newest first.

        Cron runs are flat, independent sessions whose id is
        ``cron_{job_id}_{timestamp}`` (see ``cron/scheduler.run_job``). They are
        never compression roots and never branch, so this deliberately skips the
        ``list_sessions_rich`` recursive compression-chain CTE / leading-wildcard
        ``id_query`` path — that path seeds from *every* ``source='cron'`` row in
        the DB and only filters to one job's runs after the scan, so it scales
        with the whole cron pile (a heavy history makes the desktop run-history
        endpoint time out before it eventually populates).

        Instead this binds to one job with a ``[prefix, prefix_hi)`` range over
        the id (an index range scan, not a ``%...%`` substring), filters
        ``source='cron'``, and orders by ``started_at DESC``. Work scales with
        the requested window, not the total cron history.

        Returns the same enriched row shape as ``list_sessions_rich`` (adds
        ``preview`` + ``last_active``) so callers can reuse it.
        """
        prefix = f"cron_{job_id}_"
        # Half-open upper bound for an index range scan: increment the final
        # byte of the prefix so the range covers exactly the ids that start
        # with ``prefix`` and nothing else. ``prefix`` always ends in '_', but
        # compute it generically rather than hardcoding the successor char.
        prefix_hi = prefix[:-1] + chr(ord(prefix[-1]) + 1)

        query = """
            SELECT s.*,
                COALESCE(
                    (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                     FROM messages m
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                COALESCE(
                    (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                    s.started_at
                ) AS last_active
            FROM sessions s
            WHERE s.source = 'cron' AND s.id >= ? AND s.id < ?
            ORDER BY s.started_at DESC, s.id DESC
            LIMIT ? OFFSET ?
        """
        with self._lock:
            cursor = self._conn.execute(query, (prefix, prefix_hi, limit, offset))
            rows = cursor.fetchall()

        runs: List[Dict[str, Any]] = []
        for row in rows:
            s = dict(row)
            raw = s.pop("_preview_raw", "").strip()
            if raw:
                text = raw[:60]
                s["preview"] = text + ("..." if len(raw) > 60 else "")
            else:
                s["preview"] = ""
            runs.append(s)
        return runs

    def _get_session_rich_row(self, session_id: str, compact_rows: bool = False) -> Optional[Dict[str, Any]]:
        """Fetch a single session with the same enriched columns as
        ``list_sessions_rich`` (preview + last_active). Returns None if the
        session doesn't exist.

        Pass ``compact_rows=True`` to omit the ``system_prompt`` blob (see
        ``list_sessions_rich`` for details).
        """
        _sel = self._compact_session_cols() if compact_rows else "s.*"
        query = f"""
            SELECT {_sel},
                COALESCE(
                    (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                     FROM messages m
                     WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                COALESCE(
                    (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                    s.started_at
                ) AS last_active
            FROM sessions s
            WHERE s.id = ?
        """
        with self._lock:
            cursor = self._conn.execute(query, (session_id,))
            row = cursor.fetchone()
        if not row:
            return None
        s = dict(row)
        raw = s.pop("_preview_raw", "").strip()
        if raw:
            text = raw[:60]
            s["preview"] = text + ("..." if len(raw) > 60 else "")
        else:
            s["preview"] = ""
        return s

    # =========================================================================
    # Message storage
    # =========================================================================

    # Sentinel prefix used to distinguish JSON-encoded structured content
    # (multimodal messages: lists of parts like text + image_url) from plain
    # string content. The NUL byte is not legal in normal text, so this
    # cannot collide with real user content.
    _CONTENT_JSON_PREFIX = "\x00json:"

    @classmethod
    def _encode_content(cls, content: Any) -> Any:
        """Serialize structured (list/dict) message content for sqlite.

        sqlite3 can only bind ``str``, ``bytes``, ``int``, ``float``, and ``None``
        to query parameters. Multimodal messages have ``content`` as a list of
        parts (``[{"type": "text", ...}, {"type": "image_url", ...}]``), which
        raises ``ProgrammingError: Error binding parameter N: type 'list' is
        not supported`` when bound directly.

        Returns the value unchanged when it's already a safe scalar, or a
        sentinel-prefixed JSON string for lists/dicts. Paired with
        :meth:`_decode_content` on read.
        """
        if isinstance(content, str):
            # Lone UTF-16 surrogates reach here inside tool results scraped
            # from the web/social platforms (the same input that crashed the
            # guardrail hasher). The proactive sanitizer upstream only cleans
            # the *api_messages* copy, and the recovery sanitizer only runs
            # after the API call itself raises — which it no longer does — so
            # the canonical history keeps them and this write is where they
            # land. Left raw, sqlite3 raises UnicodeEncodeError, the flush is
            # abandoned, and the session silently stops persisting for the
            # rest of its life. Scrub so persistence never fails.
            return _sanitize_surrogates(content)
        if content is None or isinstance(content, (bytes, int, float)):
            return content
        try:
            # json.dumps defaults to ensure_ascii=True, which escapes any
            # surrogate as \udXXX — already safe to bind.
            return cls._CONTENT_JSON_PREFIX + json.dumps(content)
        except (TypeError, ValueError):
            # Last-resort fallback: stringify so persistence never fails.
            return _sanitize_surrogates(str(content))

    @classmethod
    def _decode_content(cls, content: Any) -> Any:
        """Reverse :meth:`_encode_content`; returns scalars unchanged."""
        if isinstance(content, str) and content.startswith(cls._CONTENT_JSON_PREFIX):
            try:
                return json.loads(content[len(cls._CONTENT_JSON_PREFIX):])
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    "Failed to decode JSON-encoded message content; "
                    "returning raw string"
                )
                return content
        return content

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str = None,
        tool_name: str = None,
        tool_calls: Any = None,
        tool_call_id: str = None,
        token_count: int = None,
        finish_reason: str = None,
        reasoning: str = None,
        reasoning_content: str = None,
        reasoning_details: Any = None,
        codex_reasoning_items: Any = None,
        codex_message_items: Any = None,
        platform_message_id: str = None,
        observed: bool = False,
        effect_disposition: Optional[str] = None,
        timestamp: Any = None,
        api_content: Optional[str] = None,
        display_kind: Optional[str] = None,
        display_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Append a message to a session. Returns the message row ID.

        Also increments the session's message_count (and tool_call_count
        if role is 'tool' or tool_calls is present).

        ``platform_message_id`` is the external messaging platform's own
        message ID (e.g. Telegram update_id, Yuanbao msg_id).  It is
        independent of the SQLite autoincrement primary key and is used by
        platform-specific flows like yuanbao's recall guard to redact a
        message by its platform-side identifier.

        ``api_content`` is the exact content string sent to the API for this
        message when it differs from ``content`` (ephemeral memory/plugin
        injections, persist overrides).  It is a byte-fidelity sidecar for
        prompt-cache-stable replay — stored as sent, except lone surrogates
        (which sqlite3 cannot bind and which the conversation loop scrubs
        from every outgoing payload anyway, so the scrubbed form IS the
        wire bytes).
        """
        # Display metadata is presentation-only and never changes the model
        # context role/content replayed to providers.
        display_metadata_json = json.dumps(display_metadata) if display_metadata else None
        # Serialize structured fields to JSON before entering the write txn
        reasoning_details_json = (
            json.dumps(reasoning_details)
            if reasoning_details else None
        )
        codex_items_json = (
            json.dumps(codex_reasoning_items)
            if codex_reasoning_items else None
        )
        codex_message_items_json = (
            json.dumps(codex_message_items)
            if codex_message_items else None
        )
        # tool_calls may arrive as a Python list (from the live agent) or
        # as a JSON string (from import/export). Parse first to avoid
        # double-encoding.
        if isinstance(tool_calls, str):
            try:
                tool_calls = json.loads(tool_calls)
            except (json.JSONDecodeError, TypeError):
                tool_calls = []
        tool_calls_json = json.dumps(tool_calls) if tool_calls else None
        # Multimodal content (list of parts) must be JSON-encoded: sqlite3
        # cannot bind list/dict parameters directly.
        stored_content = self._encode_content(content)

        message_timestamp = time.time()
        if timestamp is not None:
            try:
                if hasattr(timestamp, "timestamp"):
                    message_timestamp = float(timestamp.timestamp())
                else:
                    message_timestamp = float(timestamp)
            except (TypeError, ValueError):
                logger.debug("Ignoring invalid explicit message timestamp: %r", timestamp)

        # Pre-compute tool call count
        num_tool_calls = 0
        if tool_calls is not None:
            num_tool_calls = len(tool_calls) if isinstance(tool_calls, list) else 1

        def _do(conn):
            cursor = conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items, platform_message_id, observed, active, api_content, display_kind, display_metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    stored_content,
                    tool_call_id,
                    tool_calls_json,
                    _scrub_surrogates(tool_name),
                    effect_disposition,
                    message_timestamp,
                    token_count,
                    finish_reason,
                    _scrub_surrogates(reasoning),
                    _scrub_surrogates(reasoning_content),
                    reasoning_details_json,
                    codex_items_json,
                    codex_message_items_json,
                    platform_message_id,
                    1 if observed else 0,
                    1,
                    _scrub_surrogates(api_content) if isinstance(api_content, str) else None,
                    _scrub_surrogates(display_kind) if isinstance(display_kind, str) else None,
                    display_metadata_json,
                ),
            )
            msg_id = cursor.lastrowid

            # Update counters
            if num_tool_calls > 0:
                conn.execute(
                    """UPDATE sessions SET message_count = message_count + 1,
                       tool_call_count = tool_call_count + ? WHERE id = ?""",
                    (num_tool_calls, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET message_count = message_count + 1 WHERE id = ?",
                    (session_id,),
                )
            return msg_id

        return self._execute_write(_do)

    def set_latest_matching_message_display_kind(
        self, session_id: str, *, role: str, content: str, display_kind: str,
        display_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Stamp presentation metadata on this turn's freshly persisted row.

        The model still receives ``role`` and ``content`` unchanged. Gateway and
        CLI synthetic inputs call this immediately after their serial turn has
        flushed, preserving producer provenance without classifying by content
        during transcript rendering.
        """
        if not session_id or not content or not display_kind:
            return False

        def _do(conn):
            row = conn.execute(
                "SELECT id FROM messages WHERE session_id = ? AND role = ? "
                "AND content = ? AND active = 1 ORDER BY id DESC LIMIT 1",
                (session_id, role, self._encode_content(content)),
            ).fetchone()
            if row is None:
                return False
            conn.execute(
                "UPDATE messages SET display_kind = ?, display_metadata = ? WHERE id = ?",
                (
                    _scrub_surrogates(display_kind),
                    json.dumps(display_metadata) if display_metadata else None,
                    row[0],
                ),
            )
            return True

        return bool(self._execute_write(_do))

    def _insert_message_rows(self, conn, session_id: str, messages: List[Dict[str, Any]]) -> tuple[int, int]:
        """Insert *messages* as fresh active rows for *session_id*.

        Shared by :meth:`replace_messages` (delete-then-insert) and
        :meth:`archive_and_compact` (soft-archive-then-insert). Runs inside the
        caller's write transaction (takes the live ``conn``). Returns
        ``(inserted_count, tool_call_count)``. Does NOT touch sessions.* counters
        — the caller owns that, since the two flows reconcile counts differently.
        """
        now_ts = time.time()
        inserted = 0
        tool_calls_total = 0
        for msg in messages:
            role = msg.get("role", "unknown")
            tool_calls = msg.get("tool_calls")
            message_timestamp = now_ts
            if msg.get("timestamp") is not None:
                try:
                    ts_value = msg.get("timestamp")
                    if hasattr(ts_value, "timestamp"):
                        message_timestamp = float(ts_value.timestamp())
                    else:
                        message_timestamp = float(ts_value)
                except (TypeError, ValueError):
                    logger.debug("Ignoring invalid explicit message timestamp: %r", msg.get("timestamp"))
            reasoning_details = msg.get("reasoning_details") if role == "assistant" else None
            codex_reasoning_items = (
                msg.get("codex_reasoning_items") if role == "assistant" else None
            )
            codex_message_items = (
                msg.get("codex_message_items") if role == "assistant" else None
            )
            reasoning_details_json = (
                json.dumps(reasoning_details) if reasoning_details else None
            )
            codex_items_json = (
                json.dumps(codex_reasoning_items) if codex_reasoning_items else None
            )
            codex_message_items_json = (
                json.dumps(codex_message_items) if codex_message_items else None
            )
            # tool_calls may arrive as a Python list (from the live agent)
            # or as a JSON string (from import_sessions / export_session,
            # which store it as TEXT). json.dumps on an already-serialized
            # string double-encodes it, so parse first.
            if isinstance(tool_calls, str):
                try:
                    tool_calls = json.loads(tool_calls)
                except (json.JSONDecodeError, TypeError):
                    tool_calls = []
            tool_calls_json = json.dumps(tool_calls) if tool_calls else None
            # Accept either `platform_message_id` (new explicit name) or
            # `message_id` (yuanbao's existing convention on message dicts).
            platform_msg_id = (
                msg.get("platform_message_id") or msg.get("message_id")
            )

            api_content = msg.get("api_content")

            conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items, platform_message_id, observed, active, api_content, display_kind, display_metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    role,
                    self._encode_content(msg.get("content")),
                    msg.get("tool_call_id"),
                    tool_calls_json,
                    _scrub_surrogates(msg.get("tool_name")),
                    msg.get("effect_disposition"),
                    message_timestamp,
                    msg.get("token_count"),
                    msg.get("finish_reason"),
                    _scrub_surrogates(msg.get("reasoning")) if role == "assistant" else None,
                    _scrub_surrogates(msg.get("reasoning_content")) if role == "assistant" else None,
                    reasoning_details_json,
                    codex_items_json,
                    codex_message_items_json,
                    platform_msg_id,
                    1 if msg.get("observed") else 0,
                    1,
                    _scrub_surrogates(api_content) if isinstance(api_content, str) else None,
                    _scrub_surrogates(msg.get("display_kind")) if isinstance(msg.get("display_kind"), str) else None,
                    json.dumps(msg["display_metadata"]) if msg.get("display_metadata") else None,
                ),
            )
            inserted += 1
            if tool_calls is not None:
                tool_calls_total += (
                    len(tool_calls) if isinstance(tool_calls, list) else 1
                )
            now_ts = max(now_ts + 1e-6, message_timestamp + 1e-6)
        return inserted, tool_calls_total

    def replace_messages(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        active_only: bool = False,
    ) -> None:
        """Atomically replace the stored messages for a session.

        Used by transcript-rewrite flows such as /retry, /undo, and /compress.
        The delete + reinsert sequence must commit as one transaction so a
        mid-rewrite failure does not leave SQLite with a partial transcript.

        DESTRUCTIVE by default: every row for the session is DELETEd (and drops
        out of the FTS index). For compaction that must preserve the
        pre-compaction transcript under the same id, use
        :meth:`archive_and_compact` instead.

        Pass ``active_only=True`` to replace ONLY the live (``active = 1``) rows,
        leaving soft-archived rows (``active = 0`` — e.g. the ``compacted = 1``
        turns that :meth:`archive_and_compact` keeps on disk for #38763
        durability, or rewind/undo rows) untouched. Callers that share a session
        id with an agent already running in-place compaction must use this so a
        full-history rewrite doesn't wipe the rows the agent deliberately
        archived. ``message_count``/``tool_call_count`` then track the live set,
        matching :meth:`archive_and_compact`.
        """

        active_clause = " AND active = 1" if active_only else ""

        def _do(conn):
            conn.execute(
                f"DELETE FROM messages WHERE session_id = ?{active_clause}",
                (session_id,),
            )
            conn.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
                (session_id,),
            )
            total_messages, total_tool_calls = self._insert_message_rows(
                conn, session_id, messages
            )
            conn.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                (total_messages, total_tool_calls, session_id),
            )

        self._execute_write(_do)

    def has_archived_messages(self, session_id: str) -> bool:
        """Return True if the session has any soft-archived (``active = 0``) rows.

        Used by callers (e.g. the ACP adapter's ``_persist``) that must decide
        whether a full-history :meth:`replace_messages` would destroy durable
        compaction-archived turns. Cheap existence probe — does not load rows.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT 1 FROM messages WHERE session_id = ? AND active = 0 LIMIT 1",
                (session_id,),
            )
            return cursor.fetchone() is not None

    def archive_and_compact(
        self, session_id: str, compacted_messages: List[Dict[str, Any]]
    ) -> int:
        """Non-destructive in-place compaction for a single durable session id.

        Soft-archives every currently-active message (``active = 0``) and
        inserts *compacted_messages* as fresh active rows — atomically, in one
        write transaction. The conversation keeps ONE session id for life
        (#38763) WITHOUT destroying history:

        - The live-context load (:meth:`get_messages_as_conversation`,
          :meth:`get_messages`) filters ``active = 1`` by default, so the model
          reloads ONLY the compacted set.
        - The archived pre-compaction turns stay on disk (active=0) and stay
          DISCOVERABLE: they are marked compacted=1, and search_messages()
          includes compacted=1 rows by default — so session_search still finds
          them, unlike rewind/undo rows (active=0, compacted=0) which stay
          hidden. They remain in the FTS index (the messages_fts* triggers
          index on INSERT / drop on DELETE and don't key on active/compacted;
          flipping to active=0 is a content-preserving UPDATE) and are
          recoverable via get_messages(..., include_inactive=True).

        This is the durability-preserving alternative to :meth:`replace_messages`
        for compaction. ``message_count`` is set to the ACTIVE (compacted) count,
        matching what the live load returns. Returns the new active count.
        """

        def _do(conn):
            # Soft-archive the live turns: active=0 hides them from the live
            # context load, compacted=1 marks them as "summarized away" (vs
            # rewind/undo's active=0+compacted=0, which means "user took it
            # back"). search_messages includes compacted=1 rows by default so
            # the pre-compaction transcript stays discoverable; live-context
            # loads (active=1 only) still exclude them.
            conn.execute(
                "UPDATE messages SET active = 0, compacted = 1 "
                "WHERE session_id = ? AND active = 1",
                (session_id,),
            )
            inserted, tool_calls_total = self._insert_message_rows(
                conn, session_id, compacted_messages
            )
            # message_count / tool_call_count reflect the LIVE (active) set —
            # the archived rows are still on disk but not part of the live count.
            conn.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                (inserted, tool_calls_total, session_id),
            )
            return inserted

        return self._execute_write(_do)

    def set_latest_user_api_content(
        self, session_id: str, content: Any, api_content: str
    ) -> int:
        """Backfill the ``api_content`` sidecar onto the newest ACTIVE user row.

        In-place preflight compaction (:meth:`archive_and_compact`) inserts the
        current turn's user row BEFORE the turn prologue composes the
        prefetch/plugin sidecar, and the subsequent crash persist identity-skips
        every compacted dict — without this backfill the stamped sidecar would
        never land in the DB and any reload would replay clean content,
        re-introducing the prompt-cache divergence the sidecar exists to close.

        The ``content`` match is a defensive guard: if the newest active user
        row is not the message the caller stamped (racing rewrite, unexpected
        tail shape), nothing is written. Returns the number of rows updated
        (0 or 1).
        """
        encoded = self._encode_content(content)

        def _do(conn):
            cursor = conn.execute(
                "UPDATE messages SET api_content = ? WHERE id = ("
                "SELECT id FROM messages "
                "WHERE session_id = ? AND role = 'user' AND active = 1 "
                "ORDER BY id DESC LIMIT 1"
                ") AND content IS ?",
                (_scrub_surrogates(api_content), session_id, encoded),
            )
            return cursor.rowcount

        return self._execute_write(_do)

    def get_messages(
        self,
        session_id: str,
        include_inactive: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Load messages for a session in insertion order.

        By default only active messages are returned. Pass
        ``include_inactive=True`` to load soft-deleted rows (e.g. for
        audit / debug views of rewound history). See
        :meth:`rewind_to_message` for the soft-delete mechanic.

        Ordered by AUTOINCREMENT id (true insertion order) rather than
        timestamp — see c03acca50 for the WSL2 clock-regression rationale.

        When ``limit`` is provided, returns at most ``limit`` messages
        starting from ``offset`` (0-based, in insertion order). Enables
        pagination for the API endpoint to avoid loading entire transcripts.
        ``offset`` alone (without ``limit``) also pages — SQLite requires a
        LIMIT clause for OFFSET, so it's emitted as ``LIMIT -1`` (unbounded).
        """
        active_clause = "" if include_inactive else " AND active = 1"
        sql = (
            "SELECT * FROM messages WHERE session_id = ?"
            f"{active_clause} ORDER BY id"
        )
        params: list = [session_id]
        if limit is not None or offset:
            # SQLite's OFFSET requires LIMIT; -1 means "no limit".
            sql += " LIMIT ? OFFSET ?"
            params.extend([-1 if limit is None else limit, offset])
        with self._lock:
            cursor = self._conn.execute(sql, params)
            rows = cursor.fetchall()
        result = []
        for row in rows:
            msg = dict(row)
            if "content" in msg:
                msg["content"] = self._decode_content(msg["content"])
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to deserialize tool_calls in get_messages, falling back to []")
                    msg["tool_calls"] = []
            result.append(msg)
        return result

    def get_messages_around(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
    ) -> Dict[str, Any]:
        """Load a window of messages anchored on a specific message id.

        Returns a dict with:
          - ``window``: up to ``window`` messages before the anchor, the anchor
            itself, and up to ``window`` messages after, ordered by id ascending.
          - ``messages_before``: count of messages strictly before the anchor
            still in the session (== window unless we hit the start).
          - ``messages_after``: count of messages strictly after the anchor
            still in the session (== window unless we hit the end).

        Used by ``session_search`` for both the discovery shape (anchored on the
        FTS5 match) and the scroll shape (anchored on any message id). The
        ``messages_before`` / ``messages_after`` counts let the caller detect
        session boundaries: when either is less than ``window``, the agent has
        reached one end of the session.

        Returns an empty window when ``around_message_id`` is not a real id in
        ``session_id`` — callers decide how to surface that.
        """
        if window < 0:
            window = 0
        with self._lock:
            # Confirm the anchor exists in this session.
            anchor_exists = self._conn.execute(
                "SELECT 1 FROM messages WHERE id = ? AND session_id = ? LIMIT 1",
                (around_message_id, session_id),
            ).fetchone()
            if not anchor_exists:
                return {"window": [], "messages_before": 0, "messages_after": 0}

            # Two queries: anchor + before (DESC, take window+1), and after
            # (ASC, take window). Final order is id ASC.
            before_rows = self._conn.execute(
                "SELECT * FROM messages "
                "WHERE session_id = ? AND id <= ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, around_message_id, window + 1),
            ).fetchall()
            after_rows = self._conn.execute(
                "SELECT * FROM messages "
                "WHERE session_id = ? AND id > ? "
                "ORDER BY id ASC LIMIT ?",
                (session_id, around_message_id, window),
            ).fetchall()

        # before_rows is DESC; reverse so it's ASC, then concatenate after_rows.
        rows = list(reversed(before_rows)) + list(after_rows)
        result = []
        for row in rows:
            msg = dict(row)
            if "content" in msg:
                msg["content"] = self._decode_content(msg["content"])
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Failed to deserialize tool_calls in get_messages_around, falling back to []"
                    )
                    msg["tool_calls"] = []
            result.append(msg)

        # before_rows includes the anchor itself; subtract 1 for the count of
        # messages strictly before the anchor in the returned slice.
        messages_before = max(0, len(before_rows) - 1)
        messages_after = len(after_rows)
        return {
            "window": result,
            "messages_before": messages_before,
            "messages_after": messages_after,
        }

    def get_anchored_view(
        self,
        session_id: str,
        around_message_id: int,
        window: int = 5,
        bookend: int = 3,
        keep_roles: Optional[Tuple[str, ...]] = ("user", "assistant"),
    ) -> Dict[str, Any]:
        """Return an anchored window plus session bookends.

        Built on top of ``get_messages_around``. Three slices:

          - ``window``: messages immediately surrounding the anchor. Filtered
            to ``keep_roles`` (tool-response noise dropped by default), EXCEPT
            the anchor itself is always preserved regardless of role.
          - ``bookend_start``: first ``bookend`` user/assistant messages of the
            session — but only those whose id is strictly before the window's
            first message id. Empty when the window already overlaps the
            session head. Empty-content messages (tool-call-only assistant
            turns) are skipped so they don't crowd out actual prose openings.
          - ``bookend_end``: last ``bookend`` user/assistant messages of the
            session, same non-overlap rule at the tail.

        Bookends let an FTS5 hit anywhere in a long session yield the goal
        (opening) and the resolution (closing) on a single call — without
        loading the whole transcript.

        Returns ``{"window": [], "messages_before": 0, "messages_after": 0,
        "bookend_start": [], "bookend_end": []}`` when the anchor isn't in
        the session.

        ``keep_roles=None`` disables role filtering (raw window + raw
        bookends).
        """
        if bookend < 0:
            bookend = 0

        # Reuse the primitive — handles anchor-existence, content decoding,
        # tool_calls deserialisation, and boundary counts.
        primitive = self.get_messages_around(
            session_id, around_message_id, window=window
        )
        window_rows = primitive["window"]
        if not window_rows:
            return {
                "window": [],
                "messages_before": 0,
                "messages_after": 0,
                "bookend_start": [],
                "bookend_end": [],
            }

        # Apply role filter to the window, but never drop the anchor itself.
        if keep_roles is not None:
            keep_set = set(keep_roles)
            filtered_window = [
                m for m in window_rows
                if m.get("id") == around_message_id or m.get("role") in keep_set
            ]
        else:
            filtered_window = window_rows

        window_min_id = window_rows[0]["id"]
        window_max_id = window_rows[-1]["id"]

        # Fetch bookends only when there's room outside the window. SQL filters
        # by id range, role, and non-empty content — tool-call-only assistant
        # turns (content='' with tool_calls populated) are excluded so they
        # don't crowd out actual prose openings/closings.
        bookend_start_rows: List[Any] = []
        bookend_end_rows: List[Any] = []
        if bookend > 0:
            with self._lock:
                role_clause = ""
                role_params: list = []
                if keep_roles is not None:
                    role_placeholders = ",".join("?" for _ in keep_roles)
                    role_clause = f" AND role IN ({role_placeholders})"
                    role_params = list(keep_roles)

                bookend_start_rows = self._conn.execute(
                    f"SELECT * FROM messages "
                    f"WHERE session_id = ? AND id < ?{role_clause} "
                    f"AND length(content) > 0 "
                    f"ORDER BY id ASC LIMIT ?",
                    (session_id, window_min_id, *role_params, bookend),
                ).fetchall()

                bookend_end_rows = self._conn.execute(
                    f"SELECT * FROM messages "
                    f"WHERE session_id = ? AND id > ?{role_clause} "
                    f"AND length(content) > 0 "
                    f"ORDER BY id DESC LIMIT ?",
                    (session_id, window_max_id, *role_params, bookend),
                ).fetchall()
                # End rows came back DESC for the LIMIT cap; flip to ASC.
                bookend_end_rows = list(reversed(bookend_end_rows))

        def _hydrate(row) -> Dict[str, Any]:
            msg = dict(row)
            if "content" in msg:
                msg["content"] = self._decode_content(msg["content"])
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(
                        "Failed to deserialize tool_calls in get_anchored_view, falling back to []"
                    )
                    msg["tool_calls"] = []
            return msg

        return {
            "window": filtered_window,
            "messages_before": primitive["messages_before"],
            "messages_after": primitive["messages_after"],
            "bookend_start": [_hydrate(r) for r in bookend_start_rows],
            "bookend_end": [_hydrate(r) for r in bookend_end_rows],
        }

    def resolve_resume_session_id(self, session_id: str) -> str:
        """Redirect a resume target to the descendant session that holds the messages.

        Context compression ends the current session and forks a new child session
        (linked via ``parent_session_id``). The flush cursor is reset, so the
        child is where new messages actually land — the parent ends up with
        ``message_count = 0`` rows unless messages had already been flushed to
        it before compression. See #15000.

        This helper walks ``parent_session_id`` forward from ``session_id`` and
        returns the descendant in the chain that has the **most recent** messages.
        Unlike the original logic, it does NOT short-circuit when the starting
        session already has messages — a descendant that was created by
        compression may hold the continuation content and should be preferred
        by the WebUI and gateway for ``--resume`` and session loading.

        If no descendant (including the starting session) has any messages,
        the original ``session_id`` is returned unchanged.

        The chain is always walked via the child whose ``started_at`` is
        latest; that matches the single-chain shape that compression creates.
        A depth cap (32) guards against accidental loops in malformed data.
        """
        if not session_id:
            return session_id

        # Follow the compression-continuation chain forward to the live tip
        # FIRST. Auto-compression ends the current session and forks a
        # continuation child, but a long-lived parent keeps its own flushed
        # message rows — so the empty-head walk below never redirects it, and
        # resuming the parent id reloads the pre-compression transcript while
        # the turns generated *after* compression (and their responses) sit in
        # the continuation. ``get_compression_tip`` is lineage-aware: it only
        # follows children whose parent ended with ``end_reason='compression'``
        # (created after the parent was ended), so delegation / branch children
        # never hijack the resume. This is the fix for the desktop "I came back
        # and the reply isn't there" report on large sessions.
        try:
            tip = self.get_compression_tip(session_id)
        except Exception:
            tip = session_id
        if tip and tip != session_id:
            session_id = tip

        with self._lock:
            current = session_id
            seen = {current}
            best = None  # tracks the last (deepest) node with messages

            for _ in range(32):
                # Check if the current node has messages.
                try:
                    row = self._conn.execute(
                        "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                        (current,),
                    ).fetchone()
                except Exception:
                    return session_id
                if row is not None:
                    best = current

                # Walk to the most-recently-started child — but skip explicit
                # branch (`_branched_from`), delegate/subagent (`_delegate_from`),
                # and tool children. They also carry a ``parent_session_id`` yet
                # are NOT compression continuations; following them would hijack
                # the resume target to an unrelated session (e.g. a subagent
                # run). This mirrors the child-exclusion in ``get_compression_tip``.
                try:
                    child_row = self._conn.execute(
                        "SELECT id FROM sessions "
                        "WHERE parent_session_id = ? "
                        "  AND json_extract(COALESCE(model_config, '{}'), '$._branched_from') IS NULL "
                        "  AND json_extract(COALESCE(model_config, '{}'), '$._delegate_from') IS NULL "
                        "  AND COALESCE(source, '') != 'tool' "
                        "ORDER BY started_at DESC, id DESC LIMIT 1",
                        (current,),
                    ).fetchone()
                except Exception:
                    return session_id
                if child_row is None:
                    break
                child_id = child_row["id"] if hasattr(child_row, "keys") else child_row[0]
                if not child_id or child_id in seen:
                    break
                seen.add(child_id)
                current = child_id

            return best if best is not None else session_id

    def get_messages_as_conversation(
        self,
        session_id: str,
        include_ancestors: bool = False,
        include_inactive: bool = False,
        repair_alternation: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Load messages in the OpenAI conversation format (role + content dicts).
        Used by the gateway to restore conversation history.

        By default only active messages are returned. Pass
        ``include_inactive=True`` to load soft-deleted (rewound) rows
        as well. See :meth:`rewind_to_message`.

        ``repair_alternation=True`` runs ``repair_message_sequence`` over the
        loaded list before returning it. Callers that restore a session for
        LIVE REPLAY should pass it: a durable alternation violation (e.g. a
        ``user;user`` pair left by a turn that persisted no assistant row)
        otherwise re-triggers the pre-request defensive repair on every
        single request for the rest of the session's life — the repair
        mutates only the per-request list, never the stored transcript.
        Inspection/export consumers keep the default and see the transcript
        verbatim.
        """
        session_ids = [session_id]
        if include_ancestors:
            session_ids = self._session_lineage_root_to_tip(session_id)

        active_clause = "" if include_inactive else " AND active = 1"
        with self._lock:
            placeholders = ",".join("?" for _ in session_ids)
            rows = self._conn.execute(
                "SELECT role, content, tool_call_id, tool_calls, tool_name, effect_disposition, "
                "finish_reason, reasoning, reasoning_content, reasoning_details, "
                "codex_reasoning_items, codex_message_items, platform_message_id, observed, timestamp, "
                "api_content, display_kind, display_metadata "
                f"FROM messages WHERE session_id IN ({placeholders})"
                # Order by AUTOINCREMENT id (true insertion order), NOT timestamp:
                # append_message stamps rows with time.time(), which is not
                # monotonic (WSL2, NTP steps, VM/laptop sleep resume). A later
                # row can carry an earlier timestamp than its predecessor, and
                # ORDER BY timestamp would then sort an assistant tool_calls row
                # after its tool response, breaking tool-call/response adjacency
                # and triggering an HTTP 400 on replay. This matches get_messages
                # — see c03acca50 for the original fix.
                f"{active_clause} ORDER BY id",
                tuple(session_ids),
            ).fetchall()

        return self._rows_to_conversation(
            rows,
            session_id=session_id,
            include_ancestors=include_ancestors,
            repair_alternation=repair_alternation,
        )

    # Columns every conversation projection decodes. Shared by
    # get_messages_as_conversation and get_resume_conversations so a single
    # SELECT can feed both the model-fed and display views.
    _CONVERSATION_ROW_COLUMNS = (
        "role, content, tool_call_id, tool_calls, tool_name, effect_disposition, "
        "finish_reason, reasoning, reasoning_content, reasoning_details, "
        "codex_reasoning_items, codex_message_items, platform_message_id, observed, timestamp, "
        "api_content, display_kind, display_metadata"
    )

    def _rows_to_conversation(
        self,
        rows,
        *,
        session_id: str,
        include_ancestors: bool,
        repair_alternation: bool,
    ) -> List[Dict[str, Any]]:
        """Decode fetched message rows into the OpenAI conversation format.

        Extracted from get_messages_as_conversation so get_resume_conversations
        can build the model-fed and display views from one SELECT. ``rows`` must
        already be ordered by ``id`` (insertion order) and filtered to the
        desired session set / active state by the caller.
        """
        messages = []
        for row in rows:
            content = self._decode_content(row["content"])
            if row["role"] in {"user", "assistant"} and isinstance(content, str):
                content = sanitize_context(content).strip()
            msg = {"role": row["role"], "content": content}
            # api_content is the byte-fidelity sidecar: the exact string sent
            # to the API when it differed from the clean content. Returned
            # VERBATIM — no sanitize_context, no strip — because the replay
            # path substitutes it for content to keep the provider prompt
            # cache prefix byte-stable across turns. Cleaning it here would
            # re-introduce the divergence it exists to remove.
            if row["api_content"]:
                msg["api_content"] = row["api_content"]
            if row["display_kind"]:
                msg["display_kind"] = row["display_kind"]
            if row["display_metadata"]:
                try:
                    msg["display_metadata"] = json.loads(row["display_metadata"])
                except (TypeError, json.JSONDecodeError):
                    logger.warning("Ignoring invalid display metadata on message row")
            if row["timestamp"]:
                msg["timestamp"] = row["timestamp"]
            if row["tool_call_id"]:
                msg["tool_call_id"] = row["tool_call_id"]
            if row["tool_name"]:
                msg["tool_name"] = row["tool_name"]
            if row["effect_disposition"]:
                msg["effect_disposition"] = row["effect_disposition"]
            if row["tool_calls"]:
                try:
                    msg["tool_calls"] = json.loads(row["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to deserialize tool_calls in conversation replay, falling back to []")
                    msg["tool_calls"] = []
            # Surface the platform-side message id (e.g. yuanbao msg_id,
            # telegram update_id) so platform-specific flows like recall
            # can match by external identifier instead of having to fall
            # back to content-match heuristics.  Exposed as ``message_id``
            # for backward compatibility with the JSONL transcript shape.
            if row["platform_message_id"]:
                msg["message_id"] = row["platform_message_id"]
            if row["observed"]:
                msg["observed"] = True
            # Restore reasoning fields on assistant messages so providers
            # that replay reasoning (OpenRouter, OpenAI, Nous) receive
            # coherent multi-turn reasoning context.
            if row["role"] == "assistant":
                if row["finish_reason"]:
                    msg["finish_reason"] = row["finish_reason"]
                if row["reasoning"]:
                    msg["reasoning"] = row["reasoning"]
                if row["reasoning_content"] is not None:
                    msg["reasoning_content"] = row["reasoning_content"]
                if row["reasoning_details"]:
                    try:
                        msg["reasoning_details"] = json.loads(row["reasoning_details"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize reasoning_details, falling back to None")
                        msg["reasoning_details"] = None
                if row["codex_reasoning_items"]:
                    try:
                        msg["codex_reasoning_items"] = json.loads(row["codex_reasoning_items"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize codex_reasoning_items, falling back to None")
                        msg["codex_reasoning_items"] = None
                if row["codex_message_items"]:
                    try:
                        msg["codex_message_items"] = json.loads(row["codex_message_items"])
                    except (json.JSONDecodeError, TypeError):
                        logger.warning("Failed to deserialize codex_message_items, falling back to None")
                        msg["codex_message_items"] = None
            if include_ancestors and self._is_duplicate_replayed_user_message(messages, msg):
                continue
            messages.append(msg)
        # DEFENSE-IN-DEPTH against background-review session pollution: a forked
        # skill/memory review that (in older builds, before the _persist_disabled
        # fix) shared the parent's session_id wrote its harness turn into this
        # real session. The harness is a user/system message instructing the
        # agent to "Review the conversation above and update the skill library /
        # save to memory" under a hard tool restriction; re-loading it as live
        # history makes the agent adopt the curator role and refuse the user's
        # actual task. Strip any such harness message AND the curator-mode
        # assistant reply immediately following it, so a polluted session
        # resumes clean even if stray rows exist.
        messages = _strip_background_review_harness(messages)
        if repair_alternation and messages:
            # Lazy import: hermes_state already depends on agent.* (see
            # sanitize_context above), but keep this optional path from
            # widening the import surface at module load.
            from agent.agent_runtime_helpers import repair_message_sequence

            repaired = repair_message_sequence(None, messages)
            if repaired:
                logger.info(
                    "Repaired %d message-alternation violation(s) while "
                    "restoring session %s — durable transcript kept them, "
                    "see repair_message_sequence",
                    repaired,
                    session_id,
                )
        return messages

    def get_resume_conversations(
        self, session_id: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return ``(model_history, display_history)`` for a session resume in ONE SELECT.

        ``session.resume`` needs two projections of the same lineage:

        - ``model_history`` — the tip session's active rows, alternation-repaired
          (the live-replay working conversation). Equivalent to
          ``get_messages_as_conversation(session_id, repair_alternation=True)``.
        - ``display_history`` — the full lineage (ancestors → tip), verbatim, with
          replayed-user dedup. Equivalent to
          ``get_messages_as_conversation(session_id, include_ancestors=True)``.

        The display fetch already reads a superset of the model fetch (the tip
        rows are part of the lineage), so serving both from one lineage SELECT
        halves the resume's DB work versus two separate calls, with byte-identical
        output (see test_get_resume_conversations_matches_separate_reads).
        """
        session_ids = self._session_lineage_root_to_tip(session_id)
        with self._lock:
            placeholders = ",".join("?" for _ in session_ids)
            rows = self._conn.execute(
                f"SELECT session_id, {self._CONVERSATION_ROW_COLUMNS} "
                f"FROM messages WHERE session_id IN ({placeholders}) AND active = 1 "
                # ORDER BY id (insertion order) — see get_messages_as_conversation
                # for why timestamp ordering is unsafe.
                "ORDER BY id",
                tuple(session_ids),
            ).fetchall()

        # Tip rows are exactly the model-fed set (get_messages_as_conversation
        # with session_ids=[session_id]); filtering the lineage fetch preserves
        # their relative id order.
        tip_rows = [r for r in rows if r["session_id"] == session_id]
        model_history = self._rows_to_conversation(
            tip_rows,
            session_id=session_id,
            include_ancestors=False,
            repair_alternation=True,
        )
        display_history = self._rows_to_conversation(
            rows,
            session_id=session_id,
            include_ancestors=True,
            repair_alternation=False,
        )
        return model_history, display_history

    def get_ancestor_display_prefix(self, session_id: str) -> List[Dict[str, Any]]:
        """Return the ancestor-only display messages for a session lineage.

        These are messages from parent/grandparent sessions (compression
        ancestors) that appear in the display transcript but NOT in the
        tip session's model-fed history. Used by ``session.resume`` to
        build the ``display_history_prefix`` that ``_live_session_payload``
        prepends to the live model history.

        Previously the prefix was calculated as
        ``display_history[:len(display) - len(raw)]``, but that overcounts
        when ``repair_message_sequence`` removes messages from the MIDDLE
        of the tip history (e.g. verification candidates collapsed by the
        consecutive-assistant merge) — the length difference includes both
        ancestor messages AND repair-removed tip messages, but the slice
        only captures the first N display messages (which are tip messages
        when there are no ancestors), causing duplication. This method
        returns ONLY the genuine ancestor messages, identified by
        ``session_id != tip_session_id``. (#65919)
        """
        session_ids = self._session_lineage_root_to_tip(session_id)
        if len(session_ids) <= 1:
            return []
        with self._lock:
            placeholders = ",".join("?" for _ in session_ids)
            rows = self._conn.execute(
                f"SELECT session_id, {self._CONVERSATION_ROW_COLUMNS} "
                f"FROM messages WHERE session_id IN ({placeholders}) AND active = 1 "
                "ORDER BY id",
                tuple(session_ids),
            ).fetchall()
        ancestor_rows = [r for r in rows if r["session_id"] != session_id]
        if not ancestor_rows:
            return []
        return self._rows_to_conversation(
            ancestor_rows,
            session_id=session_id,
            include_ancestors=True,
            repair_alternation=False,
        )

    def get_conversation_root(self, session_id: str) -> str:
        """Return the ROOT id of *session_id*'s lineage chain.

        The root is the stable "conversation id": context compression
        rotates ``session_id`` to a new segment linked via
        ``parent_session_id``, and delegate subagents hang off their
        parent the same way. Walking to the root gives every segment of
        one user-facing conversation (and its delegation tree) a single
        identifier — used for Nous Portal ``conversation=`` usage tagging.
        Returns *session_id* unchanged when it has no recorded parent.
        """
        chain = self._session_lineage_root_to_tip(session_id)
        return (chain[0] if chain and chain[0] else session_id)

    def _session_lineage_root_to_tip(self, session_id: str) -> List[str]:
        if not session_id:
            return [session_id]

        chain = []
        current = session_id
        seen = set()
        with self._lock:
            for _ in range(100):
                if not current or current in seen:
                    break
                seen.add(current)
                chain.append(current)
                row = self._conn.execute(
                    "SELECT parent_session_id FROM sessions WHERE id = ?",
                    (current,),
                ).fetchone()
                if row is None:
                    break
                current = row["parent_session_id"] if hasattr(row, "keys") else row[0]
        return list(reversed(chain)) or [session_id]

    @staticmethod
    def _is_duplicate_replayed_user_message(messages: List[Dict[str, Any]], msg: Dict[str, Any]) -> bool:
        if msg.get("role") != "user":
            return False
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            return False
        for prev in reversed(messages):
            if prev.get("role") == "user" and prev.get("content") == content:
                return True
            if prev.get("role") == "assistant" and (prev.get("content") or prev.get("tool_calls")):
                return False
        return False

    # =========================================================================
    # Rewind (soft-delete) — see /rewind slash command + issue #21910
    # =========================================================================

    def rewind_to_message(
        self, session_id: str, target_message_id: int
    ) -> Dict[str, Any]:
        """Soft-delete all messages with id >= ``target_message_id`` in *session_id*.

        The target message itself becomes inactive as well so the caller
        can pre-fill it as the next user prompt without it appearing
        twice in the replayed transcript.  Rewound rows are kept on
        disk with ``active=0`` for audit / forensic inspection — use
        :meth:`get_messages` with ``include_inactive=True`` to see them.

        Returns a dict::

            {
                "rewound_count": int,    # number of rows newly flipped to active=0
                "target_message": dict,  # full row dict of the target
                "new_head_id":   int|None  # id of the last still-active row, or None
            }

        Raises ``ValueError`` if the target message does not exist in
        *session_id* or if its role is not ``"user"``.

        Always increments ``sessions.rewind_count`` — even when the
        target is already inactive — so the counter accurately reflects
        the number of rewind operations performed against the session.
        Idempotent on the ``active`` flag: re-rewinding past the same
        target is a no-op on row state but still bumps the counter.
        """

        # 1) Validate target up-front (read-only, outside the write txn).
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM messages WHERE id = ? AND session_id = ?",
                (target_message_id, session_id),
            ).fetchone()
        if row is None:
            raise ValueError(
                f"message {target_message_id} not found in session {session_id}"
            )
        target_row = dict(row)
        if target_row.get("role") != "user":
            raise ValueError(
                f"rewind target must be a 'user' message (got role="
                f"{target_row.get('role')!r}, id={target_message_id})"
            )

        # Decode content for callers (prefill the prompt buffer).
        target_row["content"] = self._decode_content(target_row.get("content"))

        rewound: List[int] = []

        def _do(conn):
            cursor = conn.execute(
                "SELECT id FROM messages "
                "WHERE session_id = ? AND id >= ? AND active = 1",
                (session_id, target_message_id),
            )
            ids = [r[0] for r in cursor.fetchall()]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"UPDATE messages SET active = 0 WHERE id IN ({placeholders})",
                    ids,
                )
            conn.execute(
                "UPDATE sessions SET rewind_count = COALESCE(rewind_count, 0) + 1 "
                "WHERE id = ?",
                (session_id,),
            )
            return ids

        rewound = self._execute_write(_do)

        # 2) Compute new head id (largest still-active row id in session).
        with self._lock:
            head_row = self._conn.execute(
                "SELECT MAX(id) FROM messages WHERE session_id = ? AND active = 1",
                (session_id,),
            ).fetchone()
        new_head_id = head_row[0] if head_row and head_row[0] is not None else None

        return {
            "rewound_count": len(rewound),
            "target_message": target_row,
            "new_head_id": new_head_id,
        }

    def restore_rewound(self, session_id: str, since_message_id: int) -> int:
        """Mark inactive messages with id >= *since_message_id* active again.

        Returns the number of rows flipped back to ``active=1``.
        Intended for undo-of-rewind and test cleanup; not wired to a
        slash command in v1.
        """
        def _do(conn):
            cursor = conn.execute(
                "SELECT id FROM messages "
                "WHERE session_id = ? AND id >= ? AND active = 0",
                (session_id, since_message_id),
            )
            ids = [r[0] for r in cursor.fetchall()]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"UPDATE messages SET active = 1 WHERE id IN ({placeholders})",
                    ids,
                )
            return len(ids)

        return self._execute_write(_do)

    def list_recent_user_messages(
        self,
        session_id: str,
        limit: int = 20,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return the *limit* most-recent user messages, newest first.

        Each entry is a dict with keys ``id``, ``timestamp``, ``preview``.
        ``preview`` is the first 80 characters of the message content
        (with line breaks collapsed to spaces). Used by the /rewind
        slash command picker.

        By default only active messages are returned.
        """
        active_clause = "" if include_inactive else " AND active = 1"
        with self._lock:
            cursor = self._conn.execute(
                "SELECT id, timestamp, content FROM messages "
                "WHERE session_id = ? AND role = 'user'"
                f"{active_clause} "
                "ORDER BY id DESC LIMIT ?",
                (session_id, int(limit)),
            )
            rows = cursor.fetchall()

        result: List[Dict[str, Any]] = []
        for row in rows:
            decoded = self._decode_content(row["content"])
            if isinstance(decoded, list):
                # Multimodal — flatten text parts.
                text_parts = [
                    p.get("text", "") for p in decoded
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                preview = " ".join(t for t in text_parts if t).strip()
                if not preview:
                    preview = "[multimodal content]"
            elif isinstance(decoded, str):
                preview = decoded
            else:
                preview = ""
            preview = " ".join(preview.split())  # collapse whitespace
            if len(preview) > 80:
                preview = preview[:77] + "..."
            result.append(
                {
                    "id": row["id"],
                    "timestamp": row["timestamp"],
                    "preview": preview,
                }
            )
        return result

    # =========================================================================
    # Search
    # =========================================================================

    @staticmethod
    def _sanitize_fts5_query(query: str) -> str:
        """Sanitize user input for safe use in FTS5 MATCH queries.

        FTS5 has its own query syntax where characters like ``"``, ``(``, ``)``,
        ``+``, ``*``, ``{``, ``}``, the column-filter operator ``:`` and bare
        boolean operators (``AND``, ``OR``, ``NOT``) have special meaning.
        Passing raw user input directly to MATCH can cause
        ``sqlite3.OperationalError``.

        Strategy:
        - Preserve properly paired quoted phrases (``"exact phrase"``)
        - Strip unmatched FTS5-special characters that would cause errors
        - Wrap unquoted hyphenated and dotted terms in quotes so FTS5
          matches them as exact phrases instead of splitting on the
          hyphen/dot (e.g. ``chat-send``, ``P2.2``, ``my-app.config.ts``)
        """
        # Cap user-controlled FTS input before any regex processing. Search
        # queries do not need to be arbitrarily large, and bounding them keeps
        # sanitizer/runtime behavior predictable under adversarial input.
        query = query[:MAX_FTS5_QUERY_CHARS]

        # Step 1: Extract balanced double-quoted phrases and protect them
        # from further processing via numbered placeholders. Do this with a
        # single linear scan rather than a regex so pathological quote runs
        # cannot induce backtracking.
        _quoted_parts: list = []
        pieces: list[str] = []
        i = 0
        while i < len(query):
            ch = query[i]
            if ch != '"':
                pieces.append(ch)
                i += 1
                continue
            end = query.find('"', i + 1)
            if end == -1:
                # Unmatched quote: replace with whitespace like the old
                # sanitizer's special-char stripping step.
                pieces.append(" ")
                i += 1
                continue
            _quoted_parts.append(query[i:end + 1])
            pieces.append(f"\x00Q{len(_quoted_parts) - 1}\x00")
            i = end + 1

        sanitized = "".join(pieces)

        # Step 2: Strip remaining (unmatched) FTS5-special characters.  ``:`` is
        # FTS5's column-filter operator (``col:term``); since the FTS table has a
        # single ``content`` column, an unquoted colon query like ``TODO: fix``
        # parses as ``column:term`` and raises "no such column" — swallowed at
        # the execute site into zero results.  Strip it like the others.
        sanitized = re.sub(r'[+{}():\"^]', " ", sanitized)

        # Step 3: Collapse repeated * (e.g. "***") into a single one,
        # and remove leading * (prefix-only needs at least one char before *)
        sanitized = re.sub(r"\*+", "*", sanitized)
        sanitized = re.sub(r"(^|\s)\*", r"\1", sanitized)

        # Step 4: Remove dangling boolean operators at start/end that would
        # cause syntax errors (e.g. "hello AND" or "OR world")
        sanitized = re.sub(r"(?i)^(AND|OR|NOT)\b\s*", "", sanitized.strip())
        sanitized = re.sub(r"(?i)\s+(AND|OR|NOT)\s*$", "", sanitized.strip())

        # Step 5: Wrap unquoted dotted and/or hyphenated terms in double
        # quotes.  FTS5's tokenizer splits on dots and hyphens, turning
        # ``chat-send`` into ``chat AND send`` and ``P2.2`` into ``p2 AND 2``.
        # Quoting preserves phrase semantics.  A single pass avoids the
        # double-quoting bug that would occur if dotted, hyphenated and underscored
        # patterns were applied sequentially (e.g. ``my-app.config``).
        sanitized = re.sub(r"\b(\w+(?:[._-]\w+)+)\b", r'"\1"', sanitized)

        # Step 6: Restore preserved quoted phrases
        for i, quoted in enumerate(_quoted_parts):
            sanitized = sanitized.replace(f"\x00Q{i}\x00", quoted)

        return sanitized.strip()


    @staticmethod
    def _is_cjk_codepoint(cp: int) -> bool:
        return (0x4E00 <= cp <= 0x9FFF or    # CJK Unified Ideographs
                0x3400 <= cp <= 0x4DBF or    # CJK Extension A
                0x20000 <= cp <= 0x2A6DF or  # CJK Extension B
                0x3000 <= cp <= 0x303F or    # CJK Symbols
                0x3040 <= cp <= 0x309F or    # Hiragana
                0x30A0 <= cp <= 0x30FF or    # Katakana
                0xAC00 <= cp <= 0xD7AF)      # Hangul Syllables

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        """Check if text contains CJK (Chinese, Japanese, Korean) characters."""
        for ch in text:
            cp = ord(ch)
            if (0x4E00 <= cp <= 0x9FFF or    # CJK Unified Ideographs
                0x3400 <= cp <= 0x4DBF or    # CJK Extension A
                0x20000 <= cp <= 0x2A6DF or  # CJK Extension B
                0x3000 <= cp <= 0x303F or    # CJK Symbols
                0x3040 <= cp <= 0x309F or    # Hiragana
                0x30A0 <= cp <= 0x30FF or    # Katakana
                0xAC00 <= cp <= 0xD7AF):     # Hangul Syllables
                return True
        return False

    @classmethod
    def _count_cjk(cls, text: str) -> int:
        """Count CJK characters in text."""
        return sum(1 for ch in text if cls._is_cjk_codepoint(ord(ch)))

    @classmethod
    def _has_lone_cjk_run(cls, query: str) -> bool:
        """True when any maximal CJK run in the query is a single char.

        The cjk-bigram index stores bigrams for runs >=2 chars and unigrams
        only for isolated chars, so a 1-char CJK term can't match inside
        longer runs there — those queries keep the LIKE substring route.
        """
        run = 0
        for ch in query:
            if cls._is_cjk_codepoint(ord(ch)):
                run += 1
            else:
                if run == 1:
                    return True
                run = 0
        return run == 1

    @staticmethod
    def _trigram_eligible_tokens(query: str) -> bool:
        """True when every non-operator token is long enough for the trigram
        tokenizer to match (>=3 chars).

        The trigram tokenizer indexes overlapping 3-character sequences, so a
        token shorter than 3 chars produces no trigrams and can never match.
        With FTS5's implicit-AND between tokens, a single short token makes the
        whole MATCH return nothing, so the trigram path is only worth taking
        when every searchable token qualifies.
        """
        tokens = [
            t for t in query.strip('"').strip().split()
            if t.upper() not in {"AND", "OR", "NOT"}
        ]
        return bool(tokens) and all(len(t) >= 3 for t in tokens)

    def _run_trigram_search(
        self,
        raw_query: str,
        *,
        table: str = "messages_fts_trigram",
        order_by_sql: str,
        include_inactive: bool,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        """Run a search against a substring-capable FTS index.

        ``table`` is ``messages_fts_trigram`` (default) or
        ``messages_fts_cjk``. The trigram tokenizer indexes overlapping
        3-byte sequences, so it matches substrings regardless of word
        boundaries — both CJK phrases the unicode61 tokenizer splits into
        single characters and Latin runs the unicode61 tokenizer fuses onto
        adjacent CJK (e.g. ``修改youer服务端``). The cjk-bigram tokenizer
        splits Latin runs off adjacent CJK, giving the same recovery as an
        exact ranked token match. Each non-operator token is quoted to
        neutralise FTS5 special characters while boolean operators
        (AND/OR/NOT) are preserved.

        Returns the matching rows, or ``None`` when the query cannot be
        executed (e.g. the tokenizer is unavailable at runtime) so the
        caller can fall back to another strategy.
        """
        tokens = raw_query.split()
        parts = []
        for tok in tokens:
            if tok.upper() in {"AND", "OR", "NOT"}:
                parts.append(tok)
            else:
                parts.append('"' + tok.replace('"', '""') + '"')
        trigram_query = " ".join(parts)
        tri_where = [f"{table} MATCH ?"]
        tri_params: list = [trigram_query]
        if not include_inactive:
            tri_where.append("(m.active = 1 OR m.compacted = 1)")
        if source_filter is not None:
            tri_where.append(f"s.source IN ({','.join('?' for _ in source_filter)})")
            tri_params.extend(source_filter)
        if exclude_sources is not None:
            tri_where.append(f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})")
            tri_params.extend(exclude_sources)
        if role_filter:
            tri_where.append(f"m.role IN ({','.join('?' for _ in role_filter)})")
            tri_params.extend(role_filter)
        tri_sql = f"""
            SELECT
                m.id,
                m.session_id,
                m.role,
                snippet({table}, -1, '>>>', '<<<', '...', 40) AS snippet,
                m.content,
                m.timestamp,
                m.tool_name,
                s.source,
                s.model,
                s.started_at AS session_started
            FROM {table}
            JOIN messages m ON m.id = {table}.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(tri_where)}
            {order_by_sql}
            LIMIT ? OFFSET ?
        """
        tri_params.extend([limit, offset])
        with self._lock:
            try:
                tri_cursor = self._conn.execute(tri_sql, tri_params)
            except sqlite3.OperationalError:
                # Query failed at runtime — let the caller fall back.
                return None
            return [dict(row) for row in tri_cursor.fetchall()]

    def search_messages(
        self,
        query: str,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort: str = None,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        """Instrumented wrapper around :meth:`_search_messages_impl`.

        Logs one line per slow search with the routing path taken, so
        production latency stays attributable per query shape (the 2026-07
        session_search investigation needed trace archaeology to discover
        the LIKE full scans; this makes the next regression a grep).
        Threshold: HERMES_SEARCH_SLOW_MS (default 1000; 0 logs every call).
        """
        started = time.time()
        rows = None
        try:
            rows = self._search_messages_impl(
                query,
                source_filter=source_filter,
                exclude_sources=exclude_sources,
                role_filter=role_filter,
                limit=limit,
                offset=offset,
                sort=sort,
                include_inactive=include_inactive,
            )
            return rows
        finally:
            try:
                threshold = float(os.getenv("HERMES_SEARCH_SLOW_MS", "1000"))
            except (TypeError, ValueError):
                threshold = 1000.0
            elapsed_ms = (time.time() - started) * 1000.0
            if elapsed_ms >= threshold:
                logger.info(
                    "slow session search: path=%s elapsed=%.0fms rows=%s query=%r",
                    self._describe_search_path(query),
                    elapsed_ms,
                    len(rows) if rows is not None else "err",
                    query[:200],
                )

    def _describe_search_path(self, query: str) -> str:
        """Best-effort name of the routing path a query takes (log-only)."""
        try:
            sanitized = self._sanitize_fts5_query(query or "")
            if not sanitized:
                return "empty"
            if not self._contains_cjk(sanitized):
                return "fts5"
            raw = sanitized.strip('"').strip()
            if self._fts_cjk_available and not self._has_lone_cjk_run(raw):
                return "fts_cjk"
            tokens = [
                t for t in raw.split()
                if t.upper() not in {"AND", "OR", "NOT"} and self._contains_cjk(t)
            ]
            short = any(self._count_cjk(t) < 3 for t in tokens)
            if self._count_cjk(raw) >= 3 and not short and self._trigram_available:
                return "trigram"
            return "like_scan"
        except Exception:
            return "unknown"

    def _search_messages_impl(
        self,
        query: str,
        source_filter: List[str] = None,
        exclude_sources: List[str] = None,
        role_filter: List[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort: str = None,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Full-text search across session messages using FTS5.

        Supports FTS5 query syntax:
          - Simple keywords: "docker deployment"
          - Phrases: '"exact phrase"'
          - Boolean: "docker OR kubernetes", "python NOT java"
          - Prefix: "deploy*"

        Returns matching messages with session metadata, content snippet,
        and surrounding context (1 message before and after the match).

        ``sort`` controls temporal ordering:
          - ``None`` (default): FTS5 BM25 relevance only. Time-neutral.
          - ``"newest"``: order by message timestamp DESC, then by rank.
          - ``"oldest"``: order by message timestamp ASC, then by rank.

        The short-CJK LIKE fallback already orders by timestamp DESC and
        ignores ``sort``. The trigram CJK path honours ``sort`` like the main
        FTS5 path.

        Rewound (``active=0``, ``compacted=0``) rows are excluded by default —
        the user took those back. Compaction-archived rows (``active=0``,
        ``compacted=1``) ARE included by default: they were summarized away from
        the live context but remain part of the conversation's record, so the
        pre-compaction transcript stays discoverable after in-place compaction
        (#38763). Pass ``include_inactive=True`` to search every row regardless.
        """
        if not self._fts_enabled:
            return []

        if not query or not query.strip():
            return []

        query = self._sanitize_fts5_query(query)
        if not query:
            return []

        # Normalise sort. Anything not in the allowed set falls back to None
        # (FTS5 rank-only) so callers can pass through user input without
        # validation.
        if isinstance(sort, str):
            sort_norm = sort.strip().lower()
            if sort_norm not in ("newest", "oldest"):
                sort_norm = None
        else:
            sort_norm = None

        # ORDER BY shared across the main FTS5 path and trigram CJK path.
        # With sort set, timestamp is primary and rank is the tiebreaker.
        if sort_norm == "newest":
            order_by_sql = "ORDER BY m.timestamp DESC, rank"
        elif sort_norm == "oldest":
            order_by_sql = "ORDER BY m.timestamp ASC, rank"
        else:
            order_by_sql = "ORDER BY rank"

        # Build WHERE clauses dynamically
        where_clauses = ["messages_fts MATCH ?"]
        params: list = [query]
        if not include_inactive:
            # Live rows (active=1) AND compaction-archived rows (compacted=1)
            # are discoverable; only rewind/undo rows (active=0, compacted=0)
            # are hidden. See archive_and_compact() / #38763.
            where_clauses.append("(m.active = 1 OR m.compacted = 1)")

        if source_filter is not None:
            source_placeholders = ",".join("?" for _ in source_filter)
            where_clauses.append(f"s.source IN ({source_placeholders})")
            params.extend(source_filter)

        if exclude_sources is not None:
            exclude_placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({exclude_placeholders})")
            params.extend(exclude_sources)

        if role_filter:
            role_placeholders = ",".join("?" for _ in role_filter)
            where_clauses.append(f"m.role IN ({role_placeholders})")
            params.extend(role_filter)

        where_sql = " AND ".join(where_clauses)
        params.extend([limit, offset])

        sql = f"""
            SELECT
                m.id,
                m.session_id,
                m.role,
                snippet(messages_fts, -1, '>>>', '<<<', '...', 40) AS snippet,
                m.content,
                m.timestamp,
                m.tool_name,
                s.source,
                s.model,
                s.started_at AS session_started
            FROM messages_fts
            JOIN messages m ON m.id = messages_fts.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE {where_sql}
            {order_by_sql}
            LIMIT ? OFFSET ?
        """

        # CJK queries bypass the unicode61 FTS5 table.  The default tokenizer
        # splits CJK characters into individual tokens, so "大别山项目" becomes
        # "大 AND 别 AND 山 AND 项 AND 目" — producing false positives and
        # missing exact phrase matches.
        #
        # For queries with 3+ CJK characters, we use the trigram FTS5 table
        # (indexed substring matching with ranking and snippets).  For shorter
        # CJK queries (1-2 chars), trigram can't match (it needs ≥9 UTF-8
        # bytes = 3 CJK chars), so we fall back to LIKE.
        is_cjk = self._contains_cjk(query)
        if is_cjk:
            raw_query = query.strip('"').strip()
            cjk_count = self._count_cjk(raw_query)

            # Per-token CJK length check (#20494): trigram needs >=3 CJK chars
            # per token. A query like "广西 OR 桂林 OR 漓江" has cjk_count=6
            # (>=3) but each individual token is only 2 chars — trigram returns 0.
            # Route to LIKE when any non-operator CJK token is <3 CJK chars.
            _tokens_for_check = [
                t for t in raw_query.split()
                if t.upper() not in {"AND", "OR", "NOT"} and self._contains_cjk(t)
            ]
            _any_short_cjk = any(
                self._count_cjk(t) < 3 for t in _tokens_for_check
            )

            _trigram_succeeded = False
            # Tool rows are excluded from the trigram index (they're ~90% of
            # message bytes and machine noise — see FTS_TRIGRAM_SQL). A CJK
            # query explicitly filtering on role='tool' must therefore use
            # the LIKE fallback, which scans the base table directly.
            _wants_tool_rows = bool(role_filter) and "tool" in role_filter

            # ── CJK-bigram route (messages_fts_cjk, cjk_unicode61) ──────
            # When the bigram index is available it serves EVERY CJK query
            # shape the legacy code split between trigram (>=3 chars/token)
            # and LIKE full scans (1-2 char tokens) — the whole point of the
            # index (PR #65544). Exceptions stay on the legacy routes:
            #   - role_filter=['tool'] queries (tool rows aren't in the cjk
            #     index, same exclusion as trigram),
            #   - queries containing a LONE 1-char CJK run: the index stores
            #     bigrams for runs >=2, so a single-char term can only match
            #     isolated chars — LIKE substring semantics are broader.
            if (
                self._fts_cjk_available
                and not _wants_tool_rows
                and not self._has_lone_cjk_run(raw_query)
            ):
                tokens = raw_query.split()
                parts = []
                for tok in tokens:
                    if tok.upper() in {"AND", "OR", "NOT"}:
                        parts.append(tok)
                    else:
                        parts.append('"' + tok.replace('"', '""') + '"')
                cjk_query = " ".join(parts)
                cjk_where = ["messages_fts_cjk MATCH ?"]
                cjk_params: list = [cjk_query]
                if not include_inactive:
                    cjk_where.append("(m.active = 1 OR m.compacted = 1)")
                if source_filter is not None:
                    cjk_where.append(f"s.source IN ({','.join('?' for _ in source_filter)})")
                    cjk_params.extend(source_filter)
                if exclude_sources is not None:
                    cjk_where.append(f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})")
                    cjk_params.extend(exclude_sources)
                if role_filter:
                    cjk_where.append(f"m.role IN ({','.join('?' for _ in role_filter)})")
                    cjk_params.extend(role_filter)
                cjk_sql = f"""
                    SELECT
                        m.id,
                        m.session_id,
                        m.role,
                        snippet(messages_fts_cjk, -1, '>>>', '<<<', '...', 40) AS snippet,
                        m.content,
                        m.timestamp,
                        m.tool_name,
                        s.source,
                        s.model,
                        s.started_at AS session_started
                    FROM messages_fts_cjk
                    JOIN messages m ON m.id = messages_fts_cjk.rowid
                    JOIN sessions s ON s.id = m.session_id
                    WHERE {' AND '.join(cjk_where)}
                    {order_by_sql}
                    LIMIT ? OFFSET ?
                """
                cjk_params.extend([limit, offset])
                try:
                    with self._lock:
                        cjk_cursor = self._conn.execute(cjk_sql, cjk_params)
                        matches = [dict(row) for row in cjk_cursor.fetchall()]
                        _trigram_succeeded = True
                except sqlite3.OperationalError:
                    # Tokenizer missing on this connection / query syntax —
                    # the trigram + LIKE routes below still answer.
                    logger.debug(
                        "messages_fts_cjk query failed; falling back to "
                        "trigram/LIKE", exc_info=True,
                    )
                except sqlite3.DatabaseError as exc:
                    # Same corruption class as the other FTS reads: rebuild
                    # in place once and retry; on refusal/failure fall back.
                    if self._try_runtime_fts_rebuild(exc):
                        try:
                            with self._lock:
                                cjk_cursor = self._conn.execute(
                                    cjk_sql, cjk_params
                                )
                                matches = [
                                    dict(row) for row in cjk_cursor.fetchall()
                                ]
                                _trigram_succeeded = True
                        except sqlite3.DatabaseError:
                            logger.warning(
                                "CJK-bigram FTS search still failing after "
                                "in-place rebuild; falling back to "
                                "trigram/LIKE."
                            )
                    else:
                        logger.warning(
                            "CJK-bigram FTS search hit a corruption error "
                            "(%s) and no in-place rebuild was possible; "
                            "falling back to trigram/LIKE.", exc,
                        )

            if (
                not _trigram_succeeded
                and cjk_count >= 3
                and not _any_short_cjk
                and self._trigram_available
                and not _wants_tool_rows
            ):
                # Trigram FTS5 path — quote each non-operator token to handle
                # FTS5 special chars (%, *, etc.) while preserving boolean
                # operators (AND, OR, NOT) for multi-term queries.
                tokens = raw_query.split()
                parts = []
                for tok in tokens:
                    if tok.upper() in {"AND", "OR", "NOT"}:
                        parts.append(tok)
                    else:
                        parts.append('"' + tok.replace('"', '""') + '"')
                trigram_query = " ".join(parts)
                tri_where = ["messages_fts_trigram MATCH ?"]
                tri_params: list = [trigram_query]
                if not include_inactive:
                    tri_where.append("(m.active = 1 OR m.compacted = 1)")
                if source_filter is not None:
                    tri_where.append(f"s.source IN ({','.join('?' for _ in source_filter)})")
                    tri_params.extend(source_filter)
                if exclude_sources is not None:
                    tri_where.append(f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})")
                    tri_params.extend(exclude_sources)
                if role_filter:
                    tri_where.append(f"m.role IN ({','.join('?' for _ in role_filter)})")
                    tri_params.extend(role_filter)
                tri_sql = f"""
                    SELECT
                        m.id,
                        m.session_id,
                        m.role,
                        snippet(messages_fts_trigram, -1, '>>>', '<<<', '...', 40) AS snippet,
                        m.content,
                        m.timestamp,
                        m.tool_name,
                        s.source,
                        s.model,
                        s.started_at AS session_started
                    FROM messages_fts_trigram
                    JOIN messages m ON m.id = messages_fts_trigram.rowid
                    JOIN sessions s ON s.id = m.session_id
                    WHERE {' AND '.join(tri_where)}
                    {order_by_sql}
                    LIMIT ? OFFSET ?
                """
                tri_params.extend([limit, offset])
                try:
                    with self._lock:
                        tri_cursor = self._conn.execute(tri_sql, tri_params)
                        matches = [dict(row) for row in tri_cursor.fetchall()]
                        _trigram_succeeded = True
                except sqlite3.OperationalError:
                    # Trigram query failed at runtime — fall through to LIKE.
                    pass
                except sqlite3.DatabaseError as exc:
                    # Same corruption class the main FTS5 MATCH branch
                    # self-heals above: a corrupt trigram shadow table raises
                    # malformed / "fts5: corrupt structure record", which is a
                    # DatabaseError (parent of the OperationalError syntax arm
                    # caught first). Rebuild once outside the lock — the lock
                    # is released here so rebuild_fts() can re-acquire it —
                    # and retry the trigram query. If the rebuild is refused
                    # (already attempted / FTS disabled / different error
                    # class) or the retry fails again, fall through to the
                    # LIKE substring path, which reads only the canonical
                    # messages table, so CJK search stays available.
                    if self._try_runtime_fts_rebuild(exc):
                        try:
                            with self._lock:
                                tri_cursor = self._conn.execute(
                                    tri_sql, tri_params
                                )
                                matches = [
                                    dict(row) for row in tri_cursor.fetchall()
                                ]
                                _trigram_succeeded = True
                        except sqlite3.DatabaseError:
                            logger.warning(
                                "Trigram FTS search still failing after "
                                "in-place rebuild; falling back to LIKE."
                            )
                    else:
                        logger.warning(
                            "Trigram FTS search hit a corruption error (%s) "
                            "and no in-place rebuild was possible; falling "
                            "back to LIKE.", exc,
                        )
            if not _trigram_succeeded:
                # Short / mixed CJK query, trigram unavailable, or trigram
                # <3 CJK chars. Fall back to LIKE substring search.
                # For multi-token OR queries (e.g. "广西 OR 桂林 OR 漓江"),
                # build one LIKE condition per non-operator token so each term
                # is matched independently (#20494).
                non_op_tokens = [
                    t for t in raw_query.split()
                    if t.upper() not in {"AND", "OR", "NOT"}
                ] or [raw_query]
                token_clauses = []
                like_params: list = []
                for tok in non_op_tokens:
                    esc = tok.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                    token_clauses.append(
                        "(m.content LIKE ? ESCAPE '\\' OR m.tool_name LIKE ? ESCAPE '\\' OR m.tool_calls LIKE ? ESCAPE '\\')"
                    )
                    like_params += [f"%{esc}%", f"%{esc}%", f"%{esc}%"]
                like_where = [f"({' OR '.join(token_clauses)})"]
                if not include_inactive:
                    # Same visibility rule as the FTS5 paths: live rows and
                    # compaction-archived rows are discoverable; rewind/undo
                    # rows (active=0, compacted=0) are hidden (#38763).
                    like_where.append("(m.active = 1 OR m.compacted = 1)")
                if source_filter is not None:
                    like_where.append(f"s.source IN ({','.join('?' for _ in source_filter)})")
                    like_params.extend(source_filter)
                if exclude_sources is not None:
                    like_where.append(f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})")
                    like_params.extend(exclude_sources)
                if role_filter:
                    like_where.append(f"m.role IN ({','.join('?' for _ in role_filter)})")
                    like_params.extend(role_filter)
                like_sql = f"""
                    SELECT m.id, m.session_id, m.role,
                           substr(m.content,
                                  max(1, instr(m.content, ?) - 40),
                                  120) AS snippet,
                           m.content, m.timestamp, m.tool_name,
                           s.source, s.model, s.started_at AS session_started
                    FROM messages m
                    JOIN sessions s ON s.id = m.session_id
                    WHERE {' AND '.join(like_where)}
                    ORDER BY m.timestamp DESC
                    LIMIT ? OFFSET ?
                """
                like_params.extend([limit, offset])
                # instr() for snippet uses first search token
                like_params = [non_op_tokens[0]] + like_params
                with self._lock:
                    like_cursor = self._conn.execute(like_sql, like_params)
                    matches = [dict(row) for row in like_cursor.fetchall()]
        else:
            try:
                with self._lock:
                    cursor = self._conn.execute(sql, params)
                    matches = [dict(row) for row in cursor.fetchall()]
            except sqlite3.OperationalError:
                # FTS5 query syntax error despite sanitization — return empty
                return []
            except sqlite3.DatabaseError as exc:
                # A corrupt FTS index raises the malformed / "fts5: corrupt
                # structure record" class on the MATCH read, the same class the
                # write path self-heals (#66296). OperationalError (query
                # syntax) is a subclass caught above; this arm is the corruption
                # parent. Rebuild the index in place once — the lock is released
                # here, so rebuild_fts() can re-acquire it — and retry, so
                # search self-heals for read-only sessions (cron/CLI history
                # search) that never trigger a write to repair it first.
                if not self._try_runtime_fts_rebuild(exc):
                    raise
                with self._lock:
                    cursor = self._conn.execute(sql, params)
                    matches = [dict(row) for row in cursor.fetchall()]

        # Deferred-rebuild supplement (schema v23): while the background
        # backfill is pending, the FTS indexes only cover rows outside the
        # (progress, high_water] gap. Top the results up with a bounded LIKE
        # scan over just that id range so search never silently loses old
        # messages mid-rebuild. The range shrinks as the backfill advances,
        # so this cost decays to zero. The CJK LIKE-fallback path above
        # already scans the whole base table and needs no supplement.
        rebuild_status = self.fts_rebuild_status()
        if rebuild_status is not None and len(matches) < limit:
            try:
                gap_matches = self._search_unindexed_gap(
                    query,
                    limit - len(matches),
                    include_inactive=include_inactive,
                    source_filter=source_filter,
                    exclude_sources=exclude_sources,
                    role_filter=role_filter,
                )
                seen_ids = {m["id"] for m in matches}
                matches.extend(m for m in gap_matches if m["id"] not in seen_ids)
            except sqlite3.OperationalError as exc:
                logger.debug("Unindexed-gap supplement skipped: %s", exc)

        # Pure-Latin queries run against the unicode61 ``messages_fts`` table,
        # whose tokenizer does not insert a boundary between Latin letters and
        # adjacent CJK characters: "修改youer服务端" is indexed as one token,
        # so MATCH "youer" finds nothing even though the substring is present
        # (#54242). When the exact-token search returns nothing, retry on the
        # substring-capable indexes. Preference order:
        #   1. messages_fts_cjk (when built): its tokenizer splits Latin runs
        #      off adjacent CJK, so "youer" is an exact ranked token match.
        #   2. messages_fts_trigram: substring matching, needs >=3-char
        #      tokens (shorter tokens produce no trigrams).
        # Gated on a zero-result miss so successful Latin searches keep their
        # unicode61 ranking — strictly additive, never reorders existing
        # hits. Trade-off on the trigram leg: any zero-result Latin query
        # gains substring semantics (e.g. "cat" can then match
        # "concatenate"). Genuinely absent terms still return []. Skipped for
        # role_filter=['tool'] queries — both fallback indexes exclude tool
        # rows (v23), so a retry could never add hits.
        if (
            not matches
            and not is_cjk
            and not (bool(role_filter) and "tool" in role_filter)
        ):
            _fb_query = query.strip('"').strip()
            if self._fts_cjk_available:
                cjk_fb = self._run_trigram_search(
                    _fb_query,
                    table="messages_fts_cjk",
                    order_by_sql=order_by_sql,
                    include_inactive=include_inactive,
                    source_filter=source_filter,
                    exclude_sources=exclude_sources,
                    role_filter=role_filter,
                    limit=limit,
                    offset=offset,
                )
                if cjk_fb:
                    matches = cjk_fb
            if (
                not matches
                and self._trigram_available
                and self._trigram_eligible_tokens(query)
            ):
                tri_matches = self._run_trigram_search(
                    _fb_query,
                    order_by_sql=order_by_sql,
                    include_inactive=include_inactive,
                    source_filter=source_filter,
                    exclude_sources=exclude_sources,
                    role_filter=role_filter,
                    limit=limit,
                    offset=offset,
                )
                if tri_matches:
                    matches = tri_matches

        # Add surrounding context (1 message before + after each match).
        # Done outside the lock so we don't hold it across N sequential queries.
        for match in matches:
            try:
                with self._lock:
                    ctx_cursor = self._conn.execute(
                        """WITH target AS (
                               SELECT session_id, timestamp, id
                               FROM messages
                               WHERE id = ?
                           )
                           SELECT role, content
                           FROM (
                               SELECT m.id, m.timestamp, m.role, m.content
                               FROM messages m
                               JOIN target t ON t.session_id = m.session_id
                               WHERE (m.timestamp < t.timestamp)
                                  OR (m.timestamp = t.timestamp AND m.id < t.id)
                               ORDER BY m.timestamp DESC, m.id DESC
                               LIMIT 1
                           )
                           UNION ALL
                           SELECT role, content
                           FROM messages
                           WHERE id = ?
                           UNION ALL
                           SELECT role, content
                           FROM (
                               SELECT m.id, m.timestamp, m.role, m.content
                               FROM messages m
                               JOIN target t ON t.session_id = m.session_id
                               WHERE (m.timestamp > t.timestamp)
                                  OR (m.timestamp = t.timestamp AND m.id > t.id)
                               ORDER BY m.timestamp ASC, m.id ASC
                               LIMIT 1
                           )""",
                        (match["id"], match["id"]),
                    )
                    context_msgs = []
                    for r in ctx_cursor.fetchall():
                        raw = r["content"]
                        decoded = self._decode_content(raw)
                        # Multimodal context: render a compact text-only
                        # summary for search previews.
                        if isinstance(decoded, list):
                            text_parts = [
                                p.get("text", "") for p in decoded
                                if isinstance(p, dict) and p.get("type") == "text"
                            ]
                            text = " ".join(t for t in text_parts if t).strip()
                            preview = text or "[multimodal content]"
                        elif isinstance(decoded, str):
                            preview = decoded
                        else:
                            preview = ""
                        context_msgs.append(
                            {"role": r["role"], "content": preview[:200]}
                        )
                match["context"] = context_msgs
            except Exception:
                match["context"] = []

        # Remove full content from result (snippet is enough, saves tokens)
        for match in matches:
            match.pop("content", None)

        return matches

    def _search_unindexed_gap(
        self,
        fts_query: str,
        limit: int,
        *,
        include_inactive: bool = False,
        source_filter: Optional[List[str]] = None,
        exclude_sources: Optional[List[str]] = None,
        role_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """LIKE-scan the rows the deferred rebuild hasn't indexed yet.

        Only touches ids in (fts_rebuild_progress, fts_rebuild_high_water] —
        a range that shrinks to nothing as the backfill advances. The FTS
        query is degraded to per-token substring terms (AND-joined; quoted
        phrases kept whole), which is deliberately recall-over-precision:
        temporary results beat silently missing ones mid-rebuild.
        """
        status = self.fts_rebuild_status()
        if status is None or limit <= 0:
            return []
        progress, high_water = status["indexed"], status["total"]

        # Degrade the FTS query to LIKE terms: strip operators/wildcards,
        # keep quoted phrases intact, AND the rest.
        terms: List[str] = []
        for raw_tok in re.findall(r'"[^"]+"|\S+', fts_query):
            tok = raw_tok.strip('"').strip("*").strip()
            if not tok or tok.upper() in {"AND", "OR", "NOT", "NEAR"}:
                continue
            terms.append(tok)
        if not terms:
            return []

        where = ["m.id > ? AND m.id <= ?"]
        params: list = [progress, high_water]
        for term in terms:
            esc = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            where.append(
                "(m.content LIKE ? ESCAPE '\\' OR m.tool_name LIKE ? ESCAPE '\\' "
                "OR m.tool_calls LIKE ? ESCAPE '\\')"
            )
            params += [f"%{esc}%"] * 3
        if not include_inactive:
            where.append("(m.active = 1 OR m.compacted = 1)")
        if source_filter is not None:
            where.append(f"s.source IN ({','.join('?' for _ in source_filter)})")
            params.extend(source_filter)
        if exclude_sources is not None:
            where.append(f"s.source NOT IN ({','.join('?' for _ in exclude_sources)})")
            params.extend(exclude_sources)
        if role_filter:
            where.append(f"m.role IN ({','.join('?' for _ in role_filter)})")
            params.extend(role_filter)

        sql = f"""
            SELECT m.id, m.session_id, m.role,
                   substr(m.content,
                          max(1, instr(m.content, ?) - 40),
                          120) AS snippet,
                   m.content, m.timestamp, m.tool_name,
                   s.source, s.model, s.started_at AS session_started
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE {' AND '.join(where)}
            ORDER BY m.timestamp DESC
            LIMIT ?
        """
        params = [terms[0]] + params + [limit]
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def search_sessions_by_id(
        self,
        query: str,
        limit: int = 20,
        include_archived: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search surfaced sessions by exact/prefix/substring session id.

        Desktop search uses this alongside FTS message search so users can paste
        a session id from logs, CLI output, or another Hermes surface and jump
        straight to that conversation.  Matching also checks ``_lineage_root_id``
        for projected compression-chain tips, so an old root id still resolves to
        the live continuation row.
        """
        needle = (query or "").strip().lower()
        if not needle or limit <= 0:
            return []

        # SQL-bounded: list_sessions_rich pushes the id LIKE filter into the
        # query (matching the row's own id AND any id in its forward
        # compression chain), so we only materialize matching rows instead of
        # scanning every session. Fetch a small multiple of `limit` so the
        # in-Python exact/prefix/substring ranking below has enough candidates
        # to order, then truncate.
        candidates = self.list_sessions_rich(
            limit=max(limit * 4, limit),
            offset=0,
            include_archived=include_archived,
            order_by_last_active=True,
            id_query=needle,
        )

        def score(row: Dict[str, Any]) -> int:
            ids = [str(row.get("id") or ""), str(row.get("_lineage_root_id") or "")]
            normalized = [value.lower() for value in ids if value]
            if any(value == needle for value in normalized):
                return 0
            if any(value.startswith(needle) for value in normalized):
                return 1
            return 2

        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (score(item[1]), item[0]),
        )
        return [row for _, row in ranked[:limit]]

    def search_sessions(
        self,
        source: str = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by source.

        Returns rows enriched with a computed ``last_active`` column (latest
        message timestamp for the session, falling back to ``started_at``),
        ordered by most-recently-used first.
        """
        select_with_last_active = (
            "SELECT s.*, COALESCE(m.last_active, s.started_at) AS last_active "
            "FROM sessions s "
            "LEFT JOIN ("
            "SELECT session_id, MAX(timestamp) AS last_active "
            "FROM messages GROUP BY session_id"
            ") m ON m.session_id = s.id "
        )
        with self._lock:
            if source:
                cursor = self._conn.execute(
                    f"{select_with_last_active}"
                    "WHERE s.source = ? "
                    "ORDER BY last_active DESC, s.started_at DESC, s.id DESC LIMIT ? OFFSET ?",
                    (source, limit, offset),
                )
            else:
                cursor = self._conn.execute(
                    f"{select_with_last_active}"
                    "ORDER BY last_active DESC, s.started_at DESC, s.id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                )
            return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Utility
    # =========================================================================

    def session_count(
        self,
        source: str = None,
        cwd_prefix: str = None,
        min_message_count: int = 0,
        include_archived: bool = False,
        archived_only: bool = False,
        exclude_children: bool = False,
        exclude_sources: List[str] = None,
    ) -> int:
        """Count sessions, optionally filtered by source.

        Pass ``exclude_children=True`` to count only the conversations that
        ``list_sessions_rich`` surfaces (root + branch sessions), hiding
        sub-agent runs and compression continuations. Use it whenever the count
        is paired with a ``list_sessions_rich`` page (e.g. sidebar "load more"
        totals) so the total matches the number of listable rows — otherwise the
        raw row count is inflated by children and "load more" never settles.

        Pass ``exclude_sources`` to drop whole source classes from the count
        (e.g. ``["cron"]`` so the recents "load more" total matches a
        cron-excluded ``list_sessions_rich`` page and doesn't keep "load more"
        stuck on for buried scheduler sessions).
        """
        where_clauses = []
        params = []

        if exclude_children:
            # Mirror list_sessions_rich's child-exclusion clause exactly so the
            # count lines up with the rows: roots (no parent) plus branch
            # children (parent ended with end_reason='branched').
            where_clauses.append(_LISTABLE_CHILD_SQL)
            where_clauses.append(f"{_delegate_from_json('s.model_config')} IS NULL")
        if source:
            where_clauses.append("s.source = ?")
            params.append(source)
        if exclude_sources:
            placeholders = ",".join("?" for _ in exclude_sources)
            where_clauses.append(f"s.source NOT IN ({placeholders})")
            params.extend(exclude_sources)
        if cwd_prefix:
            clause, clause_params = _cwd_prefix_clause(cwd_prefix)
            where_clauses.append(clause)
            params.extend(clause_params)
        if min_message_count > 0:
            where_clauses.append("s.message_count >= ?")
            params.append(min_message_count)
        if archived_only:
            where_clauses.append("s.archived = 1")
        elif not include_archived:
            where_clauses.append("s.archived = 0")

        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with self._lock:
            cursor = self._conn.execute(f"SELECT COUNT(*) FROM sessions s{where_sql}", params)
            return cursor.fetchone()[0]

    def message_count(self, session_id: str = None) -> int:
        """Count messages, optionally for a specific session."""
        with self._lock:
            if session_id:
                cursor = self._conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
                )
            else:
                cursor = self._conn.execute("SELECT COUNT(*) FROM messages")
            return cursor.fetchone()[0]

    def has_platform_message_id(
        self, session_id: str, platform_message_id: str
    ) -> bool:
        """Check if a message with the given platform_message_id exists.

        Uses the idx_messages_platform_msg_id partial index for efficient
        lookup. Used by the gateway's transient-failure dedupe guard (#47237)
        to skip re-persisting a user message that was already saved on a
        prior retry of the same inbound platform message.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT 1 FROM messages "
                "WHERE session_id = ? AND platform_message_id = ? LIMIT 1",
                (session_id, platform_message_id),
            )
            return cursor.fetchone() is not None

    # =========================================================================
    # Export and cleanup
    # =========================================================================

    def _is_branch_child_row(self, session: Dict[str, Any]) -> bool:
        raw = session.get("model_config")
        if not raw:
            return False
        try:
            cfg = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            return False
        return isinstance(cfg, dict) and cfg.get("_branched_from") is not None

    def _is_compression_child_row(self, child: Dict[str, Any]) -> bool:
        parent_id = child.get("parent_session_id")
        if not parent_id or self._is_branch_child_row(child):
            return False
        parent = self.get_session(parent_id)
        return bool(parent and parent.get("end_reason") == "compression")

    def get_compression_lineage(self, session_id: str) -> List[str]:
        """Return compression ancestors through tip in chronological order."""
        session = self.get_session(session_id)
        if not session or self._is_branch_child_row(session):
            return [session_id] if session else []

        root = session
        ancestors = {root["id"]}
        while self._is_compression_child_row(root):
            parent = self.get_session(root["parent_session_id"])
            if not parent or parent["id"] in ancestors:
                break
            root = parent
            ancestors.add(root["id"])

        lineage = [root["id"]]
        seen = {root["id"]}
        current = root
        while current.get("end_reason") == "compression":
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT * FROM sessions
                    WHERE parent_session_id = ?
                    ORDER BY started_at ASC
                    """,
                    (current["id"],),
                ).fetchall()
            next_child = None
            for row in rows:
                candidate = dict(row)
                if not self._is_branch_child_row(candidate):
                    next_child = candidate
                    break
            if not next_child or next_child["id"] in seen:
                break
            lineage.append(next_child["id"])
            seen.add(next_child["id"])
            current = next_child
            if current["id"] == session_id:
                # Continue to include later compression tips only when the
                # requested session itself was compacted.
                continue
        return lineage if session_id in lineage else [session_id]

    def export_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a single session with all its messages as a dict."""
        session = self.get_session(session_id)
        if not session:
            return None
        messages = self.get_messages(session_id)
        return {**session, "messages": messages}

    def export_session_lineage(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Export a compression lineage as one logical session dict."""
        lineage_ids = self.get_compression_lineage(session_id)
        if not lineage_ids:
            return None
        segments = []
        for sid in lineage_ids:
            segment = self.export_session(sid)
            if segment:
                segments.append(segment)
        if not segments:
            return None
        base = dict(segments[-1])
        total_messages = sum(len(seg.get("messages") or []) for seg in segments)
        base["segments"] = segments
        base["lineage_session_ids"] = [seg["id"] for seg in segments]
        base["message_count"] = total_messages
        base["messages"] = [msg for seg in segments for msg in (seg.get("messages") or [])]
        return base

    def export_all(self, source: str = None) -> List[Dict[str, Any]]:
        """
        Export all sessions (with messages) as a list of dicts.
        Suitable for writing to a JSONL file for backup/analysis.
        """
        sessions = self.search_sessions(source=source, limit=100000)
        results = []
        for session in sessions:
            messages = self.get_messages(session["id"])
            results.append({**session, "messages": messages})
        return results

    @staticmethod
    def _import_text_or_none(value: Any, field: str) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        raise ValueError(f"{field} must be a string")

    @staticmethod
    def _import_json_object_or_none(value: Any, field: str) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{field} must be valid JSON") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"{field} must be a JSON object")
            return value
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be a JSON object")
        try:
            return json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be JSON serializable") from exc

    @staticmethod
    def _float_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _import_int_or_none(value: Any, field: str) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer") from exc

    @staticmethod
    def _int_or_default(value: Any, default: int = 0) -> int:
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _reasoning_json_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    @staticmethod
    def _import_error(index: int, session_id: str, error: str) -> Dict[str, Any]:
        item: Dict[str, Any] = {"index": index, "error": error}
        if session_id:
            item["session_id"] = session_id
        return item

    def import_sessions(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import sessions exported by :meth:`export_session` or ``export_all``.

        Existing session IDs are skipped. Imported child sessions keep their
        parent only when that parent already exists or is included in the same
        import payload; otherwise the child is detached so partial imports don't
        fail foreign-key validation. Gateway routing, handoff, rewind, and other
        live runtime state are intentionally reset: this restores conversation
        history, not ownership of a live channel or process.
        """
        if not isinstance(sessions, list):
            raise ValueError("sessions must be a list")
        if len(sessions) > self._IMPORT_MAX_SESSIONS:
            raise ValueError(
                f"sessions must contain at most {self._IMPORT_MAX_SESSIONS} entries"
            )

        normalized: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        total_messages = 0
        total_bytes = 0
        session_text_fields = (
            "source",
            "user_id",
            "model",
            "system_prompt",
            "end_reason",
            "cwd",
            "git_branch",
            "git_repo_root",
            "billing_provider",
            "billing_base_url",
            "billing_mode",
            "cost_status",
            "cost_source",
            "pricing_version",
            "title",
        )
        message_text_fields = (
            "role",
            "tool_call_id",
            "tool_name",
            "effect_disposition",
            "finish_reason",
            "reasoning",
            "reasoning_content",
            "platform_message_id",
            "message_id",
        )

        for index, raw in enumerate(sessions):
            if not isinstance(raw, dict):
                errors.append(self._import_error(index, "", "session must be an object"))
                continue
            session_id = str(raw.get("id") or "").strip()
            if not session_id:
                errors.append(self._import_error(index, "", "session id is required"))
                continue
            if session_id in seen_ids:
                errors.append(self._import_error(index, session_id, "duplicate session id"))
                continue
            messages = raw.get("messages") or []
            if not isinstance(messages, list):
                errors.append(self._import_error(index, session_id, "messages must be a list"))
                continue
            if len(messages) > self._IMPORT_MAX_MESSAGES_PER_SESSION:
                errors.append(
                    self._import_error(
                        index,
                        session_id,
                        "messages exceeds the per-session import limit",
                    )
                )
                continue
            if any(not isinstance(msg, dict) for msg in messages):
                errors.append(
                    self._import_error(
                        index,
                        session_id,
                        "messages must contain only objects",
                    )
                )
                continue

            try:
                session_bytes = len(
                    json.dumps(raw, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                )
            except (TypeError, ValueError):
                errors.append(
                    self._import_error(index, session_id, "session must be JSON serializable")
                )
                continue
            if session_bytes > self._IMPORT_MAX_SESSION_BYTES:
                errors.append(
                    self._import_error(index, session_id, "session exceeds the import size limit")
                )
                continue
            total_bytes += session_bytes
            if total_bytes > self._IMPORT_MAX_TOTAL_BYTES:
                errors.append(
                    self._import_error(index, session_id, "import exceeds the total size limit")
                )
                continue

            try:
                clean_session = dict(raw)
                clean_session["id"] = session_id
                clean_session["model_config"] = self._import_json_object_or_none(
                    clean_session.get("model_config"), "model_config"
                )
                clean_session["parent_session_id"] = self._import_text_or_none(
                    clean_session.get("parent_session_id"), "parent_session_id"
                )
                for field in session_text_fields:
                    clean_session[field] = self._import_text_or_none(
                        clean_session.get(field), field
                    )

                clean_messages: List[Dict[str, Any]] = []
                for message_index, message in enumerate(messages):
                    clean_message = dict(message)
                    role = clean_message.get("role")
                    if not isinstance(role, str) or not role:
                        raise ValueError(f"messages[{message_index}].role must be a non-empty string")
                    for field in message_text_fields:
                        if field == "role":
                            continue
                        clean_message[field] = self._import_text_or_none(
                            clean_message.get(field), field
                        )
                    clean_message["token_count"] = self._import_int_or_none(
                        clean_message.get("token_count"), "token_count"
                    )
                    clean_messages.append(clean_message)
            except ValueError as exc:
                errors.append(self._import_error(index, session_id, str(exc)))
                continue

            total_messages += len(clean_messages)
            if total_messages > self._IMPORT_MAX_TOTAL_MESSAGES:
                errors.append(
                    self._import_error(
                        index,
                        session_id,
                        "messages exceeds the total import limit",
                    )
                )
                continue
            seen_ids.add(session_id)
            normalized.append(
                {"index": index, "session": clean_session, "messages": clean_messages}
            )

        if errors:
            return {
                "ok": False,
                "imported": 0,
                "skipped": 0,
                "detached": 0,
                "errors": errors,
            }

        def _do(conn):
            imported_ids: List[str] = []
            skipped_ids: List[str] = []
            parent_updates: List[tuple[str, str]] = []
            detached = 0

            for item in normalized:
                raw = item["session"]
                messages = item["messages"]
                session_id = str(raw.get("id") or "").strip()
                exists = conn.execute(
                    "SELECT 1 FROM sessions WHERE id = ? LIMIT 1",
                    (session_id,),
                ).fetchone()
                if exists:
                    skipped_ids.append(session_id)
                    continue

                started_at = self._float_or_none(raw.get("started_at"))
                if started_at is None:
                    started_at = time.time()
                archived = 1 if raw.get("archived") else 0

                conn.execute(
                    """INSERT INTO sessions (
                           id, source, user_id, model, model_config, system_prompt,
                           parent_session_id, started_at, ended_at, end_reason,
                           message_count, tool_call_count, input_tokens, output_tokens,
                           cache_read_tokens, cache_write_tokens, reasoning_tokens,
                           cwd, git_branch, git_repo_root,
                           billing_provider, billing_base_url, billing_mode,
                           estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
                           pricing_version, title, api_call_count, archived
                       )
                       VALUES (
                           :id, :source, :user_id, :model, :model_config,
                           :system_prompt, NULL, :started_at, :ended_at,
                           :end_reason, 0, 0, :input_tokens, :output_tokens,
                           :cache_read_tokens, :cache_write_tokens,
                           :reasoning_tokens, :cwd, :git_branch, :git_repo_root,
                           :billing_provider, :billing_base_url, :billing_mode,
                           :estimated_cost_usd, :actual_cost_usd, :cost_status,
                           :cost_source, :pricing_version, :title,
                           :api_call_count, :archived
                       )""",
                    {
                        "id": session_id,
                        "source": str(raw.get("source") or "import"),
                        "user_id": raw.get("user_id"),
                        "model": raw.get("model"),
                        "model_config": raw.get("model_config"),
                        "system_prompt": raw.get("system_prompt"),
                        "started_at": started_at,
                        "ended_at": self._float_or_none(raw.get("ended_at")),
                        "end_reason": raw.get("end_reason"),
                        "input_tokens": self._int_or_default(raw.get("input_tokens")),
                        "output_tokens": self._int_or_default(raw.get("output_tokens")),
                        "cache_read_tokens": self._int_or_default(
                            raw.get("cache_read_tokens")
                        ),
                        "cache_write_tokens": self._int_or_default(
                            raw.get("cache_write_tokens")
                        ),
                        "reasoning_tokens": self._int_or_default(
                            raw.get("reasoning_tokens")
                        ),
                        "cwd": raw.get("cwd"),
                        "git_branch": raw.get("git_branch"),
                        "git_repo_root": raw.get("git_repo_root"),
                        "billing_provider": raw.get("billing_provider"),
                        "billing_base_url": raw.get("billing_base_url"),
                        "billing_mode": raw.get("billing_mode"),
                        "estimated_cost_usd": self._float_or_none(
                            raw.get("estimated_cost_usd")
                        ),
                        "actual_cost_usd": self._float_or_none(
                            raw.get("actual_cost_usd")
                        ),
                        "cost_status": raw.get("cost_status"),
                        "cost_source": raw.get("cost_source"),
                        "pricing_version": raw.get("pricing_version"),
                        "title": raw.get("title"),
                        "api_call_count": self._int_or_default(raw.get("api_call_count")),
                        "archived": archived,
                    },
                )

                sanitized_messages: List[Dict[str, Any]] = []
                for msg in messages:
                    clean = dict(msg)
                    for key in (
                        "reasoning_details",
                        "codex_reasoning_items",
                        "codex_message_items",
                    ):
                        clean[key] = self._reasoning_json_value(clean.get(key))
                    sanitized_messages.append(clean)

                total_messages, total_tool_calls = self._insert_message_rows(
                    conn,
                    session_id,
                    sanitized_messages,
                )
                conn.execute(
                    "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                    (total_messages, total_tool_calls, session_id),
                )

                parent_id = str(raw.get("parent_session_id") or "").strip()
                if parent_id:
                    parent_updates.append((session_id, parent_id))
                imported_ids.append(session_id)

            parent_by_child = dict(parent_updates)

            def _would_create_cycle(session_id: str, parent_id: str) -> bool:
                seen = {session_id}
                current = parent_id
                while current:
                    if current in seen:
                        return True
                    seen.add(current)
                    if current in parent_by_child:
                        current = parent_by_child[current]
                        continue
                    row = conn.execute(
                        "SELECT parent_session_id FROM sessions WHERE id = ? LIMIT 1",
                        (current,),
                    ).fetchone()
                    if row is None:
                        return False
                    current = row["parent_session_id"]
                return False

            for session_id, parent_id in parent_updates:
                parent_exists = conn.execute(
                    "SELECT 1 FROM sessions WHERE id = ? LIMIT 1",
                    (parent_id,),
                ).fetchone()
                if parent_exists and not _would_create_cycle(session_id, parent_id):
                    conn.execute(
                        "UPDATE sessions SET parent_session_id = ? WHERE id = ?",
                        (parent_id, session_id),
                    )
                else:
                    # Drop only the closing edge. Later entries can still attach
                    # to this now-root session, preserving the acyclic portion
                    # of a malformed imported lineage.
                    parent_by_child.pop(session_id, None)
                    detached += 1

            return {
                "ok": True,
                "imported": len(imported_ids),
                "skipped": len(skipped_ids),
                "detached": detached,
                "imported_ids": imported_ids,
                "skipped_ids": skipped_ids,
                "errors": [],
            }

        return self._execute_write(_do)

    def clear_messages(self, session_id: str) -> None:
        """Delete all messages for a session and reset its counters."""
        def _do(conn):
            conn.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            conn.execute(
                "UPDATE sessions SET message_count = 0, tool_call_count = 0 WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    @staticmethod
    def _remove_session_files(sessions_dir: Optional[Path], session_id: str) -> None:
        """Remove on-disk transcript files for a session.

        Cleans up ``{session_id}.json``, ``{session_id}.jsonl``, and any
        ``request_dump_{session_id}_*.json`` files left by the gateway.
        Silently skips files that don't exist and swallows OSError so a
        filesystem hiccup never blocks a DB operation.
        """
        if sessions_dir is None:
            return
        for suffix in (".json", ".jsonl"):
            p = sessions_dir / f"{session_id}{suffix}"
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass
        # request_dump files use session_id as a prefix component
        try:
            for p in sessions_dir.glob(f"request_dump_{session_id}_*.json"):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
        except OSError:
            pass

    def delete_session(
        self,
        session_id: str,
        sessions_dir: Optional[Path] = None,
    ) -> bool:
        """Delete a session and all its messages.

        Delegate subagent children (``model_config._delegate_from``) are
        cascade-deleted with the parent so they never resurface in session
        pickers as orphaned rows. Branch / compression children are orphaned
        (``parent_session_id → NULL``) so they remain accessible independently.
        When *sessions_dir* is provided, also removes on-disk transcript
        files (``.json`` / ``.jsonl`` / ``request_dump_*``) for every deleted
        session. Returns True if the session was found and deleted.
        """
        removed_delegate_ids: List[str] = []

        def _do(conn):
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE id = ?", (session_id,)
            )
            if cursor.fetchone()[0] == 0:
                return False
            removed_delegate_ids.extend(_delete_delegate_children(conn, [session_id]))
            # Orphan remaining child sessions (branches, etc.) so FK is satisfied.
            conn.execute(
                "UPDATE sessions SET parent_session_id = NULL "
                "WHERE parent_session_id = ?",
                (session_id,),
            )
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            return True

        deleted = self._execute_write(_do)
        if deleted:
            for delegate_id in removed_delegate_ids:
                self._remove_session_files(sessions_dir, delegate_id)
            self._remove_session_files(sessions_dir, session_id)
        return bool(deleted)

    def delete_session_if_empty(
        self,
        session_id: str,
        sessions_dir: Optional[Path] = None,
    ) -> bool:
        """Delete *session_id* only when it never gained resumable content.

        A session is considered empty when it has no messages and no
        user-assigned title. Used by CLI exit / session-rotation paths so
        immediately-started-and-quit sessions don't pile up in ``/resume``
        and ``hermes sessions list`` output. (Pattern ported from
        google-gemini/gemini-cli#27770.)

        The emptiness check and delete run in one transaction, so a message
        flushed concurrently by another writer can't be lost. Sessions with
        children (delegate subagent runs) are preserved — a parent that
        spawned work is not "empty" even if its own transcript never
        flushed. Returns True if the session was deleted.
        """
        def _do(conn):
            cursor = conn.execute(
                """
                DELETE FROM sessions
                WHERE id = ?
                  AND title IS NULL
                  AND NOT EXISTS (
                      SELECT 1 FROM messages WHERE messages.session_id = sessions.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM sessions child
                      WHERE child.parent_session_id = sessions.id
                  )
                """,
                (session_id,),
            )
            return cursor.rowcount > 0

        deleted = self._execute_write(_do)
        if deleted:
            self._remove_session_files(sessions_dir, session_id)
        return bool(deleted)

    def delete_sessions(
        self,
        session_ids: List[str],
        sessions_dir: Optional[Path] = None,
    ) -> int:
        """Delete every session in *session_ids* in a single transaction.

        Backs the dashboard's bulk-select-then-delete flow on the
        sessions page (``POST /api/sessions/bulk-delete``). Mirrors the
        single-session :meth:`delete_session` contract per row:

        * Unknown IDs are silently skipped (no 404) — selection state
          in the UI can race against another tab's delete, and we'd
          rather succeed-on-the-rest than fail-the-whole-batch.
        * Delegate subagent children (``model_config._delegate_from``) are
          cascade-deleted with their parent; branch children are orphaned
          (``parent_session_id → NULL``) so they stay accessible.
        * Messages and the session row both go in one
          ``_execute_write`` call so a partial failure can't leave the
          DB in a "messages gone but session row still there" state.
        * On-disk transcript / ``request_dump_*`` files are cleaned up
          outside the DB transaction when *sessions_dir* is provided,
          matching :meth:`prune_sessions` and
          :meth:`delete_empty_sessions`.

        Returns the count of sessions that actually existed and were
        deleted (may be less than ``len(session_ids)`` if some IDs were
        already gone).
        """
        if not session_ids:
            return 0
        # Dedup + drop any non-string entries up-front. Avoids
        # double-counting in the WHERE-IN list and protects against
        # callers that pass a list with stray ``None`` values.
        unique_ids = list({sid for sid in session_ids if isinstance(sid, str) and sid})
        if not unique_ids:
            return 0

        removed_ids: list[str] = []
        removed_delegate_ids: list[str] = []

        def _do(conn):
            placeholders = ",".join("?" * len(unique_ids))
            # First, filter to IDs that actually exist — we want to
            # return the real deleted count, not the input length.
            cursor = conn.execute(
                f"SELECT id FROM sessions WHERE id IN ({placeholders})",
                unique_ids,
            )
            existing = [row["id"] for row in cursor.fetchall()]
            if not existing:
                return 0

            existing_placeholders = ",".join("?" * len(existing))
            removed_delegate_ids.extend(_delete_delegate_children(conn, existing))
            # Orphan remaining children whose parent is in the kill list so the
            # FK constraint stays satisfied. Pin children whose parent
            # is itself in the kill list rather than NULL-ing parents
            # of survivors — the IN list on ``parent_session_id`` does
            # exactly this.
            conn.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({existing_placeholders})",
                existing,
            )
            conn.execute(
                f"DELETE FROM messages WHERE session_id IN ({existing_placeholders})",
                existing,
            )
            conn.execute(
                f"DELETE FROM sessions WHERE id IN ({existing_placeholders})",
                existing,
            )
            removed_ids.extend(existing)
            return len(existing)

        count = self._execute_write(_do)
        for sid in removed_delegate_ids:
            self._remove_session_files(sessions_dir, sid)
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count

    def count_empty_sessions(self) -> int:
        """Return the count of empty, non-active, non-archived sessions.

        "Empty" = ``message_count = 0`` AND the session has ended
        (``ended_at IS NOT NULL``) AND is not archived. The ``ended_at``
        guard matches the safety contract used by :meth:`prune_sessions`:
        only ended sessions are candidates for bulk deletion, so a freshly
        spawned session whose first message hasn't landed yet — or one
        held open by the live agent — is never sniped out from under
        the runtime.

        Backs the ``GET /api/sessions/empty/count`` endpoint that lets the
        web dashboard hide its "Delete empty" button when there's nothing
        to clean up, and pre-populate the confirm dialog with the actual
        count.
        """
        with self._lock:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM sessions "
                "WHERE message_count = 0 "
                "AND ended_at IS NOT NULL "
                "AND archived = 0"
            )
            return cursor.fetchone()[0]

    def delete_empty_sessions(
        self,
        sessions_dir: Optional[Path] = None,
    ) -> int:
        """Delete every empty, ended, non-archived session.

        Mirrors :meth:`prune_sessions`' transactional shape:

        * Selects candidate IDs first (``message_count = 0`` AND
          ``ended_at IS NOT NULL`` AND ``archived = 0``) so we never
          touch a live session or one the user deliberately archived.
        * Orphans any child whose parent is in the kill list — children
          of an empty parent are kept and re-parented to ``NULL`` rather
          than cascade-deleted, matching ``delete_session`` /
          ``prune_sessions`` semantics so branch/subagent transcripts
          survive an inadvertent parent cleanup.
        * Deletes the rows in a single ``_execute_write`` callback so
          the operation is atomic — a partial failure (e.g. SIGKILL
          mid-loop) doesn't leave the DB in a "messages-deleted but
          session-row-still-there" half-state.
        * Cleans up on-disk transcript files (``.json`` / ``.jsonl`` /
          ``request_dump_*``) outside the DB transaction when
          ``sessions_dir`` is provided. Empty sessions don't typically
          have transcript files, but the gateway can leave a stub
          ``request_dump_*`` if it crashed before the first reply —
          so we still sweep, matching ``prune_sessions``.

        Returns the number of sessions deleted.
        """
        removed_ids: list[str] = []

        def _do(conn):
            cursor = conn.execute(
                "SELECT id FROM sessions "
                "WHERE message_count = 0 "
                "AND ended_at IS NOT NULL "
                "AND archived = 0"
            )
            session_ids = {row["id"] for row in cursor.fetchall()}

            if not session_ids:
                return 0

            placeholders = ",".join("?" * len(session_ids))
            conn.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({placeholders})",
                list(session_ids),
            )

            for sid in session_ids:
                # DELETE FROM messages is paranoia — by construction
                # these rows have ``message_count = 0`` — but if a
                # bookkeeping bug ever lets the counter drift below the
                # real row count, we still leave a clean FK state.
                conn.execute(
                    "DELETE FROM messages WHERE session_id = ?", (sid,)
                )
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
                removed_ids.append(sid)
            return len(session_ids)

        count = self._execute_write(_do)
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count

    @staticmethod
    def _prune_filter_where(
        *,
        started_before: Optional[float] = None,
        started_after: Optional[float] = None,
        source: Optional[str] = None,
        title_like: Optional[str] = None,
        end_reason: Optional[str] = None,
        cwd_prefix: Optional[str] = None,
        min_messages: Optional[int] = None,
        max_messages: Optional[int] = None,
        archived: Optional[bool] = None,
        model_like: Optional[str] = None,
        provider: Optional[str] = None,
        user_id: Optional[str] = None,
        chat_id: Optional[str] = None,
        chat_type: Optional[str] = None,
        branch_like: Optional[str] = None,
        min_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        min_cost: Optional[float] = None,
        max_cost: Optional[float] = None,
        min_tool_calls: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
    ) -> Tuple[str, list]:
        """Build the shared WHERE clause for bulk prune/archive selection.

        All filters AND together. Only ended sessions are ever candidates
        (``ended_at IS NOT NULL``) so a live session is never selected.
        ``archived`` is a tri-state: ``None`` = both, ``True`` = only
        archived rows, ``False`` = only unarchived rows.

        String matching conventions: ``model_like`` / ``branch_like`` /
        ``title_like`` are case-insensitive substring matches (model slugs
        and branch names vary in prefix format); ``provider`` / ``user_id``
        / ``chat_id`` / ``chat_type`` / ``source`` / ``end_reason`` are
        exact (case-insensitive for provider). Token bounds apply to
        ``input_tokens + output_tokens``; cost bounds apply to
        ``COALESCE(actual_cost_usd, estimated_cost_usd)``.

        The clause references the ``s`` table alias — callers must select
        ``FROM sessions s``.
        """
        clauses = ["s.ended_at IS NOT NULL"]
        params: list = []
        if started_before is not None:
            clauses.append("s.started_at < ?")
            params.append(started_before)
        if started_after is not None:
            clauses.append("s.started_at >= ?")
            params.append(started_after)
        if source:
            clauses.append("s.source = ?")
            params.append(source)
        if title_like:
            clauses.append("LOWER(COALESCE(s.title, '')) LIKE ?")
            params.append(f"%{title_like.lower()}%")
        if end_reason:
            clauses.append("s.end_reason = ?")
            params.append(end_reason)
        if cwd_prefix:
            clause, clause_params = _cwd_prefix_clause(cwd_prefix)
            clauses.append(clause)
            params.extend(clause_params)
        if min_messages is not None:
            clauses.append("s.message_count >= ?")
            params.append(min_messages)
        if max_messages is not None:
            clauses.append("s.message_count <= ?")
            params.append(max_messages)
        if model_like:
            clauses.append("LOWER(COALESCE(s.model, '')) LIKE ?")
            params.append(f"%{model_like.lower()}%")
        if provider:
            clauses.append("LOWER(COALESCE(s.billing_provider, '')) = ?")
            params.append(provider.lower())
        if user_id:
            clauses.append("s.user_id = ?")
            params.append(user_id)
        if chat_id:
            clauses.append("s.chat_id = ?")
            params.append(chat_id)
        if chat_type:
            clauses.append("s.chat_type = ?")
            params.append(chat_type)
        if branch_like:
            clauses.append("LOWER(COALESCE(s.git_branch, '')) LIKE ?")
            params.append(f"%{branch_like.lower()}%")
        if min_tokens is not None:
            clauses.append(
                "(COALESCE(s.input_tokens, 0) + COALESCE(s.output_tokens, 0)) >= ?"
            )
            params.append(min_tokens)
        if max_tokens is not None:
            clauses.append(
                "(COALESCE(s.input_tokens, 0) + COALESCE(s.output_tokens, 0)) <= ?"
            )
            params.append(max_tokens)
        if min_cost is not None:
            clauses.append(
                "COALESCE(s.actual_cost_usd, s.estimated_cost_usd, 0) >= ?"
            )
            params.append(min_cost)
        if max_cost is not None:
            clauses.append(
                "COALESCE(s.actual_cost_usd, s.estimated_cost_usd, 0) <= ?"
            )
            params.append(max_cost)
        if min_tool_calls is not None:
            clauses.append("COALESCE(s.tool_call_count, 0) >= ?")
            params.append(min_tool_calls)
        if max_tool_calls is not None:
            clauses.append("COALESCE(s.tool_call_count, 0) <= ?")
            params.append(max_tool_calls)
        if archived is True:
            clauses.append("s.archived = 1")
        elif archived is False:
            clauses.append("s.archived = 0")
        return " AND ".join(clauses), params

    def list_prune_candidates(
        self,
        older_than_days: Optional[float] = None,
        source: str = None,
        **filters,
    ) -> List[Dict[str, Any]]:
        """Return the sessions a matching :meth:`prune_sessions` /
        :meth:`archive_sessions` call would touch, without modifying anything.

        Backs ``--dry-run`` and pre-confirmation counts. Accepts the same
        keyword filters as :meth:`_prune_filter_where` (unknown names raise
        ``TypeError`` there). Rows are ordered oldest-first and carry
        ``id, source, title, model, started_at, ended_at, message_count,
        archived``.
        """
        if filters.get("started_before") is None and older_than_days is not None:
            filters["started_before"] = time.time() - (older_than_days * 86400)
        where, params = self._prune_filter_where(source=source, **filters)
        with self._lock:
            cursor = self._conn.execute(
                f"""SELECT s.id, s.source, s.title, s.model, s.started_at,
                           s.ended_at, s.message_count, s.archived
                    FROM sessions s WHERE {where}
                    ORDER BY s.started_at ASC""",
                params,
            )
            return [dict(row) for row in cursor.fetchall()]

    def archive_sessions(
        self,
        older_than_days: Optional[float] = None,
        source: str = None,
        **filters,
    ) -> int:
        """Bulk-archive (soft-hide) every session matching the filters.

        Same filter surface as :meth:`prune_sessions`, but instead of deleting
        rows it flips ``archived = 1`` via :meth:`set_session_archived` so
        each match's compression lineage is archived as a unit (an unarchived
        compression root would otherwise resurrect the conversation in
        Desktop's projected list). Nothing is deleted; messages and transcript
        files are untouched. Returns the number of sessions matched.

        ``archived`` defaults to ``False`` here (only select rows not yet
        archived) so repeat runs are idempotent no-ops.
        """
        filters.setdefault("archived", False)
        rows = self.list_prune_candidates(
            older_than_days=older_than_days, source=source, **filters
        )
        for row in rows:
            self.set_session_archived(row["id"], True)
        return len(rows)

    def archive_stale_sessions(
        self, idle_days: float, *, exclude_pinned: bool = True
    ) -> int:
        """Archive every session untouched for at least ``idle_days`` days.

        "Touched" is the latest message timestamp (falling back to
        ``started_at``) — i.e. real recency, not creation time — so a session
        created long ago but active yesterday is spared, while an old
        abandoned one (even a still-open one) is swept. This differs from
        :meth:`archive_sessions`, which ages on ``started_at`` and only ended
        sessions.

        Guards:
          * ``pinned = 0`` when ``exclude_pinned`` (the Desktop "keep" flag).
          * ``archived = 0`` so repeat runs are idempotent no-ops.
          * only lineage *tips* / standalone rows are candidates
            (``end_reason <> 'compression'``); a stale tip archives its whole
            chain via :meth:`set_session_archived`, so we never resurrect an
            active conversation by matching an old compressed-away root whose
            live continuation is recent.

        Returns the number of sessions archived. Never raises for an empty or
        non-positive ``idle_days`` — it simply archives nothing.
        """
        if idle_days is None or idle_days < 0:
            return 0
        cutoff = time.time() - float(idle_days) * 86400.0
        pin_clause = "AND s.pinned = 0" if exclude_pinned else ""
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT s.id FROM sessions s
                WHERE s.archived = 0
                  AND COALESCE(s.end_reason, '') <> 'compression'
                  {pin_clause}
                  AND COALESCE(
                        (SELECT MAX(m.timestamp) FROM messages m
                         WHERE m.session_id = s.id),
                        s.started_at
                      ) < ?
                ORDER BY s.started_at ASC
                """,
                (cutoff,),
            ).fetchall()
        ids = [(r["id"] if isinstance(r, sqlite3.Row) else r[0]) for r in rows]
        for sid in ids:
            self.set_session_archived(sid, True)
        return len(ids)

    def prune_sessions(
        self,
        older_than_days: Optional[float] = 90,
        source: str = None,
        sessions_dir: Optional[Path] = None,
        **filters,
    ) -> int:
        """Delete sessions matching the filters. Returns count deleted.

        Default behavior (no keyword filters) is unchanged: delete ended
        sessions older than ``older_than_days`` days, optionally restricted
        to ``source``. Additional keyword filters AND together — the full
        set is defined by :meth:`_prune_filter_where`:

        * ``started_before`` / ``started_after`` — epoch bounds on
          ``started_at``. ``started_before`` overrides ``older_than_days``;
          pass ``older_than_days=None`` for no upper age bound (e.g. when
          only pruning a recent window via ``started_after``).
        * ``title_like`` / ``model_like`` / ``branch_like`` —
          case-insensitive substring matches.
        * ``end_reason`` / ``provider`` / ``user_id`` / ``chat_id`` /
          ``chat_type`` — exact matches (provider case-insensitive, against
          ``billing_provider``).
        * ``cwd_prefix`` — session cwd equals or is under this path.
        * ``min_messages`` / ``max_messages`` — bounds on message_count.
        * ``min_tokens`` / ``max_tokens`` — bounds on input+output tokens.
        * ``min_cost`` / ``max_cost`` — bounds on USD cost
          (actual, falling back to estimated).
        * ``min_tool_calls`` / ``max_tool_calls`` — bounds on tool_call_count.
        * ``archived`` — tri-state: None = both (default), True = only
          archived, False = only unarchived.

        Only prunes ended sessions (not active ones).  Child sessions outside
        the prune window are orphaned (parent_session_id set to NULL) rather
        than cascade-deleted.  When *sessions_dir* is provided, also removes
        on-disk transcript files (``.json`` / ``.jsonl`` /
        ``request_dump_*``) for every pruned session, outside the DB
        transaction.
        """
        if filters.get("started_before") is None and older_than_days is not None:
            filters["started_before"] = time.time() - (older_than_days * 86400)
        where, where_params = self._prune_filter_where(source=source, **filters)
        removed_ids: list[str] = []

        def _do(conn):
            cursor = conn.execute(
                f"SELECT s.id FROM sessions s WHERE {where}", where_params
            )
            session_ids = {row["id"] for row in cursor.fetchall()}

            if not session_ids:
                return 0

            # Orphan any sessions whose parent is about to be deleted
            placeholders = ",".join("?" * len(session_ids))
            conn.execute(
                f"UPDATE sessions SET parent_session_id = NULL "
                f"WHERE parent_session_id IN ({placeholders})",
                list(session_ids),
            )

            for sid in session_ids:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (sid,))
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
                removed_ids.append(sid)
            return len(session_ids)

        count = self._execute_write(_do)
        # Clean up on-disk files outside the DB transaction
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count

    # ── Meta key/value (for scheduler bookkeeping) ──

    def get_meta(self, key: str) -> Optional[str]:
        """Read a value from the state_meta key/value store."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM state_meta WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return row["value"] if isinstance(row, sqlite3.Row) else row[0]

    def set_meta(
        self, key: str, value: str, *, cursor: Optional[sqlite3.Cursor] = None
    ) -> None:
        """Write a value to the state_meta key/value store.

        When ``cursor`` is provided the write is issued on that cursor
        inline (used during ``_init_schema``, which already holds an open
        transaction — routing through ``_execute_write`` there would nest
        BEGIN IMMEDIATE and deadlock). Otherwise a normal write transaction
        is used.
        """
        if cursor is not None:
            cursor.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            return

        def _do(conn):
            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
        self._execute_write(_do)

    def apply_telegram_topic_migration(self) -> None:
        """Create Telegram DM topic-mode tables on explicit /topic opt-in.

        This migration is deliberately not part of automatic SessionDB startup
        reconciliation. Operators must be able to upgrade Hermes, keep the old
        Telegram bot behavior running, and only mutate topic-mode state when the
        user executes /topic to opt into the feature.

        Schema versions:
          v1 — initial shape (no ON DELETE CASCADE on session_id FK)
          v2 — session_id FK gets ON DELETE CASCADE so session pruning
               automatically clears bindings.
        """
        def _do(conn):
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS telegram_dm_topic_mode (
                    chat_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    activated_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    has_topics_enabled INTEGER,
                    allows_users_to_create_topics INTEGER,
                    capability_checked_at REAL,
                    intro_message_id TEXT,
                    pinned_message_id TEXT
                );

                CREATE TABLE IF NOT EXISTS telegram_dm_topic_bindings (
                    chat_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    managed_mode TEXT NOT NULL DEFAULT 'auto',
                    linked_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (chat_id, thread_id)
                );

                CREATE UNIQUE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_session
                ON telegram_dm_topic_bindings(session_id);

                CREATE INDEX IF NOT EXISTS idx_telegram_dm_topic_bindings_user
                ON telegram_dm_topic_bindings(user_id, chat_id);
                """
            )

            # v1 → v2: rebuild telegram_dm_topic_bindings if its session_id FK
            # lacks ON DELETE CASCADE. SQLite can't ALTER a foreign key, so we
            # rebuild the table. Only runs once per DB (version gate).
            current = conn.execute(
                "SELECT value FROM state_meta WHERE key = ?",
                ("telegram_dm_topic_schema_version",),
            ).fetchone()
            current_version = int(current[0]) if current and str(current[0]).isdigit() else 0
            if current_version < 2:
                fk_rows = conn.execute(
                    "PRAGMA foreign_key_list('telegram_dm_topic_bindings')"
                ).fetchall()
                needs_rebuild = any(
                    row[2] == "sessions" and (row[6] or "") != "CASCADE"
                    for row in fk_rows
                )
                if needs_rebuild:
                    conn.executescript(
                        """
                        CREATE TABLE telegram_dm_topic_bindings_new (
                            chat_id TEXT NOT NULL,
                            thread_id TEXT NOT NULL,
                            user_id TEXT NOT NULL,
                            session_key TEXT NOT NULL,
                            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                            managed_mode TEXT NOT NULL DEFAULT 'auto',
                            linked_at REAL NOT NULL,
                            updated_at REAL NOT NULL,
                            PRIMARY KEY (chat_id, thread_id)
                        );
                        INSERT INTO telegram_dm_topic_bindings_new
                            SELECT chat_id, thread_id, user_id, session_key,
                                   session_id, managed_mode, linked_at, updated_at
                            FROM telegram_dm_topic_bindings;
                        DROP TABLE telegram_dm_topic_bindings;
                        ALTER TABLE telegram_dm_topic_bindings_new
                            RENAME TO telegram_dm_topic_bindings;
                        CREATE UNIQUE INDEX idx_telegram_dm_topic_bindings_session
                            ON telegram_dm_topic_bindings(session_id);
                        CREATE INDEX idx_telegram_dm_topic_bindings_user
                            ON telegram_dm_topic_bindings(user_id, chat_id);
                        """
                    )

            conn.execute(
                "INSERT INTO state_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                ("telegram_dm_topic_schema_version", "2"),
            )
        self._execute_write(_do)

    def enable_telegram_topic_mode(
        self,
        *,
        chat_id: str,
        user_id: str,
        has_topics_enabled: Optional[bool] = None,
        allows_users_to_create_topics: Optional[bool] = None,
    ) -> None:
        """Enable Telegram DM topic mode for one private chat/user.

        This method intentionally owns the explicit topic migration. Ordinary
        SessionDB startup must not create these side tables.
        """
        self.apply_telegram_topic_migration()
        now = time.time()

        def _to_int(value: Optional[bool]) -> Optional[int]:
            if value is None:
                return None
            return 1 if value else 0

        def _do(conn):
            conn.execute(
                """
                INSERT INTO telegram_dm_topic_mode (
                    chat_id, user_id, enabled, activated_at, updated_at,
                    has_topics_enabled, allows_users_to_create_topics,
                    capability_checked_at
                ) VALUES (?, ?, 1, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    enabled = 1,
                    updated_at = excluded.updated_at,
                    has_topics_enabled = excluded.has_topics_enabled,
                    allows_users_to_create_topics = excluded.allows_users_to_create_topics,
                    capability_checked_at = excluded.capability_checked_at
                """,
                (
                    str(chat_id),
                    str(user_id),
                    now,
                    now,
                    _to_int(has_topics_enabled),
                    _to_int(allows_users_to_create_topics),
                    now,
                ),
            )
        self._execute_write(_do)

    def disable_telegram_topic_mode(
        self,
        *,
        chat_id: str,
        clear_bindings: bool = True,
    ) -> None:
        """Disable Telegram DM topic mode for one private chat.

        When ``clear_bindings`` is True (default) the (chat_id, thread_id)
        bindings for this chat are also cleared so re-enabling later
        starts from a clean slate. Set to False if the operator wants to
        preserve bindings for a later re-enable.

        Never creates the topic-mode tables from scratch; if they don't
        exist there is nothing to disable and the call is a no-op.
        """
        def _do(conn):
            try:
                conn.execute(
                    "UPDATE telegram_dm_topic_mode SET enabled = 0, updated_at = ? "
                    "WHERE chat_id = ?",
                    (time.time(), str(chat_id)),
                )
                if clear_bindings:
                    conn.execute(
                        "DELETE FROM telegram_dm_topic_bindings WHERE chat_id = ?",
                        (str(chat_id),),
                    )
            except sqlite3.OperationalError:
                # Tables don't exist yet — nothing to disable.
                return
        self._execute_write(_do)

    def is_telegram_topic_mode_enabled(self, *, chat_id: str, user_id: str) -> bool:
        """Return whether Telegram DM topic mode is enabled for this chat/user."""
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT enabled FROM telegram_dm_topic_mode
                    WHERE chat_id = ? AND user_id = ?
                    """,
                    (str(chat_id), str(user_id)),
                ).fetchone()
            except sqlite3.OperationalError:
                return False
        if row is None:
            return False
        enabled = row["enabled"] if isinstance(row, sqlite3.Row) else row[0]
        return bool(enabled)

    def get_telegram_topic_binding(
        self,
        *,
        chat_id: str,
        thread_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the session binding for a Telegram DM topic, if present."""
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT * FROM telegram_dm_topic_bindings
                    WHERE chat_id = ? AND thread_id = ?
                    """,
                    (str(chat_id), str(thread_id)),
                ).fetchone()
            except sqlite3.OperationalError:
                return None
        return dict(row) if row else None

    def list_telegram_topic_bindings_for_chat(
        self,
        *,
        chat_id: str,
    ) -> List[Dict[str, Any]]:
        """All Telegram DM topic bindings for one chat, newest first.

        Read-only; returns [] if the bindings table doesn't exist yet
        (does not trigger the topic-mode migration).
        """
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT * FROM telegram_dm_topic_bindings "
                    "WHERE chat_id = ? ORDER BY updated_at DESC",
                    (str(chat_id),),
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [dict(row) for row in rows]

    def get_telegram_topic_binding_by_session(
        self,
        *,
        session_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the Telegram DM topic binding for a given session_id, if present.

        Uses the UNIQUE INDEX on telegram_dm_topic_bindings(session_id) for an
        efficient reverse lookup. Returns None when the session has no binding or
        the table does not exist yet.
        """
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT * FROM telegram_dm_topic_bindings
                    WHERE session_id = ?
                    """,
                    (str(session_id),),
                ).fetchone()
            except sqlite3.OperationalError:
                return None
        return dict(row) if row else None

    def delete_telegram_topic_binding(
        self,
        *,
        chat_id: str,
        thread_id: str,
    ) -> int:
        """Remove the binding row for a single (chat, thread) pair.

        Called when the Telegram Bot API confirms a topic was deleted
        externally (``Thread not found`` after the same-thread retry
        already failed).  Without this prune, the stale row keeps
        living in ``telegram_dm_topic_bindings`` and the
        recovery logic in ``gateway.run._recover_telegram_topic_thread_id``
        cheerfully redirects future inbound messages to the deleted
        topic, causing tool progress, approvals, and replies to land
        in the wrong place.  Issue #31501.

        When this prune removes the chat's *last* remaining binding,
        the chat's row in ``telegram_dm_topic_mode`` is also flipped to
        ``enabled = 0`` in the same transaction.  Otherwise the chat
        would be left in topic mode with zero lanes — and
        ``gateway.run._recover_telegram_topic_thread_id`` keeps treating
        the chat as topic-enabled, lobby messages keep hunting for a
        binding that no longer exists, and a user who disabled topics in
        the Telegram client (rather than via ``/topic off``) stays stuck
        until the next send happens to fail. Clearing the flag makes
        recovery fully stand down once the dead topics are gone.

        Returns the number of binding rows deleted (0 when the binding
        was already absent or the topic-mode tables haven't been
        migrated yet — both are silent no-ops; we never raise from
        a cleanup hot path).
        """
        chat_id = str(chat_id)
        thread_id = str(thread_id)
        deleted = {"count": 0}

        def _do(conn):
            try:
                cursor = conn.execute(
                    """
                    DELETE FROM telegram_dm_topic_bindings
                    WHERE chat_id = ? AND thread_id = ?
                    """,
                    (chat_id, thread_id),
                )
                deleted["count"] = cursor.rowcount or 0
            except sqlite3.OperationalError:
                # Tables don't exist yet — nothing to prune.
                deleted["count"] = 0
                return
            if not deleted["count"]:
                return
            # If that was the chat's last binding, disable topic mode for
            # the chat so recovery stops steering lobby messages at a now
            # empty lane set. Same transaction → no read-after-prune race.
            try:
                remaining = conn.execute(
                    """
                    SELECT 1 FROM telegram_dm_topic_bindings
                    WHERE chat_id = ? LIMIT 1
                    """,
                    (chat_id,),
                ).fetchone()
                if remaining is None:
                    conn.execute(
                        "UPDATE telegram_dm_topic_mode "
                        "SET enabled = 0, updated_at = ? WHERE chat_id = ?",
                        (time.time(), chat_id),
                    )
            except sqlite3.OperationalError:
                # telegram_dm_topic_mode absent — binding prune still stands.
                pass

        self._execute_write(_do)
        return deleted["count"]

    def bind_telegram_topic(
        self,
        *,
        chat_id: str,
        thread_id: str,
        user_id: str,
        session_key: str,
        session_id: str,
        managed_mode: str = "auto",
    ) -> None:
        """Bind one Telegram DM topic thread to one Hermes session.

        A Hermes session may only be linked to one Telegram topic in MVP.
        Rebinding the same topic to the same session is idempotent; trying to
        link the same session to a different topic raises ValueError.
        """
        self.apply_telegram_topic_migration()
        now = time.time()
        chat_id = str(chat_id)
        thread_id = str(thread_id)
        user_id = str(user_id)
        session_key = str(session_key)
        session_id = str(session_id)

        def _do(conn):
            existing_session = conn.execute(
                """
                SELECT chat_id, thread_id FROM telegram_dm_topic_bindings
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if existing_session is not None:
                linked_chat = existing_session["chat_id"] if isinstance(existing_session, sqlite3.Row) else existing_session[0]
                linked_thread = existing_session["thread_id"] if isinstance(existing_session, sqlite3.Row) else existing_session[1]
                if str(linked_chat) != chat_id or str(linked_thread) != thread_id:
                    raise ValueError("session is already linked to another Telegram topic")

            conn.execute(
                """
                INSERT INTO telegram_dm_topic_bindings (
                    chat_id, thread_id, user_id, session_key, session_id,
                    managed_mode, linked_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, thread_id) DO UPDATE SET
                    user_id = excluded.user_id,
                    session_key = excluded.session_key,
                    session_id = excluded.session_id,
                    managed_mode = excluded.managed_mode,
                    updated_at = excluded.updated_at
                """,
                (
                    chat_id,
                    thread_id,
                    user_id,
                    session_key,
                    session_id,
                    managed_mode,
                    now,
                    now,
                ),
            )
        self._execute_write(_do)

    def is_telegram_session_linked_to_topic(self, *, session_id: str) -> bool:
        """Return True if a Hermes session is already bound to any Telegram DM topic.

        Read-only: does NOT trigger the telegram-topic migration. If the
        topic-mode tables have not been created yet (i.e. nobody has run
        ``/topic`` in this profile), the session is by definition unbound
        and we return False.
        """
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT 1 FROM telegram_dm_topic_bindings
                    WHERE session_id = ?
                    LIMIT 1
                    """,
                    (str(session_id),),
                ).fetchone()
            except sqlite3.OperationalError:
                return False
        return row is not None

    def list_unlinked_telegram_sessions_for_user(
        self,
        *,
        chat_id: str,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """List previous Telegram sessions for this user that are not bound to a topic.

        Read-only: does NOT trigger the telegram-topic migration. If the
        topic-mode tables are absent, fall back to a simpler query that
        just returns this user's Telegram sessions — there can't be any
        bindings yet.
        """
        with self._lock:
            try:
                rows = self._conn.execute(
                    """
                    SELECT s.*,
                        COALESCE(
                            (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                             FROM messages m
                             WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                             ORDER BY m.timestamp, m.id LIMIT 1),
                            ''
                        ) AS _preview_raw,
                        COALESCE(
                            (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                            s.started_at
                        ) AS last_active
                    FROM sessions s
                    WHERE s.source = 'telegram'
                      AND s.user_id = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM telegram_dm_topic_bindings b
                          WHERE b.session_id = s.id
                      )
                    ORDER BY last_active DESC, s.started_at DESC
                    LIMIT ?
                    """,
                    (str(user_id), int(limit)),
                ).fetchall()
            except sqlite3.OperationalError:
                # telegram_dm_topic_bindings doesn't exist yet — no bindings
                # means every telegram session for this user is "unlinked".
                rows = self._conn.execute(
                    """
                    SELECT s.*,
                        COALESCE(
                            (SELECT SUBSTR(REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' '), 1, 63)
                             FROM messages m
                             WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                             ORDER BY m.timestamp, m.id LIMIT 1),
                            ''
                        ) AS _preview_raw,
                        COALESCE(
                            (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                            s.started_at
                        ) AS last_active
                    FROM sessions s
                    WHERE s.source = 'telegram'
                      AND s.user_id = ?
                    ORDER BY last_active DESC, s.started_at DESC
                    LIMIT ?
                    """,
                    (str(user_id), int(limit)),
                ).fetchall()

        sessions: List[Dict[str, Any]] = []
        for row in rows:
            session = dict(row)
            raw = str(session.pop("_preview_raw", "") or "").strip()
            session["preview"] = raw[:60] + ("..." if len(raw) > 60 else "") if raw else ""
            sessions.append(session)
        return sessions

    # ── Space reclamation ──

    # FTS5 virtual tables whose b-tree segments we merge on optimize. The
    # trigram table is created lazily / may be disabled, and the cjk-bigram
    # table only exists (and is only queryable) when the loadable tokenizer
    # is present — so we probe each before touching it (see optimize_fts).
    _FTS_TABLES = ("messages_fts", "messages_fts_trigram", "messages_fts_cjk")

    def _fts_table_exists(self, name: str) -> bool:
        """True if an FTS5 virtual table is queryable in this DB."""
        try:
            self._conn.execute(f"SELECT 1 FROM {name} LIMIT 0")
            return True
        except sqlite3.DatabaseError:
            # OperationalError ("no such table") or the broader
            # DatabaseError class ("vtable constructor failed", raised when
            # e.g. a required tokenizer is missing or the table is mid-
            # teardown) — in every case the table is not queryable.
            return False

    def optimize_fts(self) -> int:
        """Merge fragmented FTS5 b-tree segments into one per index.

        FTS5 indexes grow as a series of incremental segments — one per
        ``INSERT`` batch driven by the message triggers. Over tens of
        thousands of messages these segments accumulate, which both bloats
        the ``*_data`` shadow tables and slows ``MATCH`` queries that must
        scan every segment. The special ``'optimize'`` command rewrites each
        index as a single merged segment.

        This is purely a maintenance operation — it changes neither search
        results nor ``snippet()`` output, only on-disk layout and query
        speed. It is complementary to VACUUM: ``optimize`` compacts the FTS
        index internally, then VACUUM returns the freed pages to the OS.

        Skips any FTS table that does not exist (e.g. the trigram index when
        disabled via ``HERMES_DISABLE_FTS_TRIGRAM`` or not yet created), so
        it is safe to call unconditionally.

        Returns the number of FTS indexes that were optimized.
        """
        optimized = 0
        with self._lock:
            for tbl in self._FTS_TABLES:
                if not self._fts_table_exists(tbl):
                    continue
                try:
                    # The column name in the INSERT must match the table name
                    # for FTS5 special commands.
                    self._conn.execute(
                        f"INSERT INTO {tbl}({tbl}) VALUES('optimize')"
                    )
                    optimized += 1
                except sqlite3.OperationalError as exc:
                    logger.warning(
                        "FTS optimize failed for %s: %s", tbl, exc
                    )
        return optimized

    def rebuild_fts(self) -> int:
        """Rebuild FTS5 indexes from the canonical ``messages`` table.

        Uses the FTS5 ``'rebuild'`` command, which rewrites the internal
        b-tree segments from the content rows. This is the documented
        recovery for a corrupt FTS index that rejects message writes while
        reads still succeed (issue #50502). Unlike ``optimize_fts`` (which
        merges existing segments), ``rebuild`` discards and recreates the
        index data entirely.

        Safe to call when FTS tables don't exist (skips them).
        Returns the number of FTS indexes that were rebuilt.
        """
        rebuilt = 0
        with self._lock:
            for tbl in self._FTS_TABLES:
                if not self._fts_table_exists(tbl):
                    continue
                try:
                    self._conn.execute(
                        f"INSERT INTO {tbl}({tbl}) VALUES('rebuild')"
                    )
                    self._conn.commit()
                    rebuilt += 1
                except sqlite3.OperationalError as exc:
                    self._conn.rollback()
                    logger.warning(
                        "FTS rebuild failed for %s: %s", tbl, exc
                    )
        return rebuilt

    def vacuum(self) -> int:
        """Run VACUUM to reclaim disk space after large deletes.

        SQLite does not shrink the database file when rows are deleted —
        freed pages just get reused on the next insert. After a prune that
        removed hundreds of sessions, the file stays bloated unless we
        explicitly VACUUM.

        VACUUM rewrites the entire DB, so it's expensive (seconds per
        100MB) and cannot run inside a transaction. It also acquires an
        exclusive lock, so callers must ensure no other writers are
        active. Safe to call at startup before the gateway/CLI starts
        serving traffic.

        FTS5 segments are merged first via :meth:`optimize_fts` so the
        subsequent VACUUM reclaims the pages freed by the merge. This is a
        layout-only optimization — search results are unchanged.

        Returns the number of FTS indexes that were optimized (0 if the
        merge step failed or no FTS tables exist).
        """
        # Merge FTS5 segments before VACUUM so the freed pages are returned
        # to the OS in the same pass. optimize_fts() manages its own lock.
        optimized = 0
        try:
            optimized = self.optimize_fts()
        except Exception as exc:
            logger.warning("FTS optimize before VACUUM failed: %s", exc)
        # VACUUM cannot be executed inside a transaction.
        with self._lock:
            # Best-effort WAL checkpoint first, then VACUUM.
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception as exc:
                logger.debug("WAL checkpoint (TRUNCATE) before VACUUM failed: %s", exc)
            self._conn.execute("VACUUM")
        return optimized

    def maybe_auto_prune_and_vacuum(
        self,
        retention_days: int = 90,
        min_interval_hours: int = 24,
        vacuum: bool = True,
        sessions_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Idempotent auto-maintenance: prune old sessions + optional VACUUM.

        Records the last run timestamp in state_meta so subsequent calls
        within ``min_interval_hours`` no-op. Designed to be called once at
        startup from long-lived entrypoints (CLI, gateway, cron scheduler).

        When *sessions_dir* is provided, on-disk transcript files
        (``.json`` / ``.jsonl`` / ``request_dump_*``) for pruned sessions
        are removed as part of the same sweep (issue #3015).

        Never raises. On any failure, logs a warning and returns a dict
        with ``"error"`` set.

        Returns a dict with keys:
          - ``"skipped"`` (bool) — true if within min_interval_hours of last run
          - ``"pruned"`` (int)   — number of sessions deleted
          - ``"vacuumed"`` (bool) — true if VACUUM ran
          - ``"error"`` (str, optional) — present only on failure
        """
        result: Dict[str, Any] = {"skipped": False, "pruned": 0, "vacuumed": False}
        try:
            # Skip if another process/call did maintenance recently.
            last_raw = self.get_meta("last_auto_prune")
            now = time.time()
            if last_raw:
                try:
                    last_ts = float(last_raw)
                    if now - last_ts < min_interval_hours * 3600:
                        result["skipped"] = True
                        return result
                except (TypeError, ValueError):
                    pass  # corrupt meta; treat as no prior run

            pruned = self.prune_sessions(
                older_than_days=retention_days,
                sessions_dir=sessions_dir,
            )
            result["pruned"] = pruned

            # Only VACUUM if we actually freed rows — VACUUM on a tight DB
            # is wasted I/O. Threshold keeps small DBs from paying the cost.
            if vacuum and pruned > 0:
                try:
                    self.vacuum()
                    result["vacuumed"] = True
                except Exception as exc:
                    logger.warning("state.db VACUUM failed: %s", exc)

            # Record the attempt even if pruned == 0, so we don't retry
            # every startup within the min_interval_hours window.
            self.set_meta("last_auto_prune", str(now))

            if pruned > 0:
                logger.info(
                    "state.db auto-maintenance: pruned %d session(s) older than %d days%s",
                    pruned,
                    retention_days,
                    " + VACUUM" if result["vacuumed"] else "",
                )
        except Exception as exc:
            # Maintenance must never block startup. Log and return error marker.
            logger.warning("state.db auto-maintenance failed: %s", exc)
            result["error"] = str(exc)

        return result

    def maybe_auto_archive(
        self,
        idle_days: float = 3,
        min_interval_hours: int = 24,
        exclude_pinned: bool = True,
    ) -> Dict[str, Any]:
        """Idempotent auto-archive: soft-hide sessions idle for ``idle_days``.

        Sibling of :meth:`maybe_auto_prune_and_vacuum` but non-destructive —
        it archives (hides) rather than deletes, and ages on last activity
        (see :meth:`archive_stale_sessions`) rather than creation. Records the
        last run in ``state_meta['last_auto_archive']`` so calls within
        ``min_interval_hours`` no-op; safe to call opportunistically (startup
        hooks, or when the Desktop backend lists sessions).

        Never raises. Returns a dict with:
          - ``"skipped"`` (bool) — within min_interval_hours of last run
          - ``"archived"`` (int) — sessions archived this run
          - ``"error"`` (str, optional) — present only on failure
        """
        result: Dict[str, Any] = {"skipped": False, "archived": 0}
        try:
            last_raw = self.get_meta("last_auto_archive")
            now = time.time()
            if last_raw:
                try:
                    if now - float(last_raw) < min_interval_hours * 3600:
                        result["skipped"] = True
                        return result
                except (TypeError, ValueError):
                    pass  # corrupt meta; treat as no prior run

            archived = self.archive_stale_sessions(
                idle_days, exclude_pinned=exclude_pinned
            )
            result["archived"] = archived

            # Record even a zero-archive run so we don't re-sweep every call
            # within the interval window.
            self.set_meta("last_auto_archive", str(now))

            if archived > 0:
                logger.info(
                    "state.db auto-archive: archived %d session(s) idle >= %s days",
                    archived,
                    idle_days,
                )
        except Exception as exc:
            logger.warning("state.db auto-archive failed: %s", exc)
            result["error"] = str(exc)

        return result

    # ── Handoff (cross-platform session transfer) ──────────────────────────
    #
    # State machine:
    #   None       — no handoff in flight
    #   "pending"  — CLI requested handoff, gateway hasn't picked it up yet
    #   "running"  — gateway is processing (session switch + synthetic turn)
    #   "completed"— gateway successfully delivered the synthetic turn
    #   "failed"   — gateway hit an error; reason in handoff_error
    #
    # The CLI writes "pending" then poll-waits for terminal state. The gateway
    # watcher transitions pending→running→{completed,failed}.

    def request_handoff(self, session_id: str, platform: str) -> bool:
        """Mark a session as pending handoff to the given platform.

        Returns True if the row was found and not already in flight; False if
        the session is already in a non-terminal handoff state.
        """
        def _do(conn):
            cur = conn.execute(
                "UPDATE sessions "
                "SET handoff_state = 'pending', "
                "    handoff_platform = ?, "
                "    handoff_error = NULL "
                "WHERE id = ? AND (handoff_state IS NULL "
                "                  OR handoff_state IN ('completed', 'failed'))",
                (platform, session_id),
            )
            return cur.rowcount > 0
        return self._execute_write(_do)

    def get_handoff_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Read the current handoff state for a session.

        Returns ``{"state", "platform", "error"}`` or None if the session has
        no handoff record.
        """
        try:
            cur = self._conn.execute(
                "SELECT handoff_state, handoff_platform, handoff_error "
                "FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "state": row["handoff_state"],
                "platform": row["handoff_platform"],
                "error": row["handoff_error"],
            }
        except Exception:
            return None

    def list_pending_handoffs(self) -> List[Dict[str, Any]]:
        """Return all sessions in handoff_state='pending', oldest first.

        Used by the gateway's handoff watcher.
        """
        try:
            cur = self._conn.execute(
                "SELECT * FROM sessions "
                "WHERE handoff_state = 'pending' "
                "ORDER BY started_at ASC"
            )
            return [dict(r) for r in cur.fetchall()]
        except Exception:
            return []

    def claim_handoff(self, session_id: str) -> bool:
        """Atomically transition pending → running. Returns True if claimed."""
        def _do(conn):
            cur = conn.execute(
                "UPDATE sessions SET handoff_state = 'running' "
                "WHERE id = ? AND handoff_state = 'pending'",
                (session_id,),
            )
            return cur.rowcount > 0
        return self._execute_write(_do)

    def complete_handoff(self, session_id: str) -> None:
        """Mark a handoff as completed."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET handoff_state = 'completed', "
                "handoff_error = NULL WHERE id = ?",
                (session_id,),
            )
        self._execute_write(_do)

    def fail_handoff(self, session_id: str, error: str) -> None:
        """Mark a handoff as failed and record the reason."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET handoff_state = 'failed', "
                "handoff_error = ? WHERE id = ?",
                (error[:500], session_id),
            )
        self._execute_write(_do)


class AsyncSessionDB:
    """Async door onto SessionDB: offloads each call via asyncio.to_thread so a blocking SQLite call never freezes the event loop. Generic forwarder — the audit confirms no method returns a live cursor/generator."""

    def __init__(self, db: "SessionDB") -> None:
        self._db = db

    def __getattr__(self, name: str):
        attr = getattr(self._db, name)
        if not callable(attr):
            return attr

        async def _offloaded(*args, **kwargs):
            return await asyncio.to_thread(attr, *args, **kwargs)

        return _offloaded
