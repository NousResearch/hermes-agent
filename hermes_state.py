#!/usr/bin/env python3
"""SQLite state store for Hermes Agent: session metadata, message history, model
config, FTS5 search. WAL mode (concurrent readers + one writer); compression
splits sessions via parent_session_id chains; sessions are source-tagged
('cli', 'telegram', ...). Batch-runner / RL trajectories live elsewhere.
"""

import asyncio
import atexit
import hashlib
import json
import logging
import os
import queue
import random
import re
import sqlite3
import sys
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path

from agent.message_sanitization import _sanitize_surrogates
from agent.memory_manager import sanitize_context  # KENSEI CUSTOM (restored)
from agent.context_compressor import _DB_PERSISTED_MARKER as _DB_PERSISTED_MARKER_KEY  # KENSEI CUSTOM (restored)
from agent.session_activity import ActivityProvenance  # KENSEI CUSTOM (restored top-level import)
from hermes_constants import get_hermes_home
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, TypeVar, cast

from hermes_state_common import (  # KENSEI CUSTOM: restored recoverable-end-reasons imports
    escape_like as _escape_like,
    stat_db_file_identity as _stat_db_file_identity,
    _RECOVERABLE_END_REASONS,
    _RECOVERABLE_END_REASONS_SQL,
    _RESET_END_REASONS,
    _sql_session_last_active,
    _sql_session_last_active_by_id)

# ── KENSEI CUSTOM (restored): module-level constants that fork code merged into this file
# references as bare names. Upstream keeps them class-level or in sibling modules; the merged
# tree has both layouts, so re-export them at module level for method-body lookups.
from agent.context_compressor import _DB_PERSISTED_MARKER  # noqa: F401
from hermes_state_errors import (  # noqa: F401  KENSEI CUSTOM: restored
    CompressionSessionBusyError, CompressionSessionClosedError, SessionTurnLeaseLostError,
)
from hermes_state_common import (  # noqa: F401
    is_automatic_end_reason,
    FTS_CJK_STALE_KEY, FTS_STALE_KEY, _legacy_reset_child_sql, _shape_preview,
    _COMPRESSION_CHILD_SQL,
    _FTS_CJK_TRIGGERS,
    _FTS_TRIGGERS,
    _LISTABLE_CHILD_SQL,
    _PREVIEW_ELIGIBLE_SQL,
    _PREVIEW_RAW_SELECT,
    _RESET_CHILD_SQL,
    _RESET_END_REASONS_SQL,
)
from hermes_state_sessions import (  # noqa: F401
    _SAME_KEY_NAMESPACE_SQL, _delegate_from_json, _MODEL_CONFIG_ROW_MISSING,
    _cwd_prefix_clause, _workspace_key_clause, _delete_delegate_children,
    _collect_delegate_child_ids, classify_session_status,
)
try:
    from hermes_state_sessions import _MODEL_CONFIG_ROW_MISSING  # noqa: F401
except ImportError:
    pass

_WRITE_PATIENCE_S, _TRANSCRIPT_WRITE_PATIENCE_S, _ACTIVITY_WRITE_PATIENCE_S = 20.0, 60.0, 0.5
_WRITE_RETRY_MIN_S, _WRITE_RETRY_MAX_S = 0.020, 0.150
_WRITE_RETRY_SLOW_AFTER_S = 2.0
_WRITE_RETRY_SLOW_MIN_S, _WRITE_RETRY_SLOW_MAX_S = 0.250, 1.000
_SLOW_AFTER_S = _WRITE_RETRY_SLOW_AFTER_S
_FTS_MERGE_EVERY_N_WRITES, _FTS_MERGE_MAX_PAGES_PER_INDEX, _FTS_MERGE_COMMANDS_PER_PASS = 1000, 500, 4
_COMMANDS_PER_PASS = _FTS_MERGE_COMMANDS_PER_PASS
_IMPORT_MAX_SESSIONS, _IMPORT_MAX_MESSAGES_PER_SESSION, _IMPORT_MAX_TOTAL_MESSAGES = 500, 10_000, 50_000
_IMPORT_MAX_SESSION_BYTES, _IMPORT_MAX_TOTAL_BYTES = 5 * 1024 * 1024, 25 * 1024 * 1024
_READ_ONLY_IOERR_RETRY_ATTEMPTS = 3
_READ_ONLY_IOERR_RETRY_BACKOFF_S = 0.05
# ── END KENSEI CUSTOM ──

from hermes_state_errors import (
    _DELETED_WAL_GENERATION_MSG, _DISK_IO_ERROR_MARKER, _STATE_DB_CORRUPT_MSG, _STATE_DB_GENERATION_KEY,
    _STATE_DB_REPLACED_MSG, DeletedWalGenerationError, SessionCompressionInProgressError, StateDbCorruptError,
    StateDbReplacedError, _is_no_more_rows, classify_persistence_error, is_malformed_db_error,
    is_malformed_schema_error,
)
from hermes_state_guard import (
    _STATE_DB_GUARD_BYPASS_ENV, _in_test_context, _is_production_state_db, _real_platform_state_root,
    _set_last_init_error, get_last_init_error,
)
from hermes_state_readpool import _READ_POOL_MAX, _proc_fd_targets, _read_budget_for
from hermes_state_sessions import SessionSessionsMixin
from hermes_state_fts import SessionFtsSetupMixin, load_fts5_cjk_extension
from hermes_state_portability import SessionPortabilityMixin
from hermes_state_telegram import SessionTelegramTopicsMixin
from hermes_state_schema import SessionSchemaMixin
import hermes_state_holders as _state_holders
from hermes_state_dbfile import (
    _canonical_sqlite_path, _connect_tracked_db, _read_sqlite_application_id, _stat_sqlite_sidecar_identity,
    _watched_sqlite_sidecar_paths, is_zeroed_state_db, quarantine_cross_process_lock, quarantine_zeroed_state_db,
    refuse_deleted_wal_generation,
)
from hermes_state_messages import SessionMessagesMixin
from hermes_state_wal import _WAL_INCOMPAT_MARKERS, apply_database_pragmas, apply_wal_with_fallback
from hermes_state_repair import _claim_repair_attempt, preflight_db_writability, repair_state_db_schema
from hermes_state_titles import SessionTitlesMixin
from hermes_state_usage import SessionUsageMixin
from hermes_state_maintenance import SessionMaintenanceMixin
from hermes_state_gateway import SessionGatewayMixin
from hermes_state_compression import SessionCompressionMixin
from hermes_state_search import SessionSearchMixin

try:  # Hard dependency, but tolerate scaffold-phase imports before pip install.
    import psutil
except ImportError:  # pragma: no cover - stripped/scaffold installs only
    psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_MAX_SAFE_MESSAGES = 20_000  # resume/export guard default


def _configured_transcript_limit(key: str, fallback: int = _MAX_SAFE_MESSAGES) -> int:
    """``sessions.<key>`` from config.yaml (lazy import: circular at load), else *fallback*; 0 disables."""
    try:
        from hermes_cli.config import load_config_readonly
        value = (load_config_readonly().get("sessions") or {}).get(key)
        if value is None:
            return fallback
        limit = int(value)
        return limit if limit >= 0 else fallback
    except Exception:
        return fallback


def resolved_max_resume_messages() -> int:
    return _configured_transcript_limit("max_resume_messages")


def resolved_max_export_messages() -> int:
    return _configured_transcript_limit("max_export_messages")


class SessionResumeTooLargeError(ValueError):
    def __init__(
        self, message_count: int, limit: int = _MAX_SAFE_MESSAGES, scope: str = "across its lineage",
    ):
        self.message_count, self.limit = message_count, limit
        super().__init__(
            f"session has at least {message_count} active messages {scope}; "
            f"safe resume limit is {limit}. Export the session instead, or set "
            "sessions.max_resume_messages: 0 in config.yaml to disable the guard."
        )


class SessionExportTooLargeError(ValueError):
    def __init__(self, session_id: str, message_count: int, limit: int = _MAX_SAFE_MESSAGES):
        self.session_id, self.message_count, self.limit = session_id, message_count, limit
        super().__init__(
            f"session '{session_id}' has at least {message_count} active messages; "
            f"safe in-memory export limit is {limit}"
        )


def _compression_lock_holder_process_is_dead(holder: str) -> bool:
    """True only when a ``pid=<n>`` lock holder's local PID is provably gone.
    Reclaim on kernel proof only: unstructured/same-process holders (another
    thread's live lease) and any probe doubt keep the lease until TTL expiry
    (PID reuse must never steal a live lease; a wrongly-kept one self-heals)."""
    match = re.search(r"(?:^|:)pid=(\d+)(?::|$)", holder or "")
    pid = int(match.group(1)) if match else 0
    if pid <= 0 or pid == os.getpid():
        return False
    if psutil is not None:
        try:
            return not psutil.pid_exists(pid)  # recycled PIDs read as alive (conservative)
        except Exception:
            return False
    # psutil-less fallback is POSIX-only: on Windows os.kill(pid, 0) maps sig=0 to
    # CTRL_C_EVENT and can kill the target's console group.
    if os.name == "nt":
        return False
    try:
        os.kill(pid, 0)  # windows-footgun: ok — nt early-returns just above
    except ProcessLookupError:
        return True
    except (OSError, OverflowError):  # PermissionError is an OSError: alive but foreign
        return False
    return False


def _scrub_surrogates(value: Any) -> Any:
    """Replace lone surrogates in text (sqlite3 raises UnicodeEncodeError, aborting the whole write)."""
    return _sanitize_surrogates(value) if isinstance(value, str) else value


# Billing buckets that aren't a routable provider identity: a session that persisted only
# one of these (never ran /model) falls back to the config default. Shared by
# session_gateway_runtime and tui_gateway.server so they cannot drift.
_BARE_BILLING_PROVIDERS = frozenset({"auto", "custom"})

T = TypeVar("T")

# Import-time snapshot lets _default_db_path() detect a re-pointed DEFAULT_DB_PATH
# (tests monkeypatch the constant directly).
DEFAULT_DB_PATH = _IMPORT_DEFAULT_DB_PATH = get_hermes_home() / "state.db"

# Back off from read-only opens after one fails: not per query, but short enough that
# transient fd pressure doesn't strand the read pool.
_READ_OPEN_RETRY_SECONDS = 60.0
# Transient SQLITE_IOERR retry budget for READ-ONLY opens (#100436): a WAL writer's checkpoint/
# reset/frame flush surfaces "disk I/O error" to a concurrent mode=ro reader for a millisecond-
# wide window — the ro connection cannot perform WAL recovery because recovery writes the -shm
# index, which mode=ro refuses. The writer closes the window on its own, so a few short retries
# make the open succeed instead of 500-ing the whole /api/sessions poll (or any other ro opener).
# Deliberately NOT for writable opens: a writer owns the transition, so an IOERR there is a real
# storage/fd problem. A persistent IOERR still exhausts the budget and propagates.
_READ_ONLY_IOERR_RETRY_ATTEMPTS, _READ_ONLY_IOERR_RETRY_BACKOFF_S = 3, 0.05


def _default_db_path() -> Path:
    """Default state DB path at CALL time: a re-pointed ``DEFAULT_DB_PATH`` wins, else
    ``get_hermes_home()`` is resolved fresh (a runtime HERMES_HOME redirect works regardless of import)."""
    return DEFAULT_DB_PATH if DEFAULT_DB_PATH != _IMPORT_DEFAULT_DB_PATH else get_hermes_home() / "state.db"


# Live-DB guard knobs live HERE (not in hermes_state_guard): the hermetic conftest monkeypatches
# ``hermes_state._STATE_DB_GUARD_BYPASS`` (``@pytest.mark.live_system_guard_bypass`` escape hatch)
# and ``_EXTRA_DENY_ROOTS`` (the pre-sandbox root, so custom-HERMES_HOME deployments are covered).
_STATE_DB_GUARD_BYPASS = False
_STATE_DB_GUARD_EXTRA_DENY_ROOTS: Tuple[Path, ...] = ()


def _ensure_test_isolation(db_path: Path) -> None:
    """Raise before any connection/mkdir/pragma/byte probe when a pytest-context process
    (env OR ancestry) resolves a production DB.

    Env alone is not enough: a child spawned with a rebuilt environment loses ``PYTEST_*`` and
    ``HERMES_HOME`` together, which is precisely the state in which it writes to production (#82770).
    """
    if _STATE_DB_GUARD_BYPASS or os.environ.get(_STATE_DB_GUARD_BYPASS_ENV) or not _in_test_context():
        return
    try:
        resolved = Path(db_path).expanduser().resolve()
    except Exception:
        return
    roots = [r for r in (_real_platform_state_root(),) if r is not None]
    for extra in _STATE_DB_GUARD_EXTRA_DENY_ROOTS:
        try:
            roots.append(Path(extra).expanduser().resolve())
        except Exception:
            continue
    for root in roots:
        if _is_production_state_db(resolved, root):
            raise RuntimeError(
                "live-system guard: test attempted to open production "
                f"state.db at {resolved} (under real Hermes root {root}). "
                "Tests must run against a temporary HERMES_HOME — pass an "
                "explicit tmp db_path or let the hermetic conftest redirect "
                "HERMES_HOME. If this test genuinely needs the live database, mark it with "
                "@pytest.mark.live_system_guard_bypass — or, for a spawned "
                f"child process, export {_STATE_DB_GUARD_BYPASS_ENV}=1 in "
                "its environment."
            )


# Openings of the background-review harness prompts (agent/background_review.py).
_REVIEW_HARNESS_PREFIXES = (
    "Review the conversation above and update the skill library",
    "Review the conversation above and consider saving to memory",
)


def _is_background_review_harness_message(msg: Dict[str, Any]) -> bool:
    """Persisted harness prompt (older builds wrote the forked curator's turns
    into real sessions; replaying them hijacks the session)."""
    if not isinstance(msg, dict) or msg.get("role") not in {"user", "system"}:
        return False
    content = msg.get("content")
    return isinstance(content, str) and content.lstrip().startswith(_REVIEW_HARNESS_PREFIXES)


def _strip_background_review_harness(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop harness messages and the curator-mode assistant reply that immediately followed each."""
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
                continue  # the curator-mode reply to the harness prompt
        out.append(msg)
    return out


# Matches a bare protocol/tool-name marker such as "[memory]" or "[skill_manage]".
_STALE_TOOL_CALL_MARKER_RE = re.compile(r"^\[[A-Za-z_][A-Za-z0-9_.-]*\]$")


def _is_stale_tool_call_marker_message(msg: Dict[str, Any]) -> bool:
    """Assistant tool-call turn whose content is a bare ``[marker]`` (an older
    conversation_loop persisted a local template's marker as the final response)."""
    if not isinstance(msg, dict) or msg.get("role") != "assistant" or not msg.get("tool_calls"):
        return False
    content = msg.get("content")
    return isinstance(content, str) and bool(_STALE_TOOL_CALL_MARKER_RE.fullmatch(content.strip()))


def _strip_stale_tool_call_markers(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Blank stale ``[marker]`` assistant content (replaying it teaches the model
    to keep emitting it); tool_call/result pairing stays intact."""
    repaired = 0
    for msg in filter(_is_stale_tool_call_marker_message, messages):
        msg["content"] = ""
        repaired += 1
    if repaired:
        logger.info(
            "Cleared %d stale tool-call marker message(s) while restoring session (#78148)", repaired,
        )
    return messages


def format_session_db_unavailable(prefix: str = "Session database not available") -> str:
    """User-facing message with the captured init cause (+ WAL-docs hint for NFS/SMB locking failures)."""
    cause = get_last_init_error()
    if not cause:
        return f"{prefix}."
    hint = " (state.db may be on NFS/SMB/FUSE/ZFS — see https://www.sqlite.org/wal.html)"
    return f"{prefix}: {cause}{hint if any(m in cause.lower() for m in _WAL_INCOMPAT_MARKERS) else ''}."


# Auto-repair at most once per DB path per process (no repair loops; serialises concurrent
# web_server / gateway opens on the same malformed file).
_repair_attempted_paths: set[str] = set()
_repair_attempt_lock = threading.Lock()
# Cross-process schema-surgery lock timeout (``_repair_attempt_lock`` covers one interpreter
# only); sized for the slowest legitimate holder (VACUUM, multi-GB DB).
_REPAIR_LOCK_TIMEOUT_SECONDS = 120.0
_IS_WINDOWS = sys.platform == "win32"


def divert_session_transcript_jsonl(session_id: str, messages) -> "Optional[Path]":
    """Append pending messages to HERMES_HOME/sessions/<id>.jsonl (state.db was replaced under a
    live process). Returns the path, or None if nothing to write."""
    sid = str(session_id or "").strip()
    if not sid or not messages:
        return None
    sessions_dir = get_hermes_home() / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"{sid}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        for msg in messages:
            if msg is not None:
                record = msg if isinstance(msg, dict) else {"content": str(msg)}
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    return path


# Process-wide shared SessionDB registry: long-lived in-process callers share ONE writer
# connection per resolved path via hermes_state_registry.acquire(); one-shots use SessionDB() + close().
def _foreign_state_db_holders(db_path: Path) -> List[Tuple[int, str]]:
    """Compatibility delegate to the state-holder authority."""
    return _state_holders.foreign_state_db_holders(db_path)


# ── Process-wide shared SessionDB registry (#90837) ── lives in hermes_state_registry.py (acquire /
# release / close_all / release_or_close). Long-lived in-process callers (gateway, tui_gateway, cron,
# in-process tools) share ONE writer connection per resolved path via hermes_state_registry.acquire(); CLI
# one-shots, recovery flows, and read-only cross-profile opens use SessionDB() directly with their own close().


class SessionDB(
    SessionSessionsMixin, SessionFtsSetupMixin, SessionSearchMixin, SessionSchemaMixin,
    SessionPortabilityMixin, SessionTelegramTopicsMixin, SessionCompressionMixin,
    SessionGatewayMixin, SessionMaintenanceMixin, SessionUsageMixin, SessionTitlesMixin,
    SessionMessagesMixin,
):
    """SQLite-backed session storage with FTS5 search; many reader threads, one writer (WAL)."""

    # Only these state-owned producers join automatic stale-open reconciliation; messaging/UI
    # sources have their own lifecycle owners; unknown sources fail closed.
    # See #60609.
    _AUTO_PRUNE_STALE_OPEN_SOURCES: Tuple[str, ...] = (
        "cli", "cron", "kanban", "acp", "api_server", "subagent", "tool",
    )

    # ── Write-contention tuning ──
    # SQLite's deterministic busy handler convoys under many hermes processes: keep its
    # timeout short (1s) and retry with random jitter. Patience is TIME-based (a sibling
    # legitimately holds the lock for seconds: checkpoint at close, VACUUM, recovery, FTS
    # optimize); attempt-counted budgets destroyed turns on a healthy store. Transcript
    # writes (failure aborts the turn) get the long budget; observation-only activity
    # writes sit on the response-critical path and get a sub-second one.
    _WRITE_PATIENCE_S, _TRANSCRIPT_WRITE_PATIENCE_S, _ACTIVITY_WRITE_PATIENCE_S = 20.0, 60.0, 0.5
    # A live compression lock gets a short wait (compression publishes in seconds), but the lease
    # is a correctness boundary: a writer still locked out afterwards is refused.
    # Observation-only activity heartbeat/label writes (#76354 review S1): these run on (or adjacent to) the
    # response-critical path and must never wait out the full routine patience under contention. Sub-second
    # budget; a skipped write is retried naturally at the next heartbeat window.
    # A live compression lock gets its own, much shorter budget than the write lock. Compression publishes
    # in a couple of seconds, so a brief wait saves the overwhelming majority of concurrent turns (#75083).
    # It deliberately stays short: the lease is a correctness boundary, not just a busy signal (see
    # test_compression_lease_blocks_non_owner_but_allows_owner_flush), so a writer that is still locked out
    # after this budget must still be refused rather than allowed to land a stale turn in a session whose
    # compression is genuinely long-running or wedged.
    _COMPRESSION_BUSY_WAIT_S = 5.0
    _WRITE_RETRY_MIN_S, _WRITE_RETRY_MAX_S = 0.020, 0.150  # fast jitter for the first _SLOW_AFTER_S
    _WRITE_RETRY_SLOW_AFTER_S = 2.0
    _WRITE_RETRY_SLOW_MIN_S, _WRITE_RETRY_SLOW_MAX_S = 0.250, 1.000
    # PASSIVE WAL checkpoint every N successful writes.
    _CHECKPOINT_EVERY_N_WRITES = 50
    # Bounded FTS ``'merge'`` (ms of lock each) instead of ``'optimize'`` (9-18s per index on a 10GB
    # DB, longer than a writer's patience); up to _COMMANDS_PER_PASS per index, stopping on no-progress.
    _FTS_MERGE_EVERY_N_WRITES, _FTS_MERGE_MAX_PAGES_PER_INDEX, _FTS_MERGE_COMMANDS_PER_PASS = 1000, 500, 4
    # Imports cap lower than exports: an import holds one BEGIN IMMEDIATE.
    _IMPORT_MAX_SESSIONS, _IMPORT_MAX_MESSAGES_PER_SESSION, _IMPORT_MAX_TOTAL_MESSAGES = 500, 10_000, 50_000
    _IMPORT_MAX_SESSION_BYTES, _IMPORT_MAX_TOTAL_BYTES = 5 * 1024 * 1024, 25 * 1024 * 1024
    # Accounting workers retire when idle so a bound-method target can't keep an abandoned SessionDB alive.
    _TOKEN_WRITER_IDLE_SECONDS = 30.0

    @staticmethod
    def _store_system_prompt(conn, system_prompt: Optional[str]) -> Optional[str]:
        if system_prompt is None:
            return None
        prompt_hash = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT OR IGNORE INTO system_prompts (hash, prompt) VALUES (?, ?)",
            (prompt_hash, system_prompt),
        )
        return prompt_hash

    @staticmethod
    def _delete_unreferenced_system_prompts(conn) -> None:
        conn.execute(
            "DELETE FROM system_prompts WHERE NOT EXISTS ("
            "SELECT 1 FROM sessions WHERE sessions.system_prompt_hash = system_prompts.hash)"
        )

    @staticmethod
    def _session_row_dict(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        if "_system_prompt_resolved" in data:
            resolved = data.pop("_system_prompt_resolved")
            if "system_prompt" in data:
                data["system_prompt"] = resolved
        return data

    @staticmethod
    def _close_connection_quietly(conn: Optional[sqlite3.Connection]) -> None:
        """Close a partially initialized connection without masking its error."""
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            logger.debug("Could not close a SessionDB connection", exc_info=True)

    def _close_conn_logged(self, conn, label: str) -> None:
        """Close *conn*; a failing close leaks a tracked fd: logged at WARNING, never swallowed."""
        try:
            conn.close()
        except Exception as exc:
            logger.warning("%s close failed for %s: %s", label, self.db_path, exc)

    def __init__(self, db_path: Path = None, read_only: bool = False):
        self.db_path = db_path or _default_db_path()
        _ensure_test_isolation(self.db_path)  # before any connection/pragma/mkdir
        self.read_only = read_only
        self._lock = threading.Lock()
        # Read-path split (WAL only): reads borrow from a BOUNDED read-only pool so they
        # never queue behind writer flushes on self._lock (see _read_ctx); unbounded
        # per-thread connections pinned fds for the process lifetime and hit EMFILE.
        self._read_pool: "queue.LifoQueue[sqlite3.Connection]" = queue.LifoQueue(maxsize=_READ_POOL_MAX)
        # Permits bound PEAK descriptors (the pool bounds only the idle set), shared per
        # DATABASE PATH; acquired non-blocking so a permitless reader degrades to the writer lock.
        # One permit per live read connection, held from before the open in _get_read_conn() until after the
        # close in _close_read_conn(). See _READ_POOL_MAX. Acquired non-blocking on purpose: a reader that
        # cannot get a permit must degrade to the writer lock, not queue here — blocking would convert fd
        # exhaustion into a stall, which is the same outage with a different stack trace. Permits are shared
        # per DATABASE PATH, not per instance: the descriptors they ration belong to the file, and one
        # process holds several SessionDB objects on the same state.db (#98573). See _PathReadBudget.
        self._read_budget = _read_budget_for(self.db_path)
        self._read_budget.register(self)
        self._read_permits = self._read_budget.permits
        self._read_conns_lock = threading.Lock()
        # Set when close() begins; an in-flight reader then closes its own connection
        # instead of re-populating a pool nobody will drain again.
        self._read_conns_closed = False
        # Read-open failure backoff is a TIMESTAMP, not a sticky bool: the likeliest trigger
        # is transient EMFILE, and a permanent flag would demote every reader forever.
        self._read_open_failed_at = 0.0
        self._wal_active, self._write_count = False, 0
        # File identity of the opened state.db, compared on every write so an out-of-band
        # replace cannot limp through in-place surgery (inode: mv/new-file; application_id: cp).
        self._db_file_identity: Optional[tuple] = None
        self._db_file_application_id: int = 0
        self._db_sidecar_identity: Dict[str, tuple] = {}
        self._db_replaced = self._db_wal_generation_lost = False
        self._db_corrupt, self._db_corrupt_reason = False, ""  # sticky quarantine (StateDbCorruptError)
        self._fts_usermerge_floor_applied = False  # one-shot usermerge-floor write guard
        self._fts_enabled = self._fts_stale = self._trigram_available = False
        # _fts_cjk_loaded: tokenizer on the writer connection; _fts_cjk_available: messages_fts_cjk
        # is queryable AND not marked stale.
        self._fts_cjk_loaded = self._fts_cjk_available = self._fts_unavailable_warned = False
        self._conn = None
        # Async token accounting; distinct from self._lock so enqueue/flush never contends with writes.
        self._token_queue: deque = deque()
        self._token_queue_cond = threading.Condition(threading.Lock())
        self._token_writer_thread: Optional[threading.Thread] = None
        self._token_writer_stop = self._token_writer_busy = False
        self._token_atexit_hook: Optional[Callable[[], None]] = None
        # Opened via hermes_state_registry.acquire(): close() releases a refcount instead.
        # Set True when this instance is opened via hermes_state_registry.acquire(). Makes close() a no-op so the
        # registry (not individual callers) controls the connection lifecycle (#90837).
        self._shared_registry_owned = False
        initialization_complete = False
        try:
            if read_only:
                self._open_read_only()
            else:
                self._open_writer()
            self._record_db_file_identity()
            initialization_complete = True
        except Exception as exc:
            # Surface WHY via /resume and friends; callers keep their ``_session_db = None`` path.
            _set_last_init_error(f"{type(exc).__name__}: {exc}")
            raise
        finally:
            if not initialization_complete:
                conn, self._conn = self._conn, None
                self._close_connection_quietly(conn)

    def _open_writer(self) -> None:
        """Writable open: preflight, zero-byte quarantine, connect + schema (one in-place repair of a
        malformed sqlite_master), generation stamp."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Read-only file/sidecar preflight BEFORE the first connection: an actionable message
        # instead of an opaque "attempt to write a readonly database" from inside _init_schema.
        preflight_db_writability(self.db_path, db_label="state.db")
        try:
            # Serialize zero-byte check, quarantine, connect and schema commit so concurrent
            # openers don't race the absent-path -> schema-commit window.
            if not self.db_path.exists() or is_zeroed_state_db(self.db_path):
                with quarantine_cross_process_lock(self.db_path) as lock_acquired:
                    if not lock_acquired:
                        logger.warning(
                            "startup quarantine lock for %s not acquired within 5s; proceeding",
                            self.db_path,
                        )
                    self._handle_quarantine_if_zeroed(already_locked=lock_acquired)
                    self._connect_and_init_with_lock_patience()
            else:
                self._handle_quarantine_if_zeroed(already_locked=False)
                self._connect_and_init_with_lock_patience()
        except sqlite3.DatabaseError as exc:
            # A malformed schema fails on the very first statement (before _init_schema), so the
            # FTS-rebuild layer never sees it: repair sqlite_master in place (backup first), reopen once.
            if not is_malformed_schema_error(exc) or not _claim_repair_attempt(self.db_path):
                raise
            logger.error(
                "state.db schema is malformed (%s) — attempting automatic "
                "repair (a backup copy is made first).", exc,
            )
            self._close_connection_quietly(self._conn)
            if not repair_state_db_schema(self.db_path).get("repaired"):
                raise
            self._connect_and_init_with_lock_patience()
        # FTS optimization is OPT-IN (`hermes db optimize`); no background worker races session lifecycle.
        self._ensure_db_file_generation()

    def _open_read_only(self) -> None:
        """Read-only attach for cross-profile aggregation: no schema init, NO write
        lock (sidebar polling never contends with that profile's backend); the DB
        must exist. FTS flags are probed with SELECTs only, and the connection is
        closed on ANY probe failure (malformed schema raises DatabaseError) so a
        leaked tracked connection cannot block the forensic backup the writable heal takes next."""
        for attempt in range(_READ_ONLY_IOERR_RETRY_ATTEMPTS + 1):
            try:
                self._conn = conn = self._connect_read_only(timeout=1.0)
                try:
                    apply_database_pragmas(conn, db_label="state.db")
                    cursor = conn.cursor()
                    self._fts_enabled = self._fts_table_probe(cursor, "messages_fts") is True
                    if self._fts_enabled:
                        self._trigram_available = (
                            self._fts_table_probe(cursor, "messages_fts_trigram") is True
                        )
                except BaseException:
                    self._conn = None
                    self._close_connection_quietly(conn)
                    raise
                return
            except sqlite3.OperationalError as ioerr:
                # In-flight WAL checkpoint/reset/frame-flush on the writer side can surface
                # SQLITE_IOERR to a mode=ro reader (it can't do the -shm recovery the read
                # needs). Closes in milliseconds: retry a bounded number of times before
                # classifying the store as failed (#100436; see _READ_ONLY_IOERR_RETRY_ATTEMPTS).
                transient = _DISK_IO_ERROR_MARKER in str(ioerr).lower()
                if attempt >= _READ_ONLY_IOERR_RETRY_ATTEMPTS or not transient:
                    raise
                time.sleep(_READ_ONLY_IOERR_RETRY_BACKOFF_S)

    def _connect_read_only(self, timeout: float) -> sqlite3.Connection:
        """``mode=ro`` tracked connection with Row factory. check_same_thread=False: pooled connections
        are borrowed by whichever thread reads next; exclusive ownership is enforced by pool checkout."""
        conn = _connect_tracked_db(
            f"file:{self.db_path}?mode=ro", tracking_path=self.db_path, uri=True,
            check_same_thread=False, timeout=timeout, isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        return conn

    def _handle_quarantine_if_zeroed(self, already_locked: bool = False) -> None:
        """Quarantine a zero-byte/headerless state.db so a fresh one can open; if quarantine failed,
        raise the clear message instead of opening the zeroed file."""
        if not (self.db_path.exists() and is_zeroed_state_db(self.db_path)):
            return
        try:
            zsize = self.db_path.stat().st_size
        except OSError:
            zsize = -1
        qpath = quarantine_zeroed_state_db(self.db_path, already_locked=already_locked)
        msg = (
            f"state.db looks ZEROED ({zsize} bytes, no SQLite header). "
            f"Preserved at {qpath or '(quarantine failed — file left in place)'}. "
            f"Restore from {self.db_path.parent / 'state-snapshots'} via `hermes snapshot list` / "
            f"`hermes snapshot restore <id>` if available. "
            "Opening a fresh empty database so the agent can start."
        )
        logger.error(msg)
        _set_last_init_error(msg)
        if qpath is None and self.db_path.exists() and is_zeroed_state_db(self.db_path):
            raise sqlite3.DatabaseError(msg)

    def _open_writer_conn(self) -> sqlite3.Connection:
        """Connect + WAL/pragma/tokenizer setup for a writer connection (no schema init). Short timeout:
        jittered application-level retry handles contention, not SQLite's busy handler;
        isolation_level=None: explicit BEGIN IMMEDIATE."""
        conn = _connect_tracked_db(
            str(self.db_path), check_same_thread=False, timeout=1.0, isolation_level=None,
        )
        try:
            conn.row_factory = sqlite3.Row
            self._wal_active = apply_wal_with_fallback(conn, db_label="state.db") == "wal"
            apply_database_pragmas(conn, db_label="state.db")
            conn.execute("PRAGMA foreign_keys=ON")
            self._fts_cjk_loaded = load_fts5_cjk_extension(conn)
        except BaseException:
            self._close_connection_quietly(conn)
            raise
        return conn

    def _connect_and_init(self) -> None:
        # Refuse before sqlite3.connect (under the startup lock) so we cannot mint
        # a replacement WAL while a live writer still holds a deleted sidecar inode.
        refuse_deleted_wal_generation(self.db_path)
        self._conn = self._open_writer_conn()
        self._init_schema()

    def _connect_and_init_with_lock_patience(self) -> None:
        """Open + init, waiting out a sibling's write lock with jittered patience:
        _init_schema's DDL runs on a 1s-timeout connection, so a sibling's VACUUM
        or checkpoint used to fail the ENTIRE open and callers disabled
        persistence for the whole run. Non-lock errors propagate immediately."""
        # Lock contention during open: _init_schema's DDL/reconcile statements run on a 1s-timeout
        # connection with no retry, so a sibling process holding the write lock (VACUUM, TRUNCATE checkpoint
        # at close, a long FTS pass from an older still-running install) used to fail the ENTIRE open —
        # callers then disable persistence for the whole run ("Failed to initialize SessionDB ... database
        # is locked", #74478). The store is healthy; wait it out with the same jittered patience the write
        # path uses.
        deadline = time.monotonic() + self._WRITE_PATIENCE_S
        while True:
            try:
                self._connect_and_init()
                return
            except sqlite3.OperationalError as exc:
                err = str(exc).lower()
                if "locked" not in err and "busy" not in err:
                    raise
                self._close_connection_quietly(self._conn)
                now = time.monotonic()
                if now >= deadline:
                    raise
                jitter = random.uniform(self._WRITE_RETRY_SLOW_MIN_S, self._WRITE_RETRY_SLOW_MAX_S)
                time.sleep(min(jitter, max(deadline - now, 0.001)))

    # ── Read-path split ──

    def _get_read_conn(self) -> Optional[sqlite3.Connection]:
        """Open a fresh read-only connection, or None when unavailable (callers
        return it to self._read_pool). WAL only: WAL readers never block on the
        writer, so reads skip self._lock; under DELETE journal mode (NFS fallback)
        readers hit SQLITE_BUSY storms, so the legacy locked path stays. Autocommit
        reads see everything committed so far (read-your-writes for flush-then-search)."""
        if not self._wal_active or self.read_only:
            return None
        with self._read_conns_lock:
            failed_at = self._read_open_failed_at
            backing_off = failed_at and time.monotonic() - failed_at < _READ_OPEN_RETRY_SECONDS
            if self._read_conns_closed or backing_off:
                return None
        # Permit BEFORE the open: openers race for permits, not descriptors.
        if not self._read_budget.acquire(self):
            logger.debug(
                "read pool at capacity (%d) for %s; serving this read from the "
                "locked writer connection", _READ_POOL_MAX, self.db_path,
            )
            return None
        conn = None  # bound before the try so the handlers can close a half-open one
        try:
            conn = self._connect_read_only(timeout=5.0)
            apply_database_pragmas(conn, db_label="state.db")
            if self._fts_cjk_loaded:  # registers in the connection, not the file: ro is fine
                load_fts5_cjk_extension(conn)
        except BaseException as exc:
            # A half-open connection (open ok, extension load failed) is a live tracked descriptor,
            # the leak shape this pool exists to fix; a stranded permit would shrink the read
            # path by one slot forever. (Not _close_read_conn: callers release their own permit.)
            if conn is not None:
                self._close_conn_logged(conn, "partially-opened read conn")
            self._read_budget.release()
            if not isinstance(exc, sqlite3.Error):
                raise
            with self._read_conns_lock:
                self._read_open_failed_at = time.monotonic()
            logger.debug("read-only connection open failed for %s", self.db_path, exc_info=True)
            return None
        return conn

    def _evict_one_idle_read_conn(self) -> bool:
        """Close one idle pooled connection (a peer on the same file wants its permit); never a live one."""
        try:
            conn = self._read_pool.get_nowait()
        except queue.Empty:
            return False
        self._close_read_conn(conn)
        return True

    def _close_read_conn(self, conn) -> None:
        """Close a pooled read connection and release its permit even when the close fails (withholding
        it would narrow the read path forever). Over-releasing the BoundedSemaphore raises ValueError."""
        try:
            self._close_conn_logged(conn, "read-conn")
        finally:
            self._read_budget.release()

    def _checkout_read_conn(self) -> Optional[sqlite3.Connection]:
        """Borrow a read connection, opening on a miss; None when the read path is unavailable.
        A pool hit costs no permit (the connection already holds one)."""
        if not self._wal_active or self.read_only:
            return None
        try:
            return self._read_pool.get_nowait()
        except queue.Empty:
            return self._get_read_conn()

    @contextmanager
    def _read_ctx(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection for read-only statements: a pooled read-only
        connection with NO lock under WAL; otherwise (non-WAL, open failure,
        ceiling reached) the writer connection under self._lock — deliberate
        degradation: slower beats EMFILE, which the supervisor cannot see."""
        conn = self._checkout_read_conn()
        if conn is not None:
            try:
                yield conn
            finally:
                returned = False
                with self._read_conns_lock:
                    if not self._read_conns_closed:
                        try:
                            self._read_pool.put_nowait(conn)
                            returned = True
                        except queue.Full:
                            pass
                if not returned:
                    # close() drained the pool (or queue.Full: unreachable while
                    # permits == maxsize, load-bearing if they drift): surplus.
                    self._close_read_conn(conn)
            return
        with self._lock:
            if self._conn is None:  # close() raced a still-unwinding reader
                self._reopen_after_close_locked(context="read")
            yield cast(sqlite3.Connection, self._conn)

    def _reopen_after_close_locked(self, context: str = "write") -> None:
        """Reopen the writer after ``close()`` raced a live caller (a teardown owner
        set ``_conn = None`` while a worker still had a transcript flush to land).
        Loud (WARNING) and bounded (only after an explicit close()). Caller holds
        ``self._lock``. No _init_schema: no DDL races with siblings during teardown."""
        if self.read_only:
            raise sqlite3.ProgrammingError(
                f"SessionDB for {self.db_path} was closed (read-only handle); "
                f"cannot serve a {context} after close()"
            )
        # A reopen resolves the PATH again: a replaced file would be written through stale WAL/shm
        # assumptions; a quarantined handle must never hand a fresh connection to a damaged file.
        if self._db_corrupt and not (self._db_replaced or self._db_file_was_replaced()):
            raise self._corrupt_error(
                f"state.db connection for {self.db_path} is quarantined after "
                f"structural corruption; refusing to reopen for a {context} "
                "after close(). "
            )
        if self._db_wal_generation_lost or self._wal_generation_was_lost():
            self._halt_deleted_wal_generation()
        logger.warning(
            "state.db connection for %s was closed while a %s was still in "
            "flight — reopening (teardown/worker race, #94736)",
            self.db_path,
            context,
        )
        try:
            conn = _connect_tracked_db(
                str(self.db_path),
                check_same_thread=False,
                timeout=1.0,
                isolation_level=None,
            )
        except Exception as exc:
            raise sqlite3.OperationalError(
                f"state.db connection was closed while a {context} was still "
                f"in flight (a session-teardown path called close() before "
                f"this worker finished — #94736) and the automatic reopen "
                f"failed: {exc}"
            ) from exc
        try:
            conn.row_factory = sqlite3.Row
            self._wal_active = (
                apply_wal_with_fallback(conn, db_label="state.db") == "wal"
            )
            apply_database_pragmas(conn, db_label="state.db")
            conn.execute("PRAGMA foreign_keys=ON")
            self._fts_cjk_loaded = load_fts5_cjk_extension(conn)
        except Exception as exc:
            self._close_connection_quietly(conn)
            raise sqlite3.OperationalError(
                f"state.db reopen after close() succeeded but connection "
                f"setup failed: {exc}"
            ) from exc
        # Schema was initialised by this instance's original open; the file
        # cannot have lost it, so no _init_schema here (no DDL races with
        # sibling processes during teardown).
        self._conn = conn

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

    @staticmethod
    def _db_has_trigram_tool_calls_projection(cursor: sqlite3.Cursor) -> bool:
        """True when the trigram vtable still includes tool_calls payload."""
        row = cursor.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'messages_fts_trigram'"
        ).fetchone()
        if row is None:
            return False
        sql = (row[0] if not isinstance(row, sqlite3.Row) else row["sql"]) or ""
        return "tool_calls" in sql.lower()

    @classmethod
    def _db_needs_fts_storage_upgrade(
        cls, cursor: sqlite3.Cursor
    ) -> bool:
        """True when the current FTS storage layout should be treated as stale."""
        return (
            cls._db_has_legacy_inline_fts(cursor)
            or cls._db_has_trigram_tool_calls_projection(cursor)
        )

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
        try:
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
        except sqlite3.OperationalError:
            # Mirror the tokenizer-loaded except below: the presence check
            # and self-heal above run before the loaded/not-loaded branch is
            # even known to be safe, so they need the same never-raises
            # guarantee the docstring promises for the rest of the method.
            logger.warning(
                "messages_fts_cjk presence check failed; CJK search stays on "
                "trigram/LIKE", exc_info=True,
            )
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
        patience_s: Optional[float] = None,
    ) -> T:
        """Execute a write transaction with BEGIN IMMEDIATE and jitter retry.

        *fn* receives the connection and should perform INSERT/UPDATE/DELETE
        statements.  The caller must NOT call ``commit()`` — that's handled
        here after *fn* returns.

        BEGIN IMMEDIATE acquires the WAL write lock at transaction start
        (not at commit time), so lock contention surfaces immediately.
        On ``database is locked``, we release the Python lock, sleep a
        random jitter, and retry — breaking the convoy pattern that
        SQLite's built-in deterministic backoff creates.

        *patience_s* is the total time budget for lock retries (default
        ``_WRITE_PATIENCE_S``).  Transcript-critical writes pass
        ``_TRANSCRIPT_WRITE_PATIENCE_S`` so a sibling process holding the
        lock for a legitimate long operation (VACUUM, TRUNCATE checkpoint,
        pre-bounded-merge FTS optimize from an older still-running
        install) exhausts routine writers' patience without destroying a
        user turn.  Jitter starts small (20-150ms) for fast reclaim on
        millisecond contention and backs off to 250ms-1s once the lock has
        been held longer than ``_WRITE_RETRY_SLOW_AFTER_S``.

        Returns whatever *fn* returns.
        """
        if patience_s is None:
            patience_s = self._WRITE_PATIENCE_S
        deadline = time.monotonic() + patience_s
        # Set on the first compression-busy collision so the short wait is
        # measured from then, not from the start of the write.
        compression_deadline: Optional[float] = None
        # One retry for SQLITE_IOERR raised by BEGIN IMMEDIATE itself. The
        # callback has not run at that point, so there is no durable effect
        # to replay and the retry is exactly-once safe (#99502's contract).
        # Once the callback starts, an IOERR leaves the write's settlement
        # unknown and must propagate — this helper owns non-idempotent
        # transcript/counter mutations, not just idempotent UPSERTs.
        ioerr_begin_retried = False

        # Transient engine-level error observed on contended WAL appends
        # (dual gateway/agent writers; FTS5 trigram sync holds the write
        # lock). The identical write succeeds standalone, so it is
        # retryable like locked/busy. The exception CLASS varies with the
        # SQLite build — some surface it as InterfaceError, which lives
        # OUTSIDE DatabaseError and escaped the retry net entirely on
        # attempt 0 — so the check is message-scoped, not class-scoped.
        def _is_no_more_rows(exc: sqlite3.Error) -> bool:
            return "no more rows available" in str(exc).lower()

        while True:
            self._raise_if_db_corrupt()
            self._raise_if_db_replaced()
            fn_started = False
            try:
                with self._lock:
                    if self._conn is None:
                        # close() ran while this writer was still unwinding
                        # (#94736) — reopen instead of dying on None.execute.
                        self._reopen_after_close_locked(context="write")
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        fn_started = True
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
                if self._write_count % self._FTS_MERGE_EVERY_N_WRITES == 0:
                    self._try_incremental_merge_fts()
                return result
            except SessionCompressionInProgressError:
                # A live foreign compression lock is transient: the compressor
                # publishes in a couple of seconds. Without any wait, a steer
                # that lands mid-compression aborts the user's turn as
                # session_persistence_failed and sends the operator hunting
                # disk space that was never the problem (#75083).
                #
                # The budget is _COMPRESSION_BUSY_WAIT_S, not the write-lock
                # patience: the lease is a correctness boundary, so a writer
                # still locked out after a short wait must be refused rather
                # than left to land a stale turn once a long-running or wedged
                # compression finally lets go.
                if compression_deadline is None:
                    compression_deadline = min(
                        time.monotonic() + self._COMPRESSION_BUSY_WAIT_S, deadline
                    )
                if self._sleep_before_write_retry(
                    compression_deadline, self._COMPRESSION_BUSY_WAIT_S
                ):
                    continue
                raise
            except sqlite3.OperationalError as exc:
                err_msg = str(exc).lower()
                if "locked" in err_msg or "busy" in err_msg:
                    if self._sleep_before_write_retry(deadline, patience_s):
                        continue
                    # Patience exhausted — say what actually happened so the
                    # surfaced error doesn't read as disk/permission damage.
                    raise sqlite3.OperationalError(
                        f"database is locked (another Hermes process held the "
                        f"state.db write lock for over {patience_s:.0f}s — "
                        "likely a long maintenance operation such as VACUUM, "
                        "a large WAL checkpoint, or an older pre-update "
                        "process; the database itself is healthy)"
                    ) from exc
                if _is_no_more_rows(exc) and self._sleep_before_write_retry(deadline, patience_s):
                    continue
                if (
                    _DISK_IO_ERROR_MARKER in err_msg
                    and not fn_started
                    and not ioerr_begin_retried
                    and self._sleep_before_write_retry(deadline, patience_s)
                ):
                    # BEGIN IMMEDIATE itself hit a transient WAL-transition
                    # IOERR. Nothing has been mutated, so retrying on the SAME
                    # connection replays nothing. Never close()+reopen to
                    # "heal" it: close() cancels this process's POSIX locks on
                    # the file for every sibling connection (howtocorrupt §2.2).
                    ioerr_begin_retried = True
                    continue
                # Non-lock error, the callback already ran (settlement is
                # unknown — do not replay), or patience exhausted.
                raise
            except sqlite3.DatabaseError as exc:
                if _is_no_more_rows(exc) and self._sleep_before_write_retry(deadline, patience_s):
                    continue
                # An out-of-band replace of state.db (restore/cp/mv under a
                # live process) surfaces as this same corruption error class.
                # In-file repair on a NEW file generation amplifies the
                # damage (#89332) — halt writes on this handle instead.
                if (
                    "not a database" in str(exc).lower()
                    or is_malformed_db_error(exc)
                    or self._is_fts_write_corruption_error(exc)
                ):
                    self._raise_if_db_replaced()
                # Corrupt FTS shadow tables make every write raise the
                # malformed/corrupt error class through the FTS sync triggers
                # while the canonical messages table is intact. Never run a
                # full-message FTS5 rebuild from this live persistence path:
                # on a multi-gigabyte state.db that can hold the writer lock
                # for minutes. Atomically detach the derived indexes instead,
                # then retry the canonical write. The existing stale-open and
                # explicit repair paths retain rebuild ownership.
                if self._enter_fts_fail_open(exc):
                    continue
                # Bare SQLITE_CORRUPT / NOTADB that survived the replaced-file
                # check and the FTS-scoped fail-open is structural damage:
                # quarantine the handle (see StateDbCorruptError).
                if self._is_structural_corruption_error(exc):
                    self._halt_db_corrupt(exc)
                raise
            except sqlite3.Error as exc:
                # Catch-all for builds that surface 'no more rows available'
                # as InterfaceError (a sibling of DatabaseError, not a
                # subclass) or another sqlite3.Error class outside the two
                # handlers above. Message-scoped: anything else propagates
                # untouched.
                if _is_no_more_rows(exc) and self._sleep_before_write_retry(deadline, patience_s):
                    continue
                raise

    def _ensure_db_file_generation(self) -> None:
        """Mint a once-per-file generation stamp (state_meta + application_id).

        First opener wins via INSERT OR IGNORE. application_id is written
        only when still 0 so racers converge on the same header value.
        PASSIVE checkpoint only — never TRUNCATE (#45383).
        """
        if self.read_only or self._conn is None:
            return
        token = uuid.uuid4().hex
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR IGNORE INTO state_meta (key, value) VALUES (?, ?)",
                    (_STATE_DB_GENERATION_KEY, token),
                )
                row = self._conn.execute(
                    "SELECT value FROM state_meta WHERE key = ?",
                    (_STATE_DB_GENERATION_KEY,),
                ).fetchone()
                if row and row[0]:
                    token = str(row[0])
                self._db_file_generation_token = token
                current = 0
                pragma_row = self._conn.execute("PRAGMA application_id").fetchone()
                if pragma_row:
                    current = int(pragma_row[0] or 0)
                if current == 0:
                    app_id = int(token[:8], 16) & 0x7FFFFFFF
                    if app_id == 0:
                        app_id = 1
                    self._conn.execute(f"PRAGMA application_id={app_id}")
                    current = app_id
                self._db_file_application_id = current
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except sqlite3.Error:
                    pass
        except sqlite3.Error as exc:
            logger.debug("state.db generation stamp skipped: %s", exc)

    def _record_db_file_identity(self) -> None:
        """Snapshot inode plus the on-disk generation header when present."""
        self._db_file_identity = _stat_db_file_identity(self.db_path)
        self._db_sidecar_identity = _stat_sqlite_sidecar_identity(self.db_path)
        disk_id = _read_sqlite_application_id(self.db_path)
        if disk_id:
            self._db_file_application_id = disk_id
        elif self._conn is not None and not self._db_file_application_id:
            try:
                with self._read_ctx() as conn:
                    pragma_row = conn.execute("PRAGMA application_id").fetchone()
                if pragma_row and pragma_row[0]:
                    self._db_file_application_id = int(pragma_row[0])
            except sqlite3.Error:
                pass

    def _db_file_was_replaced(self) -> bool:
        """True when the path no longer names the file this instance opened."""
        recorded = self._db_file_identity
        if recorded is not None:
            current = _stat_db_file_identity(self.db_path)
            if current is None or current != recorded:
                return True
        recorded_app = int(self._db_file_application_id or 0)
        if recorded_app:
            disk_app = _read_sqlite_application_id(self.db_path)
            # Header 0 means the WAL has not been checkpointed yet — not a
            # replace. A copied Hermes DB that minted its own id is nonzero.
            if disk_app and disk_app != recorded_app:
                return True
        return False

    def _halt_db_replaced(self) -> None:
        """Stop writes and raise; do not run in-file repair on a new generation."""
        self._db_replaced = True
        logger.error(_STATE_DB_REPLACED_MSG)
        raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)

    def _wal_generation_was_lost(self) -> bool:
        """True when the WAL/SHM generation this instance opened is gone.

        Steady state (a sidecar generation is recorded): pure stat — a
        recorded inode that is missing or replaced by a new file at the same
        path means the generation split. No /proc walk on healthy writes.

        Empty-identity state (fresh DB whose WAL appears only after open, or
        identity cleared by a clean ``close()``): fall back to a
        ``/proc/self/fd`` deleted-fd probe, and adopt the current sidecars as
        this handle's generation once the probe comes back clean. The full
        ``/proc/*/fd`` walk is reserved for
        :func:`refuse_deleted_wal_generation` on open, where we must see
        *foreign* deleted holders before ``sqlite3.connect`` mints a new WAL.
        """
        recorded = self._db_sidecar_identity or {}
        base = os.fspath(self.db_path)
        if recorded:
            for suffix, recorded_ident in recorded.items():
                current = _stat_db_file_identity(Path(base + suffix))
                if current is None or current != recorded_ident:
                    return True
            return False
        if not self._wal_active:
            # No WAL on this handle (journal_mode=delete/truncate fallback):
            # there is no sidecar generation to lose, and probing every write
            # would put a /proc walk on the hot path of exactly the
            # delete-mode deployments the field report used as containment.
            return False
        if sys.platform.startswith("linux"):
            watched = _watched_sqlite_sidecar_paths(self.db_path)
            fd_dir = f"/proc/{os.getpid()}/fd"
            try:
                for fd in os.listdir(fd_dir):
                    try:
                        target = os.readlink(f"{fd_dir}/{fd}")
                    except OSError:
                        continue
                    if " (deleted)" in target and _canonical_sqlite_path(target) in watched:
                        return True
            except OSError:
                return False
        # Probe clean (or unavailable on this platform): adopt whatever
        # sidecar generation exists now so subsequent writes use the cheap
        # stat check.
        current_identity = _stat_sqlite_sidecar_identity(self.db_path)
        if current_identity:
            self._db_sidecar_identity = current_identity
        return False

    def _halt_deleted_wal_generation(self) -> None:
        """Stop writes; do not mint or keep committing on a split WAL."""
        self._db_wal_generation_lost = True
        logger.error(_DELETED_WAL_GENERATION_MSG)
        raise DeletedWalGenerationError(_DELETED_WAL_GENERATION_MSG)

    def _raise_if_db_replaced(self) -> None:
        if self._db_replaced:
            raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)
        if self._db_wal_generation_lost:
            raise DeletedWalGenerationError(_DELETED_WAL_GENERATION_MSG)
        if self._db_file_was_replaced():
            self._halt_db_replaced()
        if self._wal_generation_was_lost():
            self._halt_deleted_wal_generation()

    @classmethod
    def _is_structural_corruption_error(cls, exc: BaseException) -> bool:
        """Bare SQLITE_CORRUPT/NOTADB with no FTS provenance.

        ``_is_fts_write_corruption_error`` is the positive FTS classifier;
        everything else in the ``corrupt`` bucket of
        ``classify_persistence_error`` is damage to a canonical B-tree, the
        schema, or the freelist — never repairable from the live write path.
        """
        if not isinstance(exc, sqlite3.DatabaseError):
            return False
        if isinstance(exc, StateDbCorruptError):
            return False
        if cls._is_fts_write_corruption_error(exc):
            return False
        return classify_persistence_error(exc) == "corrupt"

    def _corrupt_error(self, prefix: str = "") -> "StateDbCorruptError":
        """Build the quarantine error for this handle (message assembled once)."""
        return StateDbCorruptError(
            f"{prefix}{_STATE_DB_CORRUPT_MSG} (cause: {self._db_corrupt_reason})"
        )

    def _halt_db_corrupt(self, exc: BaseException) -> None:
        """Quarantine this handle and raise; never run in-file repair here."""
        self._db_corrupt = True
        self._db_corrupt_reason = str(exc)
        self._disable_close_time_checkpoint()
        logger.error(
            "state.db %s reported structural corruption outside the FTS "
            "indexes (%s); quarantining this handle: no further writes, no "
            "automatic reopen, no explicit WAL checkpoint at close. Stop the "
            "gateway and run `hermes sessions recover --source %s "
            "--inspect-only`.",
            self.db_path,
            exc,
            self.db_path,
        )
        err = self._corrupt_error()
        for attr in ("sqlite_errorcode", "sqlite_errorname"):
            value = getattr(exc, attr, None)
            if value is not None:
                setattr(err, attr, value)
        raise err from exc

    def _disable_close_time_checkpoint(self) -> None:
        """Best-effort: stop SQLite's own last-connection checkpoint on close.

        Skipping our explicit ``PRAGMA wal_checkpoint(PASSIVE)`` in
        ``close()`` is not enough on its own: ``sqlite3.Connection.close()``
        still runs SQLite's internal last-connection PASSIVE checkpoint and
        unlinks the ``-wal``/``-shm`` sidecars. On the field incident's file
        that close-time checkpoint is exactly what wrote 15 pages under the
        wrong page numbers. Python 3.12+ exposes the switch as
        ``Connection.setconfig(SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE)``; on 3.11
        neither the constant nor ``setconfig`` exists, so the internal
        checkpoint remains (it can only carry pre-quarantine committed
        frames — no further writes are accepted on this handle).
        """
        flag = getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None)
        if flag is None:
            return
        conn = self._conn
        setconfig = getattr(conn, "setconfig", None)
        if conn is None or setconfig is None:
            return
        try:
            setconfig(flag, True)
        except Exception:
            logger.debug(
                "Could not disable SQLite's close-time checkpoint on the "
                "quarantined handle for %s",
                self.db_path,
                exc_info=True,
            )

    def _raise_if_db_corrupt(self) -> None:
        if self._db_corrupt:
            raise self._corrupt_error()

    def _sleep_before_write_retry(
        self, deadline: float, patience_s: float
    ) -> bool:
        """Sleep one jitter interval if the patience budget still allows it.

        Returns True when the caller should retry, False when *deadline* has
        passed and the error should propagate. Jitter stays small for the
        first ``_WRITE_RETRY_SLOW_AFTER_S`` (fast reclaim on millisecond
        contention) and backs off after that, and never overshoots the
        deadline by a full slow-jitter.
        """
        now = time.monotonic()
        if now >= deadline:
            return False
        elapsed = now - (deadline - patience_s)
        if elapsed >= self._WRITE_RETRY_SLOW_AFTER_S:
            jitter = random.uniform(
                self._WRITE_RETRY_SLOW_MIN_S,
                self._WRITE_RETRY_SLOW_MAX_S,
            )
        else:
            jitter = random.uniform(
                self._WRITE_RETRY_MIN_S,
                self._WRITE_RETRY_MAX_S,
            )
        time.sleep(min(jitter, max(deadline - now, 0.001)))
        return True

    @staticmethod
    def _is_fts_write_corruption_error(exc: sqlite3.DatabaseError) -> bool:
        """Return true only when SQLite identifies corruption as FTS-scoped.

        Newer SQLite builds include ``fts5`` in the error text.  Older builds
        may emit only ``database disk image is malformed`` while exposing the
        extended ``SQLITE_CORRUPT_VTAB`` result code.  A bare
        ``SQLITE_CORRUPT``/malformed-image error is structural and must not
        trigger live FTS maintenance: it does not prove that canonical B-trees
        are intact.
        """
        corrupt_vtab = getattr(sqlite3, "SQLITE_CORRUPT_VTAB", 267)
        error_code = getattr(exc, "sqlite_errorcode", None)
        if error_code is not None:
            return error_code == corrupt_vtab
        msg = str(exc).lower()
        return msg.startswith("fts5:") and "corrupt structure" in msg

    def _foreign_state_db_holders(self) -> List[Tuple[int, str]]:
        """Return foreign processes holding this DB or its WAL sidecars."""
        return _foreign_state_db_holders(self.db_path)

    def _reap_inactive_orphan_desktop_holders(
        self, holders: List[Tuple[int, str]], *, min_age_seconds: float
    ) -> List[int]:
        """Terminate old PPID-1 Desktop ephemeral backends with no client.

        Inspection fails closed: anything whose parent, age, argv, or network
        connections cannot be proved safe remains a repair-blocking holder.
        """
        if not sys.platform.startswith("linux") or psutil is None:
            return []
        try:
            from hermes_cli.dashboard_procs import _is_ephemeral_port_zero_backend
        except Exception:
            return []

        now = time.time()
        candidates = []
        for pid in _concrete_state_db_holder_pids(self.db_path, holders):
            try:
                process = psutil.Process(pid)
                statuses = [
                    conn.status for conn in process.net_connections(kind="inet")
                ]
                if not _is_inactive_orphan_desktop_holder(
                    ppid=process.ppid(),
                    age_seconds=now - process.create_time(),
                    min_age_seconds=min_age_seconds,
                    ephemeral_backend=_is_ephemeral_port_zero_backend(process.cmdline()),
                    connection_statuses=statuses,
                ):
                    continue
            except Exception:
                continue
            candidates.append(process)

        signalled: List[int] = []
        for process in candidates:
            try:
                process.terminate()
                signalled.append(process.pid)
            except (psutil.Error, OSError):
                continue
        if not signalled:
            return []

        try:
            _gone, alive = psutil.wait_procs(candidates, timeout=1.5)
        except Exception:
            alive = []
        for process in alive:
            try:
                process.kill()
            except (psutil.Error, OSError):
                continue
        if alive:
            try:
                psutil.wait_procs(alive, timeout=1.5)
            except Exception:
                pass
        return signalled

    def _enter_fts_fail_open(self, exc: sqlite3.DatabaseError) -> bool:
        """Detach corrupt FTS indexes so canonical writes can continue.

        The stale breadcrumb and trigger removal commit atomically. Its
        ordering is load-bearing: after triggers are absent, new canonical
        rows create an index gap of unknown extent, so another process must
        never reinstall the triggers without first rebuilding every row.
        """
        if not self._fts_enabled or not self._is_fts_write_corruption_error(exc):
            return False
        self._raise_if_db_corrupt()
        if self._db_replaced or self._db_file_was_replaced():
            self._halt_db_replaced()
        if self._db_wal_generation_lost or self._wal_generation_was_lost():
            self._halt_deleted_wal_generation()

        try:
            with self._lock:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    self._conn.execute(
                        "INSERT INTO state_meta (key, value) VALUES (?, '1') "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (FTS_STALE_KEY,),
                    )
                    cjk_triggers_present = self._conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'trigger' "
                        f"AND name IN ({','.join('?' for _ in _FTS_CJK_TRIGGERS)}) "
                        "LIMIT 1",
                        _FTS_CJK_TRIGGERS,
                    ).fetchone()
                    if cjk_triggers_present:
                        self._conn.execute(
                            "INSERT INTO state_meta (key, value) VALUES (?, '1') "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                            (FTS_CJK_STALE_KEY,),
                        )
                    self._drop_all_fts_triggers(self._conn.cursor())
                    self._conn.commit()
                except BaseException:
                    self._conn.rollback()
                    raise
        except sqlite3.Error as detach_exc:
            logger.error(
                "Could not detach corrupt FTS indexes; canonical write still "
                "cannot proceed: %s",
                detach_exc,
            )
            return False

        self._fts_stale = True
        self._fts_enabled = False
        self._trigram_available = False
        self._fts_cjk_available = False
        logger.error(
            "state.db FTS indexes remain corrupt (%s); disabled FTS sync and "
            "retrying the canonical write. Search temporarily uses LIKE until "
            "a later SessionDB open rebuilds the indexes.",
            exc,
        )
        return True

    def _try_wal_checkpoint(self) -> None:
        """Best-effort PASSIVE WAL checkpoint.  Never raises.

        Flushes committed WAL frames back into the main DB file without
        requiring an exclusive lock.  PASSIVE is safe for frequent
        periodic use because it does not block concurrent writers and
        cannot corrupt B-tree pages under I/O pressure.

        PASSIVE does not truncate the WAL file — it stays at its
        high-water mark. Explicit checkpoints on the shared ``state.db`` no
        longer truncate the WAL; it is bounded by ``journal_size_limit`` and
        the writer's natural post-checkpoint reset rather than by a TRUNCATE
        at every close or maintenance command.

        Previous TRUNCATE strategy caused B-tree corruption on large
        databases (65K+ pages) due to the exclusive-lock I/O pressure
        from checkpointing thousands of frames at once (issue #45383).
        """
        if self._db_corrupt:
            return  # quarantined: never checkpoint over a damaged image
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

    def __enter__(self) -> "SessionDB":
        """Enter a scope that closes this handle on the way out.

        Ownership of a SessionDB should be released explicitly.
        Historically an instance with a started token writer pinned ITSELF
        (bound-method writer target plus a strong ``atexit`` drain hook), so
        ``__del__`` never ran for exactly the instances that leaked
        descriptors (#88033).  The writer now retires after an idle window
        and the atexit hook holds only a weak reference, so abandoned
        handles are eventually collectible — but "eventually, after the
        idle window and a GC cycle" is not a release policy.  Call sites
        owning a handle are still expected to close it deterministically
        (see the ownership comments in ``run_agent.py`` and
        ``tui_gateway/methods_session.py``).

        This makes the correct usage the easy one, so an owning scope can be
        exception-safe by construction rather than by remembering a
        ``try/finally``:

            with SessionDB(path) as db:
                db.append_message(...)

        Purely additive: it changes nothing for callers that already call
        ``close()`` directly, and ``close()`` stays idempotent, so a scope
        that closes early still exits cleanly.
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        """Close the handle, then let any exception propagate.

        Returns False (never suppressing), so ``with`` here only manages the
        descriptor lifetime and never swallows a caller's error.
        """
        self.close()
        return False

    def close(self):
        """Close the database connection.

        Drains queued token deltas first (the background writer needs the
        connection). Writable connections then attempt a PASSIVE WAL
        checkpoint (NOT TRUNCATE: transient per-cron-run connections close
        many times an hour, and a TRUNCATE fires a full WAL reset that
        races the gateway's live writer and tears B-tree pages — issue
        #45383). Read-only connections never request a checkpoint.

        When this instance is shared (opened via ``get_shared_session_db``),
        ``close()`` RELEASES one refcount instead of tearing down the
        connection: the registry owns the lifecycle and only closes on the
        final release (#90837).  This prevents one caller's close from
        tearing down the writer connection that other callers in the same
        process are still using — while still letting legacy ``close()``
        call sites return their reference instead of leaking it.
        """
        if getattr(self, "_shared_registry_owned", False):
            from hermes_state_registry import release

            release(self)
            return
        self._stop_token_writer()
        hook, self._token_atexit_hook = self._token_atexit_hook, None
        if hook is not None:
            atexit.unregister(hook)
        # Drain the read-only connection pool.  Setting the closed flag
        # under the lock first means a reader still in flight closes its own
        # connection on release instead of re-populating a pool that has
        # already been drained.
        with self._read_conns_lock:
            self._read_conns_closed = True
        while True:
            try:
                conn = self._read_pool.get_nowait()
            except queue.Empty:
                break
            self._close_read_conn(conn)
        with self._lock:
            if self._conn:
                if self._db_corrupt:
                    # Quarantined handle (see StateDbCorruptError): no explicit
                    # checkpoint over a damaged page image.
                    logger.warning(
                        "Skipping the close-time WAL checkpoint for %s: this "
                        "handle observed structural corruption (%s). Take a "
                        "snapshot of state.db, -wal and -shm before restarting, "
                        "then run `hermes sessions recover --source %s "
                        "--inspect-only`.",
                        self.db_path,
                        self._db_corrupt_reason,
                        self.db_path,
                    )
                elif not self.read_only:
                    # PASSIVE, not TRUNCATE. Every cron run_agent opens+closes a
                    # transient SessionDB, so a TRUNCATE here fires a full WAL
                    # reset many times/hour, racing the gateway's long-lived
                    # writer on large WAL databases and tearing hot B-tree
                    # pages -- the #45383 corruption this class's own periodic
                    # checkpoint was already made PASSIVE to avoid. TRUNCATE
                    # belongs only on a sole-opener/quiescent connection.
                    try:
                        self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    except Exception as exc:
                        logger.debug(
                            "WAL checkpoint (PASSIVE) at close failed: %s",
                            exc,
                        )
                conn, self._conn = self._conn, None
                self._close_connection_quietly(conn)
                # A clean close of the last connection lets SQLite unlink the
                # WAL/SHM sidecars — a legitimate end of this handle's sidecar
                # generation, not a split (#94736 late writes must still
                # self-heal). Drop the recorded generation so a teardown-race
                # reopen re-adopts whatever exists then instead of halting.
                self._db_sidecar_identity = {}

    def __del__(self) -> None:
        """Safety net: close the connection if the caller forgot.

        The async accounting worker retires when idle and its atexit hook
        holds only a weak reference, so neither can pin an otherwise orphaned
        instance. During interpreter teardown the order of module cleanup is
        undefined, so every attribute access remains guarded.

        Delegates to ``close()`` so the read pool, token writer, and atexit
        hook are all cleaned up — not just the writer connection.
        """
        if self.__dict__.get("_conn") is None:
            return
        try:
            self.close()
        except Exception:
            pass

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

    # Demoted v22 FTS shadow tables awaiting teardown (see the v23 migration:
    # DROP of a multi-GB FTS vtable blocks for minutes, so the migration
    # demotes the vtable definitions out of sqlite_master and renames the
    # orphaned shadow tables — now plain tables — to fts_v22_trash_*; the
    # worker empties them in bounded chunks, then drops them cheaply).
    _FTS_TRASH_PREFIX = "fts_v22_trash_"

    # ── CJK-bigram index backfill (dedicated marker pair) ──
    #
    # Same chunk engine as the main deferred rebuild, but on the
    # ``fts_cjk_rebuild_*`` markers so a cjk-only backfill (the common case:
    # an already-optimized v23 DB gaining the cjk index) never gates the
    # complete ``messages_fts`` / trigram triggers.

    # ── Opt-in v23 FTS storage optimization (`hermes sessions optimize-storage`) ──
    #
    # This is the ONLY path that migrates an existing legacy (v22 inline) DB
    # to the v23 external-content schema. It is deliberately foreground and
    # user-invoked, never automatic, because it is disk-heavy and long. It
    # runs the throttled/resumable chunk engine above to completion
    # synchronously — demote → new schema → chunked backfill → chunked
    # teardown — with progress callbacks, a disk preflight in the CLI
    # wrapper, a VACUUM at the end, and a defensive schema_version bump.

    def _has_fts_trash(self, conn) -> bool:
        """True when demoted v22 shadow tables are still awaiting teardown.
        Caller must hold ``self._lock`` (or pass a migration-time cursor)."""
        return bool(conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' "
            "AND name LIKE ? ESCAPE '\\' LIMIT 1",
            (self._FTS_TRASH_PREFIX.replace("_", "\\_") + "%",),
        ).fetchone())

    # =========================================================================
    # Session lifecycle
    # =========================================================================

    _PROFILE_DIR_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

    def _own_profile_name(self) -> Optional[str]:
        """The profile that owns THIS store, derived from ``db_path`` alone.

        Every profile-tree ``state.db`` belongs to exactly one profile
        (``<root>/state.db`` → ``default``,
        ``<root>/profiles/<name>/state.db`` → ``<name>``), so the derivation
        is a single match, never a guess — the same contract
        :meth:`backfill_null_session_profiles` and the web listing's
        ``row_profile`` stamp rely on. Path-based (not
        ``get_active_profile_name()``) on purpose: a gateway serving a
        NON-launch profile opens that profile's store directly, and the row
        must be stamped with the store's owner, not the serving process's
        launch profile. Returns ``None`` for stores outside the profile tree
        (explicit ``db_path`` in tests, ad-hoc copies) — those rows keep the
        legacy NULL rather than a fabricated owner.
        """
        try:
            from hermes_constants import get_default_hermes_root

            root = get_default_hermes_root().resolve()
            parent = Path(self.db_path).resolve().parent
            if parent == root:
                return "default"
            if parent.parent == root / "profiles" and self._PROFILE_DIR_RE.match(
                parent.name
            ):
                return parent.name
        except Exception:
            logger.debug("own-profile derivation failed", exc_info=True)
        return None

    def _insert_session_row(
        self,
        session_id: str,
        source: str,
        model: str = None,
        model_config: Dict[str, Any] = None,
        system_prompt: str = None,
        user_id: str = None,
        session_key: Optional[str] = None,
        chat_id: str = None,
        chat_type: str = None,
        thread_id: str = None,
        parent_session_id: str = None,
        cwd: str = None,
        profile_name: Optional[str] = None,
        git_repo_root: str = None,
        origin_json: str = None,
        display_name: str = None,
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

        When the caller passes no ``profile_name`` at all, the row is stamped
        with THIS store's own profile (:meth:`_own_profile_name`) instead of
        NULL. Every ``state.db`` belongs to exactly one profile — the same
        single-match contract :meth:`backfill_null_session_profiles` relies
        on — so the stamp is derivation, not a guess. Rows minted NULL after
        that one-shot #94724 backfill ran stayed NULL forever, and
        profile-keyed consumers (desktop sidebar scope matching,
        ``@session:<profile>/<id>`` deep links, the fail-closed owner ladder)
        treat NULL as unowned: the session vanishes from the sidebar even
        though its transcript is intact (#99222). Stores outside the profile
        tree (explicit ``db_path`` in tests, ad-hoc copies) derive nothing
        and keep NULL — never guess.
        """
        if not (profile_name or "").strip():
            profile_name = self._own_profile_name()

        def _do(conn):
            system_prompt_hash = self._store_system_prompt(conn, system_prompt)
            conn.execute(
                """INSERT INTO sessions (
                   id, source, user_id, session_key, chat_id, chat_type, thread_id,
                   model, model_config, system_prompt, system_prompt_hash,
                   parent_session_id, cwd, profile_name, git_repo_root,
                   origin_json, display_name, started_at
                )
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       model = COALESCE(sessions.model, excluded.model),
                       model_config = CASE
                           WHEN excluded.model_config IS NOT NULL
                                AND json_type(
                                    sessions.model_config, '$._reset_from'
                                ) IS NOT NULL
                                AND json_remove(
                                    sessions.model_config, '$._reset_from'
                                ) = '{}'
                           THEN json_set(
                               excluded.model_config,
                               '$._reset_from',
                               json_extract(
                                   sessions.model_config, '$._reset_from'
                               )
                           )
                           ELSE COALESCE(
                               sessions.model_config, excluded.model_config
                           )
                       END,
                       system_prompt_hash = COALESCE(
                           sessions.system_prompt_hash,
                           excluded.system_prompt_hash
                       ),
                       system_prompt = CASE
                           WHEN sessions.system_prompt_hash IS NULL
                                AND excluded.system_prompt_hash IS NOT NULL
                           THEN NULL
                           ELSE sessions.system_prompt
                       END,
                       session_key = COALESCE(sessions.session_key, excluded.session_key),
                       chat_id = COALESCE(sessions.chat_id, excluded.chat_id),
                       chat_type = COALESCE(sessions.chat_type, excluded.chat_type),
                       thread_id = COALESCE(sessions.thread_id, excluded.thread_id),
                       parent_session_id = COALESCE(sessions.parent_session_id, excluded.parent_session_id),
                       cwd = COALESCE(sessions.cwd, excluded.cwd),
                       profile_name = COALESCE(sessions.profile_name, excluded.profile_name),
                       git_repo_root = COALESCE(sessions.git_repo_root, excluded.git_repo_root),
                       origin_json = COALESCE(sessions.origin_json, excluded.origin_json),
                       display_name = COALESCE(sessions.display_name, excluded.display_name)""",
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
                    system_prompt_hash,
                    parent_session_id,
                    cwd,
                    profile_name,
                    git_repo_root,
                    origin_json,
                    display_name,
                    time.time(),
                ),
            )
            if system_prompt_hash is not None:
                self._delete_unreferenced_system_prompts(conn)
            if parent_session_id:
                conn.execute(
                    f"""UPDATE sessions
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
                                            WHERE p.id = sessions.parent_session_id
                                              AND ({_SAME_KEY_NAMESPACE_SQL})))
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
        # Session-row creation is transcript-critical: if it fails, the
        # first flush of a new session fails and the turn is aborted as
        # session_persistence_failed. Ride out long sibling holds.
        self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

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
        include_compression_ancestors: bool = False,
    ) -> None:
        """Persist the gateway routing peer for an existing session row.

        ``display_name`` / ``origin_json`` carry the gateway's presentation
        and full origin metadata (#9006) so consumers (mcp_serve, mirror,
        channel directory) can read routing data from state.db instead of
        sessions.json.  They are COALESCE'd only in the sense that ``None``
        leaves the existing value untouched.

        ``include_compression_ancestors`` keeps a logical compression lineage
        on one routing peer when an explicit gateway resume moves its tip to a
        different lane. Normal per-turn metadata refreshes update only the
        supplied row.

        Self-healing (#82616): when the target row does not exist yet — the
        gateway's ``create_session`` write failed and was deferred, or a
        crash landed between routing publication and row creation — this
        recorder INSERTs the row with the full identity instead of silently
        no-opping. Every per-turn peer refresh is therefore a repair
        opportunity: a gateway session row can no longer be first-created by
        an identity-less lazy writer (``update_token_counts`` /
        ``record_auxiliary_usage``) and stay unroutable forever.
        """
        if not session_id or not session_key:
            return

        def _do(conn):
            lineage_cte = ""
            target_clause = "WHERE id = ?"
            query_params = []
            if include_compression_ancestors:
                lineage_cte = """
                    WITH RECURSIVE compression_lineage(id) AS (
                        SELECT ?
                        UNION
                        SELECT parent.id
                        FROM compression_lineage lineage
                        JOIN sessions child ON child.id = lineage.id
                        JOIN sessions parent ON parent.id = child.parent_session_id
                        WHERE parent.end_reason = 'compression'
                          AND json_extract(
                              COALESCE(child.model_config, '{}'),
                              '$._branched_from'
                          ) IS NULL
                          AND json_extract(
                              COALESCE(child.model_config, '{}'),
                              '$._delegate_from'
                          ) IS NULL
                          AND COALESCE(child.source, '') != 'tool'
                    )
                """
                target_clause = "WHERE id IN (SELECT id FROM compression_lineage)"
                query_params.append(session_id)
            query_params.extend(
                (
                    session_key,
                    source,
                    user_id,
                    chat_id,
                    chat_type,
                    thread_id,
                    display_name,
                    origin_json,
                )
            )
            if not include_compression_ancestors:
                query_params.append(session_id)
            conn.execute(
                f"""{lineage_cte}
                   UPDATE sessions
                   SET session_key = ?, source = ?, user_id = ?, chat_id = ?,
                       chat_type = ?, thread_id = ?,
                       display_name = COALESCE(?, display_name),
                       origin_json = COALESCE(?, origin_json)
                   {target_clause}""",
                query_params,
            )
            # Self-heal (#82616): the UPDATE is a silent no-op when the row
            # is missing (create_session failed earlier, or a crash landed
            # between routing publication and row creation). Insert it with
            # the full identity so the session is durably routable — never
            # leave first-creation to an identity-less lazy writer.
            if not include_compression_ancestors:
                cur = conn.execute(
                    "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)
                )
                if cur.fetchone() is None:
                    conn.execute(
                        """INSERT INTO sessions (
                               id, source, user_id, session_key, chat_id,
                               chat_type, thread_id, display_name, origin_json,
                               profile_name, started_at
                           )
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(id) DO UPDATE SET
                               session_key = COALESCE(sessions.session_key, excluded.session_key),
                               chat_id = COALESCE(sessions.chat_id, excluded.chat_id),
                               chat_type = COALESCE(sessions.chat_type, excluded.chat_type),
                               thread_id = COALESCE(sessions.thread_id, excluded.thread_id),
                               display_name = COALESCE(sessions.display_name, excluded.display_name),
                               origin_json = COALESCE(sessions.origin_json, excluded.origin_json)""",
                        (
                            session_id,
                            source,
                            user_id,
                            session_key,
                            chat_id,
                            chat_type,
                            thread_id,
                            display_name,
                            origin_json,
                            # Same ownership stamp as _insert_session_row: a
                            # self-healed row is a first creation too, and an
                            # unowned (NULL) row vanishes from profile-keyed
                            # consumers (#99222).
                            self._own_profile_name(),
                            time.time(),
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
        with self._read_ctx() as conn:
            rows = conn.execute(
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

    def list_never_active_keyed_sessions(
        self, *, older_than_days: float
    ) -> List[Dict[str, Any]]:
        """Keyed gateway rows that were opened and then never used at all.

        Selects rows that are keyed (``session_key IS NOT NULL``), still open
        (``ended_at IS NULL``) and carry no evidence of a single turn: no
        messages, no tokens, no tool or API calls, no recorded activity, no
        title.  Such a row is indistinguishable from "never happened".

        That is exactly the shape of a leaked test fixture (#82770) — and
        also of a chat that was routed but never answered.  Both are safe to
        drop: there is no transcript to lose, and the gateway mints a fresh
        session on the next inbound message either way.

        ``bulk prune``/``archive`` cannot reach these rows: their shared
        selector is pinned to ``ended_at IS NOT NULL`` so that a live session
        is never picked, which permanently excludes every never-closed row.
        Hence a separate, narrower selector rather than another filter flag.

        ``pinned`` and ``archived`` rows are excluded — both are explicit
        user intent to keep the row around.
        """
        cutoff = time.time() - (float(older_than_days) * 86400.0)
        with self._read_ctx() as conn:
            rows = conn.execute(
                """
                SELECT s.id, s.session_key, s.source, s.chat_id,
                       s.chat_type, s.user_id, s.started_at
                  FROM sessions s
                 WHERE s.session_key IS NOT NULL
                   AND s.ended_at IS NULL
                   AND s.title IS NULL
                   AND s.last_activity_at IS NULL
                   AND COALESCE(s.message_count, 0) = 0
                   AND COALESCE(s.tool_call_count, 0) = 0
                   AND COALESCE(s.api_call_count, 0) = 0
                   AND COALESCE(s.input_tokens, 0) = 0
                   AND COALESCE(s.output_tokens, 0) = 0
                   AND COALESCE(s.pinned, 0) = 0
                   AND COALESCE(s.archived, 0) = 0
                   AND s.started_at IS NOT NULL
                   AND s.started_at < ?
                   AND NOT EXISTS (
                           SELECT 1 FROM messages m WHERE m.session_id = s.id
                       )
                 ORDER BY s.started_at
                """,
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    def _delete_routing_entries_for_sessions(self, session_ids: Set[str]) -> int:
        """Drop ``gateway_routing`` rows pointing at any of *session_ids*.

        Routing entries are keyed by ``(scope, session_key)`` and record their
        target session inside ``entry_json``, so there is no way to reach them
        by session id in SQL — the match is done in Python over all scopes.
        """
        if not session_ids:
            return 0
        with self._read_ctx() as conn:
            rows = conn.execute(
                "SELECT scope, session_key, entry_json FROM gateway_routing"
            ).fetchall()
        doomed: List[Tuple[str, str]] = []
        for row in rows:
            try:
                entry = json.loads(row["entry_json"] or "{}")
            except Exception:
                continue
            if isinstance(entry, dict) and entry.get("session_id") in session_ids:
                doomed.append((row["scope"], row["session_key"]))
        if not doomed:
            return 0

        def _do(conn):
            conn.executemany(
                "DELETE FROM gateway_routing WHERE scope = ? AND session_key = ?",
                doomed,
            )

        self._execute_write(_do)
        return len(doomed)

    def prune_never_active_keyed_sessions(
        self,
        *,
        older_than_days: float,
        sessions_dir: Optional[Path] = None,
    ) -> Tuple[int, int]:
        """Delete never-active keyed rows and the routing entries naming them.

        Returns ``(sessions_deleted, routing_entries_deleted)``.

        The routing entries go first: a stale entry that outlived its target
        would leave the gateway resuming a session id that no longer exists.
        Deleting the pair is what leaving them both would have amounted to
        anyway — the target had no transcript to resume.

        Deletion goes through :meth:`delete_session` rather than a bulk
        ``DELETE`` so the delegate cascade, FTS bookkeeping and on-disk
        transcript cleanup stay owned by one implementation.
        """
        candidates = self.list_never_active_keyed_sessions(
            older_than_days=older_than_days
        )
        if not candidates:
            return (0, 0)
        ids = {str(row["id"]) for row in candidates}
        routing_deleted = self._delete_routing_entries_for_sessions(ids)
        deleted = 0
        for session_id in ids:
            if self.delete_session(session_id, sessions_dir=sessions_dir):
                deleted += 1
        return (deleted, routing_deleted)

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
        # Full rows carry token/cost totals (MCP listings, /status) — drain
        # queued async accounting deltas so consumers see exact counters.
        self.flush_token_counts()
        query = f"""
            SELECT sessions.*,
                   COALESCE(sp.prompt, sessions.system_prompt)
                       AS _system_prompt_resolved,
                   {_sql_session_last_active("sessions")} AS last_active
            FROM sessions
            LEFT JOIN system_prompts sp
              ON sp.hash = sessions.system_prompt_hash
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
        with self._read_ctx() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._session_row_dict(r) for r in rows]

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
        with self._read_ctx() as conn:
            rows = [dict(r) for r in conn.execute(query, params).fetchall()]
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

        Ordering and emptiness (#82616): candidates are ranked by actual
        conversation recency (``last_activity_at``, falling back to
        ``started_at``) — ``started_at`` alone resurrected days-old zombie
        rows over the live conversation. Rows with messages are preferred,
        but an empty keyed row is still returned rather than ``None``:
        returning ``None`` mints a brand-new session id, which is a worse
        outcome than resuming an empty-but-correctly-keyed row (and "empty"
        may just mean the transcript lives under a compression child).

        Reset boundaries fence recovery (#68539): an intentional boundary
        such as ``session_reset`` (or any explicit non-recoverable
        end_reason) must block fallback to an *older* row for the same
        peer. Without the fence, the has-messages ranking above could reach
        behind a /new reset and silently restore the exact context the user
        reset. Each candidate is therefore rejected when a boundary row for
        the peer ended *after* the candidate's last activity — if the
        conversation's most recent event is an intentional reset, recovery
        returns nothing rather than reaching behind it.
        """
        if not session_key:
            return None
        with self._read_ctx() as conn:
            row = conn.execute(
                f"""
                SELECT s.*,
                       COALESCE(sp.prompt, s.system_prompt)
                           AS _system_prompt_resolved,
                       (COALESCE(s.message_count, 0) > 0 OR EXISTS (
                           SELECT 1 FROM messages WHERE messages.session_id = s.id LIMIT 1
                       )) AS _has_messages
                FROM sessions s
                LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash
                WHERE s.session_key = ?
                  AND s.source = ?
                  AND (s.ended_at IS NULL OR s.end_reason IN ({_RECOVERABLE_END_REASONS_SQL}))
                  AND NOT EXISTS (
                      SELECT 1 FROM sessions b
                      WHERE b.session_key = s.session_key
                        AND b.source = s.source
                        AND b.ended_at IS NOT NULL
                        AND b.end_reason IN ({_RESET_END_REASONS_SQL})
                        AND b.ended_at
                            > COALESCE(s.last_activity_at, s.started_at)
                  )
                ORDER BY _has_messages DESC,
                         COALESCE(s.last_activity_at, s.started_at) DESC
                LIMIT 1
                """,
                (session_key, source),
            ).fetchone()
            if row is not None:
                return self._session_row_dict(row)

            # Conservative fallback for rows created by current code but with a
            # temporarily-missing exact key: still require the complete peer
            # tuple so we never cross chats/threads/users.
            if chat_id is None or chat_type is None:
                return None
            # Profile fence (#74285): a Telegram DM's peer tuple is identical
            # for every bot (chat_id == user_id, no thread), so a sibling
            # profile's row written into this store before the per-profile
            # partition (legacy data) would otherwise be adopted here. Every
            # profile-tree store has one owner; a row is ours when its
            # profile_name is the owner or NULL (legacy rows this store
            # minted). Stores outside the tree derive no owner and keep the
            # historical unfenced behavior.
            owner = self._own_profile_name()
            row = conn.execute(
                f"""
                SELECT s.*,
                       COALESCE(sp.prompt, s.system_prompt)
                           AS _system_prompt_resolved,
                       (COALESCE(s.message_count, 0) > 0 OR EXISTS (
                           SELECT 1 FROM messages WHERE messages.session_id = s.id LIMIT 1
                       )) AS _has_messages
                FROM sessions s
                LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash
                WHERE s.source = ?
                  AND COALESCE(s.user_id, '') = COALESCE(?, '')
                  AND COALESCE(s.chat_id, '') = COALESCE(?, '')
                  AND COALESCE(s.chat_type, '') = COALESCE(?, '')
                  AND COALESCE(s.thread_id, '') = COALESCE(?, '')
                  AND (? IS NULL OR COALESCE(s.profile_name, ?) = ?)
                  AND (s.ended_at IS NULL OR s.end_reason IN ({_RECOVERABLE_END_REASONS_SQL}))
                  AND (COALESCE(s.message_count, 0) > 0 OR EXISTS (
                      SELECT 1 FROM messages WHERE messages.session_id = s.id LIMIT 1
                  ))
                  AND NOT EXISTS (
                      SELECT 1 FROM sessions b
                      WHERE b.source = s.source
                        AND COALESCE(b.user_id, '') = COALESCE(s.user_id, '')
                        AND COALESCE(b.chat_id, '') = COALESCE(s.chat_id, '')
                        AND COALESCE(b.chat_type, '') = COALESCE(s.chat_type, '')
                        AND COALESCE(b.thread_id, '') = COALESCE(s.thread_id, '')
                        AND b.ended_at IS NOT NULL
                        AND b.end_reason IN ({_RESET_END_REASONS_SQL})
                        AND b.ended_at
                            > COALESCE(s.last_activity_at, s.started_at)
                  )
                ORDER BY COALESCE(s.last_activity_at, s.started_at) DESC
                LIMIT 1
                """,
                (source, user_id, chat_id, chat_type, thread_id, owner, owner, owner),
            ).fetchone()
        return self._session_row_dict(row) if row else None

    # ── Orphaned gateway-session repair (#82616) ──────────────────────────
    # A write-path failure (corrupt FTS, crash between routing publication
    # and row creation) can leave the live conversation in a session row
    # that never received its identity columns. Both queries above require
    # those columns, so the row holding the real transcript is invisible to
    # recovery: the chat resolves to the last keyed row instead — days older
    # — and the conversation time-travels. Hardening the write side cannot
    # reach a row that is *already* damaged; these two methods are the
    # offline repair path behind ``hermes sessions repair-routing``.

    # Widest plausible gap between a keyed predecessor going quiet and its
    # unkeyed successor being minted. The reported incident gap was ~60s;
    # 15 minutes stays generous without spanning unrelated conversations.
    _ORPHAN_ADOPTION_MAX_GAP_S = 900.0

    def find_orphaned_gateway_sessions(
        self, *, max_gap_s: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Report message-bearing session rows that lost their routing identity.

        A row is a candidate orphan when it has messages but no
        ``session_key``. It is only *adoptable* when exactly one keyed
        predecessor can be named as the conversation it continues:

        * ``lineage`` — ``parent_session_id`` points at a keyed row of the
          same source. That is a recorded fact, so no time window applies.
        * ``contiguity`` — exactly one keyed row of the same source (and
          compatible ``user_id``) fell quiet within *max_gap_s* of the
          orphan's start, and is older than the orphan's own last activity.

        Anything ambiguous is reported with ``adoptable=False`` and a reason
        rather than guessed at: mis-adopting would splice one person's
        conversation into another person's chat. Branch/delegate/tool rows
        are excluded outright — they are unkeyed by design, not by damage.
        """
        gap = (
            self._ORPHAN_ADOPTION_MAX_GAP_S
            if max_gap_s is None
            else float(max_gap_s)
        )
        orphan_active = _sql_session_last_active("o")
        donor_active = _sql_session_last_active("d")
        donor_columns = (
            "d.id, d.session_key, d.chat_id, d.chat_type, d.thread_id, "
            "d.user_id, d.origin_json, d.display_name, d.end_reason"
        )
        records: List[Dict[str, Any]] = []

        with self._read_ctx() as conn:
            orphans = conn.execute(
                f"""
                SELECT o.id, o.source, o.user_id, o.started_at,
                       o.parent_session_id,
                       {orphan_active} AS last_active,
                       (SELECT COUNT(*) FROM messages m
                         WHERE m.session_id = o.id) AS message_count
                FROM sessions o
                WHERE o.session_key IS NULL
                  AND EXISTS (SELECT 1 FROM messages m
                               WHERE m.session_id = o.id)
                  AND COALESCE(o.source, '') != 'tool'
                  AND json_extract(COALESCE(o.model_config, '{{}}'),
                                   '$._branched_from') IS NULL
                  AND json_extract(COALESCE(o.model_config, '{{}}'),
                                   '$._delegate_from') IS NULL
                ORDER BY o.started_at ASC
                """
            ).fetchall()

            for orphan in orphans:
                donor = None
                evidence = ""
                reason = ""

                if orphan["parent_session_id"]:
                    evidence = "lineage"
                    donor = conn.execute(
                        f"""
                        SELECT {donor_columns}
                        FROM sessions d
                        WHERE d.id = ?
                          AND d.session_key IS NOT NULL
                          AND COALESCE(d.source, '') = COALESCE(?, '')
                        """,
                        (orphan["parent_session_id"], orphan["source"]),
                    ).fetchone()
                    if donor is None:
                        reason = (
                            "parent session carries no gateway identity of "
                            "this source"
                        )
                else:
                    evidence = "contiguity"
                    candidates = conn.execute(
                        f"""
                        SELECT {donor_columns}, {donor_active} AS last_active
                        FROM sessions d
                        WHERE d.session_key IS NOT NULL
                          AND d.id != ?
                          AND COALESCE(d.source, '') = COALESCE(?, '')
                          AND (COALESCE(d.user_id, '') = ''
                               OR COALESCE(?, '') = ''
                               OR d.user_id = ?)
                          AND {donor_active} BETWEEN ? AND ?
                          AND {donor_active} < ?
                        ORDER BY last_active DESC
                        LIMIT 2
                        """,
                        (
                            orphan["id"],
                            orphan["source"],
                            orphan["user_id"],
                            orphan["user_id"],
                            (orphan["started_at"] or 0) - gap,
                            (orphan["started_at"] or 0) + gap,
                            orphan["last_active"],
                        ),
                    ).fetchall()
                    if not candidates:
                        reason = (
                            f"no keyed predecessor fell quiet within {gap:.0f}s "
                            "of this session's start"
                        )
                    elif len(candidates) > 1:
                        reason = (
                            "ambiguous: more than one keyed predecessor "
                            "matches this window"
                        )
                    else:
                        donor = candidates[0]

                records.append(
                    {
                        "orphan_id": orphan["id"],
                        "source": orphan["source"],
                        "message_count": orphan["message_count"],
                        "started_at": orphan["started_at"],
                        "last_active": orphan["last_active"],
                        "donor_id": donor["id"] if donor else None,
                        "session_key": donor["session_key"] if donor else None,
                        "evidence": evidence if donor else "",
                        "adoptable": donor is not None,
                        "reason": reason,
                    }
                )

        # Two unkeyed successors claiming the same predecessor means at most
        # one of them continues that chat, and nothing here says which.
        contested = {
            r["donor_id"]
            for r in records
            if r["adoptable"]
            and sum(1 for x in records if x["donor_id"] == r["donor_id"]) > 1
        }
        for record in records:
            if record["donor_id"] in contested:
                record["adoptable"] = False
                record["reason"] = (
                    "ambiguous: more than one unkeyed session claims this "
                    "predecessor"
                )
        return records

    def adopt_orphaned_gateway_session(
        self, orphan_id: str, donor_id: str
    ) -> bool:
        """Stamp *orphan_id* with *donor_id*'s routing identity, retire *donor_id*.

        Re-verifies the pair inside the write transaction, so a concurrent
        gateway that healed either row in the meantime turns this into a
        no-op instead of a conflicting write. Existing non-NULL columns on
        the orphan are preserved. Returns True when the adoption applied.
        """
        if not orphan_id or not donor_id or orphan_id == donor_id:
            return False

        def _do(conn):
            donor = conn.execute(
                "SELECT session_key, chat_id, chat_type, thread_id, user_id, "
                "origin_json, display_name, source FROM sessions WHERE id = ?",
                (donor_id,),
            ).fetchone()
            orphan = conn.execute(
                "SELECT session_key, source FROM sessions WHERE id = ?",
                (orphan_id,),
            ).fetchone()
            if donor is None or orphan is None:
                return False
            if not donor["session_key"] or orphan["session_key"]:
                return False
            if (donor["source"] or "") != (orphan["source"] or ""):
                return False

            conn.execute(
                """UPDATE sessions
                      SET session_key = ?,
                          chat_id = COALESCE(chat_id, ?),
                          chat_type = COALESCE(chat_type, ?),
                          thread_id = COALESCE(thread_id, ?),
                          user_id = COALESCE(user_id, ?),
                          origin_json = COALESCE(origin_json, ?),
                          display_name = COALESCE(display_name, ?),
                          parent_session_id = COALESCE(parent_session_id, ?)
                    WHERE id = ? AND session_key IS NULL""",
                (
                    donor["session_key"],
                    donor["chat_id"],
                    donor["chat_type"],
                    donor["thread_id"],
                    donor["user_id"],
                    donor["origin_json"],
                    donor["display_name"],
                    donor_id,
                    orphan_id,
                ),
            )
            # Retire the predecessor under a reason recovery does NOT treat
            # as resumable — 'agent_close'/'ws_orphan_reap' would keep it in
            # the running, and the newly keyed orphan could lose the chat
            # again on the next restart.
            conn.execute(
                "UPDATE sessions SET ended_at = COALESCE(ended_at, ?), "
                "end_reason = 'superseded_by_repair' WHERE id = ?",
                (time.time(), donor_id),
            )
            return True

        return self._execute_write(_do)

    # Children that carry a ``parent_session_id`` but are NOT compression
    # continuations: branches, delegate/subagent runs, and tool sessions.
    # A marker only disqualifies a child when it points at the parent being
    # queried — compression continuations inherit the rotated agent's
    # ``model_config`` verbatim (``publish_compression_child`` callers pass
    # ``agent._session_init_model_config``), so a delegate subagent's
    # continuation carries ``_delegate_from=<the delegate's own parent>``.
    # Matching markers by mere presence misclassified those real
    # continuations as delegate children (fail-open for orphan reopen,
    # fail-closed for adoption). Bind the parent id for both markers.
    _NON_CONTINUATION_CHILD_FILTER_SQL = (
        "  AND COALESCE(json_extract(COALESCE({alias}model_config, '{{}}'),"
        " '$._branched_from'), '') != ?\n"
        "  AND COALESCE(json_extract(COALESCE({alias}model_config, '{{}}'),"
        " '$._delegate_from'), '') != ?\n"
        "  AND COALESCE({alias}source, '') != 'tool'\n"
    )

    def find_live_compression_child(
        self, parent_session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return the unique live direct child of a compression-ended session.

        A stale agent may observe that another compression path already rotated
        its parent. Recovery is safe only when the durable lineage identifies
        exactly one live direct continuation. Multiple children are treated as
        ambiguous and fail closed rather than guessing which transcript owns
        subsequent messages.
        """
        if not parent_session_id:
            return None
        with self._read_ctx() as conn:
            parent = conn.execute(
                "SELECT ended_at, end_reason FROM sessions WHERE id = ?",
                (parent_session_id,),
            ).fetchone()
            if (
                parent is None
                or parent["ended_at"] is None
                or parent["end_reason"] != "compression"
            ):
                return None
            rows = conn.execute(
                """
                SELECT s.*,
                       COALESCE(sp.prompt, s.system_prompt)
                           AS _system_prompt_resolved
                FROM sessions s
                LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash
                WHERE s.parent_session_id = ?
                  AND s.ended_at IS NULL
                """
                + self._NON_CONTINUATION_CHILD_FILTER_SQL.format(alias="s.")
                + """
                ORDER BY s.started_at ASC
                LIMIT 2
                """,
                (parent_session_id, parent_session_id, parent_session_id),
            ).fetchall()
        return self._session_row_dict(rows[0]) if len(rows) == 1 else None

    def reopen_orphaned_compression_session(self, session_id: str) -> bool:
        """Reopen a compression parent only when no continuation was published.

        Compression publication is atomic in current builds, but older builds
        could leave a closed parent behind after an interrupted handoff.  This
        recovery is deliberately conservative: an active compression lease or
        any canonical child means the lineage is still owned by another path,
        so the caller must fail closed instead of reopening the parent.
        """
        if not session_id:
            return False

        def _do(conn):
            parent = conn.execute(
                "SELECT ended_at, end_reason FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if (
                parent is None
                or parent["ended_at"] is None
                or parent["end_reason"] != "compression"
            ):
                return False

            # Treat any direct non-branch/non-delegate/non-tool child as a
            # continuation, regardless of its current ended state. Reopening
            # in that case could create a second live head for one lineage.
            child = conn.execute(
                """
                SELECT 1
                FROM sessions
                WHERE parent_session_id = ?
                """
                + self._NON_CONTINUATION_CHILD_FILTER_SQL.format(alias="")
                + """
                LIMIT 1
                """,
                (session_id, session_id, session_id),
            ).fetchone()
            if child is not None:
                return False

            # refresh_compression_lock() deliberately lets an owner revive its
            # own expired row. Reclaim that row inside this write transaction
            # before reopening: refresh-first makes the lease active and aborts
            # recovery; recovery-first deletes the holder identity so a later
            # refresh cannot resurrect it.
            now = time.time()
            lock_row = conn.execute(
                "SELECT holder, expires_at FROM compression_locks "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if lock_row is not None:
                expires_at = lock_row["expires_at"]
                if expires_at is None or float(expires_at) >= now:
                    return False
                deleted = conn.execute(
                    "DELETE FROM compression_locks "
                    "WHERE session_id = ? AND holder = ? AND expires_at = ?",
                    (session_id, lock_row["holder"], expires_at),
                )
                if deleted.rowcount != 1:
                    return False

            updated = conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL "
                "WHERE id = ? AND ended_at IS NOT NULL "
                "AND end_reason = 'compression'",
                (session_id,),
            )
            # rowcount==1 is guaranteed by the parent SELECT at the top of
            # this same BEGIN IMMEDIATE transaction. If this is ever edited
            # to return False past this point, note that the lease DELETE
            # above will still COMMIT (_execute_write commits unless _do
            # raises) — raise instead of returning False to roll back.
            return updated.rowcount == 1

        return bool(self._execute_write(_do))

    def publish_compression_child(
        self,
        *,
        parent_session_id: str,
        child_session_id: str,
        source: str,
        messages: List[Dict[str, Any]],
        model: str = None,
        model_config: Dict[str, Any] = None,
        system_prompt: str = None,
        cwd: str = None,
        profile_name: str = None,
        compression_lock_holder: str = None,
        require_compression_lease: bool = True,
        require_lease_refresh: bool = False,
        lease_ttl_seconds: float = 300.0,
        watermark: Optional[int] = None,
        watermark_ceiling: Optional[int] = None,
    ) -> None:
        """Atomically close a parent and publish its durable compression child.

        The parent closure, child row, and compacted handoff become visible in
        one transaction. Readers can therefore observe either the live parent or
        a complete child, never an ended parent with a missing/empty child.

        Concurrent-append safety (#75316): when *watermark* is provided (the
        parent's :meth:`get_active_message_watermark` captured at compression
        start), parent rows that arrived during the slow summary call
        (``id > watermark``) are cloned into the child AFTER the handoff —
        same pure-SQL column clone as :meth:`archive_and_compact`, with the
        session id rewritten — so a mid-compression append survives rotation
        instead of stranding in the closed parent.

        *watermark_ceiling* bounds the clone from above: the rotation path
        flushes its OWN un-persisted input transcript to the parent right
        before publishing (#47202), and those rows are already represented in
        the compacted handoff — cloning them would duplicate the transcript.
        The caller captures ``MAX(id)`` immediately BEFORE that flush; only
        rows in ``(watermark, watermark_ceiling]`` are foreign concurrent
        tail. ``None`` = unbounded (no internal flush happened).

        When *require_lease_refresh* is True and *compression_lock_holder* is
        set, the lease is refreshed inside the same transaction before the
        expiry check. This gives a refresher
        that stopped due to transient DB failures one final chance to extend
        the lease, preventing wasted compression work. The refresh uses the
        same ``conn`` as the publication, so there is no TOCTOU window.
        """
        def _do(conn):
            if require_lease_refresh and compression_lock_holder:
                conn.execute(
                    "UPDATE compression_locks SET expires_at = ? "
                    "WHERE session_id = ? AND holder = ?",
                    (time.time() + lease_ttl_seconds, parent_session_id,
                     compression_lock_holder),
                )
            lock_row = conn.execute(
                "SELECT holder, expires_at FROM compression_locks WHERE session_id = ?",
                (parent_session_id,),
            ).fetchone()
            if require_compression_lease and (
                lock_row is None
                or not compression_lock_holder
                or lock_row["holder"] != compression_lock_holder
                or float(lock_row["expires_at"]) <= time.time()
            ):
                raise CompressionSessionBusyError(
                    f"Compression lease lost before publication: {parent_session_id}"
                )
            parent = conn.execute(
                """SELECT ended_at, end_reason, cwd, git_branch, git_repo_root,
                          user_id, session_key, chat_id, chat_type,
                          thread_id, display_name, origin_json, profile_name
                   FROM sessions WHERE id = ?""",
                (parent_session_id,),
            ).fetchone()
            if parent is None:
                raise RuntimeError(f"Compression parent not found: {parent_session_id}")
            if parent["ended_at"] is not None:
                # A parent stamped ended by AUTOMATIC cleanup (tui_shutdown,
                # ws_disconnect, orphan reap, idle/LRU evict) while a live
                # agent is publishing its rotation is stale by construction —
                # this writer holds the compression lease and is actively
                # continuing the conversation the stamp claims is over.
                # Left in place it wedges rotation forever: every attempt
                # aborts here, nothing clears the stamp, and each attempt's
                # pre-publish flush re-grows the parent until the provider
                # rejects the request (#88197: 303 unique messages → 2,611
                # rows → HTTP 400). Clear it in this same transaction and
                # proceed; the closure UPDATE below re-stamps the parent with
                # its true boundary (end_reason='compression'). Deliberate
                # boundaries (compression, session_reset, explicit close)
                # still fail closed — those mean another path owns lineage.
                if is_automatic_end_reason(parent["end_reason"]):
                    conn.execute(
                        "UPDATE sessions SET ended_at = NULL, end_reason = NULL "
                        "WHERE id = ?",
                        (parent_session_id,),
                    )
                else:
                    raise RuntimeError(
                        f"Compression parent already ended: {parent_session_id}"
                    )
            if not messages:
                raise RuntimeError("Compression child handoff must not be empty")
            system_prompt_hash = self._store_system_prompt(conn, system_prompt)

            conn.execute(
                """INSERT INTO sessions (
                   id, source, model, model_config, system_prompt,
                   system_prompt_hash,
                   parent_session_id, cwd, git_branch, git_repo_root,
                   profile_name, user_id, session_key, chat_id, chat_type,
                   thread_id, display_name, origin_json, started_at
                ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    child_session_id,
                    source,
                    model,
                    json.dumps(model_config) if model_config else None,
                    system_prompt_hash,
                    parent_session_id,
                    cwd or parent["cwd"],
                    parent["git_branch"],
                    parent["git_repo_root"],
                    # Same inheritance contract as _insert_session_row's
                    # compression-fork backfill (#59527 / cross-profile jump
                    # fix): the child stays on the parent's profile and keeps
                    # the gateway routing/origin columns so peer recovery
                    # still works after a crash at the boundary. When neither
                    # names an owner (legacy NULL parent), stamp this store's
                    # own profile so the rotated child doesn't extend the
                    # unowned lineage (#99222).
                    profile_name
                    or parent["profile_name"]
                    or self._own_profile_name(),
                    parent["user_id"],
                    parent["session_key"],
                    parent["chat_id"],
                    parent["chat_type"],
                    parent["thread_id"],
                    parent["display_name"],
                    parent["origin_json"],
                    time.time(),
                ),
            )
            total_messages, total_tool_calls = self._insert_message_rows(
                conn, child_session_id, messages
            )
            if watermark is not None:
                # Clone the parent's concurrent tail (rows landed after the
                # watermark, at or below the ceiling — see docstring) into the
                # child, after the handoff. Column-exact except id/session_id;
                # originals stay in the (closed) parent for lineage recovery.
                _ceiling_clause = ""
                _params: list = [parent_session_id, int(watermark)]
                if watermark_ceiling is not None:
                    _ceiling_clause = " AND id <= ?"
                    _params.append(int(watermark_ceiling))
                tail_rows = conn.execute(
                    "SELECT id, tool_calls FROM messages "
                    "WHERE session_id = ? AND active = 1 AND id > ?"
                    f"{_ceiling_clause} ORDER BY id",
                    _params,
                ).fetchall()
                if tail_rows:
                    tail_ids = [int(r["id"]) for r in tail_rows]
                    placeholders = ",".join("?" for _ in tail_ids)
                    clone_cols = [
                        c for c in self._message_column_names(conn)
                        if c not in ("id", "session_id", "active", "compacted")
                    ]
                    col_list = ", ".join(clone_cols)
                    conn.execute(
                        f"INSERT INTO messages ({col_list}, session_id, active, compacted) "
                        f"SELECT {col_list}, ?, 1, 0 FROM messages "
                        f"WHERE id IN ({placeholders}) ORDER BY id",
                        [child_session_id, *tail_ids],
                    )
                    total_messages += len(tail_ids)
                    for r in tail_rows:
                        raw = r["tool_calls"]
                        if raw:
                            try:
                                parsed = json.loads(raw) if isinstance(raw, str) else raw
                                total_tool_calls += len(parsed) if isinstance(parsed, list) else 0
                            except (TypeError, ValueError):
                                pass
            conn.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                (total_messages, total_tool_calls, child_session_id),
            )
            updated = conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = 'compression' "
                "WHERE id = ? AND ended_at IS NULL",
                (time.time(), parent_session_id),
            )
            if updated.rowcount != 1:
                raise RuntimeError(
                    f"Compression parent changed during publication: {parent_session_id}"
                )

        self._execute_write(_do)

    def _bump_conversation_generation(self, conn, session_id: str, end_reason: str) -> None:
        """Advance this peer's conversation generation past a boundary.

        Called inside the transaction that writes the boundary, so the
        generation and the ``end_reason`` that caused it commit together.

        Only ``_RESET_END_REASONS`` count: ``compression`` continues one
        conversation, and an accidental close is not a replacement.  Rows with
        no ``session_key`` have no routing peer to advance.

        The counter deliberately does NOT read the session rows.  An aggregate
        over them (COUNT/MAX of boundaries) can return a pair it already
        emitted once ``delete_session()`` or bulk pruning removes an ended row,
        which would hand a new conversation a retired affinity identity.  This
        value only ever increments, so a generation is never reused for a peer
        even if every row behind it is gone.
        """
        if end_reason not in _RESET_END_REASONS:
            return
        row = conn.execute(
            "SELECT source, session_key FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return
        source = str(row["source"] or "").strip()
        session_key = str(row["session_key"] or "").strip()
        if not source or not session_key:
            return
        conn.execute(
            """
            INSERT INTO conversation_generations (source, session_key, generation)
            VALUES (?, ?, 1)
            ON CONFLICT(source, session_key) DO UPDATE
                SET generation = conversation_generations.generation + 1
            """,
            (source, session_key),
        )

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
            changed = conn.execute(
                "UPDATE sessions SET ended_at = ?, end_reason = ? "
                "WHERE id = ? AND ended_at IS NULL",
                (time.time(), end_reason, session_id),
            ).rowcount
            # Only a boundary this call actually wrote advances the generation:
            # the first end_reason wins, so a no-op must not rotate the peer.
            if changed:
                self._bump_conversation_generation(conn, session_id, end_reason)
        self._execute_write(_do)

    def reopen_session(self, session_id: str) -> None:
        """Clear ended_at/end_reason so a session can be resumed.

        Before clearing a reset boundary, stabilize markerless legacy reset
        children that still depend on the parent's mutable end_reason.
        """
        def _do(conn):
            placeholders = ",".join("?" for _ in _RESET_END_REASONS)
            # WHERE shape shared with _RESET_CHILD_SQL's fallback arm via
            # _legacy_reset_child_sql so the stamping and the listing
            # predicate cannot drift.
            conn.execute(
                "UPDATE sessions AS child SET model_config = json_set("
                "COALESCE(child.model_config, '{}'), '$._reset_from', "
                "child.parent_session_id) "
                "WHERE child.parent_session_id = ? "
                "AND json_extract(COALESCE(child.model_config, '{}'), "
                "                 '$._reset_from') IS NULL "
                f"AND {_legacy_reset_child_sql('child', placeholders)}",
                (session_id, *_RESET_END_REASONS),
            )
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
                f"OR end_reason IN ({_RECOVERABLE_END_REASONS_SQL}))",
                (now, reason, session_id),
            )
            # /new and the policy auto-resets promote rather than end_session,
            # so the generation has to advance here too — in the same
            # transaction as the boundary, and only when one was written.
            if cursor.rowcount:
                self._bump_conversation_generation(conn, session_id, reason)
            return cursor.rowcount

        try:
            rows = self._execute_write(_do)
            return bool(rows)
        except Exception:
            return False

    def update_session_cwd(
        self,
        session_id: str,
        cwd: str,
        git_branch: Optional[str] = None,
        git_repo_root: Optional[str] = None,
        replace_git_meta: bool = False,
    ) -> Optional[int]:
        """Persist the authoritative cwd and claim a Git metadata generation.

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

        ``replace_git_meta`` inverts that non-empty rule: a deliberate workspace
        MOVE (re-homing a session into another project) must overwrite the old
        repo identity even when the new cwd resolves to none — keeping the stale
        root would leave the session grouped under the project it just left.

        Every call increments ``git_metadata_generation`` in the same write
        transaction. Async Git probes must publish through
        :meth:`publish_session_git_metadata` with the returned generation, so
        an older worker cannot overwrite a newer cwd claim even after an
        A -> B -> A transition or from another process sharing this database.
        Metadata from a different cwd is cleared atomically with the move.
        """
        if not session_id or not cwd:
            return None

        branch = (git_branch or "").strip()
        repo_root = (git_repo_root or "").strip()

        def _do(conn):
            current = conn.execute(
                "SELECT cwd FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if current is None:
                return None

            current_cwd = current["cwd"] if isinstance(current, sqlite3.Row) else current[0]
            sets = [
                "cwd = ?",
                "git_metadata_generation = COALESCE(git_metadata_generation, 0) + 1",
            ]
            params: List[Any] = [cwd]
            if current_cwd != cwd or replace_git_meta:
                sets.extend(("git_branch = ?", "git_repo_root = ?"))
                params.extend((branch or None, repo_root or None))
            elif branch:
                sets.append("git_branch = ?")
                params.append(branch)
            if repo_root and current_cwd == cwd and not replace_git_meta:
                sets.append("git_repo_root = ?")
                params.append(repo_root)
            params.append(session_id)
            conn.execute(
                f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", params
            )
            row = conn.execute(
                "SELECT git_metadata_generation FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            value = row["git_metadata_generation"] if isinstance(row, sqlite3.Row) else row[0]
            return int(value)

        return self._execute_write(_do)

    def publish_session_git_metadata(
        self,
        session_id: str,
        cwd: str,
        generation: int,
        git_branch: Optional[str] = None,
        git_repo_root: Optional[str] = None,
    ) -> bool:
        """Publish async Git enrichment only while its cwd claim is current."""
        if (
            not session_id
            or not cwd
            or isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 1
        ):
            return False

        branch = (git_branch or "").strip()
        repo_root = (git_repo_root or "").strip()
        if not branch and not repo_root:
            return False

        sets: List[str] = []
        params: List[Any] = []
        if branch:
            sets.append("git_branch = ?")
            params.append(branch)
        if repo_root:
            sets.append("git_repo_root = ?")
            params.append(repo_root)
        params.extend((session_id, cwd, generation))

        def _do(conn):
            cursor = conn.execute(
                f"UPDATE sessions SET {', '.join(sets)} "
                "WHERE id = ? AND cwd = ? "
                "AND git_metadata_generation = ?",
                params,
            )
            return cursor.rowcount == 1

        return bool(self._execute_write(_do))

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
            # Merge-max with any longer live deadline so a later shorter
            # write cannot reopen the thrash window (#96775). The error
            # column always takes the latest diagnostic.
            conn.execute(
                "UPDATE sessions SET compression_failure_cooldown_until = CASE "
                "WHEN compression_failure_cooldown_until IS NOT NULL "
                " AND compression_failure_cooldown_until > ? "
                "THEN compression_failure_cooldown_until ELSE ? END, "
                "compression_failure_error = ? WHERE id = ?",
                (cooldown_until, cooldown_until, error, session_id),
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
        with self._read_ctx() as conn:
            row = conn.execute(
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

    def get_compression_failure_cooldown_row(
        self,
        session_id: str,
    ) -> Dict[str, Any]:
        """Return the exact stored cooldown columns without expiry filtering.

        Compression cancellation uses this under its session lease so rollback
        can preserve an expired row, a partially-null row, or an absent session
        exactly instead of converting those states through the active-cooldown
        API.
        """
        if not session_id:
            return {"session_exists": False, "cooldown_until": None, "error": None}
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT compression_failure_cooldown_until, compression_failure_error "
                "FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return {"session_exists": False, "cooldown_until": None, "error": None}
        cooldown_until = (
            row["compression_failure_cooldown_until"]
            if isinstance(row, sqlite3.Row)
            else row[0]
        )
        error = (
            row["compression_failure_error"]
            if isinstance(row, sqlite3.Row)
            else row[1]
        )
        return {
            "session_exists": True,
            "cooldown_until": (
                float(cooldown_until) if cooldown_until is not None else None
            ),
            "error": error,
        }

    def restore_compression_failure_cooldown_row(
        self,
        session_id: str,
        snapshot: Dict[str, Any],
    ) -> None:
        """Restore and verify an exact cooldown-row snapshot.

        Unlike the ordinary record/clear helpers, this transactional rollback
        API deliberately propagates write and verification failures. A caller
        must not report cancellation as mutation-free when compensation failed.
        """
        expected_exists = bool(snapshot.get("session_exists", False))
        if not expected_exists:
            actual = self.get_compression_failure_cooldown_row(session_id)
            if actual.get("session_exists", False):
                raise RuntimeError(
                    "cannot restore absent compression cooldown row: session now exists"
                )
            return

        deadline = snapshot.get("cooldown_until")
        error = snapshot.get("error")

        def _do(conn):
            cursor = conn.execute(
                "UPDATE sessions SET compression_failure_cooldown_until = ?, "
                "compression_failure_error = ? WHERE id = ?",
                (deadline, error, session_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"compression cooldown rollback session missing: {session_id}"
                )

        self._execute_write(_do)
        actual = self.get_compression_failure_cooldown_row(session_id)
        expected = {
            "session_exists": True,
            "cooldown_until": float(deadline) if deadline is not None else None,
            "error": error,
        }
        if actual != expected:
            raise RuntimeError(
                f"compression cooldown rollback verification failed: "
                f"expected={expected!r}, actual={actual!r}"
            )

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
        with self._read_ctx() as conn:
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

    def increment_hygiene_failure_streak(self, session_key: str) -> int:
        """Atomically increment the session-hygiene failure streak for one chat."""
        if not session_key:
            return 1
        result = []

        def _do(conn):
            conn.execute(
                """INSERT INTO gateway_hygiene_state (session_key, failure_streak)
                   VALUES (?, 1)
                   ON CONFLICT(session_key) DO UPDATE SET
                       failure_streak = gateway_hygiene_state.failure_streak + 1""",
                (session_key,),
            )
            row = conn.execute(
                "SELECT failure_streak FROM gateway_hygiene_state WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            result.append(int(row[0]))

        self._execute_write(_do)
        return result[0]

    def reset_hygiene_failure_streak(self, session_key: str) -> None:
        """Clear the persisted session-hygiene failure streak for one chat."""
        if not session_key:
            return

        def _do(conn):
            conn.execute(
                "DELETE FROM gateway_hygiene_state WHERE session_key = ?",
                (session_key,),
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
        with self._read_ctx() as conn:
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

    def get_compression_recovery_deadline(self, session_id: str) -> float:
        """Return the persisted anti-thrash recovery deadline (wall-clock epoch).

        ``0.0`` means "not armed". The deadline is the durable half of the
        #14694 recovery clock: the gateway rebuilds the compressor on every
        turn / cache eviction, so a process-local deadline restarted the
        wait on each rebuild and a tripped session never earned its probe
        (#100185).
        """
        if not session_id:
            return 0.0
        with self._read_ctx() as conn:
            if conn is None:
                return 0.0
            row = conn.execute(
                "SELECT compression_recovery_deadline FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return 0.0
        value = (
            row["compression_recovery_deadline"]
            if isinstance(row, sqlite3.Row)
            else row[0]
        )
        try:
            return max(0.0, float(value or 0.0))
        except (TypeError, ValueError):
            return 0.0

    def set_compression_recovery_deadline(self, session_id: str, deadline: float) -> None:
        """Persist the anti-thrash recovery deadline; ``0`` / ``None`` disarms it."""
        if not session_id:
            return
        try:
            normalized = max(0.0, float(deadline or 0.0))
        except (TypeError, ValueError):
            normalized = 0.0
        stored = normalized if normalized > 0.0 else None

        def _do(conn):
            conn.execute(
                "UPDATE sessions SET compression_recovery_deadline = ? WHERE id = ?",
                (stored, session_id),
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
        """Extend the compression lock lease if ``holder`` still owns it.

        Ownership is decided by the ``holder`` column alone, deliberately NOT
        by ``expires_at``: a live owner whose refresher thread was starved
        (GC pause, loaded CI runner, a slow write escaping ``_execute_write``'s
        retry budget) past its own TTL must be able to revive its still-unclaimed
        row on the next tick. Requiring ``expires_at >= now`` here made such a
        stall permanent — every later refresh matched 0 rows, so the owner kept
        compressing and rotating with no lease at all, which is exactly the
        unprotected window a competing path can fork the session lineage in.

        This does not resurrect a lock somebody else already took: SQLite
        serialises writes, so a reclaim (DELETE-expired + INSERT-or-IGNORE in
        :meth:`try_acquire_compression_lock`) and this UPDATE never interleave.
        Reclaim-first replaces ``holder``, so this UPDATE matches nothing and
        returns False; refresh-first pushes ``expires_at`` into the future, so
        the reclaimer's DELETE-expired matches nothing and its acquire fails.
        """
        if not session_id or not holder:
            return False
        now = time.time()
        expires_at = now + ttl_seconds

        def _do(conn):
            cur = conn.execute(
                "UPDATE compression_locks SET expires_at = ? "
                "WHERE session_id = ? AND holder = ?",
                (expires_at, session_id, holder),
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

    def _session_turn_lease_key_on_conn(self, conn, session_id: str) -> str:
        """Walk compression parents on ``conn`` to the conversation lease key.

        Must run on the same connection as the lease INSERT/UPDATE/DELETE.
        A prior ``get_session`` failure must not compute a child id that the
        later write then persists: refresh would walk to the parent and
        fail-close. Markers bind to ``parent_session_id`` (same contract as
        ``_NON_CONTINUATION_CHILD_FILTER_SQL``). Lock errors propagate so
        ``_execute_write`` / ``acquire_session_turn_lease`` can retry.
        """
        if not session_id:
            return session_id

        def _row(sid: str):
            row = conn.execute(
                "SELECT id, parent_session_id, source, model_config, end_reason "
                "FROM sessions WHERE id = ?",
                (sid,),
            ).fetchone()
            return dict(row) if row else None

        current = _row(session_id)
        seen = {session_id}
        while current:
            parent_id = current.get("parent_session_id")
            if (
                not parent_id
                or parent_id in seen
                or self._is_explicit_fork_child_row(current)
            ):
                break
            parent = _row(parent_id)
            if not parent or parent.get("end_reason") != "compression":
                break
            seen.add(parent_id)
            current = parent
        return str(current.get("id") or session_id) if current else session_id

    def _session_turn_lease_key(self, session_id: str) -> str:
        """Return the stable serialization key for every compression segment.

        Acquire/refresh/release resolve this inside their write transaction.
        This helper is for tests and diagnostics; it does not swallow lock
        errors (a swallowed walk plus a later successful write was the
        fail-open that replayed the post-rotation refresh miss).
        """
        if not session_id:
            return session_id
        with self._read_ctx() as conn:
            return self._session_turn_lease_key_on_conn(conn, session_id)

    def try_acquire_session_turn_lease(
        self,
        session_id: str,
        holder: str,
        *,
        ttl_seconds: float = 300.0,
        patience_s: Optional[float] = None,
    ) -> bool:
        """Atomically acquire the cross-process turn lease for a conversation.

        Compression rotates a session into child segments, so the durable key
        is the lineage root rather than the current segment id. The walk and
        INSERT share one write transaction. Expired leases and leases whose
        structured local holder PID is known dead are reclaimed in that same
        transaction.
        """
        if not session_id or not holder:
            return False
        now = time.time()
        expires_at = now + max(0.1, float(ttl_seconds))

        def _do(conn):
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            row = conn.execute(
                "SELECT holder, expires_at FROM session_turn_leases "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if row is not None:
                current_holder = row["holder"]
                if (
                    float(row["expires_at"]) <= now
                    or _compression_lock_holder_process_is_dead(current_holder)
                ):
                    conn.execute(
                        "DELETE FROM session_turn_leases "
                        "WHERE conversation_id = ? AND holder = ?",
                        (conversation_id, current_holder),
                    )
            conn.execute(
                "INSERT OR IGNORE INTO session_turn_leases "
                "(conversation_id, holder, acquired_at, expires_at) "
                "VALUES (?, ?, ?, ?)",
                (conversation_id, holder, now, expires_at),
            )
            owner = conn.execute(
                "SELECT holder FROM session_turn_leases WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            return owner is not None and owner["holder"] == holder

        return bool(self._execute_write(_do, patience_s=patience_s))

    def acquire_session_turn_lease(
        self,
        session_id: str,
        holder: str,
        *,
        ttl_seconds: float = 300.0,
        wait_seconds: float = 1800.0,
        poll_interval_seconds: float = 1.0,
        on_wait=None,
        wait_notice_interval_seconds: float = 15.0,
        should_abort=None,
        acquire_patience_s: float = 0.5,
    ) -> bool:
        """Wait for a cross-process turn lease without holding a SQLite lock.

        ``on_wait(elapsed_seconds)`` is best-effort: invoked when the first
        attempt fails (elapsed ~0) and again about every
        ``wait_notice_interval_seconds`` while still waiting, so UIs can show
        that another process holds the conversation.

        When ``should_abort()`` returns True (for example the agent received
        ``/stop`` while waiting), acquisition stops immediately and returns
        False without consuming the full ``wait_seconds`` budget.
        """
        deadline = time.monotonic() + max(0.0, float(wait_seconds))
        wait_started = None
        last_notice_at = None
        notice_every = max(0.0, float(wait_notice_interval_seconds))
        while True:
            if should_abort is not None:
                try:
                    if should_abort():
                        return False
                except Exception:
                    logger.debug(
                        "session turn lease should_abort callback failed",
                        exc_info=True,
                    )
            try:
                if self.try_acquire_session_turn_lease(
                    session_id,
                    holder,
                    ttl_seconds=ttl_seconds,
                    patience_s=acquire_patience_s,
                ):
                    return True
            except sqlite3.Error as exc:
                # Long holder transactions (compression publish, large
                # flushes) can exhaust a single write-patience budget.
                # Keep polling until wait_seconds or should_abort.
                if classify_persistence_error(exc) != "locked":
                    raise
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                return False
            if wait_started is None:
                wait_started = now
            if on_wait is not None and (
                last_notice_at is None
                or notice_every == 0.0
                or (now - last_notice_at) >= notice_every
            ):
                try:
                    on_wait(max(0.0, now - wait_started))
                except Exception:
                    logger.debug(
                        "session turn lease on_wait callback failed",
                        exc_info=True,
                    )
                last_notice_at = now
            time.sleep(min(max(0.01, float(poll_interval_seconds)), remaining))

    def refresh_session_turn_lease(
        self,
        session_id: str,
        holder: str,
        *,
        ttl_seconds: float = 300.0,
    ) -> bool:
        """Extend a turn lease only while ``holder`` still owns it."""
        if not session_id or not holder:
            return False
        expires_at = time.time() + max(0.1, float(ttl_seconds))

        def _do(conn):
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            cursor = conn.execute(
                "UPDATE session_turn_leases SET expires_at = ? "
                "WHERE conversation_id = ? AND holder = ?",
                (expires_at, conversation_id, holder),
            )
            return cursor.rowcount > 0

        return bool(self._execute_write(_do))

    def release_session_turn_lease(self, session_id: str, holder: str) -> None:
        """Release a turn lease iff ``holder`` still owns it; idempotent."""
        if not session_id or not holder:
            return

        def _do(conn):
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            conn.execute(
                "DELETE FROM session_turn_leases "
                "WHERE conversation_id = ? AND holder = ?",
                (conversation_id, holder),
            )

        self._execute_write(_do)

    def get_compression_lock_holder(self, session_id: str) -> Optional[str]:
        """Return the current (non-expired) holder for ``session_id``, or None.

        Diagnostic helper — not used by the locking protocol itself.
        """
        if not session_id:
            return None
        now = time.time()
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT holder FROM compression_locks "
                "WHERE session_id = ? AND expires_at >= ?",
                (session_id, now),
            ).fetchone()
        if row is None:
            return None
        return row["holder"] if isinstance(row, sqlite3.Row) else row[0]

    def touch_session_activity(
        self,
        session_id: str,
        ts: Optional[float] = None,
        *,
        description: Optional[str] = None,
        provenance: Optional[ActivityProvenance] = None,
    ) -> None:
        """Stamp durable mid-turn session activity (observation-only).

        Called (rate-limited) from ``AIAgent._touch_activity`` so gateway/CLI
        surfaces and stall consumers observe API/tool/compaction activity
        even when no new message row has been written yet (#72016 / #72039).

        Never moves ``last_activity_at`` backwards. When the timestamp
        advances, bounded ``last_activity_description`` /
        ``last_activity_provenance`` are written with it. No-ops when
        ``session_id`` is empty or the row does not exist.
        """
        if not session_id:
            return
        from agent.session_activity import (
            bound_activity_description,
            normalize_activity_provenance,
        )

        when = float(ts if ts is not None else time.time())
        desc = bound_activity_description(description)
        prov = normalize_activity_provenance(provenance).value

        def _do(conn):
            conn.execute(
                "UPDATE sessions SET "
                "last_activity_at = ?, "
                "last_activity_description = ?, "
                "last_activity_provenance = ? "
                "WHERE id = ? AND (last_activity_at IS NULL OR last_activity_at < ?)",
                (when, desc, prov, session_id, when),
            )

        # Observation-only write: never let it ride the full routine
        # write-patience budget (#76354 review S1). Under contention a
        # heartbeat that waits ~20s would delay the response-critical path
        # it is merely observing; give up after a sub-second budget instead
        # (the next due window retries naturally).
        self._execute_write(_do, patience_s=self._ACTIVITY_WRITE_PATIENCE_S)

    def clear_session_activity_labels(self, session_id: str) -> None:
        """Clear mid-turn activity labels after a turn ends.

        Keeps ``last_activity_at`` intact so idle / watchdog clocks stay
        continuous. Description and provenance are observation labels for
        *what was happening at* that timestamp during an active turn; once
        the turn is idle they must not keep advertising "compressing" /
        "executing tool" (#72039).

        Response-critical-path contract (#76354 review S1): runs in the
        turn's ``finally``; a no-op clear (labels already empty) skips the
        write transaction entirely, and a real clear uses the same short
        sub-second busy budget as :meth:`touch_session_activity` instead of
        the full routine write patience.
        """
        if not session_id:
            return
        from agent.session_activity import ActivityProvenance

        # No-op fast path: skip the transaction when there is nothing to
        # clear. Read-only, no write lock.
        try:
            with self._read_ctx() as conn:
                row = conn.execute(
                    "SELECT last_activity_description, last_activity_provenance "
                    "FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
        except sqlite3.Error:
            row = None
        if row is not None:
            desc = row[0] if not isinstance(row, sqlite3.Row) else row["last_activity_description"]
            prov = row[1] if not isinstance(row, sqlite3.Row) else row["last_activity_provenance"]
            if not desc and (
                not prov or prov == ActivityProvenance.UNKNOWN.value
            ):
                return

        def _do(conn):
            conn.execute(
                "UPDATE sessions SET "
                "last_activity_description = ?, "
                "last_activity_provenance = ? "
                "WHERE id = ?",
                ("", ActivityProvenance.UNKNOWN.value, session_id),
            )

        self._execute_write(_do, patience_s=self._ACTIVITY_WRITE_PATIENCE_S)

    def get_session_activity(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the durable activity snapshot for *session_id*, or None."""
        if not session_id:
            return None
        row = self.get_session(session_id)
        if not row:
            return None
        from agent.session_activity import build_activity_snapshot

        return build_activity_snapshot(
            last_activity_at=row.get("last_activity_at"),
            last_activity_description=row.get("last_activity_description"),
            last_activity_provenance=row.get("last_activity_provenance"),
        )

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
        # Barrier against queued token deltas — see update_session_model.
        self.flush_token_counts()

        def _do(conn):
            conn.execute(
                "UPDATE sessions SET model_config = ?, model = COALESCE(?, model) WHERE id = ?",
                (model_config_json, model, session_id),
            )
        self._execute_write(_do)

    def update_system_prompt(
        self, session_id: str, system_prompt: Optional[str]
    ) -> None:
        """Store the full assembled system prompt snapshot."""
        def _do(conn):
            system_prompt_hash = self._store_system_prompt(conn, system_prompt)
            conn.execute(
                "UPDATE sessions "
                "SET system_prompt_hash = ?, system_prompt = NULL WHERE id = ?",
                (system_prompt_hash, session_id),
            )
            self._delete_unreferenced_system_prompts(conn)
        self._execute_write(_do)

    def update_session_tool_names(
        self, session_id: str, tool_names: Optional[List[str]]
    ) -> None:
        """Persist the session's resolved ``tools[]`` name order (JSON array).

        Read back by ``tools.mcp_tool.restore_agent_tool_prefix`` when a fresh
        ``AIAgent`` is rebuilt for an existing session (gateway agent-cache
        eviction) so a flipped ``check_fn`` verdict can't fork the cached tool
        prefix. ``None`` clears the pin.
        """
        payload = json.dumps(list(tool_names)) if tool_names is not None else None

        def _do(conn):
            conn.execute(
                "UPDATE sessions SET tool_names = ? WHERE id = ?",
                (payload, session_id),
            )
        self._execute_write(_do)

    def update_session_model(
        self, session_id: str, model: str, provider: Optional[str] = None
    ) -> None:
        """Update the model for a session after a mid-session switch.

        Unlike ``update_token_counts`` which uses ``COALESCE(model, ?)``
        (only filling in NULL), this unconditionally sets the model column
        so that the dashboard reflects the user's latest /model choice.
        Also nulls ``system_prompt`` so stale ``Model:`` / ``Provider:``
        footer metadata is rebuilt on the next turn. A successful /model
        switch explicitly replaces any confirmed Browser runtime lock while
        preserving unrelated lineage markers in ``model_config``.

        When *provider* is given, it is merged into ``model_config``
        alongside the model (``$.model`` / ``$.provider``) so a later
        resume recombines the persisted model with the provider that
        actually serves it instead of the config.yaml primary provider
        (#79536). Callers without provider knowledge leave any stored
        provider untouched.
        """
        # This write bypasses the token queue, so deltas enqueued before the
        # switch must land first: a still-queued first delta carries the
        # pre-switch route, and applying it after this UPDATE would trip the
        # first_accounted_route overwrite in update_token_counts (row sees
        # api_call_count == 0 + a route mismatch) and resurrect the old
        # model/provider. Flushing here restores the pre-queue ordering.
        self.flush_token_counts()

        def _do(conn):
            # Use the shared merge discipline so lineage markers like
            # _branched_from / _delegate_from survive. browser_model_lock
            # is deleted via a None patch value (same semantics as the
            # old json_remove).
            patch: Dict[str, Any] = {"browser_model_lock": None}
            if model:
                patch["model"] = model
            if provider:
                patch["provider"] = provider
            merged = self._merge_model_config_json(conn, session_id, patch)
            if merged is _MODEL_CONFIG_ROW_MISSING:
                return
            conn.execute(
                "UPDATE sessions SET "
                "model = ?, model_config = ?, "
                "system_prompt = NULL, system_prompt_hash = NULL "
                "WHERE id = ?",
                (model, merged, session_id),
            )
            self._delete_unreferenced_system_prompts(conn)
        self._execute_write(_do)

    # ── KENSEI CUSTOM: persist agent mode across restarts ──
    def update_session_agent_mode(self, session_id: str, agent_mode: str) -> None:
        """Persist the agent mode (auto/plan/gods_plan/recon) to the session row."""
        def _do(conn):
            conn.execute(
                "UPDATE sessions SET agent_mode = ? WHERE id = ?",
                (agent_mode, session_id),
            )
        self._execute_write(_do)
    # ── END KENSEI CUSTOM ──

    def update_session_todo_state(
        self, session_id: str, state: Dict[str, Any]
    ) -> bool:
        """Persist one session's canonical in-progress task snapshot."""
        if not session_id or not isinstance(state, dict):
            return False
        payload = json.dumps(state, ensure_ascii=False, separators=(",", ":"))

        def _do(conn):
            cursor = conn.execute(
                "UPDATE sessions SET todo_state = ? WHERE id = ?",
                (payload, session_id),
            )
            return cursor.rowcount == 1

        return bool(self._execute_write(_do))

    def get_session_todo_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return a validated JSON object for the session task sidecar."""
        if not session_id:
            return None
        with self._read_ctx() as conn:
            if conn is None:
                return None
            row = conn.execute(
                "SELECT todo_state FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        raw = row["todo_state"] if isinstance(row, sqlite3.Row) else row[0]
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None
        return parsed if isinstance(parsed, dict) else None

    def _merge_model_config_json(
        self,
        conn,
        session_id: str,
        patch: Dict[str, Any],
        *,
        on_missing: str = "skip",
    ):
        """SELECT + tolerant-parse + merge ``patch`` into a session's model_config.

        Shared by every model_config writer (``update_session_runtime_lock``,
        ``set_session_yolo``, ``archive_and_compact``,
        ``patch_session_model_config``) so the merge discipline that keeps
        lineage markers like ``_branched_from`` / ``_delegate_from`` alive
        lives in exactly one place. A ``None`` patch value deletes that key.
        Must run inside an open write transaction (callers own the UPDATE).

        Returns the serialized merged JSON — ``None`` when the merged dict is
        empty (matching ``create_session``'s NULL convention) — or the
        ``_MODEL_CONFIG_ROW_MISSING`` sentinel when the row doesn't exist and
        ``on_missing == "skip"``; ``on_missing == "raise"`` raises ValueError.
        """
        row = conn.execute(
            "SELECT model_config FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            if on_missing == "raise":
                raise ValueError(f"Session not found: {session_id}")
            return _MODEL_CONFIG_ROW_MISSING
        raw = row["model_config"] if isinstance(row, sqlite3.Row) else row[0]
        config: Dict[str, Any] = {}
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    config = parsed
            except (json.JSONDecodeError, TypeError):
                config = {}
        elif isinstance(raw, dict):
            config = dict(raw)
        for key, value in patch.items():
            if value is None:
                config.pop(key, None)
            else:
                config[key] = value
        return json.dumps(config) if config else None

    def patch_session_model_config(
        self, session_id: str, patch: Dict[str, Any]
    ) -> None:
        """Merge ``patch`` into a session's model_config JSON atomically.

        A ``None`` patch value removes that key. No-op when the session row
        doesn't exist or the patch is empty. This is the standalone setter for
        callers that need to update model_config *without* rewriting the
        transcript (the transcript-coupled path is ``archive_and_compact``'s
        ``model_config_patch``, which shares the same merge helper).
        """
        if not session_id or not patch:
            return

        def _do(conn):
            merged = self._merge_model_config_json(conn, session_id, patch)
            if merged is _MODEL_CONFIG_ROW_MISSING:
                return
            conn.execute(
                "UPDATE sessions SET model_config = ? WHERE id = ?",
                (merged, session_id),
            )

        self._execute_write(_do)

    def get_session_model_config_value(
        self, session_id: str, key: str, default: Any = None
    ) -> Any:
        """Read one key out of a session's model_config JSON (tolerant parse)."""
        session = self.get_session(session_id) or {}
        raw = session.get("model_config")
        config: Dict[str, Any] = {}
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    config = parsed
            except (json.JSONDecodeError, TypeError):
                config = {}
        elif isinstance(raw, dict):
            config = raw
        return config.get(key, default)

    def update_session_runtime_lock(
        self,
        session_id: str,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        model_options: Optional[Dict[str, Any]] = None,
        route_source: Optional[str] = None,
        confirmed: bool = False,
    ) -> None:
        """Persist a Browser / API client runtime lock without clobbering lineage markers.

        Merges ``browser_model_lock`` into the existing ``model_config`` JSON so
        ``_branched_from`` / ``_delegate_from`` survive. Nulls ``system_prompt``
        so cached ``Model:`` / ``Provider:`` footers cannot lie after a switch.
        """
        lock = {
            "provider": provider or "",
            "model": model or "",
            "model_options": model_options or {},
            "route_source": route_source or "",
            "confirmed": bool(confirmed),
            "updated_at": time.time(),
        }

        def _do(conn):
            merged = self._merge_model_config_json(
                conn, session_id, {"browser_model_lock": lock}
            )
            if merged is _MODEL_CONFIG_ROW_MISSING:
                return
            conn.execute(
                """UPDATE sessions SET
                   model_config = ?,
                   model = COALESCE(?, model),
                   system_prompt = NULL,
                   system_prompt_hash = NULL
                   WHERE id = ?""",
                (merged, model, session_id),
            )
            self._delete_unreferenced_system_prompts(conn)
        self._execute_write(_do)

    def set_session_yolo(self, session_id: str, enabled: bool) -> None:
        """Persist the per-session YOLO bypass flag into ``model_config``.

        Merges ``yolo_mode`` into the existing ``model_config`` JSON (same
        merge discipline as ``update_session_runtime_lock`` so lineage
        markers like ``_branched_from`` / ``_delegate_from`` survive). The
        CLI resume paths read this flag back so a ``/yolo ON`` toggle — or a
        ``--yolo`` launch — survives ``hermes --resume`` into a fresh
        process. No-op when the session row doesn't exist yet; the
        creation-time ``model_config`` carries the flag for ``--yolo``
        launches.
        """
        if not session_id:
            return

        def _do(conn):
            merged = self._merge_model_config_json(
                conn, session_id, {"yolo_mode": bool(enabled)}
            )
            if merged is _MODEL_CONFIG_ROW_MISSING:
                return
            conn.execute(
                "UPDATE sessions SET model_config = ? WHERE id = ?",
                (merged, session_id),
            )
        self._execute_write(_do)

    @staticmethod
    def session_yolo_enabled(session_meta: Optional[Dict[str, Any]]) -> bool:
        """Read the persisted YOLO flag off a session row dict.

        Accepts the dict returned by ``get_session`` (``model_config`` is a
        JSON string) or an already-parsed dict. Returns False on any parse
        failure — resume must never enable the bypass by accident.
        """
        raw = (session_meta or {}).get("model_config")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                return False
        if not isinstance(raw, dict):
            return False
        return bool(raw.get("yolo_mode"))

    @staticmethod
    def session_gateway_runtime(session_meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Read the persisted runtime route off a session row dict.

        Accepts the dict returned by ``get_session`` (``model_config`` is a
        JSON string) or an already-parsed dict. Prefers the nested
        ``gateway_runtime`` key (written by the gateway's
        ``_sync_session_model_from_agent`` and the CLI ``/model`` persist),
        falling back to the top-level ``provider``/``base_url``/``api_mode``
        keys the TUI gateway's ``_runtime_model_config`` writes. As a last
        resort, falls back to the ``billing_provider`` column (written on
        every session's first accounted API call) so sessions that never ran
        ``/model`` still restore the provider that actually served them.
        Returns an empty dict on any parse failure — resume falls back to
        ambient config resolution.
        """
        raw = (session_meta or {}).get("model_config")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except Exception:
                raw = {}
        if not isinstance(raw, dict):
            raw = {}
        runtime = raw.get("gateway_runtime")
        if isinstance(runtime, dict) and runtime.get("provider"):
            # Filter None values: the persist path writes or-None to trigger
            # deletion in the top-level merge, but gateway_runtime is replaced
            # as a whole dict (not deep-merged), so None values survive here.
            return {k: v for k, v in runtime.items() if v is not None}
        top_level = {
            key: raw.get(key)
            for key in ("provider", "base_url", "api_mode")
            if raw.get(key)
        }
        if top_level:
            return top_level
        # Last resort: billing_provider column. Written via COALESCE on every
        # session's first accounted API call — the only durable record for
        # sessions that never ran /model. Mirrors the TUI gateway's
        # _stored_session_runtime_overrides fallback. Bare billing buckets
        # ("auto"/"custom") are not routable identities — filter them out so
        # resume falls back to the ambient config default instead.
        billing_provider = str(
            (session_meta or {}).get("billing_provider") or ""
        ).strip()
        if (
            billing_provider
            and billing_provider.lower() not in _BARE_BILLING_PROVIDERS
        ):
            return {"provider": billing_provider}
        return {k: v for k, v in (runtime or {}).items() if v is not None} if isinstance(runtime, dict) else {}

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
        # Barrier against queued token deltas — see update_session_model.
        self.flush_token_counts()

        def _do(conn):
            conn.execute(
                """UPDATE sessions SET
                   billing_provider = ?,
                   billing_base_url = ?,
                   billing_mode = COALESCE(?, billing_mode),
                   system_prompt = NULL,
                   system_prompt_hash = NULL
                   WHERE id = ?""",
                (provider, base_url, billing_mode, session_id),
            )
            self._delete_unreferenced_system_prompts(conn)
        self._execute_write(_do)

    # ── Async token accounting ──
    # update_token_counts() runs a sessions UPDATE (plus a per-model usage
    # upsert) inside BEGIN IMMEDIATE; against a cold multi-GB state.db one
    # call can stall the turn thread for tens to hundreds of ms, and the
    # tool loop pays it after EVERY API call (measured p50 3.3ms / p95 70ms
    # per call in production). queue_token_counts() reduces the critical
    # path to a deque append: a dedicated single-writer thread applies
    # deltas in enqueue order, coalescing consecutive same-route deltas
    # into one UPDATE when a backlog forms. Readers that need exact
    # mid-turn totals (get_session and friends) call flush_token_counts()
    # first — a plain attribute check when nothing is queued.

    # Delta fields summed when coalescing. Route fields must be equal for
    # two deltas to merge: model/billing_* feed COALESCE backfill and the
    # per-model usage attribution key, and cost_status/cost_source are
    # last-non-None-wins — equality makes the merged UPDATE byte-for-byte
    # equivalent to applying the deltas sequentially.
    _TOKEN_DELTA_SUM_FIELDS = (
        "input_tokens", "output_tokens", "cache_read_tokens",
        "cache_write_tokens", "reasoning_tokens", "api_call_count",
    )
    _TOKEN_DELTA_COST_FIELDS = ("estimated_cost_usd", "actual_cost_usd")
    _TOKEN_DELTA_ROUTE_FIELDS = (
        "model", "cost_status", "cost_source", "pricing_version",
        "billing_provider", "billing_base_url", "billing_mode",
    )

    def queue_token_counts(self, session_id: str, **kwargs) -> None:
        """Enqueue a token/cost delta for the background writer.

        Accepts the same keyword arguments as :meth:`update_token_counts`
        and applies them asynchronously with identical semantics.  Cheap
        (append + notify) — safe to call on the turn thread after every
        API call.  After close() has stopped the writer, falls back to the
        synchronous path and may raise like :meth:`update_token_counts`.
        """
        with self._token_queue_cond:
            thread = self._token_writer_thread
            writer_stopped = self._token_writer_stop and (
                thread is None or not thread.is_alive()
            )
            if not writer_stopped:
                self._token_queue.append((session_id, kwargs))
                if thread is None or not thread.is_alive():
                    # Daemon so process exit never hangs on accounting; the
                    # atexit hook drains anything still queued at interpreter
                    # shutdown (registered once per instance, on first use).
                    # ``not is_alive()`` (rather than ``is None`` only)
                    # respawns the writer if it ever died from an unexpected
                    # escape — otherwise a dead thread object would block
                    # respawn forever and deltas would pile up on the deque
                    # until a reader's flush drained them synchronously.
                    thread = threading.Thread(
                        target=self._token_writer_loop,
                        name="session-db-token-writer",
                        daemon=True,
                    )
                    self._token_writer_thread = thread
                    thread.start()
                    if self._token_atexit_hook is None:
                        self_ref = weakref.ref(self)

                        def _drain_at_exit() -> None:
                            db = self_ref()
                            if db is not None:
                                db._drain_token_queue_at_exit()

                        self._token_atexit_hook = _drain_at_exit
                        atexit.register(_drain_at_exit)
                self._token_queue_cond.notify_all()
        if writer_stopped:
            # Writer permanently stopped (close() ran; a stop-flagged but
            # still-live writer keeps accepting — its loop drains before
            # exiting). Enqueueing now would drop the delta silently: no
            # writer will run and close() already unregistered the atexit
            # hook. Apply inline instead so a closed-connection failure
            # raises at the call site, exactly like the old synchronous
            # update_token_counts path these call sites still guard for.
            self.update_token_counts(session_id, **kwargs)

    def flush_token_counts(self, timeout: float = 5.0) -> bool:
        """Block until every queued token delta has been applied.

        Returns True when the queue is fully drained, False on timeout
        (callers then read totals that are stale by the still-queued
        deltas — no worse than reading before the flush existed).
        Never raises: apply failures are logged by the writer.
        """
        # Fast path — nothing queued, nothing in flight.
        if not self._token_queue and not self._token_writer_busy:
            return True
        batch = None
        with self._token_queue_cond:
            deadline = time.monotonic() + timeout
            while self._token_queue or self._token_writer_busy:
                # A live writer is authoritative even when stop-flagged
                # (close() in progress): its loop drains the queue before
                # exiting, and draining here instead would race its
                # in-flight batch — newer deltas committing before older
                # ones breaks the last-non-None-wins / first-accounted-
                # route / COALESCE-backfill fields. Only when the writer is
                # dead (or never started for these deltas) does the caller
                # take the leftovers. Re-checked each wakeup: the writer
                # can exit mid-wait with deltas enqueued after its final
                # empty-queue check. busy is claimed while draining (same
                # protocol as the writer) so a concurrent flush cannot
                # report drained — or pop a newer delta — while this batch
                # is still unapplied; a claimed busy therefore also means
                # "wait", never "drain alongside".
                thread = self._token_writer_thread
                if (
                    (thread is None or not thread.is_alive())
                    and not self._token_writer_busy
                ):
                    self._token_writer_busy = True
                    batch = list(self._token_queue)
                    self._token_queue.clear()
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._token_queue_cond.wait(remaining)
        if batch:
            try:
                self._apply_token_batch(batch)
            finally:
                with self._token_queue_cond:
                    self._token_writer_busy = False
                    self._token_queue_cond.notify_all()
        return True

    def _token_writer_loop(self) -> None:
        while True:
            with self._token_queue_cond:
                idle_deadline = time.monotonic() + self._TOKEN_WRITER_IDLE_SECONDS
                while not self._token_queue and not self._token_writer_stop:
                    remaining = idle_deadline - time.monotonic()
                    if remaining <= 0:
                        # Publish retirement under the same lock used by
                        # queue_token_counts() to decide whether to spawn. An
                        # enqueue cannot strand a delta behind an exiting worker.
                        self._token_writer_thread = None
                        return
                    self._token_queue_cond.wait(remaining)
                if not self._token_queue:
                    self._token_writer_thread = None
                    return  # stop requested and fully drained
                # busy is set BEFORE the queue is cleared: the lock-free
                # fast path in flush_token_counts() reads queue-then-busy,
                # so this order guarantees it can never observe an empty
                # queue while the popped batch is still unapplied.
                self._token_writer_busy = True
                batch = list(self._token_queue)
                self._token_queue.clear()
            try:
                self._apply_token_batch(batch)
            finally:
                with self._token_queue_cond:
                    self._token_writer_busy = False
                    self._token_queue_cond.notify_all()

    def _apply_token_batch(self, batch: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Apply queued deltas in order, coalescing where safe. Never raises."""
        try:
            coalesced = self._coalesce_token_deltas(batch)
        except Exception as exc:
            # Coalescing must never kill the writer thread (a dead writer
            # can't be observed by callers). Fall back to applying the raw
            # batch delta-by-delta — the merge is an optimization only.
            logger.warning(
                "async token accounting: coalesce failed, applying raw "
                "batch: %s", exc,
            )
            coalesced = batch
        for session_id, kwargs in coalesced:
            try:
                self.update_token_counts(session_id, **kwargs)
            except Exception as exc:
                # Same contract as the old inline call sites: accounting
                # loss is logged, never raised into a turn.
                logger.warning(
                    "async token accounting: apply failed (session=%s): %s",
                    session_id, exc,
                )

    def _coalesce_token_deltas(
        self, batch: List[Tuple[str, Dict[str, Any]]]
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Merge consecutive incremental deltas with an identical route.

        Only adjacent deltas merge, so ordering across sessions and across
        a mid-session /model switch is preserved exactly.  absolute=True
        deltas (cumulative overwrites) never merge.
        """
        groups: List[Tuple[Optional[tuple], str, Dict[str, Any]]] = []
        for session_id, kwargs in batch:
            key = None
            if not kwargs.get("absolute"):
                key = (session_id,) + tuple(
                    kwargs.get(f) for f in self._TOKEN_DELTA_ROUTE_FIELDS
                )
            if groups and key is not None and groups[-1][0] == key:
                merged = groups[-1][2]
                for f in self._TOKEN_DELTA_SUM_FIELDS:
                    merged[f] = merged.get(f, 0) + kwargs.get(f, 0)
                for f in self._TOKEN_DELTA_COST_FIELDS:
                    value = kwargs.get(f)
                    if value is not None:
                        # None-preserving sum: an all-None run must stay
                        # None so COALESCE keeps the stored value untouched.
                        merged[f] = (merged.get(f) or 0.0) + value
            else:
                groups.append((key, session_id, dict(kwargs)))
        return [(sid, kw) for _, sid, kw in groups]

    def _stop_token_writer(self, join_timeout: float = 10.0) -> None:
        """Stop the writer thread and drain remaining deltas. Never raises."""
        with self._token_queue_cond:
            self._token_writer_stop = True
            self._token_queue_cond.notify_all()
            thread = self._token_writer_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                # Writer stuck mid-apply (pathological lock contention).
                # Leave any queued deltas unapplied rather than racing the
                # stuck apply and misordering/double-counting.
                logger.warning(
                    "async token accounting: writer did not stop within %.0fs; "
                    "%d queued delta(s) not persisted",
                    join_timeout, len(self._token_queue),
                )
                return
        # Writer exited (or never started) — apply leftovers synchronously.
        # Claim busy like the writer/flush drains do, so a concurrent
        # flush_token_counts cannot fast-path True while this batch is
        # still being applied; conversely, wait out a flush caller-drain
        # that already claimed busy — close() nulls the connection right
        # after this returns, and must not yank it mid-batch.
        with self._token_queue_cond:
            deadline = time.monotonic() + join_timeout
            while self._token_writer_busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "async token accounting: concurrent drain did not "
                        "finish within %.0fs; %d queued delta(s) not persisted",
                        join_timeout, len(self._token_queue),
                    )
                    return
                self._token_queue_cond.wait(remaining)
            # busy is claimed BEFORE the queue is cleared — same ordering
            # as the writer loop and the flush caller-drain. The lock-free
            # fast path in flush_token_counts() reads queue-then-busy
            # without the cond, so clearing first would let a concurrent
            # flush observe "empty and idle" and return True while this
            # popped batch is still unapplied.
            batch = list(self._token_queue)
            if batch:
                self._token_writer_busy = True
                self._token_queue.clear()
        if batch:
            try:
                self._apply_token_batch(batch)
            finally:
                with self._token_queue_cond:
                    self._token_writer_busy = False
                    self._token_queue_cond.notify_all()

    def _drain_token_queue_at_exit(self) -> None:
        try:
            self._stop_token_writer()
        except Exception:
            pass  # Best effort — never fatal at interpreter shutdown.

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
        api_call_count: int = 1,
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

        ``api_call_count`` defaults to 1 (one aux LLM call). Background-review
        forks record an aggregate of N fork API calls in one write with
        ``task='background_review'`` (issue #87250).

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
                api_call_count=(
                    1 if api_call_count is None else int(api_call_count)
                ),
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
                self._delete_unreferenced_system_prompts(conn)
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

    def sweep_orphaned_sessions(
        self,
        *,
        max_idle_seconds: float,
        sources: Tuple[str, ...] = ("tui", "desktop", "subagent"),
        exclude_ids: Tuple[str, ...] = (),
        exclude_pinned: bool = False,
        heartbeat_staleness_seconds: Optional[float] = None,
        heartbeat_ownership_grace_seconds: Optional[float] = None,
        respect_gateway_heartbeats: bool = True,
    ) -> List[str]:
        """Close session rows orphaned by a dead gateway process (#65194, #94895).

        The TUI/desktop gateway reaps disconnected websocket sessions with an
        in-process ``threading.Timer`` grace timer; a gateway restart destroys
        the timer and leaves the row ``ended_at IS NULL`` forever. This is the
        startup-time complement: it closes rows for the given ``sources`` whose
        ``started_at`` and canonical last-activity time are both older than
        ``max_idle_seconds``, with a distinct
        ``end_reason='startup_orphan_reap'`` for traceability.

        Canonical activity is the newest of ``last_activity_at`` (the in-turn
        heartbeat) and the newest durable message timestamp, falling back to
        ``started_at``. The separate ``started_at`` predicate protects freshly
        created compression/branch children whose copied activity is old.

        Only pass sources whose lifecycle the caller owns (never messaging-gateway
        platforms like ``telegram`` — ending those triggers the #60609 routing
        loop). ``exclude_ids`` spares rows this process still holds in memory
        (a ``session.resume`` that landed during the startup grace window).
        ``exclude_pinned`` is intended for broad automatic sweeps; pinned rows
        remain explicitly recoverable. Non-destructive: messages are preserved
        and the row remains resumable. First-reason-wins is preserved via
        ``ended_at IS NULL``.

        Cross-backend liveness (#94895): when one ``state.db`` is shared by N
        serve / gateway processes, each backend refreshes a row in
        ``gateway_heartbeats``. With ``respect_gateway_heartbeats`` enabled, a
        row is only reaped when activity staleness holds AND no live backend
        (heartbeat refreshed within ``heartbeat_staleness_seconds``, default
        ``2 * max_idle_seconds``) could plausibly own it. Disable that gate only
        for sources whose lifecycle is explicitly owned by state.db itself.

        Ownership inference: a live backend B ``owns`` a session S if
        ``B.started_at <= S.started_at + heartbeat_ownership_grace_seconds``
        (default ``heartbeat_staleness_seconds``). The grace window covers a
        migrating backend whose existing sessions predate its first heartbeat,
        but is bounded so a fresh PID-reuse respawn cannot protect rows forever.
        With no fresh heartbeat the predicate falls back to the legacy sweep.

        The SELECT, live-lease validation, and UPDATE run in one
        ``BEGIN IMMEDIATE`` transaction. Active turn leases or compression
        locks spare the row; expired/reclaimed guards are removed so their
        former owner is fenced. Returns the swept session ids.
        """
        srcs = tuple(s for s in sources if s)
        if max_idle_seconds <= 0 or not srcs:
            return []
        hb_staleness = (
            heartbeat_staleness_seconds
            if heartbeat_staleness_seconds and heartbeat_staleness_seconds > 0
            else max_idle_seconds * 2
        )
        hb_grace = (
            heartbeat_ownership_grace_seconds
            if heartbeat_ownership_grace_seconds is not None
            and heartbeat_ownership_grace_seconds >= 0
            else hb_staleness
        )
        now = time.time()
        cutoff = now - max_idle_seconds
        hb_cutoff = now - hb_staleness
        placeholders = ",".join("?" for _ in srcs)
        staleness = (
            f"started_at < ? AND {_sql_session_last_active('sessions')} < ?"
        )
        pin_scope = " AND COALESCE(pinned, 0) = 0" if exclude_pinned else ""
        heartbeat_params: Tuple[float, ...] = ()
        orphan_predicate = staleness
        if respect_gateway_heartbeats:
            orphan_predicate += (
                " AND NOT EXISTS ("
                "SELECT 1 FROM gateway_heartbeats h"
                " WHERE h.last_heartbeat >= ?"
                " AND h.started_at <= sessions.started_at + ?"
                ")"
            )
            heartbeat_params = (hb_cutoff, hb_grace)

        def _do(conn):
            rows = conn.execute(
                f"SELECT id FROM sessions WHERE ended_at IS NULL"
                f" AND source IN ({placeholders}){pin_scope}"
                f" AND {orphan_predicate}",
                (*srcs, cutoff, cutoff, *heartbeat_params),
            ).fetchall()
            excluded = {str(x) for x in exclude_ids if x}
            victims = []
            for row in rows:
                sid = str(row["id"])
                if sid in excluded:
                    continue
                try:
                    self._check_transcript_write_guards(
                        conn,
                        sid,
                        compression_lock_holder=None,
                        turn_lease_holder=None,
                        reject_active_turn_lease=True,
                        reject_active_compression_lock=True,
                    )
                except (
                    SessionCompressionInProgressError,
                    SessionTurnLeaseLostError,
                ):
                    continue
                victims.append(sid)
            if not victims:
                return []
            closed_at = time.time()
            marks = ",".join("?" for _ in victims)
            # Re-apply every scope/liveness predicate under the write lock.
            conn.execute(
                f"UPDATE sessions SET ended_at = ?, end_reason = 'startup_orphan_reap'"
                f" WHERE id IN ({marks}) AND ended_at IS NULL"
                f" AND source IN ({placeholders}){pin_scope}"
                f" AND {orphan_predicate}",
                (
                    closed_at,
                    *victims,
                    *srcs,
                    cutoff,
                    cutoff,
                    *heartbeat_params,
                ),
            )
            return victims

        return self._execute_write(_do) or []

    # ── Cross-backend heartbeat API (#94895) ───────────────────────────
    # Each serve / tui_gateway process registers a heartbeat row at startup
    # and refreshes ``last_heartbeat`` periodically. The startup orphan
    # sweep reads these rows to avoid reaping sessions owned by another
    # still-live backend that just happens to be idle. Backends remove
    # their own row on graceful shutdown; a row that survives a crash is
    # reclaimed by the staleness sweep once ``last_heartbeat`` ages out.

    def register_backend_heartbeat(
        self,
        *,
        backend_id: str,
        pid: int,
        started_at: float,
        last_heartbeat: Optional[float] = None,
        profile: str = "",
        host: str = "",
    ) -> None:
        """Upsert this backend's liveness row (#94895).

        ``backend_id`` MUST be stable for the lifetime of the process
        (e.g. ``f"{profile}@{host}:{pid}"``) so a respawn cannot accidentally
        inherit the dead predecessor's heartbeat and protect stale rows.
        ``started_at`` records when THIS process started (not the wall clock
        at first refresh) so a long-lived backend whose previous run died
        cannot be confused with a freshly-spawned sibling.
        """
        if not backend_id:
            return
        ts = time.time() if last_heartbeat is None else float(last_heartbeat)
        def _do(conn):
            conn.execute(
                "INSERT INTO gateway_heartbeats"
                " (backend_id, pid, started_at, last_heartbeat, profile, host)"
                " VALUES (?, ?, ?, ?, ?, ?)"
                " ON CONFLICT(backend_id) DO UPDATE SET"
                " pid = excluded.pid,"
                " started_at = excluded.started_at,"
                " last_heartbeat = excluded.last_heartbeat,"
                " profile = excluded.profile,"
                " host = excluded.host",
                (str(backend_id), int(pid), float(started_at), ts,
                 str(profile), str(host)),
            )
        self._execute_write(_do)

    def clear_backend_heartbeat(self, backend_id: str) -> bool:
        """Remove this backend's heartbeat row (#94895).

        Called from ``atexit`` so a graceful shutdown doesn't leave a stale
        row behind. A crashed backend's row is reclaimed later by
        ``prune_stale_heartbeats``. Returns True if a row was removed.
        """
        if not backend_id:
            return False
        def _do(conn):
            cur = conn.execute(
                "DELETE FROM gateway_heartbeats WHERE backend_id = ?",
                (str(backend_id),),
            )
            return cur.rowcount > 0
        return bool(self._execute_write(_do))

    def prune_stale_heartbeats(self, *, max_age_seconds: float) -> List[str]:
        """Drop heartbeat rows whose ``last_heartbeat`` is older than the
        staleness window. Returns the removed backend ids. Safe to call
        from any process; only stale rows are touched.
        """
        if max_age_seconds <= 0:
            return []
        cutoff = time.time() - max_age_seconds
        def _do(conn):
            cur = conn.execute(
                "DELETE FROM gateway_heartbeats WHERE last_heartbeat < ?"
                " RETURNING backend_id",
                (cutoff,),
            )
            return [str(r[0]) for r in cur.fetchall()]
        return list(self._execute_write(_do) or [])

    def list_backend_heartbeats(self) -> List[Dict[str, Any]]:
        """Snapshot of every registered backend's heartbeat (for diagnostics
        and tests). The fields mirror ``gateway_heartbeats`` exactly.
        """
        with self._read_ctx() as conn:
            rows = conn.execute(
                "SELECT backend_id, pid, started_at, last_heartbeat,"
                " profile, host FROM gateway_heartbeats"
                " ORDER BY last_heartbeat DESC"
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for r in rows:
            if isinstance(r, sqlite3.Row):
                out.append({k: r[k] for k in r.keys()})
            else:
                out.append({
                    "backend_id": r[0], "pid": r[1], "started_at": r[2],
                    "last_heartbeat": r[3], "profile": r[4], "host": r[5],
                })
        return out

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by ID."""
        # Cost/usage readers (/status, /usage, gateway endpoints) reach the
        # row through here; drain queued token deltas so they see exact
        # totals. No-op attribute check when nothing is queued.
        self.flush_token_counts()
        with self._read_ctx() as conn:
            cursor = conn.execute(
                "SELECT s.*, "
                "COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved "
                "FROM sessions s "
                "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
                "WHERE s.id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
        return self._session_row_dict(row) if row else None

    def get_dominant_session_model_route(
        self, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return the main-loop model route that served most API calls.

        ``sessions`` is a legacy aggregate row and can hold model/provider fields
        written by different route changes. ``session_model_usage`` keeps the
        coherent per-call tuple, so persisted status and billing reads should use
        its dominant main-loop route when one is available.
        """
        self.flush_token_counts()
        with self._read_ctx() as conn:
            row = conn.execute(
                """SELECT model, billing_provider, billing_base_url, billing_mode,
                          api_call_count
                     FROM session_model_usage
                    WHERE session_id = ?
                      AND task = ''
                      AND model <> 'unknown'
                      AND billing_provider <> ''
                    ORDER BY api_call_count DESC,
                             (input_tokens + output_tokens + cache_read_tokens +
                              cache_write_tokens + reasoning_tokens) DESC,
                             last_seen DESC
                    LIMIT 1""",
                (session_id,),
            ).fetchone()
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

        escaped = _escape_like(session_id_or_prefix)
        with self._read_ctx() as conn:
            cursor = conn.execute(
                "SELECT id FROM sessions WHERE id LIKE ? ESCAPE '\\' ORDER BY started_at DESC LIMIT 2",
                (f"{escaped}%",),
            )
            matches = [row["id"] for row in cursor.fetchall()]
        if len(matches) == 1:
            return matches[0]
        return None

    # Maximum length for session titles
    MAX_TITLE_LENGTH = 100

    # Title provenance, lowest to highest authority. An auto-titling write may
    # only replace a title of strictly lower authority, so the instant
    # ``derived`` title upgrades to the model's ``llm`` title exactly once and
    # nothing the agent generates can ever clobber a name the user typed.
    TITLE_SOURCE_DERIVED = "derived"
    TITLE_SOURCE_LLM = "llm"
    TITLE_SOURCE_USER = "user"
    _TITLE_SOURCE_RANK = {
        TITLE_SOURCE_DERIVED: 0,
        TITLE_SOURCE_LLM: 1,
        TITLE_SOURCE_USER: 2,
    }

    # Bot Mode's forever-chat registry: the session titled exactly this, on a
    # bot's profile, IS the bot's canonical chat — resolved by exact-title
    # lookup on every open (no session-id pointer exists). The title is the
    # identity, which is why _set_session_title refuses renames of a hidden
    # row holding it (#92473).
    CANONICAL_BOT_CHAT_TITLE = "Bot Chat"

    @classmethod
    def _title_rank(cls, source: Optional[str]) -> int:
        """Rank a stored title_source. NULL means a pre-provenance row.

        Rows written before this column existed carry NULL. They were almost
        always set by the old auto-titler, but a manual ``/title`` from that
        era is indistinguishable — so treat NULL as ``user`` and refuse to
        overwrite it. Auto-titling only ever fills genuinely empty titles on
        legacy rows, which is the conservative direction.
        """
        if source is None:
            return cls._TITLE_SOURCE_RANK[cls.TITLE_SOURCE_USER]
        return cls._TITLE_SOURCE_RANK.get(str(source), 0)

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
        source: str,
    ) -> bool:
        """Write a title, enforcing provenance precedence.

        ``source`` is one of ``TITLE_SOURCE_{DERIVED,LLM,USER}``. A ``user``
        write always lands — an explicit rename is authoritative. An automatic
        write (``derived``/``llm``) lands only when the row is untitled or the
        stored title has strictly lower authority, so the instant ``derived``
        title upgrades to ``llm`` exactly once and neither can ever overwrite a
        name the user typed. Re-running the titler on an already-``llm`` row is
        a no-op, which is what stops a session renaming itself. The one thing
        no writer may do is move a hidden canonical Bot Chat off its title.

        The read and the write are one compare-and-swap inside a single
        transaction, so a manual ``/title`` racing an in-flight generation
        cannot be clobbered by the late arrival.
        """
        title = self.sanitize_title(title)
        is_user = source == self.TITLE_SOURCE_USER
        new_rank = self._title_rank(source) if not is_user else None

        def _do(conn):
            current = conn.execute(
                "SELECT title, title_source, hidden FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if current is None:
                return 0
            # The canonical Bot Chat's NAME is its identity: Bot Mode resolves
            # the forever-chat by exact-title lookup on every open, so renaming
            # the row orphans the entire conversation — the next click mints an
            # empty replacement and UNIQUE(title) then blocks ever renaming
            # back (#92473). Refuse the rename at the single write path every
            # surface funnels through (gateway session.title, /title, CLI
            # rename, REST). Hidden is the discriminator: canonical chats are
            # born hidden; an ordinary visible session a user happens to call
            # "Bot Chat" stays freely renameable. Provenance-blind: an
            # automatic llm write outranks a derived title, so the auto-titler
            # would otherwise rename the row too (#99517) — it no-ops instead.
            if (
                (current["title"] or "") == self.CANONICAL_BOT_CHAT_TITLE
                and bool(current["hidden"])
                and title != self.CANONICAL_BOT_CHAT_TITLE
            ):
                if is_user:
                    raise ValueError(
                        "This is the bot's canonical Bot Chat — its name is its "
                        "identity, and renaming it would orphan the conversation. "
                        "To start fresh, create a new bot instead."
                    )
                return 0
            if not is_user and current["title"] is not None:
                if self._title_rank(current["title_source"]) >= new_rank:
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
            # Compare-and-swap on the exact values we just read (``IS`` is
            # NULL-safe in SQLite), so a concurrent write between the SELECT
            # and here loses instead of being silently overwritten.
            cursor = conn.execute(
                "UPDATE sessions SET title = ?, title_source = ? "
                "WHERE id = ? AND title IS ? AND title_source IS ?",
                (
                    title,
                    source if title else None,
                    session_id,
                    current["title"],
                    current["title_source"],
                ),
            )
            return cursor.rowcount

        rowcount = self._execute_write(_do)
        return rowcount > 0

    def set_session_title(self, session_id: str, title: str) -> bool:
        """Set or update a session's title on the user's behalf.

        Returns True if session was found and title was set.
        Raises ValueError if title is already in use by another session,
        or if the title fails validation (too long, invalid characters).
        Empty/whitespace-only strings are normalized to None (clearing the title).

        This records ``user`` provenance, so auto-titling will never replace
        the result. Automatic callers must use :meth:`set_auto_title`.
        """
        return self._set_session_title(
            session_id, title, source=self.TITLE_SOURCE_USER
        )

    def set_auto_title(self, session_id: str, title: str, *, source: str) -> bool:
        """Set an automatically generated title, honoring provenance precedence.

        Returns True when the title was written, False when a higher-authority
        title already holds the row (nothing is modified in that case).
        """
        if source not in (self.TITLE_SOURCE_DERIVED, self.TITLE_SOURCE_LLM):
            raise ValueError(f"invalid automatic title source: {source!r}")
        return self._set_session_title(session_id, title, source=source)

    def set_auto_title_if_empty(self, session_id: str, title: str) -> bool:
        """Back-compat shim: set an LLM title only if nothing better exists.

        Retained because older callers (and third-party plugins) reference it
        by name. New code should call :meth:`set_auto_title` with an explicit
        source.
        """
        return self.set_auto_title(
            session_id, title, source=self.TITLE_SOURCE_LLM
        )

    def get_session_title(self, session_id: str) -> Optional[str]:
        """Get the title for a session, or None."""
        with self._read_ctx() as conn:
            cursor = conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
        return row["title"] if row else None

    def get_session_title_source(self, session_id: str) -> Optional[str]:
        """Get the provenance of a session's title, or None when untitled."""
        with self._read_ctx() as conn:
            cursor = conn.execute(
                "SELECT title, title_source FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
        if not row or row["title"] is None:
            return None
        return row["title_source"]

    def set_session_title_source(self, session_id: str, source: str) -> bool:
        """Overwrite a title's provenance without touching the title text.

        Used when a title is carried across a session boundary (compression
        rotation) and the copy must keep the original's authority rather than
        the authority of whichever setter performed the copy.
        """
        if source not in self._TITLE_SOURCE_RANK:
            raise ValueError(f"invalid title source: {source!r}")

        def _do(conn):
            cursor = conn.execute(
                "UPDATE sessions SET title_source = ? "
                "WHERE id = ? AND title IS NOT NULL",
                (source, session_id),
            )
            return cursor.rowcount

        return self._execute_write(_do) > 0

    def backfill_null_session_profiles(self, profile_name: str) -> int:
        """One-shot owner backfill for legacy pre-ownership session rows.

        Sessions created before the durable-ownership work (#95407 lineage)
        carry ``profile_name = NULL``. On single-backend installs that was
        harmless, but once a Desktop registers a second connection the
        fail-closed owner ladder (which is correct for new sessions) can no
        longer route those rows anywhere — every pre-campaign session becomes
        unresumable after upgrade (#94724, field report).

        This store belongs to exactly one profile — the profile whose
        ``state.db`` this is — so stamping its own name onto rows that never
        recorded one is a single-match backfill, not a guess. Rules mirror the
        ``create_session`` COALESCE contract:

        * only ``NULL``/empty ``profile_name`` rows are touched — a non-NULL
          owner is NEVER overwritten;
        * idempotent and one-shot-per-row: a second run matches zero rows.

        Returns the number of rows stamped (0 when nothing was legacy).
        """
        stamp = (profile_name or "").strip()
        if not stamp:
            return 0

        def _do(conn):
            cursor = conn.execute(
                """UPDATE sessions
                   SET profile_name = ?
                 WHERE profile_name IS NULL OR TRIM(profile_name) = ''""",
                (stamp,),
            )
            rowcount = cursor.rowcount
            if rowcount is None or rowcount < 0:
                rowcount = conn.execute("SELECT changes()").fetchone()[0]
            return rowcount

        return int(self._execute_write(_do) or 0)

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

    # Accidental end reasons that recovery treats as resumable. Single source
    # of truth: hermes_state_common._RECOVERABLE_END_REASONS, interpolated
    # into find_latest_gateway_session_for_peer / promote_to_session_reset
    # SQL — literals cannot drift (docs/session-lifecycle.md "recoverable
    # accidental reasons").
    RECOVERABLE_END_REASONS = _RECOVERABLE_END_REASONS

    def unarchive_recoverable_session(self, session_id: str) -> bool:
        """Un-archive a session that was archived by a recoverable accident.

        Registry-style lookups (Bot Mode's canonical "Bot Chat") use this to
        resurrect a row the ws-orphan reaper (``ws_orphan_reap``) or older
        agent cleanup (``agent_close``) archived: those ends are accidents,
        not user intent, so the identity-scoped canonical chat must survive
        them (#92687). Sessions archived with no end_reason or an explicit
        boundary reason (user archived deliberately, ``session_reset``, …)
        are left untouched — returns ``False`` for those, ``True`` only when
        the row was archived for a recoverable reason and is now un-archived
        (whole compression lineage, via :meth:`set_session_archived`).
        """
        if not session_id:
            return False
        try:
            row = self.get_session(session_id)
        except Exception:
            return False
        if not row or not row.get("archived"):
            return False
        # A compressed lineage's registry row keeps end_reason='compression';
        # the accidental stamp lives on the live TIP. Judge recoverability at
        # the tip (== the row itself when uncompressed).
        tip = row
        try:
            tip_id = self.get_compression_tip(session_id) or session_id
            if tip_id != session_id:
                tip = self.get_session(tip_id) or row
        except Exception:
            tip_id = session_id
        if (tip.get("end_reason") or "") not in self.RECOVERABLE_END_REASONS:
            return False
        if not self.set_session_archived(session_id, False):
            return False

        # Clear the accidental end stamp: the session is live again, and a
        # surviving ws_orphan_reap/agent_close reason would make a LATER
        # deliberate archive (which never writes end_reason) auto-resurrect
        # on the next lookup — permanently overriding user intent.
        def _clear_end(conn):
            conn.execute(
                "UPDATE sessions SET ended_at = NULL, end_reason = NULL WHERE id = ?",
                (tip["id"],),
            )
            return 1

        self._execute_write(_clear_end)
        return True

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

    def set_session_hidden(self, session_id: str, hidden: bool) -> bool:
        """Hide or unhide a session (and its whole compression lineage).

        ``hidden`` is a generic "don't show in the global Sessions sidebar"
        flag: a hidden session is dropped from the default
        :meth:`list_sessions_rich` listing (which omits ``include_hidden``) but
        stays fully resumable by the surface that owns it — useful for plugins
        that manage their own sessions (e.g. kanban) and don't want them
        cluttering the shared recents list. Like :meth:`set_session_archived`
        / :meth:`set_session_pinned` the whole compression chain is flipped as
        a unit, so hiding the surfaced tip hides the root (and vice-versa) no
        matter which id the caller holds. Returns True when at least one row
        changed.
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
                SET hidden = ?
                WHERE id IN (SELECT id FROM lineage)
                """,
                (session_id, session_id, 1 if hidden else 0),
            )
            rowcount = cursor.rowcount
            if rowcount is None or rowcount < 0:
                rowcount = conn.execute("SELECT changes()").fetchone()[0]
            return rowcount
        rowcount = self._execute_write(_do)
        return rowcount > 0

    def set_session_read(self, session_id: str, read: bool = True) -> bool:
        """Mark a session read or unread (and its whole compression lineage).

        Read state is a watermark, not a flag: ``last_read_at`` records when
        the conversation was last read, and it counts as unread when activity
        postdates that watermark (the derived ``unread`` key on
        :meth:`list_sessions_rich` rows). New messages therefore flip a read
        conversation back to unread without any write on the message path.
        Three states:

        * NULL — never tracked (every pre-feature row): treated as read, so
          shipping the column doesn't badge a user's entire history at once.
        * 0 — explicitly marked unread: any activity postdates it.
        * timestamp — read up to that moment.

        Like :meth:`set_session_archived` / :meth:`set_session_pinned`, the
        whole compression chain is stamped as a unit, so reading the surfaced
        tip clears the root (and vice-versa) no matter which id the caller
        holds. Returns True when at least one row changed.
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
                SET last_read_at = ?
                WHERE id IN (SELECT id FROM lineage)
                """,
                (session_id, session_id, time.time() if read else 0.0),
            )
            rowcount = cursor.rowcount
            if rowcount is None or rowcount < 0:
                rowcount = conn.execute("SELECT changes()").fetchone()[0]
            return rowcount
        rowcount = self._execute_write(_do)
        return rowcount > 0

    @staticmethod
    def session_unread(session_row: Dict[str, Any]) -> bool:
        """Derive unread from a session row's watermark and activity.

        Shared by ``list_sessions_rich`` and any future surface that holds a
        row (or projected row) with ``last_read_at`` and ``last_active``.
        NULL watermark = never tracked = read.
        """
        last_read = session_row.get("last_read_at")
        if last_read is None:
            return False
        last_active = session_row.get("last_active") or session_row.get("started_at")
        return float(last_active or 0) > float(last_read)

    def get_session_by_title(self, title: str) -> Optional[Dict[str, Any]]:
        """Look up a session by exact title. Returns session dict or None."""
        with self._read_ctx() as conn:
            cursor = conn.execute(
                "SELECT s.*, "
                "COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved "
                "FROM sessions s "
                "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
                "WHERE s.title = ?",
                (title,),
            )
            row = cursor.fetchone()
        return self._session_row_dict(row) if row else None

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
        escaped = _escape_like(title)
        with self._read_ctx() as conn:
            cursor = conn.execute(
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
        escaped = _escape_like(base)
        with self._read_ctx() as conn:
            cursor = conn.execute(
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

    def get_compression_chain(self, session_id: str) -> List[str]:
        """Walk the compression-continuation chain forward and return every id.

        Root-first order, ending at the tip; ``[session_id]`` when no
        continuation exists. ``get_compression_tip`` is this walk's last
        element — kept as the single implementation so the two can never
        disagree about what the chain is.

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
        chain = [current] if current else []
        seen = {current} if current else set()
        # Bound the walk defensively — compression chains this deep are
        # pathological and shouldn't happen in practice. 100 = plenty.
        for _ in range(100):
            with self._read_ctx() as conn:
                cursor = conn.execute(
                    f"""
                    SELECT child.id
                    FROM sessions parent
                    JOIN sessions child ON child.parent_session_id = parent.id
                    WHERE parent.id = ?
                      AND parent.end_reason = 'compression'
                      AND json_extract(COALESCE(child.model_config, '{{}}'), '$._branched_from') IS NULL
                      AND json_extract(COALESCE(child.model_config, '{{}}'), '$._delegate_from') IS NULL
                      AND COALESCE(child.source, '') != 'tool'
                    ORDER BY
                      CASE
                        WHEN child.end_reason = 'compression' THEN 0
                        WHEN child.ended_at IS NULL THEN 1
                        ELSE 2
                      END,
                      {_sql_session_last_active("child")} DESC,
                      child.started_at DESC,
                      child.id DESC
                    LIMIT 1
                    """,
                    (current,),
                )
                row = cursor.fetchone()
            if row is None:
                return chain
            child_id = row["id"]
            if not child_id or child_id in seen:
                return chain
            seen.add(child_id)
            current = child_id
            chain.append(child_id)
        return chain

    def get_compression_tip(self, session_id: str) -> Optional[str]:
        """The live tip of a compression-continuation chain (see
        ``get_compression_chain`` for the walk's semantics). Returns the input
        id when no continuation exists."""
        chain = self.get_compression_chain(session_id)
        return chain[-1] if chain else session_id

    # Columns excluded from compact_rows projections: only the payload-heavy
    # blob no list consumer renders. Everything else — including gateway
    # routing fields and desktop sidebar fields like git_branch — stays, and
    # the projection is derived from SCHEMA_SQL so columns added later via
    # declarative reconciliation are included automatically instead of
    # silently dropping out of list rows.
    _SESSION_COMPACT_EXCLUDED = frozenset(
        {"system_prompt", "system_prompt_hash", "git_metadata_generation"}
    )
    _session_compact_cols_sql: Optional[str] = None

    def usage_totals(self, *, min_message_count: int = 1, include_archived: bool = False) -> Dict[str, float]:
        """Tokens and spend across this store, as one aggregate.

        The sidebar shows a profile's totals beside a page of its sessions, so
        summing the rows it happens to have loaded would report a fraction of
        the truth and shrink as paging changed. SQLite adds the columns up over
        every row instead, at the cost of one scan.

        Spend is the billed figure when the provider returned one and the
        estimate otherwise — the same precedence a single row renders.
        """
        where = ["parent_session_id IS NULL", "message_count >= ?"]
        params: List[Any] = [min_message_count]
        if not include_archived:
            where.append("COALESCE(archived, 0) = 0")

        with self._read_ctx() as conn:
            row = conn.execute(
                f"""
                SELECT COALESCE(SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)), 0),
                       COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd, 0)), 0)
                  FROM sessions
                 WHERE {' AND '.join(where)}
                """,
                params,
            ).fetchone()

        return {"tokens": int(row[0] or 0), "cost_usd": float(row[1] or 0.0)}

    def list_recent_sessions_bounded(
        self,
        *,
        limit: int = 20,
        exclude_sources: List[str] = None,
        timeout_seconds: float = 3.0,
        candidate_limit: int = None,
        lineage_limit: int = None,
    ) -> List[Dict[str, Any]]:
        """List recent user conversations without an unbounded message scan.

        This is the latency-bounded browse path used by ``session_search()``.
        It deliberately separates cheap candidate selection from expensive
        hydration:

        1. preselect a small set of session ids from the indexed durable
           activity timestamp (falling back to ``started_at``);
        2. resolve only those candidates across compression ancestry/chains;
        3. calculate message-derived activity and previews only for that
           bounded set.

        Compression ancestry and descendant traversal use ``UNION`` so a
        corrupt cycle cannot revisit the same session for one logical root,
        plus a total-row ceiling so a deep or highly branching lineage cannot
        defeat the candidate bound.  If that ceiling is reached before a
        candidate resolves to a terminal root/tip, that incomplete lineage is
        omitted from the browse result rather than expanded without bound.

        The query has a cooperative SQLite VM progress deadline.  Expensive
        statements that remain active beyond ``timeout_seconds`` are
        interrupted at the next progress callback and this method raises
        ``TimeoutError`` instead of holding a gateway callback indefinitely.
        Cheap statements may finish between callbacks; the deadline is a
        fail-safe for sustained work, not a real-time scheduler guarantee.

        This method intentionally supports only the filters needed by the
        agent-tool browse shape.  Rich dashboard/search callers keep using
        :meth:`list_sessions_rich`.
        """
        limit = max(1, int(limit))
        timeout_seconds = max(0.0, float(timeout_seconds))
        if candidate_limit is None:
            candidate_limit = max(128, limit * 8)
        candidate_limit = max(limit, min(int(candidate_limit), 2048))
        if lineage_limit is None:
            lineage_limit = min(8192, candidate_limit * 8)
        lineage_limit = max(candidate_limit, min(int(lineage_limit), 8192))

        candidate_clauses = [
            "s.archived = 0",
            "s.hidden = 0",
            f"{_delegate_from_json('s.model_config')} IS NULL",
        ]
        candidate_params: List[Any] = []
        if exclude_sources:
            placeholders = ",".join("?" for _ in exclude_sources)
            candidate_clauses.append(f"s.source NOT IN ({placeholders})")
            candidate_params.extend(exclude_sources)
        candidate_where = " AND ".join(candidate_clauses)

        # A compression continuation is an implementation edge, unlike /new
        # reset and /branch children which are independent user-visible
        # conversations.  The same predicate is used in both directions so a
        # candidate tip maps to its logical root and the root maps back to the
        # freshest live tip.
        compression_parent_edge = f"""
            parent.end_reason = 'compression'
            AND child.parent_session_id = parent.id
            AND json_extract(
                COALESCE(child.model_config, '{{}}'), '$._branched_from'
            ) IS NULL
            AND {_delegate_from_json('child.model_config')} IS NULL
            AND COALESCE(child.source, '') != 'tool'
        """

        query = f"""
            WITH RECURSIVE
            recent_candidates(id) AS (
                SELECT s.id
                FROM sessions s
                WHERE {candidate_where}
                ORDER BY COALESCE(s.last_activity_at, s.started_at) DESC,
                         s.started_at DESC, s.id DESC
                LIMIT ?
            ),
            ancestors(candidate_id, cur_id) AS (
                SELECT id, id FROM recent_candidates
                UNION
                SELECT a.candidate_id, parent.id
                FROM ancestors a
                JOIN sessions child ON child.id = a.cur_id
                JOIN sessions parent ON {compression_parent_edge}
                LIMIT ?
            ),
            candidate_roots(root_id) AS (
                SELECT DISTINCT a.cur_id
                FROM ancestors a
                JOIN sessions child ON child.id = a.cur_id
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM sessions parent
                    WHERE {compression_parent_edge}
                )
            ),
            chain(root_id, cur_id) AS (
                SELECT root_id, root_id FROM candidate_roots
                UNION
                SELECT c.root_id, child.id
                FROM chain c
                JOIN sessions parent ON parent.id = c.cur_id
                JOIN sessions child ON {compression_parent_edge}
                LIMIT ?
            ),
            chain_rows AS (
                SELECT
                    c.root_id,
                    c.cur_id,
                    {_sql_session_last_active_by_id('c.cur_id')} AS activity,
                    CASE WHEN EXISTS (
                        SELECT 1
                        FROM sessions parent
                        JOIN sessions child ON {compression_parent_edge}
                        WHERE parent.id = c.cur_id
                    ) THEN 0 ELSE 1 END AS is_tip
                FROM chain c
            ),
            ranked_tips AS (
                SELECT root_id, cur_id, activity,
                       ROW_NUMBER() OVER (
                           PARTITION BY root_id
                           ORDER BY activity DESC, cur_id DESC
                       ) AS rank_in_root
                FROM chain_rows
                WHERE is_tip = 1
            )
            SELECT
                tip.id,
                tip.source,
                tip.model,
                tip.title,
                s.started_at AS started_at,
                tip.ended_at,
                tip.end_reason,
                tip.message_count,
                tip.tool_call_count,
                rt.activity AS last_active,
                COALESCE(
                    (SELECT {_PREVIEW_RAW_SELECT}
                     FROM messages m
                     WHERE m.session_id = tip.id
                       AND m.role = 'user'
                       AND m.content IS NOT NULL
                       AND {_PREVIEW_ELIGIBLE_SQL}
                     ORDER BY m.timestamp, m.id LIMIT 1),
                    ''
                ) AS _preview_raw,
                CASE WHEN s.id != tip.id THEN s.id ELSE NULL END
                    AS _lineage_root_id
            FROM ranked_tips rt
            JOIN sessions s ON s.id = rt.root_id
            JOIN sessions tip ON tip.id = rt.cur_id
            WHERE rt.rank_in_root = 1
              AND s.archived = 0
              AND s.hidden = 0
              AND {_LISTABLE_CHILD_SQL}
              AND {_delegate_from_json('s.model_config')} IS NULL
            ORDER BY rt.activity DESC, s.started_at DESC, tip.id DESC
            LIMIT ?
        """
        params = candidate_params + [
            candidate_limit,
            lineage_limit,
            lineage_limit,
            limit,
        ]
        deadline = time.monotonic() + timeout_seconds
        interrupted_by_deadline = False

        def _deadline_progress_handler() -> int:
            nonlocal interrupted_by_deadline
            if time.monotonic() >= deadline:
                interrupted_by_deadline = True
                return 1
            return 0

        try:
            with self._read_ctx() as conn:
                conn.set_progress_handler(_deadline_progress_handler, 1000)
                try:
                    rows = conn.execute(query, params).fetchall()
                finally:
                    conn.set_progress_handler(None, 0)
        except sqlite3.OperationalError as exc:
            if interrupted_by_deadline and "interrupt" in str(exc).lower():
                raise TimeoutError(
                    f"recent-session browse exceeded {timeout_seconds:g}s deadline"
                ) from exc
            raise

        sessions = []
        for row in rows:
            session = self._session_row_dict(row)
            session["preview"] = _shape_preview(session.pop("_preview_raw", ""))
            session["unread"] = self.session_unread(session)
            sessions.append(session)
        return sessions

    def list_sessions_rich(
        self,
        source: str = None,
        sources: List[str] = None,
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
        include_pinned: bool = False,
        session_key: str = None,
        include_hidden: bool = False,
    ) -> List[Dict[str, Any]]:
        """List sessions with preview (first user message) and last active timestamp.

        Returns dicts with keys: id, source, model, title, started_at, ended_at,
        message_count, preview (first 60 chars of first user message),
        last_active (freshest of last_activity_at heartbeat and latest
        message timestamp, else started_at).

        Uses a single query with correlated subqueries instead of N+2 queries.

        By default, child sessions that represent implementation details
        (subagent runs, compression continuations) are excluded. User-visible
        branch and reset children remain listable. Pass ``include_children=True``
        to include every child.

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

        Pass ``include_pinned=True`` to back-fill any conversation carrying the
        durable ``pinned`` flag that the LIMIT/OFFSET window left out. A pin is
        a "this must always be reachable" statement, so a pinned conversation
        aging past the requested page is a bug, not a paging outcome — the
        desktop sidebar would render an empty Pinned section. Back-filled rows
        obey the same filters (source, archived, min_message_count) as the
        page: an archived or filtered-out conversation stays out.

        Pass ``session_key`` to restrict results to one stable gateway
        conversation scope (DM, group, channel, or thread, including the
        configured per-user isolation policy).
        """
        # Rows carry token/cost totals — drain queued deltas first so
        # listings (sidebar, /resume, dashboards) show exact counters.
        self.flush_token_counts()
        where_clauses = []
        params = []

        if not include_children:
            # Show roots and user-visible branch/reset sessions, while still
            # hiding sub-agent runs and compression continuations. All four
            # carry parent_session_id, so the shared predicate classifies the
            # edge from stable markers plus legacy-compatible parent metadata.
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

        include_sources = [source] if source else list(sources or [])
        if include_sources:
            placeholders = ",".join("?" for _ in include_sources)
            where_clauses.append(f"s.source IN ({placeholders})")
            params.extend(include_sources)
        if session_key:
            where_clauses.append("s.session_key = ?")
            params.append(session_key)
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
        if not include_hidden:
            where_clauses.append("s.hidden = 0")

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        # Snapshot the filter params before the query builders below extend
        # them with LIMIT/OFFSET — the pinned back-fill reuses the same WHERE.
        base_where_params = list(params)
        prompt_select = (
            "" if compact_rows
            else ", COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved"
        )
        prompt_join = (
            "" if compact_rows
            else "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash"
        )

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
            # The CTE seeds from rows the outer WHERE admits (roots +
            # user-visible branch/reset children), then recursively joins through
            # compression-continuation edges. Do NOT require
            # child.started_at >= parent.ended_at here: real desktop/gateway
            # races can insert the continuation row before the parent's
            # ended_at is written, while stale websocket siblings may satisfy
            # the timestamp test and hijack resume/list projection.
            outer_where = where_sql
            id_params: List[Any] = []
            filter_clauses: List[str] = []

            def _like_pattern(needle: str) -> str:
                return f"%{_escape_like(needle)}%"

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
                        MAX({_sql_session_last_active_by_id("cur_id")}) AS effective_last_active
                    FROM chain
                    GROUP BY root_id
                )
                SELECT {_sel}{prompt_select},
                    COALESCE(
                        (SELECT {_PREVIEW_RAW_SELECT}
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                           AND {_PREVIEW_ELIGIBLE_SQL}
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw,
                    {_sql_session_last_active("s")} AS last_active,
                    COALESCE(cm.effective_last_active, s.started_at) AS _effective_last_active
                FROM sessions s
                LEFT JOIN chain_max cm ON cm.root_id = s.id
                {prompt_join}
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
                SELECT {_sel}{prompt_select},
                    COALESCE(
                        (SELECT {_PREVIEW_RAW_SELECT}
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                           AND {_PREVIEW_ELIGIBLE_SQL}
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw,
                    {_sql_session_last_active("s")} AS last_active
                FROM sessions s
                {prompt_join}
                {where_sql}
                ORDER BY s.started_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])
        with self._read_ctx() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        sessions = []
        for row in rows:
            s = self._session_row_dict(row)
            s["preview"] = _shape_preview(s.pop("_preview_raw", ""))
            # Drop the internal ordering column so callers see a clean dict.
            s.pop("_effective_last_active", None)
            sessions.append(s)

        # Back-fill pinned conversations the page missed. A pin outlives
        # recency, so this runs BEFORE compression projection below — a
        # back-filled root then projects to its live tip exactly like a row
        # that had made the page on its own. One extra query, bounded by the
        # number of pins (a handful), never N+1 per pin.
        if include_pinned:
            seen_ids = {s["id"] for s in sessions}
            pinned_where = (
                f"{where_sql} AND s.pinned = 1" if where_sql else "WHERE s.pinned = 1"
            )
            _sel = self._compact_session_cols() if compact_rows else "s.*"
            pinned_query = f"""
                SELECT {_sel}{prompt_select},
                    COALESCE(
                        (SELECT {_PREVIEW_RAW_SELECT}
                         FROM messages m
                         WHERE m.session_id = s.id AND m.role = 'user' AND m.content IS NOT NULL
                           AND {_PREVIEW_ELIGIBLE_SQL}
                         ORDER BY m.timestamp, m.id LIMIT 1),
                        ''
                    ) AS _preview_raw,
                    COALESCE(
                        (SELECT MAX(m2.timestamp) FROM messages m2 WHERE m2.session_id = s.id),
                        s.started_at
                    ) AS last_active
                FROM sessions s
                {prompt_join}
                {pinned_where}
                ORDER BY s.started_at DESC
            """
            with self._read_ctx() as conn:
                pinned_cursor = conn.execute(pinned_query, base_where_params)
                pinned_rows = pinned_cursor.fetchall()
            for row in pinned_rows:
                s = self._session_row_dict(row)
                if s["id"] in seen_ids:
                    continue
                s["preview"] = _shape_preview(s.pop("_preview_raw", ""))
                seen_ids.add(s["id"])
                sessions.append(s)

        # Project compression roots forward to their tips. Each row whose
        # end_reason is 'compression' has a continuation child; replace the
        # surfaced fields (id, message_count, title, last_active, ended_at,
        # end_reason, preview) with the tip's values so the list entry acts
        # as the live conversation. Keep the root's started_at to preserve
        # chronological ordering by original conversation start.
        if project_compression_tips and not include_children:
            # get_compression_tip() walks each root's chain individually (it's
            # a per-session graph walk, not batchable in one query), but the
            # tip *row* fetch afterward was previously one _get_session_rich_row()
            # call per compression root. Batch that half instead: resolve
            # every tip id first, then fetch all tip rows in a single query.
            tip_ids_by_root: Dict[str, str] = {}
            chain_by_root: Dict[str, List[str]] = {}
            for s in sessions:
                if s.get("end_reason") != "compression":
                    continue
                chain = self.get_compression_chain(s["id"])
                tip_id = chain[-1] if chain else s["id"]
                if tip_id != s["id"]:
                    tip_ids_by_root[s["id"]] = tip_id
                    chain_by_root[s["id"]] = chain

            tip_rows = (
                self._get_session_rich_rows_batch(
                    set(tip_ids_by_root.values()), compact_rows=compact_rows
                )
                if tip_ids_by_root
                else {}
            )

            projected = []
            for s in sessions:
                tip_id = tip_ids_by_root.get(s["id"])
                tip_row = tip_rows.get(tip_id) if tip_id else None
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
                # Every id on the chain, intermediates included. Root and tip
                # alone are not enough client-side: a persisted tile or route
                # can hold a MIDDLE segment's id (it was the tip when opened,
                # then rotated again), and with only the root/tip pair such a
                # surface can no longer prove it names this conversation —
                # which is how one chat ends up open twice after compaction.
                merged["_lineage_ids"] = chain_by_root.get(s["id"]) or None
                projected.append(merged)
            sessions = projected

        # Derive read state per surfaced conversation. ``last_read_at`` is
        # lineage-stamped by set_session_read, so a projected row's root
        # watermark and its tip's are the same value — comparing it against
        # the tip's last_active is correct either way.
        for s in sessions:
            s["unread"] = self.session_unread(s)

        return sessions

    def session_lifecycle_statuses(
        self, session_ids: List[str]
    ) -> Dict[str, str]:
        """Classify each session's lifecycle state from its LAST message row.

        Returns ``{session_id: status}`` where status is one of:

        - ``'complete'``    — last message is a normal assistant reply
        - ``'interrupted'`` — last message is a user turn, a pending assistant
          tool call (no tool result followed), or a tool result the assistant
          never responded to
        - ``'error'``       — last message carries an error finish_reason
        - ``'empty'``       — session has no messages

        Cost-bounded by design: one query that resolves each listed session's
        newest message id via ``MAX(id)`` (an index seek on
        ``idx_messages_session_id``) and joins back for that single row's
        role/tool_calls/finish_reason. Never scans transcripts, so it stays
        cheap on large databases regardless of total message volume.
        """
        ids = [sid for sid in (session_ids or []) if sid]
        if not ids:
            return {}
        statuses: Dict[str, str] = {sid: "empty" for sid in ids}
        placeholders = ",".join("?" for _ in ids)
        query = f"""
            SELECT m.session_id, m.role,
                   m.tool_calls IS NOT NULL AS has_tool_calls,
                   m.finish_reason
            FROM messages m
            JOIN (
                SELECT session_id, MAX(id) AS max_id
                FROM messages
                WHERE session_id IN ({placeholders})
                GROUP BY session_id
            ) latest ON m.id = latest.max_id
        """
        with self._read_ctx() as conn:
            rows = conn.execute(query, ids).fetchall()
        for row in rows:
            statuses[row["session_id"]] = classify_session_status(
                role=row["role"],
                has_tool_calls=bool(row["has_tool_calls"]),
                finish_reason=row["finish_reason"],
            )
        return statuses

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

    @staticmethod
    def _encode_display_metadata(display_metadata: Any) -> Optional[str]:
        """Serialize ``display_metadata`` for its TEXT column without double-encoding.

        Import/replace paths can hand us an already-serialized JSON string (the
        same hazard ``tool_calls`` guards against above). ``json.dumps`` on that
        string would store a quoted JSON string, and the single ``json.loads``
        on read then yields a ``str`` instead of a dict.
        """
        if not display_metadata:
            return None
        if isinstance(display_metadata, str):
            try:
                parsed = json.loads(display_metadata)
            except (json.JSONDecodeError, TypeError):
                logger.warning("Ignoring non-JSON display metadata on write")
                return None
            if not isinstance(parsed, dict):
                logger.warning("Ignoring non-object display metadata on write")
                return None
            return json.dumps(parsed)
        if isinstance(display_metadata, dict):
            return json.dumps(display_metadata)
        logger.warning(
            "Ignoring unexpected display metadata type on write: %s",
            type(display_metadata).__name__,
        )
        return None

    def _check_transcript_write_guards(
        self,
        conn,
        session_id: str,
        compression_lock_holder: Optional[str],
        turn_lease_holder: Optional[str] = None,
        turn_lease_ttl_seconds: float = 300.0,
        reject_active_turn_lease: bool = False,
        reject_active_compression_lock: bool = False,
        allow_closed_compression_parent: bool = False,
    ) -> None:
        """Transcript-write admission checks, run INSIDE the write txn.

        Shared by :meth:`append_message` and :meth:`append_messages_batch` so
        the two writers can never diverge on these correctness invariants
        (this guard has already needed targeted fixes — see the #74478 patience
        note below). User-initiated transcript mutations may opt in to rejecting
        an active unowned turn lease in that same transaction.
        """
        # NOTE (#75316 redesign): appends do NOT check compression_locks.
        # The lock's job is to stop two COMPRESSIONS colliding, not to fence
        # ordinary transcript writes. Concurrent appends during a compression
        # are safe by construction: archive_and_compact() commits against a
        # watermark captured at compression start and clones every row that
        # arrived after it back into the live transcript, in the same write
        # transaction. Blocking appends here was the root cause of a whole
        # symptom family — turns dying as session_persistence_failed while a
        # slow provider summary held the lease (#74568, #77386), including
        # stale locks from dead PIDs blocking writes for the full TTL.
        # Destructive user mutations are different: a compressor that already
        # captured its watermark can otherwise publish the pre-rewind snapshot
        # after the mutation and resurrect the removed turn. Keep that narrow
        # fence opt-in so ordinary appends retain the watermark behavior.
        if reject_active_compression_lock:
            active_lock = conn.execute(
                "SELECT holder, expires_at FROM compression_locks "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if active_lock is not None:
                current_holder = active_lock["holder"]
                if (
                    float(active_lock["expires_at"]) <= time.time()
                    or _compression_lock_holder_process_is_dead(current_holder)
                ):
                    conn.execute(
                        "DELETE FROM compression_locks "
                        "WHERE session_id = ? AND holder = ?",
                        (session_id, current_holder),
                    )
                elif current_holder != compression_lock_holder:
                    raise SessionCompressionInProgressError(
                        f"Session {session_id!r} is being compressed by another writer"
                    )
        if turn_lease_holder or reject_active_turn_lease:
            conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)
            lease = conn.execute(
                "SELECT holder, expires_at FROM session_turn_leases "
                "WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            now = time.time()
            if turn_lease_holder:
                if lease is None or lease["holder"] != turn_lease_holder:
                    raise SessionTurnLeaseLostError(
                        f"Session turn lease lost; refusing transcript write "
                        f"for {session_id!r}"
                    )
                if float(lease["expires_at"]) <= now:
                    # Expiry makes the row reclaimable; it does not prove that a
                    # takeover occurred. BEGIN IMMEDIATE serializes this renewal
                    # with acquisition, so a still-matching owner can recover from
                    # a starved refresher without weakening the foreign-holder fence.
                    conn.execute(
                        "UPDATE session_turn_leases SET expires_at = ? "
                        "WHERE conversation_id = ? AND holder = ?",
                        (
                            now + max(0.1, float(turn_lease_ttl_seconds)),
                            conversation_id,
                            turn_lease_holder,
                        ),
                    )
            elif lease is not None:
                current_holder = lease["holder"]
                if (
                    float(lease["expires_at"]) <= now
                    or _compression_lock_holder_process_is_dead(current_holder)
                ):
                    # Match acquisition semantics: an expired or provably dead
                    # owner is reclaimable. Deleting it inside this BEGIN IMMEDIATE
                    # transaction also fences a stale late flush after the mutation.
                    conn.execute(
                        "DELETE FROM session_turn_leases "
                        "WHERE conversation_id = ? AND holder = ?",
                        (conversation_id, current_holder),
                    )
                else:
                    raise SessionTurnLeaseLostError(
                        f"Session has an active turn lease; refusing transcript "
                        f"mutation for {session_id!r}"
                    )
        session = conn.execute(
            "SELECT ended_at, end_reason FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if (
            session is not None
            and session["ended_at"] is not None
            and session["end_reason"] == "compression"
            and not allow_closed_compression_parent
        ):
            raise CompressionSessionClosedError(session_id)

    @staticmethod
    def _decode_display_metadata(raw: Any) -> Optional[Dict[str, Any]]:
        """Decode a ``display_metadata`` column into the dict every reader expects.

        Every message read path must go through this. Returning the raw TEXT
        instead reaches the desktop as a string, where ``'task_count' in meta``
        throws and fails the whole resume. Rows written before the encode guard
        landed are double-encoded, so unwrap a second layer when we find one.
        """
        if raw is None:
            return None
        try:
            meta = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(meta, str):
                meta = json.loads(meta)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Ignoring invalid display metadata on message row")
            return None
        if not isinstance(meta, dict):
            logger.warning("Ignoring non-object display metadata on message row")
            return None
        return meta

    @staticmethod
    def _reasoning_json_text(value: Any) -> Optional[str]:
        """Serialize a structured reasoning field for its TEXT column.

        ``reasoning_details`` / ``codex_reasoning_items`` / ``codex_message_items``
        arrive as list/dict structures from the live runtime, but callers that
        round-trip stored rows — ``get_messages`` straight into
        ``replace_messages``, e.g. the POST /api/sessions/{id}/fork handler —
        hand back the raw TEXT these columns already hold, because
        ``get_messages`` only deserializes ``content`` and ``tool_calls``.
        Re-dumping that TEXT double-encodes it, and the forked session's next
        ``get_messages_as_conversation`` json.loads then yields the inner
        string instead of the original list, so every reasoning-replay consumer
        (all of which check ``isinstance(..., list)``) silently drops it.
        Strings are therefore stored as-is; structures are dumped.
        """
        if not value:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value)

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
        _compressed_summary: bool = False,
        timestamp: Any = None,
        api_content: Optional[str] = None,
        display_kind: Optional[str] = None,
        display_metadata: Optional[Dict[str, Any]] = None,
        compression_lock_holder: Optional[str] = None,
        turn_lease_holder: Optional[str] = None,
        turn_lease_ttl_seconds: float = 300.0,
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
        display_metadata_json = self._encode_display_metadata(display_metadata)
        # Serialize structured fields to JSON before entering the write txn
        reasoning_details_json = self._reasoning_json_text(reasoning_details)
        codex_items_json = self._reasoning_json_text(codex_reasoning_items)
        codex_message_items_json = self._reasoning_json_text(codex_message_items)
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
            self._check_transcript_write_guards(
                conn,
                session_id,
                compression_lock_holder,
                turn_lease_holder=turn_lease_holder,
                turn_lease_ttl_seconds=turn_lease_ttl_seconds,
            )
            cursor = conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items, platform_message_id, observed, _compressed_summary, active, api_content, display_kind, display_metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    1 if _compressed_summary else 0,
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

        # Transcript append is THE critical write: its failure aborts the
        # user's turn (session_persistence_failed). Use the long patience so
        # a sibling process legitimately holding the write lock for seconds
        # (VACUUM, TRUNCATE checkpoint at close, an older pre-bounded-merge
        # process's FTS optimize) can't destroy a healthy turn (#74478).
        return self._execute_write(
            _do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S
        )

    def append_messages_batch(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        compression_lock_holder: Optional[str] = None,
        turn_lease_holder: Optional[str] = None,
        chunk_rows: Optional[int] = None,
        turn_lease_ttl_seconds: float = 300.0,
    ) -> int:
        """Append multiple messages atomically in ONE write transaction.

        ``messages`` is a list of dicts in the same shape
        :meth:`_insert_message_rows` already consumes for replace/compact/
        import (role, content, tool_name, tool_calls, tool_call_id,
        finish_reason, reasoning*, codex_*, timestamp, api_content,
        display_kind, display_metadata, ...). Reusing that helper keeps ONE
        row-serialization path for every multi-row writer.

        A turn-boundary flush writes the whole turn (user + assistant + tool
        rows, typically 3-8 messages) as one BEGIN IMMEDIATE / commit pair
        instead of one transaction (and, off WAL, one fsync) per row.

        Atomicity contract: all rows land or none do (the caller re-flushes
        unstamped messages on the next attempt). The same admission guards
        as :meth:`append_message` run once for the batch — same session,
        same instant.

        ``chunk_rows`` bounds the transaction size for LARGE copies (branch
        seeds can be thousands of rows; measured: 10k rows ≈ 2.4s inside one
        BEGIN IMMEDIATE because the FTS triggers run per row, which would
        monopolize the write lock and starve concurrent writers). When set,
        the batch commits in chunks of at most that many rows — same
        recovery semantics as the old per-row loops (a mid-copy failure
        leaves a partial seed), just with bounded lock holds. A turn flush
        never needs it. Returns the inserted row count.
        """
        if not messages:
            return 0

        if chunk_rows is not None and len(messages) > chunk_rows:
            inserted_total = 0
            for start in range(0, len(messages), chunk_rows):
                inserted_total += self.append_messages_batch(
                    session_id,
                    messages[start:start + chunk_rows],
                    compression_lock_holder=compression_lock_holder,
                    turn_lease_holder=turn_lease_holder,
                    turn_lease_ttl_seconds=turn_lease_ttl_seconds,
                )
            return inserted_total

        def _do(conn):
            self._check_transcript_write_guards(
                conn,
                session_id,
                compression_lock_holder,
                turn_lease_holder=turn_lease_holder,
                turn_lease_ttl_seconds=turn_lease_ttl_seconds,
            )
            from agent.transcript_repair import resolve_and_repair_transcript_batch

            inserted_rows = resolve_and_repair_transcript_batch(
                conn,
                session_id,
                messages,
                encode_content_fn=self._encode_content,
                decode_content_fn=self._decode_content,
            )
            inserted = 0
            tool_calls_total = 0
            if inserted_rows:
                inserted, tool_calls_total = self._insert_message_rows(
                    conn, session_id, inserted_rows
                )

            # One aggregated counter update for the newly inserted rows.
            if tool_calls_total > 0:
                conn.execute(
                    """UPDATE sessions SET message_count = message_count + ?,
                       tool_call_count = tool_call_count + ? WHERE id = ?""",
                    (inserted, tool_calls_total, session_id),
                )
            elif inserted > 0:
                conn.execute(
                    "UPDATE sessions SET message_count = message_count + ? WHERE id = ?",
                    (inserted, session_id),
                )
            return inserted

        # Same criticality as append_message: this IS the turn's transcript.
        return self._execute_write(
            _do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S
        )

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
                    self._encode_display_metadata(display_metadata),
                    row[0],
                ),
            )
            return True

        return bool(self._execute_write(_do))

    #: Key under which message reactions live inside ``display_metadata``.
    #: Reactions share the existing per-message JSON column rather than a side
    #: table so they survive rewind/compaction row rewrites with the row itself.
    REACTIONS_METADATA_KEY = "reactions"

    def set_message_reaction(
        self,
        session_id: str,
        message_row_id: int,
        emoji: Optional[str],
        *,
        author: str = "user",
    ) -> Optional[List[Dict[str, Any]]]:
        """Set (or with ``emoji=None`` clear) *author*'s reaction on one message.

        iOS Tapback semantics: one reaction per author per message. Re-sending
        the same emoji clears it, a different emoji replaces it. Returns the
        message's full reaction list after the write, or ``None`` when the row
        doesn't exist or isn't part of *session_id*.
        """
        if not session_id or message_row_id is None:
            return None

        def _do(conn):
            row = conn.execute(
                "SELECT display_metadata FROM messages WHERE id = ? AND session_id = ?",
                (message_row_id, session_id),
            ).fetchone()
            if row is None:
                return None

            meta = self._decode_display_metadata(row[0]) or {}
            existing = meta.get(self.REACTIONS_METADATA_KEY)
            reactions = [
                r
                for r in (existing if isinstance(existing, list) else [])
                if isinstance(r, dict) and r.get("author") != author
            ]
            previous = next(
                (
                    r
                    for r in (existing if isinstance(existing, list) else [])
                    if isinstance(r, dict) and r.get("author") == author
                ),
                None,
            )
            # Tapping the live reaction again retracts it.
            toggling_off = (
                emoji is not None and previous is not None and previous.get("emoji") == emoji
            )
            if emoji and not toggling_off:
                reactions.append(
                    {"emoji": _scrub_surrogates(emoji), "author": author, "at": time.time()}
                )

            if reactions:
                meta[self.REACTIONS_METADATA_KEY] = reactions
            else:
                meta.pop(self.REACTIONS_METADATA_KEY, None)

            conn.execute(
                "UPDATE messages SET display_metadata = ? WHERE id = ?",
                (self._encode_display_metadata(meta) if meta else None, message_row_id),
            )
            return reactions

        return self._execute_write(_do)

    def get_message_reactions(
        self, session_id: str, message_row_id: int
    ) -> List[Dict[str, Any]]:
        """Return the reaction list persisted on one message row (never ``None``)."""
        if not session_id or message_row_id is None:
            return []

        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT display_metadata FROM messages WHERE id = ? AND session_id = ?",
                (message_row_id, session_id),
            ).fetchone()

        if row is None:
            return []

        meta = self._decode_display_metadata(row[0]) or {}
        reactions = meta.get(self.REACTIONS_METADATA_KEY)

        return [r for r in reactions if isinstance(r, dict)] if isinstance(reactions, list) else []

    def take_unseen_reactions(
        self, session_id: str, *, author: str = "user"
    ) -> List[Dict[str, Any]]:
        """Return *author*'s not-yet-surfaced reactions and mark them seen.

        Powers the cache-safe model-context path: reactions are announced on the
        NEXT user turn (never by rewriting the message that was reacted to), and
        the ``seen`` stamp guarantees each one is announced exactly once.
        """
        if not session_id:
            return []

        def _do(conn):
            rows = conn.execute(
                "SELECT id, role, content, display_metadata FROM messages "
                "WHERE session_id = ? AND active = 1 AND display_metadata IS NOT NULL "
                "ORDER BY id",
                (session_id,),
            ).fetchall()

            pending = []
            for row in rows:
                meta = self._decode_display_metadata(row["display_metadata"])
                if not meta:
                    continue
                reactions = meta.get(self.REACTIONS_METADATA_KEY)
                if not isinstance(reactions, list):
                    continue

                changed = False
                for reaction in reactions:
                    if (
                        not isinstance(reaction, dict)
                        or reaction.get("author") != author
                        or reaction.get("seen")
                    ):
                        continue
                    reaction["seen"] = True
                    changed = True
                    content = self._decode_content(row["content"])
                    pending.append(
                        {
                            "row_id": row["id"],
                            "role": row["role"],
                            "emoji": reaction.get("emoji") or "",
                            "text": content if isinstance(content, str) else "",
                        }
                    )

                if changed:
                    conn.execute(
                        "UPDATE messages SET display_metadata = ? WHERE id = ?",
                        (self._encode_display_metadata(meta), row["id"]),
                    )

            return pending

        return self._execute_write(_do) or []

    def latest_message_row_id(
        self, session_id: str, *, role: str = "user", offset: int = 0, require_text: bool = True
    ) -> Optional[int]:
        """Row id of the most recent active message with *role*, or ``None``.

        Two callers, same need — "the message I mean, without an id": the agent
        defaulting to the turn that triggered it, and the desktop reacting to a
        live message that hasn't round-tripped through a resume yet.
        ``offset`` steps to earlier turns (1 = the one before the latest) so a
        reaction can land retroactively — "two messages ago" is how the caller
        thinks about it.

        ``require_text`` (default) skips rows with no plain-text content —
        tool-call-only assistant turns and attachment stubs don't render as
        bubbles, so "the latest message" as a HUMAN means it must never
        resolve to one (a reaction landing on an invisible row looks dropped,
        and its annotation quotes an empty string).
        """
        if not session_id or role not in {"user", "assistant"} or offset < 0:
            return None

        text_filter = (
            "AND content IS NOT NULL AND TRIM(content) != '' " if require_text else ""
        )

        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT id FROM messages WHERE session_id = ? AND role = ? "
                f"AND active = 1 {text_filter}ORDER BY id DESC LIMIT 1 OFFSET ?",
                (session_id, role, int(offset)),
            ).fetchone()

        return row[0] if row else None

    def latest_user_message_row_id(self, session_id: str) -> Optional[int]:
        """Row id of the most recent active user message, or ``None``.

        The agent's default reaction target: "the message that triggered me",
        so the model never has to thread row ids through a tool call (mirrors
        the photon adapter's ``_record_last_inbound``).
        """
        return self.latest_message_row_id(session_id, role="user")

    def get_message_role(self, session_id: str, row_id: int) -> Optional[str]:
        """Role of the active message at *row_id* in *session_id*, or ``None``.

        Lets a reaction event carry the target's role so a renderer can match
        a live message that doesn't know its durable row id yet.
        """
        if not session_id:
            return None

        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT role FROM messages WHERE id = ? AND session_id = ? AND active = 1",
                (int(row_id), session_id),
            ).fetchone()

        return row[0] if row else None

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
            reasoning_details_json = self._reasoning_json_text(reasoning_details)
            codex_items_json = self._reasoning_json_text(codex_reasoning_items)
            codex_message_items_json = self._reasoning_json_text(codex_message_items)
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

            cur = conn.execute(
                """INSERT INTO messages (session_id, role, content, tool_call_id,
                   tool_calls, tool_name, effect_disposition, timestamp, token_count, finish_reason,
                   reasoning, reasoning_content, reasoning_details, codex_reasoning_items,
                   codex_message_items, platform_message_id, observed, _compressed_summary, active, api_content, display_kind, display_metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    1 if msg.get("_compressed_summary") else 0,
                    1,
                    _scrub_surrogates(api_content) if isinstance(api_content, str) else None,
                    _scrub_surrogates(msg.get("display_kind")) if isinstance(msg.get("display_kind"), str) else None,
                    self._encode_display_metadata(msg.get("display_metadata")),
                ),
            )
            if isinstance(msg, dict) and cur.lastrowid is not None:
                msg["_row_id"] = cur.lastrowid
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
        archive_dropped: bool = False,
        reject_active_turn_lease: bool = False,
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

        Pass ``archive_dropped=True`` to SOFT-archive the live rows instead of
        DELETEing them: the replaced turns stay on disk with ``active = 0``,
        ``compacted = 0`` — the same "the user took it back" marking
        :meth:`rewind_to_message` applies — and stay readable via
        :meth:`get_messages` with ``include_inactive=True``. This is the mode a
        rewind/edit/regenerate must use: those flows overwrite a transcript the
        user may not have meant to drop, and a plain DELETE also evicts the rows
        from the FTS index, leaving nothing to recover from (#82756). It implies
        active-only handling — already-archived rows are never touched — so
        ``active_only`` is redundant with it. The rewritten set is inserted as
        fresh active rows exactly as in the destructive path, so the live view
        is identical either way; only the durability of the dropped turns
        differs.

        Pass ``reject_active_turn_lease=True`` for user-initiated rewrites that
        do not already own the cross-process turn lease. The lease check and
        transcript mutation then share one write transaction, so a second
        process cannot archive or replace a turn that is still being produced.
        """

        active_clause = " AND active = 1" if active_only else ""

        def _do(conn):
            if reject_active_turn_lease:
                self._check_transcript_write_guards(
                    conn,
                    session_id,
                    None,
                    reject_active_turn_lease=True,
                    reject_active_compression_lock=True,
                )
            else:
                session = conn.execute(
                    "SELECT ended_at, end_reason FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if (
                    session is not None
                    and session["ended_at"] is not None
                    and session["end_reason"] == "compression"
                ):
                    raise CompressionSessionClosedError(session_id)
            if archive_dropped:
                # Content-preserving UPDATE: the rows keep their FTS entries
                # (the messages_fts triggers fire on INSERT / DELETE / UPDATE
                # of content columns, not on `active`), so the replaced turns
                # stay readable via get_messages(include_inactive=True) and
                # searchable with include_inactive=True after the rewrite.
                conn.execute(
                    "UPDATE messages SET active = 0 "
                    "WHERE session_id = ? AND active = 1",
                    (session_id,),
                )
            else:
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

        Cheap existence probe — does not load rows. NOTE: production rewrite
        paths no longer branch on this (they pass ``active_only=True``
        unconditionally — a probe can fail open or race a concurrent
        ``archive_and_compact``, #80216); kept for tests and diagnostics.
        """
        with self._read_ctx() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM messages WHERE session_id = ? AND active = 0 LIMIT 1",
                (session_id,),
            )
            return cursor.fetchone() is not None

    def get_active_message_watermark(self, session_id: str) -> int:
        """MAX(id) of the session's active rows — the compression watermark.

        Captured at compression START (before the slow provider summary call).
        Every active row with ``id > watermark`` at commit time arrived
        concurrently and must survive the compaction verbatim. Returns 0 for
        an empty/unknown session.
        """
        if not session_id:
            return 0
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(id), 0) FROM messages "
                "WHERE session_id = ? AND active = 1",
                (session_id,),
            ).fetchone()
        return int(row[0]) if row else 0

    def archive_and_compact(
        self,
        session_id: str,
        compacted_messages: List[Dict[str, Any]],
        model_config_patch: Optional[Dict[str, Any]] = None,
        watermark: Optional[int] = None,
        lock_holder: Optional[str] = None,
        tail_count: int = 0,
    ) -> int:
        """Non-destructive in-place compaction for a single durable session id.

        Soft-archives the active messages (``active = 0``) and inserts
        *compacted_messages* as fresh active rows — atomically, in one write
        transaction. The conversation keeps ONE session id for life (#38763)
        WITHOUT destroying history:

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

        Concurrent-append safety (#75316): when *watermark* is provided (the
        value of :meth:`get_active_message_watermark` captured at compression
        START), rows that arrived during the slow provider summary call
        (``id > watermark``) are NOT summarized away. They are re-sequenced
        after the compacted set by a pure-SQL column clone (every column
        except ``id`` — content, api_content, platform_message_id, token
        counts, reasoning sidecars all survive byte-exact, and the FTS
        triggers index the clones naturally), and the originals are archived.
        NOTE: re-sequencing assigns the tail rows fresh ids; consumers that
        reference durable row ids re-resolve by content (see 3e8ab0610).
        ``watermark=None`` preserves the historical archive-everything
        behavior.

        Commit-fence safety: when *lock_holder* is provided, the commit
        verifies INSIDE the transaction that the compression lock is still
        held by that holder and unexpired — a compression whose lease was
        reclaimed (crash cleanup, TTL expiry, competing writer) fails the
        commit instead of clobbering the winner's transcript.

        *tail_count* (default 0) names how many of the LAST rows of
        *compacted_messages* are the verbatim carried-forward tail the
        compressor protected rather than summarized (#86366). Those rows'
        ORIGINALS — which this call archives as a side effect of the blanket
        soft-archive — are superseded byte-identical duplicates, not
        "summarized away" content, so they are stamped rewind-style
        (``active=0, compacted=0``, hidden from search_messages) instead of
        ``compacted=1``. Without this the tail originals satisfy the recall
        filter alongside their live clones and session_search returns every
        carried-forward message once per compaction. Callers that cannot know
        their tail shape keep the historical archive-everything behavior.

        ``message_count`` is set to the ACTIVE count after commit, matching
        what the live load returns. ``model_config_patch`` is merged into the
        session's JSON config in the same transaction; a ``None`` value
        removes that key. Returns the new active count.
        """

        def _do(conn):
            if lock_holder is not None:
                lock_row = conn.execute(
                    "SELECT holder, expires_at FROM compression_locks "
                    "WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if (
                    lock_row is None
                    or lock_row["holder"] != lock_holder
                    or float(lock_row["expires_at"]) <= time.time()
                ):
                    raise SessionCompressionInProgressError(
                        f"Compression lease for {session_id!r} lost before "
                        "commit; refusing to publish a stale compaction"
                    )

            patched_model_config = None
            if model_config_patch is not None:
                # on_missing="raise": a prune/compaction must not commit
                # against a vanished session row (the compressor's caller
                # converts the raised error into a safe keep-the-original
                # no-op), unlike the flag setters which tolerate missing rows.
                patched_model_config = self._merge_model_config_json(
                    conn, session_id, model_config_patch, on_missing="raise"
                )

            # Concurrent tail: active rows that arrived after the watermark.
            # Snapshot their ids and tool_calls now — the clone below needs a
            # stable id list, and the tool-call count keeps sessions.* honest.
            tail_ids: list[int] = []
            tail_tool_calls = 0
            if watermark is not None:
                for row in conn.execute(
                    "SELECT id, tool_calls FROM messages "
                    "WHERE session_id = ? AND active = 1 AND id > ? "
                    "ORDER BY id",
                    (session_id, int(watermark)),
                ).fetchall():
                    tail_ids.append(int(row["id"]))
                    raw = row["tool_calls"]
                    if raw:
                        try:
                            parsed = json.loads(raw) if isinstance(raw, str) else raw
                            tail_tool_calls += len(parsed) if isinstance(parsed, list) else 0
                        except (TypeError, ValueError):
                            pass

            # Soft-archive the live turns: active=0 hides them from the live
            # context load, compacted=1 marks them as "summarized away" (vs
            # rewind/undo's active=0+compacted=0, which means "user took it
            # back"). search_messages includes compacted=1 rows by default so
            # the pre-compaction transcript stays discoverable; live-context
            # loads (active=1 only) still exclude them. Tail originals whose
            # verbatim clones ride inside *compacted_messages* (tail_count)
            # are superseded duplicates instead (#86366): they get the
            # rewind-style flags so they stop matching the recall filter.
            # Rewind-target ids: the originals of the carried-forward tail
            # rows (tail_count), captured BEFORE any flag flips. Named apart
            # from the watermark `tail_ids` below on purpose — the two are
            # different sets (#86366): rewind targets sit AT/BELOW the
            # watermark (the compressor only saw rows up to it), while
            # `tail_ids` are concurrent appends ABOVE it. Without the bound,
            # a concurrent append would steal a LIMIT slot and leave a real
            # carried-forward original stamped compacted=1.
            rewind_tail_ids: Optional[list[int]] = None
            if tail_count > 0:
                if watermark is not None:
                    tail_rows = conn.execute(
                        "SELECT id FROM messages "
                        "WHERE session_id = ? AND active = 1 AND id <= ? "
                        "ORDER BY id DESC LIMIT ?",
                        (session_id, int(watermark), int(tail_count)),
                    ).fetchall()
                else:
                    tail_rows = conn.execute(
                        "SELECT id FROM messages "
                        "WHERE session_id = ? AND active = 1 ORDER BY id DESC LIMIT ?",
                        (session_id, int(tail_count)),
                    ).fetchall()
                rewind_tail_ids = [int(row["id"]) for row in tail_rows]

            # The watermark clone below re-inserts `tail_ids` rows byte-exact
            # as live rows — their originals are the SAME superseded-duplicate
            # class as the carried-forward tail (#86366), so they take the
            # rewind flags too instead of double-matching the recall filter.
            rewind_ids = [*(rewind_tail_ids or []), *tail_ids]

            if rewind_ids:
                placeholders = ",".join("?" for _ in rewind_ids)
                conn.execute(
                    "UPDATE messages SET active = 0, compacted = 0 "
                    f"WHERE session_id = ? AND id IN ({placeholders})",
                    [session_id, *rewind_ids],
                )
                conn.execute(
                    "UPDATE messages SET active = 0, compacted = 1 "
                    "WHERE session_id = ? AND active = 1 "
                    f"AND id NOT IN ({placeholders})",
                    [session_id, *rewind_ids],
                )
            else:
                conn.execute(
                    "UPDATE messages SET active = 0, compacted = 1 "
                    "WHERE session_id = ? AND active = 1",
                    (session_id,),
                )
            inserted, tool_calls_total = self._insert_message_rows(
                conn, session_id, compacted_messages
            )

            if tail_ids:
                # Re-sequence the concurrent tail after the compacted set via
                # a pure-SQL column clone: no decode/re-encode round trip, no
                # field drift — new id, active=1, compacted=0, all else exact.
                placeholders = ",".join("?" for _ in tail_ids)
                clone_cols = [
                    c for c in self._message_column_names(conn)
                    if c not in ("id", "active", "compacted")
                ]
                col_list = ", ".join(clone_cols)
                conn.execute(
                    f"INSERT INTO messages ({col_list}, active, compacted) "
                    f"SELECT {col_list}, 1, 0 FROM messages "
                    f"WHERE id IN ({placeholders}) ORDER BY id",
                    tail_ids,
                )
                inserted += len(tail_ids)
                tool_calls_total += tail_tool_calls

            # message_count / tool_call_count reflect the LIVE (active) set —
            # the archived rows are still on disk but not part of the live count.
            if model_config_patch is None:
                conn.execute(
                    "UPDATE sessions SET message_count = ?, tool_call_count = ? WHERE id = ?",
                    (inserted, tool_calls_total, session_id),
                )
            else:
                conn.execute(
                    "UPDATE sessions SET message_count = ?, tool_call_count = ?, "
                    "model_config = ? WHERE id = ?",
                    (inserted, tool_calls_total, patched_model_config, session_id),
                )
            return inserted

        return self._execute_write(_do)

    def _message_column_names(self, conn) -> List[str]:
        """Column names of the messages table, cached per-connection era."""
        cached = getattr(self, "_message_columns_cache", None)
        if cached:
            return cached
        cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
        self._message_columns_cache = cols
        return cols

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

    def _dedupe_display_generations(self, rows):
        """Collapse compaction generations so each message appears once.

        Compaction epochs copy the protected tail into each new generation, so
        one logical message can exist as several rows (identical
        role/content/timestamp) with different ``active`` flags and ids. A
        display read must surface each exactly once: prefer the live row, then
        the newest generation.

        This is the ONE definition shared by every display projection —
        :meth:`get_messages` (REST), :meth:`get_resume_conversations` and
        :meth:`get_ancestor_display_prefix` (gateway resume), and
        :meth:`get_messages_as_conversation` (warm-session payload) — so the
        surfaces cannot disagree about the same transcript. *rows* must already
        be ordered by ``id``; the returned list keeps that order.
        """
        seen: Dict[Tuple[Any, ...], Any] = {}
        for row in rows:
            dedupe_content = row["content"]
            if row["role"] == "user":
                from agent.context_compressor import split_user_originated_turn

                candidate = {
                    "role": "user",
                    "content": self._decode_content(row["content"]),
                    "display_kind": row["display_kind"],
                    "display_metadata": self._decode_display_metadata(
                        row["display_metadata"]
                    ),
                }
                handoff, live_view = split_user_originated_turn(candidate)
                if handoff is not None and live_view is not None:
                    dedupe_content = self._encode_content(live_view.get("content"))
            # Tool fields participate in the dedupe key: compaction copies them
            # verbatim, so identical tool messages across generations still
            # collapse, while distinct tool calls that happen to share
            # role/content/timestamp are never merged.
            key = (
                row["role"],
                dedupe_content,
                row["timestamp"],
                row["tool_call_id"],
                row["tool_calls"],
                row["tool_name"],
            )
            cur = seen.get(key)
            if cur is None or (row["active"], row["id"]) > (cur["active"], cur["id"]):
                seen[key] = row
        return sorted(seen.values(), key=lambda r: r["id"])

    def get_messages(
        self,
        session_id: str,
        include_inactive: bool = False,
        include_compacted: bool = False,
        limit: Optional[int] = None,
        offset: int = 0,
        latest: bool = False,
        after_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Load messages for a session in insertion order.

        By default only active messages are returned. Pass
        ``include_inactive=True`` to load soft-deleted rows (e.g. for
        audit / debug views of rewound history). See
        :meth:`rewind_to_message` for the soft-delete mechanic.

        Pass ``include_compacted=True`` to additionally load rows preserved
        by in-place context compaction (``active=0, compacted=1``). Those are
        durable display history, not soft-deleted rows — a user-visible
        transcript read must not drop them, or earlier turns silently become
        unreachable once the UI exhausts its active-only window. Soft-deleted
        Undo/Rewind rows (``active=0, compacted=0``) stay excluded; use
        ``include_inactive`` for those.

        Ordered by AUTOINCREMENT id (true insertion order) rather than
        timestamp — see c03acca50 for the WSL2 clock-regression rationale.

        When ``limit`` is provided, returns at most ``limit`` messages
        starting from ``offset`` (0-based, in insertion order). Enables
        pagination for the API endpoint to avoid loading entire transcripts.
        With ``latest=True``, the offset is measured back from the newest
        message and the selected page is still returned in chronological
        order. ``offset`` alone (without ``limit``) also pages — SQLite
        requires a LIMIT clause for OFFSET, so it's emitted as ``LIMIT -1``
        (unbounded).

        ``after_id`` enables keyset pagination (``id > after_id``): O(1)
        page seeks on huge transcripts where OFFSET degrades to O(n) per
        page. Ascending order only (incompatible with ``latest``/``offset``).
        """
        if after_id is not None and (latest or offset):
            raise ValueError("after_id is incompatible with latest/offset paging")
        if after_id is not None and include_compacted:
            raise ValueError("after_id is incompatible with include_compacted (deduped display reads use offset paging)")
        if include_inactive:
            # Audit / debug reads: every row, including soft-deleted.
            active_clause = ""
        elif include_compacted:
            # Display history: active rows plus rows preserved by in-place
            # compaction (active=0, compacted=1), but never soft-deleted
            # Undo/Rewind rows (active=0, compacted=0).
            active_clause = " AND (active = 1 OR compacted = 1)"
        else:
            active_clause = " AND active = 1"
        keyset_clause = " AND id > ?" if after_id is not None else ""
        sql = (
            "SELECT * FROM messages WHERE session_id = ?"
            f"{active_clause}{keyset_clause} ORDER BY id {'DESC' if latest else 'ASC'}"
        )
        params: list = [session_id]
        if after_id is not None:
            params.append(after_id)
        if include_compacted:
            # Read the full display set (a session's rows are bounded; the
            # UI-level 500-row cap lives in the endpoint, not here), dedupe
            # generations, then apply paging.
            with self._read_ctx() as conn:
                cursor = conn.execute(
                    "SELECT * FROM messages WHERE session_id = ?" + active_clause
                    + " ORDER BY id ASC",
                    [session_id],
                )
                all_rows = cursor.fetchall()
            rows = self._dedupe_display_generations(all_rows)
            if latest:
                rows = rows[::-1]
            rows = rows[offset:]
            if limit is not None:
                rows = rows[:limit]
            if latest:
                rows = rows[::-1]
        else:
            if limit is not None or offset:
                # SQLite's OFFSET requires LIMIT; -1 means "no limit".
                sql += " LIMIT ? OFFSET ?"
                params.extend([-1 if limit is None else limit, offset])
            with self._read_ctx() as conn:
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
            if latest:
                rows.reverse()
        result = []
        for row in rows:
            msg = dict(row)
            if msg.pop("_compressed_summary", 0):
                msg["_compressed_summary"] = True
            if "content" in msg:
                msg["content"] = self._decode_content(msg["content"])
            if msg.get("tool_calls"):
                try:
                    msg["tool_calls"] = json.loads(msg["tool_calls"])
                except (json.JSONDecodeError, TypeError):
                    logger.warning("Failed to deserialize tool_calls in get_messages, falling back to []")
                    msg["tool_calls"] = []
            if msg.get("display_metadata") is not None:
                msg["display_metadata"] = self._decode_display_metadata(msg["display_metadata"])
            result.append(msg)
        return result

    def find_pr_url_messages(self, session_ids: List[str]) -> List[Dict[str, Any]]:
        """Tool results in these sessions that mention a GitHub PR url.

        A candidate scan, deliberately loose: it hands back every tool result
        containing ``/pull/`` and leaves the caller to decide which ones make a
        claim (see the desktop's PR recovery, which only accepts an output that
        is a bare PR url — the signature of ``gh pr create``). Ordered
        oldest-first per session so the caller can take the last match.
        """
        found: List[Dict[str, Any]] = []
        ids = [s for s in session_ids if s]
        for start in range(0, len(ids), 900):  # SQLite's bound-variable ceiling.
            chunk = ids[start : start + 900]
            placeholders = ",".join("?" * len(chunk))
            with self._read_ctx() as conn:
                rows = conn.execute(
                    f"""SELECT session_id, content FROM messages
                        WHERE session_id IN ({placeholders})
                          AND role = 'tool' AND content LIKE '%/pull/%'
                        ORDER BY id ASC""",
                    chunk,
                ).fetchall()
            found.extend({"session_id": row[0], "content": row[1]} for row in rows)
        return found

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
        with self._read_ctx() as conn:
            # Confirm the anchor exists in this session.
            anchor_exists = conn.execute(
                "SELECT 1 FROM messages WHERE id = ? AND session_id = ? LIMIT 1",
                (around_message_id, session_id),
            ).fetchone()
            if not anchor_exists:
                return {"window": [], "messages_before": 0, "messages_after": 0}

            # Two queries: anchor + before (DESC, take window+1), and after
            # (ASC, take window). Final order is id ASC.
            before_rows = conn.execute(
                "SELECT * FROM messages "
                "WHERE session_id = ? AND id <= ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, around_message_id, window + 1),
            ).fetchall()
            after_rows = conn.execute(
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
            if msg.get("display_metadata") is not None:
                msg["display_metadata"] = self._decode_display_metadata(msg["display_metadata"])
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

        with self._read_ctx() as conn:
            current = session_id
            seen = {current}
            best = None  # tracks the last (deepest) node with messages

            for _ in range(32):
                # Check if the current node has messages.
                try:
                    row = conn.execute(
                        "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1",
                        (current,),
                    ).fetchone()
                except Exception:
                    return session_id
                if row is not None:
                    best = current

                # Walk to the most-recently-started child — but skip explicit
                # branch (`_branched_from`), delegate/subagent (`_delegate_from`),
                # reset-continuation (`_reset_from` or the legacy same-key
                # heuristic — a post-reset conversation must never be reached
                # by resuming the parent the user reset away), and tool
                # children. They also carry a ``parent_session_id`` yet
                # are NOT compression continuations; following them would hijack
                # the resume target to an unrelated session (e.g. a subagent
                # run). This mirrors the child-exclusion in ``get_compression_tip``.
                try:
                    child_row = conn.execute(
                        "SELECT id FROM sessions AS child "
                        "WHERE child.parent_session_id = ? "
                        "  AND json_extract(COALESCE(child.model_config, '{}'), '$._branched_from') IS NULL "
                        "  AND json_extract(COALESCE(child.model_config, '{}'), '$._delegate_from') IS NULL "
                        "  AND json_extract(COALESCE(child.model_config, '{}'), '$._reset_from') IS NULL "
                        f"  AND NOT {_legacy_reset_child_sql('child', _RESET_END_REASONS_SQL)} "
                        "  AND COALESCE(child.source, '') != 'tool' "
                        "ORDER BY child.started_at DESC, child.id DESC LIMIT 1",
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
        include_row_ids: bool = False,
        include_compacted: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Load messages in the OpenAI conversation format (role + content dicts).
        Used by the gateway to restore conversation history.

        By default only active messages are returned. Pass
        ``include_inactive=True`` to load soft-deleted (rewound) rows
        as well. See :meth:`rewind_to_message`.

        ``include_compacted=True`` additionally loads rows preserved by
        in-place compaction (``active=0, compacted=1``), deduped by
        :meth:`_dedupe_display_generations`. DISPLAY reads want this; the
        model-fed restore must NOT pass it, or a resumed session regrows the
        very history compaction just summarized away.

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
        if include_ancestors and not self._is_explicit_branch_session(session_id):
            session_ids = self._session_lineage_root_to_tip(session_id)

        if include_inactive:
            active_clause = ""
        elif include_compacted:
            active_clause = " AND (active = 1 OR compacted = 1)"
        else:
            active_clause = " AND active = 1"
        with self._read_ctx() as conn:
            placeholders = ",".join("?" for _ in session_ids)
            rows = conn.execute(
                f"SELECT {self._CONVERSATION_ROW_COLUMNS} "
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

        if include_compacted:
            rows = self._dedupe_display_generations(rows)

        return self._rows_to_conversation(
            rows,
            session_id=session_id,
            include_ancestors=include_ancestors,
            repair_alternation=repair_alternation,
            include_row_ids=include_row_ids,
        )

    # Columns every conversation projection decodes. Shared by
    # get_messages_as_conversation and get_resume_conversations so a single
    # SELECT can feed both the model-fed and display views. ``active`` rides
    # along so a display read can split the compaction-archived rows from the
    # live set (and feed _dedupe_display_generations) without a second query.
    _CONVERSATION_ROW_COLUMNS = (
        "id, role, content, tool_call_id, tool_calls, tool_name, effect_disposition, "
        "finish_reason, reasoning, reasoning_content, reasoning_details, "
        "codex_reasoning_items, codex_message_items, platform_message_id, observed, "
        "_compressed_summary, timestamp, active, "
        "api_content, display_kind, display_metadata"
    )

    def _rows_to_conversation(
        self,
        rows,
        *,
        session_id: str,
        include_ancestors: bool,
        repair_alternation: bool,
        include_row_ids: bool = False,
        include_summary_markers: bool = False,
    ) -> List[Dict[str, Any]]:
        """Decode fetched message rows into the OpenAI conversation format.

        Extracted from get_messages_as_conversation so get_resume_conversations
        can build the model-fed and display views from one SELECT. ``rows`` must
        already be ordered by ``id`` (insertion order) and filtered to the
        desired session set / active state by the caller.
        """
        messages = []
        # Watermark rotation column-clones concurrent tail rows into the child
        # after the new summary, so the copies need not be adjacent. Index the
        # exact durable clone identity while decoding instead of rescanning the
        # whole accumulated lineage for every user row.
        exact_user_clones: Dict[Tuple[Any, str], Dict[str, Any]] = {}
        for row in rows:
            content = self._decode_content(row["content"])
            if row["role"] in {"user", "assistant"} and isinstance(content, str):
                content = sanitize_context(content).strip()
            msg = {"role": row["role"], "content": content}
            # Born durable (#92231): this dict is materialized FROM a durable
            # row, so stamp the persistence marker at the source instead of
            # relying on every restore caller to thread the loaded list back
            # through a flush as ``conversation_history=`` — any
            # identity-losing handoff (compression's durable-snapshot
            # adoption, incremental persists with no history arg) would
            # otherwise re-append the ENTIRE transcript on flush.
            # Underscore-prefixed like ``_row_id``: every transport strips it
            # before the wire, and compression's assembly copies deliberately
            # strip it so rotated child handoffs still flush (see
            # _fresh_compaction_message_copy).
            msg[_DB_PERSISTED_MARKER_KEY] = True
            # Durable per-message identity for surfaces that need to address a
            # specific row later (desktop reactions). OPT-IN: only the gateway
            # asks for it — every other consumer (ACP restore, export,
            # inspection) gets the transcript in its historical shape.
            # Underscore-prefixed so every transport's convert_messages()
            # strips it before the wire.
            if include_row_ids and row["id"] is not None:
                msg["_row_id"] = row["id"]
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
                decoded = self._decode_display_metadata(row["display_metadata"])
                if decoded is not None:
                    msg["display_metadata"] = decoded
            if include_summary_markers and row["_compressed_summary"]:
                msg["_compressed_summary"] = True
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
            if include_ancestors:
                canonical_content, _is_composite = (
                    self._canonical_replayed_user_content(msg)
                )
                exact_clone_key = self._exact_replayed_user_clone_key(
                    msg.get("timestamp"), canonical_content
                )
                previous_exact = (
                    exact_user_clones.get(exact_clone_key)
                    if exact_clone_key is not None
                    else None
                )
                duplicate = None
                if previous_exact is not None:
                    previous_index = next(
                        (
                            index
                            for index, candidate in enumerate(messages)
                            if candidate is previous_exact
                        ),
                        None,
                    )
                    if previous_index is not None:
                        duplicate = (previous_index, True)
                if duplicate is None:
                    duplicate = self._find_duplicate_replayed_user_message(
                        messages, msg
                    )
                if duplicate is not None:
                    duplicate_index, prefer_current = duplicate
                    if prefer_current:
                        # A rotated compression child can carry the same live
                        # ask as the parent row plus the only surviving summary
                        # scaffold. Keep the child carrier (and its durable row
                        # id), not the simpler ancestor copy.
                        messages.pop(duplicate_index)
                    else:
                        continue
            messages.append(msg)
            if include_ancestors and exact_clone_key is not None:
                exact_user_clones[exact_clone_key] = msg
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
        # DEFENSE-IN-DEPTH against #78148: before that fix, a bare tool-call
        # marker (e.g. "[memory]") could get cached as a fallback and
        # persisted as if it were the model's real answer. Sessions written
        # before the fix can still carry those rows — clear the stray
        # content on load so replaying history doesn't re-teach the model
        # to keep emitting the marker. No-op for unaffected sessions.
        messages = _strip_stale_tool_call_markers(messages)
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
        - ``display_history`` — the full compression lineage (ancestors → tip),
          verbatim, with replayed-user dedup. Explicit ``/branch`` sessions are
          excluded from this lineage because their own rows already contain the
          copied transcript; including the live parent's rows would let messages
          written to the original after the fork leak into the branch.

        The display projection also includes rows preserved by IN-PLACE
        compaction (``active=0, compacted=1``), deduped by
        :meth:`_dedupe_display_generations`. Without them a compacted
        conversation resumes showing only its summary plus the carried-forward
        tail — the user's own turns read as deleted even though every row is
        still on disk, and the REST transcript read (which has always included
        them) disagreed with this one about the same session (#92080).

        The display fetch already reads a superset of the model fetch (the tip
        rows are part of the lineage), so serving both from one lineage SELECT
        halves the resume's DB work versus two separate calls, with byte-identical
        output (see test_get_resume_conversations_matches_separate_reads).
        """
        session_ids = self._resume_lineage_ids(session_id)
        with self._read_ctx() as conn:
            placeholders = ",".join("?" for _ in session_ids)
            rows = conn.execute(
                f"SELECT session_id, {self._CONVERSATION_ROW_COLUMNS} "
                f"FROM messages WHERE session_id IN ({placeholders}) "
                # Compaction-archived rows (active=0, compacted=1) are display
                # history; Undo/Rewind rows (active=0, compacted=0) are not.
                "AND (active = 1 OR compacted = 1) "
                # ORDER BY id (insertion order) — see get_messages_as_conversation
                # for why timestamp ordering is unsafe.
                "ORDER BY id",
                tuple(session_ids),
            ).fetchall()

        # Tip rows are exactly the model-fed set (get_messages_as_conversation
        # with session_ids=[session_id]); filtering the lineage fetch preserves
        # their relative id order. The model projection stays active-only — it
        # is the compressed working context and must not regrow the history
        # compaction just summarized away.
        tip_rows = [r for r in rows if r["session_id"] == session_id and r["active"]]
        model_history = self._rows_to_conversation(
            tip_rows,
            session_id=session_id,
            include_ancestors=False,
            repair_alternation=True,
            include_row_ids=True,
            # Pre-compress checkpointing: the resumed model history must keep
            # the summary marker so checkpoint providers can exclude derivative
            # summaries after a process restart (marker survives restart).
            include_summary_markers=True,
        )
        display_history = self._rows_to_conversation(
            self._dedupe_display_generations(rows),
            session_id=session_id,
            include_ancestors=True,
            repair_alternation=False,
            include_row_ids=True,
        )
        return model_history, display_history

    def _resume_lineage_ids(self, session_id: str) -> List[str]:
        """Session ids a full (display) resume materializes for *session_id*.

        Compression continuations need their ended ancestors' rows for the
        display transcript; an explicit ``/branch`` copy already owns its
        transcript, so its lineage is itself alone. This is the ONE definition
        shared by the resume readers (``get_resume_conversations``,
        ``get_ancestor_display_prefix``) and the resume guard
        (``assert_resume_safe`` / ``get_resume_message_count``) — the guard must
        count exactly the rows a resume would load, never a superset.
        """
        if self._is_explicit_branch_session(session_id):
            return [session_id]
        return self._session_lineage_root_to_tip(session_id)

    def get_resume_message_count(
        self, session_id: str, *, tip_only: bool = False
    ) -> int:
        """Count the rows that a resume would materialize.

        ``tip_only=True`` counts the tip segment's ACTIVE rows — the set a
        model-history restore loads (``get_messages_as_conversation`` without
        ancestors, or the deferred Desktop resume that pages the display
        transcript over REST and never materializes the ancestor prefix in
        memory).

        Otherwise this counts the full-lineage DISPLAY set — active rows plus
        the compaction-archived rows ``get_resume_conversations`` now loads
        for the transcript. Counting only active rows here would let a
        heavily-compacted conversation pass a limit sized for a handful of
        live rows and then materialize tens of thousands.
        """
        session_ids = [session_id] if tip_only else self._resume_lineage_ids(session_id)
        active_clause = "active = 1" if tip_only else "(active = 1 OR compacted = 1)"
        placeholders = ",".join("?" for _ in session_ids)
        with self._read_ctx() as conn:
            row = conn.execute(
                f"SELECT COUNT(*) FROM messages "
                f"WHERE session_id IN ({placeholders}) AND {active_clause}",
                tuple(session_ids),
            ).fetchone()
        return int(row[0] if row else 0)

    def assert_resume_safe(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
        *,
        tip_only: bool = False,
    ) -> int:
        """Return resume row count or reject a transcript too large to load.

        ``max_messages=None`` resolves the limit from config
        (``sessions.max_resume_messages``); 0 disables the guard and returns
        the (bounded) count without raising.

        ``tip_only=True`` bounds only the tip segment's ACTIVE rows, for
        callers that never materialize the ancestor lineage or the
        compaction archive in memory (tip-only model restore, deferred
        Desktop resume whose display history is REST-paginated). A
        heavily-compressed conversation — 85 compaction segments and ~29k
        lineage rows behind a ~700-row tip — is exactly the shape compression
        is supposed to produce; counting its whole lineage against a limit
        sized for in-memory materialization rejected the healthiest sessions
        (Desktop Bot Chat stuck on "Waking up…" with code 4130) while the
        process would only ever have held the tip.

        The full (non-``tip_only``) bound counts the DISPLAY set — active plus
        compaction-archived rows — because that is what
        ``get_resume_conversations`` materializes for the transcript.
        """
        if max_messages is None:
            max_messages = resolved_max_resume_messages()
        if max_messages < 0:
            raise ValueError("max_messages must be non-negative")
        if max_messages == 0:
            # Guard disabled by config — skip counting entirely. Every live
            # caller invokes this for its raise side effect and ignores the
            # return value, and an unbounded lineage COUNT here would do the
            # exact pathological work the disable exists to avoid.
            return 0
        session_ids = [session_id] if tip_only else self._resume_lineage_ids(session_id)
        active_clause = "active = 1" if tip_only else "(active = 1 OR compacted = 1)"
        placeholders = ",".join("?" for _ in session_ids)
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM ("
                f"SELECT 1 FROM messages WHERE session_id IN ({placeholders}) "
                f"AND {active_clause} LIMIT ?"
                ")",
                (*session_ids, max_messages + 1),
            ).fetchone()
        message_count = int(row[0] if row else 0)
        if message_count > max_messages:
            raise SessionResumeTooLargeError(
                message_count,
                max_messages,
                scope="in its tip segment" if tip_only else "across its lineage",
            )
        return message_count

    def assert_export_safe(
        self,
        session_id: str,
        max_messages: Optional[int] = None,
    ) -> int:
        """Return active row count or reject an unsafe in-memory export.

        Exporting one session does not include compression ancestors, so this
        guard deliberately counts only the requested segment. The limited
        subquery stops as soon as it proves the transcript exceeds the bound.

        ``max_messages=None`` resolves the limit from config
        (``sessions.max_export_messages``); 0 disables the guard and returns
        the active row count without raising.
        """
        if max_messages is None:
            max_messages = resolved_max_export_messages()
        if max_messages < 0:
            raise ValueError("max_messages must be non-negative")
        if max_messages == 0:
            # Guard disabled by config — skip the COUNT; live callers use
            # this for its raise side effect only (and skip calling it
            # entirely when the limit is 0).
            return 0
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM ("
                "SELECT 1 FROM messages WHERE session_id = ? AND active = 1 LIMIT ?"
                ")",
                (session_id, max_messages + 1),
            ).fetchone()
        message_count = int(row[0] if row else 0)
        if message_count > max_messages:
            raise SessionExportTooLargeError(session_id, message_count, max_messages)
        return message_count

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
        session_ids = self._resume_lineage_ids(session_id)
        if len(session_ids) <= 1:
            return []
        with self._read_ctx() as conn:
            placeholders = ",".join("?" for _ in session_ids)
            rows = conn.execute(
                f"SELECT session_id, {self._CONVERSATION_ROW_COLUMNS} "
                f"FROM messages WHERE session_id IN ({placeholders}) "
                # Display read: compaction-archived rows included, Undo/Rewind
                # rows excluded (see get_resume_conversations).
                "AND (active = 1 OR compacted = 1) "
                "ORDER BY id",
                tuple(session_ids),
            ).fetchall()
        rows = self._dedupe_display_generations(rows)
        ancestor_ids = {
            int(row["id"])
            for row in rows
            if row["session_id"] != session_id and row["id"] is not None
        }
        if not ancestor_ids:
            return []
        lineage = self._rows_to_conversation(
            rows,
            session_id=session_id,
            include_ancestors=True,
            repair_alternation=False,
            include_row_ids=True,
        )
        prefix: List[Dict[str, Any]] = []
        for message in lineage:
            if message.get("_row_id") not in ancestor_ids:
                continue
            projected = message.copy()
            projected.pop("_row_id", None)
            prefix.append(projected)
        return prefix

    def _is_explicit_branch_session(self, session_id: str) -> bool:
        """Return whether *session_id* is a copied user-facing branch.

        Branches and compression continuations both use ``parent_session_id``,
        but they have different history semantics: a branch owns a copied
        transcript, while a compression continuation needs its ended parent's
        archived rows for display. The durable ``_branched_from`` marker is the
        existing discriminator written by all branch creation paths.
        """
        if not session_id:
            return False
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT model_config FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return False
        raw_config = row["model_config"] if hasattr(row, "keys") else row[0]
        if not raw_config:
            return False
        try:
            config = json.loads(raw_config) if isinstance(raw_config, str) else raw_config
        except (json.JSONDecodeError, TypeError):
            return False
        return isinstance(config, dict) and bool(config.get("_branched_from"))

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
        with self._read_ctx() as conn:
            for _ in range(100):
                if not current or current in seen:
                    break
                seen.add(current)
                chain.append(current)
                row = conn.execute(
                    "SELECT parent_session_id FROM sessions WHERE id = ?",
                    (current,),
                ).fetchone()
                if row is None:
                    break
                current = row["parent_session_id"] if hasattr(row, "keys") else row[0]
        return list(reversed(chain)) or [session_id]

    @staticmethod
    def _canonical_replayed_user_content(
        msg: Dict[str, Any],
    ) -> Tuple[Any, bool]:
        """Return canonical live content and whether *msg* is composite."""
        if msg.get("role") != "user":
            return None, False

        from agent.context_compressor import split_user_originated_turn

        handoff, live_view = split_user_originated_turn(msg)
        is_composite = handoff is not None and live_view is not None
        return (
            live_view.get("content")
            if is_composite and live_view is not None
            else msg.get("content"),
            is_composite,
        )

    @staticmethod
    def _exact_replayed_user_clone_key(
        timestamp: Any, content: Any
    ) -> Optional[Tuple[Any, str]]:
        """Return a hashable key for a column-exact rotation clone."""
        if timestamp is None or content in (None, "", []):
            return None
        try:
            encoded = json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError):
            return None
        return timestamp, encoded

    @staticmethod
    def _find_duplicate_replayed_user_message(
        messages: List[Dict[str, Any]], msg: Dict[str, Any]
    ) -> Optional[Tuple[int, bool]]:
        """Return an adjacent replay duplicate and whether *msg* must win.

        Compression rotation may persist the current ask once in the parent
        and again inside a composite child carrier. Compare the canonical live
        payload for that carrier, while retaining the historical exact-string
        dedupe for ordinary replayed users. The child carrier wins because it
        owns both the current durable row identity and the retained scaffold.
        """
        if msg.get("role") != "user":
            return None

        content, prefer_current = SessionDB._canonical_replayed_user_content(msg)
        if content in (None, "", []):
            return None

        for index in range(len(messages) - 1, -1, -1):
            prev = messages[index]
            if prev.get("role") == "user":
                prev_content, prev_is_composite = (
                    SessionDB._canonical_replayed_user_content(prev)
                )
                if prev_content == content and (
                    prefer_current
                    or prev_is_composite
                    or isinstance(content, str)
                ):
                    return index, prefer_current
            if prev.get("role") == "assistant" and (prev.get("content") or prev.get("tool_calls")):
                return None
        return None

    @staticmethod
    def _is_duplicate_replayed_user_message(
        messages: List[Dict[str, Any]], msg: Dict[str, Any]
    ) -> bool:
        return SessionDB._find_duplicate_replayed_user_message(messages, msg) is not None

    # =========================================================================
    # Rewind (soft-delete) — see /rewind slash command + issue #21910
    # =========================================================================

    def get_active_message_ids(self, session_id: str) -> List[int]:
        """Return the ordered physical ids pinned by rewind CAS checks.

        Conversation projections intentionally omit legacy background-review
        harness rows.  Destructive rewinds must nevertheless pin every active
        physical row so the caller snapshot matches the transaction-local
        comparison in :meth:`rewind_to_message`.
        """
        with self._read_ctx() as conn:
            rows = conn.execute(
                "SELECT id FROM messages "
                "WHERE session_id = ? AND active = 1 ORDER BY id",
                (session_id,),
            ).fetchall()
        return [int(row[0]) for row in rows]

    @staticmethod
    def _active_transcript_counts(conn, session_id: str) -> tuple[int, int]:
        """Return active message/tool-call counts inside the caller's txn."""
        rows = conn.execute(
            "SELECT tool_calls FROM messages "
            "WHERE session_id = ? AND active = 1",
            (session_id,),
        ).fetchall()
        tool_call_count = 0
        for row in rows:
            raw = row[0]
            if not raw:
                continue
            try:
                decoded = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(decoded, list):
                tool_call_count += len(decoded)
            elif decoded:
                tool_call_count += 1
        return len(rows), tool_call_count

    def rewind_to_message(
        self,
        session_id: str,
        target_message_id: int,
        *,
        preserve_compaction_handoff: bool = False,
        expected_active_ids: Optional[List[int]] = None,
        expected_target_content: Any = None,
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
        *session_id* or if its role is not ``"user"``.  With
        ``preserve_compaction_handoff=True``, a composite summary carrier is
        split inside the same write transaction: its original row is archived
        and its canonical hidden handoff scaffold is inserted as the new head.
        That opt-in result also contains ``replacement_message_id``.

        ``expected_active_ids`` optionally pins the ordered active row set.
        ``expected_target_content`` additionally pins the selected canonical
        live-user payload.  Both checks run inside the write transaction before
        any row or counter mutation.  Presentation-only metadata changes (for
        example Desktop reactions) deliberately do not invalidate a rewind.
        A live cross-process turn lease always refuses the rewind; expired or
        provably dead holders are reclaimed inside the mutation transaction.

        Always increments ``sessions.rewind_count`` — even when the
        target is already inactive — so the counter accurately reflects
        the number of rewind operations performed against the session.
        Idempotent on the ``active`` flag: re-rewinding past the same
        target is a no-op on row state but still bumps the counter.
        """

        def _do(conn):
            # Rewind changes the active transcript and must honor the same
            # compression/closed-parent and cross-process turn guards as
            # append writers.
            self._check_transcript_write_guards(
                conn,
                session_id,
                None,
                reject_active_turn_lease=True,
                reject_active_compression_lock=True,
            )

            if expected_active_ids is not None:
                active_rows = conn.execute(
                    "SELECT id FROM messages "
                    "WHERE session_id = ? AND active = 1 ORDER BY id",
                    (session_id,),
                ).fetchall()
                active_ids = [int(active_row[0]) for active_row in active_rows]
                if active_ids != expected_active_ids:
                    raise RuntimeError(
                        "active transcript changed before the rewind could be persisted"
                    )

            row = conn.execute(
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

            replacement_message_id: Optional[int] = None
            replacement: Optional[Dict[str, Any]] = None
            if preserve_compaction_handoff or expected_target_content is not None:
                if not target_row.get("active"):
                    raise ValueError("rewind target is not active")
                from agent.context_compressor import split_user_originated_turn

                split_target = target_row.copy()
                split_target["content"] = self._decode_content(
                    split_target.get("content")
                )
                split_target["display_metadata"] = self._decode_display_metadata(
                    split_target.get("display_metadata")
                )
                handoff, live_view = split_user_originated_turn(split_target)
                if live_view is None:
                    raise ValueError("rewind target is not a user-originated turn")
                live_content = live_view.get("content")
                if isinstance(live_content, str):
                    live_content = sanitize_context(live_content).strip()
                if (
                    expected_target_content is not None
                    and live_content != expected_target_content
                ):
                    raise RuntimeError(
                        "rewind target changed before it could be persisted"
                    )
                if preserve_compaction_handoff and handoff is None:
                    raise ValueError(
                        "preserve_compaction_handoff requires an active composite carrier"
                    )
                replacement = handoff if preserve_compaction_handoff else None

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
            if replacement is not None:
                self._insert_message_rows(conn, session_id, [replacement])
                inserted = conn.execute("SELECT last_insert_rowid()").fetchone()
                replacement_message_id = int(inserted[0])
            conn.execute(
                "UPDATE sessions SET rewind_count = COALESCE(rewind_count, 0) + 1 "
                "WHERE id = ?",
                (session_id,),
            )
            message_count, tool_call_count = self._active_transcript_counts(
                conn, session_id
            )
            conn.execute(
                "UPDATE sessions SET message_count = ?, tool_call_count = ? "
                "WHERE id = ?",
                (message_count, tool_call_count, session_id),
            )
            head_row = conn.execute(
                "SELECT MAX(id) FROM messages WHERE session_id = ? AND active = 1",
                (session_id,),
            ).fetchone()
            new_head_id = (
                head_row[0] if head_row and head_row[0] is not None else None
            )
            return target_row, ids, new_head_id, replacement_message_id

        target_row, rewound, new_head_id, replacement_message_id = (
            self._execute_write(_do)
        )

        # Decode content for callers (prefill the prompt buffer) without a
        # second fallible database operation after the transaction commits.
        target_row["content"] = self._decode_content(target_row.get("content"))

        result = {
            "rewound_count": len(rewound),
            "target_message": target_row,
            "new_head_id": new_head_id,
        }
        if preserve_compaction_handoff:
            result["replacement_message_id"] = replacement_message_id
        return result

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

    # =========================================================================
    # Search
    # =========================================================================

    def search_sessions(
        self,
        source: str = None,
        limit: int = 20,
        offset: int = 0,
        workspace_key: str = None,
    ) -> List[Dict[str, Any]]:
        """List sessions, optionally filtered by source.

        Returns rows enriched with a computed ``last_active`` column
        (freshest of ``last_activity_at`` and latest message timestamp,
        else ``started_at``), ordered by most-recently-used first.

        Pass ``workspace_key`` to scope rows to one workspace - matching
        :func:`workspace_key` semantics (git repo root, else cwd). Used by
        ``hermes -c``/``--resume`` so the "last" session is the last one in
        the *current* workspace, not the global MRU.
        """
        select_with_last_active = (
            "SELECT s.*, "
            "COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved, "
            f"{_sql_session_last_active('s')} AS last_active "
            "FROM sessions s "
            "LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
        )
        where_clauses = []
        params: list = []
        if source:
            where_clauses.append("s.source = ?")
            params.append(source)
        if workspace_key:
            ws_clause, ws_params = _workspace_key_clause(workspace_key)
            where_clauses.append(ws_clause)
            params.extend(ws_params)
        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        params.extend([limit, offset])
        with self._read_ctx() as conn:
            cursor = conn.execute(
                f"{select_with_last_active}"
                f"{where_sql} "
                "ORDER BY last_active DESC, s.started_at DESC, s.id DESC LIMIT ? OFFSET ?",
                params,
            )
            return [self._session_row_dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Utility
    # =========================================================================

    def session_count(
        self,
        source: str = None,
        sources: List[str] = None,
        cwd_prefix: str = None,
        min_message_count: int = 0,
        include_archived: bool = False,
        archived_only: bool = False,
        exclude_children: bool = False,
        exclude_sources: List[str] = None,
    ) -> int:
        """Count sessions, optionally filtered by source.

        Pass ``exclude_children=True`` to count only the conversations that
        ``list_sessions_rich`` surfaces (root + branch/reset sessions), hiding
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
            # count lines up with the rows: roots plus user-visible branch/reset
            # children.
            where_clauses.append(_LISTABLE_CHILD_SQL)
            where_clauses.append(f"{_delegate_from_json('s.model_config')} IS NULL")
        include_sources = [source] if source else list(sources or [])
        if include_sources:
            placeholders = ",".join("?" for _ in include_sources)
            where_clauses.append(f"s.source IN ({placeholders})")
            params.extend(include_sources)
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

        with self._read_ctx() as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM sessions s{where_sql}", params)
            return cursor.fetchone()[0]

    def session_count_ge(self, n: int = 1) -> bool:
        """Check if at least N sessions exist (archived included).

        Short-circuits via LIMIT — much cheaper than ``session_count()``,
        which pays a full index scan for its default ``archived = 0``
        filter (measured 543us vs 4us on a 20k-session DB). Archived
        sessions count: every caller so far asks "has this install ever
        had sessions", and an archived session is still a created one.
        Use this instead of ``session_count() >= n`` when the exact count
        is irrelevant.
        """
        with self._read_ctx() as conn:
            cursor = conn.execute("SELECT 1 FROM sessions LIMIT ?", (n,))
            rows = cursor.fetchall()
        return len(rows) >= n

    def session_count_by_source(
        self,
        *,
        include_archived: bool = False,
        archived_only: bool = False,
        exclude_children: bool = False,
    ) -> Dict[str, int]:
        """Return a ``{source: count}`` dict via a single ``GROUP BY`` query.

        Replaces the O(N) ``list_sessions_rich`` histogram loop with an
        aggregate query. When ``exclude_children`` is False the query uses
        ``idx_sessions_source``; when True, the child-exclusion predicates
        require a full table scan (same as ``session_count`` and
        ``list_sessions_rich``).

        ``exclude_children=True`` mirrors ``list_sessions_rich`` visibility
        (roots + branch/reset sessions, excluding sub-agent runs, delegates,
        and compression continuations) so the source counts match what the
        Sessions page actually lists.
        """
        where_clauses = []
        params: list = []

        if exclude_children:
            where_clauses.append(_LISTABLE_CHILD_SQL)
            where_clauses.append(f"{_delegate_from_json('s.model_config')} IS NULL")
        if archived_only:
            where_clauses.append("s.archived = 1")
        elif not include_archived:
            where_clauses.append("s.archived = 0")

        where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with self._read_ctx() as conn:
            if self._conn is None:
                raise RuntimeError("SessionDB connection is closed")
            rows = conn.execute(
                "SELECT COALESCE(NULLIF(s.source, ''), 'cli') AS source, COUNT(*) AS count "
                f"FROM sessions s{where_sql} "
                "GROUP BY COALESCE(NULLIF(s.source, ''), 'cli') "
                "ORDER BY count DESC",
                params,
            ).fetchall()
        return {str(row["source"]): int(row["count"] or 0) for row in rows}

    def message_count(self, session_id: str = None) -> int:
        """Count messages, optionally for a specific session."""
        with self._read_ctx() as conn:
            if session_id:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM messages")
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
        with self._read_ctx() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM messages "
                "WHERE session_id = ? AND platform_message_id = ? LIMIT 1",
                (session_id, platform_message_id),
            )
            return cursor.fetchone() is not None

    # =========================================================================
    # Export and cleanup
    # =========================================================================

    def _is_explicit_fork_child_row(self, session: Dict[str, Any]) -> bool:
        """True when ``session`` is a branch, delegate, or tool child of its parent.

        Markers only count as a fork when they point at ``parent_session_id``.
        Compression copies ``model_config`` onto the continuation
        (``publish_compression_child`` callers pass
        ``agent._session_init_model_config``), so a delegate's continuation
        carries ``_delegate_from=<the delegate's own parent>``. Presence-only
        matching would treat that real continuation as a fork — the same
        misclassification ``_NON_CONTINUATION_CHILD_FILTER_SQL`` already
        avoids by binding both markers to the queried parent.
        """
        if session.get("source") == "tool":
            return True
        raw = session.get("model_config")
        if not raw:
            return False
        try:
            cfg = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, json.JSONDecodeError):
            return False
        if not isinstance(cfg, dict):
            return False
        parent_id = session.get("parent_session_id")
        branched = cfg.get("_branched_from")
        delegated = cfg.get("_delegate_from")
        if parent_id:
            return branched == parent_id or delegated == parent_id
        return branched is not None or delegated is not None

    def is_explicit_fork_child(self, session_id: str) -> bool:
        """True when ``session_id`` is a /branch, delegate, or tool child row.

        Read-only public view of :meth:`_is_explicit_fork_child_row` for
        callers that must respect the fork boundary without re-implementing
        its marker rules (``agent/prompt_cache_scope.py`` keeps a declared
        conversation key from crossing it). A missing row is not a fork.
        """
        session = self.get_session(session_id)
        return bool(session and self._is_explicit_fork_child_row(session))

    def declared_scope_identity(self, session_id: str) -> Tuple[bool, str]:
        """Fork verdict and recorded ``source`` for *session_id*, in ONE read.

        ``agent/prompt_cache_scope.py`` needs both to resolve a host-declared
        conversation scope, and both live on the same ``sessions`` row; asking
        for them separately read that row twice per resolution (@teknium1 on
        #98811).  The marker rules stay here, beside
        :meth:`is_explicit_fork_child`, instead of being re-implemented by the
        caller.

        A missing row is not a fork and has no source, which is the right
        answer before ``_ensure_db_session`` persists it.  This raises whatever
        :meth:`get_session` raises: the caller fails closed on a DB error, and
        merging the two reads cannot weaken that, because the fork check was
        already the first of the two.
        """
        session = self.get_session(session_id)
        if not session:
            return False, ""
        return (
            self._is_explicit_fork_child_row(session),
            str(session.get("source") or "").strip(),
        )

    def latest_conversation_boundary(
        self, session_key: str, source: str
    ) -> Optional[int]:
        """How many conversation boundaries this routing peer has crossed.

        A boundary is a row this peer ended at an intentional conversation
        break — the ``_RESET_END_REASONS`` set (``/new``, ``/switch``, idle,
        daily, suspended, resume_pending_expired).  That is the same fence
        :meth:`find_latest_gateway_session_for_peer` refuses to reach behind,
        so the two agree on where one conversation stops and the next begins
        and cannot drift.

        The peer is ``(session_key, source)``, the SAME identity tuple recovery
        uses — never the key alone.  ``X-Hermes-Session-Key`` accepts any
        authenticated caller-supplied string, so an API conversation may
        legally carry the same key as a Telegram row in one database; keying
        on the string alone would let a ``/new`` on that unrelated row rotate
        this conversation's affinity identity while recovery correctly refuses
        to cross the same line.

        Returns the count, or ``None`` when this peer has never been reset.

        The value comes from ``conversation_generations``, which
        :meth:`_bump_conversation_generation` advances inside the transaction
        that writes each boundary — NOT from an aggregate over the session
        rows.  An aggregate cannot prove non-reuse: ``delete_session()``
        orphans children and deletes the row, and bulk prune selects ended
        rows, so ``COUNT``/``MAX`` over boundaries can return a pair it already
        emitted and hand a new conversation a retired affinity identity.  It is
        also wall-clock-free, so a backwards NTP correction cannot reorder it.

        Databases upgraded mid-conversation start at no generation and take
        their first one from the next boundary written; a conversation that
        reset before the upgrade shares its predecessor's scope once, which
        costs a warm prompt-cache bucket and never crosses an identity.

        These rows are never garbage-collected, by design: dropping one resets
        the peer to "no generation", so its next boundary writes ``1`` again
        and re-issues a scope a retired conversation already used — the ABA
        this counter exists to prevent.  See the schema comment in
        ``hermes_state_common.py``.
        """
        if not session_key or not source:
            return None
        with self._read_ctx() as conn:
            row = conn.execute(
                "SELECT generation FROM conversation_generations "
                "WHERE source = ? AND session_key = ?",
                (source, session_key),
            ).fetchone()
        if row is None or row["generation"] is None:
            return None
        generation = int(row["generation"])
        return generation if generation > 0 else None

    def _is_compression_child_row(self, child: Dict[str, Any]) -> bool:
        parent_id = child.get("parent_session_id")
        if not parent_id or self._is_explicit_fork_child_row(child):
            return False
        parent = self.get_session(parent_id)
        return bool(parent and parent.get("end_reason") == "compression")

    def get_compression_lineage(self, session_id: str) -> List[str]:
        """Return compression ancestors through tip in chronological order."""
        session = self.get_session(session_id)
        if not session or self._is_explicit_fork_child_row(session):
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
            with self._read_ctx() as conn:
                rows = conn.execute(
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
                if self._is_compression_child_row(candidate):
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

    def get_session_delete_targets(self, session_id: str) -> List[str]:
        """Return every session row that :meth:`delete_session` would remove.

        The requested session is first, followed by its recursively discovered
        delegate/subagent children. Branch and compression children are not
        included because deletion preserves them by orphaning their parent
        reference.
        """
        with self._read_ctx() as conn:
            exists = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)
            ).fetchone()
            if not exists:
                return []
            # Use the borrowed read connection, never self._conn: handing the
            # shared writer connection to a helper here executes on it without
            # self._lock — the same unsynchronized-read class as #99349/#90734.
            delegate_ids = _collect_delegate_child_ids(conn, [session_id])
        return [session_id, *sorted(delegate_ids)]

    def delete_session(
        self,
        session_id: str,
        sessions_dir: Optional[Path] = None,
        expected_delete_ids: Optional[List[str]] = None,
    ) -> bool:
        """Delete a session and all its messages.

        Delegate subagent children (``model_config._delegate_from``) are
        cascade-deleted with the parent so they never resurface in session
        pickers as orphaned rows. Branch / compression children are orphaned
        (``parent_session_id → NULL``) so they remain accessible independently.
        When *sessions_dir* is provided, also removes on-disk transcript
        files (``.json`` / ``.jsonl`` / ``request_dump_*``) for every deleted
        session. When *expected_delete_ids* is provided, deletion proceeds only
        if the parent plus delegate cascade still matches that exact set. This
        lets export-before-delete callers fail closed if a new delegate appears
        after they materialize their archive. The delegate tree is re-walked
        inside the write transaction on purpose (TOCTOU guard); the cost is
        accepted for correctness. Returns True if the session was found and
        deleted.
        """
        removed_delegate_ids: List[str] = []
        expected_ids = (
            set(expected_delete_ids) if expected_delete_ids is not None else None
        )

        def _do(conn):
            cursor = conn.execute(
                "SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (session_id,)
            )
            if cursor.fetchone() is None:
                return False
            if expected_ids is not None:
                actual_ids = {
                    session_id,
                    *_collect_delegate_child_ids(conn, [session_id]),
                }
                if actual_ids != expected_ids:
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
            self._delete_unreferenced_system_prompts(conn)
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
            if cursor.rowcount > 0:
                self._delete_unreferenced_system_prompts(conn)
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
            self._delete_unreferenced_system_prompts(conn)
            removed_ids.extend(existing)
            return len(existing)

        count = self._execute_write(_do)
        for sid in removed_delegate_ids:
            self._remove_session_files(sessions_dir, sid)
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count

    #: Shared selector for :meth:`count_empty_sessions` and
    #: :meth:`delete_empty_sessions` so the badge and the sweep agree.
    #:
    #: ``message_count`` tracks live (``active = 1``) rows only; rewind
    #: (:meth:`replace_messages` w/ ``archive_dropped``) and in-place
    #: compaction (:meth:`archive_and_compact`) reset it to 0 while keeping
    #: dropped turns on disk as ``active = 0`` — the only recoverable copy
    #: (#70516 / #80763 / #82756). The ``NOT EXISTS`` probe is the authority;
    #: ``message_count = 0`` stays as a cheap prefilter. Same shape as every
    #: other emptiness guard in this module. (#95868)
    _EMPTY_SESSION_WHERE = (
        "message_count = 0 "
        "AND ended_at IS NOT NULL "
        "AND archived = 0 "
        "AND NOT EXISTS ("
        "SELECT 1 FROM messages WHERE messages.session_id = sessions.id"
        ")"
    )

    def count_empty_sessions(self) -> int:
        """Return the count of empty, non-active, non-archived sessions.

        "Empty" = the session holds no message rows at all AND has ended
        (``ended_at IS NOT NULL``) AND is not archived. The ``ended_at``
        guard matches the safety contract used by :meth:`prune_sessions`:
        only ended sessions are candidates for bulk deletion, so a freshly
        spawned session whose first message hasn't landed yet — or one
        held open by the live agent — is never sniped out from under
        the runtime.

        Emptiness is decided by :data:`_EMPTY_SESSION_WHERE` — see that
        constant for why the ``NOT EXISTS`` probe is needed instead of
        trusting ``message_count`` alone.

        Backs the ``GET /api/sessions/empty/count`` endpoint that lets the
        web dashboard hide its "Delete empty" button when there's nothing
        to clean up, and pre-populate the confirm dialog with the actual
        count.
        """
        with self._read_ctx() as conn:
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM sessions WHERE {self._EMPTY_SESSION_WHERE}"
            )
            return cursor.fetchone()[0]

    def delete_empty_sessions(
        self,
        sessions_dir: Optional[Path] = None,
    ) -> int:
        """Delete every empty, ended, non-archived session.

        Mirrors :meth:`prune_sessions`' transactional shape:

        * Selects candidate IDs first (:data:`_EMPTY_SESSION_WHERE`) so we
          never touch a live session, one the user deliberately archived,
          or one whose transcript survives as soft-archived rows.
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
                f"SELECT id FROM sessions WHERE {self._EMPTY_SESSION_WHERE}"
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
                # DELETE FROM messages is paranoia — the selector's
                # ``NOT EXISTS`` probe already proved these sessions own no
                # message rows — but a row inserted between the SELECT and
                # this statement would otherwise be left dangling, so we
                # still leave a clean FK state.
                conn.execute(
                    "DELETE FROM messages WHERE session_id = ?", (sid,)
                )
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
                removed_ids.append(sid)
            self._delete_unreferenced_system_prompts(conn)
            return len(session_ids)

        count = self._execute_write(_do)
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count

    @staticmethod
    def _prune_filter_where(
        *,
        last_active_before: Optional[float] = None,
        last_active_after: Optional[float] = None,
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
        include_pinned: bool = False,
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
        if last_active_before is not None:
            clauses.append(
                """COALESCE(
                       (SELECT MAX(m.timestamp) FROM messages m
                        WHERE m.session_id = s.id),
                       s.started_at
                   ) < ?"""
            )
            params.append(last_active_before)
            # An automatic orphan sweep closes a stale open row so the user can
            # still recover it. Age those rows from the sweep, not from their old
            # activity, or the next prune pass can delete them immediately.
            clauses.append(
                "(COALESCE(s.end_reason, '') != 'startup_orphan_reap' "
                "OR s.ended_at < ?)"
            )
            params.append(last_active_before)
        if last_active_after is not None:
            clauses.append(
                """COALESCE(
                       (SELECT MAX(m.timestamp) FROM messages m
                        WHERE m.session_id = s.id),
                       s.started_at
                   ) >= ?"""
            )
            params.append(last_active_after)
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
            clauses.append("LOWER(COALESCE(s.title, '')) LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(title_like.lower())}%")
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
            clauses.append("LOWER(COALESCE(s.model, '')) LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(model_like.lower())}%")
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
            clauses.append("LOWER(COALESCE(s.git_branch, '')) LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(branch_like.lower())}%")
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
        # Pinned sessions are a durable "keep" flag (exempt from the stale
        # auto-archive sweep). Bulk prune/delete/archive must honor that too:
        # exclude pinned rows unless the caller explicitly opts in. Without
        # this, `sessions prune`/`delete`/`archive` with a filter silently
        # destroyed pinned conversations (round-3 QA SES-01, data loss).
        if not include_pinned:
            clauses.append("COALESCE(s.pinned, 0) = 0")
        return " AND ".join(clauses), params

    @staticmethod
    def _apply_prune_age_filter(
        older_than_days: Optional[float], filters: Dict[str, Any]
    ) -> None:
        """Translate the legacy age window into the shared activity filter."""
        if (
            filters.get("last_active_before") is None
            and filters.get("started_before") is None
            and older_than_days is not None
        ):
            filters["last_active_before"] = time.time() - (
                older_than_days * 86400
            )

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
        ``id, source, title, model, started_at, last_active, ended_at,
        message_count, archived``. ``older_than_days`` is an inactivity
        threshold: it uses the latest message timestamp, falling back to
        ``started_at`` for sessions without messages.
        """
        self._apply_prune_age_filter(older_than_days, filters)
        where, params = self._prune_filter_where(source=source, **filters)
        with self._read_ctx() as conn:
            cursor = conn.execute(
                f"""SELECT s.id, s.source, s.title, s.model, s.started_at,
                           COALESCE(
                               (SELECT MAX(m.timestamp) FROM messages m
                                WHERE m.session_id = s.id),
                               s.started_at
                           ) AS last_active,
                           s.ended_at, s.message_count, s.archived
                    FROM sessions s WHERE {where}
                    ORDER BY last_active ASC, s.started_at ASC""",
                params,
            )
            return [dict(row) for row in cursor.fetchall()]

    def count_prune_matches(
        self,
        older_than_days: Optional[float] = None,
        source: str = None,
        **filters,
    ) -> int:
        """Count sessions a matching prune/archive would touch.

        Same filter surface as :meth:`list_prune_candidates` (including the
        ``include_pinned`` tri-state), but returns only a count. Used by the
        CLI to report how many pinned sessions are being spared.
        """
        self._apply_prune_age_filter(older_than_days, filters)
        where, params = self._prune_filter_where(source=source, **filters)
        with self._read_ctx() as conn:
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM sessions s WHERE {where}", params
            )
            return int(cursor.fetchone()[0])

    def count_open_prune_matches(
        self,
        older_than_days: Optional[float] = None,
        source: str = None,
        **filters,
    ) -> int:
        """Count open sessions excluded from a matching bulk prune.

        This applies every normal prune filter, but inverts only the
        ``ended_at`` safety guard. It is visibility-only: callers can explain
        why an otherwise matching session was skipped without making live
        sessions eligible for destructive pruning.
        """
        self._apply_prune_age_filter(older_than_days, filters)
        where, params = self._prune_filter_where(source=source, **filters)
        ended_guard = "s.ended_at IS NOT NULL"
        if not where.startswith(ended_guard):
            raise RuntimeError("prune filter lost its ended-session safety guard")
        open_where = f"s.ended_at IS NULL{where[len(ended_guard):]}"
        with self._read_ctx() as conn:
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM sessions s WHERE {open_where}", params
            )
            return int(cursor.fetchone()[0])

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

        "Touched" is the freshest of ``last_activity_at`` and the latest
        message timestamp (else ``started_at``) — i.e. real recency, not
        creation time — so a session
        created long ago but active yesterday is spared, while an old
        abandoned one (even a still-open one) is swept. Unlike
        :meth:`archive_sessions`, this method can also archive unended
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
        with self._read_ctx() as conn:
            rows = conn.execute(
                f"""
                SELECT s.id FROM sessions s
                WHERE s.archived = 0
                  AND COALESCE(s.end_reason, '') <> 'compression'
                  {pin_clause}
                  AND {_sql_session_last_active("s")} < ?
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
        exclude_active_write_guards: bool = False,
        **filters,
    ) -> int:
        """Delete sessions matching the filters. Returns count deleted.

        By default, delete ended sessions inactive for
        ``older_than_days`` days, optionally restricted to ``source``.
        Activity is the latest message timestamp, falling back to
        ``started_at`` for sessions without messages. Additional keyword
        filters AND together — the full set is defined by
        :meth:`_prune_filter_where`:

        * ``last_active_before`` / ``last_active_after`` — epoch bounds on
          the latest message timestamp (falling back to ``started_at``).
        * ``started_before`` / ``started_after`` — epoch bounds on
          ``started_at``. An explicit ``started_before`` overrides the
          default ``older_than_days`` inactivity cutoff; pass
          ``older_than_days=None`` for no implicit upper age bound.
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

        ``exclude_active_write_guards`` is for destructive automatic
        maintenance: rows protected by a live turn lease or compression lock
        are skipped, while expired or provably dead holders are reclaimed and
        fenced in the same write transaction.
        """
        self._apply_prune_age_filter(older_than_days, filters)
        where, where_params = self._prune_filter_where(source=source, **filters)
        removed_ids: list[str] = []

        def _do(conn):
            cursor = conn.execute(
                f"SELECT s.id FROM sessions s WHERE {where}", where_params
            )
            session_ids = {row["id"] for row in cursor.fetchall()}

            if exclude_active_write_guards:
                protected = set()
                for sid in session_ids:
                    try:
                        self._check_transcript_write_guards(
                            conn,
                            sid,
                            compression_lock_holder=None,
                            turn_lease_holder=None,
                            reject_active_turn_lease=True,
                            reject_active_compression_lock=True,
                            allow_closed_compression_parent=True,
                        )
                    except (
                        SessionCompressionInProgressError,
                        SessionTurnLeaseLostError,
                    ):
                        protected.add(sid)
                session_ids.difference_update(protected)

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
            self._delete_unreferenced_system_prompts(conn)
            return len(session_ids)

        count = self._execute_write(_do)
        # Clean up on-disk files outside the DB transaction
        for sid in removed_ids:
            self._remove_session_files(sessions_dir, sid)
        return count

    def purge_stale_tool_call_markers(
        self, *, dry_run: bool = False, backup: bool = True
    ) -> Dict[str, Any]:
        """Permanently clear bare tool-call marker content (e.g. "[memory]")
        left in the ``messages`` table by sessions persisted before the
        #78148 fix in ``agent.conversation_loop``.

        ``_strip_stale_tool_call_markers`` already repairs this in memory on
        every session load (see ``_rows_to_conversation``), so running this
        is optional — but for long-lived sessions the same rows get
        re-scanned and re-repaired on every resume, which is wasted work
        and keeps the contaminated bytes sitting in the DB (and in any
        downstream cache/backup snapshot of it) indefinitely. This rewrites
        the affected rows once, in place.

        Only the ``content`` column is touched — ``role``, ``tool_calls``,
        and every other column on the row are left exactly as they are, so
        provider tool_call/tool_result pairing is unaffected.

        Unlike the in-memory repair, this UPDATE is permanent and can't be
        undone from within the DB. Since ``backup`` defaults to True, a
        timestamped full snapshot is taken via ``VACUUM INTO`` (safe against
        a live connection, unlike the raw-copy ``_backup_db_file`` used for
        malformed-schema repair) before any row is touched — mirroring
        ``repair_state_db_schema``'s backup-by-default convention for
        destructive state.db operations. No snapshot is taken when there is
        nothing to change.

        With ``dry_run=True``, reports the affected row count/ids without
        writing or backing up (read-only, no write lock taken).

        Returns ``{"dry_run": bool, "rows_affected": int, "row_ids": [...],
        "backup_path": str|None}``.
        """

        def _find_affected(conn) -> List[int]:
            cursor = conn.execute(
                "SELECT id, content FROM messages "
                "WHERE role = 'assistant' AND tool_calls IS NOT NULL AND tool_calls != ''"
            )
            affected: List[int] = []
            for row in cursor.fetchall():
                content = row["content"]
                if isinstance(content, str) and _STALE_TOOL_CALL_MARKER_RE.fullmatch(content.strip()):
                    affected.append(row["id"])
            return affected

        with self._read_ctx() as conn:
            affected_ids = _find_affected(conn)

        if dry_run:
            return {
                "dry_run": True,
                "rows_affected": len(affected_ids),
                "row_ids": affected_ids,
                "backup_path": None,
            }

        if not affected_ids:
            return {
                "dry_run": False,
                "rows_affected": 0,
                "row_ids": [],
                "backup_path": None,
            }

        backup_path: Optional[str] = None
        if backup:
            import datetime

            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = self.db_path.with_name(
                f"{self.db_path.name}.pre-clean-markers-backup-{stamp}"
            )
            with self._lock:
                self._conn.execute("VACUUM INTO ?", (str(dest),))
            backup_path = str(dest)
            logger.info("Backed up state.db to %s before clean-markers write", backup_path)

        def _do(conn):
            ids = _find_affected(conn)
            if ids:
                placeholders = ",".join("?" * len(ids))
                conn.execute(
                    f"UPDATE messages SET content = '' WHERE id IN ({placeholders})",
                    ids,
                )
            return ids

        affected_ids = self._execute_write(_do)
        if affected_ids:
            logger.info(
                "Permanently cleared %d stale tool-call marker row(s) in state.db (#78148)",
                len(affected_ids),
            )
        return {
            "dry_run": False,
            "rows_affected": len(affected_ids),
            "row_ids": affected_ids,
            "backup_path": backup_path,
        }

    # ── Meta key/value (for scheduler bookkeeping) ──

    def get_meta(self, key: str) -> Optional[str]:
        """Read a value from the state_meta key/value store."""
        # Kept on self._lock (not _read_ctx) because callers like
        # fts_rebuild_step read progress before entering a write
        # transaction, and the read-only WAL connection sees only
        # committed data — a pending write transaction's uncommitted
        # meta writes would be invisible.  This is a cheap point lookup,
        # not the convoy bottleneck the read-path split targets.
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

    def retag_kanban_worker_sessions(self, workspaces_root: str) -> int:
        """Retag legacy kanban worker rows from ``cli`` to ``kanban``.

        Workers used to spawn without ``HERMES_SESSION_SOURCE``, so their runs
        landed as untitled ``cli`` rows and the sidebar rendered one per attempt
        labeled with the worker's own prompt. New workers tag themselves; this
        reclaims the rows already on disk so they drop out of the session lists
        too. Identified by cwd under the board's workspaces root — a path only
        the dispatcher ever runs a session in.

        Gated per workspaces root (``state_meta``) so each board reclaims its
        own rows exactly once. Returns the number of rows retagged.
        """
        prefix = str(workspaces_root).rstrip("/\\")
        if not prefix:
            return 0

        gate = f"kanban_worker_source_retagged:{prefix}"
        if self.get_meta(gate) == "1":
            return 0

        def _do(conn):
            cursor = conn.execute(
                "UPDATE sessions SET source = 'kanban' "
                "WHERE source = 'cli' AND (cwd = ? OR cwd LIKE ? ESCAPE '\\')",
                (prefix, _escape_like(prefix) + "/%"),
            )
        self._halt_if_db_generation_changed()
        logger.warning(
            "state.db connection for %s was closed while a %s was still in "
            "flight — reopening (teardown/worker race, #94736)", self.db_path, context,
        )
        try:
            self._conn = self._open_writer_conn()
        except Exception as exc:
            raise sqlite3.OperationalError(
                f"state.db connection was closed while a {context} was still "
                f"in flight (a session-teardown path called close() before "
                f"this worker finished — #94736) and the automatic reopen failed: {exc}"
            ) from exc

    def _execute_write(
        self, fn: Callable[[sqlite3.Connection], T], patience_s: Optional[float] = None,
    ) -> T:
        """Run *fn(conn)* inside BEGIN IMMEDIATE with jittered lock retry; commit
        is handled here (callers must not commit). Returns *fn*'s result.
        BEGIN IMMEDIATE takes the WAL write lock up front so contention surfaces
        immediately; on locked/busy the Python lock is released, a jitter slept,
        and the WHOLE callback retried — *fn* must stay idempotent under retry."""
        if patience_s is None:
            patience_s = self._WRITE_PATIENCE_S
        deadline = time.monotonic() + patience_s
        compression_deadline: Optional[float] = None  # set on the first compression-busy collision
        # One retry for SQLITE_IOERR raised by BEGIN IMMEDIATE itself (callback not run: nothing
        # replayed). Once fn has started, an IOERR leaves settlement unknown and must propagate.
        # The callback has not run at that point, so there is no durable effect to replay and the retry is
        # exactly-once safe (#99502's contract). Once the callback starts, an IOERR leaves the write's
        # settlement unknown and must propagate — this helper owns non-idempotent transcript/counter
        # mutations, not just idempotent UPSERTs.
        ioerr_begin_retried = False
        while True:
            self._raise_if_db_corrupt()
            self._raise_if_db_replaced()
            fn_started = False
            try:
                with self._lock:
                    if self._conn is None:  # close() raced this writer
                        self._reopen_after_close_locked(context="write")
                    self._conn.execute("BEGIN IMMEDIATE")
                    try:
                        fn_started = True
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
                if self._write_count % self._FTS_MERGE_EVERY_N_WRITES == 0:
                    self._try_incremental_merge_fts()
                return result
            except SessionCompressionInProgressError:
                # Transient (see _COMPRESSION_BUSY_WAIT_S): a steer landing mid-compression must not abort.
                # A live foreign compression lock is transient: the compressor publishes in a couple of
                # seconds. Without any wait, a steer that lands mid-compression aborts the user's turn as
                # session_persistence_failed and sends the operator hunting disk space that was never the
                # problem (#75083). The budget is _COMPRESSION_BUSY_WAIT_S, not the write-lock patience: the
                # lease is a correctness boundary, so a writer still locked out after a short wait must be
                # refused rather than left to land a stale turn once a long-running or wedged compression
                # finally lets go.
                if compression_deadline is None:
                    compression_deadline = min(time.monotonic() + self._COMPRESSION_BUSY_WAIT_S, deadline)
                if self._sleep_before_write_retry(
                    compression_deadline, self._COMPRESSION_BUSY_WAIT_S
                ):
                    continue
                raise
            except sqlite3.Error as exc:
                # 'no more rows' is a transient engine error on contended WAL appends (some builds
                # raise it as InterfaceError, a sibling of DatabaseError): retry like locked/busy.
                if _is_no_more_rows(exc) and self._sleep_before_write_retry(deadline, patience_s):
                    continue
                err_msg = str(exc).lower()
                if isinstance(exc, sqlite3.OperationalError):
                    if "locked" in err_msg or "busy" in err_msg:
                        if self._sleep_before_write_retry(deadline, patience_s):
                            continue
                        # Say what actually happened, not disk/permission damage.
                        raise sqlite3.OperationalError(
                            f"database is locked (another Hermes process held the "
                            f"state.db write lock for over {patience_s:.0f}s — "
                            "likely a long maintenance operation such as VACUUM, "
                            "a large WAL checkpoint, or an older pre-update "
                            "process; the database itself is healthy)"
                        ) from exc
                    if (
                        _DISK_IO_ERROR_MARKER in err_msg and not fn_started and not ioerr_begin_retried
                        and self._sleep_before_write_retry(deadline, patience_s)
                    ):
                        # Retry on the SAME connection: close()+reopen would cancel this process's
                        # POSIX locks for every sibling (howtocorrupt §2.2).
                        ioerr_begin_retried = True
                        continue
                    raise  # non-lock error, callback already ran, or patience exhausted
                if isinstance(exc, sqlite3.DatabaseError):
                    # An out-of-band replace surfaces as this same corruption class; in-file repair
                    # on a NEW generation amplifies the damage.
                    if (
                        "not a database" in err_msg or is_malformed_db_error(exc)
                        or self._is_fts_write_corruption_error(exc)
                    ):
                        self._raise_if_db_replaced()
                    # Corrupt FTS shadow tables fail every write via the sync triggers while canonical
                    # rows are intact: detach the derived indexes atomically and retry (never rebuild here).
                    if self._enter_fts_fail_open(exc):
                        continue
                    # What survives both checks is structural damage: quarantine.
                    if self._is_structural_corruption_error(exc):
                        self._halt_db_corrupt(exc)
                raise

    def _write_sql(
        self, sql: str, params: Any = (), *, many: bool = False, patience_s: Optional[float] = None,
    ) -> None:
        """Run one INSERT/UPDATE/DELETE through ``_execute_write``."""
        def _do(conn):
            (conn.executemany if many else conn.execute)(sql, params)
        self._execute_write(_do, patience_s=patience_s)

    def _write_rowcount(self, sql: str, params: Any = (), *, patience_s: Optional[float] = None) -> int:
        """Run one UPDATE/DELETE through ``_execute_write``; return rows changed
        (``SELECT changes()`` when the driver reports None / negative)."""
        def _do(conn):
            rowcount = conn.execute(sql, params).rowcount
            if rowcount is None or rowcount < 0:
                rowcount = conn.execute("SELECT changes()").fetchone()[0]
            return rowcount
        return self._execute_write(_do, patience_s=patience_s)

    def _read_one(self, sql: str, params: Any = ()) -> Optional[sqlite3.Row]:
        """``fetchone()`` of one read-only statement via ``_read_ctx``."""
        with self._read_ctx() as conn:
            return conn.execute(sql, params).fetchone()

    def _read_all(self, sql: str, params: Any = ()) -> List[sqlite3.Row]:
        """``fetchall()`` of one read-only statement via ``_read_ctx``."""
        with self._read_ctx() as conn:
            return conn.execute(sql, params).fetchall()

    def _ensure_db_file_generation(self) -> None:
        """Mint a once-per-file generation stamp (state_meta + application_id). First opener wins (INSERT
        OR IGNORE); application_id is written only while 0 so racers converge. PASSIVE checkpoint only.

        See #45383.
        """
        if self.read_only or self._conn is None:
            return
        token = uuid.uuid4().hex
        try:
            with self._lock:
                self._conn.execute(
                    "INSERT OR IGNORE INTO state_meta (key, value) VALUES (?, ?)",
                    (_STATE_DB_GENERATION_KEY, token),
                )
                row = self._conn.execute(
                    "SELECT value FROM state_meta WHERE key = ?", (_STATE_DB_GENERATION_KEY,),
                ).fetchone()
                if row and row[0]:
                    token = str(row[0])
                pragma_row = self._conn.execute("PRAGMA application_id").fetchone()
                current = int(pragma_row[0] or 0) if pragma_row else 0
                if current == 0:
                    current = (int(token[:8], 16) & 0x7FFFFFFF) or 1
                    self._conn.execute(f"PRAGMA application_id={current}")
                self._db_file_application_id = current
                try:
                    self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                except sqlite3.Error:
                    pass
        except sqlite3.Error as exc:
            logger.debug("state.db generation stamp skipped: %s", exc)

    def _record_db_file_identity(self) -> None:
        """Snapshot inode plus the on-disk generation header when present."""
        self._db_file_identity = _stat_db_file_identity(self.db_path)
        self._db_sidecar_identity = _stat_sqlite_sidecar_identity(self.db_path)
        disk_id = _read_sqlite_application_id(self.db_path)
        if disk_id:
            self._db_file_application_id = disk_id
        elif self._conn is not None and not self._db_file_application_id:
            try:
                pragma_row = self._read_one("PRAGMA application_id")
            except sqlite3.Error:
                pragma_row = None
            if pragma_row and pragma_row[0]:
                self._db_file_application_id = int(pragma_row[0])

    def _db_file_was_replaced(self) -> bool:
        """True when the path no longer names the file this instance opened."""
        recorded = self._db_file_identity
        if recorded is not None and _stat_db_file_identity(self.db_path) != recorded:
            return True
        recorded_app = int(self._db_file_application_id or 0)
        if not recorded_app:
            return False
        # Header 0 = WAL not yet checkpointed, not a replace; a real replacement is nonzero.
        disk_app = _read_sqlite_application_id(self.db_path)
        return bool(disk_app and disk_app != recorded_app)

    def _wal_generation_was_lost(self) -> bool:
        """True when the WAL/SHM generation this handle opened is gone. Recorded
        generation: pure stat (no /proc walk on healthy writes). Empty identity
        (WAL appeared after open, or cleared by a clean close()): probe
        /proc/self/fd for deleted sidecars and adopt the current ones once clean."""
        recorded = self._db_sidecar_identity or {}
        base = os.fspath(self.db_path)
        if recorded:
            return any(
                _stat_db_file_identity(Path(base + suffix)) != ident for suffix, ident in recorded.items()
            )
        if not self._wal_active:  # no sidecar generation to lose; keep /proc off the hot path
            return False
        if sys.platform.startswith("linux"):
            watched = _watched_sqlite_sidecar_paths(self.db_path)
            try:
                for target in _proc_fd_targets(os.getpid()):
                    if " (deleted)" in target and _canonical_sqlite_path(target) in watched:
                        return True
            except OSError:
                return False
        # Probe clean (or unavailable): adopt the current sidecar generation.
        current_identity = _stat_sqlite_sidecar_identity(self.db_path)
        if current_identity:
            self._db_sidecar_identity = current_identity
        return False

    def _halt_if_db_generation_changed(self) -> None:
        """Stop writes (logging once) when the file was replaced or its WAL/SHM generation
        is gone: never run in-file repair on a new generation, never keep committing on a
        split WAL. Both flags are sticky."""
        # A reopen resolves the PATH again — if the file at that path is no longer the one this instance
        # originally opened (out-of-band restore/cp/mv), reconnecting would write into the new generation
        # through stale WAL/shm assumptions (#89332). Refuse instead.
        if self._db_replaced or self._db_file_was_replaced():
            self._db_replaced = True
            logger.error(_STATE_DB_REPLACED_MSG)
            raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)
        if self._db_wal_generation_lost or self._wal_generation_was_lost():
            self._db_wal_generation_lost = True
            logger.error(_DELETED_WAL_GENERATION_MSG)
            raise DeletedWalGenerationError(_DELETED_WAL_GENERATION_MSG)

    def _raise_if_db_replaced(self) -> None:
        """Sticky-flag fast path (no log spam on every write), then the live probe."""
        if self._db_replaced:
            raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)
        if self._db_wal_generation_lost:
            raise DeletedWalGenerationError(_DELETED_WAL_GENERATION_MSG)
        self._halt_if_db_generation_changed()

    @classmethod
    def _is_structural_corruption_error(cls, exc: BaseException) -> bool:
        """Bare SQLITE_CORRUPT/NOTADB with no FTS provenance: canonical B-tree/schema/freelist damage,
        never repairable from the live write path."""
        return (
            isinstance(exc, sqlite3.DatabaseError)
            and not isinstance(exc, StateDbCorruptError)
            and not cls._is_fts_write_corruption_error(exc)
            and classify_persistence_error(exc) == "corrupt"
        )

    def _corrupt_error(self, prefix: str = "") -> "StateDbCorruptError":
        """Build the quarantine error for this handle (message assembled once)."""
        return StateDbCorruptError(f"{prefix}{_STATE_DB_CORRUPT_MSG} (cause: {self._db_corrupt_reason})")

    def _halt_db_corrupt(self, exc: BaseException) -> None:
        """Quarantine this handle and raise; never run in-file repair here."""
        self._db_corrupt = True
        self._db_corrupt_reason = str(exc)
        self._disable_close_time_checkpoint()
        logger.error(
            "state.db %s reported structural corruption outside the FTS "
            "indexes (%s); quarantining this handle: no further writes, no "
            "automatic reopen, no explicit WAL checkpoint at close. Stop the "
            "gateway and run `hermes sessions recover --source %s --inspect-only`.", self.db_path, exc,
            self.db_path,
        )
        err = self._corrupt_error()
        for attr in ("sqlite_errorcode", "sqlite_errorname"):
            if getattr(exc, attr, None) is not None:
                setattr(err, attr, getattr(exc, attr))
        raise err from exc

    def _disable_close_time_checkpoint(self) -> None:
        """Best-effort SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE (Python 3.12+): sqlite3's
        close() otherwise runs the internal last-connection checkpoint that wrote
        the incident's pages under wrong page numbers (see StateDbCorruptError).
        <3.12 has no setconfig; the residual checkpoint only carries
        pre-quarantine committed frames, which is tolerable."""
        flag = getattr(sqlite3, "SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE", None)
        conn = self._conn
        setconfig = getattr(conn, "setconfig", None)
        if flag is None or setconfig is None:
            return
        try:
            setconfig(flag, True)
        except Exception:
            logger.debug(
                "Could not disable SQLite's close-time checkpoint on the quarantined handle for %s",
                self.db_path, exc_info=True,
            )

    def _raise_if_db_corrupt(self) -> None:
        if self._db_corrupt:
            raise self._corrupt_error()

    def _sleep_before_write_retry(self, deadline: float, patience_s: float) -> bool:
        """Sleep one jitter interval if the budget allows; True = retry, False = deadline passed. Small
        jitter for the first _WRITE_RETRY_SLOW_AFTER_S, then slow; never overshoots the deadline."""
        now = time.monotonic()
        if now >= deadline:
            return False
        slow = now - (deadline - patience_s) >= self._WRITE_RETRY_SLOW_AFTER_S
        jitter = random.uniform(*(
            (self._WRITE_RETRY_SLOW_MIN_S, self._WRITE_RETRY_SLOW_MAX_S) if slow
            else (self._WRITE_RETRY_MIN_S, self._WRITE_RETRY_MAX_S)
        ))
        time.sleep(min(jitter, max(deadline - now, 0.001)))
        return True

    def _foreign_state_db_holders(self) -> List[Tuple[int, str]]:
        """Foreign processes holding this DB or its WAL sidecars (see hermes_state_holders)."""
        return _foreign_state_db_holders(self.db_path)

    def _try_wal_checkpoint(self) -> None:
        """Best-effort PASSIVE WAL checkpoint; never raises. PASSIVE never blocks writers;
        TRUNCATE corrupted B-trees on 65K+ page databases under exclusive-lock I/O pressure.

        Previous TRUNCATE strategy caused B-tree corruption on large databases (65K+ pages) due to the
        exclusive-lock I/O pressure from checkpointing thousands of frames at once (issue #45383).
        """
        if self._db_corrupt:
            return  # quarantined: never checkpoint over a damaged image
        try:
            with self._lock:
                result = self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone()
                if result and result[1] > 0:
                    logger.debug("WAL checkpoint: %d/%d pages checkpointed", result[2], result[1])
        except Exception as exc:
            logger.warning("WAL checkpoint (PASSIVE) failed: %s", exc)

    def __enter__(self) -> "SessionDB":
        """``with SessionDB(path) as db:`` closes on exit; owners must release deterministically.

        Ownership of a SessionDB should be released explicitly. Historically an instance with a started
        token writer pinned ITSELF (bound-method writer target plus a strong ``atexit`` drain hook), so
        ``__del__`` never ran for exactly the instances that leaked descriptors (#88033). The writer now
        retires after an idle window and the atexit hook holds only a weak reference, so abandoned handles
        are eventually collectible — but "eventually, after the idle window and a GC cycle" is not a release
        policy. Call sites owning a handle are still expected to close it deterministically (see the
        ownership comments in ``run_agent.py`` and ``tui_gateway/methods_session.py``).
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False  # never suppress the caller's exception

    def close(self):
        """Drain queued token deltas, then a PASSIVE checkpoint on writable handles
        (NOT TRUNCATE: a full WAL reset races the gateway's live writer, tearing
        B-tree pages). A registry-shared instance RELEASES one refcount instead.

        Drains queued token deltas first (the background writer needs the connection). Read-only connections
        never request a checkpoint. See #45383.
        When this instance is shared (opened via ``hermes_state_registry.acquire``), ``close()`` RELEASES one
        refcount instead of tearing down the connection: the registry owns the lifecycle and only closes on
        the final release (#90837). This prevents one caller's close from tearing down the writer connection
        that other callers in the same process are still using — while still letting legacy ``close()`` call
        sites return their reference instead of leaking it.
        """
        if self._shared_registry_owned:
            from hermes_state_registry import release
            release(self)
            return
        self._stop_token_writer()
        hook, self._token_atexit_hook = self._token_atexit_hook, None
        if hook is not None:
            atexit.unregister(hook)
        # Closed flag first: an in-flight reader then closes its own connection.
        with self._read_conns_lock:
            self._read_conns_closed = True
        while self._evict_one_idle_read_conn():
            pass
        with self._lock:
            if self._conn:
                if self._db_corrupt:  # quarantined: no checkpoint over a damaged image
                    logger.warning(
                        "Skipping the close-time WAL checkpoint for %s: this "
                        "handle observed structural corruption (%s). Take a "
                        "snapshot of state.db, -wal and -shm before restarting, "
                        "then run `hermes sessions recover --source %s --inspect-only`.", self.db_path,
                        self._db_corrupt_reason, self.db_path,
                    )
                elif not self.read_only:  # PASSIVE, not TRUNCATE (see docstring)
                    try:
                        # Every cron run_agent opens+closes a transient SessionDB, so a TRUNCATE here fires
                        # a full WAL reset many times/hour, racing the gateway's long-lived writer on large
                        # WAL databases and tearing hot B-tree pages -- the #45383 corruption this class's
                        # own periodic checkpoint was already made PASSIVE to avoid. TRUNCATE belongs only
                        # on a sole-opener/quiescent connection.
                        self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    except Exception as exc:
                        logger.debug("WAL checkpoint (PASSIVE) at close failed: %s", exc)
                conn, self._conn = self._conn, None
                self._close_connection_quietly(conn)
                # A clean close lets SQLite unlink the sidecars (a legitimate end of the
                # generation, not a split): a teardown-race reopen must re-adopt.
                self._db_sidecar_identity = {}

    def __del__(self) -> None:
        """Safety net: close() if the caller forgot. Attribute access stays
        guarded: module teardown order is undefined."""
        if self.__dict__.get("_conn") is not None:
            try:
                self.close()
            except Exception:
                pass

    # ── Async token accounting (SessionUsageMixin) ──
    # queue_token_counts() is a deque append; a single-writer thread applies deltas in
    # order, coalescing consecutive deltas whose route fields are EQUAL (so the merged
    # UPDATE equals applying them sequentially). Exact readers call flush_token_counts().
    _TOKEN_DELTA_SUM_FIELDS = (
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens",
        "api_call_count",
    )
    _TOKEN_DELTA_COST_FIELDS = ("estimated_cost_usd", "actual_cost_usd")
    _TOKEN_DELTA_ROUTE_FIELDS = (
        "model", "cost_status", "cost_source", "pricing_version", "billing_provider", "billing_base_url",
        "billing_mode",
    )

    MAX_TITLE_LENGTH = 100

    # Title provenance, lowest to highest authority: auto-titling may only replace a
    # strictly lower-authority title (``derived`` -> ``llm`` once; never a user-typed name).
    TITLE_SOURCE_DERIVED, TITLE_SOURCE_LLM, TITLE_SOURCE_USER = "derived", "llm", "user"
    _TITLE_SOURCE_RANK = {TITLE_SOURCE_DERIVED: 0, TITLE_SOURCE_LLM: 1, TITLE_SOURCE_USER: 2}

    # Bot Mode's canonical chat is resolved by exact-title lookup: the title IS the identity,
    # so _set_session_title refuses renames of a hidden row holding it.
    # Bot Mode's forever-chat registry: the session titled exactly this, on a bot's profile, IS the bot's
    # canonical chat — resolved by exact-title lookup on every open (no session-id pointer exists). See
    # #92473.
    CANONICAL_BOT_CHAT_TITLE = "Bot Chat"

    # ── Message storage constants (SessionMessagesMixin) ──
    # Prefix marking JSON-encoded structured content; NUL cannot collide with text.
    _CONTENT_JSON_PREFIX = "\x00json:"
    #: Reactions live inside ``display_metadata`` so they survive row rewrites.
    REACTIONS_METADATA_KEY = "reactions"
    # Columns every conversation projection decodes; ``active`` rides along so a display read
    # can split compaction-archived rows without a second query.
    _CONVERSATION_ROW_COLUMNS = (
        "id, role, content, tool_call_id, tool_calls, tool_name, effect_disposition, "
        "finish_reason, reasoning, reasoning_content, reasoning_details, "
        "codex_reasoning_items, codex_message_items, platform_message_id, observed, "
        "_compressed_summary, timestamp, active, api_content, display_kind, display_metadata"
    )

    # ── Meta key/value (scheduler bookkeeping) ──

    def get_meta(self, key: str) -> Optional[str]:
        """Read state_meta[key] on self._lock (not _read_ctx): fts_rebuild_step reads progress before its
        write transaction and a WAL reader would not see it."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM state_meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def set_meta(self, key: str, value: str, *, cursor: Optional[sqlite3.Cursor] = None) -> None:
        """Upsert state_meta[key]; with ``cursor`` the write is inline (the caller already holds a
        transaction — nesting BEGIN IMMEDIATE would deadlock)."""
        sql = (
            "INSERT INTO state_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
        )
        if cursor is not None:
            cursor.execute(sql, (key, value))
        else:
            self._write_sql(sql, (key, value))

    def retag_kanban_worker_sessions(self, workspaces_root: str) -> int:
        """Retag legacy kanban worker rows from ``cli`` to ``kanban`` by cwd under the board's workspaces
        root; gated once per root via state_meta. Returns rows retagged."""
        prefix = str(workspaces_root).rstrip("/\\")
        if not prefix:
            return 0
        gate = f"kanban_worker_source_retagged:{prefix}"
        if self.get_meta(gate) == "1":
            return 0
        def _do(conn):
            cursor = conn.execute(
                "UPDATE sessions SET source = 'kanban' "
                "WHERE source = 'cli' AND (cwd = ? OR cwd LIKE ? ESCAPE '\\')",
                (prefix, _escape_like(prefix) + "/%"),
            )
            # rowcount BEFORE set_meta reuses this cursor for its INSERT.
            retagged = cursor.rowcount or 0
            self.set_meta(gate, "1", cursor=cursor)
            return retagged
        return self._execute_write(_do)

    def list_meta_prefix(self, prefix: str) -> List[Tuple[str, str]]:
        """``[(key, value), ...]`` for state_meta keys starting with the literal
        ``prefix`` (LIKE wildcards escaped) — e.g. ``loop:<session_id>`` rows."""
        if not prefix:
            return []
        rows = self._read_all(
            "SELECT key, value FROM state_meta WHERE key LIKE ? ESCAPE '\\'", (_escape_like(prefix) + "%",),
        )
        return [(row[0], row[1]) for row in rows]


class AsyncSessionDB:
    """Async door onto SessionDB: every call runs via asyncio.to_thread so a blocking SQLite call
    never freezes the event loop (no method returns a live cursor)."""

    def __init__(self, db: "SessionDB") -> None:
        self._db = db

    def __getattr__(self, name: str):
        attr = getattr(self._db, name)
        if not callable(attr):
            return attr
        async def _offloaded(*args, **kwargs):
            return await asyncio.to_thread(attr, *args, **kwargs)
        return _offloaded


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
from typing import Set  # noqa: F401,E402
import contextlib  # noqa: F401,E402
import errno  # noqa: F401,E402
import struct  # noqa: F401,E402
import weakref  # noqa: F401,E402

MAX_SAFE_EXPORT_MESSAGES = 20_000

MAX_SAFE_RESUME_MESSAGES = 20_000


_PLUGIN_COMPAT_LAZY = {
    'AUTO_VACUUM_MIN_FREELIST_RATIO': ('hermes_state_common', 'AUTO_VACUUM_MIN_FREELIST_RATIO'),
    'ActivityProvenance': ('agent.session_activity', 'ActivityProvenance'),
    'CompressionSessionBusyError': ('hermes_state_errors', 'CompressionSessionBusyError'),
    'CompressionSessionClosedError': ('hermes_state_errors', 'CompressionSessionClosedError'),
    'DEFERRED_INDEX_SQL': ('hermes_state_common', 'DEFERRED_INDEX_SQL'),
    'FTS_CJK_STALE_KEY': ('hermes_state_common', 'FTS_CJK_STALE_KEY'),
    'FTS_CJK_TABLE_SQL': ('hermes_state_fts', 'FTS_CJK_TABLE_SQL'),
    'FTS_CJK_TRIGGER_SQL': ('hermes_state_fts', 'FTS_CJK_TRIGGER_SQL'),
    'FTS_REBUILD_DEFERRAL_KEY': ('hermes_state_common', 'FTS_REBUILD_DEFERRAL_KEY'),
    'FTS_SQL': ('hermes_state_common', 'FTS_SQL'),
    'FTS_STALE_KEY': ('hermes_state_common', 'FTS_STALE_KEY'),
    'FTS_STORAGE_VERSION': ('hermes_state_common', 'FTS_STORAGE_VERSION'),
    'FTS_TRIGRAM_SQL': ('hermes_state_common', 'FTS_TRIGRAM_SQL'),
    'LEGACY_FTS_SQL': ('hermes_state_common', 'LEGACY_FTS_SQL'),
    'LEGACY_FTS_TRIGRAM_SQL': ('hermes_state_common', 'LEGACY_FTS_TRIGRAM_SQL'),
    'MAX_FTS5_QUERY_CHARS': ('hermes_state_common', 'MAX_FTS5_QUERY_CHARS'),
    'PERSISTENCE_ERROR_CAUSES': ('hermes_state_errors', 'PERSISTENCE_ERROR_CAUSES'),
    'SCHEMA_SQL': ('hermes_state_common', 'SCHEMA_SQL'),
    'SCHEMA_VERSION': ('hermes_state_common', 'SCHEMA_VERSION'),
    'SESSION_STATUS_COMPLETE': ('hermes_state_sessions', 'SESSION_STATUS_COMPLETE'),
    'SESSION_STATUS_EMPTY': ('hermes_state_sessions', 'SESSION_STATUS_EMPTY'),
    'SESSION_STATUS_ERROR': ('hermes_state_sessions', 'SESSION_STATUS_ERROR'),
    'SESSION_STATUS_INTERRUPTED': ('hermes_state_sessions', 'SESSION_STATUS_INTERRUPTED'),
    'SKILL_EXCERPT_JOINT': ('agent.skill_commands', 'SKILL_EXCERPT_JOINT'),
    'SKILL_SCAFFOLD_SQL_LIKE': ('agent.skill_commands', 'SKILL_SCAFFOLD_SQL_LIKE'),
    'SessionTurnLeaseLostError': ('hermes_state_errors', 'SessionTurnLeaseLostError'),
    'WalUnsupportedError': ('hermes_state_wal', 'WalUnsupportedError'),
    'apply_durability_barriers': ('hermes_state_repair', 'apply_durability_barriers'),
    'classify_session_status': ('hermes_state_sessions', 'classify_session_status'),
    'collect_state_db_stats': ('hermes_state_dbfile', 'collect_state_db_stats'),
    'count_db_holders': ('hermes_state_dbfile', 'count_db_holders'),
    'describe_skill_invocation': ('agent.skill_commands', 'describe_skill_invocation'),
    'fts5_cjk_so_path': ('hermes_state_fts', 'fts5_cjk_so_path'),
    'is_advisory_lock_contention': ('hermes_state_common', 'is_advisory_lock_contention'),
    'is_automatic_end_reason': ('hermes_state_common', 'is_automatic_end_reason'),
    'is_disk_full_error': ('hermes_state_errors', 'is_disk_full_error'),
    'is_sqlite_wal_reset_vulnerable': ('hermes_state_wal', 'is_sqlite_wal_reset_vulnerable'),
    'is_transient_sqlite_error': ('hermes_state_errors', 'is_transient_sqlite_error'),
    'iter_deleted_sqlite_sidecar_holders': ('hermes_state_dbfile', 'iter_deleted_sqlite_sidecar_holders'),
    'release_or_close': ('hermes_state_registry', 'release_or_close'),
    'report_startup_progress': ('hermes_startup_watchdog', 'report_startup_progress'),
    'resolve_journal_mode': ('hermes_state_wal', 'resolve_journal_mode'),
    'resolve_synchronous_level': ('hermes_state_wal', 'resolve_synchronous_level'),
    'sanitize_context': ('agent.memory_manager', 'sanitize_context'),
    'sqlite_source_id': ('hermes_state_wal', 'sqlite_source_id'),
    'workspace_key': ('hermes_state_sessions', 'workspace_key'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
