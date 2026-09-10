#!/usr/bin/env python3
"""SQLite state store for Hermes Agent: session metadata, message history, model
config, FTS5 search. WAL mode (concurrent readers + one writer); compression
splits sessions via parent_session_id chains; sessions are source-tagged
('cli', 'telegram', ...). Batch-runner / RL trajectories live elsewhere.
"""

import asyncio
import atexit
import contextlib
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
from dataclasses import dataclass
from pathlib import Path

from agent.message_sanitization import _sanitize_surrogates
from hermes_constants import get_hermes_home
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, TypeVar, cast

from hermes_state_common import escape_like as _escape_like, stat_db_file_identity as _stat_db_file_identity
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
    _watched_sqlite_sidecar_paths, has_invalid_sqlite_header_preopen, is_zeroed_state_db, quarantine_cross_process_lock,
    quarantine_invalid_state_db,
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


@contextlib.contextmanager
def _cross_process_repair_lock(db_path: Path):
    """Serialize state.db schema surgery across processes.

    Yields True when this process holds the repair lock for *db_path*, False
    when the bounded acquire timed out or the lock file could not be opened at
    all.  Unlike the kanban init lock — whose critical section is idempotent,
    so proceeding without the lock is merely redundant work — proceeding here
    would be exactly the unsafe interleaving we are trying to prevent, so a
    caller that gets False must NOT do surgery.

    ``flock`` is the right primitive for this: the kernel drops the lock when
    the holding process dies, so a crashed repairer cannot leave a stale lock
    that wedges every future repair (a pidfile would).  One exception exists
    (issue #100108): a forked child that inherited the lock fd keeps the
    flock alive after the acquirer dies, so the acquire path records the
    holder's pid + start time and breaks the lock when that holder is
    provably dead (see ``_acquire_db_flock``).  The acquire is still
    bounded because a *live* repairer can legitimately sit in ``VACUUM`` for
    minutes on a large DB, and an unbounded wait would hang the caller's open
    with no traceback (the failure shape of #36644).
    """
    lock_path = db_path.with_name(db_path.name + ".repair.lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
    except OSError as exc:
        # Fail closed, exactly as a timed-out acquire does.  A lock file we
        # cannot even open means the filesystem is out of space, inodes or
        # descriptors — and a sibling that opened ITS handle before the disk
        # filled is still inside writable_schema surgery or VACUUM.  Yielding
        # True here let two processes run schema surgery on the same live
        # state.db concurrently, which is itself the corruption source this
        # lock exists to remove (#100368: the disk-full trigger, then a fresh
        # corruption on every boot with other writers alive).  Callers already
        # handle False by re-probing and reporting, and on a read-only
        # directory no repair strategy could have written anyway.
        logger.warning(
            "Could not open state.db repair lock %s (%s) — skipping schema "
            "surgery rather than running it without cross-process authority.",
            lock_path, exc,
        )
        yield False
        return

    acquired = False
    try:
        if _IS_WINDOWS:
            deadline = time.monotonic() + _REPAIR_LOCK_TIMEOUT_SECONDS
            while True:
                try:
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    acquired = True
                    break
                except (BlockingIOError, OSError) as exc:
                    if not is_advisory_lock_contention(exc):
                        logger.warning(
                            "Could not acquire state.db repair lock %s (%s) — "
                            "skipping schema surgery on a non-contention error.",
                            lock_path, exc,
                        )
                        acquired = None
                        break
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(_REPAIR_LOCK_POLL_SECONDS)
        else:
            acquired, handle = _acquire_db_flock(
                str(lock_path),
                handle,
                _REPAIR_LOCK_TIMEOUT_SECONDS,
                _REPAIR_LOCK_POLL_SECONDS,
                "state.db repair lock",
            )
        if acquired is None:
            # Non-contention failure already logged with its errno.
            acquired = False
        elif not acquired:
            record = None if _IS_WINDOWS else _read_lock_holder_record(handle)
            logger.warning(
                "state.db repair lock %s held by another process for more "
                "than %.0fs — skipping schema surgery in this process to "
                "avoid racing the repairer. Recorded holder: %s.",
                lock_path, _REPAIR_LOCK_TIMEOUT_SECONDS,
                _describe_lock_holder(record),
            )
        yield acquired
    finally:
        try:
            if acquired:
                if _IS_WINDOWS:
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    _clear_lock_holder_record(handle)
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:  # pragma: no cover - best effort release
            pass
        finally:
            handle.close()


def _try_acquire_auto_maintenance_lock(db_path: Path) -> Optional[Any]:
    """Non-blocking cross-process lock for one auto-maintenance pass.

    The kernel releases this advisory lock if the holder exits, unlike a
    durable pid/meta marker. A caller that cannot acquire it must skip the
    pass: otherwise two startups can both pass the interval check and the
    second can prune a row the first has only just closed recoverably.
    """
    lock_path = db_path.with_name(db_path.name + ".auto-maintenance.lock")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
    except OSError as exc:
        logger.warning(
            "Could not open state.db auto-maintenance lock %s (%s) — skipping "
            "automatic maintenance.",
            lock_path,
            exc,
        )
        return None

    try:
        if _IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(  # type: ignore[attr-defined]
                handle.fileno(), msvcrt.LK_NBLCK, 1  # type: ignore[attr-defined]
            )
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        handle.close()
        return None
    return handle


def _release_auto_maintenance_lock(handle: Any) -> None:
    """Release a handle returned by :func:`_try_acquire_auto_maintenance_lock`."""
    try:
        if _IS_WINDOWS:
            import msvcrt

            handle.seek(0)
            msvcrt.locking(  # type: ignore[attr-defined]
                handle.fileno(), msvcrt.LK_UNLCK, 1  # type: ignore[attr-defined]
            )
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:  # pragma: no cover - best effort release
        pass
    finally:
        handle.close()


def _bump_schema_cookie(conn: sqlite3.Connection) -> None:
    """Increment the schema cookie after direct ``sqlite_master`` surgery.

    Ordinary DDL bumps this counter for free, and every other connection
    compares it before running a prepared statement — that is how they learn
    to discard a cached schema.  Editing ``sqlite_master`` under
    ``PRAGMA writable_schema=ON`` does NOT bump it, so live connections in
    other processes keep compiling statements against the schema we just
    deleted objects from — e.g. writing ``messages`` rows through triggers
    into ``messages_fts*`` shadow tables that no longer exist.  SQLite's
    writable_schema documentation calls out incrementing ``schema_version``
    as the required companion to such an edit.

    Best-effort and never raises: a failed bump leaves exactly the
    pre-existing behaviour, and the repair itself is still worth completing.
    """
    try:
        current = conn.execute("PRAGMA schema_version").fetchone()[0]
        # Wraps within the 32-bit signed range SQLite stores this in; the
        # comparison other connections make is equality, not ordering.
        conn.execute(f"PRAGMA schema_version={(int(current) + 1) & 0x7FFFFFFF}")
    except (sqlite3.DatabaseError, TypeError, IndexError) as exc:
        logger.warning("Could not bump state.db schema cookie: %s", exc)


# ── Repair-loop bounding + dead-backup hygiene (#86747) ─────────────────────
#
# ``_claim_repair_attempt`` above is an in-memory set: it bounds the loop
# only WITHIN one process. A corruption class the strategies cannot heal
# (b-tree page damage) failed repair on EVERY process start, and each pass
# took a fresh ~900MB forensic backup — 105 attempts / 89GB of identical
# dead copies in the reporting install. Two persistent bounds fix the class:
#
# * a sidecar attempt ledger (``<db>.repair-attempts.json``) that refuses
#   further surgery after ``_MAX_PERSISTENT_REPAIR_ATTEMPTS`` failures on
#   the SAME damaged file (fingerprint = size + a bounded content sample; any
#   successful repair or replacement changes it and resets the count);
# * backup dedupe + a retention cap in ``_backup_db_file`` — an identical
#   damaged file is never copied twice, and only the newest
#   ``_MAX_MALFORMED_BACKUPS`` forensic copies are kept.

_MAX_PERSISTENT_REPAIR_ATTEMPTS = 3
_MAX_MALFORMED_BACKUPS = 3

# Sidecars copied alongside a damaged DB and pruned with it. ``-journal`` is
# included because rollback-journal (DELETE) mode — Hermes's fallback on
# NFS/SMB/FUSE/ZFS and on WAL-reset-vulnerable SQLite builds — leaves a hot
# journal on disk whenever a transaction was open, and that file is what
# interprets the damaged bytes. Omitting it from the forensic copy means the
# backup cannot be rolled back to a consistent state by hand.
_DB_SIDECAR_SUFFIXES = ("-wal", "-shm", "-journal")

# Head/tail bytes sampled by ``_db_fingerprint``. Enough to change whenever
# the DB is genuinely repaired, truncated or restored (SQLite rewrites the
# header on any real recovery), while staying O(1) on a multi-GB file.
_FINGERPRINT_SAMPLE_BYTES = 65536

# Byte ranges inside SQLite's 100-byte database header that move on ordinary
# commits rather than on repair, and are therefore masked out of the content
# sample. In rollback-journal (DELETE) mode a commit writes the main file
# directly, bumping the file change counter (24-27) and version-valid-for
# (92-95); a malformed-SCHEMA DB still accepts those writes, so without the
# mask any live session write re-keys the ledger and the repair budget resets
# to 1 forever — the exact unbounded loop this ledger exists to stop. (WAL mode
# routes commits to the -wal sidecar, so the main file's header only moves on
# checkpoint; masking is harmless there and correct for both.) Everything that
# matters for repair identity — the page-1 sqlite_master b-tree — sits after
# byte 100 and stays in the sample.
_FINGERPRINT_VOLATILE_HEADER_RANGES = ((24, 28), (92, 96))


def _mask_volatile_header(head: bytes) -> bytes:
    """Zero the commit-counter fields so ordinary writes don't re-key the ledger."""
    if len(head) < 96:
        return head
    buf = bytearray(head)
    for start, end in _FINGERPRINT_VOLATILE_HEADER_RANGES:
        buf[start:end] = b"\x00" * (end - start)
    return bytes(buf)

# Free-space headroom for the pre-repair forensic backup. The backup is a
# full raw copy of the damaged DB (plus its -wal/-shm sidecars), so a repair
# loop on a large state.db is a disk amplifier: the reporting incident wrote
# ~98MB every ~10s until the volume was nearly full, which would have taken
# down every agent on the host.
#
# Proportional, not a flat floor: an absolute multi-GB reserve would refuse
# backups that fit comfortably on small container/VM volumes, and because a
# refused backup is a HARD STOP (#69603) that would silently convert "repair
# loops" into "repair never runs" for those deployments. Require the copy
# itself plus a small slice of the volume, clamped to a modest floor.
_REPAIR_BACKUP_MIN_FREE_BYTES = 256 * 1024 * 1024  # 256 MiB absolute floor
_REPAIR_BACKUP_FREE_FRACTION = 0.02  # plus 2% of the volume


def _repair_backup_headroom_bytes(total_bytes: int) -> int:
    """Free space required *beyond* the copy itself, for a volume of *total_bytes*."""
    return max(
        _REPAIR_BACKUP_MIN_FREE_BYTES,
        int(total_bytes * _REPAIR_BACKUP_FREE_FRACTION),
    )


def _repair_scratch_space_error(db_path: Path) -> Optional[str]:
    """Return an error unless snapshot, VACUUM and promotion can fit safely."""
    import shutil

    try:
        main_bytes = db_path.stat().st_size
        snapshot_bytes = main_bytes
        for suffix in _DB_SIDECAR_SUFFIXES:
            sidecar = db_path.with_name(db_path.name + suffix)
            if sidecar.exists():
                snapshot_bytes += sidecar.stat().st_size
        usage = shutil.disk_usage(db_path.parent)
        headroom = _repair_backup_headroom_bytes(usage.total)
        # Strategy 2 runs VACUUM on the staged database. SQLite documents that
        # VACUUM may need up to twice the database size in additional free
        # space while it builds the replacement and journals the overwrite.
        # Reserve that beyond the snapshot itself; after VACUUM releases its
        # temporary files, the same reserve also covers transactional
        # promotion into the live database.
        required = snapshot_bytes + (2 * snapshot_bytes) + headroom
        if usage.free >= required:
            return None
        return (
            f"only {usage.free / 1e9:.2f}GB free on {db_path.parent}; the "
            f"repair snapshot needs up to {snapshot_bytes / 1e9:.2f}GB, "
            f"VACUUM may need another {(2 * snapshot_bytes) / 1e9:.2f}GB, and "
            f"{headroom / 1e9:.2f}GB must remain as headroom. Free disk space, "
            "then retry."
        )
    except OSError as exc:
        return (
            f"could not determine free space on {db_path.parent} ({exc}); "
            "refusing the repair snapshot rather than risk filling the volume"
        )


def _repair_snapshot_timeout_seconds(source_path: Path) -> float:
    """Bound one SQLite snapshot by source size, including live sidecars.

    A WAL can contain committed canonical rows which are not yet present in
    the main database file.  Count it (and the rollback journal where
    present), both to describe the work honestly and to avoid applying the
    repair-lock timeout to an otherwise healthy large-database copy.
    """
    source_bytes = 0
    for suffix in ("", *_DB_SIDECAR_SUFFIXES):
        candidate = (
            source_path
            if not suffix
            else source_path.with_name(source_path.name + suffix)
        )
        try:
            source_bytes += candidate.stat().st_size
        except FileNotFoundError:
            continue
    return max(
        _REPAIR_LOCK_TIMEOUT_SECONDS,
        source_bytes / _REPAIR_SNAPSHOT_MIN_THROUGHPUT_BYTES_PER_SECOND,
    )


def _repair_failure_consumes_attempt(exc: BaseException) -> bool:
    """Whether a pre-strategy SQLite failure proves deterministic corruption.

    Lock contention, timeouts, disk-full, I/O and filesystem failures are
    environmental aborts: retrying later may succeed and must not exhaust the
    repair ledger. Only SQLite's corruption/image result codes prove the
    deterministic damage the bounded ledger exists to stop from retrying
    forever, even when SQLite cannot stage a snapshot far enough to run a
    named strategy.
    """
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    error_code = getattr(exc, "sqlite_errorcode", None)
    if isinstance(error_code, int):
        # Extended result codes retain the primary code in the low byte.
        primary_code = error_code & 0xFF
        return primary_code in (sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB)

    # sqlite3 versions before exception result-code attributes need a narrow,
    # conservative compatibility path. Do not turn generic DatabaseError
    # messages such as "disk is full" or "readonly" into permanent failures.
    message = str(exc).lower()
    return (
        "file is not a database" in message
        or "database disk image is malformed" in message
    )


def _repair_ledger_path(db_path: Path) -> Path:
    return db_path.with_name(db_path.name + ".repair-attempts.json")


def _db_fingerprint(db_path: Path) -> "Optional[str]":
    """Cheap identity for a damaged DB file: size + a bounded content sample.

    Deliberately EXCLUDES mtime. The original ledger keyed on
    ``size:mtime_ns`` on the assumption that "nothing can successfully write
    to a damaged file", but that does not hold for the malformed-schema
    class: the DB still opens and accepts writes (only ``sqlite_master`` is
    unreadable), so live writers, WAL checkpoints and the in-place repair
    strategies themselves all move mtime between passes. Every pass then
    looked like a NEW file — the attempt counter reset to 1 forever, never
    reaching ``_MAX_PERSISTENT_REPAIR_ATTEMPTS``, and the ``_backup_db_file``
    dedupe (which compares mtime too) never matched, so each pass wrote
    another full-size forensic copy. Observed: a repair every ~10s, a fresh
    98MB copy each time, 2.3GB in 20 minutes, disk heading to zero.

    Hashing a multi-GB corrupt file on every open is the repeated cost this
    ledger exists to avoid, so sample instead of digesting the whole file:
    size plus the head/tail slices that any real repair, truncation or
    restore necessarily changes. Stable across passes that merely touch
    mtime; still resets the attempt count after genuine recovery.

    The content read runs under ``offline_file_access`` because it takes a raw
    descriptor, and ``close()`` on ANY descriptor cancels every POSIX advisory
    lock this process holds on the file — including a peer connection's
    RESERVED lock (see ``hermes_cli.sqlite_safe_read`` rule 1). This function
    is reached from ``repair_state_db_schema``'s exhaustion probe BEFORE
    ``_backup_db_file``'s ``has_live_connection`` guard, and the repair path is
    entered by one SessionDB while the gateway holds others, so a live peer is
    the expected case rather than a theoretical one.

    Returns ``None`` when a live connection makes the read unsafe. Callers MUST
    NOT substitute a differently-shaped key (an earlier revision fell back to
    ``size:mtime_ns``): the ledger compares keys for equality, so alternating
    between a content key and an mtime key across passes never matches, the
    counter resets to 1 every time and the unbounded repair loop this ledger
    exists to stop comes straight back. ``None`` means "identity unavailable",
    and the ledger helpers below keep using the key already on record.
    """
    try:
        st = db_path.stat()
        try:
            from hermes_cli.sqlite_safe_read import (
                LiveConnectionError,
                offline_file_access,
            )
        except ImportError:
            # Scaffold/embed installs ship hermes_state without hermes_cli. No
            # tracked connections exist there, so the raw read is safe.
            @contextmanager
            def offline_file_access(_path, **_kw):
                yield

            class LiveConnectionError(Exception):
                pass

        try:
            with offline_file_access(db_path, what="fingerprint"):
                with open(db_path, "rb") as fh:
                    head = fh.read(_FINGERPRINT_SAMPLE_BYTES)
                    if st.st_size > _FINGERPRINT_SAMPLE_BYTES:
                        fh.seek(max(0, st.st_size - _FINGERPRINT_SAMPLE_BYTES))
                        tail = fh.read(_FINGERPRINT_SAMPLE_BYTES)
                    else:
                        tail = b""
        except LiveConnectionError:
            return None
        digest = hashlib.sha256(_mask_volatile_header(head) + tail).hexdigest()[:32]
        return f"{st.st_size}:{digest}"
    except OSError:
        return None


def _backup_content_identity(db_path: Path) -> "Optional[str]":
    """Recovery-image identity for forensic-backup dedupe: whole-file + sidecars.

    This is a DIFFERENT equivalence relation from :func:`_db_fingerprint`, and
    the two MUST NOT be conflated. ``_db_fingerprint`` answers "same repair
    epoch?" — it masks SQLite's commit counters and samples only the head/tail
    so an ordinary write does not mint a fresh repair budget. That is exactly
    the wrong predicate for "may I reuse an existing forensic copy?": a live
    writer can commit new transcript/session rows into an *interior* page while
    preserving file size and leaving the first/last 64 KiB untouched, so two
    materially different recovery images share one ``_db_fingerprint``. Reusing
    a backup on that basis hands the operator a snapshot that predates real
    user data (and #87409 shows a failed in-place repair can still VACUUM
    canonical tables away), so the forensic copy must claim byte identity, not
    epoch identity.

    So this digests the ENTIRE main file plus every present sidecar
    (``-wal``/``-shm``/``-journal``) — the WAL can hold committed frames not yet
    checkpointed, so it is part of the recovery image. The cost is an O(n) read;
    on a miss the caller is about to do an O(n) *write* (the full raw copy), so
    the read is the cheaper half and never the dominant cost. Runs under
    ``offline_file_access`` for the same POSIX-advisory-lock reason as
    ``_db_fingerprint``; returns ``None`` when a live connection makes the read
    unsafe (caller then declines to dedupe and takes a fresh backup — the safe
    side, never a false reuse).
    """
    try:
        from hermes_cli.sqlite_safe_read import (
            LiveConnectionError,
            offline_file_access,
        )
    except ImportError:
        @contextmanager
        def offline_file_access(_path, **_kw):
            yield

        class LiveConnectionError(Exception):
            pass

    def _hash_whole(path: Path, hasher: "Any") -> None:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(chunk)

    try:
        hasher = hashlib.sha256()
        with offline_file_access(db_path, what="backup-identity"):
            # Length-delimit every member (main file included) so the
            # concatenation is prefix-free — otherwise a main-file tail could
            # coincide with a main+sidecar split and dedupe two different
            # recovery images together.
            hasher.update(f"\0main:{db_path.stat().st_size}\0".encode())
            _hash_whole(db_path, hasher)
            for suffix in _DB_SIDECAR_SUFFIXES:
                sidecar = db_path.with_name(db_path.name + suffix)
                if sidecar.exists():
                    hasher.update(f"\0{suffix}:{sidecar.stat().st_size}\0".encode())
                    _hash_whole(sidecar, hasher)
        return hasher.hexdigest()
    except LiveConnectionError:
        return None
    except OSError:
        return None


def _read_repair_ledger(db_path: Path) -> "Dict[str, Any]":
    try:
        raw = json.loads(_repair_ledger_path(db_path).read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except (OSError, ValueError):
        pass
    return {}


def _persistent_repair_attempts_exhausted(db_path: Path) -> bool:
    """Whether *db_path* has already burned its cross-restart repair budget.

    True only when the ledger records ``_MAX_PERSISTENT_REPAIR_ATTEMPTS``
    failed attempts against the CURRENT file fingerprint. Never raises; a
    missing/corrupt ledger or unstatable DB reads as "not exhausted" (the
    in-process claim and cross-process lock still bound a single run).

    When the fingerprint is unavailable because a live connection makes the
    content read unsafe, fall back to the SIZE the ledger recorded rather than
    reading as "not exhausted". Otherwise a peer connection is enough to hide
    an exhausted budget on every pass, which is the unbounded loop again.
    """
    ledger = _read_repair_ledger(db_path)
    recorded = ledger.get("fingerprint")
    fp = _db_fingerprint(db_path)
    if fp is None:
        # Size is the one component both key shapes share and that a raw read
        # is not needed for; an unchanged size means the damaged file is very
        # likely the same one the budget was burned on.
        try:
            size_prefix = f"{db_path.stat().st_size}:"
        except OSError:
            return False
        if not isinstance(recorded, str) or not recorded.startswith(size_prefix):
            return False
    elif recorded != fp:
        return False
    return int(ledger.get("failed_attempts", 0)) >= _MAX_PERSISTENT_REPAIR_ATTEMPTS


def _persistent_repair_exhausted_error(db_path: Path) -> str:
    """The stable operator-facing diagnostic for an exhausted repair budget."""
    return (
        f"automatic repair has already failed "
        f"{_MAX_PERSISTENT_REPAIR_ATTEMPTS} times on this exact file — "
        "the corruption is beyond the schema/FTS repair strategies "
        "(likely b-tree page damage). Manual recovery required: restore "
        f"a backup, or salvage with `sqlite3 {db_path} \".recover\"`. "
        f"Delete {_repair_ledger_path(db_path).name} to force another "
        "automatic attempt."
    )


def _record_repair_outcome(
    db_path: Path, *, repaired: bool, fingerprint: "Optional[str]" = None
) -> None:
    """Update the persistent attempt ledger after a repair pass. Never raises.

    Defaults to the post-attempt fingerprint — the file state the NEXT
    attempt's exhaustion probe will observe.

    When the fingerprint is unavailable (a live connection makes the content
    read unsafe), keep the key already on record and still increment: dropping
    the pass would let a peer connection reset the budget every time, which is
    the unbounded loop this ledger exists to stop. Never write a differently
    shaped key — the probe compares for equality, so mixing key shapes across
    passes never matches.
    """
    ledger_path = _repair_ledger_path(db_path)
    try:
        if repaired:
            ledger_path.unlink(missing_ok=True)
            return
        ledger = _read_repair_ledger(db_path)
        recorded = ledger.get("fingerprint")
        fp = fingerprint if fingerprint is not None else _db_fingerprint(db_path)
        if fp is None:
            if not isinstance(recorded, str):
                # No prior key to extend and no way to mint one safely: the
                # in-process claim and cross-process lock still bound this run.
                return
            fp = recorded
        attempts = (
            int(ledger.get("failed_attempts", 0)) + 1 if recorded == fp else 1
        )
        import datetime

        ledger_path.write_text(
            json.dumps(
                {
                    "fingerprint": fp,
                    "failed_attempts": attempts,
                    "last_attempt": datetime.datetime.now().isoformat(
                        timespec="seconds"
                    ),
                }
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Could not update state.db repair ledger: %s", exc)


def _existing_malformed_backups(db_path: Path) -> "List[Path]":
    """Timestamped forensic backups of *db_path*, newest first."""
    prefix = f"{db_path.name}.malformed-backup-"
    try:
        found = [
            p
            for p in db_path.parent.iterdir()
            if p.name.startswith(prefix)
            and not p.name.endswith(_DB_SIDECAR_SUFFIXES)
        ]
    except OSError:
        return []
    return sorted(found, key=lambda p: p.name, reverse=True)


def _prune_malformed_backups(db_path: Path, keep: int = _MAX_MALFORMED_BACKUPS) -> None:
    """Delete all but the *keep* newest forensic backups (and sidecars)."""
    for stale in _existing_malformed_backups(db_path)[keep:]:
        for victim in (
            stale,
            *(stale.with_name(stale.name + suffix) for suffix in _DB_SIDECAR_SUFFIXES),
        ):
            try:
                victim.unlink(missing_ok=True)
            except OSError as exc:  # pragma: no cover - best effort
                logger.warning("Could not prune stale DB backup %s: %s", victim, exc)


def _backup_db_file(db_path: Path) -> "Tuple[Optional[Path], Optional[str]]":
    """Copy a (possibly malformed) DB file to a timestamped backup beside it.
    Raw file copy on purpose: the DB won't open cleanly, so we preserve the
    bytes exactly for forensics / manual restore. WAL, SHM and rollback-journal
    sidecars are copied too when present. Returns ``(backup_path, None)`` on success or
    ``(None, reason)`` on failure — callers on the repair path treat a
    refused backup as a HARD STOP (see #69603). Repair strategies run on a
    scratch snapshot, but the forensic bundle remains the recovery path when
    corruption defeats them.

    Refuses when a connection to this database is still live in the process:
    reading the file would ``close()`` a descriptor for it and cancel that
    connection's POSIX advisory locks (see ``hermes_cli.sqlite_safe_read``).
    The repair path can be entered by one SessionDB while the gateway holds
    others, so this is a real possibility rather than a theoretical one.
    """
    import datetime
    import shutil

    try:
        from hermes_cli.sqlite_safe_read import has_live_connection
    except ImportError:
        has_live_connection = None  # type: ignore[assignment]

    if has_live_connection is not None and has_live_connection(db_path):
        reason = (
            f"a connection to {db_path} is still open in this process; "
            "raw-copying it would cancel that connection's POSIX advisory "
            "locks. Close all SessionDB handles first."
        )
        logger.error("Refusing to raw-copy %s for backup: %s", db_path, reason)
        return None, reason

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.malformed-backup-{stamp}")
    # Same-second collision (two distinct damaged states within one second)
    # must not silently overwrite the earlier forensic copy.
    seq = 1
    while backup_path.exists():
        backup_path = db_path.with_name(
            f"{db_path.name}.malformed-backup-{stamp}_{seq}"
        )
        seq += 1
    try:
        # Sweep staging debris from an earlier interrupted pass (kill mid-copy)
        # BEFORE the dedupe below. A leftover staging file is a byte-identical
        # copy of the damaged DB, so its fingerprint MATCHES and the dedupe
        # would otherwise hand it back as a legitimate forensic backup.
        # Matches sidecar staging names (``.backup-staging-<stamp>-wal``) too.
        # The second pattern is the pre-merge ``.incomplete`` spelling, swept so
        # a host that ran that build does not keep prefix-matching debris that
        # sorts NEWEST and survives prune forever.
        for pattern in (
            f"{db_path.name}.backup-staging-*",
            f"{db_path.name}.malformed-backup-*.incomplete*",
        ):
            for old in db_path.parent.glob(pattern):
                try:
                    old.unlink(missing_ok=True)
                except OSError:  # pragma: no cover - best effort
                    pass
        # Dedupe (#86747): a repair loop used to copy the SAME damaged bytes
        # on every restart — ~900MB a pass, 89GB over 11 days in the
        # reporting install. If the newest existing backup is byte-identical to
        # the current recovery image, reuse it.
        #
        # Matching on mtime made this dedupe miss exactly when it mattered
        # most: the malformed-SCHEMA class still accepts writes, so live
        # writers and the in-place repair strategies move mtime between
        # passes and every pass wrote another full-size copy (2.3GB in 20
        # minutes).
        #
        # Use ``_backup_content_identity`` (whole file + sidecars), NOT the
        # repair-epoch ``_db_fingerprint``. They are different equivalence
        # relations: the fingerprint masks commit counters and samples only
        # head/tail so an ordinary interior-page write does not re-key the
        # repair budget — but that same write DOES change the recovery image,
        # and deduping on the fingerprint would hand back a stale backup that
        # predates the write. A forensic copy must prove byte identity, so it
        # pays the O(n) read (cheaper than the O(n) write it avoids on a hit).
        try:
            # Only hash the source when there is actually a candidate to dedupe
            # against — on the common first-corruption pass there is no prior
            # backup, and hashing the (possibly multi-GB) source then would be
            # pure waste right before the copy reads it again anyway.
            existing_backups = _existing_malformed_backups(db_path)[:1]
            if existing_backups:
                src_id = _backup_content_identity(db_path)
                for existing in existing_backups:
                    if src_id is not None and _backup_content_identity(existing) == src_id:
                        logger.info(
                            "Reusing existing forensic backup %s (identical to the "
                            "damaged DB).", existing,
                        )
                        return existing, None
        except OSError:
            pass
        # Disk guard: this is a full raw copy of a possibly multi-GB DB plus
        # its sidecars. On a host whose volume is already nearly full — which
        # a preceding repair loop may itself have caused — taking it can
        # finish off the disk and take down every process on the machine.
        # Refuse while there is still room to refuse in.
        try:
            need = db_path.stat().st_size
            for suffix in _DB_SIDECAR_SUFFIXES:
                sidecar = db_path.with_name(db_path.name + suffix)
                if sidecar.exists():
                    need += sidecar.stat().st_size
            usage = shutil.disk_usage(db_path.parent)
            headroom = _repair_backup_headroom_bytes(usage.total)
            if usage.free - need < headroom:
                reason = (
                    f"only {usage.free / 1e9:.2f}GB free on {db_path.parent}; "
                    f"copying the damaged DB needs {need / 1e9:.2f}GB and must "
                    f"leave {headroom / 1e9:.2f}GB headroom. Free disk space, "
                    f"then retry (or recover manually with `sqlite3 {db_path} "
                    '".recover"`).'
                )
                logger.error("Refusing forensic backup of %s: %s", db_path, reason)
                return None, reason
        except OSError as exc:
            # Fail CLOSED. This guard exists for the nearly-full volume, which
            # is exactly where stat()/disk_usage() is most likely to fail — and
            # proceeding would take the multi-GB copy that finishes off the
            # disk. A refused backup is a HARD STOP (#69603), so repair simply
            # does not run until a human frees space, which is the safe side.
            reason = (
                f"could not determine free space on {db_path.parent} ({exc}); "
                "refusing the forensic copy rather than risk filling the "
                f"volume. Free disk space, then retry (or recover manually "
                f'with `sqlite3 {db_path} ".recover"`).'
            )
            logger.error("Refusing forensic backup of %s: %s", db_path, reason)
            return None, reason
        # Copy to a staging name OUTSIDE the ``.malformed-backup-`` prefix, then
        # rename into place only once every copy has succeeded. The prefix
        # matters: ``_existing_malformed_backups`` matches on
        # ``startswith(f"{db}.malformed-backup-")`` and excludes only ``-wal``/
        # ``-shm`` suffixes, so a staging name derived from the backup name (e.g.
        # ``…malformed-backup-<stamp>.incomplete``) still counts as a backup —
        # it sorts NEWEST (``.incomplete`` > the bare stamp), so prune's
        # keep-3-newest slice retained partials and deleted intact copies, and
        # the dedupe could hand a partial back as the official ``backup_path``,
        # passing the #69603 hard-stop gate with no real forensic copy on disk.
        staging = db_path.with_name(f"{db_path.name}.backup-staging-{stamp}")
        # (staging_src, final_dst) pairs. ORDER MATTERS for publication: the
        # main-DB backup name is the bundle's commit marker —
        # ``_existing_malformed_backups`` matches ``{db}.malformed-backup-*``
        # and excludes only the ``-wal``/``-shm``/``-journal`` suffixes, so the
        # main file appearing is what makes the bundle "count". Sidecars are
        # therefore staged/published FIRST and the main DB LAST, so a failure
        # partway through never leaves a countable main backup standing over a
        # missing sidecar (an incomplete recovery image that would pass the
        # #69603 hard stop and dedupe as legitimate on the next pass).
        staged_sidecars: "List[Tuple[Path, Path, Path]]" = []
        for suffix in _DB_SIDECAR_SUFFIXES:
            sidecar = db_path.with_name(db_path.name + suffix)
            if sidecar.exists():
                side_staging = staging.with_name(staging.name + suffix)
                side_dst = backup_path.with_name(backup_path.name + suffix)
                staged_sidecars.append((sidecar, side_staging, side_dst))
        main_pair = (staging, backup_path)
        published: "List[Path]" = []
        all_staging_srcs = [staging] + [s for _src, s, _d in staged_sidecars]
        try:
            shutil.copy2(db_path, staging)
            for sidecar, side_staging, _side_dst in staged_sidecars:
                shutil.copy2(sidecar, side_staging)
            # Publish sidecars first, main DB LAST (the commit marker), so a
            # mid-publish failure never leaves a countable-but-incomplete bundle.
            publish_order = [
                (s, d) for _src, s, d in staged_sidecars
            ] + [main_pair]
            for src, dst in publish_order:
                os.replace(src, dst)
                published.append(dst)
        except Exception:
            # Roll back BOTH unpublished staging files AND anything already
            # promoted — the old code unlinked only staging srcs, so a failure
            # after the main os.replace left the official backup_path on disk.
            for src in all_staging_srcs:
                try:
                    src.unlink(missing_ok=True)
                except OSError:
                    pass
            for dst in published:
                try:
                    dst.unlink(missing_ok=True)
                except OSError:
                    pass
            try:
                staging.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        # Retention cap (#86747): keep only the newest few forensic copies.
        _prune_malformed_backups(db_path)
        return backup_path, None
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Could not back up malformed DB %s: %s", db_path, exc)
        return None, f"backup copy failed: {exc}"


def preflight_db_writability(
    db_path: Path,
    *,
    db_label: str = "state.db",
) -> None:
    """Refuse-or-repair read-only DB files BEFORE the first connection opens.

    Port of Kilo-Org/kilocode#12508's startup preflight. A stray read-only
    ``state.db`` / ``-wal`` / ``-shm`` (sudo run, restored backup, copied
    dotfiles) previously surfaced as an opaque
    ``sqlite3.OperationalError: attempt to write a readonly database`` raised
    from deep inside ``_init_schema`` — naming no file and no fix — and the
    obvious wrong "fix" (deleting the ``-wal``) silently loses committed
    transactions. This preflight:

    - **Repairs** permissions with ``chmod u+rw`` when the file lives inside
      the Hermes home tree (``get_hermes_home()``) — the safe repair scope:
      Hermes owns those files, and the OS makes ``chmod`` fail on files the
      user doesn't own, which bounds the repair exactly.
    - **Fails fast with an actionable error** naming the exact file and the
      exact ``chmod`` command for anything else (root-owned files, read-only
      mounts, custom paths outside the home tree).
    - Never deletes or truncates a WAL sidecar — once writable, the normal
      open path checkpoints its committed frames into the DB as intended.

    ``:memory:`` and ``file:`` URI paths are skipped (no plain on-disk files
    to check). Shared by :class:`SessionDB` and ``hermes_cli.kanban_db``.
    """
    raw = str(db_path)
    if raw == ":memory:" or raw.startswith("file:"):
        return

    try:
        home: Optional[Path] = Path(get_hermes_home()).resolve()
    except Exception:  # pragma: no cover - defensive
        home = None

    def _in_repair_scope(p: Path) -> bool:
        if home is None:
            return False
        try:
            return p.resolve().is_relative_to(home)
        except (OSError, ValueError):
            return False

    def _ensure_writable(p: Path, *, is_dir: bool = False) -> None:
        import stat as _stat

        if os.access(p, os.R_OK | os.W_OK):
            return
        if _in_repair_scope(p):
            try:
                add = _stat.S_IRUSR | _stat.S_IWUSR | (_stat.S_IXUSR if is_dir else 0)
                os.chmod(p, p.stat().st_mode | add)
            except OSError:
                pass
            if os.access(p, os.R_OK | os.W_OK):
                logger.info(
                    "%s preflight: repaired read-only %s (chmod u+rw%s)",
                    db_label,
                    p,
                    "x" if is_dir else "",
                )
                return
        kind = "directory" if is_dir else "file"
        wal_note = (
            " Do NOT delete the -wal file — it contains committed data that "
            "will be merged into the database once it is writable."
            if p.name.endswith("-wal")
            else ""
        )
        raise sqlite3.OperationalError(
            f"{db_label} is not writable: {kind} {p} is read-only for this "
            f"user. Hermes needs read-write access to open the database. "
            f"Fix with: chmod u+rw{'x' if is_dir else ''} '{p}'"
            f" (files owned by another user may need sudo/chown).{wal_note}"
        )

    parent = db_path.parent
    if parent.is_dir():
        # SQLite needs a writable directory in every journal mode (WAL and
        # SHM sidecars in WAL mode; the rollback journal in DELETE mode).
        _ensure_writable(parent, is_dir=True)

    for suffix in ("", "-wal", "-shm"):
        p = db_path.with_name(db_path.name + suffix) if suffix else db_path
        if p.is_file():
            _ensure_writable(p)


def _connect_repair_durable(
    db_path: Path, *, timeout: float = 5.0
) -> sqlite3.Connection:
    """``sqlite3.connect`` for the repair/probe paths, with macOS write barriers.

    These paths open ``state.db`` directly rather than through ``SessionDB``
    (which routes via :func:`apply_wal_with_fallback`), so they inherited
    SQLite's ``synchronous=NORMAL`` default and no ``checkpoint_fullfsync``.
    On Darwin that is exactly the combination :func:`_enforce_macos_synchronous_full`
    exists to prevent: ``fsync()`` there guarantees neither data-on-platter nor
    write ordering, so a rewrite interrupted by process or OS termination can
    leave half-written b-tree pages behind.

    That matters more here than anywhere else in the module, because what runs
    through these connections is ``REINDEX``, ``VACUUM`` and ``writable_schema``
    surgery — the operations that rewrite nearly every page of the file.  The
    2026-08-19 recurrence tore ``messages`` (root page 5) and
    ``idx_messages_session``, reporting the unmistakable signature: repeated
    "2nd reference to page", a rowid out of order, and long runs of leaked
    "never used" pages.

    Autocommit (``isolation_level=None``) is preserved: callers run DDL and
    ``VACUUM``, which are illegal inside an implicit transaction.

    Applying the barriers is best-effort *by necessity*: SQLite loads the
    schema before it runs any statement, so on a malformed schema even
    ``PRAGMA synchronous=FULL`` raises ``DatabaseError`` ("malformed database
    schema (messages_fts) - table messages_fts already exists").  A malformed
    database is precisely this helper's input, so raising there would leave
    repair unable to open the file it exists to fix.  Strategies that go on to
    rewrite the whole file call :func:`_reapply_durability_barriers` once the
    schema parses again, which is the point at which the pragmas can stick.
    """
    conn = sqlite3.connect(str(db_path), timeout=timeout, isolation_level=None)
    _reapply_durability_barriers(conn)
    return conn


def _reapply_durability_barriers(conn: sqlite3.Connection) -> bool:
    """Best-effort (re)application of the macOS write barriers.  Never raises.

    Returns True when the pragmas were accepted.  Callers about to rewrite the
    file wholesale (``VACUUM``, ``REINDEX``) should call this after the schema
    becomes parseable, because a connection opened against a malformed schema
    could not take them at open time.
    """
    try:
        _apply_macos_checkpoint_barrier(conn)
        _enforce_macos_synchronous_full(conn)
        return True
    except sqlite3.DatabaseError:
        # Schema still unparseable — the pragmas cannot be set yet.
        return False
    except Exception:
        return False


def apply_durability_barriers(conn: sqlite3.Connection) -> bool:
    """Apply state-store durability barriers without changing journal mode.

    This is the public entry point for secondary users of ``state.db`` that
    must inherit its owner's journal mode while retaining per-connection
    durability settings. Also applies the configured ``database.synchronous``
    level (a per-connection pragma that would otherwise only ride on the
    journal-mode setup path guest connections must not run).
    """
    ok = _reapply_durability_barriers(conn)
    try:
        # Local import avoids a circular import with hermes_cli.config.
        from hermes_cli.config import cfg_get, load_config_readonly

        cfg = load_config_readonly()
        raw_synchronous = cfg_get(cfg, "database", "synchronous", default=None)
        if raw_synchronous is not None:
            _apply_synchronous_pragma(
                conn, raw_synchronous, db_label="state.db (guest)"
            )
    except Exception:
        pass
    return ok


@contextmanager
def _exclusive_repair_db_guard(db_path: Path):
    """Yield one live connection that excludes writers for repair surgery.

    ``locking_mode=EXCLUSIVE`` retains SQLite's file-level exclusion after the
    short ``BEGIN EXCLUSIVE`` transaction is rolled back.  That rollback is
    essential: ``Connection.backup`` may use the guarded connection as a
    *source* while it is transaction-free, and it must be transaction-free
    when it is later the promotion *destination*.  The connection itself
    remains open for the entire snapshot -> strategies -> promotion window,
    so another writer cannot commit a change that promotion could overwrite.

    Existing WAL readers make exclusive acquisition fail rather than being
    disturbed.  In DELETE mode an existing reader similarly prevents
    ``BEGIN EXCLUSIVE``; a future reader/writer waits behind the guard.  A
    repair therefore fails closed whenever this process cannot own that whole
    window.
    """
    guard: Optional[sqlite3.Connection] = None
    try:
        # The cross-process repair lock already serializes repairers.  Do not
        # wait behind an ordinary application connection: a partial repair is
        # less safe than an explicit "stop the gateway and retry" result.
        guard = _connect_repair_durable(db_path, timeout=0.0)
        guard.execute("PRAGMA locking_mode=EXCLUSIVE")
        guard.execute("BEGIN EXCLUSIVE")
        guard.execute("ROLLBACK")
    except (sqlite3.Error, OSError) as exc:
        if guard is not None:
            try:
                guard.execute("PRAGMA locking_mode=NORMAL")
            except Exception:
                pass
            guard.close()
        yield None, exc
        return

    try:
        yield guard, None
    finally:
        try:
            # Let SQLite release the exclusive locks before close; this also
            # avoids a connection-close checkpoint being mistaken for a
            # repair write in callers that immediately reopen state.db.
            guard.execute("PRAGMA locking_mode=NORMAL")
        except Exception:
            pass
        guard.close()


def _copy_database_snapshot(
    source_path: Path,
    destination_path: Path,
    *,
    source_connection: Optional[sqlite3.Connection] = None,
    destination_connection: Optional[sqlite3.Connection] = None,
) -> None:
    """Copy one complete SQLite snapshot without replacing either file inode.

    SQLite's online backup API incorporates committed WAL frames into the
    source snapshot and writes the destination inside one transaction. This
    avoids both the main-file-only staging gap and replacing ``state.db`` from
    under handles that already refer to it. If backup is interrupted, SQLite
    rolls the destination transaction back.
    """
    # Work out the deadline before opening an owned source connection.  A
    # sidecar disappearing while we stat it is an ordinary staging failure,
    # but it must not leak a just-opened SQLite descriptor.
    deadline_seconds = _repair_snapshot_timeout_seconds(source_path)
    deadline = time.monotonic() + deadline_seconds
    source = source_connection or _connect_repair_durable(source_path)
    destination = destination_connection
    own_source = source_connection is None
    own_destination = destination_connection is None

    def _check_deadline(_status: int, _remaining: int, _total: int) -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out copying SQLite repair snapshot after "
                f"{deadline_seconds:.0f}s"
            )

    try:
        if destination is None:
            destination = _connect_repair_durable(destination_path)
        elif destination.in_transaction:
            # sqlite3_backup requires a transaction-free destination.  The
            # exclusive repair guard deliberately retains file exclusion via
            # locking_mode, not an active transaction, so it satisfies this.
            raise sqlite3.ProgrammingError(
                "SQLite repair backup destination has an active transaction"
            )
        source.backup(
            destination,
            pages=256,
            progress=_check_deadline,
            sleep=_REPAIR_LOCK_POLL_SECONDS,
        )
    finally:
        if own_destination and destination is not None:
            destination.close()
        if own_source:
            source.close()


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
    conn = _connect_repair_durable(db_path)
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


def _foreign_state_db_holders(db_path: Path) -> List[Tuple[int, str]]:
    """Compatibility delegate to the state-holder authority."""
    return _state_holders.foreign_state_db_holders(db_path)


def _live_writer_holds_db(db_path: Path) -> bool:
    """Compatibility delegate to the repair-admission authority."""
    return _state_holders.live_writer_holds_db(
        db_path,
        connect_repair_durable=_connect_repair_durable,
    )


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

    Canonical ``sessions`` / ``messages`` rows are never modified by a failed
    attempt. Mutating strategies run against a complete SQLite snapshot and a
    successful result is copied back transactionally. A timestamped raw backup
    is taken first unless ``backup=False``.

    The surgery below is serialised across processes (see
    :func:`_cross_process_repair_lock`): the gateway service, the Desktop
    app's backend and interactive CLI sessions all open the same file, and
    two of them running ``writable_schema`` surgery concurrently is itself a
    corruption source.

    Returns a report dict: ``{repaired: bool, strategy: str|None,
    backup_path: str|None, error: str|None}``.
    """
    report: Dict[str, Any] = {
        "repaired": False,
        "strategy": None,
        "backup_path": None,
        "error": None,
    }

    # Startup-watchdog progress lease: repair (raw backup copy + surgery +
    # VACUUM) is I/O-bound — near-zero CPU on a multi-GB file — which the
    # watchdog's CPU fallback would misread as a parked deadlock (OOF-298).
    # Single lease is deliberate (clamped to _MAX_LEASE_S=900): honest worst
    # case is up to the lease duration of zombie time on a wedged repair,
    # accepted over per-chunk renewal complexity in the repair loop.
    report_startup_progress(900.0, phase="state_db_repair")

    db_path = Path(db_path)
    if not db_path.exists():
        report["error"] = f"{db_path} does not exist"
        return report

    # Cross-restart attempt cap (#86747): the in-memory claim bounds one
    # process, but a corruption class the strategies below cannot heal
    # (b-tree page damage) previously re-ran the whole surgery — and took a
    # fresh multi-hundred-MB forensic backup — on EVERY restart, forever.
    # After _MAX_PERSISTENT_REPAIR_ATTEMPTS failures against the same
    # damaged file, stop retrying and surface a terminal, actionable error.
    if _persistent_repair_attempts_exhausted(db_path):
        report["error"] = _persistent_repair_exhausted_error(db_path)
        logger.error("state.db repair skipped: %s", report["error"])
        return report

    result = report
    with _cross_process_repair_lock(db_path) as holding_lock:
        if not holding_lock:
            # Another process is still inside its critical section, or the
            # lock file itself could not be opened (full disk / no fds). It
            # may nonetheless have healed the file already (long VACUUM after
            # a successful strategy), so re-probe before reporting failure.
            if _db_opens_cleanly(db_path) is None:
                report["repaired"] = True
                report["strategy"] = "repaired_by_other_process"
            else:
                report["error"] = (
                    "could not obtain the state.db repair lock (held by "
                    "another process, or the lock file was unopenable); "
                    "skipped schema surgery to avoid racing a concurrent "
                    "repairer"
                )
        else:
            # The fast check above avoids taking the lock for a known-exhausted
            # image. Recheck after acquisition: a queued repairer can have
            # recorded the final failure while this process waited, and this
            # process must not start a fourth attempt.
            if _persistent_repair_attempts_exhausted(db_path):
                report["error"] = _persistent_repair_exhausted_error(db_path)
                logger.error("state.db repair skipped: %s", report["error"])
            # Keep the existing WAL-holder preflight: it preserves the
            # established fail-closed behaviour for active readers before we
            # create a forensic backup. It is not the race defence; the
            # retained exclusive guard inside the locked routine is what
            # excludes writers continuously through promotion. DELETE-mode
            # readers which this probe cannot see are still rejected by the
            # later BEGIN EXCLUSIVE acquisition.
            elif _live_writer_holds_db(db_path):
                report["error"] = (
                    "a live writer still holds state.db; skipped schema surgery "
                    "to avoid tearing b-tree pages under a concurrent writer. "
                    "Stop the gateway (hermes gateway stop) and retry."
                )
                logger.error("state.db repair skipped: %s", report["error"])
            else:
                # Probe the mode BEFORE surgery (#89674): every repair
                # strategy rewrites the file, and a rebuilt SQLite file comes
                # back in the default journal mode (delete) — silently moving
                # a WAL store out of WAL with nothing in the logs recording
                # the flip. The open-time WAL-reset gate never sees this flip
                # because it happens inside the repair path (distinct from
                # the open-time flip #89393 warns about). A probe of the
                # damaged file may fail, in which case the canonical
                # database.journal_mode setting is the restore target.
                before_mode = _probe_journal_mode_for_repair(db_path)
                result = _repair_state_db_schema_locked(
                    db_path, backup=backup, report=report
                )
                if result.get("repaired"):
                    result["journal_mode_before"] = before_mode
                    _restore_journal_mode_after_repair(db_path, before_mode)
            # Environmental aborts happen before a strategy gets to mutate the
            # isolated snapshot. They are retriable operating conditions, not
            # proof that the damaged database exhausted a repair strategy.
            # Keep that private signal out of the public report while
            # successful health checks still clear a stale persistent failure
            # record. This ledger update stays under the same cross-process
            # lock as surgery, so two repairers cannot lose each other's
            # attempt updates. A queued loser must not record at all: its
            # owner is responsible for its outcome.
            attempted = bool(result.pop("_repair_attempted", False))
            if attempted or result.get("repaired"):
                _record_repair_outcome(
                    db_path, repaired=bool(result.get("repaired"))
                )
    return result


def _probe_journal_mode_for_repair(db_path: Path) -> Optional[str]:
    """Best-effort journal-mode probe for a (possibly malformed) DB file.

    Returns the on-disk mode (``wal``/``delete``), or ``None`` when the file
    cannot be opened or probed — a malformed header or a concurrent opener's
    locks are both expected on the repair path. Callers fall back to the
    configured ``database.journal_mode`` for ``None``.
    """
    try:
        conn = _connect_repair_durable(db_path)
        try:
            return _on_disk_journal_mode(conn)
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return None


def _restore_journal_mode_after_repair(db_path: Path, before_mode: Optional[str]) -> None:
    """Re-apply the journal mode after schema surgery (#89674).

    A repaired/rebuilt SQLite file comes back in the default journal mode
    (delete). Without this restore, a corruption event deterministically
    moves a WAL store out of WAL and nothing records the change — the
    WAL-reset gate at open time never sees the flip because it happened
    inside the repair path, not at open (the open-time flip #89393 warns
    about is a different door).

    The restore runs through :func:`apply_wal_with_fallback` — the canonical
    journal-mode path — rather than issuing a switch pragma directly, so it
    inherits the vulnerable-SQLite WAL-reset gate (a rebuilt file IS a new
    database: on a vulnerable runtime the gate deliberately keeps it in
    DELETE, and "restore could not reach WAL" there is the expected outcome,
    not a failure), the macOS-NFS silent-refusal handling, and the WAL
    companions (size limit, checkpoint barrier, synchronous=FULL) that the
    front door applies. ``before_mode`` is the pre-surgery probe (None when
    the damaged file could not be probed) and is only used for the log
    comparison — the restore target itself is whatever the canonical path
    resolves from ``database.journal_mode``.

    Best-effort by design: the repair itself already succeeded, so failures
    to re-apply are logged at WARNING, never raised.
    """
    try:
        conn = _connect_repair_durable(db_path)
        try:
            after = apply_wal_with_fallback(conn, db_label=db_path.name)
        finally:
            conn.close()
        if before_mode and after != before_mode:
            logger.warning(
                "state.db repair changed journal_mode %r -> %r "
                "(pre-surgery probe %r; restore resolved through "
                "apply_wal_with_fallback per database.journal_mode and the "
                "WAL-reset gate)",
                before_mode, after, before_mode,
            )
    except (sqlite3.Error, OSError) as exc:
        logger.warning(
            "state.db repair at %s: post-surgery journal-mode restore "
            "failed (%s); verify with PRAGMA journal_mode on the next open",
            db_path, exc,
        )


def _repair_state_db_schema_locked(
    db_path: Path, *, backup: bool, report: Dict[str, Any]
) -> Dict[str, Any]:
    """Repair strategies for :func:`repair_state_db_schema`.

    Caller must hold the cross-process repair lock for *db_path*.

    The strategies run on a SCRATCH COPY and the result is copied back through
    SQLite's transactional backup API only once it is proven to open cleanly.
    A repair that does not succeed therefore cannot modify or lose committed
    canonical data. In WAL mode SQLite may checkpoint already-committed WAL
    frames into the main file while the exclusive guard is released; that is
    not a repair mutation and does not change the committed database image.

    They used to run in place, and Strategy 2 ends in ``VACUUM``. VACUUM does
    not preserve what it cannot parse: it rebuilds the file from the schema
    SQLite can still read, so when the damage IS in the schema b-tree — page
    1's child pointers resolving to data pages, which is exactly the
    ``malformed database schema ()`` class this function exists to handle —
    every table hanging off the unreadable part is silently dropped. The probe
    afterwards then correctly reports the file is STILL malformed, so the
    function returns ``repaired=False`` and advises a manual restore, having
    already destroyed the thing it was asked to save. Destroying the data and
    reporting the repair failed are not mutually exclusive outcomes, and
    nothing here treated them as a contradiction.

    The pre-repair backup (#69603) does not close this: it is a forensic
    artefact that nothing reads back, so recovery still depends on a human
    noticing a ``.malformed-backup-*`` file and knowing what to do with it.
    Not mutating the original in the first place is the property that holds
    without a human in the loop.
    """
    scratch = db_path.with_name(f"{db_path.name}.repair-scratch")
    cleanup_error = _unlink_db_triple(scratch)
    if cleanup_error is not None:
        report["error"] = (
            "could not remove a stale repair snapshot before probing state.db: "
            f"{cleanup_error}"
        )
        logger.error("state.db repair aborted: %s", report["error"])
        return report

    # Re-probe under the lock: a process we queued behind may have just
    # repaired the file, in which case redoing the surgery would undo its
    # work on a now-healthy DB (the repair/re-corrupt cascade this lock
    # exists to break).
    if _db_opens_cleanly(db_path) is None:
        report["repaired"] = True
        report["strategy"] = "already_healthy"
        return report

    if backup:
        bpath, backup_error = _backup_db_file(db_path)
        report["backup_path"] = str(bpath) if bpath else None
        if bpath is None:
            # HARD STOP (#69603). The forensic recovery image remains required
            # when corruption defeats every strategy, even though the
            # strategies themselves now run against an isolated snapshot.
            report["error"] = (
                "pre-repair backup refused; aborting schema repair to avoid "
                f"mutating the only copy of the damaged DB: {backup_error}"
            )
            logger.error("state.db repair aborted: %s", report["error"])
            return report

    # The forensic copy intentionally happens before this guard: its raw-copy
    # safety checks inspect real live holders and would be poisoned by our
    # exclusive connection.  Everything that can affect the repair image or
    # live promotion happens only after writer exclusion is held.
    with _exclusive_repair_db_guard(db_path) as (live_guard, guard_error):
        if live_guard is None:
            report["error"] = (
                "could not acquire exclusive state.db repair ownership; "
                "skipped schema surgery to avoid overwriting a concurrent "
                f"writer. Stop the gateway and retry: {guard_error}"
            )
            if guard_error is not None and _repair_failure_consumes_attempt(
                guard_error
            ):
                report["_repair_attempted"] = True
            logger.error("state.db repair skipped: %s", report["error"])
            return report

        space_error = _repair_scratch_space_error(db_path)
        if space_error is not None:
            report["error"] = space_error
            logger.error("state.db repair aborted: %s", report["error"])
            return report

        try:
            # Reuse live_guard rather than opening a second source connection:
            # the guard owns the exclusion, so a second connection could be
            # blocked by our own EXCLUSIVE lock on some SQLite builds.
            _copy_database_snapshot(
                db_path, scratch, source_connection=live_guard
            )
        except (OSError, sqlite3.Error, TimeoutError) as exc:
            report["error"] = (
                f"could not stage a complete SQLite repair snapshot of {db_path}: {exc}"
            )
            if _repair_failure_consumes_attempt(exc):
                report["_repair_attempted"] = True
            logger.error("state.db repair aborted: %s", report["error"])
            _unlink_db_triple(scratch)
            return report

        try:
            # This private marker is consumed by the outer wrapper. A strategy
            # failure is a genuine repair outcome and consumes the persistent
            # budget. A later promotion failure is classified separately:
            # full disks, I/O, permission and lock failures are environmental
            # aborts, not evidence that the strategy cannot repair this image.
            report["_repair_attempted"] = True
            _run_repair_strategies(scratch, report)
            if report.get("repaired"):
                try:
                    # Do not os.replace the live DB: Windows rejects
                    # replacement under open handles, while POSIX would leave
                    # those handles on the old inode.  The same transaction-
                    # free guard that staged the live image receives the
                    # promotion, retaining writer exclusion throughout.
                    _copy_database_snapshot(
                        scratch,
                        db_path,
                        destination_connection=live_guard,
                    )
                except (OSError, sqlite3.Error, TimeoutError) as exc:
                    report["repaired"] = False
                    report["strategy"] = None
                    report["_repair_attempted"] = _repair_failure_consumes_attempt(
                        exc
                    )
                    report["error"] = (
                        "repaired snapshot could not be promoted transactionally: "
                        f"{exc}"
                    )
                    logger.error("state.db repair promotion failed: %s", exc)
                else:
                    logger.warning(
                        "state.db repaired via '%s' and promoted transactionally: %s",
                        report.get("strategy"),
                        db_path,
                    )
            if not report.get("repaired"):
                # Logged HERE, not inside the strategies: they run against the
                # scratch copy, and naming that throwaway path in the one
                # message a human is meant to act on would send them to a file
                # that no longer exists by the time they read it.
                logger.error(
                    "state.db schema repair could not recover %s automatically "
                    "(no committed canonical data was modified or lost; backup: %s); "
                    "manual restore from backup may be required.",
                    db_path,
                    report["backup_path"],
                )
            return report
        finally:
            # Never leave a half-repaired file beside the DB for a later probe
            # — or a later human — to mistake for the real thing.
            cleanup_error = _unlink_db_triple(scratch)
            if cleanup_error is not None:
                logger.warning(
                    "Could not remove state.db repair snapshot after repair: %s",
                    cleanup_error,
                )


def _unlink_db_triple(path: Path) -> Optional[str]:
    """Remove *path* and every SQLite sidecar; return any cleanup failure."""
    failures: List[str] = []
    for suffix in ("", *_DB_SIDECAR_SUFFIXES):
        victim = path if not suffix else path.with_name(path.name + suffix)
        for attempt in range(10):
            try:
                victim.unlink()
                break
            except FileNotFoundError:
                break
            except PermissionError as exc:
                # Windows may retain a just-closed SQLite handle for a few
                # scheduler ticks. Bound the retry; a later backup open still
                # fails safely if the handle truly remains live.
                if _IS_WINDOWS and attempt < 9:
                    time.sleep(0.05)
                    continue
                failures.append(f"{victim}: {exc}")
                break
            except OSError as exc:
                failures.append(f"{victim}: {exc}")
                break
    return "; ".join(failures) or None


def _run_repair_strategies(
    db_path: Path, report: Dict[str, Any]
) -> Dict[str, Any]:
    """Escalating repair attempts, applied to *db_path* IN PLACE.

    Every strategy here mutates its argument — FTS rebuild, REINDEX,
    ``writable_schema`` surgery, ``VACUUM``. It is therefore only ever called
    by :func:`_repair_state_db_schema_locked` on a scratch copy that nothing
    else holds open, never on the user's database.
    """
    # ── Strategy 0: rebuild FTS indexes in place (FTS write-corruption) ──
    # The FTS5 'rebuild' command rewrites the internal index from the canonical
    # content table. This is the recommended, least-destructive recovery for a
    # corrupt FTS index that rejects message writes while reads still succeed.
    try:
        conn = _connect_repair_durable(db_path)
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
        conn = _connect_repair_durable(db_path)
        try:
            # REINDEX rewrites every index b-tree; take the barriers now that
            # the schema parses, in case the open-time attempt was refused.
            _reapply_durability_barriers(conn)
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
        conn = _connect_repair_durable(db_path)
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
            if dupes:
                _bump_schema_cookie(conn)
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
    #
    # The destructive one, and the reason this whole path now runs on a
    # scratch copy. VACUUM rebuilds the file from the schema SQLite can still
    # parse, so on a damaged schema b-tree it silently drops every table
    # hanging off the unreadable part — and the probe below then correctly
    # reports the result is still malformed. On a scratch copy that is merely
    # a discarded attempt; on the live file it was data loss.
    try:
        conn = _connect_repair_durable(db_path)
        try:
            conn.execute("PRAGMA writable_schema=ON")
            conn.execute("DELETE FROM sqlite_master WHERE name LIKE 'messages_fts%'")
            _bump_schema_cookie(conn)
            conn.execute("PRAGMA writable_schema=OFF")
            conn.commit()
            # The schema is repaired and parseable now, so the barriers can
            # finally stick — and VACUUM, which rewrites the entire file, is
            # the single most damaging operation to lose halfway.
            _reapply_durability_barriers(conn)
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

    # The "could not recover" log lives in the caller: it must name the user's
    # database, not the scratch copy these strategies were handed.
    return report


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

CREATE TRIGGER IF NOT EXISTS messages_fts_cjk_update
AFTER UPDATE OF content, tool_name, tool_calls, role ON messages
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


class CompressionSessionClosedError(RuntimeError):
    """A durable write targeted a parent already closed by compression."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        super().__init__(
            f"Session {session_id!r} is closed by compression; "
            "adopt its live continuation before appending messages"
        )


class CompressionSessionBusyError(RuntimeError):
    """A non-owner tried to write while compression owns the session."""


class SessionCompressionInProgressError(CompressionSessionBusyError):
    """A concurrent writer collided with a *live* compression lock.

    Split out from :class:`CompressionSessionBusyError` because the two
    conditions that class covers need opposite handling. This one is
    transient: a healthy compressor holds the session for a few seconds and
    the lock row carries its own ``expires_at``, so the write can simply wait
    (see ``_execute_write``'s patience loop). The other case, a compressor
    discovering its own lease is gone, is permanent and must fail fast rather
    than spin out the whole patience budget.

    Subclassing keeps every existing ``except CompressionSessionBusyError``
    handler working unchanged.
    """


class SessionTurnLeaseLostError(RuntimeError):
    """A transcript write presented a turn-lease holder that no longer owns it.

    Fail-fast fencing: do not retry inside ``_execute_write``. The caller
    either still thinks it owns the conversation after expiry/reclaim, or
    the lease row is gone. A later writer may already be persisting a
    newer turn; landing this write would interleave a stale reply.
    """


class ConversationWorktreeConflict(RuntimeError):
    """A root session attempted to change its claimed Git identity."""


@dataclass(frozen=True)
class ConversationWorktreeRecord:
    """Durable immutable Git identity plus the lifecycle state of one root."""

    root_session_id: str
    worktree_path: str
    branch: str
    base_commit: str
    repo_common_dir: str
    state: str
    failure_phase: Optional[str]
    failure_message: Optional[str]
    created_at: float
    updated_at: float


class StateDbReplacedError(RuntimeError):
    """The state.db path no longer names the file this SessionDB opened.

    Raised when an out-of-band ``cp``/``mv``/restore replaces the database
    under a live gateway. In-place FTS repair and fail-open trigger
    dropping cannot fix a generation mismatch; they amplify it.
    """


class DeletedWalGenerationError(StateDbReplacedError):
    """A live process holds a deleted state.db-wal / -shm generation.

    Opening or writing through this handle would mint a second WAL inode
    (or keep committing on the orphan) — the split-brain that produces
    intermittent SQLITE_CORRUPT / SQLITE_IOERR. Stop the writers; do not
    unlink the WAL yourself. ``database.journal_mode: delete`` is operator
    containment, not a default change.

    Subclasses :class:`StateDbReplacedError` so every downstream consumer
    that already stops SQLite writes and diverts pending transcripts on a
    replaced store (gateway retry queue, run_agent flush) handles the split
    WAL generation identically — the correct response is the same: stop
    writing, preserve the transcript tail on disk.
    """


# SQLite header: 4-byte big-endian application_id at offset 68. Distinct from
# inode: ``cp`` onto the same path keeps st_ino and truncates+rewrites.
_STATE_DB_APPLICATION_ID_OFFSET = 68
_STATE_DB_GENERATION_KEY = "db_file_generation"
_STATE_DB_REPLACED_MSG = (
    "FATAL: state.db was replaced underneath the gateway; refusing further "
    "writes to this file. Divert transcripts to sessions/<id>.jsonl (and the "
    "gateway pending_messages spool) and restore or reopen after operator "
    "intervention."
)
_DELETED_WAL_GENERATION_MSG = (
    "FATAL: a live process holds a deleted state.db-wal or state.db-shm "
    "inode while the path names a different (or missing) generation. "
    "Refusing to open or write so a second WAL cannot be minted. "
    "Stop the gateway, dashboard, and cron writers that hold the deleted "
    "sidecar, then reopen. Do not delete the WAL yourself. "
    "database.journal_mode: delete is operator containment, not a new default."
)


class StateDbCorruptError(sqlite3.DatabaseError):
    """A live SessionDB observed structural (non-FTS) corruption and is quarantined.

    Raised once a write on this handle reports bare ``SQLITE_CORRUPT`` /
    ``SQLITE_NOTADB`` that is neither FTS-scoped (``_is_fts_write_corruption_error``)
    nor a replaced-file case (``StateDbReplacedError``). Subclasses
    ``sqlite3.DatabaseError`` so every existing ``except sqlite3.Error``
    degrade path keeps working; ``sqlite_errorcode``/``sqlite_errorname``
    are copied from the originating error.

    The quarantine is sticky for the life of the handle: later writes fail
    fast, the handle never reopens after ``close()``, and ``close()`` skips
    its own WAL checkpoint. Field evidence (the #90837 lost/reordered-page
    signature, the #90950 page-1 clobber): a handle that kept writing for ~50
    minutes after the first structural error checkpointed 15 pages under the
    wrong page numbers on shutdown, turning a still-readable file into
    ``file is not a database``. Stopping the writes is what prevents that;
    skipping the explicit checkpoint is the second line of defence. SQLite
    still runs its own last-connection checkpoint inside ``close()`` (and
    deletes the ``-wal`` sidecar) unless ``SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE``
    is set — Python exposes it via ``Connection.setconfig()`` on 3.12+, so
    quarantine disables the close-time checkpoint there and the WAL survives
    on disk for forensics; on 3.11 the internal checkpoint is unavoidable
    (post-quarantine it can only carry pre-corruption committed frames, since
    no further writes are accepted). The
    recovery boundary is a process restart on a repaired or restored file.
    """


_STATE_DB_CORRUPT_MSG = (
    "FATAL: state.db reported structural corruption (database disk image is "
    "malformed outside the FTS shadow tables) on a live handle; refusing further "
    "writes, automatic reopen, and the close-time WAL checkpoint on this file. "
    "Stop the gateway, then run `hermes sessions recover --source <state.db> "
    "--inspect-only` or restore a snapshot. Unwritten transcripts are diverted to "
    "sessions/<id>.jsonl (and the gateway pending_messages spool)."
)


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
            if not self.db_path.exists() or has_invalid_sqlite_header_preopen(self.db_path):
                with quarantine_cross_process_lock(self.db_path) as lock_acquired:
                    if not lock_acquired:
                        logger.warning(
                            "startup quarantine lock for %s not acquired within 5s; proceeding",
                            self.db_path,
                        )
                    self._handle_quarantine_if_invalid(already_locked=lock_acquired)
                    self._connect_and_init_with_lock_patience()
            else:
                self._handle_quarantine_if_invalid(already_locked=False)
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

    def _handle_quarantine_if_invalid(self, already_locked: bool = False) -> None:
        """Quarantine a zero-byte/headerless state.db so a fresh one can open; if quarantine failed,
        raise the clear message instead of opening the zeroed file."""
        if not (self.db_path.exists() and has_invalid_sqlite_header_preopen(self.db_path)):
            return
        try:
            zsize = self.db_path.stat().st_size
        except OSError:
            zsize = -1
        qpath = quarantine_invalid_state_db(self.db_path, already_locked=already_locked)
        msg = (
            f"state.db has no SQLite header ({zsize} bytes). "
            f"Preserved at {qpath or '(quarantine failed — file left in place)'}. "
            f"Restore from {self.db_path.parent / 'state-snapshots'} via `hermes snapshot list` / "
            f"`hermes snapshot restore <id>` if available, or salvage the preserved bytes with "
            f"`hermes sessions recover --source {qpath or self.db_path}`. "
            "Opening a fresh empty database so the agent can start."
        )
        logger.error(msg)
        _set_last_init_error(msg)
        if qpath is None and self.db_path.exists() and has_invalid_sqlite_header_preopen(self.db_path):
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
            self._disable_close_time_checkpoint()
            logger.error(_STATE_DB_REPLACED_MSG)
            raise StateDbReplacedError(_STATE_DB_REPLACED_MSG)
        if self._db_wal_generation_lost or self._wal_generation_was_lost():
            self._db_wal_generation_lost = True
            self._disable_close_time_checkpoint()
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
        the incident's pages under wrong page numbers (see StateDbCorruptError and
        the generation-loss halts).
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

    def _quarantine_reason(self) -> Optional[str]:
        """Why this handle must not checkpoint or run in-file repair, or None. A corrupted image has
        torn B-trees; a replaced file or a deleted/replaced WAL generation would checkpoint under
        wrong page numbers into the main DB -- the shutdown-time cause of #105670. Same precedence
        as the halt path (replaced is checked before generation loss)."""
        if self._db_corrupt:
            return f"structural corruption ({self._db_corrupt_reason})"
        if self._db_replaced:
            return "a replaced state.db file"
        if self._db_wal_generation_lost:
            return "a deleted WAL generation (split-brain)"
        return None

    def _try_wal_checkpoint(self) -> None:
        """Best-effort PASSIVE WAL checkpoint; never raises. PASSIVE never blocks writers;
        TRUNCATE corrupted B-trees on 65K+ page databases under exclusive-lock I/O pressure.

        Previous TRUNCATE strategy caused B-tree corruption on large databases (65K+ pages) due to the
        exclusive-lock I/O pressure from checkpointing thousands of frames at once (issue #45383).
        """
        if self._quarantine_reason() is not None:
            return
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
                quarantine_reason = self._quarantine_reason()
                if quarantine_reason is not None:
                    logger.warning(
                        "Skipping the close-time WAL checkpoint for %s: this "
                        "handle observed %s. Take a snapshot of state.db, -wal and -shm "
                        "before restarting, then run `hermes sessions recover --source %s --inspect-only`.",
                        self.db_path, quarantine_reason, self.db_path,
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

    @staticmethod
    def _conversation_worktree_record(row: sqlite3.Row) -> ConversationWorktreeRecord:
        return ConversationWorktreeRecord(
            root_session_id=row["root_session_id"],
            worktree_path=row["worktree_path"],
            branch=row["branch"],
            base_commit=row["base_commit"],
            repo_common_dir=row["repo_common_dir"],
            state=row["state"],
            failure_phase=row["failure_phase"],
            failure_message=row["failure_message"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @classmethod
    def _conversation_worktree_record_on_conn(
        cls, conn: sqlite3.Connection, root_session_id: str
    ) -> Optional[ConversationWorktreeRecord]:
        row = conn.execute(
            "SELECT root_session_id, worktree_path, branch, base_commit, "
            "repo_common_dir, state, failure_phase, failure_message, "
            "created_at, updated_at "
            "FROM conversation_worktree_bindings WHERE root_session_id = ?",
            (root_session_id,),
        ).fetchone()
        return cls._conversation_worktree_record(row) if row is not None else None

    def get_conversation_worktree(
        self, root_session_id: str
    ) -> Optional[ConversationWorktreeRecord]:
        """Return the durable binding for a root session, if it was claimed."""
        with self._read_ctx() as conn:
            return self._conversation_worktree_record_on_conn(conn, root_session_id)

    def claim_conversation_worktree(
        self,
        *,
        root_session_id: str,
        worktree_path: str,
        branch: str,
        base_commit: str,
        repo_common_dir: str,
    ) -> ConversationWorktreeRecord:
        """Claim an immutable Git identity, or return the identical claim.

        Claims are made before session persistence, so the binding table has no
        foreign key to ``sessions``. A retry may observe the same identity,
        but it may never replace any identity field of an existing root.
        """
        identity = (worktree_path, branch, base_commit, repo_common_dir)

        def _do(conn: sqlite3.Connection) -> ConversationWorktreeRecord:
            existing = self._conversation_worktree_record_on_conn(
                conn, root_session_id
            )
            if existing is not None:
                existing_identity = (
                    existing.worktree_path,
                    existing.branch,
                    existing.base_commit,
                    existing.repo_common_dir,
                )
                if existing_identity != identity:
                    raise ConversationWorktreeConflict(
                        "conversation worktree identity already claimed for "
                        f"root session {root_session_id!r}"
                    )
                return existing

            now = time.time()
            conn.execute(
                "INSERT INTO conversation_worktree_bindings ("
                "root_session_id, worktree_path, branch, base_commit, "
                "repo_common_dir, state, created_at, updated_at"
                ") VALUES (?, ?, ?, ?, ?, 'creating', ?, ?)",
                (root_session_id, *identity, now, now),
            )
            record = self._conversation_worktree_record_on_conn(conn, root_session_id)
            if record is None:  # pragma: no cover - INSERT is in this transaction.
                raise RuntimeError("conversation worktree claim was not persisted")
            return record

        return self._execute_write(_do)

    def _set_conversation_worktree_state(
        self,
        root_session_id: str,
        *,
        state: str,
        allowed_current_states: Tuple[str, ...],
        failure_phase: Optional[str] = None,
        failure_message: Optional[str] = None,
    ) -> ConversationWorktreeRecord:
        def _do(conn: sqlite3.Connection) -> ConversationWorktreeRecord:
            existing = self._conversation_worktree_record_on_conn(
                conn, root_session_id
            )
            if existing is None:
                raise RuntimeError(
                    "cannot transition an unclaimed conversation worktree "
                    f"for root session {root_session_id!r}"
                )
            if existing.state == state:
                return existing
            if existing.state not in allowed_current_states:
                raise ConversationWorktreeConflict(
                    "cannot transition conversation worktree for root session "
                    f"{root_session_id!r} from {existing.state!r} to {state!r}"
                )

            conn.execute(
                "UPDATE conversation_worktree_bindings "
                "SET state = ?, failure_phase = ?, failure_message = ?, updated_at = ? "
                "WHERE root_session_id = ?",
                (state, failure_phase, failure_message, time.time(), root_session_id),
            )
            record = self._conversation_worktree_record_on_conn(conn, root_session_id)
            if record is None:  # pragma: no cover - guarded by prior SELECT.
                raise RuntimeError("conversation worktree transition was not persisted")
            return record

        return self._execute_write(_do)

    def mark_conversation_worktree_ready(
        self, root_session_id: str
    ) -> ConversationWorktreeRecord:
        """Record that a claimed worktree was verified and bootstrapped."""
        return self._set_conversation_worktree_state(
            root_session_id,
            state="ready",
            allowed_current_states=("creating", "creation_failed"),
        )

    def mark_conversation_worktree_failed(
        self,
        root_session_id: str,
        *,
        failure_phase: str,
        failure_message: str,
    ) -> ConversationWorktreeRecord:
        """Retain a failed creation claim with its safe diagnostic summary."""
        return self._set_conversation_worktree_state(
            root_session_id,
            state="creation_failed",
            allowed_current_states=("creating", "creation_failed"),
            failure_phase=failure_phase,
            failure_message=failure_message,
        )

    def mark_conversation_worktree_removed(
        self, root_session_id: str
    ) -> ConversationWorktreeRecord:
        """Record explicit cleanup without discarding the immutable identity."""
        return self._set_conversation_worktree_state(
            root_session_id,
            state="removed",
            allowed_current_states=("creating", "ready", "creation_failed", "retained"),
        )

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
