"""
Cron job storage and management.

Jobs are stored in ~/.hermes/cron/jobs.json
Output is saved to ~/.hermes/cron/output/{job_id}/{timestamp}.md
"""

import contextlib
import copy
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import json
import logging
import shutil
import tempfile
import threading
import time
import os
import re
import uuid

# Cross-process advisory file locking for jobs.json critical sections.
# fcntl is Unix-only; on Windows fall back to msvcrt. Either may be absent,
# in which case _jobs_lock() degrades to in-process locking only (the old
# behaviour) rather than failing.
try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix
    fcntl = None
try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None
from datetime import datetime, timedelta
from pathlib import Path
from hermes_constants import get_hermes_home
from typing import Optional, Dict, List, Any, Set, Tuple, Union, Collection

logger = logging.getLogger(__name__)

from hermes_time import now as _hermes_now
from utils import atomic_replace, atomic_write_text

# ``croniter`` compiles ~15 ms of regexes at import and only matters for
# 5-field cron expressions. Resolve lazily; ``HAS_CRONITER`` stays a module
# attribute (tests monkeypatch it, and a monkeypatched value wins because
# ``_ensure_croniter`` only probes while it's still None).
croniter = None
HAS_CRONITER: Optional[bool] = None


def _ensure_croniter() -> bool:
    """Import croniter on first use; honor a pre-set HAS_CRONITER override."""
    global croniter, HAS_CRONITER
    if HAS_CRONITER is None:
        try:
            from croniter import croniter as _croniter
            croniter = _croniter
            HAS_CRONITER = True
        except ImportError:
            HAS_CRONITER = False
    return bool(HAS_CRONITER)

# =============================================================================
# Configuration
# =============================================================================

# Cron is per-profile by design (issue #4707). Each profile owns its own cron
# store under its own HERMES_HOME, and a profile-scoped gateway runs that
# profile's jobs under that same HERMES_HOME — so a job authored in profile
# `coder` lives in `~/.hermes/profiles/coder/cron/jobs.json` and executes with
# `coder`'s `.env`, `config.yaml`, and skills. We deliberately anchor on
# `get_hermes_home()` (the active profile home), NOT `get_default_hermes_root()`
# (the shared root). Anchoring at the root would funnel every profile's jobs
# into one shared `jobs.json` and run them under whatever HERMES_HOME the
# ticker process happens to have — leaking config/credentials/skills across
# profiles (the security boundary #4707 was filed for). Do NOT change this to
# the default root: that re-breaks per-profile isolation. See also the dynamic
# `_get_hermes_home()` / `_get_lock_paths()` resolution in cron/scheduler.py.
HERMES_DIR = get_hermes_home().resolve()
# These constants remain the default-profile fallback and a compatibility
# surface for existing callers/tests. Cross-profile callers must scope paths
# with use_cron_store() instead of mutating them process-wide.
CRON_DIR = HERMES_DIR / "cron"
JOBS_FILE = CRON_DIR / "jobs.json"
# Heartbeat file the in-process ticker touches on every loop iteration. The
# gateway process and the (separate) ``hermes cron status`` process share it
# so status can tell whether the ticker THREAD is alive, not just whether the
# gateway PROCESS exists — a ticker that dies silently inside a live gateway
# would otherwise report healthy (#32612, #32895).
TICKER_HEARTBEAT_FILE = CRON_DIR / "ticker_heartbeat"
# Last tick that completed WITHOUT raising. Distinguishing this from the plain
# heartbeat lets status detect a ticker that is alive but failing every tick.
TICKER_SUCCESS_FILE = CRON_DIR / "ticker_last_success"
# Default ticker loop interval (seconds). The single source of truth shared by
# the in-process ticker (cron/scheduler_provider.py) and the staleness
# threshold in `hermes cron status` (hermes_cli/cron.py), so the two never
# drift apart.
TICKER_INTERVAL_SECONDS = 60

# In-process lock protecting load_jobs→modify→save_jobs cycles.
# Required when tick() runs jobs in parallel threads — without this,
# concurrent mark_job_run / advance_next_run calls can clobber each other.
_jobs_file_lock = threading.RLock()
_jobs_lock_state = threading.local()
_contextual_accounting_thread_lock = threading.RLock()
_contextual_authority_thread_lock = threading.RLock()

# Upper bound on waiting for the cross-process .jobs.lock flock (#60703).
# Every cron function in the process funnels through _jobs_lock(), and the
# flock is taken while holding the process-wide RLock — so an unbounded wait
# on a lock held by a wedged sibling process silently freezes the ticker
# heartbeat and every job forever.  30s is orders of magnitude above any
# legitimate critical section (field updates only) while keeping the ticker's
# worst-case stall well under one status-alarm threshold.
_JOBS_LOCK_TIMEOUT_SECONDS = 30.0
_CONTEXTUAL_ACCOUNTING_LOCK_TIMEOUT_SECONDS = 30.0
_CONTEXTUAL_AUTHORITY_LOCK_TIMEOUT_SECONDS = 30.0
OUTPUT_DIR = CRON_DIR / "output"
ONESHOT_GRACE_SECONDS = 120


@dataclass(frozen=True)
class _CronStorePaths:
    cron_dir: Path
    jobs_file: Path
    output_dir: Path


_cron_store_override: ContextVar[Optional[_CronStorePaths]] = ContextVar(
    "cron_store_override",
    default=None,
)


# Import-time snapshot of the compatibility constants, so deliberate
# re-pointing of the module surface (monkeypatched CRON_DIR/JOBS_FILE/
# OUTPUT_DIR — the documented escape hatch existing tests/embedders use)
# is distinguishable from the constants merely being stale.
_IMPORT_STORE = _CronStorePaths(CRON_DIR, JOBS_FILE, OUTPUT_DIR)


def _current_cron_store() -> _CronStorePaths:
    """Return paths pinned to this execution context's profile.

    Precedence, most explicit first:

    1. an active use_cron_store() override (ContextVar);
    2. deliberately re-pointed module constants — if CRON_DIR/JOBS_FILE/
       OUTPUT_DIR no longer match their import-time values, someone chose
       the documented process-wide compatibility surface; honor it;
    3. the ACTIVE profile home, resolved fresh via get_hermes_home()
       (context-local override, then the HERMES_HOME env var) — so a test
       or embedder that re-points HERMES_HOME after this module was
       imported reads/writes ITS OWN store, not whatever jobs.json the
       import happened to freeze (the filed incident: fixtures that patched
       the env too late silently rewrote the user's real jobs file);
    4. the import-time constants (home unchanged since import — the common
       path, returned unchanged).
    """
    override = _cron_store_override.get()
    if override is not None:
        return override
    live_constants = _CronStorePaths(CRON_DIR, JOBS_FILE, OUTPUT_DIR)
    if live_constants != _IMPORT_STORE:
        return live_constants
    home = get_hermes_home().resolve()
    if home == HERMES_DIR:
        return live_constants
    cron_dir = home / "cron"
    return _CronStorePaths(cron_dir, cron_dir / "jobs.json", cron_dir / "output")


@contextlib.contextmanager
def use_cron_store(home: Union[str, Path]):
    """Route cron storage to ``home`` without mutating process globals."""
    cron_dir = Path(home).expanduser().resolve() / "cron"
    token = _cron_store_override.set(
        _CronStorePaths(
            cron_dir=cron_dir,
            jobs_file=cron_dir / "jobs.json",
            output_dir=cron_dir / "output",
        )
    )
    try:
        yield
    finally:
        _cron_store_override.reset(token)


def get_cron_output_dir() -> Path:
    """Return the output directory for the active cron store context."""
    return _current_cron_store().output_dir


# Fallback stale-recovery window for a one-shot's running-claim (#59229) when
# the cron inactivity timeout is disabled (HERMES_CRON_TIMEOUT=0 → unlimited),
# in which case no finite run bound exists to derive from. Also acts as the
# floor for the derived value so a very short configured timeout can't make the
# claim expire mid-run.
ONESHOT_RUN_CLAIM_TTL_SECONDS = 1800

# The derived TTL is the cron inactivity timeout times this headroom multiplier.
# A healthy run clears its claim via mark_job_run() long before the TTL; the
# TTL only recovers a claim left by a tick that DIED mid-run. HERMES_CRON_TIMEOUT
# is an *inactivity* limit, not a wall-clock cap — a job that keeps producing
# output legitimately runs past it — so the multiplier gives comfortable
# headroom over any healthy run before we treat a claim as stale.
_ONESHOT_RUN_CLAIM_TTL_HEADROOM = 3

_DEFAULT_CRON_INACTIVITY_TIMEOUT = 600.0


def _oneshot_run_claim_ttl_seconds() -> float:
    """Resolve the one-shot running-claim stale-recovery TTL.

    Derived from ``HERMES_CRON_TIMEOUT`` (the cron inactivity timeout the
    scheduler enforces on each run) so the safety valve tracks how long a run
    is actually allowed to go quiet, instead of a magic constant:

    - unset / invalid → default 600s inactivity limit → TTL = 1800s
    - ``0`` (unlimited runs) → no finite bound to derive from → fall back to
      ``ONESHOT_RUN_CLAIM_TTL_SECONDS``
    - positive N → ``max(N * headroom, ONESHOT_RUN_CLAIM_TTL_SECONDS)`` so a
      tiny configured timeout can never expire a claim mid-run.
    """
    raw = os.getenv("HERMES_CRON_TIMEOUT", "").strip()
    timeout = _DEFAULT_CRON_INACTIVITY_TIMEOUT
    if raw:
        try:
            timeout = float(raw)
        except (ValueError, TypeError):
            timeout = _DEFAULT_CRON_INACTIVITY_TIMEOUT
    if timeout <= 0:
        # Unlimited runs — cannot bound; use the fixed fallback floor.
        return float(ONESHOT_RUN_CLAIM_TTL_SECONDS)
    return max(
        timeout * _ONESHOT_RUN_CLAIM_TTL_HEADROOM,
        float(ONESHOT_RUN_CLAIM_TTL_SECONDS),
    )


def _job_running_in_this_process(job_id: str) -> bool:
    """Return True when the scheduler in THIS process is still running ``job_id``.

    Direct liveness signal for stale-entry recovery (#62002): the run_claim
    TTL alone cannot distinguish "the claiming tick died" from "the run is
    alive but slow" — a run stalled on network I/O (or a laptop that slept
    mid-run) legitimately outlives the TTL. The in-process ticker and the run
    share this process, so the scheduler's running set settles the common
    single-gateway case without any claim-age guesswork.

    Imported lazily: the scheduler imports this module at load, so a
    module-level import here would be circular.
    """
    try:
        from cron.scheduler import get_running_job_ids
        return job_id in get_running_job_ids()
    except Exception:
        logger.warning(
            "Cron running-set liveness check failed for job %r; keeping the "
            "entry to avoid deleting a possibly live one-shot run",
            job_id,
            exc_info=True,
        )
        return True


def _jobs_lock_file() -> Path:
    """Return the advisory lock path for the current cron directory."""
    return _current_cron_store().cron_dir / ".jobs.lock"


@contextlib.contextmanager
def _jobs_lock():
    """Serialize a load_jobs→modify→save_jobs critical section.

    Combines the in-process threading lock (cheap mutual exclusion between
    the gateway's parallel tick threads) with a cross-process advisory file
    lock on ``<cron dir>/.jobs.lock`` (mutual exclusion between the gateway process
    and standalone ``hermes`` CLI invocations, which previously shared no lock
    at all — a `cron pause` could be silently clobbered by a concurrent
    gateway write, leaving a "paused" job still firing).

    The flock is blocking, but every critical section that uses it is short
    (field updates only — no agent execution), so contention resolves in
    milliseconds. If neither fcntl nor msvcrt is available the manager still
    provides in-process locking, matching the historical behaviour.

    Nested calls in the same thread reuse the held lock so legacy callers that
    invoke save_jobs() inside a broader mutation section don't deadlock or try
    to reacquire the advisory file lock.
    """
    depth = getattr(_jobs_lock_state, "depth", 0)
    if depth:
        _jobs_lock_state.depth = depth + 1
        try:
            yield bool(getattr(_jobs_lock_state, "cross_process_acquired", False))
        finally:
            _jobs_lock_state.depth -= 1
        return

    with _jobs_file_lock:
        _jobs_lock_state.depth = 1
        # Stamp of jobs.json as of this section's load_jobs() (#80703's
        # fast-path, credit @JoaoMarcos44): lets _save_jobs_unlocked skip the
        # shrink-merge parse when the file provably hasn't changed since this
        # section read it. Reset on entry/exit so stale stamps from unlocked
        # loads or prior sections can never suppress a needed merge.
        _jobs_lock_state.load_stamp = None
        _jobs_lock_state.cross_process_acquired = False
        lock_fd = None
        try:
            try:
                ensure_dirs()
                lock_fd = open(_jobs_lock_file(), "a+", encoding="utf-8")
                lock_fd.seek(0)
                if fcntl is not None:
                    # Bounded acquisition (#60703): a plain blocking
                    # fcntl.flock(LOCK_EX) here has NO timeout, and it is
                    # taken while holding the process-wide _jobs_file_lock
                    # RLock above.  If another process wedges while holding
                    # .jobs.lock (e.g. an old gateway draining through a
                    # restart), a single blocked acquirer freezes EVERY cron
                    # function in this process — including the ticker's
                    # get_due_jobs() — silently and forever: the heartbeat
                    # file stops updating and all jobs stop firing with no
                    # error logged.  Poll LOCK_NB against a deadline instead;
                    # on timeout, log loudly and fall through to the same
                    # in-process-only degraded mode used when locking is
                    # unavailable.  A briefly-torn cross-process write is
                    # strictly better than a permanently dead scheduler.
                    _deadline = time.monotonic() + _JOBS_LOCK_TIMEOUT_SECONDS
                    while True:
                        try:
                            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            _jobs_lock_state.cross_process_acquired = True
                            break
                        except (OSError, IOError):
                            if time.monotonic() >= _deadline:
                                logger.error(
                                    "Timed out after %.0fs waiting for the cron "
                                    "jobs lock (%s) — another process is holding "
                                    "it. Proceeding with in-process locking only "
                                    "so the scheduler stays alive (#60703).",
                                    _JOBS_LOCK_TIMEOUT_SECONDS,
                                    _jobs_lock_file(),
                                )
                                try:
                                    lock_fd.close()
                                except OSError:
                                    pass
                                lock_fd = None
                                break
                            time.sleep(0.1)
                elif msvcrt is not None:
                    getattr(msvcrt, "locking")(lock_fd.fileno(), getattr(msvcrt, "LK_LOCK"), 1)
                    _jobs_lock_state.cross_process_acquired = True
            except (OSError, IOError) as e:
                # Never let a locking failure take down cron writes — fall back to
                # in-process-only protection (still held via _jobs_file_lock).
                logger.warning("jobs.json cross-process lock unavailable (%s); "
                               "proceeding with in-process lock only", e)
            try:
                yield bool(_jobs_lock_state.cross_process_acquired)
            finally:
                if lock_fd is not None:
                    try:
                        if fcntl is not None:
                            fcntl.flock(lock_fd, fcntl.LOCK_UN)
                        elif msvcrt is not None:
                            getattr(msvcrt, "locking")(lock_fd.fileno(), getattr(msvcrt, "LK_UNLCK"), 1)
                    except (OSError, IOError):
                        pass
                    finally:
                        lock_fd.close()
        finally:
            _jobs_lock_state.depth = 0
            _jobs_lock_state.load_stamp = None
            _jobs_lock_state.cross_process_acquired = False


@contextlib.contextmanager
def contextual_accounting_lock():
    """Serialize the jobs-file + execution-ledger accounting transaction.

    Unlike the general jobs lock this lock fails closed on acquisition errors.
    Proceeding without cross-process exclusion can consume a finite repeat
    twice, so reconciliation must retry later instead of degrading safety.
    """
    with _contextual_accounting_thread_lock:
        ensure_dirs()
        lock_path = _current_cron_store().cron_dir / ".contextual-accounting.lock"
        lock_fd = open(lock_path, "a+b")
        acquired = False
        try:
            if fcntl is not None:
                deadline = time.monotonic() + _CONTEXTUAL_ACCOUNTING_LOCK_TIMEOUT_SECONDS
                while True:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        break
                    except (OSError, IOError) as exc:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Timed out acquiring contextual accounting lock {lock_path}"
                            ) from exc
                        time.sleep(0.1)
            elif msvcrt is not None:
                if os.fstat(lock_fd.fileno()).st_size == 0:
                    lock_fd.write(b"\0")
                    lock_fd.flush()
                lock_fd.seek(0)
                deadline = time.monotonic() + _CONTEXTUAL_ACCOUNTING_LOCK_TIMEOUT_SECONDS
                while True:
                    try:
                        getattr(msvcrt, "locking")(
                            lock_fd.fileno(), getattr(msvcrt, "LK_NBLCK"), 1
                        )
                        acquired = True
                        break
                    except (OSError, IOError) as exc:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Timed out acquiring contextual accounting lock {lock_path}"
                            ) from exc
                        time.sleep(0.1)
            else:  # pragma: no cover - supported hosts provide one primitive
                raise RuntimeError("No cross-process file-lock primitive is available")

            yield
        finally:
            if acquired:
                try:
                    if fcntl is not None:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    elif msvcrt is not None:
                        lock_fd.seek(0)
                        getattr(msvcrt, "locking")(
                            lock_fd.fileno(), getattr(msvcrt, "LK_UNLCK"), 1
                        )
                except (OSError, IOError):
                    logger.error(
                        "Failed to release contextual accounting lock %s",
                        lock_path,
                        exc_info=True,
                    )
            lock_fd.close()


@contextlib.contextmanager
def contextual_authority_lock():
    """Serialize contextual-authority publication with every jobs replace.

    This lock is intentionally separate from the ordinary jobs lock. The jobs
    lock may degrade to process-local protection to keep legacy cron alive;
    this transition lock never degrades because a check/replace race could
    erase the first contextual job.
    """
    with _contextual_authority_thread_lock:
        ensure_dirs()
        lock_path = _current_cron_store().cron_dir / ".contextual-authority.lock"
        lock_fd = open(lock_path, "a+b")
        acquired = False
        try:
            if fcntl is not None:
                deadline = time.monotonic() + _CONTEXTUAL_AUTHORITY_LOCK_TIMEOUT_SECONDS
                while True:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        break
                    except (OSError, IOError) as exc:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Timed out acquiring contextual authority lock {lock_path}"
                            ) from exc
                        time.sleep(0.1)
            elif msvcrt is not None:
                if os.fstat(lock_fd.fileno()).st_size == 0:
                    lock_fd.write(b"\0")
                    lock_fd.flush()
                lock_fd.seek(0)
                deadline = time.monotonic() + _CONTEXTUAL_AUTHORITY_LOCK_TIMEOUT_SECONDS
                while True:
                    try:
                        getattr(msvcrt, "locking")(
                            lock_fd.fileno(), getattr(msvcrt, "LK_NBLCK"), 1
                        )
                        acquired = True
                        break
                    except (OSError, IOError) as exc:
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                f"Timed out acquiring contextual authority lock {lock_path}"
                            ) from exc
                        time.sleep(0.1)
            else:  # pragma: no cover - supported hosts provide one primitive
                raise RuntimeError("No cross-process file-lock primitive is available")

            yield
        finally:
            if acquired:
                try:
                    if fcntl is not None:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    elif msvcrt is not None:
                        lock_fd.seek(0)
                        getattr(msvcrt, "locking")(
                            lock_fd.fileno(), getattr(msvcrt, "LK_UNLCK"), 1
                        )
                except (OSError, IOError):
                    logger.error(
                        "Failed to release contextual authority lock %s",
                        lock_path,
                        exc_info=True,
                    )
            lock_fd.close()


@contextlib.contextmanager
def contextual_transcript_lock():
    """Serialize transcript identity-check, append, and outbox acknowledgement.

    The accounting lock is deliberately shared: both operations bridge the
    execution ledger and another durable store, and neither critical section
    calls the other. Reusing one fail-closed cross-process lock avoids a second
    lock-order domain while keeping both crash gaps linearizable.
    """
    with contextual_accounting_lock():
        yield

# Fields on a cron job that must never change after creation. ``id`` is used
# as a filesystem path component under ``OUTPUT_DIR``; allowing it to be
# updated lets an unsafe value (``../escape``, absolute path, nested) leak
# into output writes/deletes.
_IMMUTABLE_JOB_FIELDS = frozenset(
    {"id", "session_key", "context_binding", "origin", "_contextual_binding_version"}
)


def _job_output_dir(job_id: str) -> Path:
    """Resolve a job's output directory, rejecting any path-escape attempt.

    Job IDs are filesystem path components under ``OUTPUT_DIR``. A legacy or
    crafted ID containing ``..``, absolute paths, or nested separators would
    allow output writes/deletes to escape the cron output sandbox. Reject
    anything that isn't a single safe path component.
    """
    text = str(job_id or "").strip()
    if not text or text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(f"Invalid cron job id for output path: {job_id!r}")
    if Path(text).is_absolute() or Path(text).drive:
        raise ValueError(f"Invalid cron job id for output path: {job_id!r}")
    return _current_cron_store().output_dir / text


def _normalize_skill_list(skill: Optional[str] = None, skills: Optional[Any] = None) -> List[str]:
    """Normalize legacy/single-skill and multi-skill inputs into a unique ordered list."""
    if skills is None:
        raw_items = [skill] if skill else []
    elif isinstance(skills, str):
        raw_items = [skills]
    else:
        raw_items = list(skills)

    normalized: List[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if text and text not in normalized:
            normalized.append(text)
    return normalized


def _apply_skill_fields(job: Dict[str, Any]) -> Dict[str, Any]:
    """Return a job dict with canonical `skills` and legacy `skill` fields aligned."""
    normalized = dict(job)
    skills = _normalize_skill_list(normalized.get("skill"), normalized.get("skills"))
    normalized["skills"] = skills
    normalized["skill"] = skills[0] if skills else None
    return normalized


def _coerce_job_text(value: Any, fallback: str = "") -> str:
    """Coerce legacy/hand-edited nullable cron fields to strings for readers."""
    if value is None:
        return fallback
    return str(value)


def _schedule_display_for_job(job: Dict[str, Any]) -> str:
    display = _coerce_job_text(job.get("schedule_display")).strip()
    if display:
        return display

    schedule = job.get("schedule")
    if isinstance(schedule, dict):
        for key in ("display", "value", "expr", "run_at"):
            text = _coerce_job_text(schedule.get(key)).strip()
            if text:
                return text
    elif schedule is not None:
        return str(schedule)

    return "?"


def _normalize_job_record(job: Dict[str, Any]) -> Dict[str, Any]:
    """Return a read-safe cron job shape for UI/API/tool/scheduler consumers.

    Older or hand-edited jobs can have nullable fields like ``prompt``,
    ``name``, or ``schedule_display``.  Keep storage untouched on read, but
    ensure consumers never crash while formatting or running those records.
    """
    normalized = _apply_skill_fields(job)
    # Migration-safe public default: jobs written before contextual cron are
    # ordinary isolated jobs. Never infer context from a stray legacy key.
    from cron.contextual import normalize_session_target
    normalized["session_target"] = normalize_session_target(
        normalized.get("session_target")
    )
    if normalized["session_target"] != "current":
        normalized.pop("session_key", None)
    job_id = _coerce_job_text(normalized.get("id"), "unknown")
    prompt = _coerce_job_text(normalized.get("prompt"))
    normalized["id"] = job_id
    normalized["prompt"] = prompt

    name = _coerce_job_text(normalized.get("name")).strip()
    if not name:
        script = _coerce_job_text(normalized.get("script")).strip()
        label_source = (
            prompt
            or (normalized["skills"][0] if normalized.get("skills") else "")
            or script
            or job_id
            or "cron job"
        )
        name = label_source[:50].strip() or "cron job"
    normalized["name"] = name
    normalized["schedule_display"] = _schedule_display_for_job(normalized)

    # Display state is derived from the scheduler-honoured ``enabled`` flag so a
    # half-paused record (enabled=true + state/paused_at) cannot render as
    # "paused" while the fleet is still live. See effective_job_state().
    normalized["state"] = effective_job_state(normalized)

    return normalized


def _has_pause_marker(job: Dict[str, Any]) -> bool:
    """True when the record carries any operator-facing pause signal."""
    if _coerce_job_text(job.get("state")).strip() == "paused":
        return True
    return bool(job.get("paused_at"))


def is_job_runnable(job: Dict[str, Any]) -> bool:
    """True iff the scheduler may fire this job.

    ``enabled`` is the scheduler-honoured flag. Pause markers (``state`` /
    ``paused_at``) are a second gate so a contradictory half-paused record
    never fires even before self-heal runs.
    """
    if not job.get("enabled", True):
        return False
    if _has_pause_marker(job):
        return False
    return True


def effective_job_state(job: Dict[str, Any]) -> str:
    """Operator-facing state derived from the scheduler-honoured flag.

    A job with ``enabled=true`` must never display as paused — that was the
    07-30 outage failure mode (list looked frozen, fleet kept merging).
    Terminal states (completed/error) are preserved regardless of enabled.
    """
    stored = _coerce_job_text(job.get("state")).strip()
    if stored in {"completed", "error"}:
        return stored
    if not job.get("enabled", True):
        if _has_pause_marker(job) or stored == "paused":
            return "paused"
        return stored or "paused"
    # enabled=true is authoritative: never claim paused
    if stored == "paused" or job.get("paused_at"):
        return "scheduled"
    return stored or "scheduled"


_PRIVATE_CONTEXTUAL_JOB_FIELDS = frozenset(
    {
        "session_key",
        "context_binding",
        "_contextual_binding_version",
        "_contextual_occurrence_claim",
        "_pending_accounting_execution_ids",
        "_last_accounted_execution_id",
    }
)
_PUBLIC_EXECUTION_FIELDS = frozenset(
    {
        "id",
        "status",
        "outcome",
        "phase",
        "created_at",
        "started_at",
        "finished_at",
        "updated_at",
    }
)


def public_job_record(job: Dict[str, Any]) -> Dict[str, Any]:
    """Return the stable public API shape without routing or ledger authority.

    Internal bindings, concrete session identities, result payloads, claims,
    and delivery state are scheduler authority and must never cross a public
    list/get/create/update boundary.
    """
    from cron.contextual import normalize_session_target

    source = copy.deepcopy(dict(job))
    public = source
    for key in _PRIVATE_CONTEXTUAL_JOB_FIELDS:
        public.pop(key, None)
    public_target = normalize_session_target(source.get("session_target"))
    if public_target == "current":
        public["session_target"] = "current"
        public.pop("origin", None)
        # Contextual claim payloads are feature-private authority. Isolated
        # jobs retain their legacy run/fire claim projection unchanged.
        public.pop("run_claim", None)
        public.pop("fire_claim", None)
    else:
        public.pop("session_target", None)
    execution = source.get("latest_execution")
    if public_target == "current":
        if isinstance(execution, dict):
            public["latest_execution"] = {
                key: execution.get(key)
                for key in _PUBLIC_EXECUTION_FIELDS
                if execution.get(key) is not None
            }
        else:
            public.pop("latest_execution", None)
    return public


def _secure_dir(path: Path):
    """Set directory to owner-only access (0700). No-op on Windows."""
    try:
        os.chmod(path, 0o700)
    except (OSError, NotImplementedError):
        pass  # Windows or other platforms where chmod is not supported


def _secure_file(path: Path):
    """Set file to owner-only read/write (0600). No-op on Windows."""
    try:
        if path.exists():
            os.chmod(path, 0o600)
    except (OSError, NotImplementedError):
        pass


def _preserve_file_ownership(path: Path, before: Optional[os.stat_result]) -> None:
    """Restore a rewritten file's previous owner (POSIX, privileged writer only).

    The atomic-write pattern (mkstemp + replace) makes the rewritten file owned
    by the *writer's* euid. When a root shell runs a state-writing cron CLI
    command (``docker exec hermes hermes cron create ...`` — ``docker exec``
    defaults to root) against a store owned by the unprivileged gateway user,
    the replace flips ``jobs.json`` to ``root:root`` mode 600 and the gateway's
    ticker (uid 1000) is silently locked out of every subsequent tick (#68483).

    Root can always hand ownership back, so do exactly that: when the euid is 0
    and the pre-replace owner differs, chown the new file to the previous
    uid/gid. Unprivileged writers are a no-op (their own rewrite already heals
    a root-owned file back to their uid, and they couldn't chown anyway).
    No-op on Windows. Best-effort: a failure must never break the save.
    """
    if before is None or os.name != "posix":
        return
    geteuid = getattr(os, "geteuid", None)
    getegid = getattr(os, "getegid", None)
    if geteuid is None or getegid is None:
        return
    try:
        euid = geteuid()
        if euid != 0:
            return  # unprivileged writer — nothing to (or we could) restore
        if (before.st_uid, before.st_gid) == (euid, getegid()):
            return  # already ours before the rewrite — nothing changed
        os.chown(path, before.st_uid, before.st_gid)
    except OSError as e:
        logger.warning(
            "Could not restore ownership of %s to uid=%s gid=%s after rewrite: %s "
            "— if the gateway runs as a different user, its cron ticker may now "
            "be locked out (see issue #68483).",
            path, before.st_uid, before.st_gid, e,
        )


def ensure_dirs():
    """Ensure cron directories exist with secure permissions."""
    store = _current_cron_store()
    store.cron_dir.mkdir(parents=True, exist_ok=True)
    store.output_dir.mkdir(parents=True, exist_ok=True)
    _secure_dir(store.cron_dir)
    _secure_dir(store.output_dir)


def _contextual_jobs_authority_marker() -> Path:
    """Permanent profile marker that makes degraded whole-file writes strict."""
    return _current_cron_store().cron_dir / ".contextual-jobs-authority"


def _contextual_jobs_authority_pending_marker() -> Path:
    """Crash-recovery receipt for the first contextual jobs-file commit."""
    return _current_cron_store().cron_dir / ".contextual-jobs-authority.pending"


def _fsync_contextual_authority_dir_unlocked() -> None:
    """Best-effort durability barrier for authority sidecar changes."""
    marker = _contextual_jobs_authority_marker()
    try:
        dir_fd = os.open(marker.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        # Directory fsync is unavailable on some platforms/filesystems.
        pass


def _jobs_payload_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_contextual_jobs_authority_pending_unlocked(tmp_path: str) -> None:
    """Durably record the exact payload expected to complete first authority."""
    pending = _contextual_jobs_authority_pending_marker()
    atomic_write_text(
        pending,
        json.dumps(
            {
                "version": 1,
                "jobs_payload_sha256": _jobs_payload_sha256(tmp_path),
            },
            sort_keys=True,
        ),
        create_mode=0o600,
    )
    _fsync_contextual_authority_dir_unlocked()


def _remove_contextual_authority_path_unlocked(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    _fsync_contextual_authority_dir_unlocked()


def _recover_contextual_jobs_authority_unlocked(jobs_file: Path) -> bool:
    """Resolve a crash-interrupted first-authority transition.

    The pending receipt is durable before the permanent marker is published.
    Therefore a matching jobs payload proves the first contextual commit
    crossed the atomic replacement boundary; a mismatch proves that it did
    not and both sidecars must be rolled back.
    """
    marker = _contextual_jobs_authority_marker()
    pending = _contextual_jobs_authority_pending_marker()
    if not pending.exists():
        return marker.exists()

    try:
        payload = json.loads(pending.read_text(encoding="utf-8"))
        expected = payload.get("jobs_payload_sha256")
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Unreadable contextual authority transition receipt {pending}"
        ) from exc
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(
            f"Invalid contextual authority transition receipt {pending}"
        )

    try:
        committed = _jobs_payload_sha256(jobs_file) == expected
    except FileNotFoundError:
        committed = False

    if committed:
        if not marker.exists():
            _publish_contextual_jobs_authority_unlocked()
        _remove_contextual_authority_path_unlocked(pending)
        return True

    _remove_contextual_authority_path_unlocked(marker)
    _remove_contextual_authority_path_unlocked(pending)
    return False


def _contextual_jobs_authority_enabled(jobs_file: Optional[Path] = None) -> bool:
    """Return committed authority after resolving any interrupted transition."""
    target = jobs_file or _current_cron_store().jobs_file
    with contextual_authority_lock():
        return _recover_contextual_jobs_authority_unlocked(target)


def _publish_contextual_jobs_authority_unlocked() -> None:
    """Publish the marker while the caller holds contextual_authority_lock()."""
    marker = _contextual_jobs_authority_marker()
    fd = os.open(marker, os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    _secure_file(marker)
    _fsync_contextual_authority_dir_unlocked()


# =============================================================================
# Schedule Parsing
# =============================================================================

def parse_duration(s: str) -> int:
    """
    Parse duration string into minutes.
    
    Examples:
        "30m" → 30
        "2h" → 120
        "1d" → 1440
    """
    s = s.strip().lower()
    match = re.match(r'^(\d+)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$', s)
    if not match:
        raise ValueError(f"Invalid duration: '{s}'. Use format like '30m', '2h', or '1d'")
    
    value = int(match.group(1))
    unit = match.group(2)[0]  # First char: m, h, or d
    
    multipliers = {'m': 1, 'h': 60, 'd': 1440}
    return value * multipliers[unit]


def parse_schedule(schedule: str) -> Dict[str, Any]:
    """
    Parse schedule string into structured format.
    
    Returns dict with:
        - kind: "once" | "interval" | "cron"
        - For "once": "run_at" (ISO timestamp)
        - For "interval": "minutes" (int)
        - For "cron": "expr" (cron expression)
    
    Examples:
        "30m"              → once in 30 minutes
        "2h"               → once in 2 hours
        "every 30m"        → recurring every 30 minutes
        "every 2h"         → recurring every 2 hours
        "0 9 * * *"        → cron expression
        "2026-02-03T14:00" → once at timestamp
    """
    schedule = schedule.strip()
    original = schedule
    schedule_lower = schedule.lower()
    
    # "every X" pattern → recurring interval
    if schedule_lower.startswith("every "):
        duration_str = schedule[6:].strip()
        minutes = parse_duration(duration_str)
        return {
            "kind": "interval",
            "minutes": minutes,
            "display": f"every {minutes}m"
        }
    
    # Check for cron expression (5 or 6 space-separated fields)
    # Cron fields: minute hour day month weekday [year]
    parts = schedule.split()
    if len(parts) >= 5 and all(
        re.match(r'^[\d\*\-,/]+$', p) for p in parts[:5]
    ):
        if not _ensure_croniter():
            raise ValueError("Cron expressions require 'croniter' package. Install with: pip install croniter")
        # Validate cron expression
        try:
            croniter(schedule)
        except Exception as e:
            raise ValueError(f"Invalid cron expression '{schedule}': {e}")
        return {
            "kind": "cron",
            "expr": schedule,
            "display": schedule
        }
    
    # ISO timestamp (contains T or looks like date)
    if 'T' in schedule or re.match(r'^\d{4}-\d{2}-\d{2}', schedule):
        try:
            # Parse and validate
            dt = datetime.fromisoformat(schedule.replace('Z', '+00:00'))
            # Make naive timestamps timezone-aware at parse time so the stored
            # value doesn't depend on the system timezone matching at check time.
            #
            # Anchor to the CONFIGURED Hermes timezone, not the server's local
            # timezone. The due-check (`get_due_jobs`) compares `next_run_at`
            # against `hermes_time.now()`, which uses the configured zone. If a
            # naive "20:07" were interpreted as server-local (e.g. UTC) while
            # now() runs in Asia/Kolkata, the stored instant would land hours
            # off from the user's wall-clock intent — far enough that one-shots
            # never become due and recurring jobs fire at the wrong time. Using
            # the configured zone makes "20:07" mean 20:07 on the same clock the
            # scheduler checks against (#51021).
            if dt.tzinfo is None:
                hermes_tz = _hermes_now().tzinfo
                dt = dt.replace(tzinfo=hermes_tz)
            return {
                "kind": "once",
                "run_at": dt.isoformat(),
                "display": f"once at {dt.strftime('%Y-%m-%d %H:%M')}"
            }
        except ValueError as e:
            raise ValueError(f"Invalid timestamp '{schedule}': {e}")
    
    # Duration like "30m", "2h", "1d" → one-shot from now
    try:
        minutes = parse_duration(schedule)
        run_at = _hermes_now() + timedelta(minutes=minutes)
        return {
            "kind": "once",
            "run_at": run_at.isoformat(),
            "display": f"once in {original}"
        }
    except ValueError:
        pass
    
    raise ValueError(
        f"Invalid schedule '{original}'. Use:\n"
        f"  - Duration: '30m', '2h', '1d' (one-shot)\n"
        f"  - Interval: 'every 30m', 'every 2h' (recurring)\n"
        f"  - Cron: '0 9 * * *' (cron expression)\n"
        f"  - Timestamp: '2026-02-03T14:00:00' (one-shot at time)"
    )


def _schedule_dict(job: Dict[str, Any]) -> Dict[str, Any]:
    """Return a structured schedule without mutating legacy persisted strings."""
    value = job.get("schedule")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return parse_schedule(value)
        except ValueError:
            return {}
    return {}


def _ensure_aware(dt: datetime) -> datetime:
    """Return a timezone-aware datetime in Hermes configured timezone.

    Backward compatibility:
    - Older stored timestamps may be naive.
    - Naive values are interpreted as *system-local wall time* (the timezone
      `datetime.now()` used when they were created), then converted to the
      configured Hermes timezone.

    This preserves relative ordering for legacy naive timestamps across
    timezone changes and avoids false not-due results.
    """
    target_tz = _hermes_now().tzinfo
    if dt.tzinfo is None:
        local_tz = datetime.now().astimezone().tzinfo
        return dt.replace(tzinfo=local_tz).astimezone(target_tz)
    return dt.astimezone(target_tz)


def _timezone_offset_mismatch(stored: datetime, current: datetime) -> bool:
    """Return True when a stored aware timestamp uses a different UTC offset.

    Naive stored timestamps return False: they carry no offset to compare, and
    are normalized by ``_ensure_aware`` instead — they intentionally never take
    the offset-repair path.
    """
    if stored.tzinfo is None or current.tzinfo is None:
        return False
    return stored.utcoffset() != current.utcoffset()


def _stored_wall_clock_is_future(stored: datetime, current: datetime) -> bool:
    """Return True when the stored local wall-clock time has not arrived yet.

    Cron schedules express local wall-clock intent. If Hermes/system local time
    changes after next_run_at was persisted, an old offset can make a future
    wall-clock run look due at the converted absolute time (for example
    21:00+10 becomes 13:00+02). Comparing naive wall-clock values lets us
    distinguish that migration case from a genuinely missed run whose scheduled
    wall time has already passed.
    """
    return stored.replace(tzinfo=None) > current.replace(tzinfo=None)


def _recoverable_oneshot_run_at(
    schedule: Dict[str, Any],
    now: datetime,
    *,
    last_run_at: Optional[str] = None,
) -> Optional[str]:
    """Return a one-shot run time if it is still eligible to fire.

    One-shot jobs get a small grace window so jobs created a few seconds after
    their requested minute still run on the next tick. Once a one-shot has
    already run, it is never eligible again.
    """
    if not isinstance(schedule, dict) or schedule.get("kind") != "once":
        return None
    if last_run_at:
        return None

    run_at = schedule.get("run_at")
    if not run_at:
        return None

    try:
        run_at_dt = _ensure_aware(datetime.fromisoformat(run_at))
    except Exception:
        return None
    if run_at_dt >= now - timedelta(seconds=ONESHOT_GRACE_SECONDS):
        return run_at
    return None


def _compute_grace_seconds(schedule: dict) -> int:
    """Compute how late a job can be and still catch up instead of fast-forwarding.

    Uses half the schedule period, clamped between 120 seconds and 2 hours.
    This ensures daily jobs can catch up if missed by up to 2 hours,
    while frequent jobs (every 5-10 min) still fast-forward quickly.
    """
    MIN_GRACE = 120
    MAX_GRACE = 7200  # 2 hours

    kind = schedule.get("kind")

    if kind == "interval":
        period_seconds = schedule.get("minutes", 1) * 60
        grace = period_seconds // 2
        return max(MIN_GRACE, min(grace, MAX_GRACE))

    if kind == "cron" and _ensure_croniter():
        expr = schedule.get("expr")
        if expr:
            try:
                now = _hermes_now()
                cron = croniter(expr, now)
                first = cron.get_next(datetime)
                second = cron.get_next(datetime)
                period_seconds = int((second - first).total_seconds())
                grace = period_seconds // 2
                return max(MIN_GRACE, min(grace, MAX_GRACE))
            except Exception:
                pass

    return MIN_GRACE


def compute_next_run(schedule: Dict[str, Any], last_run_at: Optional[str] = None) -> Optional[str]:
    """
    Compute the next run time for a schedule.

    Returns ISO timestamp string, or None if no more runs.
    """
    now = _hermes_now()

    if not isinstance(schedule, dict):
        return None
    kind = schedule.get("kind")
    if kind is None:
        return None

    if kind == "once":
        return _recoverable_oneshot_run_at(schedule, now, last_run_at=last_run_at)

    elif kind == "interval":
        minutes = schedule.get("minutes")
        if minutes is None:
            return None
        if last_run_at:
            try:
                last = _ensure_aware(datetime.fromisoformat(last_run_at))
                next_run = last + timedelta(minutes=minutes)
            except Exception:
                next_run = now + timedelta(minutes=minutes)
        else:
            # First run is now + interval
            next_run = now + timedelta(minutes=minutes)
        return next_run.isoformat()

    elif kind == "cron":
        expr = schedule.get("expr")
        if not expr:
            return None
        if not _ensure_croniter():
            logger.warning(
                "Cannot compute next run for cron schedule %r: 'croniter' is "
                "not installed. croniter is a core dependency as of v0.9.x; "
                "reinstall hermes-agent or run 'pip install croniter' in your "
                "runtime env.",
                expr,
            )
            return None
        # Use last_run_at as the croniter base when available, consistent
        # with interval jobs.  This ensures that after a crash/restart,
        # the next run is anchored to the actual last execution time
        # rather than to an arbitrary restart time.
        base_time = now
        if last_run_at:
            try:
                base_time = _ensure_aware(datetime.fromisoformat(last_run_at))
            except Exception:
                base_time = now
        cron = croniter(expr, base_time)
        next_run = cron.get_next(datetime)
        return next_run.isoformat()

    return None


# =============================================================================
# Ticker heartbeat (liveness signal for `hermes cron status`)
# =============================================================================

def _atomic_write_epoch(path: Path) -> None:
    """Atomically write the current epoch time to ``path``.

    Delegates to :func:`utils.atomic_write_text` (tmpfile + fsync +
    ``atomic_replace``, same pattern as ``save_jobs``) so a concurrent reader
    in another process (``hermes cron status``) never sees a torn/truncated
    file. Best-effort: failures are swallowed by callers.
    """
    ensure_dirs()
    atomic_write_text(path, str(time.time()), tmp_prefix=".hb_")


def _atomic_write_counter(path: Path, value: int) -> None:
    """Atomically persist a non-negative integer counter."""
    ensure_dirs()
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".count_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(max(0, value)))
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def record_ticker_heartbeat(success: bool = False) -> None:
    """Record a ticker liveness signal, and optionally a successful-tick signal.

    The ticker calls this once per loop iteration. ``success=True`` additionally
    bumps the *last successful tick* marker. We track two distinct signals so
    `hermes cron status` can tell a thread that is merely *alive and looping*
    (heartbeat fresh, success stale) from one that is actually *firing jobs*
    (both fresh) — a ticker stuck failing every tick would otherwise keep the
    plain heartbeat fresh and falsely report healthy (#32612, #32895).

    Resolution uses ``_current_cron_store()`` so the heartbeat is correctly
    scoped to the active profile's store — critical under multiplex_profiles
    where each profile needs its own liveness signal (#69377).

    Best-effort: a write failure must never disrupt the tick loop.
    """
    store = _current_cron_store()
    try:
        _atomic_write_epoch(store.cron_dir / "ticker_heartbeat")
    except Exception:
        pass
    if success:
        try:
            _atomic_write_epoch(store.cron_dir / "ticker_last_success")
        except Exception:
            pass


def _epoch_file_age(path: Path) -> Optional[float]:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return max(0.0, time.time() - float(raw))
    except Exception:
        return None


def get_ticker_heartbeat_age() -> Optional[float]:
    """Seconds since the ticker loop last iterated, or None if unknown.

    None = heartbeat file missing/unreadable (older build, never ran, or a
    torn read). Callers treat None as "cannot determine", not "dead".

    Resolution uses ``_current_cron_store()`` so the heartbeat is correctly
    scoped to the active profile — critical under multiplex_profiles where
    ``hermes cron status`` must report per-profile liveness (#69377).
    """
    store = _current_cron_store()
    return _epoch_file_age(store.cron_dir / "ticker_heartbeat")


def get_ticker_success_age() -> Optional[float]:
    """Seconds since the ticker last completed a tick WITHOUT raising, or None.

    Resolution uses ``_current_cron_store()`` so the heartbeat is correctly
    scoped to the active profile — critical under multiplex_profiles where
    ``hermes cron status`` must report per-profile liveness (#69377).
    """
    store = _current_cron_store()
    return _epoch_file_age(store.cron_dir / "ticker_last_success")


def record_catch_up_occurrence() -> None:
    """Increment the profile-local stale-schedule catch-up counter, best effort."""
    path = _current_cron_store().cron_dir / "catch_up_occurrences"
    try:
        try:
            value = int(path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            value = 0
        _atomic_write_counter(path, max(0, value) + 1)
    except Exception:
        pass


def record_ticker_error(message: str) -> None:
    """Persist the most recent tick failure so other processes can surface it.

    The ticker thread lives inside the gateway process; ``hermes cron
    status``/``list`` run in a separate process and previously could only
    infer "ticks may be failing" from marker staleness, with no clue WHY.
    A root-owned ``jobs.json`` (#68483) failed every tick for ~14h with the
    reason visible only in the gateway's errors.log. Writing the last error
    next to the heartbeat markers gives the CLI something concrete to show.

    Best-effort: a write failure must never disrupt the tick loop.
    """
    store = _current_cron_store()
    path = store.cron_dir / "ticker_last_error"
    try:
        ensure_dirs()
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), suffix=".tmp", prefix=".terr_"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"{time.time()}\n{message.strip()}\n")
                f.flush()
                os.fsync(f.fileno())
            atomic_replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception:
        pass


def get_catch_up_occurrence_count() -> int:
    """Return the profile-local stale-schedule catch-up count."""
    path = _current_cron_store().cron_dir / "catch_up_occurrences"
    try:
        return max(0, int(path.read_text(encoding="utf-8").strip()))
    except (OSError, ValueError):
        return 0


def clear_ticker_error() -> None:
    """Remove the last-tick-error marker after a successful tick. Best-effort."""
    store = _current_cron_store()
    try:
        (store.cron_dir / "ticker_last_error").unlink()
    except OSError:
        pass


def get_ticker_last_error() -> Optional[str]:
    """Return the most recent recorded tick error message, or None."""
    store = _current_cron_store()
    try:
        raw = (store.cron_dir / "ticker_last_error").read_text(encoding="utf-8")
    except Exception:
        return None
    lines = raw.splitlines()
    if len(lines) < 2:
        return None
    message = "\n".join(lines[1:]).strip()
    return message or None


# =============================================================================
# Job CRUD Operations
# =============================================================================

def _parse_jobs_file(jobs_file: Path) -> Tuple[Any, bool]:
    """Tolerantly parse jobs.json; shared by load_jobs and the save-path peek.

    Returns ``(data, used_strict_fallback)``. utf-8-sig absorbs a Windows
    BOM; a strict parse failure is retried with ``strict=False`` to survive
    bare control characters in string values. IO errors from the open and
    parse errors from the fallback propagate to the caller, which decides
    between repair (load_jobs) and bail-out (peek).
    """
    with open(jobs_file, "r", encoding="utf-8-sig") as f:
        raw = f.read()
    try:
        return json.loads(raw), False
    except json.JSONDecodeError:
        return json.loads(raw, strict=False), True


def load_jobs() -> List[Dict[str, Any]]:
    """Load all jobs from storage."""
    jobs_file = _current_cron_store().jobs_file
    ensure_dirs()
    # Stamp BEFORE reading (fail-safe direction — see _record_load_stamp):
    # a sibling write racing this load leaves the stamp older than disk, so
    # the save-path merge runs instead of being wrongly skipped.
    pre_read_stamp = _jobs_file_stamp(jobs_file)
    if not jobs_file.exists():
        _record_load_stamp(None)
        return []

    try:
        data, _strict_retry = _parse_jobs_file(jobs_file)
    except IOError as e:
        logger.error("IOError reading jobs.json: %s", e)
        raise RuntimeError(f"Failed to read cron database: {e}") from e
    except Exception as e:
        logger.error("Failed to auto-repair jobs.json: %s", e)
        raise RuntimeError(f"Cron database corrupted and unrepairable: {e}") from e

    # Validate the top-level JSON shape: accept a dict (expected) or a bare
    # list (auto-repair). Anything else (str/number/null) is corruption that
    # would otherwise raise an uncaught AttributeError on ``.get()`` and take
    # down the whole cron subsystem.
    if isinstance(data, dict):
        jobs = data.get("jobs", [])
        if _strict_retry and jobs:
            # Hit control-character corruption — rewrite with proper escaping.
            save_jobs(jobs)
            logger.warning("Auto-repaired jobs.json (had invalid control characters)")
        _record_load_stamp(pre_read_stamp)
        return jobs
    if isinstance(data, list):
        # Bare array — likely saved/edited outside save_jobs(). Wrap it back
        # into the expected {"jobs": [...]} structure.
        if data:
            save_jobs(data)
            logger.warning("Auto-repaired jobs.json (bare list wrapped as dict)")
        _record_load_stamp(pre_read_stamp)
        return data

    raise RuntimeError(
        f"Cron database corrupted: expected {{'jobs': [...]}}, got {type(data).__name__}"
    )


def _peek_jobs_unlocked() -> Optional[List[Dict[str, Any]]]:
    """Best-effort read of on-disk jobs without repair side-effects.

    Caller must hold ``_jobs_lock()``. Returns ``[]`` when the file is
    missing, ``None`` when the payload is unreadable/corrupt (caller should
    not attempt a shrink-merge against an unknown baseline). Never calls
    ``save_jobs`` — the repair-free property is what keeps the save path
    re-entrancy-safe (a repairing read here would recurse through
    ``_save_jobs_unlocked``).
    """
    jobs_file = _current_cron_store().jobs_file
    if not jobs_file.exists():
        return []
    try:
        data, _ = _parse_jobs_file(jobs_file)
    except Exception:
        return None
    if isinstance(data, dict):
        jobs = data.get("jobs", [])
        return jobs if isinstance(jobs, list) else None
    if isinstance(data, list):
        return data
    return None


def _jobs_file_stamp(jobs_file: Path) -> Optional[Tuple[int, int, int]]:
    """Cheap change-detection stamp for jobs.json: ``(mtime_ns, size, ino)``.

    ``None`` means the file is missing/unstatable. Used as a fast-path gate
    in front of the shrink-merge so the healthy no-race save costs one
    ``stat()`` instead of a full read+parse (the ``advance_next_runs``
    batching exists because this path is hot — see its docstring).
    ``st_ino`` is included because every legitimate writer goes through
    mkstemp+rename (new inode), so even a same-size write inside one mtime
    quantum on a coarse-clock filesystem (ext4 jiffies, network mounts)
    cannot false-match.
    """
    try:
        st = jobs_file.stat()
        return (st.st_mtime_ns, st.st_size, st.st_ino)
    except OSError:
        return None


def _record_load_stamp(stamp: Optional[Tuple[int, int, int]]) -> None:
    """Remember jobs.json's stamp for the enclosing _jobs_lock() section.

    No-op outside a critical section. Lets the save path skip the
    shrink-merge parse when the file provably hasn't changed since this
    section loaded it (#80703's fast-path). The caller must capture the
    stamp BEFORE reading the file: a sibling landing mid-read then leaves
    the recorded stamp OLDER than disk — a mismatch, so the merge runs
    (fail-safe direction). Stamping after the read would let that sibling's
    write be certified as "seen" without being in the loaded payload,
    wrongly suppressing the recovery.
    """
    if not getattr(_jobs_lock_state, "depth", 0):
        return
    _jobs_lock_state.load_stamp = stamp


def _merge_unexpected_disk_jobs(
    jobs: List[Dict[str, Any]],
    *,
    removed_ids: Optional[Collection[str]] = None,
) -> List[Dict[str, Any]]:
    """Return *jobs* plus any on-disk jobs missing from the save payload (#80624).

    Under ``_jobs_lock()``'s degraded flock-timeout path (#60703), two
    processes can both believe they own the store. A writer that loaded an
    older/smaller snapshot then calls ``save_jobs`` and would otherwise
    clobber concurrent creates (the filed ``no_agent`` watchdog pattern:
    CLI/tool create succeeds, then a gateway tick/remove rewrites
    ``jobs.json`` empty or without the new id).

    Intentional deletes pass ``removed_ids``. Any other id present on disk
    but absent from *jobs* is treated as a concurrent create and merged
    back before the atomic write. The caller's list is never mutated — a
    new list is returned when anything was recovered.

    Fast path: when the enclosing critical section recorded a load stamp
    and the file's ``(mtime_ns, size)`` still matches, nothing can have
    changed underneath us, so the read+parse is skipped entirely — one
    ``stat()`` on the healthy no-race save.
    """
    stamp = getattr(_jobs_lock_state, "load_stamp", None)
    if stamp is not None and _jobs_file_stamp(_current_cron_store().jobs_file) == stamp:
        return jobs

    disk_jobs = _peek_jobs_unlocked()
    if disk_jobs is None:
        return jobs

    intended_remove = {str(i) for i in (removed_ids or ()) if i}
    new_ids: Set[str] = set()
    for job in jobs:
        if isinstance(job, dict) and job.get("id"):
            new_ids.add(str(job["id"]))

    recovered: List[Dict[str, Any]] = []
    for disk_job in disk_jobs:
        if not isinstance(disk_job, dict):
            continue
        disk_id = disk_job.get("id")
        if not disk_id:
            continue
        disk_id = str(disk_id)
        if disk_id in new_ids or disk_id in intended_remove:
            continue
        recovered.append(disk_job)
        new_ids.add(disk_id)

    if not recovered:
        return jobs
    logger.warning(
        "Preserved %d cron job(s) present on disk but missing from the "
        "in-memory save payload (concurrent create under degraded lock "
        "or stale writer) (#80624): %s",
        len(recovered),
        [j.get("id") for j in recovered],
    )
    return jobs + recovered


def _replace_jobs_payload_with_authority(
    tmp_path: str,
    jobs_file: Path,
    jobs: List[Dict[str, Any]],
) -> None:
    """Replace jobs.json while atomically enforcing contextual authority."""
    publishes_contextual = any(
        isinstance(job, dict)
        and job.get("session_target", "isolated") == "current"
        for job in jobs
    )
    marker = _contextual_jobs_authority_marker()
    pending = _contextual_jobs_authority_pending_marker()
    with contextual_authority_lock():
        cross_process_acquired = bool(
            getattr(_jobs_lock_state, "cross_process_acquired", False)
        )
        marker_preexisting = _recover_contextual_jobs_authority_unlocked(jobs_file)
        if not cross_process_acquired and (marker_preexisting or publishes_contextual):
            raise RuntimeError(
                "Refusing a degraded jobs.json write after contextual jobs "
                "authority was enabled for this profile"
            )

        marker_created = publishes_contextual and not marker_preexisting
        payload_replaced = False
        try:
            if marker_created:
                _publish_contextual_jobs_authority_pending_unlocked(tmp_path)
                _publish_contextual_jobs_authority_unlocked()
            atomic_replace(tmp_path, jobs_file)
            payload_replaced = True
            if marker_created:
                _remove_contextual_authority_path_unlocked(pending)
        except BaseException:
            if marker_created and not payload_replaced:
                try:
                    _remove_contextual_authority_path_unlocked(marker)
                    _remove_contextual_authority_path_unlocked(pending)
                except OSError:
                    logger.error(
                        "Failed to roll back contextual authority transition %s",
                        pending,
                        exc_info=True,
                    )
            raise


def _save_jobs_unlocked(
    jobs: List[Dict[str, Any]],
    *,
    removed_ids: Optional[Collection[str]] = None,
    replace: bool = False,
):
    """Save all jobs to storage. Caller must hold _jobs_lock().

    ``removed_ids`` lists job ids this mutation intentionally deleted.
    ``replace=True`` skips the shrink-merge guard (tests / disaster recovery
    that mean to rewrite the store wholesale).
    """
    jobs_file = _current_cron_store().jobs_file
    ensure_dirs()
    # Snapshot the current owner BEFORE the atomic replace so a privileged
    # writer (root CLI in Docker) can hand ownership back to the gateway user
    # afterwards instead of locking its ticker out (#68483). When the file is
    # being created for the first time, inherit the cron dir's owner — in the
    # Docker image that is the PUID/PGID gateway user who must be able to
    # read the store on the next tick.
    try:
        _stat_before = os.stat(jobs_file)
    except OSError:
        try:
            _stat_before = os.stat(jobs_file.parent)
        except OSError:
            _stat_before = None

    # Shrink-merge + rewrite loop (#80624): under the degraded flock-timeout
    # path another process can create a job between our load and our write.
    # Merge unexpected disk ids into the payload, stage the write, then
    # re-peek; if new ids appeared, merge again and restage before replace.
    # The merge itself fast-paths to a single stat() when the enclosing
    # section's load stamp still matches (see _merge_unexpected_disk_jobs).
    tmp_path = None
    try:
        for _attempt in range(5):
            if not replace:
                jobs = _merge_unexpected_disk_jobs(jobs, removed_ids=removed_ids)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(jobs_file.parent), suffix=".tmp", prefix=".jobs_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(
                        {"jobs": jobs, "updated_at": _hermes_now().isoformat()},
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
                    f.flush()
                    os.fsync(f.fileno())
            except BaseException:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                tmp_path = None
                raise

            if not replace:
                # Verify-after-stage: a sibling landing while we serialized
                # the payload must trigger another merge round. Same stamp
                # fast path as the merge — an unchanged stamp proves nothing
                # was written, so the full parse is skipped.
                _stamp = getattr(_jobs_lock_state, "load_stamp", None)
                _unchanged = (
                    _stamp is not None and _jobs_file_stamp(jobs_file) == _stamp
                )
                disk_jobs = None if _unchanged else _peek_jobs_unlocked()
                if disk_jobs is not None:
                    payload_ids = {
                        str(j["id"])
                        for j in jobs
                        if isinstance(j, dict) and j.get("id")
                    }
                    intended = {str(i) for i in (removed_ids or ()) if i}
                    if any(
                        isinstance(dj, dict)
                        and dj.get("id")
                        and str(dj["id"]) not in payload_ids
                        and str(dj["id"]) not in intended
                        for dj in disk_jobs
                    ):
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                        tmp_path = None
                        continue

            _replace_jobs_payload_with_authority(tmp_path, jobs_file, jobs)
            tmp_path = None
            _secure_file(jobs_file)
            _preserve_file_ownership(jobs_file, _stat_before)
            # Invalidate (never refresh) the stamp after writing: the stamp
            # certifies "this section's loaded payload still matches disk",
            # which stops being provable the moment anyone writes. A refresh
            # here would let a nested save (e.g. create_job inside a broader
            # section) certify disk against an OUTER caller's stale payload
            # and deterministically clobber the nested create; it also races
            # a degraded sibling landing between replace and stat. Later
            # saves in this section simply take the full merge (fail-safe).
            _record_load_stamp(None)
            return

        # Exhausted retries — last merge + write without another re-peek.
        if not replace:
            jobs = _merge_unexpected_disk_jobs(jobs, removed_ids=removed_ids)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(jobs_file.parent), suffix=".tmp", prefix=".jobs_"
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                {"jobs": jobs, "updated_at": _hermes_now().isoformat()},
                f,
                indent=2,
                ensure_ascii=False,
            )
            f.flush()
            os.fsync(f.fileno())
        _replace_jobs_payload_with_authority(tmp_path, jobs_file, jobs)
        tmp_path = None
        _secure_file(jobs_file)
        _preserve_file_ownership(jobs_file, _stat_before)
    except BaseException:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        raise


def save_jobs(
    jobs: List[Dict[str, Any]],
    *,
    removed_ids: Optional[Collection[str]] = None,
    replace: bool = False,
):
    """Save all jobs to storage.

    See ``_save_jobs_unlocked`` for ``removed_ids`` / ``replace`` semantics
    (shrink-merge guard against concurrent-create clobber, #80624).
    """
    publishes_contextual = any(
        isinstance(job, dict)
        and job.get("session_target", "isolated") == "current"
        for job in jobs
    )
    with _jobs_lock() as cross_process_locked:
        if not cross_process_locked and (
            publishes_contextual or _contextual_jobs_authority_enabled()
        ):
            raise RuntimeError(
                "Refusing a degraded jobs.json write for a profile with "
                "contextual jobs authority"
            )
        _save_jobs_unlocked(jobs, removed_ids=removed_ids, replace=replace)


def _normalize_workdir(workdir: Optional[str]) -> Optional[str]:
    """Normalize and validate a cron job workdir.

    Rules:
      - Empty / None → None (feature off, preserves old behaviour).
      - ``~`` is expanded.  Relative paths are rejected — cron jobs run detached
        from any shell cwd, so relative paths have no stable meaning.
      - The path must exist and be a directory at create/update time.  We do
        NOT re-check at run time (a user might briefly unmount the dir; the
        scheduler will just fall back to old behaviour with a logged warning).

    Returns the absolute path string, or None when disabled.
    Raises ValueError on invalid input.
    """
    if workdir is None:
        return None
    raw = str(workdir).strip()
    if not raw:
        return None
    expanded = Path(raw).expanduser()
    if not expanded.is_absolute():
        raise ValueError(
            f"Cron workdir must be an absolute path (got {raw!r}). "
            f"Cron jobs run detached from any shell cwd, so relative paths are ambiguous."
        )
    resolved = expanded.resolve()
    if not resolved.exists():
        raise ValueError(f"Cron workdir does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Cron workdir is not a directory: {resolved}")
    return str(resolved)


def _resolve_default_model_snapshot() -> Optional[str]:
    """Resolve the global default model the same way the cron ticker does.

    Mirrors the unpinned-model resolution in ``cron/scheduler.py`` ``run_job``:
    read ``config.yaml`` ``model.default`` (or the ``model`` alias / bare string
    form), applying the managed-scope overlay and env expansion. Used by
    ``create_job`` to snapshot the default model for unpinned jobs so a later
    swap of the global default is detected at fire time (#44585).

    Returns the resolved model string, or ``None`` if config is missing/empty
    or resolution fails (fail-open — caller treats ``None`` as "no snapshot").
    """
    try:
        from hermes_cli.config import _expand_env_vars, read_user_config_raw

        cfg_path = get_hermes_home() / "config.yaml"
        if not cfg_path.exists():
            return None
        cfg = read_user_config_raw(cfg_path)
        try:
            from hermes_cli import managed_scope
            cfg = managed_scope.apply_managed_overlay(cfg)
        except Exception:
            pass
        cfg = _expand_env_vars(cfg)
        # Mirror run_job's precedence: the explicit cron-fleet default
        # (cron.model) beats the global chat model for unpinned cron jobs.
        cron_cfg = cfg.get("cron") or {}
        if isinstance(cron_cfg, dict):
            cron_model = cron_cfg.get("model")
            if isinstance(cron_model, str) and cron_model.strip():
                return cron_model.strip()
        model_cfg = cfg.get("model") or {}
        if isinstance(model_cfg, str):
            return model_cfg.strip() or None
        if isinstance(model_cfg, dict):
            default = model_cfg.get("default") or model_cfg.get("model")
            if isinstance(default, str):
                return default.strip() or None
        return None
    except Exception:
        return None


def _normalize_job_optional_text(value: Any, *, strip_trailing_slash: bool = False) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if strip_trailing_slash:
        text = text.rstrip("/")
    return text or None


def _compute_provider_model_snapshots(
    *,
    provider: Any,
    model: Any,
    base_url: Any,
    no_agent: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """Snapshot unpinned inference axes for the provider/model drift guard.

    Agent cron jobs with unpinned provider/model follow global config at fire
    time. Capture the current resolution for each unpinned axis so a later
    global switch fails closed instead of silently changing spend. Pinned axes
    and no-agent script jobs intentionally carry no snapshot.
    """
    normalized_provider = _normalize_job_optional_text(provider)
    normalized_model = _normalize_job_optional_text(model)
    normalized_base_url = _normalize_job_optional_text(
        base_url,
        strip_trailing_slash=True,
    )
    if bool(no_agent):
        return None, None

    provider_snapshot: Optional[str] = None
    model_snapshot: Optional[str] = None
    if normalized_provider is None:
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider

            runtime_kwargs = {"requested": None}
            if normalized_base_url:
                runtime_kwargs["explicit_base_url"] = normalized_base_url
            snap = resolve_runtime_provider(**runtime_kwargs)
            snap_provider = str(snap.get("provider") or "").strip().lower()
            provider_snapshot = snap_provider or None
        except Exception:
            provider_snapshot = None
    if normalized_model is None:
        try:
            model_snapshot = _resolve_default_model_snapshot() or None
        except Exception:
            model_snapshot = None
    return provider_snapshot, model_snapshot


def _normalized_inference_axes(job: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    """Return the stored inference-routing fields in their semantic form."""
    return (
        _normalize_job_optional_text(job.get("provider")),
        _normalize_job_optional_text(job.get("model")),
        _normalize_job_optional_text(job.get("base_url"), strip_trailing_slash=True),
        bool(job.get("no_agent")),
    )


def _validate_job_mode_invariants(
    monitor_script: Optional[str],
    monitor_url: Optional[str],
    no_agent: bool,
    script: Optional[str],
) -> None:
    """Shared create/update validation for job execution-mode invariants.

    ONE owner for the class: create_job and update_job both call this so an
    invariant enforced at create time cannot be violated through the update
    door (monitor jobs silently degrading when no_agent is flipped on, etc.).
    """
    if monitor_script and monitor_url:
        raise ValueError(
            "monitor_script and monitor_url are mutually exclusive — a job "
            "can only have one monitor source."
        )
    if (monitor_script or monitor_url) and no_agent:
        raise ValueError(
            "monitor_script/monitor_url cannot be combined with no_agent=True — "
            "the whole point of a monitor job is to suppress or wake the AGENT "
            "based on source changes. Use a plain no_agent script job instead."
        )
    if no_agent and not script:
        raise ValueError(
            "no_agent=True requires a script — with no agent and no script "
            "there is nothing for the job to run."
        )


def create_job(
    prompt: Optional[str],
    schedule: str,
    name: Optional[str] = None,
    repeat: Optional[int] = None,
    deliver: Optional[str] = None,
    origin: Optional[Dict[str, Any]] = None,
    skill: Optional[str] = None,
    skills: Optional[List[str]] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    script: Optional[str] = None,
    context_from: Optional[Union[str, List[str]]] = None,
    enabled_toolsets: Optional[List[str]] = None,
    workdir: Optional[str] = None,
    no_agent: bool = False,
    attach_to_session: Optional[bool] = None,
    monitor_script: Optional[str] = None,
    monitor_url: Optional[str] = None,
    session_target: str = "isolated",
) -> Dict[str, Any]:
    """
    Create a new cron job.

    Args:
        prompt: The prompt to run (must be self-contained, or a task instruction when skill is set).
                Ignored when ``no_agent=True`` except as an optional name hint.
        schedule: Schedule string (see parse_schedule)
        name: Optional friendly name
        repeat: How many times to run (None = forever, 1 = once)
        deliver: Where to deliver output ("origin", "local", "telegram", etc.)
        origin: Source info where job was created (for "origin" delivery)
        skill: Optional legacy single skill name to load before running the prompt
        skills: Optional ordered list of skills to load before running the prompt
        model: Optional per-job model override
        provider: Optional per-job provider override
        base_url: Optional per-job base URL override
        script: Optional path to a script whose stdout feeds the job. With
                ``no_agent=True`` the script IS the job — its stdout is
                delivered verbatim. Without ``no_agent``, its stdout is
                injected into the agent's prompt as context (data-collection /
                change-detection pattern). Paths resolve under
                ~/.hermes/scripts/; ``.sh`` / ``.bash`` files run via bash,
                anything else via Python.
        context_from: Optional job ID (or list of job IDs) whose most recent output
                      is injected into the prompt as context before each run.
                      Useful for chaining cron jobs: job A finds data, job B processes it.
        enabled_toolsets: Optional list of toolset names to restrict the agent to.
                          When set, only tools from these toolsets are loaded, reducing
                          token overhead. When omitted, all default tools are loaded.
                          Ignored when ``no_agent=True``.
        workdir: Optional absolute path.  When set, the job runs as if launched
                from that directory: AGENTS.md / CLAUDE.md / .cursorrules from
                that directory are injected into the system prompt, and the
                terminal/file/code_exec tools use it as their working directory
                (via TERMINAL_CWD).  When unset, the old behaviour is preserved
                (no context files injected, tools use the scheduler's cwd).
                With ``no_agent=True``, ``workdir`` is still applied as the
                script's cwd so relative paths inside the script behave
                predictably.
        no_agent: When True, skip the agent entirely — run ``script`` on schedule
                and deliver its stdout directly. Empty stdout = silent (no
                delivery). Requires ``script`` to be set. Ideal for classic
                watchdogs and periodic alerts that don't need LLM reasoning.
        monitor_script: Optional path to a cheap monitor source script (same
                resolution/containment rules as ``script``: relative to
                ~/.hermes/scripts/, .sh/.bash via bash, else Python). Each
                tick the script runs FIRST and its output is hashed as exact
                bytes: unchanged output suppresses the agent run entirely
                (recorded as a silent 'no_change' tick); changed output
                injects a MONITOR CHANGE DETECTED block (unified diff + new
                output) into the prompt before a normal agent run. Scripts
                should emit stable output (no timestamps). Mutually exclusive
                with ``monitor_url``; incompatible with ``no_agent=True``.
        monitor_url: Optional http(s) URL used as the monitor source instead
                of a script — fetched with a bounded GET each tick. Same
                hash-suppression semantics as ``monitor_script``.
        session_target: ``isolated`` (default) or explicit ``current`` to bind
                the job immutably to the live gateway conversation. Contextual
                jobs cannot use monitor mode or other isolated-run overrides.

    Returns:
        The created job dict
    """
    parsed_schedule = parse_schedule(schedule)

    # Normalize repeat: treat 0 or negative values as None (infinite)
    if repeat is not None and repeat <= 0:
        repeat = None

    # Auto-set repeat=1 for one-shot schedules if not specified
    if parsed_schedule["kind"] == "once" and repeat is None:
        repeat = 1

    # Default delivery to origin if available, otherwise local
    if deliver is None:
        deliver = "origin" if origin else "local"

    job_id = uuid.uuid4().hex[:12]
    now = _hermes_now().isoformat()

    normalized_skills = _normalize_skill_list(skill, skills)
    normalized_model = _normalize_job_optional_text(model)
    normalized_provider = _normalize_job_optional_text(provider)
    normalized_base_url = _normalize_job_optional_text(base_url, strip_trailing_slash=True)
    normalized_script = str(script).strip() if isinstance(script, str) else None
    normalized_script = normalized_script or None
    normalized_toolsets = [str(t).strip() for t in enabled_toolsets if str(t).strip()] if enabled_toolsets else None
    normalized_toolsets = normalized_toolsets or None
    normalized_workdir = _normalize_workdir(workdir)
    normalized_no_agent = bool(no_agent)
    normalized_attach = attach_to_session if isinstance(attach_to_session, bool) else None
    normalized_monitor_script = str(monitor_script).strip() if isinstance(monitor_script, str) else None
    normalized_monitor_script = normalized_monitor_script or None
    normalized_monitor_url = str(monitor_url).strip() if isinstance(monitor_url, str) else None
    normalized_monitor_url = normalized_monitor_url or None

    # Monitor-mode validation: exactly one source, and monitor mode only
    # makes sense when there IS an agent to suppress/wake. Surface failures at
    # create time so bad configs never reach the scheduler.
    _validate_job_mode_invariants(
        normalized_monitor_script,
        normalized_monitor_url,
        normalized_no_agent,
        normalized_script,
    )

    from cron.contextual import (
        contextual_fields_for_write,
        validate_contextual_job_shape,
    )
    contextual_fields = contextual_fields_for_write(session_target)
    if contextual_fields.get("session_target") == "current":
        from cron.contextual import contextual_origin_from_binding

        # Execution identity and delivery identity are one atomic binding. A
        # stale/caller-supplied origin may never diverge from the live key.
        origin = contextual_origin_from_binding(contextual_fields["context_binding"])

    # Normalize context_from: accept str or list of str, store as list or None
    if isinstance(context_from, str):
        context_from = [context_from.strip()] if context_from.strip() else None
    elif isinstance(context_from, list):
        context_from = [str(j).strip() for j in context_from if str(j).strip()] or None
    else:
        context_from = None

    prompt_text = _coerce_job_text(prompt)

    # Reject cron jobs that schedule gateway-lifecycle commands. Prevents
    # agent-driven SIGTERM-respawn loops under launchd/systemd KeepAlive
    # (#30719). Enforced here (not only in the CLI layer) so the agent's
    # `cronjob` model tool — which calls create_job directly — is also
    # covered, not just `hermes cron create`.
    from cron.lifecycle_guard import check_gateway_lifecycle
    check_gateway_lifecycle(prompt_text, normalized_script)

    label_source = (prompt_text or (normalized_skills[0] if normalized_skills else None) or (normalized_script if normalized_no_agent else None)) or "cron job"

    provider_snapshot, model_snapshot = _compute_provider_model_snapshots(
        provider=normalized_provider,
        model=normalized_model,
        base_url=normalized_base_url,
        no_agent=normalized_no_agent,
    )

    next_run_at = compute_next_run(parsed_schedule)
    if parsed_schedule.get("kind") == "once" and next_run_at is None:
        run_at = parsed_schedule.get("run_at") or schedule
        logger.warning(
            "Rejecting one-shot cron job '%s': run_at %s is outside the %ss grace window",
            name or label_source[:50].strip(),
            run_at,
            ONESHOT_GRACE_SECONDS,
        )
        raise ValueError(
            f"Requested one-shot time {run_at} is more than "
            f"{ONESHOT_GRACE_SECONDS}s in the past and cannot be scheduled."
        )

    job = {
        "id": job_id,
        "name": name or label_source[:50].strip(),
        "prompt": prompt_text,
        "skills": normalized_skills,
        "skill": normalized_skills[0] if normalized_skills else None,
        "model": normalized_model,
        "provider": normalized_provider,
        # Provider/model resolution captured at creation for unpinned jobs
        # (#44585). None for pinned axes, no_agent jobs, resolution failures, and
        # any pre-existing job written before these fields existed (back-compat).
        "provider_snapshot": provider_snapshot,
        "model_snapshot": model_snapshot,
        "base_url": normalized_base_url,
        "script": normalized_script,
        "no_agent": normalized_no_agent,
        "monitor_script": normalized_monitor_script,
        "monitor_url": normalized_monitor_url,
        # Hash-suppression state for monitor jobs: {"last_output_hash": ...,
        # "last_changed_at": ...}. None until the first monitor tick.
        "monitor_state": None,
        "context_from": context_from,
        "schedule": parsed_schedule,
        "schedule_display": parsed_schedule.get("display", schedule),
        "repeat": {
            "times": repeat,  # None = forever
            "completed": 0
        },
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "created_at": now,
        "next_run_at": next_run_at,
        "last_run_at": None,
        "last_status": None,
        "last_error": None,
        "last_delivery_error": None,
        # Delivery configuration
        "deliver": deliver,
        "origin": origin,  # Tracks where job was created for "origin" delivery
        "enabled_toolsets": normalized_toolsets,
        "workdir": normalized_workdir,
        **contextual_fields,
    }
    # Only persist attach_to_session when explicitly set, so existing jobs and
    # the common case stay byte-identical (absent key => fall back to the
    # global cron.mirror_delivery config, default off).
    if normalized_attach is not None:
        job["attach_to_session"] = normalized_attach

    validate_contextual_job_shape(job)

    with _jobs_lock():
        jobs = load_jobs()
        jobs.append(job)
        save_jobs(jobs)

    return job


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get a job by ID."""
    jobs = load_jobs()
    for job in jobs:
        if job["id"] == job_id:
            return _normalize_job_record(job)
    return None


class AmbiguousJobReference(LookupError):
    """Raised when a job name matches more than one job."""

    def __init__(self, ref: str, matches: List[Dict[str, Any]]):
        self.ref = ref
        self.matches = matches
        ids = ", ".join(m["id"] for m in matches)
        super().__init__(
            f"Job name '{ref}' is ambiguous — matches {len(matches)} jobs: {ids}. "
            f"Use the job ID instead."
        )


def resolve_job_ref(ref: str) -> Optional[Dict[str, Any]]:
    """Resolve a job reference (ID or name) to a job record.

    - Exact ID match wins (works even if a different job's name equals this ID).
    - Otherwise, case-insensitive name match.
    - If a name matches more than one job, raises AmbiguousJobReference so the
      caller can surface the matching IDs rather than silently picking one.
    """
    if not ref:
        return None
    jobs = load_jobs()
    for job in jobs:
        if job["id"] == ref:
            return _normalize_job_record(job)
    ref_lower = ref.lower()
    name_matches = [j for j in jobs if (j.get("name") or "").lower() == ref_lower]
    if not name_matches:
        return None
    if len(name_matches) > 1:
        raise AmbiguousJobReference(
            ref, [_normalize_job_record(j) for j in name_matches]
        )
    return _normalize_job_record(name_matches[0])


def list_jobs(include_disabled: bool = False) -> List[Dict[str, Any]]:
    """List all jobs, optionally including disabled ones."""
    jobs = [_normalize_job_record(j) for j in load_jobs()]
    if not include_disabled:
        jobs = [j for j in jobs if j.get("enabled", True)]
    try:
        from cron.executions import latest_executions

        latest = latest_executions([job.get("id", "") for job in jobs])
    except Exception:
        latest = {}
    for job in jobs:
        job["latest_execution"] = latest.get(job.get("id", ""))
    return jobs


def update_job(job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a job by ID, refreshing derived schedule fields when needed."""
    # Block mutation of immutable fields. ``id`` in particular is a filesystem
    # path component under OUTPUT_DIR — letting an update change it leaks
    # path-escape values into output writes/deletes.
    bad_fields = _IMMUTABLE_JOB_FIELDS.intersection(updates or {})
    if bad_fields:
        raise ValueError(
            f"Cron job field(s) cannot be updated: {', '.join(sorted(bad_fields))}"
        )

    with _jobs_lock():
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] != job_id:
                continue

            # Validate / normalize workdir if present in updates.  Empty string
            # or None both mean "clear the field" (restore old behaviour).
            if "workdir" in updates:
                _wd = updates["workdir"]
                if _wd in {None, "", False}:
                    updates["workdir"] = None
                else:
                    updates["workdir"] = _normalize_workdir(_wd)

            # Normalize monitor fields the same way create_job does (empty
            # string clears the field).
            for _mon_field in ("monitor_script", "monitor_url"):
                if _mon_field in updates:
                    _mv = updates[_mon_field]
                    _mv = str(_mv).strip() if isinstance(_mv, str) else None
                    updates[_mon_field] = _mv or None

            # Only an explicit session-target edit re-captures the trusted
            # task-local gateway key. Ordinary edits (including pause/resume)
            # preserve the original binding so they remain safe outside a live
            # request context.
            from cron.contextual import (
                contextual_fields_for_write,
                normalize_session_target,
                validate_contextual_job_shape,
            )
            requested_target = normalize_session_target(
                updates.get("session_target", job.get("session_target"))
            )

            previous_inference_axes = _normalized_inference_axes(job)
            updated = _apply_skill_fields({**job, **updates})

            # Re-check execution-mode invariants on the MERGED record when
            # any participating field changes, so create-time invariants
            # cannot be violated through the update door.
            if {"monitor_script", "monitor_url", "no_agent", "script"}.intersection(updates):
                _upd_script = updated.get("script")
                _upd_script = str(_upd_script).strip() if isinstance(_upd_script, str) else None
                _validate_job_mode_invariants(
                    updated.get("monitor_script") or None,
                    updated.get("monitor_url") or None,
                    bool(updated.get("no_agent")),
                    _upd_script or None,
                )

            previous_target = normalize_session_target(job.get("session_target"))
            was_contextual = bool(
                job.get("_contextual_binding_version")
                or previous_target == "current"
                or job.get("session_key")
                or job.get("context_binding")
            )
            if was_contextual and requested_target != "current":
                raise ValueError(
                    "A contextual cron job's session binding is permanent. "
                    "Delete it and create a new isolated job instead."
                )
            if requested_target == "current":
                if previous_target == "current" and job.get("session_key"):
                    # V1 bindings are immutable. An explicit no-op current
                    # update validates the shape but does not silently move the
                    # job to the editing chat.
                    updated["session_target"] = "current"
                else:
                    fields = contextual_fields_for_write("current")
                    updated.update(fields)
                    from cron.contextual import contextual_origin_from_binding

                    updated["origin"] = contextual_origin_from_binding(
                        fields["context_binding"]
                    )
            else:
                updated.pop("session_target", None)
                updated.pop("session_key", None)
                updated.pop("context_binding", None)
                updated.pop("_contextual_binding_version", None)
            validate_contextual_job_shape(updated)

            schedule_changed = "schedule" in updates
            inference_fields_changed = bool(
                {"provider", "model", "base_url", "no_agent"}.intersection(updates)
            ) and _normalized_inference_axes(updated) != previous_inference_axes

            if "skills" in updates or "skill" in updates:
                normalized_skills = _normalize_skill_list(updated.get("skill"), updated.get("skills"))
                updated["skills"] = normalized_skills
                updated["skill"] = normalized_skills[0] if normalized_skills else None

            if schedule_changed:
                updated_schedule = updated["schedule"]
                # The API may pass schedule as a raw string (e.g. "every 10m")
                # instead of a pre-parsed dict.  Normalize it the same way
                # create_job() does so downstream code can call .get() safely.
                if isinstance(updated_schedule, str):
                    updated_schedule = parse_schedule(updated_schedule)
                    updated["schedule"] = updated_schedule
                updated["schedule_display"] = updates.get(
                    "schedule_display",
                    updated_schedule.get("display", updated.get("schedule_display")),
                )
                if updated.get("state") != "paused":
                    updated_next_run = compute_next_run(updated_schedule)
                    # Same guard as create_job: an UPDATE that sets a one-shot
                    # to a time >ONESHOT_GRACE_SECONDS in the past would store
                    # next_run_at=None with state="scheduled", re-creating the
                    # ghost job that never fires (#59395). Reject it here too so
                    # the bug can't re-enter through the update door.
                    if (
                        updated_next_run is None
                        and updated_schedule.get("kind") == "once"
                    ):
                        run_at = updated_schedule.get("run_at") or updated_schedule
                        logger.warning(
                            "Rejecting one-shot cron job update '%s': run_at %s "
                            "is outside the %ss grace window",
                            updated.get("name", job_id),
                            run_at,
                            ONESHOT_GRACE_SECONDS,
                        )
                        raise ValueError(
                            f"Requested one-shot time {run_at} is more than "
                            f"{ONESHOT_GRACE_SECONDS}s in the past and cannot be scheduled."
                        )
                    updated["next_run_at"] = updated_next_run

            if inference_fields_changed:
                provider_snapshot, model_snapshot = _compute_provider_model_snapshots(
                    provider=updated.get("provider"),
                    model=updated.get("model"),
                    base_url=updated.get("base_url"),
                    no_agent=updated.get("no_agent"),
                )
                updated["provider_snapshot"] = provider_snapshot
                updated["model_snapshot"] = model_snapshot

            if updated.get("enabled", True) and updated.get("state") != "paused" and not updated.get("next_run_at"):
                next_run = compute_next_run(updated["schedule"])
                if next_run is None and updated["schedule"].get("kind") == "once":
                    run_at = updated["schedule"].get("run_at", "unknown")
                    raise ValueError(
                        f"Requested one-shot time {run_at} is in the past "
                        f"(grace window: {ONESHOT_GRACE_SECONDS}s) and cannot be scheduled."
                    )
                updated["next_run_at"] = next_run

            jobs[i] = updated
            save_jobs(jobs)
            return _normalize_job_record(jobs[i])
    return None


def pause_job(job_id: str, reason: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Pause a job without deleting it. Accepts a job ID or name."""
    job = resolve_job_ref(job_id)
    if not job:
        return None
    return update_job(
        job["id"],
        {
            "enabled": False,
            "state": "paused",
            "paused_at": _hermes_now().isoformat(),
            "paused_reason": reason,
        },
    )


def resume_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Resume a paused job and compute the next future run from now. Accepts a job ID or name."""
    job = resolve_job_ref(job_id)
    if not job:
        return None

    schedule = _schedule_dict(job)
    next_run_at = compute_next_run(schedule)
    if next_run_at is None and schedule.get("kind") == "once":
        run_at = schedule.get("run_at", "unknown")
        raise ValueError(
            f"Cannot resume: one-shot time {run_at} is in the past "
            f"(grace window: {ONESHOT_GRACE_SECONDS}s) and will never fire."
        )
    return update_job(
        job["id"],
        {
            "enabled": True,
            "state": "scheduled",
            "paused_at": None,
            "paused_reason": None,
            "next_run_at": next_run_at,
        },
    )


def trigger_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Schedule a job to run on the next scheduler tick. Accepts a job ID or name."""
    job = resolve_job_ref(job_id)
    if not job:
        return None
    return update_job(
        job["id"],
        {
            "enabled": True,
            "state": "scheduled",
            "paused_at": None,
            "paused_reason": None,
            "next_run_at": _hermes_now().isoformat(),
        },
    )


def remove_job(job_id: str) -> bool:
    """Remove a job by ID or name."""
    job = resolve_job_ref(job_id)
    if not job:
        return False
    canonical_id = job["id"]
    with _jobs_lock():
        jobs = load_jobs()
        original_len = len(jobs)
        jobs = [j for j in jobs if j["id"] != canonical_id]
        if len(jobs) < original_len:
            # Resolve the output dir BEFORE saving so a legacy unsafe ID (e.g.
            # left over from before the create-time guard) fails closed without
            # half-applying the removal.
            job_output_dir = _job_output_dir(canonical_id)
            save_jobs(jobs, removed_ids={canonical_id})
            # Clean up output directory to prevent orphaned dirs accumulating
            if job_output_dir.exists():
                shutil.rmtree(job_output_dir)
            # Clean up the job's durable notepad (cron/notepad.db) — without
            # this, removed jobs orphan their KV rows forever. Best effort:
            # a notepad failure must never block the removal itself.
            try:
                from cron.notepad import clear_notepad
                clear_notepad(canonical_id)
            except Exception:
                logger.debug(
                    "Failed to clear notepad for removed job %s",
                    canonical_id, exc_info=True,
                )
            return True
    return False


def _set_preflight_alerted(job_id: str, value: bool) -> bool:
    """Set/clear the preflight alert-dedup marker; return the PRIOR value.

    The marker records that the operator was already alerted about this
    job's blocked configuration, so the scheduler alerts exactly once and
    stays silent on subsequent ticks until the config heals (same
    alert-once shape as the dead-pin auto-pause in #73506). Persisted on
    the job record so the dedup survives gateway restarts.
    """
    with _jobs_lock():
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] == job_id:
                prior = bool(job.get("preflight_alerted"))
                if value:
                    job["preflight_alerted"] = True
                else:
                    job.pop("preflight_alerted", None)
                if prior != value:
                    jobs[i] = job
                    save_jobs(jobs)
                return prior
    return False


def mark_preflight_alerted(job_id: str) -> bool:
    """Mark the job as preflight-alerted; return True if it already was."""
    return _set_preflight_alerted(job_id, True)


def clear_preflight_alerted(job_id: str) -> None:
    """Clear the preflight alert-dedup marker (config validates again)."""
    _set_preflight_alerted(job_id, False)


def mark_job_run(
    job_id: str,
    success: bool,
    error: Optional[str] = None,
    delivery_error: Optional[str] = None,
    status: Optional[str] = None,
    *,
    execution_id: Optional[str] = None,
    require_cross_process_lock: bool = False,
) -> bool:
    """
    Mark a job as having been run.
    
    Updates last_run_at, last_status, increments completed count,
    computes next_run_at, and auto-deletes if repeat limit reached.

    ``delivery_error`` is tracked separately from the agent error — a job
    can succeed (agent produced output) but fail delivery (platform down).

    ``status`` overrides the derived ``last_status`` ("ok"/"error") with a
    specific terminal status for this run — e.g. ``"blocked_config"`` when
    the pre-dispatch configuration validation refused to run the agent
    (T1-26), so `cronjob list` distinguishes "your config is broken" from
    "the run itself failed".
    """
    with _jobs_lock() as cross_process_locked:
        if require_cross_process_lock and not cross_process_locked:
            logger.error(
                "mark_job_run for contextual execution %s refused because the "
                "jobs cross-process lock is unavailable",
                execution_id or job_id,
            )
            return False
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] == job_id:
                if execution_id:
                    markers = {
                        str(item)
                        for item in (job.get("_pending_accounting_execution_ids") or [])
                        if str(item)
                    }
                    if str(execution_id) in markers:
                        return True
                    markers.add(str(execution_id))
                    job["_pending_accounting_execution_ids"] = sorted(markers)
                now = _hermes_now().isoformat()
                job["last_run_at"] = now
                job["last_status"] = status or ("ok" if success else "error")
                job["last_error"] = error if not success else None
                # A healthy run means the configuration validates again — drop
                # the preflight alert-dedup marker so a FUTURE config break
                # re-alerts instead of being silently swallowed.
                if success:
                    job.pop("preflight_alerted", None)
                # Track delivery failures separately — cleared on successful delivery
                job["last_delivery_error"] = delivery_error
                if execution_id:
                    # The job file is the counter authority. Persist the
                    # occurrence marker in the same atomic save as the counters
                    # so restart recovery can replay accounting without double
                    # incrementing repeat/completion state.
                    job["_last_accounted_execution_id"] = str(execution_id)
                # Clear any external-fire claim so a re-armed recurring job can
                # be claimed again on its next fire (Phase 4C CAS).
                job["fire_claim"] = None
                # Clear the one-shot running-claim (#59229): the run is over, so
                # a re-armed recurring job or a re-dispatched one-shot recovery
                # is claimable again. No-op if the job never carried a claim.
                if job.get("run_claim") is not None:
                    job["run_claim"] = None
                # The execution-specific admission marker has served its purpose;
                # clear it in the same durable accounting write that re-anchors
                # the job so a later occurrence may claim the cursor.
                job.pop("_contextual_occurrence_claim", None)
                
                # Increment completed count.  Finite one-shot jobs are
                # pre-claimed by claim_dispatch() BEFORE the side effect runs
                # (issue #38758), which already incremented completed — do not
                # double-count them here.  Recurring jobs and direct callers
                # with no pre-run claim still get the legacy increment.
                if job.get("repeat"):
                    repeat = job["repeat"]
                    times = repeat.get("times")
                    completed = repeat.get("completed", 0)
                    kind = _schedule_dict(job).get("kind")
                    preclaimed_oneshot = (
                        kind == "once"
                        and times is not None
                        and times > 0
                        and completed > 0
                    )
                    if not preclaimed_oneshot:
                        completed += 1
                        repeat["completed"] = completed

                    # Check if we've hit the repeat limit
                    if times is not None and times > 0 and completed >= times:
                        # Limit reached: retain the record as a terminal
                        # completion instead of popping it. Deleting the job
                        # here discarded the last_status / last_error /
                        # last_delivery_error written above — a finished
                        # one-shot vanished from `cronjob list` with no
                        # inspectable outcome, and a failed delivery was
                        # invisible. Mirror the terminal shape of the
                        # next_run_at-is-None branch below; the retention
                        # sweep prunes these after
                        # COMPLETED_ONESHOT_RETENTION_DAYS.
                        job["enabled"] = False
                        job["state"] = "completed"
                        job["next_run_at"] = None
                        save_jobs(jobs)
                        return True
                
                # Compute next run
                job["next_run_at"] = compute_next_run(_schedule_dict(job), now)

                # If no next run, decide whether this is terminal completion
                # (one-shot) or a transient failure (recurring schedule couldn't
                # compute — e.g. 'croniter' missing from the runtime env).
                # Recurring jobs must NEVER be silently disabled: that turns a
                # missing runtime dep into "job completed" and the user's
                # schedule quietly goes off. See issue #16265.
                if job["next_run_at"] is None:
                    kind = _schedule_dict(job).get("kind")
                    if kind in {"cron", "interval"}:
                        job["state"] = "error"
                        if not job.get("last_error"):
                            job["last_error"] = (
                                "Failed to compute next run for recurring "
                                "schedule (is the 'croniter' package "
                                "installed in the gateway's Python env?)"
                            )
                        logger.error(
                            "Job '%s' (%s) could not compute next_run_at; "
                            "leaving enabled and marking state=error so the "
                            "job is not silently disabled.",
                            job.get("name", job.get("id", "?")),
                            kind,
                        )
                    else:
                        job["enabled"] = False
                        job["state"] = "completed"
                elif job.get("state") != "paused":
                    job["state"] = "scheduled"

                save_jobs(jobs)
                return True

        logger.warning("mark_job_run: job_id %s not found, skipping save", job_id)
        return False


def clear_job_accounting_marker(job_id: str, execution_id: str) -> bool:
    """Remove a crash-gap marker after the ledger confirms accounting."""
    with _jobs_lock() as cross_process_locked:
        if not cross_process_locked:
            logger.error(
                "Accounting marker cleanup for execution %s deferred because "
                "the jobs cross-process lock is unavailable",
                execution_id,
            )
            return False
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] != job_id:
                continue
            markers = [
                str(item)
                for item in (job.get("_pending_accounting_execution_ids") or [])
                if str(item) and str(item) != str(execution_id)
            ]
            if markers:
                job["_pending_accounting_execution_ids"] = sorted(set(markers))
            else:
                job.pop("_pending_accounting_execution_ids", None)
            jobs[i] = job
            save_jobs(jobs)
            return True
        return False


def _write_wedged_oneshot_diagnostic(job: Dict[str, Any]) -> None:
    """Leave an operator-visible trace when a wedged one-shot is removed.

    A finite one-shot whose dispatch was claimed (``repeat.completed`` >=
    ``repeat.times``) but which never reached ``mark_job_run`` (``last_run_at``
    is null) was interrupted mid-run — scheduler restart, gateway kill, or a
    non-Exception escape (#73973). The recovery guards remove such jobs so
    they stop appearing due, but a silent removal leaves the user with no
    output, no error, and no job record. Write a small diagnostic file into
    the job's output directory so the removal is observable and debuggable.

    Best-effort: diagnostics must never break the removal itself.
    """
    if job.get("last_run_at") is not None:
        return  # a prior run was recorded — normal completion race, not a wedge
    try:
        repeat = job.get("repeat") or {}
        claim = job.get("run_claim") or {}
        text = (
            "# Cron job removed without producing output\n\n"
            f"- job id: {job.get('id')}\n"
            f"- name: {job.get('name')}\n"
            f"- dispatch claimed: {repeat.get('completed', '?')}/{repeat.get('times', '?')}\n"
            f"- run claimed at: {claim.get('at', 'unknown')} by {claim.get('by', 'unknown')}\n"
            f"- removed at: {_hermes_now().isoformat()}\n\n"
            "This one-shot job's dispatch was claimed, but the run never "
            "completed (`last_run_at` was never written) — the scheduler "
            "process was most likely killed or restarted mid-execution. The "
            "job has been removed to stop it re-firing; recreate it to run "
            "again.\n"
        )
        save_job_output(job.get("id", ""), text)
        logger.warning(
            "Job '%s': removed without a completed run — diagnostic written to "
            "its output directory",
            job.get("name", job.get("id", "?")),
        )
    except Exception as e:
        logger.debug(
            "Failed to write wedged-oneshot diagnostic for job %r: %s",
            job.get("id"), e,
        )


def claim_dispatch(job_id: str) -> bool:
    """Atomically claim a finite one-shot job dispatch BEFORE execution.

    Increments ``repeat.completed`` under the cross-process jobs lock and
    persists the claim immediately, so that if the tick dies mid-execution
    (gateway kill, OOM, segfault, hard-timeout) the dispatch is not lost.
    This converts finite one-shot jobs from *at-least-once* to *at-most-times*
    semantics — a job that self-destructs fires at most ``repeat.times`` times
    instead of infinitely (issue #38758).

    Returns ``True`` if the caller may proceed to run the job, ``False`` if the
    dispatch limit is already reached (in which case the stale job is removed).

    Only claims jobs with ``schedule.kind == "once"`` and ``repeat.times > 0``.
    Recurring jobs (they use ``advance_next_run``) and infinite-repeat / no-repeat
    jobs are left unchanged and always allowed to proceed.
    """
    with _jobs_lock() as cross_process_locked:
        if not cross_process_locked:
            logger.error("claim_dispatch failed closed: jobs lock unavailable")
            return False
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] != job_id:
                continue
            if _schedule_dict(job).get("kind") != "once":
                return True  # recurring jobs use advance_next_run(), not dispatch claims
            repeat = job.get("repeat")
            if not repeat:
                return True  # no repeat limit — always dispatch
            times = repeat.get("times")
            if times is None or times <= 0:
                return True  # infinite — always dispatch
            completed = repeat.get("completed", 0)
            if completed >= times:
                # Already dispatched the max number of times.
                if job.get("last_run_at") is not None:
                    # A prior run completed normally (e.g. mark_job_run raced
                    # with this tick). Retain the terminal record — same shape
                    # as mark_job_run's repeat-limit branch — instead of
                    # deleting the job and its final status/delivery error.
                    job["enabled"] = False
                    job["state"] = "completed"
                    job["next_run_at"] = None
                    save_jobs(jobs)
                    logger.info(
                        "Job '%s': dispatch limit reached (%d/%d) — marking completed",
                        job.get("name", job.get("id", "?")),
                        completed,
                        times,
                    )
                    return False
                # A prior tick claimed the dispatch then died before the run
                # completed (#73973) — a genuinely wedged claim. Remove it so
                # it stops appearing as due, and leave an operator-visible
                # diagnostic instead of vanishing silently.
                jobs.pop(i)
                save_jobs(jobs, removed_ids={job_id})
                _write_wedged_oneshot_diagnostic(job)
                logger.info(
                    "Job '%s': dispatch limit reached (%d/%d) — removing",
                    job.get("name", job.get("id", "?")),
                    completed,
                    times,
                )
                return False
            # Claim this dispatch before the side effect runs.
            repeat["completed"] = completed + 1
            save_jobs(jobs)
            logger.debug(
                "Job '%s': claimed dispatch %d/%d",
                job.get("name", job.get("id", "?")),
                repeat["completed"],
                times,
            )
            return True

        logger.debug(
            "claim_dispatch: job_id %s not in store — proceeding without claim "
            "(handed-in job dict; nothing to persist a claim against)",
            job_id,
        )
        return True


def claim_contextual_occurrence(
    job_id: str,
    *,
    execution_id: str,
    expected_next_run_at: str,
) -> bool:
    """CAS one built-in contextual occurrence after its ledger row exists.

    The execution id is persisted with the jobs-file claim so a worker can
    prove it owns the exact occurrence it was handed. Recurring jobs advance
    their cursor here; one-shots acquire their durable in-flight ``run_claim``
    here. Finite repeat accounting remains the responsibility of
    :func:`claim_dispatch` immediately before the worker side effect.
    """
    occurrence_id = str(execution_id or "").strip()
    expected_cursor = str(expected_next_run_at or "").strip()
    if not occurrence_id or not expected_cursor:
        return False
    with _jobs_lock() as cross_process_locked:
        if not cross_process_locked:
            logger.error(
                "Contextual occurrence claim for job %s failed closed: jobs "
                "lock unavailable",
                job_id,
            )
            return False
        jobs = load_jobs()
        for job in jobs:
            if job.get("id") != job_id:
                continue
            if job.get("session_target", "isolated") != "current":
                return False
            existing = job.get("_contextual_occurrence_claim")
            if isinstance(existing, dict):
                return (
                    existing.get("execution_id") == occurrence_id
                    and existing.get("scheduled_for") == expected_cursor
                )
            if str(job.get("next_run_at") or "") != expected_cursor:
                return False
            schedule = _schedule_dict(job)
            kind = schedule.get("kind")
            claimed_at = _hermes_now().isoformat()
            if kind in {"cron", "interval"}:
                new_next = compute_next_run(schedule, claimed_at)
                if not new_next or new_next == expected_cursor:
                    return False
                job["next_run_at"] = new_next
            elif kind == "once":
                existing_run_claim = job.get("run_claim")
                if isinstance(existing_run_claim, dict):
                    return existing_run_claim.get("execution_id") == occurrence_id
                job["run_claim"] = {
                    "at": claimed_at,
                    "by": _machine_id(),
                    "durable": True,
                    "execution_id": occurrence_id,
                }
            else:
                return False
            job["_contextual_occurrence_claim"] = {
                "execution_id": occurrence_id,
                "scheduled_for": expected_cursor,
                "claimed_at": claimed_at,
            }
            save_jobs(jobs)
            return True
    return False


def contextual_occurrence_claim_state(
    job_id: str, *, execution_id: str
) -> Optional[bool]:
    """Return True/False for a definitive claim check, or None when unavailable."""
    occurrence_id = str(execution_id or "").strip()
    if not occurrence_id:
        return False
    with _jobs_lock() as cross_process_locked:
        if not cross_process_locked:
            return None
        for job in load_jobs():
            if job.get("id") != job_id:
                continue
            claim = job.get("_contextual_occurrence_claim")
            if not isinstance(claim, dict) or claim.get("execution_id") != occurrence_id:
                return False
            if _schedule_dict(job).get("kind") == "once":
                run_claim = job.get("run_claim")
                return bool(
                    isinstance(run_claim, dict)
                    and run_claim.get("durable") is True
                    and run_claim.get("execution_id") == occurrence_id
                )
            return True
        return False


def verify_contextual_occurrence_claim(job_id: str, *, execution_id: str) -> bool:
    """Return whether ``execution_id`` still owns the live contextual occurrence."""
    return contextual_occurrence_claim_state(
        job_id, execution_id=execution_id
    ) is True


def heartbeat_run_claim(job_id: str, *, expected_owner: str) -> bool:
    """Refresh a one-shot's ``run_claim`` timestamp while its run is alive.

    Called periodically from the scheduler's run monitor (#62002) so a
    legitimately long run keeps its claim fresh: an expired claim then really
    does mean "the claiming process died", and neither another process's tick
    nor this process's own next tick will re-dispatch or stale-remove the job
    while the run is in flight. mark_job_run() clears the claim on completion.

    ``expected_owner`` is the stable owner copied from the dispatched job. The
    compare-and-refresh prevents a stale runner that resumes after a long sleep
    from extending a claim another scheduler process has since taken over.

    Returns True if this owner's one-shot claim was refreshed; False when the
    job, claim, or ownership no longer matches.
    """
    with _jobs_lock() as cross_process_locked:
        if not cross_process_locked:
            return False
        jobs = load_jobs()
        for job in jobs:
            if job.get("id") != job_id:
                continue
            if _schedule_dict(job).get("kind") != "once":
                return False
            claim = job.get("run_claim")
            if not isinstance(claim, dict) or claim.get("by") != expected_owner:
                return False
            claim["at"] = _hermes_now().isoformat()
            save_jobs(jobs)
            return True
    return False


def advance_next_runs(
    job_ids, *, require_cross_process_lock: bool = False
) -> int:
    """Batch form of :func:`advance_next_run` for the due-dispatch loop.

    One ``load_jobs()`` + at most one ``save_jobs()`` for the whole due
    set, instead of one of each per job — the per-job form costs
    O(N loads + N saves) for N due jobs (~110 ms at N=50, measured), the
    batch form O(1 + 1) (~2 ms). ``job_ids`` may contain ids of one-shot
    or unknown jobs; they are skipped exactly as the per-job form skips
    them. Returns the number of jobs whose ``next_run_at`` was advanced.

    Crash semantics: the batch persists once at the end, so a crash
    mid-batch re-fires the whole set on restart (at-least-once burst)
    rather than advancing a prefix — acceptable given the sub-10ms window,
    and identical to the per-job form once the batch completes.
    """
    ids = set(job_ids)
    if not ids:
        return 0
    with _jobs_lock() as cross_process_locked:
        if require_cross_process_lock and not cross_process_locked:
            logger.error(
                "advance_next_runs failed closed: jobs lock unavailable"
            )
            return 0
        jobs = load_jobs()
        now = _hermes_now().isoformat()
        advanced = 0
        for job in jobs:
            if job["id"] not in ids:
                continue
            kind = job.get("schedule", {}).get("kind")
            if kind not in {"cron", "interval"}:
                continue
            new_next = compute_next_run(job["schedule"], now)
            if new_next and new_next != job.get("next_run_at"):
                job["next_run_at"] = new_next
                advanced += 1
        if advanced:
            save_jobs(jobs)
        return advanced


def advance_next_run(
    job_id: str, *, require_cross_process_lock: bool = False
) -> bool:
    """Preemptively advance next_run_at for a recurring job before execution.

    Call this BEFORE run_job() so that if the process crashes mid-execution,
    the job won't re-fire on the next gateway restart.  This converts the
    scheduler from at-least-once to at-most-once for recurring jobs — missing
    one run is far better than firing dozens of times in a crash loop.

    One-shot jobs are left unchanged so they can still retry on restart.

    Returns True if next_run_at was advanced, False otherwise.
    """
    # >= 1 (not == 1): a corrupted jobs file with duplicate ids advances
    # every matching record; the wrapper still reports the advance.
    return (
        advance_next_runs(
            [job_id], require_cross_process_lock=require_cross_process_lock
        )
        >= 1
    )


def _machine_id() -> str:
    """Stable-ish identifier for claim attribution/debugging (NOT correctness).

    Uses ``HERMES_MACHINE_ID`` if set, else hostname + pid. The CAS correctness
    comes from the file lock + the fresh-claim check, not from this value.
    """
    explicit = os.getenv("HERMES_MACHINE_ID", "").strip()
    if explicit:
        return explicit
    try:
        import socket
        host = socket.gethostname()
    except Exception:
        host = "unknown"
    return f"{host}:{os.getpid()}"


def claim_job_for_fire(job_id: str, *, claim_ttl_seconds: int = 300) -> bool:
    """Atomically claim a job for one existing external-provider fire.

    Contextual jobs are rejected at creation whenever an external provider is
    configured; this function therefore preserves the isolated-cron behavior
    that predates contextual V1.
    """
    with _jobs_lock() as cross_process_locked:
        if not cross_process_locked:
            logger.error("claim_job_for_fire failed closed: jobs lock unavailable")
            return False
        jobs = load_jobs()
        for job in jobs:
            if job["id"] != job_id:
                continue
            # enabled + pause markers must both clear — a half-paused record
            # (enabled=true, state=paused/paused_at set) must not claim.
            if not is_job_runnable(job):
                return False
            now = _hermes_now()
            existing = job.get("fire_claim")
            if existing:
                try:
                    claimed_at = _ensure_aware(datetime.fromisoformat(existing["at"]))
                    age = (now - claimed_at).total_seconds()
                    if 0 <= age < claim_ttl_seconds:
                        return False
                except Exception:
                    pass
            job["fire_claim"] = {"at": now.isoformat(), "by": _machine_id()}
            schedule = _schedule_dict(job)
            if schedule.get("kind") in {"cron", "interval"}:
                nxt = compute_next_run(schedule, now.isoformat())
                if nxt:
                    job["next_run_at"] = nxt
            save_jobs(jobs)
            return True
        return False


# Completed one-shot job records are retained in jobs.json (final status +
# delivery error stay inspectable via `cronjob list`) instead of being deleted
# at completion, then pruned by _sweep_completed_oneshots once they age out.
COMPLETED_ONESHOT_RETENTION_DAYS = 7


def _completed_oneshot_retention_days() -> float:
    """Resolve the completed one-shot retention window from config.

    ``cron.completed_retention_days`` (number, default
    ``COMPLETED_ONESHOT_RETENTION_DAYS``). A non-positive value disables the
    sweep, retaining completed one-shot records indefinitely.
    """
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        return float(
            cron_cfg.get(
                "completed_retention_days", COMPLETED_ONESHOT_RETENTION_DAYS
            )
        )
    except Exception:
        return float(COMPLETED_ONESHOT_RETENTION_DAYS)


def _sweep_completed_oneshots(
    raw_jobs: List[Dict[str, Any]],
    now: datetime,
    *,
    removed_ids: Optional[Set[str]] = None,
) -> bool:
    """Prune terminal ``state == "completed"`` one-shot records past retention.

    Mutates *raw_jobs* in place; returns True when anything was removed (the
    caller persists). Ids removed are added to *removed_ids* when provided so
    ``save_jobs``'s shrink-merge guard (#80624) allows the intentional delete.
    Only one-shot (``schedule.kind == "once"``) records in the terminal
    completed state are candidates; recurring jobs and non-terminal one-shots
    are never touched. Age is measured from ``last_run_at`` — a completed
    record without a parseable ``last_run_at`` is kept (never guess a record
    into deletion).
    """
    retention_days = _completed_oneshot_retention_days()
    if retention_days <= 0:
        return False
    cutoff = now - timedelta(days=retention_days)
    removed = False
    for rj in list(raw_jobs):
        try:
            if rj.get("state") != "completed":
                continue
            schedule = rj.get("schedule")
            kind = schedule.get("kind") if isinstance(schedule, dict) else None
            if kind != "once":
                continue
            last_run = rj.get("last_run_at")
            if not isinstance(last_run, str):
                continue
            try:
                last_run_dt = _ensure_aware(datetime.fromisoformat(last_run))
            except Exception:
                continue
            if last_run_dt >= cutoff:
                continue
            raw_jobs.remove(rj)
            removed = True
            rid = rj.get("id")
            if removed_ids is not None and rid:
                removed_ids.add(str(rid))
            logger.info(
                "Job '%s': pruning completed one-shot record "
                "(finished %s, retention %.1f days)",
                rj.get("name", rj.get("id", "?")),
                last_run,
                retention_days,
            )
        except Exception:
            logger.debug(
                "Retention sweep skipped malformed job record %r",
                rj.get("id", "?"),
                exc_info=True,
            )
    return removed


def get_due_jobs() -> List[Dict[str, Any]]:
    """Get all jobs that are due to run now.

    For recurring jobs (cron/interval), if the scheduled time is stale (more
    than one period in the past, e.g. because the gateway was down OR because a
    long-running previous execution overran the interval), the accumulated
    missed runs are collapsed — ``next_run_at`` is fast-forwarded to the next
    future occurrence so a backlog does NOT burst-fire on restart — but the job
    still fires ONCE now. This prevents the perpetual-defer loop (#33315) where
    a job whose runtime exceeds ``interval + grace`` would be skipped forever.

    Note: firing once on catch-up flows through ``mark_job_run``, so a job with
    a ``repeat.times`` limit consumes one of its runs on that catch-up fire.
    """
    with _jobs_lock() as cross_process_locked:
        if not cross_process_locked:
            # jobs.json is a whole-file authority.  A degraded ordinary writer
            # could otherwise rewrite stale contextual state even when it does
            # not touch that job directly, so any store containing contextual
            # records becomes read-only until cross-process exclusion recovers.
            try:
                contains_contextual = any(
                    isinstance(job, dict)
                    and job.get("session_target", "isolated") == "current"
                    for job in load_jobs()
                )
            except Exception:
                # Failure to inspect a potentially mixed store cannot authorize
                # a whole-file rewrite without the advisory lock.
                logger.exception(
                    "Cron due scan skipped because degraded lock mode could not "
                    "prove the store contains only isolated jobs"
                )
                return []
            if contains_contextual:
                logger.error(
                    "Cron due scan skipped because jobs cross-process locking is "
                    "unavailable for a store containing contextual jobs"
                )
                return []
        return _get_due_jobs_locked()


def _get_due_jobs_locked() -> List[Dict[str, Any]]:
    """Inner implementation of get_due_jobs(); must be called with _jobs_lock held."""
    now = _hermes_now()
    raw_jobs = load_jobs()
    needs_save = False
    intentionally_removed: Set[str] = set()

    # Repair id-less records BEFORE anything keys off ``job["id"]``. A direct
    # jobs.json edit that bypassed add_job() can leave a record without an "id"
    # (older writers used "job_id"). Every downstream site — the logging
    # helpers and the ``for rj in raw_jobs: if rj["id"] == job["id"]``
    # persistence loops — indexes job["id"] eagerly, so a single malformed
    # record raised KeyError mid-tick, aborting the whole scan before
    # save_jobs() ran. That froze the entire profile's scheduler in a
    # per-minute fast-forward loop (healthy jobs recomputed in memory, then
    # discarded when the exception unwound). Recover the id from the drifted
    # "job_id" key when present, else synthesize one, and persist.
    for rj in raw_jobs:
        if not rj.get("id"):
            rj["id"] = rj.pop("job_id", None) or uuid.uuid4().hex[:12]
            needs_save = True

    jobs = [_apply_skill_fields(j) for j in copy.deepcopy(raw_jobs)]
    due = []

    # Normalize schedules for execution without destroying supported legacy
    # string schedules. Valid strings are parsed only in the working copy; the
    # persisted public shape remains byte-for-byte compatible.
    for j in jobs:
        value = j.get("schedule")
        if isinstance(value, dict):
            continue
        parsed = _schedule_dict(j)
        if parsed:
            j["schedule"] = parsed
        else:
            j["schedule"] = {}
            needs_save = True
    for rj in raw_jobs:
        value = rj.get("schedule")
        if isinstance(value, dict):
            continue
        if isinstance(value, str) and _schedule_dict(rj):
            continue
        rj["schedule"] = {}
        needs_save = True

    # Normalize malformed "next_run_at" records (direct jobs.json edit,
    # corruption, migration, or buggy writer). If present but not a valid
    # ISO string, datetime.fromisoformat(next_run) later raises and aborts
    # the entire scan *before* save_jobs(). Healthy siblings then lose any
    # fast-forwarded next_run_at (same class of bug as bad "id" or "schedule").
    # Strip the bad value so the existing "no next_run_at" recovery path
    # recomputes a sane value and persists it for this job.
    for j in jobs:
        nr = j.get("next_run_at")
        if nr is not None:
            if not isinstance(nr, str):
                j.pop("next_run_at", None)
                needs_save = True
            else:
                try:
                    datetime.fromisoformat(nr)
                except Exception:
                    j.pop("next_run_at", None)
                    needs_save = True
    for rj in raw_jobs:
        nr = rj.get("next_run_at")
        if nr is not None:
            if not isinstance(nr, str):
                rj.pop("next_run_at", None)
                needs_save = True
            else:
                try:
                    datetime.fromisoformat(nr)
                except Exception:
                    rj.pop("next_run_at", None)
                    needs_save = True

    # Same treatment for last_run_at (used as base in recovery / compute_next_run).
    for j in jobs:
        lr = j.get("last_run_at")
        if lr is not None and not isinstance(lr, str):
            j.pop("last_run_at", None)
            needs_save = True
        elif isinstance(lr, str):
            try:
                datetime.fromisoformat(lr)
            except Exception:
                j.pop("last_run_at", None)
                needs_save = True
    for rj in raw_jobs:
        lr = rj.get("last_run_at")
        if lr is not None and not isinstance(lr, str):
            rj.pop("last_run_at", None)
            needs_save = True
        elif isinstance(lr, str):
            try:
                datetime.fromisoformat(lr)
            except Exception:
                rj.pop("last_run_at", None)
                needs_save = True

    # Resolve the one-shot running-claim stale-recovery TTL once per scan
    # (derived from HERMES_CRON_TIMEOUT). See _oneshot_run_claim_ttl_seconds.
    _run_claim_ttl = _oneshot_run_claim_ttl_seconds()

    # Retention sweep: completed one-shots are retained (so their final
    # status / delivery error stay inspectable via `cronjob list`) instead of
    # being deleted on completion, but they must not accumulate in jobs.json
    # forever. Prune terminal one-shot records older than the retention
    # window each scan.
    if _sweep_completed_oneshots(raw_jobs, now, removed_ids=intentionally_removed):
        needs_save = True
        jobs = [j for j in jobs if any(rj.get("id") == j.get("id") for rj in raw_jobs)]

    for job in jobs:
        # Per-job containment (structural guard): one malformed or
        # unexpected job record must never abort the whole scan. The id /
        # schedule / timestamp normalizations above repair the known shapes;
        # this guard catches every FUTURE variant, degrading to "skip this
        # job this tick" so healthy siblings still run and their recovered
        # state still reaches save_jobs() below.
        try:
            if not job.get("enabled", True):
                continue

            # Contradiction self-heal: enabled=true with pause markers means the
            # operator believes the job is frozen while the scheduler would still
            # fire it (07-30 outage). Refuse to run and force enabled=false so
            # the next list/report is honest. Log loudly — this should be rare
            # after pause_job sets both fields atomically.
            if _has_pause_marker(job):
                jid = job.get("id")
                logger.error(
                    "Job '%s' (%s) has pause markers while enabled=true; "
                    "self-disabling so it cannot fire (pause must be authoritative).",
                    job.get("name", jid),
                    jid,
                )
                for rj in raw_jobs:
                    if rj.get("id") != jid:
                        continue
                    rj["enabled"] = False
                    rj["state"] = "paused"
                    if not rj.get("paused_at"):
                        rj["paused_at"] = now.isoformat()
                    if not rj.get("paused_reason"):
                        rj["paused_reason"] = (
                            "auto-disabled: enabled+paused contradiction"
                        )
                    needs_save = True
                    break
                continue

            # Cross-process running-claim guard (#59229): if another scheduler
            # process already claimed this one-shot and its run is still in flight
            # (claim younger than the TTL), skip it — do NOT re-dispatch. The
            # claim is stamped just before we return the job as due (below) and
            # cleared by mark_job_run() on completion. A claim older than the TTL
            # is treated as stale (the claiming tick died mid-run) and allowed
            # through so the job is recovered rather than wedged forever.
            existing_claim = job.get("run_claim")
            if existing_claim and _schedule_dict(job).get("kind") == "once":
                # Contextual one-shots are conservative: once dispatched, an
                # unacknowledged run is never retried merely because a TTL
                # elapsed. The execution ledger will classify a dead owner as
                # unknown; an operator can then resolve it without risking a
                # duplicate hidden turn or external notification.
                if existing_claim.get("durable") is True:
                    continue
                try:
                    claimed_at = _ensure_aware(
                        datetime.fromisoformat(existing_claim["at"])
                    )
                    # 0 <= age: a future-dated claim (clock/TZ skew across a
                    # restart) must be treated as stale, not eternally fresh,
                    # or the one-shot is skipped forever (#60703).
                    _age = (now - claimed_at).total_seconds()
                    if 0 <= _age < _run_claim_ttl:
                        continue  # a fresh claim is held by an in-flight run
                except (KeyError, ValueError, TypeError):
                    pass  # malformed claim → fall through and (re)claim

            next_run = job.get("next_run_at")
            if not next_run:
                schedule = job.get("schedule", {})
                kind = schedule.get("kind")

                # One-shot jobs use a small grace window via the dedicated helper.
                recovered_next = _recoverable_oneshot_run_at(
                    schedule,
                    now,
                    last_run_at=job.get("last_run_at"),
                )
                recovery_kind = "one-shot" if recovered_next else None

                # Recurring jobs reach here only when something — typically a
                # direct jobs.json edit that bypassed add_job() — left
                # next_run_at unset.  Without this branch, such jobs are
                # silently skipped forever; recompute next_run_at from the
                # schedule so they pick up at their next scheduled tick.
                if not recovered_next and kind in {"cron", "interval"}:
                    recovered_next = compute_next_run(schedule, now.isoformat())
                    if recovered_next:
                        recovery_kind = kind

                if not recovered_next:
                    continue

                job["next_run_at"] = recovered_next
                next_run = recovered_next
                logger.info(
                    "Job '%s' had no next_run_at; recovering %s run at %s",
                    job.get("name", job.get("id", "?")),
                    recovery_kind,
                    recovered_next,
                )
                for rj in raw_jobs:
                    if rj["id"] == job["id"]:
                        rj["next_run_at"] = recovered_next
                        needs_save = True
                        break

            raw_next_run_dt = datetime.fromisoformat(next_run)
            schedule = job.get("schedule", {})
            kind = schedule.get("kind")

            next_run_dt = _ensure_aware(raw_next_run_dt)
            # Migration repair: a cron job persists next_run_at as an absolute
            # instant, but the cron expr describes local wall-clock intent. If the
            # configured/system timezone changed after persistence, the stored
            # instant's offset no longer matches now's, and its converted time can
            # look due hours early (21:00+10 -> 13:00+02). When the stored *wall
            # clock* is still in the future, recompute from the schedule so we fire
            # at the intended local time instead of early-then-again.
            #
            # TRADE-OFF: this cannot distinguish a config/host TZ migration from a
            # legitimate DST offset change. A DST boundary that satisfies all four
            # conditions will recompute (and thus SKIP the pending occurrence, no
            # catch-up) rather than fire it. Accepted: in the pure-migration case
            # the recompute lands on the same wall-clock time later the same period,
            # and DST-boundary collisions with a still-future stored wall clock are
            # rare relative to the double-fire bug this prevents (#28934).
            if (
                kind == "cron"
                and next_run_dt <= now
                and _timezone_offset_mismatch(raw_next_run_dt, now)
                and _stored_wall_clock_is_future(raw_next_run_dt, now)
            ):
                new_next = compute_next_run(schedule, now.isoformat())
                if new_next:
                    logger.info(
                        "Job '%s' next_run_at offset changed (%s -> %s). "
                        "Recomputing cron run to preserve local wall-clock intent: %s",
                        job.get("name", job.get("id", "?")),
                        raw_next_run_dt.utcoffset(),
                        now.utcoffset(),
                        new_next,
                    )
                    for rj in raw_jobs:
                        if rj["id"] == job["id"]:
                            rj["next_run_at"] = new_next
                            needs_save = True
                            break
                    continue

            if next_run_dt <= now:

                # For recurring jobs, check if the scheduled time is stale
                # (gateway was down and missed the window). Fast-forward to
                # the next future occurrence instead of firing a stale run.
                grace = _compute_grace_seconds(schedule)
                if kind in {"cron", "interval"} and (now - next_run_dt).total_seconds() > grace:
                    # Job is past its catch-up grace window — skip accumulated
                    # missed runs but still execute once now to avoid deferring
                    # indefinitely (e.g. a long-running job just finished).
                    new_next = compute_next_run(schedule, now.isoformat())
                    if new_next:
                        logger.info(
                            "Job '%s' missed its scheduled time (%s, grace=%ds). "
                            "Running now; next run provisionally set to: %s "
                            "(re-anchored on completion)",
                            job.get("name", job.get("id", "?")),
                            next_run,
                            grace,
                            new_next,
                        )
                        # Persist the fast-forward to storage now (skip accumulated
                        # slots). In the built-in ticker path this is shortly
                        # overwritten by advance_next_run + mark_job_run, but it is
                        # NOT redundant: it (a) protects the crash window between
                        # here and mark_job_run, and (b) covers the external
                        # fire_due provider path, which does not call
                        # advance_next_run. mark_job_run re-anchors next_run_at off
                        # the actual completion time, so this value is provisional.
                        for rj in raw_jobs:
                            if rj["id"] == job["id"]:
                                rj["next_run_at"] = new_next
                                needs_save = True
                                break
                        record_catch_up_occurrence()
                        # Fall through to due.append(job) — execute once now

                # One-shot dispatch-limit guard (issue #38758): a finite one-shot
                # claimed via claim_dispatch() but whose tick died before
                # mark_job_run could remove it will have completed >= times while
                # still looking due (last_run_at was never written, so the
                # recovery helper re-armed it). Remove it instead of re-firing.
                if kind == "once":
                    repeat = job.get("repeat")
                    if repeat:
                        times = repeat.get("times")
                        completed = repeat.get("completed", 0)
                        if times is not None and times > 0 and completed >= times:
                            # A live run must never have its job record deleted
                            # underneath it (#62002): a run that outlives the
                            # run_claim TTL (stream stall, laptop asleep
                            # mid-run) satisfies the same completed >= times +
                            # expired-claim condition as a dead tick, but
                            # mark_job_run() still needs the record to land
                            # last_run_at / last_status / last_delivery_error.
                            # If this process is still running the job, it is
                            # slow, not stale — keep the entry and skip.
                            if _job_running_in_this_process(job.get("id", "")):
                                logger.info(
                                    "Job '%s': dispatch limit reached (%d/%d) "
                                    "but its run is still in flight in this "
                                    "process — keeping entry",
                                    job.get("name", job.get("id", "?")),
                                    completed,
                                    times,
                                )
                                continue
                            logger.info(
                                "Job '%s': one-shot dispatch limit reached (%d/%d) "
                                "— removing stale due entry",
                                job.get("name", job.get("id", "?")),
                                completed,
                                times,
                            )
                            for rj in raw_jobs:
                                if rj["id"] == job["id"]:
                                    raw_jobs.remove(rj)
                                    intentionally_removed.add(str(rj["id"]))
                                    needs_save = True
                                    break
                            # The claimed run never completed here by
                            # definition (last_run_at unwritten is what made
                            # the entry look due) — leave an operator-visible
                            # diagnostic instead of vanishing silently (#73973).
                            _write_wedged_oneshot_diagnostic(job)
                            continue

                # Durably claim a one-shot for the DURATION of its run before
                # returning it as due, so a second scheduler process (gateway +
                # desktop both run in-process 60s tickers on one HERMES_HOME)
                # cannot re-dispatch it while the first run is still in flight
                # (#59229). A plain one-shot's due-state is not resolved until
                # mark_job_run() completes it minutes later, so advancing
                # next_run_at by a fixed window is not enough — a job that outlives
                # one tick (e.g. a 2.5-min research prompt) would simply re-fire on
                # the next tick after the window. Instead we stamp a run_claim under
                # the same lock get_due_jobs already holds; the other process reads
                # a fresh claim on its next tick and skips (handled at the top of
                # this loop). mark_job_run() clears the claim on completion. The TTL
                # is only a safety valve: a claiming tick that DIES mid-run leaves a
                # stale claim that expires after the resolved run-claim TTL
                # (_oneshot_run_claim_ttl_seconds, derived from HERMES_CRON_TIMEOUT),
                # so the job is re-dispatched rather than wedged forever.
                if (
                    kind == "once"
                    and job.get("session_target", "isolated") != "current"
                ):
                    claim = {
                        "at": now.isoformat(),
                        "by": _machine_id(),
                        "durable": job.get("session_target") == "current",
                    }
                    job["run_claim"] = claim
                    for rj in raw_jobs:
                        if rj["id"] == job["id"]:
                            rj["run_claim"] = claim
                            needs_save = True
                            break

                due.append(job)
        except Exception:
            logger.exception(
                "Skipping malformed cron job %r during due scan",
                job.get("name") or job.get("id") or "?",
            )
            continue

    if needs_save:
        save_jobs(raw_jobs, removed_ids=intentionally_removed or None)

    return due


# Per-run cron output (`cron/output/<job>/<timestamp>.md`) is written once per
# execution. Unlike the quick-snapshot store (`hermes_cli.backup`, capped at 20)
# it had no retention, so a frequently-scheduled job on a long-running deploy
# accumulated one file per run forever and could fill the disk (#52383). Keep the
# most recent N files per job; a non-positive value disables pruning (opt-out).
_CRON_OUTPUT_DEFAULT_KEEP = 50


def _cron_output_keep() -> int:
    """Resolve the per-job output-file retention cap from config (``cron.output_retention``)."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        return int(cron_cfg.get("output_retention", _CRON_OUTPUT_DEFAULT_KEEP))
    except Exception:
        return _CRON_OUTPUT_DEFAULT_KEEP


def _prune_job_output(job_output_dir: Path, keep: int) -> int:
    """Remove the oldest ``*.md`` run-output files beyond *keep*. Returns count deleted.

    Mirrors the quick-snapshot retention in ``hermes_cli.backup._prune_quick_snapshots``:
    output filenames are timestamp-based (``%Y-%m-%d_%H-%M-%S.md``) so a reverse
    lexical sort orders newest-first, and everything past *keep* is the tail to
    drop. A non-positive *keep* disables pruning. Pruning failures are swallowed
    so they can never break output saving.
    """
    if keep <= 0:
        return 0
    try:
        files = sorted(
            (f for f in job_output_dir.glob("*.md") if f.is_file()),
            key=lambda f: f.name,
            reverse=True,
        )
    except OSError:
        return 0
    deleted = 0
    for stale in files[keep:]:
        try:
            stale.unlink()
            deleted += 1
        except OSError as exc:
            logger.debug("Failed to prune cron output %s: %s", stale.name, exc)
    return deleted


def save_job_output(job_id: str, output: str):
    """Save job output to file."""
    ensure_dirs()
    job_output_dir = _job_output_dir(job_id)
    job_output_dir.mkdir(parents=True, exist_ok=True)
    _secure_dir(job_output_dir)

    timestamp = _hermes_now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = job_output_dir / f"{timestamp}.md"

    fd, tmp_path = tempfile.mkstemp(dir=str(job_output_dir), suffix='.tmp', prefix='.output_')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(output)
            f.flush()
            os.fsync(f.fileno())
        atomic_replace(tmp_path, output_file)
        _secure_file(output_file)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Bound per-job output growth so long-running deploys don't fill the disk (#52383).
    _prune_job_output(job_output_dir, _cron_output_keep())

    return output_file


# =============================================================================
# Skill reference rewriting (curator integration)
# =============================================================================

def _canonical_skill_ref(raw: Any) -> str:
    """Reduce one job skill reference to the bare name the curator matches on.

    A job may store an absolute path under ``HERMES_HOME/skills`` or an
    external skills dir; the scheduler resolves those through
    ``normalize_skill_lookup_name`` before handing them to ``skill_view``.
    The curator compares this set against bare skill names, so it has to
    resolve them the same way — otherwise a path-referencing job's skill
    looks unreferenced and gets archived out from under it.

    Best-effort: if the resolver is unavailable or rejects the value, fall
    back to the plain cleanup so a broken import can never lose a name.
    """
    value = str(raw or "").strip()
    if not value:
        return ""
    try:
        from agent.skill_utils import normalize_skill_lookup_name
        value = normalize_skill_lookup_name(value) or value
    except Exception:
        logger.debug(
            "referenced_skill_names: could not normalize skill ref %r", raw,
            exc_info=True,
        )
    return value.strip().lstrip("/")


def referenced_skill_names() -> Set[str]:
    """Return the set of skill names referenced by ANY cron job.

    Includes paused and disabled jobs deliberately: a paused job never
    fires, so its skills never get a ``bump_use`` from the scheduler, yet
    resuming it must still find its skills present. The curator uses this
    set to protect referenced skills from inactivity archival — a skill a
    live job depends on is "in use" regardless of when it was last loaded.

    Names are canonicalized the way the scheduler resolves them at load
    time, so a job that stores an absolute skill path is protected too.

    Best-effort: a corrupt/unreadable jobs store returns an empty set
    rather than raising, so a cron issue can never break the curator.
    """
    try:
        jobs = load_jobs()
    except Exception:
        logger.debug("referenced_skill_names: failed to load cron jobs", exc_info=True)
        return set()

    names: Set[str] = set()
    for job in jobs:
        if not isinstance(job, dict):
            continue
        for name in _normalize_skill_list(job.get("skill"), job.get("skills")):
            cleaned = _canonical_skill_ref(name)
            if cleaned:
                names.add(cleaned)
    return names


def rewrite_skill_refs(
    consolidated: Optional[Dict[str, str]] = None,
    pruned: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Rewrite cron job skill references after a curator consolidation pass.

    When the curator consolidates a skill X into umbrella Y (or archives X
    as pruned), any cron job that lists ``X`` in its ``skills`` field will
    fail to load ``X`` at run time — the scheduler logs a warning and
    skips the skill, so the job runs without the instructions it was
    scheduled to follow. See cron/scheduler.py where ``skill_view`` is
    called per skill name.

    This function repairs cron jobs in-place:

    - A skill listed in ``consolidated`` is replaced with its umbrella
      target (the ``into`` value). If the umbrella is already in the
      job's skill list, the stale name is dropped without duplication.
    - A skill listed in ``pruned`` is dropped outright — there is no
      forwarding target.
    - Ordering and other skills in the list are preserved.
    - The legacy ``skill`` field is realigned via ``_apply_skill_fields``.

    Args:
        consolidated: mapping of ``old_skill_name -> umbrella_skill_name``.
        pruned: list of skill names that were archived with no forwarding
            target.

    Returns a report dict::

        {
            "rewrites": [
                {
                    "job_id": ...,
                    "job_name": ...,
                    "before": [...],
                    "after": [...],
                    "mapped": {"old": "new", ...},
                    "dropped": ["old", ...],
                },
                ...
            ],
            "jobs_updated": N,
            "jobs_scanned": M,
        }

    Best-effort: exceptions from loading/saving propagate to the caller so
    tests can assert behaviour; the curator invocation site wraps this
    call in a try/except so a failure here never breaks the curator.
    """
    consolidated = dict(consolidated or {})
    pruned_set = set(pruned or [])
    # A skill listed in both wins as "consolidated" — it has a target,
    # which is the more useful of the two outcomes.
    pruned_set -= set(consolidated.keys())

    if not consolidated and not pruned_set:
        return {"rewrites": [], "jobs_updated": 0, "jobs_scanned": 0}

    with _jobs_lock():
        jobs = load_jobs()
        rewrites: List[Dict[str, Any]] = []
        changed = False

        for job in jobs:
            skills_before = _normalize_skill_list(job.get("skill"), job.get("skills"))
            if not skills_before:
                continue

            mapped: Dict[str, str] = {}
            dropped: List[str] = []
            new_skills: List[str] = []

            for name in skills_before:
                if name in consolidated:
                    target = consolidated[name]
                    mapped[name] = target
                    if target and target not in new_skills:
                        new_skills.append(target)
                elif name in pruned_set:
                    dropped.append(name)
                elif name not in new_skills:
                    new_skills.append(name)

            if not mapped and not dropped:
                continue

            job["skills"] = new_skills
            job["skill"] = new_skills[0] if new_skills else None
            changed = True

            rewrites.append({
                "job_id": job.get("id"),
                "job_name": job.get("name") or job.get("id"),
                "before": list(skills_before),
                "after": list(new_skills),
                "mapped": mapped,
                "dropped": dropped,
            })

        if changed:
            save_jobs(jobs)
            logger.info(
                "Curator rewrote skill references in %d cron job(s)", len(rewrites)
            )

        return {
            "rewrites": rewrites,
            "jobs_updated": len(rewrites),
            "jobs_scanned": len(jobs),
        }
