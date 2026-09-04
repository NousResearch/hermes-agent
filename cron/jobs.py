"""
Cron job storage and management.

Jobs are stored in ~/.hermes/cron/jobs.json
Output is saved to ~/.hermes/cron/output/{job_id}/{timestamp}.md
"""

import contextlib
import copy
from contextvars import ContextVar
from dataclasses import dataclass
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
_fire_fence_locks: Dict[str, threading.RLock] = {}
_fire_fence_locks_guard = threading.Lock()
_fire_fence_lock_state = threading.local()

# Upper bound on waiting for the cross-process .jobs.lock flock (#60703).
# Every cron function in the process funnels through _jobs_lock(), and the
# flock is taken while holding the process-wide RLock — so an unbounded wait
# on a lock held by a wedged sibling process silently freezes the ticker
# heartbeat and every job forever.  30s is orders of magnitude above any
# legitimate critical section (field updates only) while keeping the ticker's
# worst-case stall well under one status-alarm threshold.
_JOBS_LOCK_TIMEOUT_SECONDS = 30.0
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
            yield
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
            except (OSError, IOError) as e:
                # Never let a locking failure take down cron writes — fall back to
                # in-process-only protection (still held via _jobs_file_lock).
                logger.warning("jobs.json cross-process lock unavailable (%s); "
                               "proceeding with in-process lock only", e)
            try:
                yield
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


@contextlib.contextmanager
def _fire_job_lock(job_id: str):
    """Serialize one job's owner mutations and external side effects.

    Unlike the global jobs lock, this lock may be held across network delivery.
    It is scoped to one profile + job, so unrelated cron jobs keep progressing.
    Fencing fails closed when cross-process locking is unavailable.
    """
    cron_dir = _current_cron_store().cron_dir
    lock_key = f"{cron_dir.resolve()}::{job_id}"
    with _fire_fence_locks_guard:
        local_lock = _fire_fence_locks.setdefault(lock_key, threading.RLock())

    if not local_lock.acquire(timeout=_JOBS_LOCK_TIMEOUT_SECONDS):
        logger.error("Timed out waiting for local fire fence %s; failing closed", lock_key)
        yield False
        return

    held_locks = getattr(_fire_fence_lock_state, "held", None)
    if held_locks is None:
        held_locks = {}
        _fire_fence_lock_state.held = held_locks
    if lock_key in held_locks:
        try:
            yield held_locks[lock_key]
        finally:
            local_lock.release()
        return

    try:
        ensure_dirs()
        lock_name = uuid.uuid5(uuid.NAMESPACE_URL, lock_key).hex
        lock_path = cron_dir / f".fire-{lock_name}.lock"
        lock_fd = None
        acquired = False
        try:
            lock_fd = open(lock_path, "a+", encoding="utf-8")
            lock_fd.seek(0)
            if fcntl is not None:
                deadline = time.monotonic() + _JOBS_LOCK_TIMEOUT_SECONDS
                while True:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        acquired = True
                        break
                    except (OSError, IOError):
                        if time.monotonic() >= deadline:
                            logger.error(
                                "Timed out waiting for fire fence %s; failing closed",
                                lock_path,
                            )
                            break
                        time.sleep(0.1)
            elif msvcrt is not None:
                getattr(msvcrt, "locking")(
                    lock_fd.fileno(), getattr(msvcrt, "LK_LOCK"), 1
                )
                acquired = True
            else:  # pragma: no cover - supported platforms provide one backend
                logger.error("No cross-process lock backend for cron fire fence")
        except (OSError, IOError) as exc:
            logger.error("Cron fire fence unavailable for %s: %s", job_id, exc)

        held_locks[lock_key] = acquired
        try:
            yield acquired
        finally:
            held_locks.pop(lock_key, None)
            if lock_fd is not None:
                try:
                    if acquired and fcntl is not None:
                        fcntl.flock(lock_fd, fcntl.LOCK_UN)
                    elif acquired and msvcrt is not None:
                        getattr(msvcrt, "locking")(
                            lock_fd.fileno(), getattr(msvcrt, "LK_UNLCK"), 1
                        )
                except (OSError, IOError):
                    pass
                finally:
                    lock_fd.close()
    finally:
        local_lock.release()


@contextlib.contextmanager
def fire_claim_fence(job_id: str, *, expected_owner: str):
    """Hold a per-job fence while an owner performs an external side effect."""
    with _fire_job_lock(job_id) as acquired:
        if not acquired:
            yield False
            return
        with _jobs_lock():
            job = next((item for item in load_jobs() if item.get("id") == job_id), None)
            claim = job.get("fire_claim") if isinstance(job, dict) else None
            owns_claim = (
                isinstance(claim, dict) and claim.get("by") == expected_owner
            )
        yield owns_claim

# Fields on a cron job that must never change after creation. ``id`` is used
# as a filesystem path component under ``OUTPUT_DIR``; allowing it to be
# updated lets an unsafe value (``../escape``, absolute path, nested) leak
# into output writes/deletes.
_IMMUTABLE_JOB_FIELDS = frozenset({"id"})


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


# Fields whose presence in an update can turn a runnable job into an empty one.
_PAYLOAD_FIELDS = frozenset({"prompt", "script", "skill", "skills", "no_agent"})

EMPTY_PAYLOAD_ERROR = (
    "Cron job has nothing to run: the prompt is blank and no script or "
    "skill(s) are set. Provide a prompt, a script, or at least one skill."
)

NO_AGENT_WITHOUT_SCRIPT_ERROR = (
    "no_agent=True requires a script — with no agent and no script "
    "there is nothing for the job to run."
)


def job_payload_is_empty(job: Dict[str, Any]) -> bool:
    """True when a job record has nothing runnable at all.

    A blank/whitespace prompt with no script and no skills would hand the
    agent an empty instruction on every fire (incident a5e29e688dc0).
    ``no_agent`` needs no special case here — it already requires a script.
    """
    if _coerce_job_text(job.get("prompt")).strip():
        return False
    if _coerce_job_text(job.get("script")).strip():
        return False
    if _normalize_skill_list(job.get("skill"), job.get("skills")):
        return False
    # Only flag if at least one payload field is explicitly present in the record
    if "prompt" in job or "script" in job or "skill" in job or "skills" in job:
        return True
    return False


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


def is_terminal_job(job: Dict[str, Any]) -> bool:
    """Return whether a job record is in a terminal scheduler state."""
    return job.get("state") in {"completed", "error"}


def _is_recoverable_error_job(job: Dict[str, Any]) -> bool:
    """True for a recurring job stuck in ``state=error``.

    ``state=error`` is set ONLY on a cron/interval job when
    ``compute_next_run()`` fails to produce a next occurrence (e.g. the
    ``croniter`` package is missing, or a malformed schedule) — see
    ``_mark_job_run_locked``'s issue #16265 comment: recurring jobs must
    NEVER be silently disabled. Unlike ``state=completed`` (a one-shot
    that genuinely has no more occurrences, ever), an error-state
    recurring job still has a schedule with future occurrences once the
    underlying issue resolves — it is stuck pending a ``next_run_at``
    recompute, not truly done.

    ``is_terminal_job()`` treats both states identically, which is correct
    for blocking bare reactivation through ``update_job`` on a genuinely
    completed job, but wrong here: it also blocks the due-scan's own
    ``next_run_at`` self-heal (``_get_due_jobs_locked`` already recomputes
    it for ``cron``/``interval`` jobs, but never reaches that code), the
    at-most-once pre-advance (``advance_next_runs``), the dispatch claim
    (``_claim_job_for_fire_locked``), and manual recovery (``resume_job``)
    — wedging the job forever with no exit except deleting and recreating
    it. Callers that need "is this job truly done" should keep using
    ``is_terminal_job()`` alone; callers that need "can this job still
    reach a future occurrence" should exclude this case.
    """
    return (
        job.get("state") == "error"
        and (job.get("schedule") or {}).get("kind") in {"cron", "interval"}
    )


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


def _is_named_profile_path(path: Path) -> bool:
    """Return True if *path* is inside a named profile home.

    Named profiles live under ``<hermes_home>/profiles/<name>/``.  The
    default profile lives at ``<hermes_home>`` directly (no ``profiles``
    parent), as do custom ``HERMES_HOME`` paths outside ``~/.hermes``.

    Checks both the resolved path (handles symlinks in the parent chain)
    and the raw path (catches symlinked profile homes whose resolve()
    target no longer contains ``profiles``).
    """
    try:
        if "profiles" in path.resolve().parts:
            return True
    except (OSError, RuntimeError):
        pass
    return "profiles" in path.parts


def _ensure_cron_dir(cron_dir: Path) -> None:
    """Create a cron directory without resurrecting a deleted profile home.

    Named profiles are created by the profile lifecycle, not cron.  A stale
    multiplex scheduler may still hold a path to a deleted profile after the
    user removes it; ``parents=False`` makes that race fail closed
    (FileNotFoundError) instead of silently restoring the directory tree.
    Default and custom Hermes homes keep ``parents=True`` so first-run
    directory creation still works.
    """
    if _is_named_profile_path(cron_dir):
        cron_dir.mkdir(exist_ok=True)
        return
    cron_dir.mkdir(parents=True, exist_ok=True)


def ensure_dirs():
    """Ensure cron directories exist with secure permissions."""
    store = _current_cron_store()
    _ensure_cron_dir(store.cron_dir)
    _ensure_cron_dir(store.output_dir)
    _secure_dir(store.cron_dir)
    _secure_dir(store.output_dir)


# =============================================================================
# Schedule Parsing
# =============================================================================

def normalize_repeat_value(repeat: Any) -> Optional[int]:
    """Coerce a repeat value from any entry point into ``Optional[int]``.

    The tool schema exposes ``repeat`` as an integer, but agents and users
    legitimately pass the user-facing strings ``'forever'``/``'once'`` or
    numeric strings (``'3'``). Uncoerced strings previously died with
    ``'<=' not supported between instances of 'str' and 'int'`` at create
    (#66824/#64520/#7142/#71987/#95706) and were stored raw by update paths,
    breaking ``mark_job_run`` later. Semantics: ``'forever'``-family -> None
    (infinite), ``'once'``-family -> 1, numeric -> int, 0/negative -> None,
    anything else -> ValueError (never store garbage).
    """
    if repeat is None:
        return None
    if isinstance(repeat, str):
        repeat_str = repeat.strip().lower()
        if repeat_str in ("forever", "infinite", "inf", "none", ""):
            return None
        if repeat_str in ("once", "one", "1x"):
            return 1
        try:
            repeat = int(repeat_str)
        except ValueError:
            raise ValueError(
                f"Invalid repeat value {repeat!r}: use an integer, "
                f"'forever', or 'once'."
            )
    return None if repeat <= 0 else int(repeat)


def parse_duration(s: str) -> int:
    """
    Parse duration string into minutes.
    
    Examples:
        "30m" → 30
        "2h" → 120
        "1d" → 1440
        "hour" → 60 (bare unit, no leading number)
    """
    s = s.strip().lower()
    match = re.match(r'^(\d*)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$', s)
    if not match:
        raise ValueError(
            f"Invalid duration: '{s}'. Use format like '30m', '2h', '1d', "
            "or a bare unit like 'hour' (defaults to 1)."
        )
    
    value = int(match.group(1)) if match.group(1) else 1
    unit = match.group(2)[0]  # First char: m, h, or d

    multipliers = {'m': 1, 'h': 60, 'd': 1440}
    return value * multipliers[unit]


# Natural-language day-spec phrases for the documented "every monday 9am" /
# "every day at 9am" schedule forms. Cron weekday numbering is
# 0=Sunday … 6=Saturday (croniter's default).
_WEEKDAY_TO_CRON_DOW = {
    "sunday": "0", "sun": "0",
    "monday": "1", "mon": "1",
    "tuesday": "2", "tue": "2", "tues": "2",
    "wednesday": "3", "wed": "3", "weds": "3",
    "thursday": "4", "thu": "4", "thur": "4", "thurs": "4",
    "friday": "5", "fri": "5",
    "saturday": "6", "sat": "6",
}

# Keyword day-specs that expand to a cron weekday field.
_DAYSPEC_TO_CRON_DOW = {
    "day": "*", "daily": "*", "everyday": "*",
    "weekday": "1-5", "weekdays": "1-5",
    "weekend": "0,6", "weekends": "0,6",
}


def _parse_clock_time(text: str) -> Optional[tuple]:
    """Parse a wall-clock time into a ``(hour, minute)`` 24-hour tuple.

    Accepts ``9am``, ``9:30am``, ``9 am``, ``14:00``, ``7`` (bare hour, 24h),
    ``noon``/``midday``, and ``midnight``. Returns None when the text is not a
    recognized clock time so the caller can reject the schedule cleanly.
    """
    t = text.strip().lower().replace(" ", "")
    if not t:
        return None
    if t in ("noon", "midday"):
        return (12, 0)
    if t == "midnight":
        return (0, 0)
    match = re.match(r'^(\d{1,2})(?::(\d{2}))?(am|pm)?$', t)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    meridiem = match.group(3)
    if meridiem:
        if not 1 <= hour <= 12:
            return None
        if meridiem == "am":
            hour = 0 if hour == 12 else hour
        else:  # pm
            hour = 12 if hour == 12 else hour + 12
    if hour > 23 or minute > 59:
        return None
    return (hour, minute)


def _natural_every_to_cron(rest: str) -> Optional[str]:
    """Convert a documented ``every <when> [at] <time>`` phrase to a 5-field
    cron expression, or None when *rest* is not such a phrase.

    Examples::

        "monday 9am"      -> "0 9 * * 1"
        "day at 9am"      -> "0 9 * * *"
        "weekday at 9am"  -> "0 9 * * 1-5"
        "monday, wednesday at 9am" -> "0 9 * * 1,3"

    Returning None lets ``parse_schedule`` fall back to the interval
    (``every 30m``) path, so existing duration schedules are unaffected.
    """
    tokens = rest.lower().replace(",", " ").split()
    if not tokens:
        return None

    # Consume one or more leading day tokens: a keyword spec ("weekdays"),
    # a single weekday, or a comma/"and"-separated weekday list
    # ("monday, wednesday at 9am").
    day_token = tokens[0]
    dow = _DAYSPEC_TO_CRON_DOW.get(day_token)
    idx = 1
    if dow is None:
        days = []
        while idx <= len(tokens):
            tok = tokens[idx - 1]
            if tok == "and":
                idx += 1
                continue
            mapped = _WEEKDAY_TO_CRON_DOW.get(tok)
            if mapped is None:
                break
            if mapped not in days:
                days.append(mapped)
            idx += 1
        if not days:
            return None
        dow = ",".join(days)
        idx -= 1

    time_tokens = tokens[idx:]
    # Optional "at" separator: "every day at 9am".
    if time_tokens and time_tokens[0] == "at":
        time_tokens = time_tokens[1:]
    if not time_tokens:
        return None

    parsed = _parse_clock_time(" ".join(time_tokens))
    if parsed is None:
        return None
    hour, minute = parsed
    return f"{minute} {hour} * * {dow}"


def parse_schedule(schedule: str) -> Dict[str, Any]:
    """
    Parse schedule string into structured format.
    
    Returns dict with:
        - kind: "once" | "interval" | "cron"
        - For "once": "run_at" (ISO timestamp)
        - For "interval": "minutes" (int)
        - For "cron": "expr" (cron expression)
    
    Examples:
        "30m"              → every 30 minutes (recurring)
        "2h"               → every 2 hours (recurring)
        "every 30m"        → recurring every 30 minutes
        "every 2h"         → recurring every 2 hours
        "every monday 9am" → recurring weekly (cron)
        "every day at 9am" → recurring daily (cron)
        "0 9 * * *"        → cron expression
        "2026-02-03T14:00" → once at timestamp
    """
    schedule = schedule.strip()
    original = schedule
    schedule_lower = schedule.lower()

    # "every X" pattern → recurring interval, OR a documented natural-language
    # day/time phrase ("every monday 9am", "every day at 9am") → cron.
    if schedule_lower.startswith("every "):
        rest = schedule[6:].strip()
        cron_expr = _natural_every_to_cron(rest)
        if cron_expr is not None:
            if not _ensure_croniter():
                raise ValueError(
                    "Weekday/time schedules like 'every monday 9am' require the "
                    "'croniter' package. Install with: pip install croniter"
                )
            try:
                croniter(cron_expr)
            except Exception as e:
                raise ValueError(f"Invalid schedule '{original}': {e}")
            return {
                "kind": "cron",
                "expr": cron_expr,
                "display": original,
            }
        minutes = parse_duration(rest)
        return {
            "kind": "interval",
            "minutes": minutes,
            "display": f"every {minutes}m"
        }

    # No-"every" natural day/time phrases advertised by the Desktop dialog:
    # "weekdays at 9am", "monday at 9:30", "daily at 7am" (#51975). Reuse the
    # same helper — the phrase shape is identical without the "every " prefix.
    cron_expr = _natural_every_to_cron(schedule_lower)
    if cron_expr is not None:
        if not _ensure_croniter():
            raise ValueError(
                "Weekday/time schedules like 'weekdays at 9am' require the "
                "'croniter' package. Install with: pip install croniter"
            )
        try:
            croniter(cron_expr)
        except Exception as e:
            raise ValueError(f"Invalid schedule '{original}': {e}")
        return {
            "kind": "cron",
            "expr": cron_expr,
            "display": original,
        }

    # Check for cron expression (5 or 6 space-separated fields)
    # Cron fields: minute hour day month weekday [year]
    # Allow letters so named months/weekdays (JAN-DEC, MON-SUN, incl. ranges
    # and lists like MON-FRI or MON,WED,FRI) are routed to croniter, which
    # supports them. The previous digit-only pattern silently rejected these
    # valid expressions as "Invalid schedule".
    parts = schedule.split()
    if len(parts) >= 5 and all(
        re.match(r'^[A-Za-z\d\*\-,/]+$', p) for p in parts[:5]
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
    
    # Duration like "30m", "2h", "1d" → RECURRING interval, matching the
    # documented tool contract ("30m (every 30 minutes)"). Previously this
    # returned kind="once", silently creating a one-shot job for a schedule
    # the schema documents as recurring — an agent passing '30m' for "every
    # 30 minutes" got a job that ran once and died (cron contract bug, fixed
    # 2026-08-04). Explicit one-shot-by-duration is "in 30m"/"in 2h".
    if schedule_lower.startswith("in "):
        duration_str = schedule[3:].strip()
        try:
            minutes = parse_duration(duration_str)
        except ValueError:
            raise ValueError(
                f"Invalid duration '{duration_str}' after 'in '. Use e.g. 'in 30m', 'in 2h'."
            )
        run_at = _hermes_now() + timedelta(minutes=minutes)
        return {
            "kind": "once",
            "run_at": run_at.isoformat(),
            "display": f"once in {duration_str}",
        }
    try:
        minutes = parse_duration(schedule)
        return {
            "kind": "interval",
            "minutes": minutes,
            "display": f"every {minutes}m",
        }
    except ValueError:
        pass
    
    raise ValueError(
        f"Invalid schedule '{original}'. Use:\n"
        f"  - Interval: '30m', 'every 30m', 'every 2h' (recurring)\n"
        f"  - One-shot delay: 'in 30m', 'in 2h' (fires once)\n"
        f"  - Weekly/daily: 'every monday 9am', 'weekdays at 9am' (recurring)\n"
        f"  - Cron: '0 9 * * *' (cron expression)\n"
        f"  - Timestamp: '2026-02-03T14:00:00' (one-shot at time)"
    )


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

    Uses half the schedule period (via ``_schedule_cadence_seconds``, the
    single cadence-measurement implementation), clamped between 120 seconds
    and 2 hours.  This ensures daily jobs can catch up if missed by up to
    2 hours, while frequent jobs (every 5-10 min) still fast-forward quickly.
    """
    MIN_GRACE = 120
    MAX_GRACE = 7200  # 2 hours

    period_seconds = _schedule_cadence_seconds(schedule)
    if not period_seconds:
        return MIN_GRACE
    grace = int(period_seconds) // 2
    return max(MIN_GRACE, min(grace, MAX_GRACE))


# Missed-run visibility (#99879): a recurring dispatch within this many
# seconds of its scheduled instant renders as "on time". The built-in ticker
# runs once a minute and a busy tick can push dispatch a couple of minutes
# past the scheduled instant — that is normal cadence, not gateway downtime.
_LATE_DISPATCH_TOLERANCE_SECONDS = 300


def _classify_dispatch_lateness(lateness_seconds: float, grace_seconds: int) -> str:
    """Classify a recurring dispatch by how late it fired.

    ``on_time``  — within normal ticker cadence slack;
    ``late``     — missed the scheduled instant but within the catch-up
                   grace window (e.g. gateway briefly down);
    ``catch_up`` — beyond the grace window; the due-scan skipped the
                   accumulated misses and executed once now.
    """
    if lateness_seconds > grace_seconds:
        return "catch_up"
    if lateness_seconds > _LATE_DISPATCH_TOLERANCE_SECONDS:
        return "late"
    return "on_time"


# Durable (persisted-state) recovery counter for a recurring job wedged in a
# stale ``last_status == "error"`` state with ``next_run_at`` parked in the
# future.  This is the restart-surviving half of the recurring-cron wedge
# (t_20e23f84 / t_8b5480b3): the in-memory stale-claim sweep in scheduler.py
# heals a leaked ``_running_job_ids`` claim in-process, but a job whose
# persisted ``next_run_at`` was re-armed into the future (the normal
# post-error re-arm by ``mark_job_run``) is invisible to that sweep — it is
# not in the running set, so nothing force-releases it, and it is not due, so
# ``get_due_jobs`` never returns it.  It just sits.  This recovery re-arms
# ``next_run_at`` to now so the next tick re-dispatches it, exactly like the
# operator's force-run / mech_red_guard's ``cron resume`` but built-in.
_persisted_error_recoveries: int = 0
# Bounded in-memory history kept by every probe-visible fire-path counter.
_TELEMETRY_RECENT_HISTORY = 20
_persisted_error_recoveries_recent: list = []


def _job_is_stale_error_recurring(
    job: Dict[str, Any],
    schedule: Dict[str, Any],
    now: datetime,
) -> bool:
    """True when a recurring job is wedged in a stale persisted error state.

    Condition (all must hold):
      * it is a recurring (cron/interval) job (checked by caller);
      * its persisted ``last_status == "error"`` (a prior fire errored and the
        job never recovered);
      * it has NOT successfully re-fired within its natural cadence — its
        ``last_run_at`` is older than ``cadence + grace``, so this is not a
        normal transient-error retry that will fire on its own soon, it is a
        job that has been sitting errored for a full period with no recovery;
      * it is not currently running in this process (a live run must never be
        re-armed underneath itself, #62002-style).

    ``last_run_at`` being older than one cadence is the key discriminator: a
    job that errors and is retried on its normal schedule keeps ``last_run_at``
    fresh (mark_job_run stamps it on every fire, success or failure), so the
    stale check does not fire for a job that is merely erroring-and-retrying.
    """
    if job.get("last_status") != "error":
        return False
    if _job_running_in_this_process(str(job.get("id") or "")):
        return False
    last_run = job.get("last_run_at")
    if not last_run:
        return False
    try:
        last_run_dt = _ensure_aware(datetime.fromisoformat(last_run))
    except (ValueError, TypeError):
        return False
    age_seconds = (now - last_run_dt).total_seconds()
    if age_seconds < 0:
        return False
    cadence_seconds = _schedule_cadence_seconds(schedule)
    if cadence_seconds is None:
        # Unknown cadence (croniter unavailable / malformed expr): fall back to
        # the grace window so a badly-parked job is still recovered, but never
        # re-arm anything younger than the 2h grace cap.
        cadence_seconds = _compute_grace_seconds(schedule)
    return age_seconds > (cadence_seconds + _compute_grace_seconds(schedule))


def _schedule_cadence_seconds(schedule: Dict[str, Any]) -> Optional[float]:
    """Approximate the natural period of a schedule, in seconds, or None.

    Interval jobs use ``minutes * 60``.  Cron jobs measure the gap between the
    next two fire times with croniter (falling back to None when croniter is
    missing or the expr is malformed).  Cron results are cached per expr —
    this runs inside ``_jobs_lock`` on every tick for every stale-errored
    job, and two croniter evaluations per call add up (the same reason
    ``scheduler.py`` caches ``_cron_interval_minutes``).  The measured gap
    can vary with the base time for irregular exprs; the cache trades that
    precision for not re-evaluating croniter under the lock, which is fine
    for a staleness *threshold*.
    """
    if not isinstance(schedule, dict):
        return None
    kind = schedule.get("kind")
    if kind == "interval":
        minutes = schedule.get("minutes")
        try:
            return float(minutes) * 60.0 if minutes else None
        except (TypeError, ValueError):
            return None
    if kind == "cron":
        if not _ensure_croniter():
            return None
        expr = schedule.get("expr")
        if not expr:
            return None
        if expr in _cron_cadence_cache:
            return _cron_cadence_cache[expr]
        try:
            base = _hermes_now()
            it = croniter(expr, base)
            first = it.get_next(datetime)
            second = it.get_next(datetime)
            gap = (second - first).total_seconds()
            result = gap if gap > 0 else None
        except Exception:
            result = None
        # Hard bound so deleted/edited exprs can never grow the cache
        # unboundedly in a long-lived gateway; a rare full clear costs two
        # croniter evaluations per live expr to rebuild.
        if len(_cron_cadence_cache) >= 256:
            _cron_cadence_cache.clear()
        _cron_cadence_cache[expr] = result
        return result
    return None


# Per-expr cache for _schedule_cadence_seconds' croniter measurements.
_cron_cadence_cache: Dict[str, Optional[float]] = {}


def _append_telemetry_record(filename: str, entry: Dict[str, Any], recent: list) -> None:
    """Keep ``entry`` in the bounded in-memory ``recent`` list and append it to
    ``<cron_dir>/<filename>`` (best effort — telemetry must never break a tick).

    Shared by the probe-visible fire-path counters (persisted-error recovery,
    timezone-migration catch-up); each keeps its own module-level int counter
    because tests reset those by name.
    """
    recent.append(entry)
    del recent[:-_TELEMETRY_RECENT_HISTORY]
    try:
        path = _current_cron_store().cron_dir / filename
        _ensure_cron_dir(path.parent)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        logger.debug("Could not append %s record: %s", filename, exc)


def _record_persisted_error_recovery(job: Dict[str, Any], previous_next_run: str) -> None:
    """Persist a countable, probe-visible signal for one stale-error re-arm."""
    global _persisted_error_recoveries
    entry = {
        "job_id": job.get("id"),
        "name": job.get("name") or job.get("id"),
        "previous_next_run_at": previous_next_run,
        "rearmed_at": _hermes_now().isoformat(),
    }
    _persisted_error_recoveries += 1
    _append_telemetry_record(
        "persisted_error_recoveries.jsonl", entry, _persisted_error_recoveries_recent
    )


def get_persisted_error_recovery_stats() -> Dict[str, Any]:
    """Probe-visible snapshot of persisted-error recoveries."""
    return {
        "persisted_error_recoveries": _persisted_error_recoveries,
        "recent": list(_persisted_error_recoveries_recent),
    }


def _cron_next_run_matches_expr(
    schedule: Dict[str, Any],
    next_run_dt: datetime,
) -> bool:
    """Whether ``next_run_dt`` is an occurrence of the schedule's current expr.

    A direct ``jobs.json`` edit can change ``schedule.expr`` while leaving the
    stored ``next_run_at`` computed under the *old* expression (#93049). The
    stored instant is stale exactly when it is not an occurrence of the
    current expression. Validation is best-effort: anything that cannot be
    checked (non-cron kind, missing expr, croniter unavailable, malformed
    input) reports a match so the fire path keeps its existing semantics.
    """
    if schedule.get("kind") != "cron":
        return True
    expr = schedule.get("expr")
    if not expr or not _ensure_croniter() or croniter is None:
        return True
    try:
        # The last occurrence at-or-before the stored instant: base croniter
        # a second past it so an exact occurrence is included, then compare
        # at second granularity (croniter returns second-precision datetimes).
        base = next_run_dt + timedelta(seconds=1)
        prev = croniter(str(expr), base).get_prev(datetime)
        return abs((prev - next_run_dt).total_seconds()) < 1.0
    except Exception:
        return True


# Classification results for a due cron instant that is NOT an occurrence of
# the job's current expression (see _classify_stale_cron_next_run).
STALE_CRON_MATCH = "match"
STALE_CRON_TIMEZONE_MIGRATION = "timezone_migration"
STALE_CRON_EXPR_EDIT = "expr_edit"


def _classify_stale_cron_next_run(
    schedule: Dict[str, Any],
    raw_next_run_dt: datetime,
    next_run_dt: datetime,
) -> str:
    """Explain WHY a stored ``next_run_at`` misses the current cron lattice.

    ``_cron_next_run_matches_expr`` answers "does the stored instant occur in
    the current expression?" but not "why not?", and the two answers call for
    opposite actions:

    * ``expr_edit`` — a direct ``jobs.json`` edit changed ``schedule.expr``
      while leaving ``next_run_at`` computed under the old one (#93049). The
      stored instant is a time the current expression *excludes*, so it must
      be re-anchored WITHOUT firing.
    * ``timezone_migration`` — the expression never changed; only the stored
      value's *offset representation* did. Upgrading from a UTC-scheduling
      build to one that honours the profile timezone leaves legacy rows like
      ``2026-09-02T04:00:00+00:00`` for ``0 4 * * *``; normalizing to
      Europe/Brussels turns that into ``06:00+02``, which the expression
      excludes. Treating it as a stale edit re-anchored to tomorrow and
      silently skipped a due occurrence that had never fired.

    The discriminator is whether *normalization itself* moved the wall clock.
    Cron expressions describe local wall-clock intent, so a stored instant
    whose OWN wall clock is a legal occurrence, and which only left the
    lattice because ``_ensure_aware`` converted it into a different offset, is
    a representation migration — not a schedule edit. When the offsets agree
    (the common case, including every value this build wrote) the wall clock
    is unchanged, so a genuine ``expr`` edit can never be misread as a
    migration.
    """
    if _cron_next_run_matches_expr(schedule, next_run_dt):
        return STALE_CRON_MATCH
    wall_clock_shifted = (
        raw_next_run_dt.replace(tzinfo=None) != next_run_dt.replace(tzinfo=None)
    )
    if wall_clock_shifted and _cron_next_run_matches_expr(schedule, raw_next_run_dt):
        return STALE_CRON_TIMEZONE_MIGRATION
    return STALE_CRON_EXPR_EDIT


# Durable, probe-visible counter for offset-representation migrations caught
# on the fire path. Kept separate from `catch_up_occurrences` (runs skipped
# past their grace window — a migrated row that is ALSO past grace increments
# both) because this one means "an upgrade rewrote how next_run_at is
# represented" — an operator seeing it climb after a deploy is seeing the
# migration drain, and seeing it climb steadily afterwards is seeing a
# timezone that keeps changing under the store.
_timezone_migration_catchups: int = 0
_timezone_migration_catchups_recent: list = []


def _record_timezone_migration_catchup(
    job: Dict[str, Any],
    raw_next_run_dt: datetime,
    next_run_dt: datetime,
) -> None:
    """Persist a countable signal for one offset-migration catch-up fire."""
    global _timezone_migration_catchups
    entry = {
        "job_id": job.get("id"),
        "name": job.get("name") or job.get("id"),
        "expr": (job.get("schedule") or {}).get("expr"),
        "stored_next_run_at": raw_next_run_dt.isoformat(),
        "normalized_next_run_at": next_run_dt.isoformat(),
        "fired_at": _hermes_now().isoformat(),
    }
    _timezone_migration_catchups += 1
    _append_telemetry_record(
        "timezone_migration_catchups.jsonl", entry, _timezone_migration_catchups_recent
    )


def get_timezone_migration_catchup_stats() -> Dict[str, Any]:
    """Probe-visible snapshot of offset-migration catch-up fires."""
    return {
        "timezone_migration_catchups": _timezone_migration_catchups,
        "recent": list(_timezone_migration_catchups_recent),
    }


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
        needs_shape_repair = False
        if isinstance(jobs, dict):
            # ID-keyed map ({"jobs": {"<job_id>": {...}, ...}}) — written by
            # external tools or hand edits, never by save_jobs(). The public
            # load_jobs() contract and every CRUD/scheduler consumer expect a
            # list; iterating the dict would yield bare id strings and blow up
            # downstream (e.g. _normalize_job_record -> dict(<str>)). Flatten
            # with an id-preserving merge: an inline "id" on the value wins,
            # otherwise the map key is adopted as the id (external tools often
            # key by id and omit the inline copy). Non-dict values are junk
            # (a flattened record would not be a job) — skip them with a
            # warning instead of crashing downstream.
            # (_peek_jobs_unlocked() deliberately does NOT flatten —
            # it returns None on a dict-shaped store so the save path never
            # merges against an unrepaired baseline; the flatten lives only
            # here at the load boundary.)
            skipped = [k for k, v in jobs.items() if not isinstance(v, dict)]
            if skipped:
                logger.warning(
                    "Skipping %d non-dict entr%s in id-keyed jobs map: %s",
                    len(skipped),
                    "y" if len(skipped) == 1 else "ies",
                    ", ".join(map(repr, skipped)),
                )
            jobs = [
                {**v, "id": v.get("id") or k}
                for k, v in jobs.items()
                if isinstance(v, dict)
            ]
            needs_shape_repair = True
        if jobs and (_strict_retry or needs_shape_repair):
            # Rewrite into the canonical {"jobs": [...]} form: either the parse
            # hit control-character corruption (_strict_retry) or the store was
            # an id-keyed map. save_jobs() re-emits the list shape every reader
            # expects.
            save_jobs(jobs)
            if needs_shape_repair:
                logger.warning("Auto-repaired jobs.json (id-keyed jobs map flattened to list)")
            else:
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

            atomic_replace(tmp_path, jobs_file)
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
        atomic_replace(tmp_path, jobs_file)
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
    with _jobs_lock():
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


def _normalize_reasoning_effort(value: Any) -> Optional[str]:
    """Validate a per-job reasoning effort against the canonical grammar.

    Spelling-only validation at the storage choke point: the SAME parser
    every other effort surface uses (``hermes_constants.parse_reasoning_effort``)
    decides validity, so the cron knob can never be stricter or looser than
    its config.yaml sibling. Capability (whether the resolved model supports
    the level) is intentionally NOT checked here — the model is not knowable
    at create time (unpinned jobs, auth fallback), and the provider
    transports already clamp/omit at send time.

    Returns None for unset (None/empty string), the normalized lowercase
    level for valid input, and raises ValueError otherwise so nothing
    invalid ever persists for a fire-and-forget job.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    from hermes_constants import parse_reasoning_effort

    if parse_reasoning_effort(text) is None:
        raise ValueError(
            f"Invalid reasoning_effort {value!r}. Valid levels: "
            "none, minimal, low, medium, high, xhigh, max, ultra "
            "(empty string clears the override)."
        )
    # parse_reasoning_effort accepts disable aliases ("false", "disabled");
    # store the canonical spelling so the record reads unambiguously.
    if text in {"false", "disabled"}:
        return "none"
    return text


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
        raise ValueError(NO_AGENT_WITHOUT_SCRIPT_ERROR)


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
    reasoning_effort: Optional[str] = None,
    failure_deliver: Optional[str] = None,
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
        reasoning_effort: Optional per-job reasoning effort pin. One of the
                canonical Hermes levels (none|minimal|low|medium|high|xhigh|
                max|ultra, case-insensitive). When set, it wins over BOTH the
                global ``agent.reasoning_effort`` and per-model
                ``agent.reasoning_overrides`` at fire time. Capability is NOT
                validated here: levels above what the resolved model supports
                are clamped or omitted by the provider transport at send time,
                exactly like config-set effort. Inert with ``no_agent=True``
                (no LLM call to configure). None/empty = unset (job follows
                config resolution, pre-existing behavior).

    Returns:
        The created job dict
    """
    parsed_schedule = parse_schedule(schedule)

    # Normalize repeat: treat 0 or negative values as None (infinite).
    # String forms ('forever'/'once'/numeric) coerce via
    # normalize_repeat_value — the shared chokepoint with update paths
    # (#66824/#64520/#7142/#71987/#95706).
    repeat = normalize_repeat_value(repeat)

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
    normalized_reasoning_effort = _normalize_reasoning_effort(reasoning_effort)
    # failure_deliver shares deliver's value grammar; the str/list
    # flatten below mirrors the tool layer's _normalize_deliver_param for
    # direct create_job callers (the tool pre-normalizes). Semantic
    # validation happens at resolution time via the shared deliver path.
    normalized_failure_deliver = (
        str(failure_deliver).strip() if isinstance(failure_deliver, str) else None
    )
    if isinstance(failure_deliver, (list, tuple)):
        normalized_failure_deliver = ",".join(
            str(p).strip() for p in failure_deliver if str(p).strip()
        )
    normalized_failure_deliver = normalized_failure_deliver or None
    normalized_monitor_script = str(monitor_script).strip() if isinstance(monitor_script, str) else None
    normalized_monitor_script = normalized_monitor_script or None
    normalized_monitor_url = str(monitor_url).strip() if isinstance(monitor_url, str) else None
    normalized_monitor_url = normalized_monitor_url or None

    # Monitor-mode validation: exactly one source, and monitor mode only
    # makes sense when there IS an agent to suppress/wake.
    # no_agent jobs are meaningless without a script — the script IS the job.
    # Surface these as clear ValueErrors at create time so bad configs never
    # reach the scheduler (shared with update_job, see
    # _validate_job_mode_invariants).
    _validate_job_mode_invariants(
        normalized_monitor_script,
        normalized_monitor_url,
        normalized_no_agent,
        normalized_script,
    )

    # Normalize context_from: accept str or list of str, store as list or None
    if isinstance(context_from, str):
        context_from = [context_from.strip()] if context_from.strip() else None
    elif isinstance(context_from, list):
        context_from = [str(j).strip() for j in context_from if str(j).strip()] or None
    else:
        context_from = None

    prompt_text = _coerce_job_text(prompt).strip()

    if not prompt_text and not normalized_script and not normalized_skills:
        raise ValueError(EMPTY_PAYLOAD_ERROR)

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
        # Live-adapter targets whose last send was acked with no message_id /
        # raw_response (accepted, but UNVERIFIED — surfaced by cron list/doctor).
        "last_delivery_unverified": None,
        "failure_streak": 0,
        # Delivery configuration
        "deliver": deliver,
        "origin": origin,  # Tracks where job was created for "origin" delivery
        "enabled_toolsets": normalized_toolsets,
        "workdir": normalized_workdir,
    }
    # Only persist attach_to_session when explicitly set, so existing jobs and
    # the common case stay byte-identical (absent key => fall back to the
    # global cron.mirror_delivery config, default off).
    if normalized_attach is not None:
        job["attach_to_session"] = normalized_attach
    # Same conditional-persist rule for the per-job reasoning effort pin:
    # absent key = job follows config resolution (pre-feature behavior).
    if normalized_reasoning_effort is not None:
        job["reasoning_effort"] = normalized_reasoning_effort
    # Conditional-persist for failure_deliver too: absent key = failures
    # follow deliver, byte-identical to pre-feature jobs (NS-788).
    if normalized_failure_deliver is not None:
        job["failure_deliver"] = normalized_failure_deliver

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

            # Validate/normalize the per-job reasoning effort pin the same
            # way create_job does: canonical grammar only, empty string (or
            # None) clears. Invalid values raise BEFORE the merge so the
            # stored value stays untouched.
            if "reasoning_effort" in updates:
                updates["reasoning_effort"] = _normalize_reasoning_effort(
                    updates["reasoning_effort"]
                )

            # Normalize repeat the same way create_job does. Callers pass
            # either the stored dict shape ({"times": N, "completed": M}) or
            # a bare value ("forever", "once", 3, "3"); bare values coerce
            # through normalize_repeat_value and preserve the completed
            # counter. A raw string stored here previously broke
            # mark_job_run ('str' has no .get) and repeat accounting.
            if "repeat" in updates:
                _rp = updates["repeat"]
                if isinstance(_rp, dict):
                    _rp = dict(_rp)
                    _rp["times"] = normalize_repeat_value(_rp.get("times"))
                    _rp.setdefault("completed", (job.get("repeat") or {}).get("completed", 0))
                    updates["repeat"] = _rp
                else:
                    updates["repeat"] = {
                        "times": normalize_repeat_value(_rp),
                        "completed": (job.get("repeat") or {}).get("completed", 0),
                    }

            previous_inference_axes = _normalized_inference_axes(job)
            updated = _apply_skill_fields({**job, **updates})

            if (
                is_terminal_job(job)
                and not _is_recoverable_error_job(job)
                and (
                    updated.get("state") not in {"completed", "error"}
                    or updated.get("enabled") is True
                    or updated.get("next_run_at") is not None
                )
            ):
                raise ValueError(
                    f"Cannot activate terminal cron job '{job.get('name', job_id)}' "
                    "through update_job; use cron resume --run-now or --at."
                )

            # Re-check execution-mode invariants on the MERGED record when
            # any participating field changes, so create-time invariants
            # can't be violated through the update door (e.g. flipping
            # no_agent=True on a monitor job would silently disable the
            # monitor: the scheduler's no_agent short-circuit runs before
            # the monitor gate). Scoped to changed fields so legacy records
            # untouched by this update keep loading.
            if {"monitor_script", "monitor_url", "no_agent", "script"}.intersection(updates):
                _upd_script = updated.get("script")
                _upd_script = str(_upd_script).strip() if isinstance(_upd_script, str) else None
                _validate_job_mode_invariants(
                    updated.get("monitor_script") or None,
                    updated.get("monitor_url") or None,
                    bool(updated.get("no_agent")),
                    _upd_script or None,
                )

            if any(k in updates for k in _PAYLOAD_FIELDS):
                if job_payload_is_empty(updated):
                    raise ValueError(EMPTY_PAYLOAD_ERROR)
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

            if (
                is_terminal_job(job)
                and not _is_recoverable_error_job(job)
                and (
                    updated.get("state") not in {"completed", "error"}
                    or updated.get("enabled") is True
                    or updated.get("next_run_at") is not None
                )
            ):
                raise ValueError(
                    f"Cannot activate terminal cron job '{job.get('name', job_id)}' "
                    "through update_job; use cron resume --run-now or --at."
                )

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

    next_run_at = compute_next_run(job["schedule"])
    if next_run_at is None and job["schedule"].get("kind") == "once":
        run_at = job["schedule"].get("run_at", "unknown")
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


def trigger_job(
    job_id: str, extra_prompt: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Schedule a job to run on the next scheduler tick. Accepts a job ID or name.

    ``extra_prompt``: optional transient per-run context for the manual fire
    (from ``cronjob(action='run', prompt=...)`` forwarded through the gateway
    api_server). Stamped as ``manual_run_prompt`` alongside ``manual_run_at``
    and consumed by ``run_one_job`` for that single fire only —
    ``mark_job_run`` clears it, so it never persists into the job definition
    or later scheduled fires.
    """
    job = resolve_job_ref(job_id)
    if not job:
        return None
    if is_terminal_job(job):
        state = job.get("state")
        name = job.get("name", job_id)
        raise ValueError(
            f"Cannot run: job '{name}' is {state} (terminal). "
            f"Create a new occurrence with 'hermes cron resume {name} "
            "--run-now' or '--at <ISO-8601>'."
        )
    manual_run_at = _hermes_now().isoformat()
    return update_job(
        job["id"],
        {
            "enabled": True,
            "state": "scheduled",
            "paused_at": None,
            "paused_reason": None,
            "next_run_at": manual_run_at,
            # Persist run-now intent alongside the arbitrary instant so cron
            # expression/TZ repair guards do not mistake it for stale state.
            "manual_run_at": manual_run_at,
            # Transient run context rides with the run-now intent (None
            # clears any stale prompt from a previous trigger).
            "manual_run_prompt": (extra_prompt or None),
        },
    )


def _claim_is_live(claim: Any, now: datetime, ttl_seconds: float) -> bool:
    if not isinstance(claim, dict) or not claim.get("at"):
        return False
    try:
        age = (now - _ensure_aware(datetime.fromisoformat(claim["at"]))).total_seconds()
    except (TypeError, ValueError):
        return False
    return 0 <= age < ttl_seconds


def rearm_oneshot(job_id: str, run_at: Any) -> Optional[Dict[str, Any]]:
    """Re-arm a completed one-shot as an explicit new occurrence."""
    job_ref = resolve_job_ref(job_id)
    if not job_ref:
        return None
    if isinstance(run_at, datetime):
        run_at = run_at.isoformat()
    parsed_schedule = parse_schedule(str(run_at))
    if parsed_schedule.get("kind") != "once":
        raise ValueError(
            "Cannot re-arm recurring jobs: re-arm is one-shot-only; "
            "use plain resume or cron run."
        )
    next_run_at = compute_next_run(parsed_schedule)
    if next_run_at is None:
        requested = parsed_schedule.get("run_at") or run_at
        raise ValueError(
            f"Requested one-shot time {requested} is more than "
            f"{ONESHOT_GRACE_SECONDS}s in the past and cannot be scheduled."
        )

    with _jobs_lock():
        jobs = load_jobs()
        for index, job in enumerate(jobs):
            if job.get("id") != job_ref["id"]:
                continue
            now = _hermes_now()
            if _claim_is_live(job.get("run_claim"), now, _oneshot_run_claim_ttl_seconds()):
                raise ValueError("Cannot re-arm one-shot over a live run claim.")
            if _claim_is_live(job.get("fire_claim"), now, 300):
                raise ValueError("Cannot re-arm one-shot over a live fire claim.")
            if job.get("schedule", {}).get("kind") != "once":
                raise ValueError(
                    "Cannot re-arm recurring jobs: re-arm is one-shot-only; "
                    "use plain resume or cron run."
                )
            repeat = job.get("repeat") or {}
            repeat["completed"] = 0
            job["schedule"] = parsed_schedule
            job["schedule_display"] = parsed_schedule.get("display", str(run_at))
            job["repeat"] = repeat
            job["run_claim"] = None
            job["fire_claim"] = None
            job["enabled"] = True
            job["state"] = "scheduled"
            job["paused_at"] = None
            job["paused_reason"] = None
            job["next_run_at"] = next_run_at
            jobs[index] = job
            save_jobs(jobs)
            return _normalize_job_record(job)
    return None


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
            # Prune the per-job fire-fence lock entry so the registry does
            # not grow monotonically across create/remove cycles.
            _fence_key = f"{_current_cron_store().cron_dir.resolve()}::{canonical_id}"
            with _fire_fence_locks_guard:
                _fire_fence_locks.pop(_fence_key, None)
            return True
    return False


def mark_job_run(
    job_id: str,
    success: bool,
    error: Optional[str] = None,
    delivery_error: Optional[str] = None,
    status: Optional[str] = None,
    *,
    expected_fire_owner: Optional[str] = None,
) -> bool:
    with _fire_job_lock(job_id) as acquired:
        if not acquired:
            return False
        return _mark_job_run_locked(
            job_id,
            success,
            error,
            delivery_error,
            status=status,
            expected_fire_owner=expected_fire_owner,
        )


def _set_alert_flag(job_id: str, field: str, value: bool) -> bool:
    """Set/clear a persisted alert-dedup marker; return the PRIOR value.

    The marker records that the operator was already alerted about this
    job's condition, so the scheduler alerts exactly once and stays silent
    on subsequent ticks until the condition heals (same alert-once shape as
    the dead-pin auto-pause in #73506). Persisted on the job record so the
    dedup survives gateway restarts. Fields: ``preflight_alerted`` (blocked
    config, T1-26) and ``drift_alerted`` (#44585 drift-guard skip).
    """
    with _jobs_lock():
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] == job_id:
                prior = bool(job.get(field))
                if value:
                    job[field] = True
                else:
                    job.pop(field, None)
                if prior != value:
                    jobs[i] = job
                    save_jobs(jobs)
                return prior
    return False


def _set_preflight_alerted(job_id: str, value: bool) -> bool:
    """Set/clear the preflight alert-dedup marker; return the PRIOR value."""
    return _set_alert_flag(job_id, "preflight_alerted", value)


def mark_preflight_alerted(job_id: str) -> bool:
    """Mark the job as preflight-alerted; return True if it already was."""
    return _set_preflight_alerted(job_id, True)


def clear_preflight_alerted(job_id: str) -> None:
    """Clear the preflight alert-dedup marker (config validates again)."""
    _set_preflight_alerted(job_id, False)


def mark_drift_alerted(job_id: str) -> bool:
    """Mark the job as drift-alerted; return True if it already was."""
    return _set_alert_flag(job_id, "drift_alerted", True)


def clear_drift_alerted(job_id: str) -> None:
    """Clear the drift alert-dedup marker (resolution matches again)."""
    _set_alert_flag(job_id, "drift_alerted", False)


def note_fire_forward_failure(job_id: str, detail: str) -> bool:
    """Durably record that a scheduled fire could not be handed to the runner.

    Written by the dashboard fire webhook when the loopback forward to the
    gateway api_server fails (gateway unreachable / listener not bound) —
    the shape behind "job runs manually but never auto-fires". Without this
    stamp the miss is invisible outside gui.log: no execution row is created
    (the claim never happens) and ``last_status``/``last_error`` only cover
    runs that actually started.

    Stored as ``last_fire_error`` (``{"at": iso, "detail": str}``) on the job
    record so `cronjob list`, the CLI, and the dashboard all surface it.
    Cleared by the next successful run (``mark_job_run``). Repeated failures
    overwrite in place — latest miss wins; per-fire history lives in the
    scheduler's own logs.

    Returns True when a job record was found and stamped.
    """
    with _jobs_lock():
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] == job_id:
                job["last_fire_error"] = {
                    "at": _hermes_now().isoformat(),
                    "detail": str(detail or "")[:500],
                }
                jobs[i] = job
                save_jobs(jobs)
                return True
    return False


def _mark_job_run_locked(
    job_id: str,
    success: bool,
    error: Optional[str] = None,
    delivery_error: Optional[str] = None,
    *,
    status: Optional[str] = None,
    expected_fire_owner: Optional[str] = None,
) -> bool:
    """
    Mark a job as having been run.
    
    Updates last_run_at, last_status, increments completed count,
    computes next_run_at, and auto-deletes if repeat limit reached.

    ``delivery_error`` is tracked separately from the agent error — a job
    can succeed (agent produced output) but fail delivery (platform down).

    A run that succeeded but failed delivery records
    ``last_status = "delivery_failed"`` (never "ok") so the failure is
    visible to every reader, while ``failure_streak`` stays untouched —
    the agent did its job.

    ``status`` overrides the derived ``last_status`` ("ok"/"error"/
    "delivery_failed") with a
    specific terminal status for this run — e.g. ``"blocked_config"`` when
    the pre-dispatch configuration validation refused to run the agent
    (T1-26), so `cronjob list` distinguishes "your config is broken" from
    "the run itself failed".
    """
    with _jobs_lock():
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] == job_id:
                if expected_fire_owner is not None:
                    claim = job.get("fire_claim")
                    if not isinstance(claim, dict) or claim.get("by") != expected_fire_owner:
                        logger.warning(
                            "mark_job_run: job_id %s fire claim owner changed; "
                            "discarding stale completion",
                            job_id,
                        )
                        return False
                now = _hermes_now().isoformat()
                job["last_run_at"] = now
                job.pop("manual_run_at", None)
                # The transient manual-run context is single-fire: whatever
                # run just completed consumed it (or superseded it).
                job.pop("manual_run_prompt", None)
                # A run whose agent succeeded but whose delivery failed is NOT
                # "ok": recording it as such hid last_delivery_error behind a
                # green status in `cron list`/the UI and made a job that never
                # reached the user look like a quiet success (#83993). It gets
                # its own status so every reader that keys off "ok" (CLI list,
                # doctor, cronjob_tools) sees the failure. An explicit
                # ``status`` override (e.g. "blocked_config") still wins.
                if status:
                    job["last_status"] = status
                elif not success:
                    job["last_status"] = "error"
                elif isinstance(delivery_error, str) and delivery_error.strip():
                    job["last_status"] = "delivery_failed"
                else:
                    job["last_status"] = "ok"
                job["last_error"] = error if not success else None
                # A healthy run means the configuration validates again — drop
                # the preflight alert-dedup marker so a FUTURE config break
                # re-alerts instead of being silently swallowed. Same contract
                # for the drift marker (#44585 alert-once): a run that made it
                # through the guard means resolution matches again.
                if success:
                    job.pop("preflight_alerted", None)
                    job.pop("drift_alerted", None)
                    # The fire hand-off demonstrably works again — clear the
                    # forward-failure stamp so it only ever describes the
                    # CURRENT auto-fire health, not a healed past incident.
                    job.pop("last_fire_error", None)
                # Consecutive agent-failure streak. Any successful run resets
                # it; delivery failures alone do NOT count (the agent did its
                # job). Read by the scheduler's failure-delivery path to nudge
                # the user to review a repeatedly-failing automation
                # (Poke-inspired; see cron/scheduler._failure_streak_nudge).
                if success:
                    job["failure_streak"] = 0
                else:
                    job["failure_streak"] = int(job.get("failure_streak") or 0) + 1
                # Track delivery failures separately — cleared on successful delivery
                job["last_delivery_error"] = delivery_error
                # Clear any external-fire claim so a re-armed recurring job can
                # be claimed again on its next fire (Phase 4C CAS).
                job["fire_claim"] = None
                # Clear the one-shot running-claim (#59229): the run is over, so
                # a re-armed recurring job or a re-dispatched one-shot recovery
                # is claimable again. No-op if the job never carried a claim.
                if job.get("run_claim") is not None:
                    job["run_claim"] = None
                
                # Increment completed count.  Finite one-shot jobs are
                # pre-claimed by claim_dispatch() BEFORE the side effect runs
                # (issue #38758), which already incremented completed — do not
                # double-count them here.  Recurring jobs and direct callers
                # with no pre-run claim still get the legacy increment.
                if job.get("repeat"):
                    repeat = job["repeat"]
                    times = repeat.get("times")
                    completed = repeat.get("completed", 0)
                    kind = job.get("schedule", {}).get("kind")
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
                job["next_run_at"] = compute_next_run(job["schedule"], now)

                # If no next run, decide whether this is terminal completion
                # (one-shot) or a transient failure (recurring schedule couldn't
                # compute — e.g. 'croniter' missing from the runtime env).
                # Recurring jobs must NEVER be silently disabled: that turns a
                # missing runtime dep into "job completed" and the user's
                # schedule quietly goes off. See issue #16265.
                if job["next_run_at"] is None:
                    kind = job.get("schedule", {}).get("kind")
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


def _write_missed_oneshot_diagnostic(job: Dict[str, Any], next_run: str) -> None:
    """Leave an operator-visible trace when a never-ran one-shot is retired
    because its persisted run time fell outside the grace window.

    The record is removed because the scheduler will never fire it (see the
    grace gate in ``_get_due_jobs_locked``); without a diagnostic the job
    would just vanish. Best-effort: a diagnostic failure never blocks the
    removal itself.
    """
    try:
        text = (
            "# Cron job removed before firing (run time outside grace window)\n\n"
            f"- job id: {job.get('id')}\n"
            f"- name: {job.get('name')}\n"
            f"- scheduled run time: {next_run}\n"
            f"- grace window: {ONESHOT_GRACE_SECONDS}s\n"
            f"- removed at: {_hermes_now().isoformat()}\n\n"
            "This one-shot's run time is more than the grace window in the "
            "past (scheduler down past the window, host asleep, or jobs.json "
            "edited), which is outside the 'will never fire' contract "
            "enforced at create/update/resume time. The job was removed "
            "without running; recreate it (or use the Run button) to "
            "schedule it again.\n"
        )
        save_job_output(job.get("id", ""), text)
    except Exception as e:
        logger.debug(
            "Failed to write missed-oneshot diagnostic for job %r: %s",
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
    with _jobs_lock():
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job["id"] != job_id:
                continue
            if job.get("schedule", {}).get("kind") != "once":
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
    with _jobs_lock():
        jobs = load_jobs()
        for job in jobs:
            if job.get("id") != job_id:
                continue
            if job.get("schedule", {}).get("kind") != "once":
                return False
            claim = job.get("run_claim")
            if not isinstance(claim, dict) or claim.get("by") != expected_owner:
                return False
            claim["at"] = _hermes_now().isoformat()
            save_jobs(jobs)
            return True
    return False


def clear_run_claim(job_id: str) -> bool:
    """Clear a one-shot job's ``run_claim`` when its dispatch fails.

    ``get_due_jobs`` stamps a ``run_claim`` before returning a one-shot as
    due (#59229).  ``mark_job_run`` clears it on *successful* completion.
    When dispatch itself fails (interpreter shutdown, executor submit error,
    execution-creation error) the job never reaches ``mark_job_run`` and the
    stale claim blocks re-dispatch until the TTL expires (default 30 min).

    Calling this on every early-exit path restores the "the job stays due
    and will fire on the next healthy tick" invariant that the scheduler
    comment promises (#86522).
    """
    with _jobs_lock():
        jobs = load_jobs()
        for job in jobs:
            if job.get("id") != job_id:
                continue
            if job.get("schedule", {}).get("kind") != "once":
                return False
            if job.get("run_claim") is not None:
                job["run_claim"] = None
                save_jobs(jobs)
                return True
            return False  # already cleared
    return False


def advance_next_runs(job_ids) -> int:
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
    with _jobs_lock():
        jobs = load_jobs()
        now = _hermes_now().isoformat()
        advanced = 0
        for job in jobs:
            if job["id"] not in ids:
                continue
            if is_terminal_job(job) and not _is_recoverable_error_job(job):
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


def advance_next_run(job_id: str) -> bool:
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
    return advance_next_runs([job_id]) >= 1


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


def claim_job_for_fire(
    job_id: str,
    *,
    claim_ttl_seconds: int = 300,
    force: bool = False,
    return_job: bool = False,
) -> Union[bool, Dict[str, Any]]:
    with _fire_job_lock(job_id) as acquired:
        if not acquired:
            return False
        return _claim_job_for_fire_locked(
            job_id,
            claim_ttl_seconds=claim_ttl_seconds,
            force=force,
            return_job=return_job,
        )


def _claim_job_for_fire_locked(
    job_id: str,
    *,
    claim_ttl_seconds: int = 300,
    force: bool = False,
    return_job: bool = False,
) -> Union[bool, Dict[str, Any]]:
    """Atomically claim a job for a single external 'fire' (multi-machine
    at-most-once). Returns True iff THIS caller won the claim.

    Used by the external-provider fire path (``CronScheduler.fire_due``) when an
    external scheduler (Chronos) signals a job is due across N gateway replicas:
    exactly one wins. Single-machine deployments always win.

    Under the file lock: reject if the job is missing/disabled/paused. An
    explicit manual fire may pass ``force=True`` to atomically enable and
    resume the job as part of the claim; external scheduler callbacks must
    leave it false so a stale callback cannot resurrect a paused job. If a
    fresh claim (younger than ``claim_ttl_seconds``) already exists, lose.
    Otherwise stamp a ``fire_claim`` and, for recurring jobs, advance
    ``next_run_at`` (mirrors ``advance_next_run``'s at-most-once bump so a stale
    re-delivery for the old time can't re-fire). One-shots keep ``next_run_at``
    but the fresh ``fire_claim`` blocks a duplicate retry for the same fire.
    ``mark_job_run`` clears the claim on completion so a re-armed recurring job
    is claimable again next fire.

    The stale-claim TTL means a machine that crashed after claiming but before
    completing doesn't wedge the job forever — after the TTL another fire can
    reclaim it.
    """
    with _jobs_lock():
        jobs = load_jobs()
        for job in jobs:
            if job["id"] != job_id:
                continue
            if is_terminal_job(job) and not _is_recoverable_error_job(job):
                return False
            # enabled + pause markers must both clear — a half-paused record
            # (enabled=true, state=paused/paused_at set) must not claim. An
            # explicit ``force`` (Trigger-now on a paused job) bypasses the
            # gate and atomically resumes the job below.
            if not force and not is_job_runnable(job):
                return False
            now = _hermes_now()
            existing = job.get("fire_claim")
            if existing:
                try:
                    claimed_at = _ensure_aware(datetime.fromisoformat(existing["at"]))
                    # Bounded on BOTH sides (#60703): a claim stamped in the
                    # future (clock/TZ skew across a restart, or a corrupted
                    # timestamp) would otherwise have a negative age and stay
                    # "fresh" forever — the job becomes permanently unfireable
                    # and every manual `cron run` reports "already being
                    # fired". Treat future-dated claims as stale/overwritable.
                    _age = (now - claimed_at).total_seconds()
                    if 0 <= _age < claim_ttl_seconds:
                        return False  # someone holds a fresh claim
                except Exception:
                    pass  # malformed claim → overwrite
            if force:
                job["enabled"] = True
                job["state"] = "scheduled"
                job["paused_at"] = None
                job["paused_reason"] = None
            # Per-acquisition token: a process may legitimately reclaim its own
            # stale lease, and the previous runner must not heartbeat the new
            # claim merely because hostname + PID are unchanged.
            owner = f"{_machine_id()}:{uuid.uuid4().hex}"
            job["fire_claim"] = {"at": now.isoformat(), "by": owner}
            kind = job.get("schedule", {}).get("kind")
            if kind in {"cron", "interval"}:
                nxt = compute_next_run(job["schedule"], now.isoformat())
                if nxt:
                    job["next_run_at"] = nxt
            save_jobs(jobs)
            return copy.deepcopy(job) if return_job else True
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


def heartbeat_fire_claim(job_id: str, *, expected_owner: str) -> Optional[bool]:
    """Confirm or refresh an active ``fire_claim`` for ``expected_owner``.

    Returns tri-state so callers can distinguish a *positively observed* loss
    from a transient failure to verify:

    - ``True`` — the claim is still owned by ``expected_owner`` and was
      refreshed.
    - ``False`` — ownership is positively observed to have changed (the claim
      is absent or belongs to another owner). Callers may interrupt the run.
    - ``None`` — ownership could NOT be verified (the per-job fire fence could
      not be acquired within ``_JOBS_LOCK_TIMEOUT_SECONDS``). This is NOT
      evidence of loss: under memory/CPU starvation a healthy owner can fail
      to reach the lock. Callers should treat ``None`` as "unknown" and feed
      the existing grace path, not the immediate-kill path.
    """
    with _fire_job_lock(job_id) as acquired:
        if not acquired:
            return None
        return _heartbeat_fire_claim_locked(
            job_id,
            expected_owner=expected_owner,
        )


def _heartbeat_fire_claim_locked(job_id: str, *, expected_owner: str) -> bool:
    """Refresh an active ``fire_claim`` without extending another owner's lease.

    A cron execution can legitimately outlive the fire-claim TTL.  The shared
    run wrapper calls this periodically so another scheduler process cannot
    treat a live execution as abandoned and dispatch it again.  Comparing the
    owner copied at dispatch prevents a stale runner from refreshing a claim
    that has since been recovered by another process.
    """
    with _jobs_lock():
        jobs = load_jobs()
        for job in jobs:
            if job.get("id") != job_id:
                continue
            claim = job.get("fire_claim")
            if not isinstance(claim, dict) or claim.get("by") != expected_owner:
                return False
            claim["at"] = _hermes_now().isoformat()
            save_jobs(jobs)
            return True
    return False


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
    with _jobs_lock():
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

    # Normalize malformed "schedule" records (direct jobs.json edit, old writers,
    # corruption, etc.). "schedule" must be a dict; a null/string/etc. value
    # makes `schedule.get("kind")` or direct `schedule["kind"]` / ["expr"] /
    # ["minutes"] later raise and abort the entire scan *before* save_jobs().
    # Healthy jobs then lose their fast-forwarded next_run_at (exactly the
    # failure mode of the id-less job bug fixed above). Repair early at the
    # source so the rest of the tick can proceed and persist progress for
    # siblings.
    for j in jobs:
        if not isinstance(j.get("schedule"), dict):
            j["schedule"] = {}
            needs_save = True
    for rj in raw_jobs:
        if not isinstance(rj.get("schedule"), dict):
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
            if is_terminal_job(job) and not _is_recoverable_error_job(job):
                continue
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
            if existing_claim and job.get("schedule", {}).get("kind") == "once":
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
            # Intentionally string-exact (raw stored values, not normalized
            # datetimes): trigger_job stamps the SAME isoformat string into
            # both fields, and any rewrite of next_run_at (schedule edit,
            # recovery re-anchor, fire-claim advance) must invalidate the
            # marker. Do not "fix" this with _ensure_aware normalization.
            manual_run = job.get("manual_run_at") == next_run
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
                and not manual_run
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

            # Persisted-state stale-error recovery (t_8b5480b3): a recurring
            # job whose persisted state shows last_status=error and whose
            # last_run_at is older than a full cadence has been sitting wedged
            # (errored once, next_run_at re-armed into the future by
            # mark_job_run, but never re-dispatched — the restart-surviving
            # half of the 2026-08-14 incident, invisible to the in-memory
            # in-flight sweep). Re-arm so the job re-dispatches WITHOUT
            # force-run/resume:
            #   * interval jobs re-arm to now — interval schedules have no
            #     excluded times, so an immediate catch-up retry is always a
            #     legal fire (the 2026-08-14 incident jobs were intervals);
            #   * cron jobs re-arm to the next LEGAL occurrence from now —
            #     re-arming to now would fire at times the expression
            #     explicitly excludes (e.g. a weekday-only 9am job whose
            #     Friday run errored must fire Monday 9am, not Saturday).
            #     This still repairs values parked beyond the next legal
            #     occurrence; a correctly-parked cron value is left as-is.
            if (
                kind in ("cron", "interval")
                and next_run_dt > now
                and _job_is_stale_error_recurring(job, schedule, now)
            ):
                jid = job.get("id")
                if kind == "interval":
                    recovered_next = now.isoformat()
                    recovered_next_dt = now
                else:
                    recovered_next = compute_next_run(schedule, now.isoformat())
                    try:
                        recovered_next_dt = (
                            _ensure_aware(datetime.fromisoformat(recovered_next))
                            if recovered_next
                            else None
                        )
                    except (ValueError, TypeError):
                        recovered_next_dt = None
                if recovered_next and recovered_next_dt is not None and recovered_next_dt < next_run_dt:
                    logger.warning(
                        "cron.persisted_error.recovered job='%s' id=%s — recurring "
                        "job wedged in stale last_status=error without re-firing for "
                        "a full cadence; re-arming next_run_at to %s so it "
                        "re-dispatches without force-run/resume",
                        job.get("name", jid),
                        jid,
                        recovered_next,
                    )
                    _record_persisted_error_recovery(job, next_run)
                    job["next_run_at"] = recovered_next
                    next_run_dt = recovered_next_dt
                    for rj in raw_jobs:
                        if rj["id"] == jid:
                            rj["next_run_at"] = recovered_next
                            needs_save = True
                            break

            if next_run_dt <= now:

                # Stale-schedule guard (#93049): a direct jobs.json edit that
                # changed schedule.expr leaves next_run_at computed under the
                # old expression, so the job would fire at an instant the
                # current expression excludes. Both the within-grace fire and
                # the catch-up "run once now" below inherit that wrong instant,
                # so re-anchor before either can fire. Recomputation uses the
                # current expression, so this converges — it cannot defer
                # forever.
                #
                # Not every mismatch is an edit, though: an offset-representation
                # migration (UTC-scheduling build -> profile-timezone build)
                # moves a legacy instant off the lattice without the expression
                # ever changing, and re-anchoring THAT silently swallowed a due
                # occurrence. Classify first, and only the edit case skips.
                stale_class = (
                    _classify_stale_cron_next_run(schedule, raw_next_run_dt, next_run_dt)
                    if not manual_run and kind == "cron"
                    else STALE_CRON_MATCH
                )
                if stale_class == STALE_CRON_EXPR_EDIT:
                    new_next = compute_next_run(schedule, now.isoformat())
                    logger.info(
                        "Job '%s' next_run_at %s does not match its current "
                        "cron expression %r (direct jobs.json edit?); "
                        "re-anchoring to %s without firing.",
                        job.get("name", job.get("id", "?")),
                        next_run,
                        schedule.get("expr"),
                        new_next,
                    )
                    if new_next:
                        for rj in raw_jobs:
                            if rj["id"] == job["id"]:
                                rj["next_run_at"] = new_next
                                needs_save = True
                                break
                    continue
                if stale_class == STALE_CRON_TIMEZONE_MIGRATION:
                    # Fall through to the normal due path: the occurrence is
                    # real and overdue, so it fires ONCE here and the usual
                    # advance/mark_job_run re-anchor writes the value back in
                    # the current offset. At-most-once is preserved because
                    # nothing re-reads the legacy instant after that.
                    logger.warning(
                        "cron.timezone_migration.catch_up job='%s' id=%s expr=%r "
                        "stored=%s normalized=%s — stored next_run_at carries a "
                        "pre-migration UTC offset (%s, now %s) and is a legal "
                        "occurrence at its own wall clock; firing the due run "
                        "instead of re-anchoring past it.",
                        job.get("name", job.get("id", "?")),
                        job.get("id"),
                        schedule.get("expr"),
                        next_run,
                        next_run_dt.isoformat(),
                        raw_next_run_dt.utcoffset(),
                        now.utcoffset(),
                    )
                    _record_timezone_migration_catchup(job, raw_next_run_dt, next_run_dt)

                # For recurring jobs, check if the scheduled time is stale
                # (gateway was down and missed the window). Fast-forward to
                # the next future occurrence instead of firing a stale run.
                grace = _compute_grace_seconds(schedule)
                if (
                    not manual_run
                    and kind in {"cron", "interval"}
                    and (now - next_run_dt).total_seconds() > grace
                ):
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

                # One-shot grace gate: a one-shot whose persisted run time is
                # beyond the grace window must never fire. create_job /
                # update_job / resume_job all reject such schedules ("will
                # never fire") and _recoverable_oneshot_run_at never recovers
                # them — only the due-scan acted differently and dispatched a
                # wall-clock one-shot hours late (gateway down past the
                # window, host asleep, hand-edited jobs.json).
                if kind == "once" and (now - next_run_dt).total_seconds() > ONESHOT_GRACE_SECONDS:
                    if not (job.get("run_claim") or job.get("fire_claim")):
                        # Nothing was ever dispatched — retire the record with
                        # a diagnostic so it stops being scanned and the miss
                        # is operator-visible (retire-not-silently-delete).
                        _write_missed_oneshot_diagnostic(job, next_run)
                        for rj in raw_jobs:
                            if rj["id"] == job["id"]:
                                raw_jobs.remove(rj)
                                intentionally_removed.add(str(job["id"]))
                                needs_save = True
                                break
                    # A (possibly stale) claim may mean a run is still in
                    # flight in another process — skip this scan but keep the
                    # record so its mark_job_run can still land.
                    continue

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
                            if job.get("last_run_at") is not None:
                                # A record with last_run_at completed a real
                                # run and was later re-armed without a budget
                                # reset (e.g. a schedule edit before the
                                # #93524 fix, or a hand-edited store). This is
                                # NOT the dead-tick recovery case this guard
                                # was built for, and the wedged-oneshot
                                # diagnostic below will (correctly) not fire
                                # — so removing it silently at INFO would
                                # vanish the user's rescheduled run without a
                                # trace. Make it operator-visible.
                                logger.warning(
                                    "Job '%s': one-shot dispatch limit reached "
                                    "(%d/%d) on a record that already completed "
                                    "a run (last_run_at=%s) — removing it "
                                    "WITHOUT firing. This record was re-armed "
                                    "without a budget reset (pre-#93615 store "
                                    "or hand edit); re-run it with "
                                    "'hermes cron resume <job> --run-now' "
                                    "(#93524).",
                                    job.get("name", job.get("id", "?")),
                                    completed,
                                    times,
                                    job.get("last_run_at"),
                                )
                            else:
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
                if kind == "once":
                    claim = {"at": now.isoformat(), "by": _machine_id()}
                    job["run_claim"] = claim
                    for rj in raw_jobs:
                        if rj["id"] == job["id"]:
                            rj["run_claim"] = claim
                            needs_save = True
                            break

                # Missed-run visibility (#99879): persist scheduled-vs-actual
                # dispatch timing on the job record so `hermes cron list` /
                # `hermes cron status` (separate CLI processes) can show a
                # late catch-up run as such instead of an ordinary on-time
                # run. Recurring schedules only — one-shots beyond grace are
                # retired above, and manual triggers have no scheduled
                # instant to be late against.
                if not manual_run and kind in {"cron", "interval"}:
                    lateness = max(0.0, (now - next_run_dt).total_seconds())
                    dispatch_stamp = {
                        "scheduled_at": next_run,
                        "dispatched_at": now.isoformat(),
                        "lateness_seconds": round(lateness, 1),
                        "kind": _classify_dispatch_lateness(lateness, grace),
                    }
                    job["last_dispatch"] = dispatch_stamp
                    for rj in raw_jobs:
                        if rj["id"] == job["id"]:
                            rj["last_dispatch"] = dispatch_stamp
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
    _ensure_cron_dir(job_output_dir)
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
