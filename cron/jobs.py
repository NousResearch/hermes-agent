"""Cron job storage: ~/.hermes/cron/jobs.json; output in
~/.hermes/cron/output/{job_id}/{timestamp}.md"""

import contextlib
import copy
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import logging
import shutil
import tempfile
import threading
import time
import os
import re
import uuid

# Cross-process advisory locking for jobs.json: fcntl (Unix) or msvcrt (Windows). If both are
# absent, _jobs_lock() degrades to in-process locking rather than failing.
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
from typing import Optional, Dict, List, Any, Callable, Set, Tuple, Union, Collection

logger = logging.getLogger(__name__)

from hermes_time import now as _hermes_now
from utils import atomic_replace, atomic_write_text

# croniter is imported lazily (slow import, only needed for cron exprs). HAS_CRONITER stays a
# module attribute: a monkeypatched value wins because _ensure_croniter only probes while None.
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


# --- Configuration ---

# Cron is per-profile by design: anchor at get_hermes_home() (active profile home), NOT
# get_default_hermes_root() — the shared root would funnel every profile's jobs into one jobs.json
# and run them under the ticker's HERMES_HOME, leaking config/credentials/skills across profiles.
# Each profile owns its own cron store under its own HERMES_HOME, and a profile-scoped gateway runs that
# profile's jobs under that same HERMES_HOME — so a job authored in profile `coder` lives in
# `~/.hermes/profiles/coder/cron/jobs.json` and executes with `coder`'s `.env`, `config.yaml`, and skills.
# Do NOT change this to the default root: that re-breaks per-profile isolation. See also the dynamic
# `_get_hermes_home()` / `_get_lock_paths()` resolution in cron/scheduler.py. See #4707.
HERMES_DIR = get_hermes_home().resolve()
# Default-profile fallback and compatibility surface for callers/tests. Cross-profile callers must
# scope paths with use_cron_store() instead of mutating these process-wide.
CRON_DIR = HERMES_DIR / "cron"
JOBS_FILE = CRON_DIR / "jobs.json"
# Heartbeat: touched every ticker loop so `hermes cron status` can tell the ticker THREAD is alive,
# not just the gateway PROCESS; success = last tick that completed WITHOUT raising.
# The gateway process and the (separate) ``hermes cron status`` process share it so status can tell whether
# the ticker THREAD is alive, not just whether the gateway PROCESS exists — a ticker that dies silently
# inside a live gateway would otherwise report healthy (#32612, #32895).
TICKER_HEARTBEAT_FILE = CRON_DIR / "ticker_heartbeat"
TICKER_SUCCESS_FILE = CRON_DIR / "ticker_last_success"
# Single source of truth for the ticker interval (scheduler_provider.py) and the staleness
# threshold in `hermes cron status` (hermes_cli/cron.py), so they never drift apart.
TICKER_INTERVAL_SECONDS = 60

# In-process lock for load_jobs→modify→save_jobs cycles; without it, parallel tick threads'
# mark_job_run / advance_next_run calls clobber each other.
_jobs_file_lock = threading.RLock()
_jobs_lock_state = threading.local()
_fire_fence_locks: Dict[str, threading.RLock] = {}
_fire_fence_locks_guard = threading.Lock()
_fire_fence_lock_state = threading.local()

# Upper bound on waiting for the cross-process .jobs.lock. Every cron function funnels through
# _jobs_lock(), so blocking forever on a wedged sibling process would freeze the ticker and every
# job. 30s is far above any legitimate critical section yet under one status-alarm threshold.
_JOBS_LOCK_TIMEOUT_SECONDS = 30.0
OUTPUT_DIR = CRON_DIR / "output"
ONESHOT_GRACE_SECONDS = 120


@dataclass(frozen=True)
class _CronStorePaths:
    cron_dir: Path
    jobs_file: Path
    output_dir: Path

    @classmethod
    def for_dir(cls, cron_dir: Path) -> "_CronStorePaths":
        return cls(cron_dir, cron_dir / "jobs.json", cron_dir / "output")


_cron_store_override: ContextVar[Optional[_CronStorePaths]] = ContextVar(
    "cron_store_override", default=None)

# Import-time snapshot so deliberate re-pointing of CRON_DIR/JOBS_FILE/OUTPUT_DIR (the documented
# escape hatch for tests/embedders) is distinguishable from the constants merely being stale.
_IMPORT_STORE = _CronStorePaths(CRON_DIR, JOBS_FILE, OUTPUT_DIR)


def _current_cron_store() -> _CronStorePaths:
    """Paths pinned to this execution context's profile. Precedence: (1) active use_cron_store()
    override; (2) deliberately re-pointed module constants; (3) the ACTIVE profile home via
    get_hermes_home(), so re-pointing HERMES_HOME after import uses ITS OWN store rather than the
    user's real jobs.json frozen at import; (4) import-time constants."""
    override = _cron_store_override.get()
    if override is not None:
        return override
    live_constants = _CronStorePaths(CRON_DIR, JOBS_FILE, OUTPUT_DIR)
    if live_constants != _IMPORT_STORE:
        return live_constants
    home = get_hermes_home().resolve()
    if home == HERMES_DIR:
        return live_constants
    return _CronStorePaths.for_dir(home / "cron")


@contextlib.contextmanager
def use_cron_store(home: Union[str, Path]):
    """Route cron storage to ``home`` without mutating process globals."""
    token = _cron_store_override.set(
        _CronStorePaths.for_dir(Path(home).expanduser().resolve() / "cron"))
    try:
        yield
    finally:
        _cron_store_override.reset(token)


def get_cron_output_dir() -> Path:
    """Return the output directory for the active cron store context."""
    return _current_cron_store().output_dir


# Fallback stale-recovery window for a one-shot's running-claim when HERMES_CRON_TIMEOUT=0
# (unlimited, no bound to derive from); also the floor so a tiny timeout can't expire a claim
# mid-run.
ONESHOT_RUN_CLAIM_TTL_SECONDS = 1800

# Derived TTL = inactivity timeout × this headroom. The TTL only recovers a claim left by a tick
# that DIED mid-run; the timeout is an *inactivity* limit, not a wall-clock cap, so healthy runs may
# legitimately exceed it — hence the headroom.
_ONESHOT_RUN_CLAIM_TTL_HEADROOM = 3

_DEFAULT_CRON_INACTIVITY_TIMEOUT = 600.0


def _oneshot_run_claim_ttl_seconds() -> float:
    """One-shot running-claim TTL from ``HERMES_CRON_TIMEOUT``: unset/invalid → 600s → 1800s;
    ``0`` (unlimited) → the fixed floor; positive N → ``max(N * headroom, floor)``."""
    raw = os.getenv("HERMES_CRON_TIMEOUT", "").strip()
    try:
        timeout = float(raw) if raw else _DEFAULT_CRON_INACTIVITY_TIMEOUT
    except (ValueError, TypeError):
        timeout = _DEFAULT_CRON_INACTIVITY_TIMEOUT
    if timeout <= 0:
        return float(ONESHOT_RUN_CLAIM_TTL_SECONDS)
    return max(timeout * _ONESHOT_RUN_CLAIM_TTL_HEADROOM, float(ONESHOT_RUN_CLAIM_TTL_SECONDS))


def _job_running_in_this_process(job_id: str) -> bool:
    """True when the scheduler in THIS process is still running ``job_id``: the run_claim TTL alone
    cannot distinguish "claiming tick died" from "alive but slow". Lazy import: scheduler imports
    us.

    Direct liveness signal for stale-entry recovery (#62002): the run_claim TTL alone cannot distinguish
    "the claiming tick died" from "the run is alive but slow" — a run stalled on network I/O (or a laptop
    that slept mid-run) legitimately outlives the TTL. The in-process ticker and the run share this process,
    so the scheduler's running set settles the common single-gateway case without any claim-age guesswork.
    """
    try:
        from cron.scheduler import get_running_job_ids
        return job_id in get_running_job_ids()
    except Exception:
        logger.warning(
            "Cron running-set liveness check failed for job %r; keeping the "
            "entry to avoid deleting a possibly live one-shot run",
            job_id, exc_info=True)
        return True


def _jobs_lock_file() -> Path:
    """Return the advisory lock path for the current cron directory."""
    return _current_cron_store().cron_dir / ".jobs.lock"


def _acquire_flock(lock_fd, timeout: float) -> Optional[bool]:
    """Bounded exclusive lock: True when acquired, False on timeout, None when no backend exists. A
    blocking flock(LOCK_EX) taken under the in-process lock would let a wedged sibling freeze
    EVERY cron function forever, so poll LOCK_NB against a deadline; the caller picks the
    degraded mode."""
    if fcntl is not None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except (OSError, IOError):
                if time.monotonic() >= deadline:
                    return False
                time.sleep(0.1)
    if msvcrt is not None:
        getattr(msvcrt, "locking")(lock_fd.fileno(), getattr(msvcrt, "LK_LOCK"), 1)
        return True
    return None


def _release_flock(lock_fd) -> None:
    """Unlock (best effort) and close a lock file opened for ``_acquire_flock``."""
    try:
        if fcntl is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        elif msvcrt is not None:
            getattr(msvcrt, "locking")(lock_fd.fileno(), getattr(msvcrt, "LK_UNLCK"), 1)
    except (OSError, IOError):
        pass
    finally:
        lock_fd.close()


@contextlib.contextmanager
def _jobs_lock():
    """Serialize a load_jobs→modify→save_jobs critical section: in-process RLock (parallel tick
    threads) plus a cross-process flock on ``<cron dir>/.jobs.lock`` (gateway vs. CLI writes —
    otherwise a `cron pause` could be clobbered and keep firing). Nested calls in one thread
    reuse the held lock. Without a flock backend, or on flock timeout (logged loudly), it
    degrades to in-process-only locking: a briefly torn cross-process write beats a dead
    scheduler."""
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
        # jobs.json stamp as of this section's load_jobs(): lets _save_jobs_unlocked skip the
        # shrink-merge parse when the file provably hasn't changed. Reset on entry/exit so stale
        # stamps from unlocked loads or prior sections can never suppress a needed merge.
        # See #80703.
        _jobs_lock_state.load_stamp = None
        lock_fd = None
        try:
            try:
                ensure_dirs()
                lock_fd = open(_jobs_lock_file(), "a+", encoding="utf-8")
                lock_fd.seek(0)
                if _acquire_flock(lock_fd, _JOBS_LOCK_TIMEOUT_SECONDS) is False:
                    logger.error(
                        "Timed out after %.0fs waiting for the cron "
                        "jobs lock (%s) — another process is holding "
                        "it. Proceeding with in-process locking only "
                        "so the scheduler stays alive (#60703).",
                        _JOBS_LOCK_TIMEOUT_SECONDS, _jobs_lock_file())
                    with contextlib.suppress(OSError):
                        lock_fd.close()
                    lock_fd = None
            except (OSError, IOError) as e:
                # A locking failure must never take down cron writes — in-process lock still held.
                logger.warning("jobs.json cross-process lock unavailable (%s); "
                               "proceeding with in-process lock only", e)
            try:
                yield
            finally:
                if lock_fd is not None:
                    _release_flock(lock_fd)
        finally:
            _jobs_lock_state.depth = 0
            _jobs_lock_state.load_stamp = None


@contextlib.contextmanager
def _fire_job_lock(job_id: str):
    """Serialize one job's owner mutations and external side effects. Unlike the global jobs lock
    this may be held across network delivery; scoped to one profile + job so unrelated jobs keep
    progressing. Fails closed when cross-process locking is unavailable."""
    cron_dir = _current_cron_store().cron_dir
    lock_key = f"{cron_dir.resolve()}::{job_id}"
    with _fire_fence_locks_guard:
        local_lock = _fire_fence_locks.setdefault(lock_key, threading.RLock())

    if not local_lock.acquire(timeout=_JOBS_LOCK_TIMEOUT_SECONDS):
        logger.error("Timed out waiting for local fire fence %s; failing closed", lock_key)
        yield False
        return

    held_locks = _fire_fence_lock_state.__dict__.setdefault("held", {})
    if lock_key in held_locks:
        try:
            yield held_locks[lock_key]
        finally:
            local_lock.release()
        return

    try:
        ensure_dirs()
        lock_path = cron_dir / f".fire-{uuid.uuid5(uuid.NAMESPACE_URL, lock_key).hex}.lock"
        lock_fd = None
        acquired = False
        try:
            lock_fd = open(lock_path, "a+", encoding="utf-8")
            lock_fd.seek(0)
            result = _acquire_flock(lock_fd, _JOBS_LOCK_TIMEOUT_SECONDS)
            if result is None:  # pragma: no cover - supported platforms provide one backend
                logger.error("No cross-process lock backend for cron fire fence")
            elif not result:
                logger.error("Timed out waiting for fire fence %s; failing closed", lock_path)
            acquired = bool(result)
        except (OSError, IOError) as exc:
            logger.error("Cron fire fence unavailable for %s: %s", job_id, exc)

        held_locks[lock_key] = acquired
        try:
            yield acquired
        finally:
            held_locks.pop(lock_key, None)
            if lock_fd is not None:
                if acquired:
                    _release_flock(lock_fd)
                else:
                    lock_fd.close()
    finally:
        local_lock.release()


def _under_fire_fence(job_id: str, fn: Callable[[], Any]) -> Any:
    """Run ``fn()`` holding the job's fire fence; False (fail closed) when it can't be acquired."""
    with _fire_job_lock(job_id) as acquired:
        if not acquired:
            return False
        return fn()


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
            owns_claim = isinstance(claim, dict) and claim.get("by") == expected_owner
        yield owns_claim


# Fields that must never change after creation: ``id`` is a path component under OUTPUT_DIR, so an
# update could leak ``../escape``/absolute/nested values into output writes/deletes.
_IMMUTABLE_JOB_FIELDS = frozenset({"id"})


def _job_output_dir(job_id: str) -> Path:
    """Resolve a job's output directory, rejecting any path-escape attempt (``..``, absolute
    paths, separators): only a single safe path component is accepted."""
    text = str(job_id or "").strip()
    if (
        not text or text in {".", ".."} or "/" in text or "\\" in text
        or Path(text).is_absolute() or Path(text).drive
    ):
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
    return fallback if value is None else str(value)


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
    """True when a job record has nothing runnable (blank prompt, no script, no skills) AND at
    least one payload field is explicitly present. ``no_agent`` already requires a script."""
    if _coerce_job_text(job.get("prompt")).strip() or _coerce_job_text(job.get("script")).strip():
        return False
    if _coerce_job_text(job.get("prompt_file")).strip():
        return False
    if _normalize_skill_list(job.get("skill"), job.get("skills")):
        return False
    return any(k in job for k in ("prompt", "script", "skill", "skills", "prompt_file"))


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
    """Read-safe job shape: legacy/hand-edited records may have nullable ``prompt``, ``name``,
    ``schedule_display``. Storage is untouched; consumers never crash on formatting."""
    normalized = _apply_skill_fields(job)
    job_id = normalized["id"] = _coerce_job_text(normalized.get("id"), "unknown")
    prompt = normalized["prompt"] = _coerce_job_text(normalized.get("prompt"))
    name = _coerce_job_text(normalized.get("name")).strip()
    if not name:
        label_source = (
            prompt
            or (normalized["skills"][0] if normalized.get("skills") else "")
            or _coerce_job_text(normalized.get("script")).strip()
            or job_id
            or "cron job"
        )
        name = label_source[:50].strip() or "cron job"
    normalized["name"] = name
    normalized["schedule_display"] = _schedule_display_for_job(normalized)
    # Derived from the scheduler-honoured ``enabled`` flag so a half-paused record cannot render
    # "paused" while still firing. See effective_job_state().
    normalized["state"] = effective_job_state(normalized)
    return normalized


def _has_pause_marker(job: Dict[str, Any]) -> bool:
    """True when the record carries any operator-facing pause signal."""
    return _coerce_job_text(job.get("state")).strip() == "paused" or bool(job.get("paused_at"))


def is_job_runnable(job: Dict[str, Any]) -> bool:
    """True iff the scheduler may fire this job: ``enabled`` plus pause markers as a second gate so
    a contradictory half-paused record never fires even before self-heal runs."""
    return bool(job.get("enabled", True)) and not _has_pause_marker(job)


def effective_job_state(job: Dict[str, Any]) -> str:
    """Operator-facing state derived from ``enabled``: an enabled job must never display as paused
    (list looked frozen while jobs kept firing). Terminal states are preserved regardless."""
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
    """True for a recurring job stuck in ``state=error`` (set ONLY when ``compute_next_run()`` fails
    for a cron/interval job: croniter missing, malformed schedule). Such a job still has future
    occurrences once the issue resolves, so treating it as terminal would block due-scan self-heal,
    pre-advance, dispatch claim and ``resume_job`` — wedging it forever.

    Unlike ``state=completed`` (a one-shot that genuinely has no more occurrences, ever), an error-state
    recurring job still has a schedule with future occurrences once the underlying issue resolves — it is
    stuck pending a ``next_run_at`` recompute, not truly done. See #16265.
    """
    return (
        job.get("state") == "error"
        and (job.get("schedule") or {}).get("kind") in {"cron", "interval"}
    )


def _secure_dir(path: Path):
    """Set directory to owner-only access (0700). No-op where chmod is unsupported (Windows)."""
    with contextlib.suppress(OSError, NotImplementedError):
        os.chmod(path, 0o700)


def _secure_file(path: Path):
    """Set file to owner-only read/write (0600). No-op where chmod is unsupported (Windows)."""
    with contextlib.suppress(OSError, NotImplementedError):
        if path.exists():
            os.chmod(path, 0o600)


def _preserve_file_ownership(path: Path, before: Optional[os.stat_result]) -> None:
    """Restore a rewritten file's previous owner (POSIX, root writer only): atomic replace makes the
    file owned by the writer's euid, so a root CLI write (e.g. ``docker exec``) against the
    unprivileged gateway's store would flip jobs.json to root:root 0600 and lock the ticker out."""
    if before is None or os.name != "posix":
        return
    geteuid = getattr(os, "geteuid", None)
    getegid = getattr(os, "getegid", None)
    if geteuid is None or getegid is None:
        return
    try:
        euid = geteuid()
        if euid != 0 or (before.st_uid, before.st_gid) == (euid, getegid()):
            return  # unprivileged writer, or already ours before the rewrite
        os.chown(path, before.st_uid, before.st_gid)
    except OSError as e:
        logger.warning(
            "Could not restore ownership of %s to uid=%s gid=%s after rewrite: %s "
            "— if the gateway runs as a different user, its cron ticker may now "
            "be locked out (see issue #68483).",
            path, before.st_uid, before.st_gid, e)


def _is_named_profile_path(path: Path) -> bool:
    """True if *path* is under ``<hermes_home>/profiles/<name>/`` (default/custom homes are not).
    Checks the resolved path (symlinked parents) and the raw path (symlinked profile homes)."""
    with contextlib.suppress(OSError, RuntimeError):
        if "profiles" in path.resolve().parts:
            return True
    return "profiles" in path.parts


def _ensure_cron_dir(cron_dir: Path) -> None:
    """Create a cron directory without resurrecting a deleted profile home: a stale multiplex
    scheduler may still hold a deleted profile's path, so named profiles use ``parents=False`` and
    fail closed. Default/custom homes keep ``parents=True`` so first-run creation works."""
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


# --- Schedule Parsing ---

def normalize_repeat_value(repeat: Any) -> Optional[int]:
    """Coerce a repeat value (int or user-facing string) into ``Optional[int]``:
    ``'forever'``-family -> None, ``'once'``-family -> 1, numeric -> int, 0/negative -> None,
    else ValueError.

    The tool schema exposes ``repeat`` as an integer, but agents and users legitimately pass the user-facing
    strings ``'forever'``/``'once'`` or numeric strings (``'3'``). Uncoerced strings previously died with
    ``'<=' not supported between instances of 'str' and 'int'`` at create
    (#66824/#64520/#7142/#71987/#95706) and were stored raw by update paths, breaking ``mark_job_run``
    later.
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


_DURATION_MULTIPLIERS = {'m': 1, 'h': 60, 'd': 1440}


def parse_duration(s: str) -> int:
    """Parse a duration into minutes: "30m" → 30, "2h" → 120, "1d" → 1440, bare "hour" → 60."""
    s = s.strip().lower()
    match = re.match(r'^(\d*)\s*(m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)$', s)
    if not match:
        raise ValueError(
            f"Invalid duration: '{s}'. Use format like '30m', '2h', '1d', "
            "or a bare unit like 'hour' (defaults to 1).")
    value = int(match.group(1)) if match.group(1) else 1
    return value * _DURATION_MULTIPLIERS[match.group(2)[0]]


# Day-spec phrases for "every monday 9am" / "every day at 9am". Cron weekday numbering is
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
    """Parse ``9am``/``9:30am``/``14:00``/``7`` (bare 24h hour)/``noon``/``midnight`` into a
    24-hour ``(hour, minute)`` tuple, or None when unrecognized."""
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
        hour = hour % 12 + (12 if meridiem == "pm" else 0)
    if hour > 23 or minute > 59:
        return None
    return (hour, minute)


def _natural_every_to_cron(rest: str) -> Optional[str]:
    """Convert ``<when> [at] <time>`` ("monday 9am", "weekday at 9am", "monday, wednesday at 9am")
    to a 5-field cron expr, or None so ``parse_schedule`` can fall back to the interval path."""
    tokens = rest.lower().replace(",", " ").split()
    if not tokens:
        return None
    # Leading day tokens: a keyword spec ("weekdays") or a comma/"and"-separated weekday list.
    dow = _DAYSPEC_TO_CRON_DOW.get(tokens[0])
    idx = 1
    if dow is None:
        days = []
        idx = len(tokens)
        for i, tok in enumerate(tokens):
            if tok == "and":
                continue
            mapped = _WEEKDAY_TO_CRON_DOW.get(tok)
            if mapped is None:
                idx = i
                break
            if mapped not in days:
                days.append(mapped)
        if not days:
            return None
        dow = ",".join(days)
    time_tokens = tokens[idx:]
    if time_tokens and time_tokens[0] == "at":  # optional separator: "every day at 9am"
        time_tokens = time_tokens[1:]
    if not time_tokens:
        return None
    parsed = _parse_clock_time(" ".join(time_tokens))
    if parsed is None:
        return None
    hour, minute = parsed
    return f"{minute} {hour} * * {dow}"


def _cron_schedule(
    expr: str, display: str, missing_croniter: str, invalid_label: str
) -> Dict[str, Any]:
    """Validate a cron expression with croniter and build the stored schedule dict."""
    if not _ensure_croniter():
        raise ValueError(f"{missing_croniter} Install with: pip install croniter")
    try:
        croniter(expr)
    except Exception as e:
        raise ValueError(f"Invalid {invalid_label} '{display}': {e}")
    return {"kind": "cron", "expr": expr, "display": display}


def _interval_schedule(minutes: int) -> Dict[str, Any]:
    return {"kind": "interval", "minutes": minutes, "display": f"every {minutes}m"}


def parse_schedule(schedule: str) -> Dict[str, Any]:
    """Parse a schedule string into ``{"kind": "once"|"interval"|"cron", ...}`` with ``run_at`` /
    ``minutes`` / ``expr``. "30m" and "every 30m" are recurring intervals; "every monday 9am" and
    "0 9 * * *" are cron; an ISO timestamp is once."""
    schedule = schedule.strip()
    original = schedule
    schedule_lower = schedule.lower()

    # Natural day/time phrase → cron ("every monday 9am", or sans prefix "weekdays at 9am");
    # any other "every X" → recurring interval.
    is_every = schedule_lower.startswith("every ")
    rest = schedule[6:].strip() if is_every else schedule_lower
    cron_expr = _natural_every_to_cron(rest)
    # Reuse the same helper — the phrase shape is identical without the "every " prefix. See #51975.
    if cron_expr is not None:
        example = "every monday 9am" if is_every else "weekdays at 9am"
        return _cron_schedule(
            cron_expr, original,
            f"Weekday/time schedules like '{example}' require the 'croniter' package.", "schedule")
    if is_every:
        return _interval_schedule(parse_duration(rest))

    # Cron expression (5-6 fields). Letters are allowed so named months/weekdays (JAN-DEC, MON-FRI)
    # reach croniter, which supports them.
    parts = schedule.split()
    if len(parts) >= 5 and all(re.match(r'^[A-Za-z\d\*\-,/]+$', p) for p in parts[:5]):
        return _cron_schedule(
            schedule, schedule, "Cron expressions require 'croniter' package.", "cron expression")

    # ISO timestamp (contains T or looks like date)
    if 'T' in schedule or re.match(r'^\d{4}-\d{2}-\d{2}', schedule):
        try:
            dt = datetime.fromisoformat(schedule.replace('Z', '+00:00'))
            # Naive timestamps become aware in the CONFIGURED Hermes timezone (not server-local):
            # the due-check compares against hermes_time.now().
            # Make naive timestamps timezone-aware at parse time so the stored value doesn't depend on the
            # system timezone matching at check time. UTC) while now() runs in Asia/Kolkata, the stored
            # instant would land hours off from the user's wall-clock intent — far enough that one-shots
            # never become due and recurring jobs fire at the wrong time. Using the configured zone makes
            # "20:07" mean 20:07 on the same clock the scheduler checks against (#51021).
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=_hermes_now().tzinfo)
            return {
                "kind": "once",
                "run_at": dt.isoformat(),
                "display": f"once at {dt.strftime('%Y-%m-%d %H:%M')}"
            }
        except ValueError as e:
            raise ValueError(f"Invalid timestamp '{schedule}': {e}")

    # "in 30m"/"in 2h" is the explicit one-shot-by-duration form; a bare duration ("30m") is a
    # RECURRING interval per the documented tool contract.
    if schedule_lower.startswith("in "):
        duration_str = schedule[3:].strip()
        try:
            minutes = parse_duration(duration_str)
        except ValueError:
            raise ValueError(
                f"Invalid duration '{duration_str}' after 'in '. Use e.g. 'in 30m', 'in 2h'.")
        run_at = _hermes_now() + timedelta(minutes=minutes)
        return {"kind": "once", "run_at": run_at.isoformat(), "display": f"once in {duration_str}"}
    with contextlib.suppress(ValueError):
        return _interval_schedule(parse_duration(schedule))

    raise ValueError(
        f"Invalid schedule '{original}'. Use:\n"
        f"  - Interval: '30m', 'every 30m', 'every 2h' (recurring)\n"
        f"  - One-shot delay: 'in 30m', 'in 2h' (fires once)\n"
        f"  - Weekly/daily: 'every monday 9am', 'weekdays at 9am' (recurring)\n"
        f"  - Cron: '0 9 * * *' (cron expression)\n"
        f"  - Timestamp: '2026-02-03T14:00:00' (one-shot at time)"
    )


def _ensure_aware(dt: datetime) -> datetime:
    """Aware datetime in the configured Hermes timezone. Legacy naive values are read as
    *system-local* wall time (what created them) then converted, preserving ordering across
    timezone changes and avoiding false not-due results."""
    target_tz = _hermes_now().tzinfo
    if dt.tzinfo is None:
        return dt.replace(tzinfo=datetime.now().astimezone().tzinfo).astimezone(target_tz)
    return dt.astimezone(target_tz)


def _parse_aware(value: Any) -> Optional[datetime]:
    """``_ensure_aware(datetime.fromisoformat(value))``, or None when *value* is not a parseable ISO
    string."""
    try:
        return _ensure_aware(datetime.fromisoformat(value))
    except Exception:
        return None


def _timezone_offset_mismatch(stored: datetime, current: datetime) -> bool:
    """True when a stored aware timestamp uses a different UTC offset. Naive values return False:
    they are normalized by ``_ensure_aware`` and intentionally never take the offset-repair path."""
    if stored.tzinfo is None or current.tzinfo is None:
        return False
    return stored.utcoffset() != current.utcoffset()


def _stored_wall_clock_is_future(stored: datetime, current: datetime) -> bool:
    """True when the stored local wall-clock time has not arrived yet. Cron expresses wall-clock
    intent; after a timezone change an old offset can make a future run look due (21:00+10 →
    13:00+02). Comparing naive wall clocks separates that from a genuine miss."""
    return stored.replace(tzinfo=None) > current.replace(tzinfo=None)


def _recoverable_oneshot_run_at(
    schedule: Dict[str, Any], now: datetime, *, last_run_at: Optional[str] = None,
) -> Optional[str]:
    """One-shot run time if still eligible: a small grace window covers jobs created just after
    their minute; once run, a one-shot is never eligible again."""
    if not isinstance(schedule, dict) or schedule.get("kind") != "once" or last_run_at:
        return None
    run_at = schedule.get("run_at")
    run_at_dt = _parse_aware(run_at) if run_at else None
    if run_at_dt is not None and run_at_dt >= now - timedelta(seconds=ONESHOT_GRACE_SECONDS):
        return run_at
    return None


_MIN_GRACE_SECONDS = 120
_MAX_GRACE_SECONDS = 7200


def _compute_grace_seconds(schedule: dict) -> int:
    """How late a job can be and still catch up rather than fast-forward: half the period, clamped
    to [120s, 2h], so daily jobs catch up but frequent jobs fast-forward quickly."""
    period_seconds = _schedule_cadence_seconds(schedule)
    if not period_seconds:
        return _MIN_GRACE_SECONDS
    return max(_MIN_GRACE_SECONDS, min(int(period_seconds) // 2, _MAX_GRACE_SECONDS))


# A recurring dispatch within this many seconds of schedule renders "on time": a busy once-a-minute
# ticker can slip a couple of minutes — normal cadence, not gateway downtime.
# See #99879.
_LATE_DISPATCH_TOLERANCE_SECONDS = 300


def _classify_dispatch_lateness(lateness_seconds: float, grace_seconds: int) -> str:
    """``on_time`` (within ticker slack), ``late`` (within the catch-up grace window), or
    ``catch_up`` (beyond grace; accumulated misses skipped, executed once now)."""
    if lateness_seconds > grace_seconds:
        return "catch_up"
    if lateness_seconds > _LATE_DISPATCH_TOLERANCE_SECONDS:
        return "late"
    return "on_time"


# Recovery counter for recurring jobs wedged in stale ``last_status == "error"`` with a future
# next_run_at.
_persisted_error_recoveries: int = 0
# Bounded in-memory history kept by every probe-visible fire-path counter.
_TELEMETRY_RECENT_HISTORY = 20
_persisted_error_recoveries_recent: list = []


def _job_is_stale_error_recurring(
    job: Dict[str, Any], schedule: Dict[str, Any], now: datetime,
) -> bool:
    """True when a recurring job (caller-checked) is wedged in a stale persisted error state:
    ``last_status == "error"``, not running in this process (never re-arm a live run underneath
    itself), and ``last_run_at`` older than ``cadence + grace`` (a job merely erroring-and-retrying
    on schedule stays fresh and is not flagged).

    Condition (all must hold): * it has NOT successfully re-fired within its natural cadence — its
    ``last_run_at`` is older than ``cadence + grace``, so this is not a normal transient-error retry that
    will fire on its own soon, it is a job that has been sitting errored for a full period with no recovery;
    See #62002.
    """
    if job.get("last_status") != "error":
        return False
    if _job_running_in_this_process(str(job.get("id") or "")):
        return False
    last_run = job.get("last_run_at")
    last_run_dt = _parse_aware(last_run) if last_run else None
    if last_run_dt is None:
        return False
    age_seconds = (now - last_run_dt).total_seconds()
    if age_seconds < 0:
        return False
    grace = _compute_grace_seconds(schedule)
    cadence_seconds = _schedule_cadence_seconds(schedule)
    if cadence_seconds is None:
        # Unknown cadence: fall back to the grace window, never re-arming anything younger than it.
        cadence_seconds = grace
    return age_seconds > (cadence_seconds + grace)


# Per-expr cache for _schedule_cadence_seconds' croniter measurements.
_cron_cadence_cache: Dict[str, Optional[float]] = {}


def _schedule_cadence_seconds(schedule: Dict[str, Any]) -> Optional[float]:
    """Approximate schedule period in seconds, or None (croniter missing / malformed expr). Cron
    results are cached per expr because this runs under ``_jobs_lock`` every tick; the gap can vary
    with base time for irregular exprs, acceptable for a staleness *threshold*."""
    if not isinstance(schedule, dict):
        return None
    kind = schedule.get("kind")
    if kind == "interval":
        minutes = schedule.get("minutes")
        try:
            return float(minutes) * 60.0 if minutes else None
        except (TypeError, ValueError):
            return None
    if kind != "cron" or not _ensure_croniter():
        return None
    expr = schedule.get("expr")
    if not expr:
        return None
    if expr in _cron_cadence_cache:
        return _cron_cadence_cache[expr]
    try:
        it = croniter(expr, _hermes_now())
        first = it.get_next(datetime)
        gap = (it.get_next(datetime) - first).total_seconds()
        result = gap if gap > 0 else None
    except Exception:
        result = None
    # Hard bound so deleted/edited exprs can't grow the cache unboundedly in a long-lived gateway.
    if len(_cron_cadence_cache) >= 256:
        _cron_cadence_cache.clear()
    _cron_cadence_cache[expr] = result
    return result


def _append_telemetry_record(filename: str, entry: Dict[str, Any], recent: list) -> None:
    """Record ``entry`` in the bounded ``recent`` list and append to ``<cron_dir>/<filename>``
    (best effort — telemetry must never break a tick). Counters stay module-level ints per
    metric because tests reset them by name."""
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
        "persisted_error_recoveries.jsonl", entry, _persisted_error_recoveries_recent)


def get_persisted_error_recovery_stats() -> Dict[str, Any]:
    """Probe-visible snapshot of persisted-error recoveries."""
    return {
        "persisted_error_recoveries": _persisted_error_recoveries,
        "recent": list(_persisted_error_recoveries_recent),
    }


def _cron_next_run_matches_expr(schedule: Dict[str, Any], next_run_dt: datetime) -> bool:
    """Whether ``next_run_dt`` is an occurrence of the schedule's current expr (detects a
    hand-edited ``schedule.expr`` whose stored ``next_run_at`` came from the old one).
    Best-effort: anything uncheckable (non-cron, no expr, no croniter, malformed) reports a
    match.

    A direct ``jobs.json`` edit can change ``schedule.expr`` while leaving the stored ``next_run_at``
    computed under the *old* expression (#93049). The stored instant is stale exactly when it is not an
    occurrence of the current expression. Validation is best-effort: anything that cannot be checked
    (non-cron kind, missing expr, croniter unavailable, malformed input) reports a match so the fire path
    keeps its existing semantics.
    """
    if schedule.get("kind") != "cron":
        return True
    expr = schedule.get("expr")
    if not expr or not _ensure_croniter() or croniter is None:
        return True
    try:
        # Last occurrence at-or-before the instant: base one second past it so an exact hit is
        # included, then compare at second granularity (croniter is second-precision).
        prev = croniter(str(expr), next_run_dt + timedelta(seconds=1)).get_prev(datetime)
        return abs((prev - next_run_dt).total_seconds()) < 1.0
    except Exception:
        return True


# Classifications for a due cron instant NOT on the current expr (see
# _classify_stale_cron_next_run).
STALE_CRON_MATCH = "match"
STALE_CRON_TIMEZONE_MIGRATION = "timezone_migration"
STALE_CRON_EXPR_EDIT = "expr_edit"


def _classify_stale_cron_next_run(
    schedule: Dict[str, Any], raw_next_run_dt: datetime, next_run_dt: datetime,
) -> str:
    """Explain WHY a stored ``next_run_at`` misses the current cron lattice; the causes need
    opposite actions. ``expr_edit``: a hand edit changed ``schedule.expr`` — re-anchor WITHOUT
    firing. ``timezone_migration``: only the offset representation changed (legacy UTC rows
    normalized into the profile tz); treating it as an edit would skip a due, never-fired
    occurrence. Discriminator: normalization moved the wall clock AND the stored instant's OWN
    wall clock is a legal occurrence (when offsets agree a genuine expr edit can never be misread
    as a migration).

    * ``expr_edit`` — a direct ``jobs.json`` edit changed ``schedule.expr`` while leaving ``next_run_at``
    computed under the old one (#93049). Upgrading from a UTC-scheduling build to one that honours the
    profile timezone leaves legacy rows like ``2026-09-02T04:00:00+00:00`` for ``0 4 * * *``; normalizing to
    Europe/Brussels turns that into ``06:00+02``, which the expression excludes. Treating it as a stale edit
    re-anchored to tomorrow and silently skipped a due occurrence that had never fired.
    """
    if _cron_next_run_matches_expr(schedule, next_run_dt):
        return STALE_CRON_MATCH
    wall_clock_shifted = raw_next_run_dt.replace(tzinfo=None) != next_run_dt.replace(tzinfo=None)
    if wall_clock_shifted and _cron_next_run_matches_expr(schedule, raw_next_run_dt):
        return STALE_CRON_TIMEZONE_MIGRATION
    return STALE_CRON_EXPR_EDIT


# Offset-migration catch-ups on the fire path: climbing after a deploy = draining; steady = TZ
# churn.
_timezone_migration_catchups: int = 0
_timezone_migration_catchups_recent: list = []


def _record_timezone_migration_catchup(
    job: Dict[str, Any], raw_next_run_dt: datetime, next_run_dt: datetime,
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
        "timezone_migration_catchups.jsonl", entry, _timezone_migration_catchups_recent)


def get_timezone_migration_catchup_stats() -> Dict[str, Any]:
    """Probe-visible snapshot of offset-migration catch-up fires."""
    return {
        "timezone_migration_catchups": _timezone_migration_catchups,
        "recent": list(_timezone_migration_catchups_recent),
    }


def compute_next_run(schedule: Dict[str, Any], last_run_at: Optional[str] = None) -> Optional[str]:
    """Compute the next run time for a schedule as an ISO string, or None if no more runs."""
    now = _hermes_now()
    if not isinstance(schedule, dict):
        return None
    kind = schedule.get("kind")
    if kind == "once":
        return _recoverable_oneshot_run_at(schedule, now, last_run_at=last_run_at)
    # Recurring kinds anchor on last_run_at so a restart doesn't re-anchor the schedule.
    base_time = (_parse_aware(last_run_at) if last_run_at else None) or now
    if kind == "interval":
        minutes = schedule.get("minutes")
        if minutes is None:
            return None
        return (base_time + timedelta(minutes=minutes)).isoformat()
    if kind == "cron":
        expr = schedule.get("expr")
        if not expr:
            return None
        if not _ensure_croniter():
            logger.warning(
                "Cannot compute next run for cron schedule %r: 'croniter' is "
                "not installed. croniter is a core dependency as of v0.9.x; "
                "reinstall hermes-agent or run 'pip install croniter' in your runtime env.",
                expr)
            return None
        return croniter(expr, base_time).get_next(datetime).isoformat()
    return None


# --- Ticker heartbeat (liveness signal for `hermes cron status`) ---

def _write_marker(name: str, text: str, tmp_prefix: str) -> None:
    """Atomic (never torn) best-effort marker write; failures swallowed so markers never break the
    tick."""
    try:
        ensure_dirs()
        atomic_write_text(_current_cron_store().cron_dir / name, text, tmp_prefix=tmp_prefix)
    except Exception:
        pass


def record_ticker_heartbeat(success: bool = False) -> None:
    """Record ticker liveness (+ last-success marker when ``success``) so `cron status` can tell
    "alive but failing" from "firing"; scoped per profile store.

    The ticker calls this once per loop iteration. ``success=True`` additionally bumps the *last successful
    tick* marker. We track two distinct signals so `hermes cron status` can tell a thread that is merely
    *alive and looping* (heartbeat fresh, success stale) from one that is actually *firing jobs* (both
    fresh) — a ticker stuck failing every tick would otherwise keep the plain heartbeat fresh and falsely
    report healthy (#32612, #32895).
    Resolution uses ``_current_cron_store()`` so the heartbeat is correctly scoped to the active profile's
    store — critical under multiplex_profiles where each profile needs its own liveness signal (#69377).
    """
    _write_marker("ticker_heartbeat", str(time.time()), ".hb_")
    if success:
        _write_marker("ticker_last_success", str(time.time()), ".hb_")


def _epoch_file_age(name: str) -> Optional[float]:
    """Seconds since the epoch stamp stored in ``<cron_dir>/<name>``; None = missing/unreadable."""
    try:
        raw = (_current_cron_store().cron_dir / name).read_text(encoding="utf-8").strip()
        return max(0.0, time.time() - float(raw))
    except Exception:
        return None


def get_ticker_heartbeat_age() -> Optional[float]:
    """Seconds since the ticker loop last iterated; None = missing/unreadable ("cannot determine",
    not "dead").

    Resolution uses ``_current_cron_store()`` so the heartbeat is correctly scoped to the active profile —
    critical under multiplex_profiles where ``hermes cron status`` must report per-profile liveness
    (#69377).
    """
    return _epoch_file_age("ticker_heartbeat")


def get_ticker_success_age() -> Optional[float]:
    """Seconds since the ticker last completed a tick WITHOUT raising, or None.

    Resolution uses ``_current_cron_store()`` so the heartbeat is correctly scoped to the active profile —
    critical under multiplex_profiles where ``hermes cron status`` must report per-profile liveness
    (#69377).
    """
    return _epoch_file_age("ticker_last_success")


def get_catch_up_occurrence_count() -> int:
    """Return the profile-local stale-schedule catch-up count."""
    path = _current_cron_store().cron_dir / "catch_up_occurrences"
    try:
        return max(0, int(path.read_text(encoding="utf-8").strip()))
    except (OSError, ValueError):
        return 0


def record_catch_up_occurrence() -> None:
    """Increment the profile-local stale-schedule catch-up counter, best effort."""
    _write_marker("catch_up_occurrences", str(get_catch_up_occurrence_count() + 1), ".count_")


def record_ticker_error(message: str) -> None:
    """Persist the latest tick failure so `cron status` (another process) can show WHY, not just
    staleness."""
    _write_marker("ticker_last_error", f"{time.time()}\n{message.strip()}\n", ".terr_")


def clear_ticker_error() -> None:
    """Remove the last-tick-error marker after a successful tick. Best-effort."""
    with contextlib.suppress(OSError):
        (_current_cron_store().cron_dir / "ticker_last_error").unlink()


def get_ticker_last_error() -> Optional[str]:
    """Return the most recent recorded tick error message, or None."""
    try:
        raw = (_current_cron_store().cron_dir / "ticker_last_error").read_text(encoding="utf-8")
    except Exception:
        return None
    lines = raw.splitlines()
    if len(lines) < 2:
        return None
    return "\n".join(lines[1:]).strip() or None


# --- Job CRUD Operations ---

def _parse_jobs_file(jobs_file: Path) -> Tuple[Any, bool]:
    """Tolerant jobs.json parse -> ``(data, used_strict_fallback)``: utf-8-sig absorbs a BOM, strict
    failure retries with ``strict=False``. IO/fallback errors propagate (caller decides repair vs
    bail)."""
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
    # Stamp BEFORE reading (fail-safe, see _record_load_stamp): a racing write then forces the
    # merge.
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

    # Accept the canonical dict, or a bare list (auto-repair); any other top-level shape is
    # corruption.
    repair = "had invalid control characters" if _strict_retry else None
    if isinstance(data, dict):
        jobs = data.get("jobs", [])
        if isinstance(jobs, dict):
            # ID-keyed map from external tools: flatten (inline "id" wins, else the key), skip junk.
            # _peek_jobs_unlocked deliberately does NOT flatten, so saves never merge against it.
            skipped = [k for k, v in jobs.items() if not isinstance(v, dict)]
            if skipped:
                logger.warning(
                    "Skipping %d non-dict entr%s in id-keyed jobs map: %s",
                    len(skipped), "y" if len(skipped) == 1 else "ies",
                    ", ".join(map(repr, skipped)))
            jobs = [{**v, "id": v.get("id") or k} for k, v in jobs.items() if isinstance(v, dict)]
            repair = "id-keyed jobs map flattened to list"
    elif isinstance(data, list):
        jobs = data
        repair = "bare list wrapped as dict"
    else:
        raise RuntimeError(
            f"Cron database corrupted: expected {{'jobs': [...]}}, got {type(data).__name__}")
    if jobs and repair:
        save_jobs(jobs)
        logger.warning("Auto-repaired jobs.json (%s)", repair)
    _record_load_stamp(pre_read_stamp)
    return jobs


def _peek_jobs_unlocked() -> Optional[List[Dict[str, Any]]]:
    """Repair-free read under ``_jobs_lock()``: ``[]`` if missing, ``None`` if corrupt (never
    shrink-merge against an unknown baseline). Never saves — that would recurse."""
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
    return data if isinstance(data, list) else None


def _jobs_file_stamp(jobs_file: Path) -> Optional[Tuple[int, int, int]]:
    """Shrink-merge fast-path stamp ``(mtime_ns, size, ino)``; ``None`` if unstatable. ``st_ino`` is
    included because every writer uses mkstemp+rename, so a same-size write in one mtime quantum
    can't false-match."""
    try:
        st = jobs_file.stat()
        return (st.st_mtime_ns, st.st_size, st.st_ino)
    except OSError:
        return None


def _record_load_stamp(stamp: Optional[Tuple[int, int, int]]) -> None:
    """Remember jobs.json's stamp for the enclosing _jobs_lock() section (no-op outside one) so the
    save path can skip the shrink-merge when disk provably hasn't changed. Capture it BEFORE
    reading: a mid-read sibling then mismatches (fail-safe); stamping after would certify an
    unseen write.

    Stamping after the read would let that sibling's write be certified as "seen" without being in the
    loaded payload, wrongly suppressing the recovery. See #80703.
    """
    if getattr(_jobs_lock_state, "depth", 0):
        _jobs_lock_state.load_stamp = stamp


def _unmerged_disk_jobs(
    jobs: List[Dict[str, Any]], removed_ids: Optional[Collection[str]]
) -> List[Dict[str, Any]]:
    """On-disk jobs missing from *jobs* and not intentionally removed. Stamp match => nothing
    landed, return without parsing; unreadable store => ``[]`` (never merge against an unknown
    baseline)."""
    stamp = getattr(_jobs_lock_state, "load_stamp", None)
    if stamp is not None and _jobs_file_stamp(_current_cron_store().jobs_file) == stamp:
        return []
    disk_jobs = _peek_jobs_unlocked()
    if disk_jobs is None:
        return []
    seen = {str(j["id"]) for j in jobs if isinstance(j, dict) and j.get("id")}
    seen |= {str(i) for i in (removed_ids or ()) if i}
    recovered: List[Dict[str, Any]] = []
    for disk_job in disk_jobs:
        if not isinstance(disk_job, dict) or not disk_job.get("id"):
            continue
        disk_id = str(disk_job["id"])
        if disk_id not in seen:
            recovered.append(disk_job)
            seen.add(disk_id)
    return recovered


def _merge_unexpected_disk_jobs(
    jobs: List[Dict[str, Any]], *, removed_ids: Optional[Collection[str]] = None,
) -> List[Dict[str, Any]]:
    """*jobs* plus on-disk jobs absent from the payload (under the degraded flock-timeout path a
    stale writer would otherwise clobber concurrent creates). Deletes pass ``removed_ids``; never
    mutates *jobs*."""
    recovered = _unmerged_disk_jobs(jobs, removed_ids)
    if not recovered:
        return jobs
    logger.warning(
        "Preserved %d cron job(s) present on disk but missing from the "
        "in-memory save payload (concurrent create under degraded lock "
        "or stale writer) (#80624): %s",
        len(recovered), [j.get("id") for j in recovered])
    return jobs + recovered


def _unlink_quiet(path: Optional[str]) -> None:
    if path is not None:
        with contextlib.suppress(OSError):
            os.unlink(path)


def _stage_jobs_payload(jobs_file: Path, jobs: List[Dict[str, Any]]) -> str:
    """Serialize the store payload to a fsynced temp file next to *jobs_file*; return its path."""
    fd, tmp_path = tempfile.mkstemp(dir=str(jobs_file.parent), suffix=".tmp", prefix=".jobs_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                {"jobs": jobs, "updated_at": _hermes_now().isoformat()},
                f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        _unlink_quiet(tmp_path)
        raise
    return tmp_path


_SAVE_JOBS_MERGE_ATTEMPTS = 5


def _save_jobs_unlocked(
    jobs: List[Dict[str, Any]], *, removed_ids: Optional[Collection[str]] = None,
    replace: bool = False,
):
    """Save all jobs; caller must hold _jobs_lock(). ``removed_ids`` = intentional deletes;
    ``replace=True`` skips the shrink-merge guard (wholesale rewrite for tests/disaster
    recovery)."""
    jobs_file = _current_cron_store().jobs_file
    ensure_dirs()
    # Owner snapshot BEFORE replace so a root writer can hand the file back to the gateway user.
    _stat_before = None
    for probe in (jobs_file, jobs_file.parent):
        with contextlib.suppress(OSError):
            _stat_before = os.stat(probe)
            break

    # Shrink-merge loop: merge, stage, re-peek, repeat; the last attempt writes without a re-peek.
    tmp_path = None
    try:
        for attempt in range(_SAVE_JOBS_MERGE_ATTEMPTS + 1):
            if not replace:
                jobs = _merge_unexpected_disk_jobs(jobs, removed_ids=removed_ids)
            tmp_path = _stage_jobs_payload(jobs_file, jobs)
            # Verify-after-stage: a sibling landing during serialization forces another merge round.
            if (
                not replace
                and attempt < _SAVE_JOBS_MERGE_ATTEMPTS
                and _unmerged_disk_jobs(jobs, removed_ids)
            ):
                _unlink_quiet(tmp_path)
                tmp_path = None
                continue
            atomic_replace(tmp_path, jobs_file)
            tmp_path = None
            _secure_file(jobs_file)
            _preserve_file_ownership(jobs_file, _stat_before)
            # Invalidate (never refresh) the stamp: a refresh would let a nested save certify disk
            # against an OUTER caller's stale payload. Later saves take the full merge (fail-safe).
            _record_load_stamp(None)
            return
    except BaseException:
        _unlink_quiet(tmp_path)
        raise


def save_jobs(
    jobs: List[Dict[str, Any]], *, removed_ids: Optional[Collection[str]] = None,
    replace: bool = False,
):
    """Save all jobs under the lock; see ``_save_jobs_unlocked`` for ``removed_ids``/``replace``."""
    with _jobs_lock():
        _save_jobs_unlocked(jobs, removed_ids=removed_ids, replace=replace)


_MISSING = object()


def _with_job(
    job_id: Any, fn: Callable[[List[Dict[str, Any]], int, Dict[str, Any]], Any], missing: Any = None
) -> Any:
    """Run ``fn(jobs, i, job)`` on the first match under ``_jobs_lock()``; ``fn`` saves. *missing*
    if none."""
    with _jobs_lock():
        jobs = load_jobs()
        for i, job in enumerate(jobs):
            if job.get("id") == job_id:
                return fn(jobs, i, job)
    return missing


def _complete_job_record(job: Dict[str, Any]) -> None:
    """Retire *job* in place as a terminal completion (record kept for `cronjob list`)."""
    job.update(enabled=False, state="completed", next_run_at=None)


def _activate_job_record(job: Dict[str, Any]) -> None:
    """Clear pause markers in place so *job* is runnable again."""
    job.update(enabled=True, state="scheduled", paused_at=None, paused_reason=None)


def _normalize_workdir(workdir: Optional[str]) -> Optional[str]:
    """Workdir -> absolute path, or None when empty. ``~`` expands; relative paths are rejected
    (cron runs detached from any cwd); must be an existing dir now but is deliberately NOT
    re-checked at run time (scheduler falls back with a warning). ValueError when invalid."""
    if workdir is None:
        return None
    raw = str(workdir).strip()
    if not raw:
        return None
    expanded = Path(raw).expanduser()
    if not expanded.is_absolute():
        raise ValueError(
            f"Cron workdir must be an absolute path (got {raw!r}). "
            f"Cron jobs run detached from any shell cwd, so relative paths are ambiguous.")
    resolved = expanded.resolve()
    if not resolved.exists():
        raise ValueError(f"Cron workdir does not exist: {resolved}")
    if not resolved.is_dir():
        raise ValueError(f"Cron workdir is not a directory: {resolved}")
    return str(resolved)


def _resolve_default_model_snapshot() -> Optional[str]:
    """Default model resolved as the ticker's ``run_job`` does, so unpinned jobs can snapshot it and
    detect a later swap. ``None`` on missing config or failure (fail-open: "no snapshot")."""
    try:
        from hermes_cli.config import _expand_env_vars, read_user_config_raw

        cfg_path = get_hermes_home() / "config.yaml"
        if not cfg_path.exists():
            return None
        cfg = read_user_config_raw(cfg_path)
        with contextlib.suppress(Exception):
            from hermes_cli import managed_scope
            cfg = managed_scope.apply_managed_overlay(cfg)
        cfg = _expand_env_vars(cfg)
        cron_cfg = cfg.get("cron") or {}
        if isinstance(cron_cfg, dict):
            cron_model = cron_cfg.get("model")
            if isinstance(cron_model, str) and cron_model.strip():
                return cron_model.strip()
        model_cfg = cfg.get("model") or {}
        if isinstance(model_cfg, dict):
            model_cfg = model_cfg.get("default") or model_cfg.get("model")
        return model_cfg.strip() or None if isinstance(model_cfg, str) else None
    except Exception:
        return None


def _normalize_job_optional_text(
    value: Any, *, strip_trailing_slash: bool = False
) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return (value.strip().rstrip("/") if strip_trailing_slash else value.strip()) or None


def _normalize_base_url(value: Any) -> Optional[str]:
    return _normalize_job_optional_text(value, strip_trailing_slash=True)


def _normalize_str_list(items: Any) -> Optional[List[str]]:
    """Non-blank stripped items of *items*, or None when nothing remains."""
    return [str(j).strip() for j in items if str(j).strip()] or None


def _normalize_context_from(value: Any) -> Optional[List[str]]:
    """Accept a job id or a list of ids; anything else is None."""
    if isinstance(value, str):
        value = [value]
    return _normalize_str_list(value) if isinstance(value, list) else None


def _normalize_failure_deliver(value: Any) -> Optional[str]:
    """failure_deliver shares deliver's value grammar; flatten str/list like the tool layer's
    _normalize_deliver_param for direct create_job callers. Semantic validation happens at
    resolution time via the shared deliver path."""
    if isinstance(value, (list, tuple)):
        return ",".join(str(p).strip() for p in value if str(p).strip()) or None
    return _normalize_job_optional_text(value)


def _normalize_reasoning_effort(value: Any) -> Optional[str]:
    """Spelling-only validation via the shared parser (cron knob never stricter/looser than
    config.yaml); model capability is deliberately NOT checked (model unknowable at create time,
    transports clamp at send time). None for unset, lowercase level, or ValueError."""
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
            "(empty string clears the override).")
    if text in {"false", "disabled"}:
        return "none"
    return text


# Normalizers for create_job (all fields) / update_job (present fields). Invalid values raise BEFORE
# storing.
_CREATE_FIELD_NORMALIZERS: Dict[str, Callable[[Any], Any]] = {
    "model": _normalize_job_optional_text,
    "provider": _normalize_job_optional_text,
    "base_url": _normalize_base_url,
    "script": _normalize_job_optional_text,
    "monitor_script": _normalize_job_optional_text,
    "monitor_url": _normalize_job_optional_text,
    "enabled_toolsets": lambda v: _normalize_str_list(v) if v else None,
    "workdir": _normalize_workdir,
    "no_agent": bool,
    "context_from": _normalize_context_from,
    "failure_deliver": _normalize_failure_deliver,
}
_UPDATE_FIELD_NORMALIZERS: Dict[str, Callable[[Any], Any]] = {
    "workdir": lambda v: None if v in {None, "", False} else _normalize_workdir(v),
    "monitor_script": _normalize_job_optional_text,
    "monitor_url": _normalize_job_optional_text,
    "reasoning_effort": _normalize_reasoning_effort,
}


def _compute_provider_model_snapshots(
    *, provider: Any, model: Any, base_url: Any, no_agent: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """Snapshot unpinned provider/model resolution so a later global switch fails closed at fire
    time instead of silently changing spend. Pinned axes and no-agent jobs carry no snapshot."""
    normalized_provider = _normalize_job_optional_text(provider)
    normalized_model = _normalize_job_optional_text(model)
    normalized_base_url = _normalize_base_url(base_url)
    if bool(no_agent):
        return None, None

    provider_snapshot: Optional[str] = None
    model_snapshot: Optional[str] = None
    if normalized_provider is None:
        with contextlib.suppress(Exception):
            from hermes_cli.runtime_provider import resolve_runtime_provider

            runtime_kwargs = {"requested": None}
            # Delegate all rate-limit / 5xx retry to hermes's outer conversation loop, which honors
            # Retry-After. The SDK default (max_retries=2) uses its own 1-2s backoff that ignores
            # Retry-After and double-retries inside our loop — burning request slots against a bucket that
            # won't refill for minutes. (#26293)
            if normalized_base_url:
                runtime_kwargs["explicit_base_url"] = normalized_base_url
            snap = resolve_runtime_provider(**runtime_kwargs)
            provider_snapshot = str(snap.get("provider") or "").strip().lower() or None
    if normalized_model is None:
        with contextlib.suppress(Exception):
            model_snapshot = _resolve_default_model_snapshot() or None
    return provider_snapshot, model_snapshot


def _normalized_inference_axes(
    job: Dict[str, Any],
) -> Tuple[Optional[str], Optional[str], Optional[str], bool]:
    """Return the stored inference-routing fields in their semantic form."""
    return (
        _normalize_job_optional_text(job.get("provider")),
        _normalize_job_optional_text(job.get("model")), _normalize_base_url(job.get("base_url")),
        bool(job.get("no_agent")),
    )


def _validate_job_mode_invariants(
    monitor_script: Optional[str],
    monitor_url: Optional[str],
    no_agent: bool,
    script: Optional[str],
) -> None:
    """Execution-mode invariants shared by create_job and update_job (no bypass via the update
    door)."""
    if monitor_script and monitor_url:
        raise ValueError(
            "monitor_script and monitor_url are mutually exclusive — a job "
            "can only have one monitor source.")
    if (monitor_script or monitor_url) and no_agent:
        raise ValueError(
            "monitor_script/monitor_url cannot be combined with no_agent=True — "
            "the whole point of a monitor job is to suppress or wake the AGENT "
            "based on source changes. Use a plain no_agent script job instead.")
    if no_agent and not script:
        raise ValueError(NO_AGENT_WITHOUT_SCRIPT_ERROR)


def _oneshot_past_grace_error(run_at: Any) -> ValueError:
    return ValueError(
        f"Requested one-shot time {run_at} is more than "
        f"{ONESHOT_GRACE_SECONDS}s in the past and cannot be scheduled.")


def _next_run_or_reject_past_oneshot(
    parsed_schedule: Dict[str, Any], label: str, fallback_run_at: Any, what: str,
) -> Optional[str]:
    """``compute_next_run`` that raises (after a warning log) for a one-shot outside the grace
    window, so a ghost job with ``next_run_at=None`` can never be stored."""
    next_run_at = compute_next_run(parsed_schedule)
    if parsed_schedule.get("kind") == "once" and next_run_at is None:
        run_at = parsed_schedule.get("run_at") or fallback_run_at
        logger.warning(
            "Rejecting one-shot cron job %s'%s': run_at %s is outside the %ss grace window",
            what, label, run_at, ONESHOT_GRACE_SECONDS)
        raise _oneshot_past_grace_error(run_at)
    return next_run_at


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
    prompt_file: Optional[str] = None,
    no_cron_hint: Optional[bool] = None,
) -> Dict[str, Any]:
    """Create a new cron job and return the stored record.

    deliver defaults to "origin" when ``origin`` is given, else "local"; repeat None = forever.
    script: stdout is injected as prompt context, or with ``no_agent=True`` IS the job (stdout
    delivered verbatim, requires ``script``). context_from: job id(s) whose latest output is
    injected. workdir: absolute cwd for tools/scripts. monitor_script/monitor_url: cheap monitor
    source run FIRST each tick; unchanged output suppresses the agent run (mutually exclusive,
    incompatible with ``no_agent``). reasoning_effort: per-job pin; capability NOT validated."""
    parsed_schedule = parse_schedule(schedule)
    # Normalize repeat: treat 0 or negative values as None (infinite). String forms
    # ('forever'/'once'/numeric) coerce via normalize_repeat_value — the shared chokepoint with update paths
    # (#66824/#64520/#7142/#71987/#95706).
    repeat = normalize_repeat_value(repeat)
    if parsed_schedule["kind"] == "once" and repeat is None:
        repeat = 1
    if deliver is None:
        deliver = "origin" if origin else "local"
    job_id = uuid.uuid4().hex[:12]
    now = _hermes_now().isoformat()

    raw = locals()
    f = {key: norm(raw[key]) for key, norm in _CREATE_FIELD_NORMALIZERS.items()}
    normalized_skills = _normalize_skill_list(skill, skills)
    normalized_attach = attach_to_session if isinstance(attach_to_session, bool) else None
    normalized_reasoning_effort = _normalize_reasoning_effort(reasoning_effort)
    normalized_prompt_file = str(prompt_file).strip() if isinstance(prompt_file, str) else None
    normalized_prompt_file = normalized_prompt_file or None
    normalized_no_cron_hint = no_cron_hint if isinstance(no_cron_hint, bool) else None
    # prompt_file must resolve to a readable file at create time so a typo fails
    # fast here instead of silently at every fire. It is read fresh each run (see
    # scheduler_prompt._build_job_prompt), so it stays the single source of truth for a
    # routine that lives in a git-backed file.
    if normalized_prompt_file is not None:
        _pf = Path(normalized_prompt_file).expanduser()
        if not _pf.is_file():
            raise ValueError(
                f"prompt_file does not exist or is not a readable file: {normalized_prompt_file}"
            )
        # Store the resolved absolute path so create-time validation and the
        # fire-time read (scheduler runs under its own cwd) refer to the same
        # file -- a relative path would resolve differently at each. Mirrors
        # _normalize_workdir's expand+resolve contract.
        normalized_prompt_file = str(_pf.resolve())

    _validate_job_mode_invariants(f["monitor_script"], f["monitor_url"], f["no_agent"], f["script"])
    prompt_text = _coerce_job_text(prompt).strip()
    if not prompt_text and not f["script"] and not normalized_skills and not normalized_prompt_file:
        raise ValueError(EMPTY_PAYLOAD_ERROR)
    # Reject gateway-lifecycle commands (respawn loops) here, not just in the CLI: covers the tool.
    from cron.lifecycle_guard import check_gateway_lifecycle
    check_gateway_lifecycle(prompt_text, f["script"])

    label_source = (
        prompt_text
        or (normalized_skills[0] if normalized_skills else None)
        or (f["script"] if f["no_agent"] else None)
        or "cron job"
    )
    name = name or label_source[:50].strip()
    provider_snapshot, model_snapshot = _compute_provider_model_snapshots(
        provider=f["provider"], model=f["model"], base_url=f["base_url"], no_agent=f["no_agent"])
    next_run_at = _next_run_or_reject_past_oneshot(parsed_schedule, name, schedule, "")

    job = {
        "id": job_id,
        "name": name,
        "prompt": prompt_text,
        "skills": normalized_skills,
        "skill": normalized_skills[0] if normalized_skills else None,
        "model": f["model"],
        "provider": f["provider"],
        "provider_snapshot": provider_snapshot,
        "model_snapshot": model_snapshot,
        "base_url": f["base_url"],
        "script": f["script"],
        "no_agent": f["no_agent"],
        "monitor_script": f["monitor_script"],
        "monitor_url": f["monitor_url"],
        "monitor_state": None,
        "context_from": f["context_from"],
        "schedule": parsed_schedule,
        "schedule_display": parsed_schedule.get("display", schedule),
        "repeat": {"times": repeat, "completed": 0},  # times None = forever
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
        # Targets acked without message_id/raw_response (accepted but UNVERIFIED).
        "last_delivery_unverified": None,
        "failure_streak": 0,
        "deliver": deliver,
        "origin": origin,  # Tracks where job was created for "origin" delivery
        "enabled_toolsets": f["enabled_toolsets"],
        "workdir": f["workdir"],
    }
    # Optional keys are persisted only when explicitly set: an absent key falls back to global
    # config (attach/reasoning) or to ``deliver`` (failure_deliver), byte-identical to pre-feature
    # jobs.
    for key, value in (
        ("attach_to_session", normalized_attach), ("reasoning_effort", normalized_reasoning_effort),
        ("failure_deliver", f["failure_deliver"]),
        ("prompt_file", normalized_prompt_file), ("no_cron_hint", normalized_no_cron_hint),
    ):
        if value is not None:
            job[key] = value

    with _jobs_lock():
        save_jobs(load_jobs() + [job])
    return job


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Get a job by ID."""
    job = next((j for j in load_jobs() if j["id"] == job_id), None)
    return _normalize_job_record(job) if job is not None else None


class AmbiguousJobReference(LookupError):
    """Raised when a job name matches more than one job."""

    def __init__(self, ref: str, matches: List[Dict[str, Any]]):
        self.ref = ref
        self.matches = matches
        ids = ", ".join(m["id"] for m in matches)
        super().__init__(
            f"Job name '{ref}' is ambiguous — matches {len(matches)} jobs: {ids}. "
            f"Use the job ID instead.")


def resolve_job_ref(ref: str) -> Optional[Dict[str, Any]]:
    """Resolve an ID or name to a job record. Exact ID wins, then case-insensitive name; an
    ambiguous name raises AmbiguousJobReference rather than silently picking one."""
    if not ref:
        return None
    jobs = load_jobs()
    by_id = next((j for j in jobs if j["id"] == ref), None)
    if by_id is not None:
        return _normalize_job_record(by_id)
    ref_lower = ref.lower()
    name_matches = [j for j in jobs if (j.get("name") or "").lower() == ref_lower]
    if not name_matches:
        return None
    if len(name_matches) > 1:
        raise AmbiguousJobReference(ref, [_normalize_job_record(j) for j in name_matches])
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


def _reject_terminal_activation(job: Dict[str, Any], updated: Dict[str, Any], job_id: str) -> None:
    """A genuinely terminal job cannot be reactivated through update_job (use cron resume)."""
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
            "through update_job; use cron resume --run-now or --at.")


def _normalize_job_updates(job: Dict[str, Any], updates: Dict[str, Any]) -> None:
    """Normalize updates in place like create_job; invalid values raise BEFORE the merge. ``repeat``
    accepts the stored dict or a bare value (coerced, completed counter preserved)."""
    for key, norm in _UPDATE_FIELD_NORMALIZERS.items():
        if key in updates:
            updates[key] = norm(updates[key])
    if "repeat" in updates:
        _rp = updates["repeat"]
        completed = (job.get("repeat") or {}).get("completed", 0)
        if isinstance(_rp, dict):
            _rp = dict(_rp)
            _rp["times"] = normalize_repeat_value(_rp.get("times"))
            _rp.setdefault("completed", completed)
            updates["repeat"] = _rp
        else:
            updates["repeat"] = {"times": normalize_repeat_value(_rp), "completed": completed}


def _apply_schedule_update(updated: Dict[str, Any], updates: Dict[str, Any], job_id: str) -> None:
    """Parse a string schedule, refresh ``schedule_display`` and (unless paused) ``next_run_at``."""
    updated_schedule = updated["schedule"]
    if isinstance(updated_schedule, str):
        updated_schedule = parse_schedule(updated_schedule)
        updated["schedule"] = updated_schedule
    updated["schedule_display"] = updates.get(
        "schedule_display", updated_schedule.get("display", updated.get("schedule_display")))
    if updated.get("state") != "paused":
        updated["next_run_at"] = _next_run_or_reject_past_oneshot(
            updated_schedule, updated.get("name", job_id), updated_schedule, "update ")


def _fill_missing_next_run(updated: Dict[str, Any]) -> None:
    """An enabled, unpaused record must never persist without ``next_run_at`` (it would never fire).
    """
    if (
        not updated.get("enabled", True)
        or updated.get("state") == "paused"
        or updated.get("next_run_at")
    ):
        return
    next_run = compute_next_run(updated["schedule"])
    if next_run is None and updated["schedule"].get("kind") == "once":
        run_at = updated["schedule"].get("run_at", "unknown")
        raise ValueError(
            f"Requested one-shot time {run_at} is in the past "
            f"(grace window: {ONESHOT_GRACE_SECONDS}s) and cannot be scheduled.")
    updated["next_run_at"] = next_run


def update_job(job_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a job by ID, refreshing derived schedule fields when needed."""
    # ``id`` is a path component under OUTPUT_DIR — changing it would leak path-escape values.
    bad_fields = _IMMUTABLE_JOB_FIELDS.intersection(updates or {})
    if bad_fields:
        raise ValueError(f"Cron job field(s) cannot be updated: {', '.join(sorted(bad_fields))}")

    def apply(jobs, i, job):
        _normalize_job_updates(job, updates)
        previous_inference_axes = _normalized_inference_axes(job)
        updated = _apply_skill_fields({**job, **updates})
        _reject_terminal_activation(job, updated, job_id)
        # Re-check on the MERGED record; scoped to changed fields so legacy records keep loading.
        if {"monitor_script", "monitor_url", "no_agent", "script"}.intersection(updates):
            _validate_job_mode_invariants(
                updated.get("monitor_script") or None,
                updated.get("monitor_url") or None,
                bool(updated.get("no_agent")),
                _normalize_job_optional_text(updated.get("script")))
        if any(k in updates for k in _PAYLOAD_FIELDS) and job_payload_is_empty(updated):
            raise ValueError(EMPTY_PAYLOAD_ERROR)
        inference_fields_changed = bool(
            {"provider", "model", "base_url", "no_agent"}.intersection(updates)
        ) and _normalized_inference_axes(updated) != previous_inference_axes

        if "schedule" in updates:
            _apply_schedule_update(updated, updates, job_id)
        if inference_fields_changed:
            snapshots = _compute_provider_model_snapshots(
                provider=updated.get("provider"),
                model=updated.get("model"),
                base_url=updated.get("base_url"),
                no_agent=updated.get("no_agent"))
            updated["provider_snapshot"], updated["model_snapshot"] = snapshots
        _fill_missing_next_run(updated)
        _reject_terminal_activation(job, updated, job_id)
        jobs[i] = updated
        save_jobs(jobs)
        return _normalize_job_record(updated)

    return _with_job(job_id, apply)


def pause_job(job_id: str, reason: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Pause a job without deleting it. Accepts a job ID or name."""
    job = resolve_job_ref(job_id)
    if not job:
        return None
    return update_job(job["id"], {
        "enabled": False,
        "state": "paused",
        "paused_at": _hermes_now().isoformat(),
        "paused_reason": reason,
    })


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
            f"(grace window: {ONESHOT_GRACE_SECONDS}s) and will never fire.")
    return update_job(job["id"], {
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "next_run_at": next_run_at,
    })


def trigger_job(job_id: str, extra_prompt: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Schedule a job for the next tick (ID or name). ``extra_prompt`` is stamped as
    ``manual_run_prompt`` for that single fire only; ``mark_job_run`` clears it."""
    job = resolve_job_ref(job_id)
    if not job:
        return None
    if is_terminal_job(job):
        name = job.get("name", job_id)
        raise ValueError(
            f"Cannot run: job '{name}' is {job.get('state')} (terminal). "
            f"Create a new occurrence with 'hermes cron resume {name} "
            "--run-now' or '--at <ISO-8601>'.")
    manual_run_at = _hermes_now().isoformat()
    return update_job(job["id"], {
        "enabled": True,
        "state": "scheduled",
        "paused_at": None,
        "paused_reason": None,
        "next_run_at": manual_run_at,
        # Run-now intent, so cron expression/TZ repair guards don't treat it as stale state.
        "manual_run_at": manual_run_at,
        "manual_run_prompt": (extra_prompt or None),
    })


def _claim_is_live(claim: Any, now: datetime, ttl_seconds: float) -> bool:
    """True for a well-formed claim aged within ``[0, ttl)``: future-dated (clock/TZ skew) or
    malformed claims count as stale so they can never wedge a job."""
    if not isinstance(claim, dict) or not claim.get("at"):
        return False
    claimed_at = _parse_aware(claim["at"])
    return claimed_at is not None and 0 <= (now - claimed_at).total_seconds() < ttl_seconds


_REARM_RECURRING_ERROR = (
    "Cannot re-arm recurring jobs: re-arm is one-shot-only; use plain resume or cron run."
)


def rearm_oneshot(job_id: str, run_at: Any) -> Optional[Dict[str, Any]]:
    """Re-arm a completed one-shot as an explicit new occurrence."""
    job_ref = resolve_job_ref(job_id)
    if not job_ref:
        return None
    if isinstance(run_at, datetime):
        run_at = run_at.isoformat()
    parsed_schedule = parse_schedule(str(run_at))
    if parsed_schedule.get("kind") != "once":
        raise ValueError(_REARM_RECURRING_ERROR)
    next_run_at = compute_next_run(parsed_schedule)
    if next_run_at is None:
        raise _oneshot_past_grace_error(parsed_schedule.get("run_at") or run_at)

    def apply(jobs, _i, job):
        now = _hermes_now()
        if _claim_is_live(job.get("run_claim"), now, _oneshot_run_claim_ttl_seconds()):
            raise ValueError("Cannot re-arm one-shot over a live run claim.")
        if _claim_is_live(job.get("fire_claim"), now, 300):
            raise ValueError("Cannot re-arm one-shot over a live fire claim.")
        if job.get("schedule", {}).get("kind") != "once":
            raise ValueError(_REARM_RECURRING_ERROR)
        repeat = job.get("repeat") or {}
        repeat["completed"] = 0
        job.update(
            schedule=parsed_schedule, schedule_display=parsed_schedule.get("display", str(run_at)),
            repeat=repeat, run_claim=None, fire_claim=None)
        _activate_job_record(job)
        job["next_run_at"] = next_run_at
        save_jobs(jobs)
        return _normalize_job_record(job)

    return _with_job(job_ref["id"], apply)


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
        if len(jobs) == original_len:
            return False
        # Resolve BEFORE saving so a legacy unsafe ID fails closed without a half-applied removal.
        job_output_dir = _job_output_dir(canonical_id)
        save_jobs(jobs, removed_ids={canonical_id})
        if job_output_dir.exists():
            shutil.rmtree(job_output_dir)
        try:
            from cron.notepad import clear_notepad
            clear_notepad(canonical_id)
        except Exception:
            logger.debug("Failed to clear notepad for removed job %s", canonical_id, exc_info=True)
        # Prune the fire-fence lock entry so the registry doesn't grow monotonically.
        _fence_key = f"{_current_cron_store().cron_dir.resolve()}::{canonical_id}"
        with _fire_fence_locks_guard:
            _fire_fence_locks.pop(_fence_key, None)
        return True


def _set_alert_flag(job_id: str, field: str, value: bool) -> bool:
    """Set/clear a persisted alert-dedup marker (alert exactly once until the condition heals;
    survives restarts) and return the PRIOR value. Fields: ``preflight_alerted``,
    ``drift_alerted``.

    The marker records that the operator was already alerted about this job's condition, so the scheduler
    alerts exactly once and stays silent on subsequent ticks until the condition heals (same alert-once
    shape as the dead-pin auto-pause in #73506). Fields: ``preflight_alerted`` (blocked config, T1-26) and
    ``drift_alerted`` (#44585 drift-guard skip).
    """
    def apply(jobs, _i, job):
        prior = bool(job.get(field))
        if value:
            job[field] = True
        else:
            job.pop(field, None)
        if prior != value:
            save_jobs(jobs)
        return prior

    return _with_job(job_id, apply, False)


def mark_preflight_alerted(job_id: str) -> bool:
    """Mark the job as preflight-alerted; return True if it already was."""
    return _set_alert_flag(job_id, "preflight_alerted", True)


def clear_preflight_alerted(job_id: str) -> None:
    """Clear the preflight alert-dedup marker (config validates again)."""
    _set_alert_flag(job_id, "preflight_alerted", False)


def mark_drift_alerted(job_id: str) -> bool:
    """Mark the job as drift-alerted; return True if it already was."""
    return _set_alert_flag(job_id, "drift_alerted", True)


def note_fire_forward_failure(job_id: str, detail: str) -> bool:
    """Durably record (as ``last_fire_error``) that a scheduled fire could not be handed to the
    runner — written by the dashboard fire webhook when the loopback forward fails. Without it
    the miss is invisible (no execution row, last_status only covers started runs); mark_job_run
    clears it."""
    def apply(jobs, _i, job):
        job["last_fire_error"] = {
            "at": _hermes_now().isoformat(), "detail": str(detail or "")[:500]}
        save_jobs(jobs)
        return True

    return _with_job(job_id, apply, False)


def _record_run_outcome(
    job: Dict[str, Any], success: bool, error: Optional[str], delivery_error: Optional[str],
    status: Optional[str], now: str,
) -> None:
    """Stamp one completed run onto *job*: status fields, failure streak, alert markers, claims."""
    job["last_run_at"] = now
    job.pop("manual_run_at", None)
    # The transient manual-run context is single-fire: the run that just completed consumed it.
    job.pop("manual_run_prompt", None)
    delivery_failed = isinstance(delivery_error, str) and bool(delivery_error.strip())
    job["last_status"] = status or (
        "error" if not success else ("delivery_failed" if delivery_failed else "ok"))
    job["last_error"] = None if success else error
    if success:
        # Healthy run: drop the alert-once dedup markers so a FUTURE break re-alerts, and clear
        # the forward-failure stamp so it only describes CURRENT auto-fire health.
        job.pop("preflight_alerted", None)
        job.pop("drift_alerted", None)
        job.pop("last_fire_error", None)
        job["failure_streak"] = 0
    else:
        # Consecutive agent-failure streak; delivery failures do NOT count
        # (scheduler._failure_streak_nudge).
        job["failure_streak"] = int(job.get("failure_streak") or 0) + 1
    job["last_delivery_error"] = delivery_error
    # Clear both claims: the run is over, so the job is claimable again.
    job["fire_claim"] = None
    if job.get("run_claim") is not None:  # keep key absence for legacy records
        job["run_claim"] = None


def _advance_after_run(job: Dict[str, Any], now: str) -> None:
    """Bump ``repeat.completed`` and recompute ``next_run_at``; retire the record as a terminal
    completion when the repeat limit is reached or a one-shot has no further run."""
    # If no next run, decide whether this is terminal completion (one-shot) or a transient failure
    # (recurring schedule couldn't compute — e.g. 'croniter' missing from the runtime env). Recurring jobs
    # must NEVER be silently disabled: that turns a missing runtime dep into "job completed" and the user's
    # schedule quietly goes off. See issue #16265.
    kind = job.get("schedule", {}).get("kind")
    # One-shot dispatch-limit guard (issue #38758): a finite one-shot claimed via claim_dispatch() but whose
    # tick died before mark_job_run could remove it will have completed >= times while still looking due
    # (last_run_at was never written, so the recovery helper re-armed it). Remove it instead of re-firing.
    repeat = job.get("repeat")
    if repeat:
        times = repeat.get("times")
        finite = times is not None and times > 0
        completed = repeat.get("completed", 0)
        # Finite one-shots were pre-claimed by claim_dispatch() (completed already incremented) —
        # do not double-count; recurring jobs and direct callers still get the increment.
        if not (kind == "once" and finite and completed > 0):
            completed += 1
            repeat["completed"] = completed
        if finite and completed >= times:
            # Limit reached: retain a terminal record instead of popping it, so the status just
            # written stays inspectable in `cronjob list`; the retention sweep prunes it later.
            _complete_job_record(job)
            return

    job["next_run_at"] = compute_next_run(job["schedule"], now)
    if job["next_run_at"] is not None:
        if job.get("state") != "paused":
            job["state"] = "scheduled"
    elif kind in {"cron", "interval"}:
        # Recurring: transient failure (e.g. croniter missing) — disabling it would turn a missing
        # dep into "job completed" and silently drop the schedule.
        job["state"] = "error"
        if not job.get("last_error"):
            job["last_error"] = (
                "Failed to compute next run for recurring schedule (is the 'croniter' package "
                "installed in the gateway's Python env?)")
        logger.error(
            "Job '%s' (%s) could not compute next_run_at; "
            "leaving enabled and marking state=error so the job is not silently disabled.",
            job.get("name", job.get("id", "?")), kind)
    else:
        _complete_job_record(job)  # one-shot: terminal completion


def mark_job_run(
    job_id: str,
    success: bool,
    error: Optional[str] = None,
    delivery_error: Optional[str] = None,
    status: Optional[str] = None,
    *,
    expected_fire_owner: Optional[str] = None,
) -> bool:
    """Mark a job as run: update last_run_at/last_status, bump completed, recompute next_run_at,
    and retire the record as a terminal completion when the repeat limit is reached.

    ``delivery_error`` is separate from the agent error: agent succeeded but delivery failed records
    ``last_status = "delivery_failed"`` (never "ok") while ``failure_streak`` is left alone. An
    explicit ``status`` (e.g. "blocked_config") overrides the derived value. False when the fence
    can't be taken, the job is missing, or ``expected_fire_owner`` no longer holds the fire claim.
    """
    def apply(jobs, _i, job):
        if expected_fire_owner is not None:
            claim = job.get("fire_claim")
            if not isinstance(claim, dict) or claim.get("by") != expected_fire_owner:
                logger.warning(
                    "mark_job_run: job_id %s fire claim owner changed; discarding stale completion",
                    job_id)
                return False
        now = _hermes_now().isoformat()
        _record_run_outcome(job, success, error, delivery_error, status, now)
        _advance_after_run(job, now)
        save_jobs(jobs)
        return True

    def locked():
        found = _with_job(job_id, apply, missing=_MISSING)
        if found is _MISSING:
            logger.warning("mark_job_run: job_id %s not found, skipping save", job_id)
            return False
        return found

    return _under_fire_fence(job_id, locked)


def _write_oneshot_diagnostic(job: Dict[str, Any], text: str, what: str) -> bool:
    """Best-effort operator-visible trace in the job's output dir; never breaks the caller."""
    try:
        save_job_output(job.get("id", ""), text)
        return True
    except Exception as e:
        logger.debug("Failed to write %s diagnostic for job %r: %s", what, job.get("id"), e)
        return False


def _write_wedged_oneshot_diagnostic(job: Dict[str, Any]) -> None:
    """Trace for a wedged one-shot removal: dispatch was claimed but mark_job_run never ran
    (interrupted mid-run); removing it silently would leave no output, error, or record.

    A finite one-shot whose dispatch was claimed (``repeat.completed`` >= ``repeat.times``) but which never
    reached ``mark_job_run`` (``last_run_at`` is null) was interrupted mid-run — scheduler restart, gateway
    kill, or a non-Exception escape (#73973). The recovery guards remove such jobs so they stop appearing
    due, but a silent removal leaves the user with no output, no error, and no job record. Write a small
    diagnostic file into the job's output directory so the removal is observable and debuggable.
    """
    if job.get("last_run_at") is not None:
        return  # a prior run was recorded — normal completion race, not a wedge
    repeat = job.get("repeat") or {}
    claim = job.get("run_claim") or {}
    written = _write_oneshot_diagnostic(
        job,
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
        "again.\n",
        "wedged-oneshot")
    if written:
        logger.warning(
            "Job '%s': removed without a completed run — diagnostic written to "
            "its output directory",
            job.get("name", job.get("id", "?")))


def _write_missed_oneshot_diagnostic(job: Dict[str, Any], next_run: str) -> None:
    """Trace for a never-ran one-shot retired outside the grace window (else it would just vanish).
    """
    _write_oneshot_diagnostic(
        job,
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
        "schedule it again.\n",
        "missed-oneshot")


def claim_dispatch(job_id: str) -> bool:
    """Atomically claim a finite one-shot dispatch BEFORE execution: ``repeat.completed`` is bumped
    and persisted under the jobs lock so a tick dying mid-execution cannot lose the dispatch
    (*at-most-times* instead of *at-least-once*). True if the caller may run the job; False when
    the limit is already reached. Only ``kind == "once"`` with ``repeat.times > 0`` is claimed.

    Increments ``repeat.completed`` under the cross-process jobs lock and persists the claim immediately, so
    that if the tick dies mid-execution (gateway kill, OOM, segfault, hard-timeout) the dispatch is not
    lost. This converts finite one-shot jobs from *at-least-once* to *at-most-times* semantics — a job that
    self-destructs fires at most ``repeat.times`` times instead of infinitely (issue #38758).
    """
    def apply(jobs, i, job):
        repeat = job.get("repeat") or {}
        times = repeat.get("times")
        # Recurring jobs use advance_next_run(); no/infinite repeat limit always dispatches.
        if job.get("schedule", {}).get("kind") != "once" or times is None or times <= 0:
            return True
        completed = repeat.get("completed", 0)
        label = job.get("name", job.get("id", "?"))
        if completed >= times:
            if job.get("last_run_at") is not None:
                # A prior run completed normally (mark_job_run raced this tick). Retain the terminal
                # record, as mark_job_run's repeat-limit branch does, instead of deleting the
                # status.
                _complete_job_record(job)
                save_jobs(jobs)
                logger.info(
                    "Job '%s': dispatch limit reached (%d/%d) — marking completed",
                    label, completed, times)
                return False
            # A prior tick claimed the dispatch then died — a genuinely wedged claim. Remove it so
            # it stops appearing due, leaving an operator-visible diagnostic.
            jobs.pop(i)
            # See #73973.
            save_jobs(jobs, removed_ids={job_id})
            _write_wedged_oneshot_diagnostic(job)
            logger.info(
                "Job '%s': dispatch limit reached (%d/%d) — removing", label, completed, times)
            return False
        # Claim this dispatch before the side effect runs.
        repeat["completed"] = completed + 1
        save_jobs(jobs)
        logger.debug("Job '%s': claimed dispatch %d/%d", label, repeat["completed"], times)
        return True

    claimed = _with_job(job_id, apply, missing=_MISSING)
    if claimed is _MISSING:
        logger.debug(
            "claim_dispatch: job_id %s not in store — proceeding without claim "
            "(handed-in job dict; nothing to persist a claim against)",
            job_id)
        return True
    return claimed


def _refresh_claim(jobs: List[Dict[str, Any]], claim: Any, expected_owner: str) -> bool:
    """Compare-and-refresh a claim's ``at`` stamp; False unless *expected_owner* still holds it."""
    if not isinstance(claim, dict) or claim.get("by") != expected_owner:
        return False
    claim["at"] = _hermes_now().isoformat()
    save_jobs(jobs)
    return True


def heartbeat_run_claim(job_id: str, *, expected_owner: str) -> bool:
    """Refresh a one-shot's ``run_claim`` timestamp while its run is alive, so an expired claim
    really means the claiming process died. Compare-and-refresh on ``expected_owner`` stops a stale
    runner from extending a claim another process has since taken over.

    Called periodically from the scheduler's run monitor (#62002) so a legitimately long run keeps its claim
    fresh: an expired claim then really does mean "the claiming process died", and neither another process's
    tick nor this process's own next tick will re-dispatch or stale-remove the job while the run is in
    flight. mark_job_run() clears the claim on completion.
    """
    def apply(jobs, _i, job):
        if job.get("schedule", {}).get("kind") != "once":
            return False
        return _refresh_claim(jobs, job.get("run_claim"), expected_owner)

    return _with_job(job_id, apply, False)


def clear_run_claim(job_id: str) -> bool:
    """Clear a one-shot's ``run_claim`` when dispatch itself fails: such a job never reaches
    mark_job_run, so the stale claim would block re-dispatch until the TTL expires.

    Calling this on every early-exit path restores the "the job stays due and will fire on the next healthy
    tick" invariant that the scheduler comment promises (#86522).
    """
    def apply(jobs, _i, job):
        if job.get("schedule", {}).get("kind") != "once" or job.get("run_claim") is None:
            return False  # recurring, or already cleared
        job["run_claim"] = None
        save_jobs(jobs)
        return True

    return _with_job(job_id, apply, False)


def advance_next_runs(job_ids) -> int:
    """Batch form of :func:`advance_next_run`: one load + at most one save for the whole due set;
    one-shot/unknown ids are skipped. Returns the count advanced. Persisted once at the end, so a
    crash mid-batch re-fires the whole set on restart rather than a prefix (sub-10ms window)."""
    ids = set(job_ids)
    if not ids:
        return 0
    with _jobs_lock():
        jobs = load_jobs()
        now = _hermes_now().isoformat()
        advanced = 0
        for job in jobs:
            if (
                job["id"] not in ids
                or (is_terminal_job(job) and not _is_recoverable_error_job(job))
                or job.get("schedule", {}).get("kind") not in {"cron", "interval"}
            ):
                continue
            new_next = compute_next_run(job["schedule"], now)
            if new_next and new_next != job.get("next_run_at"):
                job["next_run_at"] = new_next
                advanced += 1
        if advanced:
            save_jobs(jobs)
        return advanced


def advance_next_run(job_id: str) -> bool:
    """Advance a recurring job's next_run_at BEFORE run_job() so a mid-run crash cannot re-fire it
    on restart (at-most-once for recurring jobs — one missed run beats a crash-loop burst).
    One-shots are left unchanged so they can retry. Returns True if next_run_at was advanced."""
    # >= 1 (not == 1): duplicate ids in a corrupted file all advance; still report the advance.
    return advance_next_runs([job_id]) >= 1


def _machine_id() -> str:
    """Claim attribution/debugging id (NOT correctness — that comes from the file lock and the
    fresh-claim check): ``HERMES_MACHINE_ID`` if set, else hostname:pid."""
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
    job_id: str, *, claim_ttl_seconds: int = 300, force: bool = False, return_job: bool = False,
) -> Union[bool, Dict[str, Any]]:
    """Atomically claim a job for one external 'fire' (multi-machine at-most-once); True iff THIS
    caller won (``CronScheduler.fire_due``: exactly one of N replicas runs a job). Under the
    fence + file lock: reject missing/terminal/paused jobs unless ``force`` (explicit manual
    fire, which also resumes the job atomically; external callbacks must leave it false so a
    stale callback cannot resurrect a paused job). Lose if a claim younger than
    ``claim_ttl_seconds`` exists (the TTL lets another fire reclaim after a crash; mark_job_run
    clears the claim). Otherwise stamp ``fire_claim`` and, for recurring jobs, advance
    ``next_run_at`` so a stale re-delivery cannot re-fire."""
    def apply(jobs, _i, job):
        if is_terminal_job(job) and not _is_recoverable_error_job(job):
            return False
        # Both enabled and pause markers must clear — a half-paused record must not claim. ``force``
        # (Trigger-now on a paused job) bypasses the gate and atomically resumes the job below.
        if not force and not is_job_runnable(job):
            return False
        now = _hermes_now()
        if _claim_is_live(job.get("fire_claim"), now, claim_ttl_seconds):
            return False  # someone holds a fresh claim
        if force:
            _activate_job_record(job)
        # Per-acquisition token: a process may legitimately reclaim its own stale lease, and the
        # previous runner must not heartbeat the new claim merely because hostname + PID match.
        job["fire_claim"] = {"at": now.isoformat(), "by": f"{_machine_id()}:{uuid.uuid4().hex}"}
        if job.get("schedule", {}).get("kind") in {"cron", "interval"}:
            nxt = compute_next_run(job["schedule"], now.isoformat())
            if nxt:
                job["next_run_at"] = nxt
        save_jobs(jobs)
        return copy.deepcopy(job) if return_job else True

    return _under_fire_fence(job_id, lambda: _with_job(job_id, apply, False))


def heartbeat_fire_claim(job_id: str, *, expected_owner: str) -> bool:
    """Refresh an active ``fire_claim`` without extending another owner's lease: an execution may
    outlive the TTL, and the owner check stops a stale runner from refreshing a recovered claim."""
    def apply(jobs, _i, job):
        return _refresh_claim(jobs, job.get("fire_claim"), expected_owner)

    return _under_fire_fence(job_id, lambda: _with_job(job_id, apply, False))


# Completed one-shots are retained in jobs.json (final status stays inspectable) and pruned by
# _sweep_completed_oneshots once they age out.
COMPLETED_ONESHOT_RETENTION_DAYS = 7


def _cron_config_number(key: str, default: Any, cast: Callable[[Any], Any]) -> Any:
    """Read ``cron.<key>`` from config as *cast*, falling back to *default* on any failure."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        return cast(cron_cfg.get(key, default))
    except Exception:
        return cast(default)


def _completed_oneshot_retention_days() -> float:
    """``cron.completed_retention_days``; non-positive disables the sweep (records kept forever)."""
    return _cron_config_number("completed_retention_days", COMPLETED_ONESHOT_RETENTION_DAYS, float)


def _sweep_completed_oneshots(
    raw_jobs: List[Dict[str, Any]], now: datetime, *, removed_ids: Optional[Set[str]] = None,
) -> bool:
    """Prune completed one-shot records past retention (in place; True when anything was removed).
    Removed ids go into *removed_ids* so save_jobs's shrink-merge guard allows the delete. Age is
    measured from ``last_run_at``; a record without a parseable one is kept (never guess into
    deletion)."""
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
            if (schedule.get("kind") if isinstance(schedule, dict) else None) != "once":
                continue
            last_run = rj.get("last_run_at")
            last_run_dt = _parse_aware(last_run) if isinstance(last_run, str) else None
            if last_run_dt is None or last_run_dt >= cutoff:
                continue
            raw_jobs.remove(rj)
            removed = True
            rid = rj.get("id")
            if removed_ids is not None and rid:
                removed_ids.add(str(rid))
            logger.info(
                "Job '%s': pruning completed one-shot record (finished %s, retention %.1f days)",
                rj.get("name", rj.get("id", "?")), last_run, retention_days)
        except Exception:
            logger.debug(
                "Retention sweep skipped malformed job record %r", rj.get("id", "?"), exc_info=True)
    return removed


# --- Due scan ---

def get_due_jobs() -> List[Dict[str, Any]]:
    """Return all jobs due now. A recurring job more than one period stale (gateway down, or a run
    overran the interval) has its backlog collapsed — next_run_at fast-forwards so nothing
    burst-fires — but still fires ONCE now (via mark_job_run, consuming one ``repeat.times`` run),
    avoiding the perpetual-defer loop for runs longer than interval + grace.

    This prevents the perpetual-defer loop (#33315) where a job whose runtime exceeds ``interval + grace``
    would be skipped forever.
    """
    with _jobs_lock():
        return _get_due_jobs_locked()


@dataclass
class _DueScan:
    """Mutable state threaded through one due scan: the raw store records plus what to persist."""

    raw_jobs: List[Dict[str, Any]]
    now: datetime
    needs_save: bool = False
    removed: Set[str] = field(default_factory=set)

    def find(self, job_id: Any) -> Optional[Dict[str, Any]]:
        return next((rj for rj in self.raw_jobs if rj["id"] == job_id), None)

    def persist(self, job_id: Any, **fields: Any) -> None:
        """Write *fields* onto the raw record for *job_id* and flag a save (no-op if missing)."""
        rj = self.find(job_id)
        if rj is not None:
            rj.update(fields)
            self.needs_save = True

    def retire(self, job_id: Any) -> None:
        """Drop the raw record for *job_id* as an intentional removal."""
        rj = self.find(job_id)
        if rj is not None:
            self.raw_jobs.remove(rj)
            self.removed.add(str(job_id))
            self.needs_save = True


def _normalize_due_scan_records(raw_jobs: List[Dict[str, Any]]) -> bool:
    """Repair malformed store records in place BEFORE the due scan keys off them: a missing ``id``
    (older writers used ``job_id``), non-dict ``schedule``, or non-ISO timestamp used to abort the
    whole scan before save_jobs(), freezing the scheduler in a fast-forward loop."""
    changed = False
    for rj in raw_jobs:
        if not rj.get("id"):
            rj["id"] = rj.pop("job_id", None) or uuid.uuid4().hex[:12]
            changed = True
        if not isinstance(rj.get("schedule"), dict):
            rj["schedule"] = {}
            changed = True
        for key in ("next_run_at", "last_run_at"):
            value = rj.get(key)
            if value is not None and _parse_aware(value) is None:
                rj.pop(key, None)  # the "no next_run_at" path recomputes
                changed = True
    return changed


def _self_disable_half_paused(job: Dict[str, Any], scan: _DueScan) -> None:
    """Self-heal enabled=true with pause markers: the operator believes the job is frozen while the
    scheduler would still fire it. Force enabled=false so listings are honest; logged loudly since
    pause_job sets both fields atomically, so this should be rare."""
    jid = job.get("id")
    logger.error(
        "Job '%s' (%s) has pause markers while enabled=true; "
        "self-disabling so it cannot fire (pause must be authoritative).",
        job.get("name", jid), jid)
    rj = scan.find(jid)
    if rj is None:
        return
    rj.update(enabled=False, state="paused")
    if not rj.get("paused_at"):
        rj["paused_at"] = scan.now.isoformat()
    if not rj.get("paused_reason"):
        rj["paused_reason"] = "auto-disabled: enabled+paused contradiction"
    scan.needs_save = True


def _recover_missing_next_run(job: Dict[str, Any], scan: _DueScan) -> Optional[str]:
    """Recompute and persist a missing ``next_run_at``; None when unrecoverable. One-shots use the
    grace window; recurring jobs only get here after a direct jobs.json edit bypassed add_job(),
    and would otherwise be silently skipped forever."""
    schedule = job.get("schedule", {})
    kind = schedule.get("kind")
    recovered_next = _recoverable_oneshot_run_at(
        schedule, scan.now, last_run_at=job.get("last_run_at"))
    recovery_kind = "one-shot" if recovered_next else None
    if not recovered_next and kind in {"cron", "interval"}:
        recovered_next = compute_next_run(schedule, scan.now.isoformat())
        if recovered_next:
            recovery_kind = kind
    if not recovered_next:
        return None
    job["next_run_at"] = recovered_next
    logger.info(
        "Job '%s' had no next_run_at; recovering %s run at %s",
        job.get("name", job.get("id", "?")), recovery_kind, recovered_next)
    scan.persist(job["id"], next_run_at=recovered_next)
    return recovered_next


@dataclass
class _DueJob:
    """One candidate under evaluation: its record, schedule and the stored next_run in raw/aware
    form."""

    job: Dict[str, Any]
    scan: _DueScan
    next_run: str  # stored ISO string, compared string-exact against manual_run_at
    raw_next_run_dt: datetime  # as stored (may carry a pre-migration offset)
    next_run_dt: datetime  # normalized to the configured tz

    @property
    def schedule(self) -> Dict[str, Any]:
        return self.job.get("schedule", {})

    @property
    def kind(self) -> Optional[str]:
        return self.schedule.get("kind")

    @property
    def label(self) -> Any:
        return self.job.get("name", self.job.get("id", "?"))

    def recompute_next(self) -> Optional[str]:
        return compute_next_run(self.schedule, self.scan.now.isoformat())


def _repair_timezone_shifted_cron(d: _DueJob) -> bool:
    """Repair a cron job whose stored offset no longer matches now's (TZ migration).

    next_run_at is an absolute instant but the expr means local wall clock, so a TZ change can make
    it look due hours early. If the stored wall clock is still in the future, recompute so we fire
    at the intended local time. True when re-anchored (caller skips this tick). TRADE-OFF: a DST
    offset change meeting the same conditions SKIPS the pending occurrence; accepted as rare."""
    now = d.scan.now
    if not (
        d.next_run_dt <= now
        and _timezone_offset_mismatch(d.raw_next_run_dt, now)
        and _stored_wall_clock_is_future(d.raw_next_run_dt, now)
    ):
        return False
    new_next = d.recompute_next()
    if not new_next:
        return False
    logger.info(
        "Job '%s' next_run_at offset changed (%s -> %s). "
        "Recomputing cron run to preserve local wall-clock intent: %s",
        d.label, d.raw_next_run_dt.utcoffset(), now.utcoffset(), new_next)
    d.scan.persist(d.job["id"], next_run_at=new_next)
    return True


def _rearm_stale_error_recurring(d: _DueJob) -> datetime:
    """Re-arm a recurring job wedged in persisted last_status=error; returns the effective
    next_run_dt.

    Such a job errored, mark_job_run parked next_run_at in the future, and nothing re-dispatched it
    (the in-memory stale-claim sweep cannot see it). Interval jobs re-arm to now (always a legal
    fire); cron jobs re-arm to the next LEGAL occurrence, since re-arming to now would fire at times
    the expression excludes. A correctly-parked cron value is left as-is.
    """
    now = d.scan.now
    if not (
        d.kind in ("cron", "interval")
        and d.next_run_dt > now
        and _job_is_stale_error_recurring(d.job, d.schedule, now)
    ):
        return d.next_run_dt
    if d.kind == "interval":
        recovered_next = now.isoformat()
        recovered_next_dt: Optional[datetime] = now
    else:
        recovered_next = d.recompute_next()
        recovered_next_dt = _parse_aware(recovered_next) if recovered_next else None
    if not (recovered_next and recovered_next_dt is not None and recovered_next_dt < d.next_run_dt):
        return d.next_run_dt
    jid = d.job.get("id")
    logger.warning(
        "cron.persisted_error.recovered job='%s' id=%s — recurring "
        "job wedged in stale last_status=error without re-firing for "
        "a full cadence; re-arming next_run_at to %s so it re-dispatches without force-run/resume",
        d.job.get("name", jid), jid, recovered_next)
    _record_persisted_error_recovery(d.job, d.next_run)
    d.job["next_run_at"] = recovered_next
    d.scan.persist(jid, next_run_at=recovered_next)
    return recovered_next_dt


def _reanchor_stale_cron(d: _DueJob) -> bool:
    """Stale-schedule guard for a due cron instant; True when re-anchored without firing.

    A direct edit of schedule.expr leaves next_run_at on the old lattice, so re-anchor first (from
    the current expr, so this converges). An offset-representation migration also moves a legacy
    instant off the lattice, and re-anchoring THAT swallowed a due occurrence — so classify, and
    let
    the migration case fall through to fire ONCE (at-most-once holds: nothing re-reads the legacy
    instant after advance/mark_job_run rewrites it)."""
    stale_class = _classify_stale_cron_next_run(d.schedule, d.raw_next_run_dt, d.next_run_dt)
    if stale_class == STALE_CRON_EXPR_EDIT:
        new_next = d.recompute_next()
        logger.info(
            "Job '%s' next_run_at %s does not match its current "
            "cron expression %r (direct jobs.json edit?); re-anchoring to %s without firing.",
            d.label, d.next_run, d.schedule.get("expr"), new_next)
        if new_next:
            d.scan.persist(d.job["id"], next_run_at=new_next)
        return True
    if stale_class == STALE_CRON_TIMEZONE_MIGRATION:
        logger.warning(
            "cron.timezone_migration.catch_up job='%s' id=%s expr=%r "
            "stored=%s normalized=%s — stored next_run_at carries a "
            "pre-migration UTC offset (%s, now %s) and is a legal "
            "occurrence at its own wall clock; firing the due run instead of re-anchoring past it.",
            d.label, d.job.get("id"), d.schedule.get("expr"), d.next_run,
            d.next_run_dt.isoformat(), d.raw_next_run_dt.utcoffset(), d.scan.now.utcoffset())
        _record_timezone_migration_catchup(d.job, d.raw_next_run_dt, d.next_run_dt)
    return False


def _fast_forward_missed_recurring(d: _DueJob, grace: int) -> None:
    """Recurring job past its grace window: skip the accumulated misses, fire once now.

    The fast-forward is persisted immediately — NOT redundant with advance_next_run/mark_job_run:
    it
    protects the crash window before mark_job_run and covers the external fire_due path, which never
    calls advance_next_run. mark_job_run re-anchors on completion, so the value is provisional.
    """
    if (d.scan.now - d.next_run_dt).total_seconds() <= grace:
        return
    new_next = d.recompute_next()
    if not new_next:
        return
    logger.info(
        "Job '%s' missed its scheduled time (%s, grace=%ds). "
        "Running now; next run provisionally set to: %s (re-anchored on completion)",
        d.label, d.next_run, grace, new_next)
    d.scan.persist(d.job["id"], next_run_at=new_next)
    record_catch_up_occurrence()


def _retire_expired_oneshot(d: _DueJob) -> bool:
    """One-shot grace gate; True when the job must not fire this tick.

    A one-shot beyond the grace window must never fire (create/update/resume reject such schedules
    and recovery never revives them; only the due scan used to dispatch them hours late). With no
    claim stamped, retire it with a diagnostic (never silently delete). A claim may mean a run is
    still in flight elsewhere — skip but keep the record so its mark_job_run can land."""
    if (d.scan.now - d.next_run_dt).total_seconds() <= ONESHOT_GRACE_SECONDS:
        return False
    if not (d.job.get("run_claim") or d.job.get("fire_claim")):
        _write_missed_oneshot_diagnostic(d.job, d.next_run)
        d.scan.retire(d.job["id"])
    return True


def _oneshot_dispatch_limit_reached(job: Dict[str, Any], scan: _DueScan) -> bool:
    """One-shot dispatch-limit guard; True when the job must not fire this tick.

    A finite one-shot claimed via claim_dispatch() whose tick died before mark_job_run has
    completed >= times while still looking due. Remove it instead of re-firing — unless THIS
    process is still running it (a run outliving the run_claim TTL is slow, not stale)."""
    repeat = job.get("repeat") or {}
    times = repeat.get("times")
    completed = repeat.get("completed", 0)
    if times is None or times <= 0 or completed < times:
        return False
    name = job.get("name", job.get("id", "?"))
    # A live run must never have its job record deleted underneath it (#62002): a run that outlives the
    # run_claim TTL (stream stall, laptop asleep mid-run) satisfies the same completed >= times +
    # expired-claim condition as a dead tick, but mark_job_run() still needs the record to land last_run_at
    # / last_status / last_delivery_error. If this process is still running the job, it is slow, not stale —
    # keep the entry and skip.
    if _job_running_in_this_process(job.get("id", "")):
        logger.info(
            "Job '%s': dispatch limit reached (%d/%d) but its run is still in flight in this "
            "process — keeping entry",
            name, completed, times)
        return True
    if job.get("last_run_at") is not None:
        # A record with last_run_at completed a real run and was re-armed without a budget reset
        # (old build or hand edit) — not the dead-tick case; warn so the removal leaves a trace.
        logger.warning(
            "Job '%s': one-shot dispatch limit reached (%d/%d) on a record that already completed "
            "a run (last_run_at=%s) — removing it WITHOUT firing. This record was re-armed "
            "without a budget reset (pre-#93615 store or hand edit); re-run it with "
            "'hermes cron resume <job> --run-now' (#93524).",
            name, completed, times, job.get("last_run_at"))
    else:
        logger.info(
            "Job '%s': one-shot dispatch limit reached (%d/%d) — removing stale due entry",
            name, completed, times)
    scan.retire(job["id"])
    # The claimed run never completed here by definition — leave an operator-visible diagnostic.
    _write_wedged_oneshot_diagnostic(job)
    return True


def _evaluate_due_job(job: Dict[str, Any], scan: _DueScan, run_claim_ttl: float) -> bool:
    """Decide whether one enabled, non-terminal job fires this tick, persisting any repairs.
    Ordering matters: recover missing next_run_at, repair timezone shifts, re-arm stale-error
    recurring jobs; then once due: re-anchor stale cron instants, fast-forward missed recurring
    runs, retire/guard one-shots, and finally stamp the run claim / dispatch record."""
    now = scan.now
    # Cross-process guard: another process's live one-shot run_claim (younger than TTL) — do NOT
    # re-dispatch. Malformed/future-dated claims (clock/TZ skew) count as stale, never eternally
    # fresh.
    if (
        job.get("schedule", {}).get("kind") == "once"
        and _claim_is_live(job.get("run_claim"), now, run_claim_ttl)
    ):
        return False

    next_run = job.get("next_run_at") or _recover_missing_next_run(job, scan)
    if not next_run:
        return False
    raw_next_run_dt = datetime.fromisoformat(next_run)
    d = _DueJob(job, scan, next_run, raw_next_run_dt, _ensure_aware(raw_next_run_dt))
    kind = d.kind
    recurring = kind in {"cron", "interval"}
    # Intentionally string-exact on raw stored values: trigger_job stamps the SAME isoformat string
    # into both fields, and any rewrite of next_run_at (edit, re-anchor, fire-claim advance) must
    # invalidate the marker. Do not "fix" this with _ensure_aware normalization.
    manual_run = job.get("manual_run_at") == next_run
    if kind == "cron" and not manual_run and _repair_timezone_shifted_cron(d):
        return False
    d.next_run_dt = _rearm_stale_error_recurring(d)
    if d.next_run_dt > now:
        return False

    if not manual_run and kind == "cron" and _reanchor_stale_cron(d):
        return False
    grace = _compute_grace_seconds(d.schedule)
    if not manual_run and recurring:
        _fast_forward_missed_recurring(d, grace)
    if kind == "once":
        if _retire_expired_oneshot(d) or _oneshot_dispatch_limit_reached(job, scan):
            return False
        # Durably claim the one-shot for the DURATION of its run: a second scheduler process on the
        # same HERMES_HOME must not re-dispatch it while in flight, and advancing next_run_at by a
        # fixed window is not enough for a run that outlives a tick. The other process sees the
        # fresh claim and skips; mark_job_run() clears it. The TTL only covers a tick that DIES.
        claim = {"at": now.isoformat(), "by": _machine_id()}
        job["run_claim"] = claim
        scan.persist(job["id"], run_claim=claim)

    # Missed-run visibility: persist scheduled-vs-actual timing so separate CLI processes can show a
    # late catch-up. Recurring only — expired one-shots were retired above; manual triggers aren't
    # late.
    if not manual_run and recurring:
        lateness = max(0.0, (now - d.next_run_dt).total_seconds())
        # See #99879.
        dispatch_stamp = {
            "scheduled_at": next_run,
            "dispatched_at": now.isoformat(),
            "lateness_seconds": round(lateness, 1),
            "kind": _classify_dispatch_lateness(lateness, grace),
        }
        job["last_dispatch"] = dispatch_stamp
        scan.persist(job["id"], last_dispatch=dispatch_stamp)
    return True


def _get_due_jobs_locked() -> List[Dict[str, Any]]:
    """Inner implementation of get_due_jobs(); must be called with _jobs_lock held."""
    raw_jobs = load_jobs()
    scan = _DueScan(raw_jobs, _hermes_now())
    scan.needs_save = _normalize_due_scan_records(raw_jobs)
    jobs = [_apply_skill_fields(j) for j in copy.deepcopy(raw_jobs)]
    # One-shot run-claim TTL, resolved once per scan (see _oneshot_run_claim_ttl_seconds).
    run_claim_ttl = _oneshot_run_claim_ttl_seconds()

    # Retention sweep: completed one-shots are kept for inspection but must not accumulate forever.
    if _sweep_completed_oneshots(raw_jobs, scan.now, removed_ids=scan.removed):
        scan.needs_save = True
        jobs = [j for j in jobs if scan.find(j.get("id")) is not None]

    due = []
    for job in jobs:
        # Per-job containment: one malformed record must never abort the whole scan. Normalization
        # above repairs known shapes; this catches FUTURE variants so healthy siblings still
        # run/persist.
        try:
            if is_terminal_job(job) and not _is_recoverable_error_job(job):
                continue
            if not job.get("enabled", True):
                continue
            if _has_pause_marker(job):
                _self_disable_half_paused(job, scan)
                continue
            if _evaluate_due_job(job, scan, run_claim_ttl):
                due.append(job)
        except Exception:
            logger.exception(
                "Skipping malformed cron job %r during due scan",
                job.get("name") or job.get("id") or "?")

    if scan.needs_save:
        save_jobs(raw_jobs, removed_ids=scan.removed or None)
    return due


# --- Run output ---

# Per-run output files (`cron/output/<job>/<timestamp>.md`) are capped so a frequent job can't fill
# the disk.
# Unlike the quick-snapshot store (`hermes_cli.backup`, capped at 20) it had no retention, so a
# frequently-scheduled job on a long-running deploy accumulated one file per run forever and could fill the
# disk (#52383). Keep the most recent N files per job; a non-positive value disables pruning (opt-out).
_CRON_OUTPUT_DEFAULT_KEEP = 50


def _cron_output_keep() -> int:
    """Per-job output-file retention cap (``cron.output_retention``)."""
    return _cron_config_number("output_retention", _CRON_OUTPUT_DEFAULT_KEEP, int)


def _prune_job_output(job_output_dir: Path, keep: int) -> int:
    """Remove the oldest ``*.md`` run-output files beyond *keep*; returns count deleted. Filenames
    are timestamps, so a reverse lexical sort is newest-first. Non-positive *keep* disables pruning;
    failures are swallowed so they can never break output saving."""
    if keep <= 0:
        return 0
    try:
        files = sorted(
            (f for f in job_output_dir.glob("*.md") if f.is_file()),
            key=lambda f: f.name, reverse=True)
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
    output_file = job_output_dir / f"{_hermes_now().strftime('%Y-%m-%d_%H-%M-%S')}.md"
    atomic_write_text(output_file, output, tmp_prefix=".output_")
    _secure_file(output_file)
    # Bound per-job output growth so long-running deploys don't fill the disk (#52383).
    _prune_job_output(job_output_dir, _cron_output_keep())
    return output_file


# --- Skill reference rewriting (curator integration) ---

def _canonical_skill_ref(raw: Any) -> str:
    """Reduce a job skill reference (possibly an absolute path) to the bare name the curator matches
    on, resolving as the scheduler does — otherwise a path-referencing job's skill looks
    unreferenced and gets archived. Falls back to plain cleanup so a broken import can never lose
    a name."""
    value = str(raw or "").strip()
    if not value:
        return ""
    try:
        from agent.skill_utils import normalize_skill_lookup_name
        value = normalize_skill_lookup_name(value) or value
    except Exception:
        logger.debug("referenced_skill_names: could not normalize skill ref %r", raw, exc_info=True)
    return value.strip().lstrip("/")


def referenced_skill_names() -> Set[str]:
    """Skill names referenced by ANY cron job, deliberately including paused/disabled ones (resuming
    must still find them); the curator protects these from inactivity archival. Canonicalized as the
    scheduler does, so absolute paths are protected too. A corrupt store yields an empty set."""
    try:
        jobs = load_jobs()
    except Exception:
        logger.debug("referenced_skill_names: failed to load cron jobs", exc_info=True)
        return set()
    return {
        cleaned
        for job in jobs
        if isinstance(job, dict)
        for name in _normalize_skill_list(job.get("skill"), job.get("skills"))
        if (cleaned := _canonical_skill_ref(name))
    }


def rewrite_skill_refs(
    consolidated: Optional[Dict[str, str]] = None, pruned: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Rewrite cron job skill references after a curator consolidation pass (a job listing a
    consolidated/pruned skill would otherwise run without it). Consolidated names map to their
    umbrella target without duplication, pruned names are dropped, ordering is preserved, and the
    legacy ``skill`` field is realigned. Returns ``{"rewrites": [{job_id, job_name, before,
    after, mapped, dropped}, ...], "jobs_updated": N, "jobs_scanned": M}``. Load/save exceptions
    propagate."""
    consolidated = dict(consolidated or {})
    # A skill listed in both wins as "consolidated" — it has a target, the more useful outcome.
    pruned_set = set(pruned or []) - set(consolidated.keys())
    if not consolidated and not pruned_set:
        return {"rewrites": [], "jobs_updated": 0, "jobs_scanned": 0}

    with _jobs_lock():
        jobs = load_jobs()
        rewrites: List[Dict[str, Any]] = []
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
            rewrites.append({
                "job_id": job.get("id"),
                "job_name": job.get("name") or job.get("id"),
                "before": list(skills_before),
                "after": list(new_skills),
                "mapped": mapped,
                "dropped": dropped,
            })
        if rewrites:
            save_jobs(jobs)
            logger.info("Curator rewrote skill references in %d cron job(s)", len(rewrites))
        return {"rewrites": rewrites, "jobs_updated": len(rewrites), "jobs_scanned": len(jobs)}


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def clear_drift_alerted(job_id: str) -> None:
    """Clear the drift alert-dedup marker (resolution matches again)."""
    _set_alert_flag(job_id, "drift_alerted", False)
# ---- END PLUGIN-COMPAT ----
