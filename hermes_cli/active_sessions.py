"""Cross-process active chat session leases.

The session database records persisted conversations; this module records
currently open chat surfaces, including idle CLI/TUI sessions that have not
written a transcript row yet.
"""

from __future__ import annotations

import json
import logging
import collections
import math
import os
import time
import uuid
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from hermes_constants import get_default_hermes_root, get_hermes_home
from utils import atomic_json_write

logger = logging.getLogger(__name__)


class ActiveSessionRegistryError(RuntimeError):
    """The liveness registry could not prove a safe ownership decision."""


def coerce_max_concurrent_sessions(value: Any, key: str = "max_concurrent_sessions") -> Optional[int]:
    """Return a positive integer cap, or None when disabled/invalid."""
    if value is None:
        return None
    try:
        if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
            raise ValueError(value)
        parsed = int(value.strip(), 10) if isinstance(value, str) else int(value)
    except (TypeError, ValueError):
        logger.warning(
            "Ignoring invalid %s=%r (expected a positive integer; 0/null disables)", key, value
        )
        return None
    return parsed if parsed > 0 else None


def resolve_max_concurrent_sessions(config: Any) -> Optional[int]:
    """Resolve top-level max_concurrent_sessions with gateway.* fallback."""
    raw: Any = None
    key = "max_concurrent_sessions"
    if isinstance(config, dict):
        if "max_concurrent_sessions" in config:
            raw = config.get("max_concurrent_sessions")
        else:
            gateway_cfg = config.get("gateway")
            if isinstance(gateway_cfg, dict):
                raw = gateway_cfg.get("max_concurrent_sessions")
                key = "gateway.max_concurrent_sessions"
    else:
        raw = getattr(config, "max_concurrent_sessions", None)
    return coerce_max_concurrent_sessions(raw, key=key)


def format_age(seconds: float) -> str:
    minutes = max(0, int(seconds // 60))
    if minutes < 60:
        return f"{minutes}m"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h" if not minutes else f"{hours}h{minutes}m"


def summarize_holders(entries: list[dict[str, Any]]) -> str:
    """Compact "who is holding the slots" phrase, e.g. ``desktop x4, cli``."""
    if not entries:
        return ""
    counts = collections.Counter(str(e.get("surface") or "unknown") for e in entries)
    held = ", ".join(
        f"{surface} x{n}" if n > 1 else surface
        for surface, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    )
    started = [t for t in (_optional_float(e.get("started_at")) for e in entries) if t]
    if started:
        held += f", oldest {format_age(time.time() - min(started))} ago"
    return held


def active_session_limit_message(
    active_count: int, max_sessions: int, entries: Optional[list[dict[str, Any]]] = None
) -> str:
    # Name the holders: slots are shared across CLI, desktop/TUI and gateway,
    # so the rejected surface is usually NOT the one squatting on them.
    held = summarize_holders(entries or [])
    detail = f" Held by: {held}." if held else ""
    return (
        f"Hermes is at the active session limit ({active_count}/{max_sessions})."
        f"{detail} Try again when another session finishes."
    )


# Machine-readable refusal reasons (the reason is the contract, the message is for
# people). Capacity = "busy, come back later"; ownership = "a live owner exists and
# writing would interleave with theirs".
SESSION_NOT_OWNED = "SESSION_NOT_OWNED"
MAX_CONCURRENT_SESSIONS = "MAX_CONCURRENT_SESSIONS"
# Ownership could not be PROVEN (registry unreadable/corrupt). Deliberately distinct
# from SESSION_NOT_OWNED: treating "can't tell" as a go-ahead is the fail-open hole
# that let two writers share one session.
# Distinct from SESSION_NOT_OWNED on purpose -- "someone else owns this" and "I cannot tell who owns this"
# call for different operator action, and collapsing the second into a silent go-ahead is exactly the
# fail-open hole that let two writers share one session (#94595 review, blocker 2).
SESSION_COORDINATION_UNAVAILABLE = "SESSION_COORDINATION_UNAVAILABLE"
# Preemptible leases (#auto-yield -> preemptible leases): a LIVE foreign holder is either
# stolen from atomically (idle / stalled turn / past-grace auto work / heartbeat-dead) or
# refused fast. SESSION_BUSY means "a live owner is mid-turn or protected; retry or preempt
# later" — machine-distinguishable from the SESSION_NOT_OWNED a heartbeat-less LEGACY
# (pre-steal) holder still produces, so callers can shape retries and messaging per class.
SESSION_BUSY = "SESSION_BUSY"
# Holder-side next-touch fence: the lease this process cached was taken over (or the
# registry no longer proves it ours). The session must close and reload from the DB;
# continuing to write on it is exactly the double-writer the fence exists to prevent.
SESSION_DISPLACED = "SESSION_DISPLACED"

# Advertised through the gateway. A module constant, not a config flag: it holds
# because try_acquire_active_session checks atomically, so it cannot drift from the
# enforcement without this file changing.
PER_SESSION_EXCLUSIVE_SUBMIT = True

# --- Cross-surface yield requests (#auto-yield patch) -----------------------------
# A requester refused with SESSION_NOT_OWNED by a LIVE pid on the same machine asks
# that owner to close its idle session by writing a sidecar request file next to the
# lease registry. The holder backend's yield watcher honors fresh requests for idle
# sessions (never mid-turn), then the requester re-claims. Requests are one-shot,
# expire after YIELD_REQUEST_TTL_S, and the pid+create-time pair in the request must
# match the current lease entry, so a stale or forged request cannot fence a live
# session out of its own ownership.
#
# Preemptible leases: this file channel is now a COMPAT path only — OLD-code requesters
# (pre-steal builds) keep their 8s write-and-poll dance and a restarted (fixed) holder
# honors it, but NEW requesters never write request files to acquire; they steal or
# refuse inside try_acquire_active_session's flock instead. Remove the channel once the
# fleet has converged.
YIELD_REQUEST_FILENAME = "yield_requests"
YIELD_REQUEST_TTL_S = 15.0
# Bump when the payload shape changes; poll_yield_requests leaves FRESH own-pid files of a
# FUTURE protocol alone (a newer build may need them) while still unlinking expired ones.
YIELD_REQUEST_PROTOCOL = 2


def _env_float(name: str, default: float) -> float:
    # Tolerant env parsing (mirrors tui_gateway._env.env_float): a bare float() would raise
    # at import on a typo and kill the process before it serves a command.
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "") or "").strip().lower() in ("1", "true", "yes", "on")


# --- Preemptible-lease tunables (#auto-yield -> preemptible leases) -----------------
# Steal-decision windows, all in epoch seconds, all env-overridable.
# HERMES_LEASE_HEARTBEAT_FRESH_S is floored at 2x the 1.5s maintenance tick + margin: a
# freshness window shorter than the tick would make every HEALTHY holder look stale
# between ticks (freshness-headroom rule).
HERMES_LEASE_HEARTBEAT_FRESH_S = max(2.0, _env_float("HERMES_LEASE_HEARTBEAT_FRESH_S", 6.0))
# A busy_kind='user' turn is protected while its activity clock (last visible streaming/
# tool progress, refreshed from the turn thread itself) is at most this old.
HERMES_LEASE_ACTIVITY_FRESH_S = _env_float("HERMES_LEASE_ACTIVITY_FRESH_S", 15.0)
# Past this activity age a 'user' turn counts as stalled (blocked approval, hung tool,
# silent compute) and becomes stealable — the owner-reported "UI looked done" shape.
# 0 disables the stall tier: a user turn is then never interruptible, however silent.
HERMES_LEASE_STALL_ACTIVITY_S = _env_float("HERMES_LEASE_STALL_ACTIVITY_S", 60.0)
# bg-review grace: observed review API calls run 65.6s/79.7s — 90s lets most reviews
# finish before a steal may interrupt them (best-effort work, never a hard guarantee).
HERMES_LEASE_AUTO_BUSY_GRACE_S = _env_float("HERMES_LEASE_AUTO_BUSY_GRACE_S", 90.0)
# An idle lease whose heartbeat is staler than this is stealable — the escape hatch that
# bounds a dead maintenance watcher (a heartbeat-writing holder runs new code whose
# admission fence bounds it at next touch, so stealing from it is safe).
HERMES_LEASE_HEARTBEAT_STALE_STEAL_S = _env_float("HERMES_LEASE_HEARTBEAT_STALE_STEAL_S", 30.0)
# Operator escape for LEGACY (pre-steal) holders — default OFF. A legacy holder has no
# displacement detector and short-circuits on its cached lease between turns, so stealing
# from it opens a blind double-writer; absence of heartbeat = unknown = fail-closed refuse.
HERMES_STEAL_LEGACY_HOLDER = _env_flag("HERMES_STEAL_LEGACY_HOLDER")
# Turn-path activity piggyback throttle: at most one non-blocking activity refresh per
# window, so streaming never pays a registry write per token.
HERMES_LEASE_ACTIVITY_REFRESH_MIN_S = _env_float("HERMES_LEASE_ACTIVITY_REFRESH_MIN_S", 2.5)


def _yield_request_dir(registry_home: str | Path | None = None) -> Path:
    return _registry_home(registry_home) / "runtime" / YIELD_REQUEST_FILENAME


def request_cross_surface_yield(
    session_id: str, entry: dict[str, Any], *, registry_home: str | Path | None = None,
    requested_at: float | None = None,
) -> bool:
    """Ask the live owner in ``entry`` to close its idle session for ``session_id``.

    Returns True when a request file was written. The holder honors it only while the
    request stays fresh (TTL) and the lease still matches pid + process_start_time —
    a request can never close a session that changed owners since it was written.

    ``requested_at`` (987d564112 repair): a REQUEUE must pass the ORIGINAL mint time or
    the fresh stamp below silently extends a busy chain's TTL on every hop — the opposite
    of what the old requeue comment claimed. New requests leave it None (mint now).
    """
    target = str(session_id or "")
    holder_pid = entry.get("pid")
    holder_start = _optional_float(entry.get("process_start_time"))
    if not target or holder_pid is None:
        return False
    req_dir = _yield_request_dir(registry_home)
    try:
        req_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "session_id": target,
            "holder_pid": int(holder_pid),
            "holder_process_start_time": holder_start,
            "requested_at": time.time() if requested_at is None else float(requested_at),
            "protocol": YIELD_REQUEST_PROTOCOL,
        }
        tmp = req_dir / f".{int(holder_pid)}.{uuid.uuid4().hex}.tmp"
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, req_dir / f"{int(holder_pid)}-{uuid.uuid4().hex[:8]}.json")
        return True
    except Exception:
        logger.debug("cross-surface yield request write failed", exc_info=True)
        return False


def poll_yield_requests(
    *, registry_home: str | Path | None = None, max_age_s: float = YIELD_REQUEST_TTL_S,
) -> list[dict[str, Any]]:
    """Fresh, well-formed requests targeting THIS process. Files addressed to another pid
    are LEFT in place for their rightful owner (every backend sweeps every home, so the
    first poller must not consume a foreign request); corrupt and expired files are removed
    regardless of protocol (an expired future-protocol file must not leak forever); a FRESH
    own-pid file of a future protocol is left in place for a future-version reader."""
    req_dir = _yield_request_dir(registry_home)
    mine: list[dict[str, Any]] = []
    try:
        candidates = list(req_dir.glob("*.json"))
    except Exception:
        return mine
    now = time.time()
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        file_pid: Optional[int] = None
        if isinstance(payload, dict):
            try:
                file_pid = int(payload.get("holder_pid") or 0)
            except (TypeError, ValueError):
                file_pid = None
        expired = (
            not isinstance(payload, dict)
            or now - float(payload.get("requested_at") or 0.0) > max_age_s
        )
        # Expired/corrupt determination comes FIRST, before the foreign/future-protocol
        # guards, so nothing downstream can keep a dead file alive on disk.
        if expired:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            continue
        if file_pid is not None and file_pid != os.getpid():
            continue  # not ours, still fresh: leave it for the addressed holder
        try:
            protocol = int(payload.get("protocol") or 1)
        except (TypeError, ValueError):
            protocol = 1
        if protocol > YIELD_REQUEST_PROTOCOL:
            continue  # fresh own-pid future-protocol file: leave it for its future reader
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        if file_pid == os.getpid():
            mine.append(payload)
    return mine


class ActiveSessionRefusal(str):
    """Refusal message (a ``str``, so callers are untouched) with a machine-readable ``reason``.

    #auto-yield patch: ``holder_entry`` carries the blocking lease entry (when known) so a
    same-machine requester can ask the live owner to yield an idle session instead of failing.
    """

    reason: str
    holder_entry: Optional[dict[str, Any]]

    def __new__(cls, message: str, reason: str, holder_entry: Optional[dict[str, Any]] = None):
        obj = super().__new__(cls, message)
        obj.reason = reason
        obj.holder_entry = holder_entry
        return obj


def format_refusal_stderr(message: str) -> str:
    """Keep the refusal contract across the one-shot CLI subprocess boundary."""
    reason = getattr(message, "reason", "")
    return f"hermes-refusal-reason: {reason}\n{message}" if reason else str(message)


def _is_same_writer(entry: dict[str, Any], metadata: Optional[dict[str, Any]]) -> bool:
    """True when an existing lease belongs to the very caller re-acquiring it.
    Identity is (pid, live_session_id): pid alone lets two live sessions in one process
    steal each other's lease; the live id alone lets another process with an equal id."""
    try:
        if int(entry.get("pid") or -1) != os.getpid():
            return False
    except (TypeError, ValueError):
        return False
    existing_live = str((entry.get("metadata") or {}).get("live_session_id") or "")
    incoming_live = str((metadata or {}).get("live_session_id") or "")
    return bool(existing_live and incoming_live) and existing_live == incoming_live


def _format_state_age(seconds: float) -> str:
    # Seconds precision under a minute (the phone renders "~4s mid-turn", not "~0m"),
    # minute granularity past that via the existing lease-age formatter.
    return f"{max(0, int(seconds))}s" if seconds < 60 else format_age(seconds)


def _holder_state_phrase(entry: dict[str, Any], now: float) -> str:
    """Truthful parenthetical about what the holder is doing (preemptible leases). The
    phone renders the refusal string verbatim as its 'Not sent' detail — the only
    UX channel a frozen client gives us, so it must never claim 'mid-turn' for a holder
    that may be idle with a lagging maintenance watcher."""
    heartbeat = _optional_float(entry.get("heartbeat_at"))
    if heartbeat is None:
        # Legacy pre-steal holder: no heartbeat field was ever published. Name the needed
        # operator action instead of a state we cannot know.
        return "; its backend must restart once to enable handover"
    busy_kind = entry.get("busy_kind") if entry.get("busy") else None
    activity = _optional_float(entry.get("activity_at"))
    if busy_kind == "user" and activity is not None:
        elapsed = now - activity
        if elapsed <= max(0.0, HERMES_LEASE_ACTIVITY_FRESH_S):
            return f", mid-turn ~{_format_state_age(elapsed)}"
        return f", no visible progress for ~{_format_state_age(elapsed)}"
    if busy_kind == "auto":
        since = _optional_float(entry.get("busy_since")) or now
        return f", background review ~{_format_state_age(now - since)}"
    if (now - heartbeat) > HERMES_LEASE_HEARTBEAT_FRESH_S:
        # Live pid but the lease's heartbeat lags: holder maintenance (not the user's
        # turn) is the thing that looks stuck — never call this 'mid-turn'.
        return ", lease stale — holder maintenance lagging"
    return ""


def session_already_owned_message(session_id: str, entry: dict[str, Any]) -> str:
    surface = str(entry.get("surface") or "another surface")
    pid = entry.get("pid")
    started = _optional_float(entry.get("started_at"))
    # NOTE: this "running Xm" is LEASE age, not turn state — never conflate the two (the
    # turn state lives in the busy/activity fields via _holder_state_phrase).
    age = f", running {format_age(time.time() - started)}" if started else ""
    state = _holder_state_phrase(entry, time.time())
    return (
        f"Session {session_id} already has a live owner ({surface}, pid {pid}{age}{state}). "
        "Only one surface at a time may run a session, because a second one would "
        "reason from a transcript that does not include the first one's work. "
        "Do not delete a live owner's lease to force a takeover."
    )


def _registry_home(registry_home: str | Path | None = None) -> Path:
    return Path(registry_home) if registry_home is not None else Path(get_hermes_home())


def _state_path(registry_home: str | Path | None = None) -> Path:
    return _registry_home(registry_home) / "runtime" / "active_sessions.json"


def _lock_path(registry_home: str | Path | None = None) -> Path:
    return _registry_home(registry_home) / "runtime" / "active_sessions.lock"


def _lease_paths(
    lease: Optional["ActiveSessionLease"] = None, registry_home: str | Path | None = None
) -> tuple[Path, Path]:
    if lease is not None and lease.state_path is not None and lease.lock_path is not None:
        return lease.state_path, lease.lock_path
    home = _registry_home(registry_home)
    return home / "runtime" / "active_sessions.json", home / "runtime" / "active_sessions.lock"


def _flock(fh, *, lock: bool, blocking: bool = True) -> None:
    """Exclusive whole-file lock/unlock on ``fh`` (fcntl on POSIX, msvcrt on Windows).
    ``blocking=False`` (acquire only) skips on contention via LOCK_NB — callers degrade
    to "try again on the next touch" instead of waiting behind a slow registry writer."""
    if os.name == "nt":
        import msvcrt
        if lock and not blocking:
            # msvcrt.locking has no LOCK_NB; a non-blocking acquire on Windows degrades
            # to the blocking form (callers on this platform accept the wait).
            blocking = True
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_LOCK if lock else msvcrt.LK_UNLCK, 1)
    else:
        import fcntl
        op = fcntl.LOCK_EX if lock else fcntl.LOCK_UN
        if lock and not blocking:
            op |= fcntl.LOCK_NB
        fcntl.flock(fh.fileno(), op)


class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+b")
        try:
            _flock(self._fh, lock=True)
        except Exception as exc:
            self._fh.close()
            self._fh = None
            raise RuntimeError("active session file lock unavailable") from exc
        return self

    def __exit__(self, exc_type, exc, tb):
        fh, self._fh = self._fh, None
        if fh is not None:
            with suppress(Exception):
                _flock(fh, lock=False)
            fh.close()


def _read_entries(path: Path, *, strict: bool = False) -> list[dict[str, Any]]:
    def invalid(what: str) -> ActiveSessionRegistryError:
        return ActiveSessionRegistryError(f"active session registry {what}: {path}")

    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return []
    except Exception as exc:
        if strict:
            raise invalid("unreadable") from exc
        logger.warning("Ignoring corrupt active session registry at %s", path)
        return []
    entries = data.get("entries") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        if strict:
            raise invalid("has invalid shape")
        return []
    valid = [entry for entry in entries if isinstance(entry, dict)]
    if not strict:
        return valid
    if len(valid) != len(entries):
        raise invalid("contains invalid entries")
    seen_leases: set[str] = set()
    for entry in valid:
        lease_id = entry.get("lease_id")
        # (problem-if-True predicate, message fragment) — checked lazily, in
        # this order, so an unhashable lease id is reported before the dup check.
        for bad, what in (
            (lambda: not _nonblank_str(lease_id), "an invalid lease id"),
            (lambda: lease_id in seen_leases, "a duplicate lease id"),
            (lambda: not _nonblank_str(entry.get("session_id")), "an invalid session id"),
            (lambda: _registry_pid(entry.get("pid")) <= 0, "an invalid pid"),
            (lambda: not _optional_isinstance(entry.get("surface"), str), "an invalid surface"),
            (lambda: not _optional_isinstance(entry.get("track_liveness"), bool), "an invalid liveness marker"),
            (lambda: not _optional_isinstance(entry.get("metadata"), dict), "invalid metadata"),
            (lambda: not _valid_process_start(entry.get("process_start_time")), "an invalid process start time"),
            # Preemptible-lease fields: absence stays VALID everywhere — entries written by
            # OLD code must parse under NEW rules and vice versa (additive-only schema).
            (lambda: "epoch" in entry and (isinstance(entry.get("epoch"), bool)
                                           or not isinstance(entry.get("epoch"), int)
                                           or entry["epoch"] < 1), "an invalid epoch"),
            (lambda: "busy" in entry and not isinstance(entry.get("busy"), bool), "an invalid busy marker"),
            (lambda: "busy_kind" in entry and entry.get("busy_kind") not in ("user", "auto"),
             "an invalid busy kind"),
            (lambda: "busy_detail" in entry and entry.get("busy_detail") is not None
             and not isinstance(entry.get("busy_detail"), str), "an invalid busy detail"),
            (lambda: any(_valid_process_start(entry.get(k)) is False
                         for k in ("busy_since", "activity_at", "heartbeat_at")),
             "an invalid lease clock"),
        ):
            if bad():
                raise invalid(f"contains {what}")
        seen_leases.add(lease_id)
    return valid


def _nonblank_str(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _optional_isinstance(v: Any, typ) -> bool:
    return v is None or isinstance(v, typ)


def _registry_pid(pid: Any) -> int:
    """Registry pid as int; 0 for bools, non-int/str, or unparseable values."""
    if isinstance(pid, bool) or not isinstance(pid, (int, str)):
        return 0
    try:
        return int(pid)
    except (TypeError, ValueError):
        return 0


def _valid_process_start(v: Any) -> bool:
    if v in (None, ""):
        return True
    parsed = _optional_float(v)
    return parsed is not None and math.isfinite(parsed)


def _write_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    atomic_json_write(path, {"entries": entries}, indent=None, sort_keys=True)


def _process_start_time(pid: int) -> Optional[float]:
    # Pair pid with create_time when psutil can read it, so a recycled pid does not
    # keep a stale lease alive indefinitely.
    try:
        import psutil  # type: ignore
        return float(psutil.Process(pid).create_time())
    except Exception:
        return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pid_liveness(pid: Any, process_start_time: Any = None, *, lenient: bool = False) -> Optional[bool]:
    """True/False for live/dead, or None when unknowable. ``lenient`` never returns None:
    an unparseable pid or failed existence probe counts as dead, an unreadable start as alive."""
    unknown_dead = False if lenient else None
    try:
        pid_int = int(pid)
    except (TypeError, ValueError):
        pid_int = 0
    if pid_int <= 0:
        return unknown_dead
    try:
        from gateway.status import _pid_exists
        exists = bool(_pid_exists(pid_int))
    except Exception:
        return unknown_dead
    if not exists:
        return False
    expected_start = _optional_float(process_start_time)
    if expected_start is None:
        return True
    current_start = _process_start_time(pid_int)
    if current_start is None:
        return True if lenient else None
    return abs(current_start - expected_start) < 0.001


def _prune_dead(entries: list[dict[str, Any]], *, strict: bool = False) -> list[dict[str, Any]]:
    """Keep entries whose owner is alive; tracked/strict entries must be provably so."""
    live: list[dict[str, Any]] = []
    for entry in entries:
        tracked = strict or bool(entry.get("track_liveness"))
        state = _pid_liveness(
            entry.get("pid"), entry.get("process_start_time"), lenient=not tracked
        )
        if state is None:
            raise ActiveSessionRegistryError("active session owner liveness is unknown")
        if state:
            live.append(entry)
    return live


@dataclass
class ActiveSessionLease:
    lease_id: str
    session_id: str
    surface: str
    enabled: bool = True
    released: bool = False
    # Pinned at acquisition: a lease taken under the root HERMES_HOME must release
    # against the same registry even inside a profile-home override, or phantom
    # leases fill the session cap.
    # See #85431.
    state_path: Optional[Path] = None
    lock_path: Optional[Path] = None
    track_liveness: bool = False
    # Fencing generation (preemptible leases): 1 on first acquire, prev+1 on every steal,
    # preserved on same-writer re-entrancy. A holder's cached lease is only valid while
    # the registry entry still carries THIS epoch — a bump means someone stole it.
    epoch: int = 1

    def release(self) -> None:
        if self.released or not self.enabled:
            return
        release_active_session(self)


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {str(k): v for k, v in metadata.items() if isinstance(k, str)}


def _drop_lease(
    state_path: Path, entries: list[dict[str, Any]], lease_id: str
) -> list[dict[str, Any]]:
    """Remove ``lease_id`` from ``entries``, writing the registry only if it was present."""
    kept = [e for e in entries if str(e.get("lease_id") or "") != lease_id]
    if len(kept) != len(entries):
        _write_entries(state_path, kept)
    return kept


def _holds_session(entries: list[dict[str, Any]], session_id: str) -> bool:
    target = str(session_id or "")
    return bool(target) and any(str(e.get("session_id") or "") == target for e in entries)


def _read_live_entries(
    state_path: Path, *, track_liveness: bool, warn: str,
) -> Optional[tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """``(raw, pruned)`` from the registry, or None when it is unreadable.

    Liveness-tracked callers re-raise instead (they must not proceed on an unprovable
    registry); untracked callers get ``warn`` logged and decide how to degrade.
    """
    try:
        raw_entries = _read_entries(state_path, strict=True)
        return raw_entries, _prune_dead(raw_entries, strict=track_liveness)
    except ActiveSessionRegistryError:
        if track_liveness:
            raise
        logger.warning(warn)
        return None


def _lease_entry(
    *, lease_id: str, session_id: str, surface: str,
    metadata: Optional[dict[str, Any]] = None, track_liveness: bool = False,
    epoch: int = 1, busy_kind: Optional[str] = None, busy_detail: Optional[str] = None,
) -> dict[str, Any]:
    now = time.time()
    entry: dict[str, Any] = {
        "lease_id": lease_id,
        "session_id": str(session_id),
        "surface": str(surface),
        "pid": os.getpid(),
        "process_start_time": _process_start_time(os.getpid()),
        "started_at": now,
        "updated_at": now,
        # Every NEW-code entry stamps epoch + heartbeat_at: epoch fences revalidation,
        # and a heartbeat-writing holder is by definition new code whose admission gate
        # bounds it at next touch (the property that makes the stale-heartbeat steal safe).
        "epoch": max(1, int(epoch)),
        "heartbeat_at": now,
    }
    if track_liveness:
        entry["track_liveness"] = True
    if metadata:
        entry["metadata"] = _clean_metadata(metadata)
    if busy_kind:
        # Busy trio + clocks (holder-published turn state). Absent on non-admission
        # acquires — the entry then reads "idle", which is exactly what it is.
        entry["busy"] = True
        entry["busy_kind"] = str(busy_kind)
        if busy_detail:
            entry["busy_detail"] = str(busy_detail)
        entry["busy_since"] = now
        entry["activity_at"] = now
    return entry


def _stealable_entry(existing: dict[str, Any], now: float) -> bool:
    """Preemptible-lease steal decision for a live foreign holder (post dead-holder
    pruning, under the caller's flock). Decision table, in precedence order:

    1. heartbeat_at ABSENT → NOT stealable (legacy pre-steal holder) unless
       HERMES_STEAL_LEGACY_HOLDER. A legacy holder has no displacement detector and
       short-circuits on its cached lease between turns; stealing would create a blind
       double-writer. Fail-closed (I2 precedent).
    2. busy_kind='user' + activity FRESH → NOT stealable, EVER, in any heartbeat state
       — a stale heartbeat plus busy=user means "mid-user-turn with dead maintenance",
       which must refuse (the F1 carve-out).
    3. busy_kind='user' + activity stale in (FRESH, STALL] → NOT stealable ("no visible
       progress yet" — could be a slow-but-live computation the user is watching).
    4. busy_kind='user' + activity stale > STALL (>0) → stealable (the stall shape:
       blocked approval, hung tool, silent compute; the owner-reported flagship case).
    5. busy_kind='auto' → not stealable while now-busy_since <= AUTO_BUSY_GRACE_S
       (rides out observed 65-80s bg-review API calls), stealable past grace.
    6. busy=True with an unknown kind → NOT stealable (absence = unknown = fail-closed).
    7. idle (busy False/absent) + heartbeat fresh → stealable immediately (the common
       idle-tab case).
    8. idle + heartbeat stale in (FRESH, STALE_STEAL_S] → NOT stealable (unknown; the
       holder may be perfectly healthy with a lagging watcher — refuse honestly).
    9. idle + heartbeat stale > STALE_STEAL_S → stealable (dead-watcher escape hatch:
       guarantees the system can never reach a permanently-unyieldable state).
    """
    heartbeat_at = _optional_float(existing.get("heartbeat_at"))
    if heartbeat_at is None:
        return HERMES_STEAL_LEGACY_HOLDER
    busy = bool(existing.get("busy"))
    busy_kind = existing.get("busy_kind")
    activity_at = _optional_float(existing.get("activity_at"))
    if busy and busy_kind == "user":
        # Tier 2-4: the activity clock (kept fresh by the turn thread itself) decides —
        # NOT the heartbeat, so protection survives a dead maintenance watcher.
        if activity_at is None:
            return False  # busy without a readable clock: unknown, never steal
        if (now - activity_at) <= max(0.0, HERMES_LEASE_ACTIVITY_FRESH_S):
            return False
        if HERMES_LEASE_STALL_ACTIVITY_S <= 0:
            return False  # stall-steal disabled: a user turn is never interruptible
        return (now - activity_at) > HERMES_LEASE_STALL_ACTIVITY_S
    if busy:
        if busy_kind == "auto":
            busy_since = _optional_float(existing.get("busy_since"))
            return busy_since is not None and (now - busy_since) > max(0.0, HERMES_LEASE_AUTO_BUSY_GRACE_S)
        return False  # unknown busy kind: fail-closed
    heartbeat_age = now - heartbeat_at
    if heartbeat_age <= HERMES_LEASE_HEARTBEAT_FRESH_S:
        return True  # idle + provably-maintained holder: the common idle-tab steal
    if heartbeat_age <= HERMES_LEASE_HEARTBEAT_STALE_STEAL_S:
        return False  # unknown maintenance state: refuse with honest "lease stale" text
    return True  # dead-watcher escape hatch (tier 2 already protected live user turns)


def try_acquire_active_session(
    *, session_id: str, surface: str, config: Any, metadata: Optional[dict[str, Any]] = None,
    registry_home: str | Path | None = None, track_liveness: bool = False,
    mark_busy: bool = False,
) -> tuple[Optional[ActiveSessionLease], Optional[str]]:
    """Acquire an active-session slot: ``(lease, None)`` or ``(None, ActiveSessionRefusal)``.

    Per-session exclusivity is CORRECTNESS, enforced unconditionally (at most one live
    owner per stored session); ``max_concurrent_sessions`` is resource POLICY, applied
    only when configured. ``registry_home`` lets profile-scoped backends share the owning
    profile's registry. Ownership uncertainty fails CLOSED (SESSION_COORDINATION_UNAVAILABLE).

    Preemptible leases (#auto-yield -> preemptible leases): a live FOREIGN holder is
    either stolen from atomically in this same flock (idle / stalled / past-grace /
    heartbeat-dead — epoch+1, busy marked) or refused fast with SESSION_BUSY (or
    SESSION_NOT_OWNED against a heartbeat-less legacy holder, which is never stolen
    from). ``mark_busy=True`` stamps the busy trio on the acquired entry — a turn
    admission publishing "mid-turn" from the first instant, so ownership cannot
    ping-pong between two requesters inside the first turn.

    Liveness tracking keeps richer desktop lifecycle semantics; ``registry_home`` lets profile-scoped
    backends share the owning profile's registry even when launched from another home. See #94595.
    """
    max_sessions = resolve_max_concurrent_sessions(config)
    lease_id = uuid.uuid4().hex
    key = str(session_id or "")

    # No stored id yet => nothing to fence or record (and the strict schema
    # refuses empty session ids): hand back a no-op lease.
    if not key and not track_liveness:
        return ActiveSessionLease(
            lease_id=lease_id, session_id=key, surface=str(surface), enabled=False
        ), None

    entry = _lease_entry(
        lease_id=lease_id, session_id=key, surface=str(surface), metadata=metadata,
        track_liveness=track_liveness, busy_kind="user" if mark_busy else None,
    )
    state_path, lock_path = _lease_paths(registry_home=registry_home)
    lease = ActiveSessionLease(
        lease_id=lease_id, session_id=key, surface=str(surface), state_path=state_path,
        lock_path=lock_path, track_liveness=track_liveness,
    )
    with _FileLock(lock_path):
        # A capacity cap could degrade open; exclusivity cannot: "could not
        # prove ownership" must never become "no owner exists".
        loaded = _read_live_entries(
            state_path, track_liveness=track_liveness,
            warn="Active-session registry is unavailable; refusing the session "
                 "rather than risking a concurrent writer",
        )
        if loaded is None:
            return None, ActiveSessionRefusal(
                "Hermes could not read the active-session registry at "
                f"{state_path}, so it cannot prove this session has no other "
                "live owner. Fix or remove that file and try again.",
                SESSION_COORDINATION_UNAVAILABLE,
            )
        raw_entries, entries = loaded
        pruned = len(raw_entries) - len(entries)
        if pruned:
            logger.info("Pruned %d stale active session lease(s)", pruned)

        def refuse(message: str, reason: str, log: str, *args, holder_entry: Optional[dict[str, Any]] = None) -> tuple[None, ActiveSessionRefusal]:
            _write_entries(state_path, entries)  # persist the prune even when refusing
            logger.info(log, *args)
            return None, ActiveSessionRefusal(message, reason, holder_entry=holder_entry)

        # Correctness first, under the same lock that just pruned dead owners.
        # An empty key is exempt: treating "" as an identity would make every
        # unsaved draft exclude every other one.
        if key:
            for index, existing in enumerate(entries):
                if str(existing.get("session_id") or "") != key:
                    continue
                # The same writer is not a second writer: a live session that
                # leaked its lease reference would otherwise be fenced out of
                # its own session permanently (pruning only removes entries
                # whose PROCESS is dead). Re-entrancy, not concurrency. The
                # fencing epoch is preserved so the holder's cached lease stays
                # valid across the re-acquire.
                if _is_same_writer(existing, metadata):
                    own_epoch = int(existing.get("epoch") or 1)
                    entries[index] = _lease_entry(
                        lease_id=lease_id, session_id=key, surface=str(surface),
                        metadata=metadata, track_liveness=track_liveness,
                        epoch=own_epoch, busy_kind="user" if mark_busy else None)
                    lease.epoch = own_epoch
                    _write_entries(state_path, entries)
                    return lease, None
                # The holder is LIVE (dead ones were pruned above). Decide the steal
                # RIGHT HERE, under the same flock a holder's admission revalidate
                # takes — exactly one of {requester steals, holder marks busy} can win.
                if (_pid_liveness(existing.get("pid"), existing.get("process_start_time"),
                                  lenient=False) is True
                        and _stealable_entry(existing, time.time())):
                    new_epoch = int(existing.get("epoch") or 0) + 1
                    # A steal ALWAYS marks busy (the acquisition is a turn admission):
                    # without it two requesters could ping-pong ownership inside the
                    # first turn. Net entry count is unchanged, so the capacity check
                    # below stays untouched (the foreign entry already held the slot).
                    entries[index] = _lease_entry(
                        lease_id=lease_id, session_id=key, surface=str(surface),
                        metadata=metadata, track_liveness=track_liveness,
                        epoch=new_epoch, busy_kind="user")
                    lease.epoch = new_epoch
                    _write_entries(state_path, entries)
                    logger.info(
                        "Stole active session lease for %s: pid=%s surface=%s epoch=%s -> %s",
                        key, existing.get("pid"), existing.get("surface"),
                        int(existing.get("epoch") or 0), new_epoch)
                    return lease, None
                # Not stealable: refuse fast and truthfully. SESSION_BUSY when the holder
                # publishes a heartbeat (new code); SESSION_NOT_OWNED for a heartbeat-less
                # LEGACY holder — machine-distinguishable so callers never retry-burn or
                # fabricate a takeover against old code.
                reason = SESSION_BUSY if existing.get("heartbeat_at") is not None else SESSION_NOT_OWNED
                return refuse(
                    session_already_owned_message(key, existing), reason,
                    "Refused active session %s: already held by pid=%s surface=%s busy=%s",
                    key, existing.get("pid"), existing.get("surface"),
                    bool(existing.get("busy")),
                    holder_entry=existing)

        # Capacity second, and only when an operator asked for one.
        if max_sessions is not None and len(entries) >= max_sessions:
            return refuse(
                active_session_limit_message(len(entries), max_sessions, entries),
                MAX_CONCURRENT_SESSIONS,
                "Active session limit reached: active=%d max=%d surface=%s",
                len(entries), max_sessions, surface,
            )
        entries.append(entry)
        _write_entries(state_path, entries)

    return lease, None


def _entry_still_owned(entry: Optional[dict[str, Any]], lease: ActiveSessionLease) -> bool:
    """Whether ``entry`` is still THIS lease: lease_id, this process (pid + process
    start pairing), and the same fencing epoch. The ONLY sanctioned identity predicate
    for holder-side busy/heartbeat writes — never find-by-session_id (a stolen session's
    new owner must never be mutated by the displaced holder's cleanup)."""
    if not isinstance(entry, dict):
        return False
    if str(entry.get("lease_id") or "") != lease.lease_id:
        return False
    if _registry_pid(entry.get("pid")) != os.getpid():
        return False
    expected_start = _optional_float(entry.get("process_start_time"))
    current_start = _process_start_time(os.getpid())
    if expected_start is not None and current_start is not None:
        if abs(current_start - expected_start) >= 0.001:
            return False  # this pid is a different incarnation: the lease died with the old one
    return int(entry.get("epoch") or 0) == int(getattr(lease, "epoch", 1) or 1)


def revalidate_active_session(
    lease: ActiveSessionLease, *, mark_busy: bool = False,
    busy_kind: Optional[str] = None, busy_detail: Optional[str] = None,
    convert_user_to_auto: bool = False,
) -> tuple[Optional[ActiveSessionLease], Optional[str]]:
    """Holder-side next-touch displacement check AND busy marker in ONE flock critical
    section (preemptible leases): ``(lease, None)`` when the lease is provably still
    ours, else ``(None, ActiveSessionRefusal(SESSION_DISPLACED))`` with the current
    entry attached as ``holder_entry`` — and NEVER a write in that case.

    ``mark_busy=True`` publishes the busy trio under the same lock: the flock-serialized
    point that makes "holder starts a turn" mutually exclusive with "requester steals".
    This is the ONLY sanctioned busy-marking primitive — no find-by-session_id mutation
    exists anywhere. Semantics:
      * busy_kind='user' sets busy/user + fresh activity (busy_since preserved when the
        entry already reads user — a followup chain is ONE continuous busy span).
      * busy_kind='auto' NEVER downgrades a live 'user' mark (a foreground turn wins);
        the settle transition below is what hands the mark over.
      * convert_user_to_auto (with busy_kind='auto'): the user-turn-settle transition —
        clears any live 'user' mark AND writes the auto token in THIS one critical
        section. Without it the guard above would block both ends of the handover (the
        begin cannot downgrade the live turn, and a plain auto-mark cannot either),
        stranding busy_kind='user' with a frozen activity clock for the review's whole
        lifetime (stall-stealable at 61s; the 90s auto grace unreachable).
      * mark_busy=False (a user turn settling) clears the mark UNLESS the entry still
        reads busy_kind='auto' (a bg-review token is active); busy_kind='auto' with
        mark_busy=False is the conditional auto-clear (no-op while a user turn holds it).
    """
    if (not getattr(lease, "enabled", True) or getattr(lease, "released", False)
            or getattr(lease, "state_path", None) is None or getattr(lease, "lock_path", None) is None):
        return lease, None  # disabled no-op lease (or a leaseless session stub): nothing to fence
    state_path, lock_path = _lease_paths(lease)
    now = time.time()
    with _FileLock(lock_path):
        try:
            entries = _read_entries(state_path, strict=True)
        except ActiveSessionRegistryError as exc:
            # Unknown ownership never proceeds (the #94595 fail-closed rule): report it
            # as coordination failure, NOT displacement — a corrupt registry must not
            # close live sessions.
            logger.warning("Lease revalidation could not read the registry: %s", exc)
            return None, ActiveSessionRefusal(
                "Hermes could not read the active-session registry, so it cannot prove "
                "this session is still owned by this surface. Try again.",
                SESSION_COORDINATION_UNAVAILABLE)
        entry = next((e for e in entries if str(e.get("lease_id") or "") == lease.lease_id), None)
        if not _entry_still_owned(entry, lease):
            return None, ActiveSessionRefusal(
                "Session taken over by another surface; reload to continue",
                SESSION_DISPLACED,
                holder_entry=entry if isinstance(entry, dict) else None)
        assert entry is not None  # _entry_still_owned implies it
        current_kind = entry.get("busy_kind") if entry.get("busy") else None
        if mark_busy:
            if convert_user_to_auto and busy_kind == "auto":
                # The settle transition: whatever mark the finishing turn left ('user'
                # with a frozen activity clock, or a prior 'auto'), the token now owns
                # the entry — fresh busy_since so the auto grace measures from NOW.
                entry["busy"] = True
                entry["busy_kind"] = "auto"
                entry["busy_detail"] = str(busy_detail or "auto")
                entry["busy_since"] = now
                entry.pop("activity_at", None)  # auto work has no visible-activity clock
            elif busy_kind == "user":
                entry["busy"] = True
                entry["busy_kind"] = "user"
                if busy_detail:
                    entry["busy_detail"] = str(busy_detail)
                else:
                    entry.pop("busy_detail", None)
                # Preserve busy_since across a continuous user span (chain shape); a
                # fresh span (idle -> user) mints now.
                if current_kind != "user":
                    entry["busy_since"] = now
                entry["activity_at"] = now
            elif busy_kind == "auto" and current_kind != "user":
                entry["busy"] = True
                entry["busy_kind"] = "auto"
                entry["busy_detail"] = str(busy_detail or "auto")
                if current_kind != "auto":
                    entry["busy_since"] = now
            # busy_kind='auto' while a user turn holds the mark: deliberate no-op above —
            # the settle transition (convert_user_to_auto) hands the mark over when the
            # turn finishes.
        else:
            if busy_kind == "auto":
                if current_kind == "auto":
                    entry["busy"] = False
                    entry.pop("busy_kind", None)
                    entry.pop("busy_detail", None)
                    entry.pop("busy_since", None)
                    entry.pop("activity_at", None)
            else:
                # Plain clear: NO local work claims the entry anymore (a turn settle with
                # a live review token takes the convert branch; a review end with a live
                # turn uses the conditional auto-clear) — so whatever mark is present
                # comes from bookkeeping that already finished. Clear it unconditionally:
                # leaving it would strand busy='user'/'auto' forever (the review round's
                # stuck-mark finding).
                entry["busy"] = False
                entry.pop("busy_kind", None)
                entry.pop("busy_detail", None)
                entry.pop("busy_since", None)
                entry.pop("activity_at", None)
        entry["heartbeat_at"] = now
        entry["updated_at"] = now
        _write_entries(state_path, entries)
    return lease, None


def heartbeat_leases(leases: list[ActiveSessionLease]) -> set[str]:
    """Refresh ``heartbeat_at`` on every held lease whose entry still matches (lease_id,
    pid, process_start_time, epoch); returns the DISPLACED lease_ids so the maintenance
    watcher can close those sessions (theft detection). Preserves all busy fields
    verbatim — a heartbeat never rewrites turn state. Registry read failures skip the
    group (no refresh, no displacement): a transient error must not close live sessions,
    and a persistent one ages the heartbeat into the stale-steal escape instead."""
    displaced: set[str] = set()
    groups: dict[Path, list[ActiveSessionLease]] = {}
    for lease in leases:
        if (not getattr(lease, "enabled", True) or getattr(lease, "released", False)
                or not getattr(lease, "lease_id", None)
                or getattr(lease, "state_path", None) is None or getattr(lease, "lock_path", None) is None):
            continue
        groups.setdefault(lease.lock_path, []).append(lease)
    now = time.time()
    for lock_path, group in groups.items():
        try:
            with _FileLock(lock_path):
                entries = _read_entries(group[0].state_path, strict=True)
                by_id = {str(e.get("lease_id") or ""): e for e in entries}
                wrote = False
                for lease in group:
                    entry = by_id.get(lease.lease_id)
                    if not _entry_still_owned(entry, lease):
                        displaced.add(lease.lease_id)
                        continue
                    entry["heartbeat_at"] = now
                    entry["updated_at"] = now
                    wrote = True
                if wrote:
                    _write_entries(group[0].state_path, entries)
        except ActiveSessionRegistryError:
            continue
        except OSError:
            continue
    return displaced


def refresh_lease_activity(lease: ActiveSessionLease, *, activity_at: Optional[float] = None) -> bool:
    """Turn-thread activity piggyback: refresh ``activity_at`` (and ``heartbeat_at``)
    under a NON-BLOCKING flock — skip entirely on contention (the next touch retries).
    Keeps a visibly streaming turn protected even when the maintenance watcher thread is
    dead; callers throttle to >= 1 write per HERMES_LEASE_ACTIVITY_REFRESH_MIN_S."""
    if (not getattr(lease, "enabled", True) or getattr(lease, "released", False)
            or lease.state_path is None or lease.lock_path is None):
        return False
    state_path, lock_path = _lease_paths(lease)
    now = time.time() if activity_at is None else float(activity_at)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "a+b")
    except OSError:
        return False
    try:
        try:
            _flock(fh, lock=True, blocking=False)
        except (BlockingIOError, OSError):
            return False  # contended: never stall the streaming path on the registry
        try:
            entries = _read_entries(state_path, strict=True)
            entry = next((e for e in entries if str(e.get("lease_id") or "") == lease.lease_id), None)
            if not _entry_still_owned(entry, lease):
                return False
            entry["activity_at"] = now
            entry["heartbeat_at"] = now
            entry["updated_at"] = now
            _write_entries(state_path, entries)
            return True
        finally:
            with suppress(Exception):
                _flock(fh, lock=False)
    except ActiveSessionRegistryError:
        return False
    finally:
        with suppress(Exception):
            fh.close()


def release_active_session(lease: ActiveSessionLease) -> None:
    # Prefer the registry the lease was acquired against: the caller may be
    # running under a profile HERMES_HOME override.
    # See #85431.
    state_path, lock_path = _lease_paths(lease)
    with _FileLock(lock_path):
        if lease.released:
            return
        loaded = _read_live_entries(
            state_path, track_liveness=lease.track_liveness,
            warn="Active-session registry is unavailable; preserving it while "
                 "releasing an untracked lease",
        )
        if loaded is not None:
            _drop_lease(state_path, loaded[1], lease.lease_id)
        lease.released = True


def transfer_active_session(
    lease: ActiveSessionLease, *, session_id: str, metadata: Optional[dict[str, Any]] = None
) -> bool:
    """Move an existing lease to a new session id without dropping the slot."""
    new_session_id = str(session_id or "")
    if not new_session_id or lease.released:
        return False
    if not lease.enabled:
        lease.session_id = new_session_id
        return True

    state_path, lock_path = _lease_paths(lease)
    with _FileLock(lock_path):
        # release() may have won after the optimistic precheck but before this
        # thread acquired the file lock. Never resurrect a durably removed lease.
        if lease.released:
            return False
        loaded = _read_live_entries(
            state_path, track_liveness=lease.track_liveness,
            warn="Active-session registry is unavailable; refusing to overwrite "
                 "it during lease transfer",
        )
        if loaded is None:
            return False
        entries = loaded[1]
        own = next((e for e in entries if str(e.get("lease_id") or "") == lease.lease_id), None)
        if own is not None:
            own["session_id"] = new_session_id
            own["updated_at"] = time.time()
            if metadata:
                own["metadata"] = _clean_metadata(metadata)
        elif lease.track_liveness:
            # Resurrect guard (preemptible leases): never re-append a lease onto a session
            # a LIVE foreign entry already holds — the displaced holder's transfer must
            # fail (lease treated as gone), not create a second writer. The epoch
            # comparison is skipped entirely when the foreign entry LACKS an epoch: an
            # old-code owner (epoch absent = 0) must still win this guard under skew.
            foreign = next((
                e for e in entries
                if str(e.get("session_id") or "") == new_session_id
                and str(e.get("lease_id") or "") != lease.lease_id), None)
            if foreign is not None and _pid_liveness(
                    foreign.get("pid"), foreign.get("process_start_time")) is not False:
                return False
            entries.append(_lease_entry(
                lease_id=lease.lease_id, session_id=new_session_id, surface=lease.surface,
                metadata=metadata, track_liveness=True,
                # Preserve the lease object's fencing epoch: a resurrected entry minted at
                # epoch 1 while the cached lease carries >1 would self-displace on the
                # very next revalidate (epoch mismatch) — the lease would look stolen
                # when it was not.
                epoch=int(getattr(lease, "epoch", 1) or 1),
            ))
        else:
            return False
        _write_entries(state_path, entries)
        lease.session_id = new_session_id
        return True


# A lease this process wrote in the last few seconds may not be in the caller's
# ``own_live_lease_ids`` yet: ``try_acquire_active_session`` writes the registry entry under
# the file lock and the server attaches the lease to its session record only after that
# returns. A concurrent finalize that snapshotted its live ids in between would otherwise
# read the brand-new lease as an orphan and drop it. Real orphans are minutes old.
# See #101415.
_SELF_ORPHAN_GRACE_SECONDS = 30.0


def _drop_self_orphans(
    entries: list[dict[str, Any]], own_live_lease_ids: set[str] | None
) -> list[dict[str, Any]]:
    """Drop this process's leases only when its caller can vouch for owners."""
    if own_live_lease_ids is None:
        return entries
    pid = os.getpid()
    cutoff = time.time() - _SELF_ORPHAN_GRACE_SECONDS
    return [
        entry for entry in entries
        if entry.get("pid") != pid
        or str(entry.get("lease_id") or "") in own_live_lease_ids
        or (_optional_float(entry.get("started_at")) or 0.0) > cutoff
    ]


def _release_orphaned_leases_in_home(registry_home: Path, live_lease_ids: set[str]) -> int:
    state_path = _state_path(registry_home)
    # No registry file yet means no leases have ever been written under this
    # home — don't take a lock (or create its file) on the idle-reaper tick.
    if not state_path.exists():
        return 0
    with _FileLock(_lock_path(registry_home)):
        loaded = _read_live_entries(
            state_path, track_liveness=False,
            warn="Active-session registry is unavailable; skipping orphaned-lease sweep",
        )
        if loaded is None:
            return 0
        entries = loaded[1]
        kept = _drop_self_orphans(entries, live_lease_ids)
        dropped = len(entries) - len(kept)
        if dropped:
            _write_entries(state_path, kept)
        return dropped


def release_orphaned_leases(live_lease_ids: set[str]) -> int:
    """Drop this process's registry entries that no live session owns.

    ``_prune_dead`` only reclaims leases of dead processes, so on a days-long server a
    lease whose session skipped teardown is held until restart. The owning process is the
    only authority on its own leases — exact, no heartbeat on the turn path, no threshold.
    Sweeps the root home and every profile home (a multiplexed server leases across them).
    """
    root = get_default_hermes_root()
    homes = [root]
    try:
        homes.extend(p for p in (root / "profiles").iterdir()
                     if p.is_dir() and not p.name.startswith("."))
    except OSError:
        pass

    dropped = 0
    for home in homes:
        try:
            dropped += _release_orphaned_leases_in_home(home, live_lease_ids)
        except OSError as exc:
            logger.debug("orphaned-lease sweep failed for %s: %s", home, exc)
    return dropped


def active_session_registry_snapshot(
    registry_home: str | Path | None = None, *, strict: bool = False,
) -> list[dict[str, Any]]:
    """Return live leases; attachment callers require provable liveness."""
    state_path, lock_path = _lease_paths(registry_home=registry_home)
    with _FileLock(lock_path):
        raw_entries = _read_entries(state_path, strict=True)
        entries = _prune_dead(raw_entries, strict=strict)
        if entries != raw_entries:
            _write_entries(state_path, entries)
        return entries


@contextmanager
def active_session_liveness_guard(
    session_id: str, *, registry_home: str | Path | None = None,
    own_live_lease_ids: set[str] | None = None,
) -> Iterator[bool]:
    """Hold the registry lock while reporting whether ``session_id`` is leased, so no
    new backend can acquire a lease between the check and the caller's ``end_session``."""
    state_path, lock_path = _lease_paths(registry_home=registry_home)
    with _FileLock(lock_path):
        entries = _prune_dead(_read_entries(state_path, strict=True), strict=True)
        entries = _drop_self_orphans(entries, own_live_lease_ids)
        _write_entries(state_path, entries)
        yield _holds_session(entries, session_id)


@contextmanager
def release_active_session_liveness_guard(
    lease: ActiveSessionLease, session_id: str, *, own_live_lease_ids: set[str] | None = None,
) -> Iterator[bool]:
    """Remove ``lease`` and hold its registry lock through a lifecycle write, making
    cleanup one atomic decision (release, check siblings, end the durable row)."""
    if not lease.enabled or lease.released:
        home = lease.state_path.parent.parent if lease.state_path is not None else None
        with active_session_liveness_guard(
            session_id, registry_home=home, own_live_lease_ids=own_live_lease_ids,
        ) as active:
            yield active
        return

    state_path, lock_path = _lease_paths(lease)
    with _FileLock(lock_path):
        entries = _prune_dead(_read_entries(state_path, strict=True), strict=True)
        kept = [e for e in entries if str(e.get("lease_id") or "") != lease.lease_id]
        kept = _drop_self_orphans(kept, own_live_lease_ids)
        if len(kept) != len(entries):
            _write_entries(state_path, kept)
        lease.released = True
        yield _holds_session(kept, session_id)
