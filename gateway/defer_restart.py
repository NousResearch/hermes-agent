"""Deferred self-restart request channel (``hermes gateway defer-restart``).

An agent that needs its gateway to restart (config/API-key change) cannot run
``hermes gateway restart`` from inside the gateway: the terminal-tool lifecycle
guard refuses any command that would kill the host gateway from within. This
module implements the *request* half of a deferred, drain-first restart that is
safe to trigger from inside a turn:

1. The CLI writes a protected request file (``restart_requests/pending-<id>.json``)
   naming the target process generation (pid + process start time) and a stable
   service identity, then sends SIGUSR1 as a wake-up only.
2. On wake-up the gateway validates the request against *its own* generation and
   service identity, atomically claims it, reserves the cross-process cooldown
   budget, and only then schedules the existing ``request_restart()`` drain-first
   restart. The ACCEPTED result is persisted to ``restart_requests/result-<id>.json``
   BEFORE the generation exits (protocol invariant I1).
3. The CLI polls for its own result file and reports ACCEPTED / unknown.

Protocol invariants (see ``openspec/changes/gateway-deferred-restart``):

- I1: only a valid request bound to the *current* generation can be ACCEPTED, and
  the acceptance record survives the accepting generation's exit.
- I2: within a cooldown window one service identity never gains extra restart
  budget via generation changes, sibling multiplex profiles, or duplicates.
- I3: pre-existing SIGUSR1 callers with no deferred request keep the legacy
  behaviour; a deferred request that FAILS validation must never fall through to
  the legacy unconditional restart.

Security posture: the request file is metadata only (no commands, exit codes, or
delays), lives in a 0700 directory with 0600 files, and is validated for owner,
regular-file type, and generation binding. ``actor_ref`` is audit-only and is
never an authorization credential.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROTOCOL_SCHEMA_VERSION = 1
REQUESTS_DIR_NAME = "restart_requests"
#: A pending request the gateway never saw (CLI crashed before signalling) expires.
PENDING_TTL_S = 300.0
#: Accepted/unknown results stay queryable across generations (I1), then get pruned.
RESULT_RETENTION_S = 7 * 24 * 3600.0
#: Cross-process cooldown between accepted deferred restarts, per service identity (I2).
COOLDOWN_WINDOW_S = 600.0

# Result outcomes. The CLI maps these onto its exit-code contract.
OUTCOME_ACCEPTED = "accepted"
OUTCOME_DUPLICATE = "duplicate"
OUTCOME_SUPERSEDED = "superseded"
OUTCOME_REJECTED_INVALID = "rejected_invalid"
OUTCOME_REJECTED_TARGET_MISMATCH = "rejected_target_mismatch"
OUTCOME_REJECTED_COOLDOWN = "rejected_cooldown"
OUTCOME_EXPIRED = "expired"
#: Restart scheduled by this request but the outcome could not be persisted — the CLI
#: treats an absent result as "unknown" (exit 6) and must not blindly retry.
OUTCOMES_FINAL = {
    OUTCOME_ACCEPTED, OUTCOME_DUPLICATE, OUTCOME_SUPERSEDED,
    OUTCOME_REJECTED_INVALID, OUTCOME_REJECTED_TARGET_MISMATCH,
    OUTCOME_REJECTED_COOLDOWN, OUTCOME_EXPIRED,
}


def requests_dir() -> Path:
    """The protected request directory (``<hermes_home>/restart_requests``, 0700)."""
    from hermes_constants import get_hermes_home

    path = get_hermes_home() / REQUESTS_DIR_NAME
    try:
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "posix":
            os.chmod(path, 0o700)
    except OSError as exc:
        logger.warning("defer-restart: cannot create request dir %s: %s", path, exc)
        raise
    return path


def process_generation() -> dict[str, Any]:
    """This process's generation identity: ``{"pid", "start_time"}``."""
    from gateway.status import _get_process_start_time

    start = _get_process_start_time(os.getpid())
    return {"pid": os.getpid(), "start_time": start}


def compute_service_identity() -> Optional[dict[str, str]]:
    """Stable service identity ``(hermes home, supervisor kind, service id)``.

    Shared by CLI and gateway so both sides derive the same key; ``None`` when no
    supported supervisor manages this install (the deferred channel requires one —
    without a supervisor a drained gateway would simply never come back).
    """
    try:
        from hermes_constants import get_hermes_home
        from hermes_cli.service_manager import detect_service_manager

        kind = detect_service_manager()
        if kind not in ("launchd", "systemd", "s6"):
            return None
        from hermes_cli.gateway import get_service_name

        return {
            "kind": str(kind),
            "service_id": get_service_name(),
            "hermes_home": str(get_hermes_home().resolve()),
        }
    except Exception as exc:  # pragma: no cover - defensive: identity must never crash a signal path
        logger.warning("defer-restart: service identity unavailable: %s", exc)
        return None


def service_identity_key(identity: dict[str, str]) -> str:
    """Canonical cooldown key for a service identity (I2)."""
    return f'{identity.get("kind", "")}|{identity.get("service_id", "")}|{identity.get("hermes_home", "")}'


def _request_path(request_id: str) -> Path:
    return requests_dir() / f"pending-{request_id}.json"


def _result_path(request_id: str) -> Path:
    return requests_dir() / f"result-{request_id}.json"


def _cooldown_path() -> Path:
    return requests_dir() / "cooldown.json"


def _fsync_dir(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> bool:
    """tmp + rename + dir-fsync at 0600. Symlink-safe: writes to a fresh name."""
    from utils import atomic_json_write

    try:
        atomic_json_write(path, payload, mode=0o600, default=str)
        _fsync_dir(path.parent)
        return True
    except Exception as exc:
        logger.warning("defer-restart: atomic write %s failed: %s", path, exc)
        return False


def _safe_regular_private_file(path: Path) -> bool:
    """True when ``path`` is a regular file we own with no group/other bits (POSIX)."""
    try:
        st = os.lstat(path)
    except OSError:
        return False
    import stat as stat_mod

    if not stat_mod.S_ISREG(st.st_mode):
        return False
    if os.name == "posix":
        if st.st_uid != os.getuid():
            return False
        if st.st_mode & 0o077:
            return False
    return True


# ---------------------------------------------------------------------------
# Cooldown (I2) — shared store, gateway is the writer of record.
# ---------------------------------------------------------------------------

def read_cooldown() -> dict[str, Any]:
    try:
        raw = _cooldown_path().read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def cooldown_active_for(identity_key: str, *, now: Optional[float] = None) -> Optional[dict[str, Any]]:
    """The blocking cooldown record for ``identity_key``, or None."""
    entry = read_cooldown().get(identity_key)
    if not isinstance(entry, dict):
        return None
    try:
        accepted_at = float(entry.get("accepted_at", 0.0))
    except (TypeError, ValueError):
        return None
    now = time.time() if now is None else now
    if 0.0 < (now - accepted_at) <= COOLDOWN_WINDOW_S:
        return entry
    return None


def record_cooldown(identity_key: str, request_id: str) -> bool:
    data = read_cooldown()
    data[identity_key] = {"accepted_at": time.time(), "request_id": request_id}
    return _atomic_write_json(_cooldown_path(), data)


# ---------------------------------------------------------------------------
# Result records (I1) — persist before the accepting generation exits.
# ---------------------------------------------------------------------------

def write_result(
    request_id: str,
    outcome: str,
    *,
    generation: Optional[dict[str, Any]] = None,
    detail: str = "",
) -> bool:
    """Persist the decision for ``request_id``. Queryable after the writer exits."""
    payload = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "request_id": request_id,
        "outcome": outcome,
        "generation": generation or process_generation(),
        "decided_at": time.time(),
        "detail": detail,
    }
    return _atomic_write_json(_result_path(request_id), payload)


def read_result(request_id: str) -> Optional[dict[str, Any]]:
    """The persisted decision for ``request_id`` (CLI polling), or None."""
    path = _result_path(request_id)
    try:
        if not _safe_regular_private_file(path):
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("request_id") != request_id:
        return None
    if payload.get("outcome") not in OUTCOMES_FINAL:
        return None
    return payload


# ---------------------------------------------------------------------------
# Request validation & consumption (gateway side).
# ---------------------------------------------------------------------------

def _validate_payload_shape(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload is not an object"
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        return False, "unsupported schema_version"
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        return False, "bad request_id"
    gen = payload.get("target_generation")
    if not isinstance(gen, dict) or not isinstance(gen.get("pid"), int):
        return False, "bad target_generation"
    if not isinstance(payload.get("service_identity"), dict):
        return False, "bad service_identity"
    return True, ""


def _generation_matches(recorded: dict[str, Any]) -> bool:
    from gateway.status import start_time_fingerprints_match

    me = process_generation()
    if recorded.get("pid") != me.get("pid"):
        return False
    if me.get("start_time") is None or recorded.get("start_time") is None:
        return False
    try:
        # Same comparator the liveness reconciler uses: same-host readings of the
        # start-time fingerprint can drift by ~1 s (#117505), far beyond the 1 ms
        # equality tolerance — a recycled PID is still never this close in time.
        return bool(start_time_fingerprints_match(recorded.get("start_time"), me["start_time"]))
    except (TypeError, ValueError):
        return False


def _identity_matches(recorded: dict[str, Any], mine: Optional[dict[str, str]]) -> bool:
    if mine is None:
        return False
    return all(recorded.get(k) == v for k, v in mine.items())


def list_pending_request_files() -> list[Path]:
    try:
        return sorted(requests_dir().glob("pending-*.json"))
    except OSError:
        return []


def might_have_pending_request() -> bool:
    """Cheap name-only check for the SIGUSR1 fast path (no request-file reads).

    Used by the signal handler to pick between the legacy synchronous restart
    and the deferred-request task; protocol ordering (write request file, then
    signal) guarantees a real deferred request is visible here before its
    wake-up arrives. Failures report ``False`` — the legacy path is the safe
    default for an ordinary restart signal.
    """
    try:
        with os.scandir(requests_dir()) as entries:
            return any(e.name.startswith("pending-") for e in entries)
    except OSError:
        return False


def consume_pending_request(
    *, now: Optional[float] = None
) -> tuple[Optional[dict[str, Any]], str, str]:
    """Validate and claim the newest pending request for THIS generation.

    Returns ``(payload_or_None, outcome, detail)``. Every consumed request gets a
    persisted result record (explicit outcomes only — never silently dropped) and
    its pending file is removed so a re-sent SIGUSR1 cannot reprocess it. A
    request aimed at another generation is left in place (it belongs to its own
    target) but gets a ``rejected_target_mismatch`` result only when it targets a
    dead/other generation — a live sibling claim is that generation's business.
    """
    now = time.time() if now is None else now
    pending = list_pending_request_files()
    if not pending:
        return None, "", "no pending request"

    newest = pending[-1]
    request_id = newest.name[len("pending-"):-len(".json")]
    try:
        raw = newest.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except (OSError, ValueError) as exc:
        with _best_effort():
            write_result(request_id, OUTCOME_REJECTED_INVALID, detail=f"unreadable request: {exc}")
            _unlink_quiet(newest)
        return None, OUTCOME_REJECTED_INVALID, f"unreadable request: {exc}"

    ok, why = _validate_payload_shape(payload)
    if not ok or payload.get("request_id") != request_id:
        with _best_effort():
            write_result(request_id, OUTCOME_REJECTED_INVALID, detail=why or "request_id mismatch")
            _unlink_quiet(newest)
        return None, OUTCOME_REJECTED_INVALID, why

    if not _safe_regular_private_file(newest):
        with _best_effort():
            write_result(request_id, OUTCOME_REJECTED_INVALID, detail="request file permissions/owner")
            _unlink_quiet(newest)
        return None, OUTCOME_REJECTED_INVALID, "request file permissions/owner"

    if not _generation_matches(payload["target_generation"]):
        # Not ours: leave the file for its real target (or expiry), but tell *this*
        # wake-up's caller nothing was claimed.
        return None, OUTCOME_REJECTED_TARGET_MISMATCH, "target generation mismatch"

    mine = compute_service_identity()
    if mine is None or not _identity_matches(payload["service_identity"], mine):
        with _best_effort():
            write_result(request_id, OUTCOME_REJECTED_INVALID, detail="service identity mismatch")
            _unlink_quiet(newest)
        return None, OUTCOME_REJECTED_INVALID, "service identity mismatch"

    # Same request id already decided: return the existing decision (idempotent
    # replay). A replayed ACCEPTED is reported as DUPLICATE so the wake-up path
    # never schedules a second restart for the same request (I2).
    existing = read_result(request_id)
    if existing is not None:
        with _best_effort():
            _unlink_quiet(newest)
        prior = str(existing.get("outcome") or OUTCOME_REJECTED_INVALID)
        if prior == OUTCOME_ACCEPTED:
            return payload, OUTCOME_DUPLICATE, "replay of decided request"
        return payload, prior, "replay of decided request"

    identity_key = service_identity_key(mine)
    blocking = cooldown_active_for(identity_key, now=now)
    if blocking is not None and blocking.get("request_id") != request_id:
        with _best_effort():
            write_result(request_id, OUTCOME_REJECTED_COOLDOWN,
                         detail=f'cooldown window active until +{COOLDOWN_WINDOW_S:.0f}s')
            _unlink_quiet(newest)
        return None, OUTCOME_REJECTED_COOLDOWN, "cooldown active"

    # Claim: the pending file is consumed exactly once.
    _unlink_quiet(newest)
    return payload, OUTCOME_ACCEPTED, ""


class _best_effort:
    """Context manager that swallows persistence failures (logged, never raised)."""

    def __enter__(self) -> "_best_effort":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is not None:
            logger.warning("defer-restart: post-decision persistence failed: %s", exc)
        return True


def _unlink_quiet(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# SIGUSR1 wake-up processing (runs as a loop task, never inside the raw signal
# callback — no blocking file I/O in signal handlers).
# ---------------------------------------------------------------------------

def process_sigusr1_request(schedule_restart) -> None:
    """Handle one SIGUSR1 wake-up with deferred-request awareness (I3).

    ``schedule_restart`` is the legacy scheduling callable (no args, returns
    truthy when a NEW restart was scheduled). Behaviour:

    - No pending deferred request → legacy path: call ``schedule_restart()``
      exactly as the pre-deferred handler did.
    - Pending request valid for this generation → claim, then schedule; persist
      ACCEPTED only when the schedule actually succeeded (linearization point),
      before this generation can exit; otherwise persist SUPERSEDED (the restart
      is already scheduled — an explicit outcome, never a silent drop).
    - Pending request present but INVALID → persist the rejection and do NOT
      restart: an invalid request must never fall through to the legacy
      unconditional restart (attacker-dropped files must not cause restarts).
    """
    cleanup_expired()
    if not list_pending_request_files():
        # Legacy compatibility: plain SIGUSR1 restart behaviour, unchanged.
        schedule_restart()
        return

    payload, outcome, detail = consume_pending_request()
    if payload is None:
        logger.warning(
            "SIGUSR1: deferred restart request not claimed (outcome=%s detail=%s) — "
            "no legacy restart fallback (invariant I3)", outcome, detail)
        return

    request_id = payload["request_id"]
    if outcome == OUTCOME_DUPLICATE:
        # Replay of an already-ACCEPTED request: the original decision stands (I2)
        # — nothing to schedule and the original result is left untouched.
        logger.info("SIGUSR1: deferred restart %s is a duplicate of an accepted request", request_id)
        return
    if outcome != OUTCOME_ACCEPTED:
        # Replay of an already-decided (rejected) request: repeat that decision.
        with _best_effort():
            write_result(request_id, outcome, detail=f"replay: {detail}")
        return

    scheduled = False
    try:
        scheduled = bool(schedule_restart())
    except Exception as exc:
        logger.warning("defer-restart: scheduling failed after claim: %s", exc)
    if scheduled:
        with _best_effort():
            record_cooldown(service_identity_key(payload["service_identity"]), request_id)
            write_result(request_id, OUTCOME_ACCEPTED,
                         generation=payload.get("target_generation"))
    else:
        with _best_effort():
            write_result(request_id, OUTCOME_SUPERSEDED,
                         detail="restart already scheduled for this gateway")


# ---------------------------------------------------------------------------
# Expiry cleanup — never deletes accepted/unknown audit evidence (spec).
# ---------------------------------------------------------------------------

def cleanup_expired(*, now: Optional[float] = None) -> None:
    now = time.time() if now is None else now
    try:
        files = list(requests_dir().iterdir())
    except OSError:
        return
    for path in files:
        name = path.name
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        if name.startswith("pending-") and now - mtime > PENDING_TTL_S:
            request_id = name[len("pending-"):-len(".json")]
            with _best_effort():
                # Expired without a decision → persist an explicit EXPIRED outcome so
                # the CLI of a crashed sender has something attributable to read.
                if read_result(request_id) is None:
                    write_result(request_id, OUTCOME_EXPIRED, detail="pending request expired unconsumed")
                _unlink_quiet(path)
        elif name.startswith("result-") and now - mtime > RESULT_RETENTION_S:
            _unlink_quiet(path)
        elif name == "cooldown.json" and now - mtime > RESULT_RETENTION_S:
            _unlink_quiet(path)
