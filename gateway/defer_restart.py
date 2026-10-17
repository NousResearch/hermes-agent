"""Deferred self-restart request protocol (``hermes gateway defer-restart``).

An agent that needs its gateway to restart (config/API-key change) cannot run
``hermes gateway restart`` from inside the gateway process — the terminal-tool
lifecycle guard refuses every host-killing command, by design. This module is
the gateway half of a two-phase, drain-first deferred restart:

1. The CLI writes a small validated request file naming its own request id,
   the service identity and the target generation (pid + start-time
   fingerprint), then sends SIGUSR1 — the existing graceful-restart signal.
2. The gateway claims the request and schedules the existing drain-first
   restart; the per-``request_id`` result record is the CLI's attributable
   confirmation. The CLI exits immediately after confirmation and never waits
   for the restart itself.

Security posture (see openspec/changes/gateway-deferred-restart for the
threat model): the request file is not a credential — same-user processes can
already signal the gateway. It exists for *attribution and binding* (no
cross-service hijack, no PID-reuse races, no stale requests), and for the
cross-process cooldown (I2). The directory is 0700, request files are 0600,
and validation refuses symlinks, wrong owners and loose modes.

Protocol invariants (reviewed):
I1  the acceptance decision is persisted per ``request_id`` before this
    generation may exit, and a final outcome is NEVER overwritten within its
    retention window (first decision wins across rejections, replays and
    expiry cleanup);
I2  the restart budget is keyed by a stable service identity (not a pid), a
    replay never spends a second restart, and the cooldown is recoverable
    from persisted records if ``cooldown.json`` is lost;
I3  no valid deferred request → the legacy SIGUSR1 restart is unchanged; a
    deferred request that fails validation is rejected and never falls
    through to the unconditional legacy restart; once a wake observes
    protocol traffic, later empty-dir wakes take no action (double-wake
    races cannot resurrect legacy restarts).

All request-file I/O runs outside the raw signal callback (see
``process_sigusr1_request``).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
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
#: Acceptance writes retry briefly (transient FS pressure) before a scheduled
#: restart is cancelled — an exit must never run without a persisted decision (I1).
RESULT_WRITE_ATTEMPTS = 3
RESULT_WRITE_BACKOFF_S = 0.1

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

# Result outcomes. The CLI maps these onto its exit-code contract.
OUTCOME_ACCEPTED = "accepted"
OUTCOME_DUPLICATE = "duplicate"
OUTCOME_SUPERSEDED = "superseded"
OUTCOME_REJECTED_INVALID = "rejected_invalid"
OUTCOME_REJECTED_TARGET_MISMATCH = "rejected_target_mismatch"
OUTCOME_REJECTED_COOLDOWN = "rejected_cooldown"
OUTCOME_REJECTED_STOPPING = "rejected_stopping"
OUTCOME_EXPIRED = "expired"
#: Every outcome above is final: the CLI maps an absent result to "unknown" (exit 6)
#: and must not blindly retry. ``duplicate`` never overwrites the original
#: ``accepted`` record — the CLI polls the result file and reads the first decision.
OUTCOMES_FINAL = {
    OUTCOME_ACCEPTED, OUTCOME_DUPLICATE, OUTCOME_SUPERSEDED,
    OUTCOME_REJECTED_INVALID, OUTCOME_REJECTED_TARGET_MISMATCH,
    OUTCOME_REJECTED_COOLDOWN, OUTCOME_REJECTED_STOPPING, OUTCOME_EXPIRED,
}
#: Outcomes that mean "a restart for this gateway is already happening".
OUTCOMES_SUCCESS = {OUTCOME_ACCEPTED, OUTCOME_DUPLICATE, OUTCOME_SUPERSEDED}

#: Once any wake has seen protocol traffic this flag is set for the generation:
#: later wakes with an empty/unknown request dir take NO action (I3, double-wake).
_deferred_engaged = False


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


def _claim_path(request_id: str) -> Path:
    return requests_dir() / f"claim-{request_id}.json"


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


def _exclusive_write_json(path: Path, payload: dict[str, Any]) -> bool:
    """Create ``path`` ONLY if absent (``O_EXCL``) — the exclusive claim token.

    Unlike tmp+rename (which replaces), this has true create-if-absent atomic
    semantics, so exactly one consumer can own a given request id even across
    re-publishes of the same id (review gate 1 round 2, item 6).
    """
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    except OSError as exc:
        logger.warning("defer-restart: exclusive create %s failed: %s", path, exc)
        return False
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        _fsync_dir(path.parent)
        return True
    except Exception as exc:
        logger.warning("defer-restart: exclusive write %s failed: %s", path, exc)
        try:
            path.unlink()
        except OSError:
            pass
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


def _unlink_exclusive(path: Path) -> bool:
    """Delete ``path``; True only when THIS caller removed it (claim-by-unlink).

    A missing file means another consumer/cleaner won the race — the caller must
    NOT continue as the claimant (review gate 1, item 5).
    """
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False
    except OSError as exc:
        logger.warning("defer-restart: claim unlink %s failed: %s", path, exc)
        return False


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
# Cooldown (I2) — shared store, gateway is the writer of record. Read failures
# are fail-CLOSED (treated as active), and the window is recoverable from
# persisted result/claim records if cooldown.json is lost.
# ---------------------------------------------------------------------------

def read_cooldown() -> tuple[dict[str, Any], bool]:
    """``(data, ok)`` — ``ok=False`` when the store exists but cannot be read."""
    try:
        raw = _cooldown_path().read_text(encoding="utf-8")
        data = json.loads(raw)
        return (data if isinstance(data, dict) else {}), True
    except FileNotFoundError:
        return {}, True
    except (OSError, ValueError):
        return {}, False


def _recover_cooldown_from_records(identity_key: str, now: float) -> Optional[dict[str, Any]]:
    """Derive a blocking cooldown entry from persisted accepted/claim records (I2)."""
    try:
        files = list(requests_dir().iterdir())
    except OSError:
        return None
    newest: Optional[dict[str, Any]] = None
    for path in files:
        name = path.name
        if not (name.startswith("result-") or name.startswith("claim-")):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        outcome = data.get("outcome")
        if name.startswith("result-") and outcome != OUTCOME_ACCEPTED:
            continue
        if data.get("service_key") not in (None, identity_key):
            continue
        try:
            when = float(data.get("decided_at") or data.get("claimed_at") or 0.0)
        except (TypeError, ValueError):
            continue
        if 0.0 < (now - when) <= COOLDOWN_WINDOW_S:
            if newest is None or when > float(newest.get("accepted_at", 0.0)):
                newest = {"accepted_at": when,
                          "request_id": str(data.get("request_id", "")),
                          "recovered": True}
    return newest


def cooldown_active_for(identity_key: str, *, now: Optional[float] = None) -> Optional[dict[str, Any]]:
    """The blocking cooldown record for ``identity_key``, or None.

    Fail-closed on read errors (a phantom cooldown only delays a restart; a
    phantom miss spends the budget twice — I2), and the store entry is NEVER
    trusted as the latest state on its own: records are always re-scanned and
    the NEWER of (store entry, recovered record) wins. A stale-but-valid
    ``cooldown.json`` plus a failed update cannot reopen the window
    (review gate 1 round 2, item 4.2).
    """
    now = time.time() if now is None else now
    data, ok = read_cooldown()
    if not ok:
        logger.warning("defer-restart: cooldown store unreadable — treating as active (fail-closed)")
        return {"accepted_at": now, "request_id": "<unreadable>", "unreadable": True}
    entry = data.get(identity_key)
    store_entry: Optional[dict[str, Any]] = None
    if isinstance(entry, dict):
        try:
            accepted_at = float(entry.get("accepted_at", 0.0))
        except (TypeError, ValueError):
            accepted_at = 0.0
        if 0.0 < (now - accepted_at) <= COOLDOWN_WINDOW_S:
            store_entry = entry
    recovered = _recover_cooldown_from_records(identity_key, now)
    if recovered is None:
        return store_entry
    if store_entry is None:
        return recovered
    try:
        if float(recovered.get("accepted_at", 0.0)) > float(store_entry.get("accepted_at", 0.0)):
            return recovered
    except (TypeError, ValueError):
        return recovered
    return store_entry


def record_cooldown(identity_key: str, request_id: str) -> bool:
    data, ok = read_cooldown()
    if not ok:
        logger.warning("defer-restart: cooldown store unreadable at record time")
    data[identity_key] = {"accepted_at": time.time(), "request_id": request_id}
    return _atomic_write_json(_cooldown_path(), data)


# ---------------------------------------------------------------------------
# Result records (I1) — persist before the accepting generation exits, and a
# final outcome is NEVER overwritten (first decision wins).
# ---------------------------------------------------------------------------

#: Append-only audit trail (spec: Audit logging and request retention).
AUDIT_FILE_NAME = "audit.jsonl"


def _audit_path() -> Path:
    return requests_dir() / AUDIT_FILE_NAME


def _sanitize_actor_ref(value: Any) -> str:
    """Audit-only attribution: a bounded string, never an authorization input."""
    return value if isinstance(value, str) and len(value) <= 256 else ""


def append_audit(
    request_id: str,
    outcome: str,
    *,
    actor_ref: str = "",
    generation: Optional[dict[str, Any]] = None,
    service_key: Optional[str] = None,
    detail: str = "",
) -> None:
    """Atomically append one JSONL audit record (O_APPEND write + fsync).

    Audit is best-effort evidence, never a gate: a failed append is logged and
    the decision still stands (the result record is the authoritative outcome).
    """
    record = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "request_id": request_id,
        "outcome": outcome,
        "actor_ref": _sanitize_actor_ref(actor_ref),
        "recorded_at": time.time(),
        "generation": generation if generation is not None else process_generation(),
        "service_key": service_key,
        "detail": detail,
    }
    try:
        fd = os.open(_audit_path(), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception as exc:
        logger.warning("defer-restart: audit append failed: %s", exc)


def write_result(
    request_id: str,
    outcome: str,
    *,
    generation: Optional[dict[str, Any]] = None,
    detail: str = "",
    service_key: Optional[str] = None,
    actor_ref: str = "",
) -> bool:
    """Persist the decision for ``request_id``. Queryable after the writer exits.

    Refuses to overwrite an existing final decision (I1): rejections, replays
    and expiry cleanup must all respect the first decision. Every recorded
    decision is additionally appended to the audit trail (spec).
    """
    path = _result_path(request_id)
    if path.exists():
        logger.info("defer-restart: result for %s already final — not overwriting", request_id)
        return False
    payload = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "request_id": request_id,
        "outcome": outcome,
        "generation": generation or process_generation(),
        "decided_at": time.time(),
        "detail": detail,
    }
    if service_key is not None:
        payload["service_key"] = service_key
    ok = _atomic_write_json(path, payload)
    if ok and not path.exists():  # pragma: no cover - defensive
        return False
    if ok:
        append_audit(request_id, outcome, actor_ref=actor_ref,
                     generation=payload["generation"], service_key=service_key, detail=detail)
    return ok


def write_result_with_retry(request_id: str, outcome: str, **kwargs: Any) -> bool:
    """``write_result`` with brief retries; returns True only once persisted (I1).

    The recovery branch trusts CONTENT, not path existence (review gate 1
    round 3, R1): only a readable, valid terminal record whose outcome matches
    the decision being written proves "this acceptance is persisted". A corrupt,
    unreadable or differently-decided file is NOT success — the caller cancels
    the scheduled restart (no exit without a verifiable decision).
    """
    for attempt in range(RESULT_WRITE_ATTEMPTS):
        if write_result(request_id, outcome, **kwargs):
            return True
        prior = read_result(request_id)
        if prior is not None and prior.get("outcome") == outcome:
            # A valid record matching the attempted decision already landed
            # (e.g. the write succeeded but dir-fsync reported failure).
            return True
        if attempt + 1 < RESULT_WRITE_ATTEMPTS:
            time.sleep(RESULT_WRITE_BACKOFF_S)
    return False


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

def _is_number(value: Any) -> bool:
    """Real finite int/float — rejects bool (a subclass of int) and inf/nan
    (JSON's default decoder accepts ``Infinity``/``NaN``, and ``1e309`` parses
    to inf — all refused at the schema layer, review gate 1 round 2)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        # Huge JSON ints (e.g. "1" + "0"*400) overflow the float conversion
        # instead of returning False — refuse them like any illegal number
        # (review gate 1 round 3, R2).
        return False


def _validate_payload_shape(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload is not an object"
    if type(payload.get("schema_version")) is not int or payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        return False, "unsupported schema_version"
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not _REQUEST_ID_RE.match(request_id):
        return False, "bad request_id"
    gen = payload.get("target_generation")
    if not isinstance(gen, dict):
        return False, "bad target_generation"
    if type(gen.get("pid")) is not int or gen.get("pid") <= 0:
        return False, "bad target_generation.pid"
    if not _is_number(gen.get("start_time")) or gen.get("start_time") <= 0:
        return False, "bad target_generation.start_time"
    identity = payload.get("service_identity")
    if not isinstance(identity, dict):
        return False, "bad service_identity"
    for key in ("kind", "service_id", "hermes_home"):
        if not isinstance(identity.get(key), str):
            return False, f"bad service_identity.{key}"
    actor = payload.get("actor_ref")
    if actor is not None and (not isinstance(actor, str) or len(actor) > 256):
        return False, "bad actor_ref"
    return True, ""


def _generation_matches(recorded: dict[str, Any]) -> bool:
    from gateway.status import start_time_fingerprints_match

    me = process_generation()
    if type(recorded.get("pid")) is not int or recorded.get("pid") != me.get("pid"):
        return False
    if not _is_number(recorded.get("start_time")) or not _is_number(me.get("start_time")):
        return False
    if me.get("start_time") is None or recorded.get("start_time") is None:
        return False
    try:
        # Same comparator the liveness reconciler uses: same-host readings of the
        # start-time fingerprint can drift by ~1 s (#117505), far beyond the 1 ms
        # equality tolerance. This is a bounded engineering tolerance, NOT a proof
        # of generation uniqueness — pid equality is the primary gate and a
        # recycled pid could in principle land within the window (review gate 1).
        return bool(start_time_fingerprints_match(recorded.get("start_time"), me["start_time"]))
    except (TypeError, ValueError):
        return False


def _identity_matches(recorded: dict[str, Any], mine: Optional[dict[str, str]]) -> bool:
    if mine is None:
        return False
    return all(recorded.get(k) == v for k, v in mine.items())


def list_pending_request_files() -> Optional[list[Path]]:
    """Pending request files, or ``None`` when the directory cannot be listed.

    ``None`` means "unknown" and MUST NOT be treated as "confirmed empty" (I3).
    """
    try:
        with os.scandir(requests_dir()) as entries:
            names = sorted(e.name for e in entries if e.name.startswith("pending-") and e.name.endswith(".json"))
        return [requests_dir() / n for n in names]
    except OSError:
        return None


def might_have_pending_request() -> Optional[bool]:
    """Cheap name-only check for wake classification: True / False / None (unknown)."""
    try:
        with os.scandir(requests_dir()) as entries:
            for e in entries:
                if e.name.startswith("pending-"):
                    return True
            return False
    except OSError:
        return None


def consume_pending_request(
    *, now: Optional[float] = None
) -> tuple[Optional[dict[str, Any]], str, str]:
    """Validate and claim the newest pending request for THIS generation.

    Order matters (review gate 1, item 2): file-safety checks first, then any
    EXISTING decision for the same request id (replay — never overwritten),
    and only then the "new request" checks (generation, identity, cooldown).

    Returns ``(payload_or_None, outcome, detail)``. Every consumed request gets a
    persisted result record (explicit outcomes only — never silently dropped) and
    its pending file is removed so a re-sent SIGUSR1 cannot reprocess it. A
    request aimed at another generation is left in place (it belongs to its own
    target). Claim-by-unlink: only the consumer whose unlink() succeeded may
    continue to scheduling.
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
            if not _unlink_exclusive(newest):
                return None, "", "claim lost to another consumer"
            write_result(request_id, OUTCOME_REJECTED_INVALID, detail=f"unreadable request: {exc}")
        return None, OUTCOME_REJECTED_INVALID, f"unreadable request: {exc}"

    ok, why = _validate_payload_shape(payload)
    if not ok or payload.get("request_id") != request_id:
        with _best_effort():
            if not _unlink_exclusive(newest):
                return None, "", "claim lost to another consumer"
            write_result(request_id, OUTCOME_REJECTED_INVALID, detail=why or "request_id mismatch")
        return None, OUTCOME_REJECTED_INVALID, why

    if not _safe_regular_private_file(newest):
        with _best_effort():
            if not _unlink_exclusive(newest):
                return None, "", "claim lost to another consumer"
            write_result(request_id, OUTCOME_REJECTED_INVALID, detail="request file permissions/owner")
        return None, OUTCOME_REJECTED_INVALID, "request file permissions/owner"

    actor = _sanitize_actor_ref(payload.get("actor_ref"))

    # Existing decision first (I1): a decided request id is never re-decided,
    # regardless of which generation now sees it.
    existing = read_result(request_id)
    if existing is not None:
        with _best_effort():
            _unlink_exclusive(newest)
        prior = str(existing.get("outcome") or OUTCOME_REJECTED_INVALID)
        if prior == OUTCOME_ACCEPTED:
            return payload, OUTCOME_DUPLICATE, "replay of decided request"
        return payload, prior, "replay of decided request"

    # A result file that exists but is NOT a valid terminal record (corrupt,
    # unreadable, wrong attribution or illegal outcome) must never be treated as
    # "no decision" — fail-closed before scheduling (review gate 1 round 3, R1).
    # The corrupt file is left in place as evidence; the CLI of this id gets no
    # acceptance and times out conservatively.
    if _result_path(request_id).exists():
        with _best_effort():
            _unlink_exclusive(newest)
        return None, OUTCOME_REJECTED_INVALID, "corrupt result record (fail-closed)"

    # A claim without a result means a previous claimer never committed its
    # decision (failed acceptance write / crash). Never schedule twice for one
    # id and never let a re-published id race a stale claim.
    if _claim_path(request_id).exists():
        with _best_effort():
            _unlink_exclusive(newest)
            write_result(request_id, OUTCOME_REJECTED_INVALID,
                         detail="orphaned claim: a previous acceptance was never committed",
                         actor_ref=actor)
        return None, OUTCOME_REJECTED_INVALID, "orphaned claim"

    if not _generation_matches(payload["target_generation"]):
        # Not ours: leave the file for its real target (or expiry), but tell *this*
        # wake-up's caller nothing was claimed.
        return None, OUTCOME_REJECTED_TARGET_MISMATCH, "target generation mismatch"

    mine = compute_service_identity()
    if mine is None or not _identity_matches(payload["service_identity"], mine):
        with _best_effort():
            if not _unlink_exclusive(newest):
                return None, "", "claim lost to another consumer"
            write_result(request_id, OUTCOME_REJECTED_INVALID, detail="service identity mismatch",
                         actor_ref=actor)
        return None, OUTCOME_REJECTED_INVALID, "service identity mismatch"

    identity_key = service_identity_key(mine)
    blocking = cooldown_active_for(identity_key, now=now)
    if blocking is not None and blocking.get("request_id") != request_id:
        with _best_effort():
            if not _unlink_exclusive(newest):
                return None, "", "claim lost to another consumer"
            write_result(request_id, OUTCOME_REJECTED_COOLDOWN,
                         detail=f'cooldown window active until +{COOLDOWN_WINDOW_S:.0f}s',
                         service_key=identity_key, actor_ref=actor)
        return None, OUTCOME_REJECTED_COOLDOWN, "cooldown active"

    # Claim: the O_EXCL create IS the consume-once gate — only its winner may
    # reach scheduling (review items 5/6). The pending file is then removed so a
    # re-sent SIGUSR1 cannot reprocess it.
    me = process_generation()
    if not _exclusive_write_json(_claim_path(request_id), {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "request_id": request_id,
        "generation": me,
        "claimed_at": now,
        "service_key": identity_key,
    }):
        return None, "", "claim lost to another consumer"
    _unlink_quiet(newest)
    return payload, OUTCOME_ACCEPTED, ""


# ---------------------------------------------------------------------------
# SIGUSR1 wake-up processing (the handler only dispatches this as a task;
# no filesystem work happens inside the raw signal callback).
# ---------------------------------------------------------------------------

def process_sigusr1_request(schedule_restart, cancel_restart=None) -> None:
    """Handle one SIGUSR1 wake-up with deferred-request awareness (I3).

    ``schedule_restart()`` returns one of:
      ``"scheduled"``       — a NEW drain-first restart was scheduled (I1 writes
                              the acceptance only now);
      ``"already"``         — a restart was already scheduled (explicit
                              ``superseded`` outcome, still success for the CLI);
      ``"stopping"``        — the gateway is stopping WITHOUT a restart path:
                              explicit ``rejected_stopping`` outcome, never a
                              fake success.
    ``cancel_restart`` (optional) is invoked when the acceptance cannot be
    persisted after retries: a controlled exit must never run without a
    persisted decision (I1), so the scheduled restart is cancelled instead.

    Serialization guarantee (review gate 1 round 2, items 2/6/7): this whole
    function runs SYNCHRONOUSLY in one event-loop turn — it contains no
    ``await``, and the retry backoff uses blocking ``time.sleep`` (which stalls
    the loop rather than yielding to other tasks). Therefore the restart task
    created by ``schedule_restart()`` cannot run, another wake cannot consume,
    and no result writer can interleave between claim, scheduling and the
    acceptance write. An ``"already"`` answer can consequently only refer to a
    schedule whose own turn already committed (or cancelled) its decision.

    Behaviour:
    - No pending deferred request AND protocol traffic never seen this
      generation → legacy path: schedule exactly as the pre-deferred handler.
    - Pending request valid for this generation → claim, then schedule, then
      persist ACCEPTED; unknown/failed listing or a second wake after the
      traffic was consumed takes NO action (double-wake races cannot resurrect
      a legacy restart — review item 1).
    - Pending request present but INVALID → persist the rejection and do NOT
      restart: an invalid request must never fall through to the legacy
      unconditional restart (attacker-dropped files must not cause restarts).
    """
    global _deferred_engaged

    cleanup_expired()
    pending = list_pending_request_files()
    if pending is None:
        logger.error(
            "SIGUSR1: cannot list deferred restart requests — refusing both the "
            "deferred path and the legacy restart (invariant I3, fail-closed)")
        return
    if not pending:
        if _deferred_engaged:
            logger.info(
                "SIGUSR1: deferred restart protocol traffic was consumed earlier "
                "this generation — no legacy restart fallback (invariant I3)")
            return
        # Legacy compatibility: plain SIGUSR1 restart behaviour, unchanged.
        schedule_restart()
        return

    _deferred_engaged = True

    # Decide on EVERY pending request (one wake must not leave other CLIs
    # without their explicit outcomes): the first claimable one schedules, the
    # rest get explicit cooldown/superseded/replay outcomes from their own runs.
    while True:
        payload, outcome, detail = consume_pending_request()
        if payload is None and outcome == "":
            return  # listing vanished mid-loop, or claim lost: no action (I3)
        if payload is None:
            logger.warning(
                "SIGUSR1: deferred restart request not claimed (outcome=%s detail=%s) — "
                "no legacy restart fallback (invariant I3)", outcome, detail)
            if outcome == OUTCOME_REJECTED_TARGET_MISMATCH:
                return  # addressed elsewhere: leave the rest alone
            continue

        request_id = payload["request_id"]
        if outcome == OUTCOME_DUPLICATE:
            # Replay of an already-ACCEPTED request: the original decision stands
            # (I2) — nothing to schedule and the original result is untouched.
            logger.info("SIGUSR1: deferred restart %s duplicates an accepted request", request_id)
            with _best_effort():
                append_audit(request_id, OUTCOME_DUPLICATE,
                             actor_ref=_sanitize_actor_ref(payload.get("actor_ref")),
                             generation=payload.get("target_generation"),
                             service_key=service_identity_key(payload["service_identity"]),
                             detail="replay of an accepted request")
            continue
        if outcome != OUTCOME_ACCEPTED:
            # Replay of an already-decided (rejected) request: repeat that decision
            # without overwriting it (write_result refuses overwrites anyway).
            continue

        scheduled = "already"
        try:
            scheduled = schedule_restart()
        except Exception as exc:
            logger.warning("defer-restart: scheduling failed after claim: %s", exc)
            scheduled = "stopping"
        key = service_identity_key(payload["service_identity"])
        if scheduled == "scheduled":
            # Linearization point (I1): acceptance is recorded only for a restart
            # that actually got scheduled — and must persist before this
            # generation can exit, else the restart is cancelled.
            if not write_result_with_retry(request_id, OUTCOME_ACCEPTED,
                                           generation=payload.get("target_generation"),
                                           service_key=key,
                                           actor_ref=_sanitize_actor_ref(payload.get("actor_ref"))):
                logger.critical(
                    "defer-restart: could not persist acceptance for %s — cancelling "
                    "the scheduled restart (no exit without a decision, invariant I1)",
                    request_id)
                if cancel_restart is not None:
                    with _best_effort():
                        cancel_restart()
                continue
            with _best_effort():
                record_cooldown(key, request_id)
        elif scheduled == "already":
            with _best_effort():
                write_result(request_id, OUTCOME_SUPERSEDED,
                             detail="restart already scheduled for this gateway",
                             service_key=key,
                             actor_ref=_sanitize_actor_ref(payload.get("actor_ref")))
        else:  # "stopping" or unexpected value
            with _best_effort():
                write_result(request_id, OUTCOME_REJECTED_STOPPING,
                             detail="gateway is stopping without a scheduled restart",
                             service_key=key,
                             actor_ref=_sanitize_actor_ref(payload.get("actor_ref")))


# ---------------------------------------------------------------------------
# Expiry cleanup — never overwrites an existing decision (I1) and never
# deletes accepted/unknown audit evidence inside its retention window.
# Consuming an expired pending request is protocol traffic: it engages the
# I3 latch so a same-generation wake can never resurrect a legacy restart.
# ---------------------------------------------------------------------------

def cleanup_expired(*, now: Optional[float] = None) -> None:
    global _deferred_engaged
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
                # write_result never overwrites an earlier decision (I1).
                if read_result(request_id) is None:
                    write_result(request_id, OUTCOME_EXPIRED, detail="pending request expired unconsumed")
                _unlink_quiet(path)
            _deferred_engaged = True
        elif name.startswith("result-") and now - mtime > RESULT_RETENTION_S:
            _unlink_quiet(path)
        elif name.startswith("claim-") and now - mtime > RESULT_RETENTION_S:
            _unlink_quiet(path)
        elif name == "cooldown.json" and now - mtime > RESULT_RETENTION_S:
            _unlink_quiet(path)
