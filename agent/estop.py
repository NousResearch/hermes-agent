"""Global emergency stop (ESTOP) — a resumable pause for NEW work only.

``hermes pause`` writes a sentinel at ``$HERMES_HOME/ESTOP``; ``hermes resume``
removes it. While it exists the cron scheduler, kanban dispatcher and new gateway
turns skip work; in-flight work is never killed. The check is one or two uncached
``os.stat`` calls (process home + fleet root when they differ). The body is optional
JSON ``{"generation", "reason", "engaged_at", "expires_at", "allow"}``; a corrupt/empty
file still counts as engaged (fail safe, e.g. ``touch ~/.hermes/ESTOP``).

Three fields extend the primitive, all so an unattended hold neither strands the fleet
nor silently stops holding:

``generation``
    Opaque token minted per engagement. It is the sentinel's IDENTITY, so the deadman
    retires exactly the generation it EXAMINED and never a successor that re-armed in
    the interval since. Legacy/hand-written sentinels carry no token; their identity
    falls back to the file's ``mtime_ns:size``.
``expires_at``
    A DEADMAN (ISO-8601). Past that instant the sentinel is NOT engaged, so a window
    job that dies between arm and release cannot freeze the fleet forever. The lift is
    logged once, loudly, per engagement, and the dead generation is retired (a re-arm
    starts a fresh engagement).
``allow``
    ``{"user_ids": [...], "profiles": [...]}`` — who keeps working THROUGH the pause.
    ``user_ids`` is the operator's authenticated id and is the ONLY grant; ``profiles``
    NARROWS admission to those serving profiles and never grants on its own. Absent
    allowlist = nobody exempt.

Every hold on ``_candidate_sentinel_paths()`` composes: admission requires at least one
ACTIVE hold and the consent of EVERY one of them (deny wins), so a permissive
profile-local sentinel cannot weaken a stricter fleet-root one.

Ported from gastownhall/gastown estop.go (MIT).
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# Same profile-aware / fleet-root resolvers the file-safety guards use (fail-open to ~/.hermes).
from agent.file_safety import _hermes_home_path as _hermes_home, _hermes_root_path as _canonical_root

SENTINEL_NAME = "ESTOP"
# Per-engagement identity token — what lets the deadman tell the generation it examined
# apart from a successor that re-armed in the interval since.
GENERATION_FIELD = "generation"
# Retiring a dead generation may reveal a successor re-armed at the same path; re-check it,
# but bound the passes so a pathological re-arm loop can never spin inside a gate check.
_RETIRE_PASSES = 4

logger = logging.getLogger(__name__)

# Per-component "logged already for this engagement" flags: log once per engagement, not per tick.
_log_lock = threading.Lock()
_logged_components: set[str] = set()
# Sentinel path -> GENERATION of the last logged EXPIRY, so the deadman's lift is one loud
# line per engagement rather than one per check (a re-arm earns a fresh token, so it logs).
_expired_logged: dict[str, str] = {}

_DURATION_RE = re.compile(r"^\s*(\d+)\s*([smhd]?)\s*$")
_DURATION_UNITS = {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_duration(value: Any) -> Optional[int]:
    """Seconds for ``45m`` / ``90m`` / ``2h`` / ``90`` / ``90``-as-int; None when unusable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if value > 0 else None
    match = _DURATION_RE.match(str(value))
    if not match:
        return None
    seconds = int(match.group(1)) * _DURATION_UNITS[match.group(2)]
    return seconds or None


def sentinel_path() -> Path:
    """Path of the ESTOP sentinel this process would write on `hermes pause`."""
    return _hermes_home() / SENTINEL_NAME


def _candidate_sentinel_paths() -> list:
    """Profile home first, then the fleet root if it is a different directory: a profile
    gateway (HERMES_HOME=~/.hermes/profiles/<n>) must still honor an operator's ~/.hermes/ESTOP."""
    primary = sentinel_path()
    try:
        root = _canonical_root() / SENTINEL_NAME
    except Exception:
        return [primary]
    try:
        distinct = root.resolve() != primary.resolve()
    except Exception:
        # Non-Path test doubles fail .resolve(); plain equality still dedupes.
        distinct = root != primary
    return [primary, root] if distinct else [primary]


def _read_payload(path: Path) -> Optional[dict]:
    """Parsed sentinel body, or None when absent/unreadable/not a JSON object.

    None means "no usable body", which every caller must treat as ENGAGED (fail safe) —
    it is never a reason to lift the pause.

    ``utf-8-sig`` so a BOM left by a Windows editor does not turn the body into an
    unparsable one (a BOM'd ``expires_at`` would otherwise read as "no expiry").
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, AttributeError):
        return None
    return raw if isinstance(raw, dict) else None


def _parse_stamp(value: Any) -> Optional[datetime]:
    """Aware datetime for an ISO-8601 stamp; None when absent or unparsable (a naive
    stamp is read as UTC). Unparsable NEVER means expired — the pause stays held."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _normalize_allow(allow: Any) -> dict:
    """``{"user_ids": [...], "profiles": [...]}`` with string values and no empties."""
    if not isinstance(allow, dict):
        return {}
    normalized: dict = {}
    for key in ("user_ids", "profiles"):
        value = allow.get(key)
        if isinstance(value, (str, int, float)):
            value = [value]
        entries = [str(item).strip() for item in value or [] if str(item).strip()]
        if entries:
            normalized[key] = entries
    return normalized


def _resolve_expiry(value: Any) -> Optional[datetime]:
    """Absolute expiry from an ISO-8601 string or an aware datetime; None otherwise."""
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    return _parse_stamp(value)


def _new_generation() -> str:
    """Opaque per-engagement identity, minted on every write."""
    return uuid.uuid4().hex


def _file_key(path: Path) -> str:
    """Fallback identity for a token-less (legacy or hand-written) sentinel."""
    try:
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    except (OSError, AttributeError, TypeError, ValueError):
        return "unknown"


def _engagement_key(payload: Optional[dict], path: Path) -> str:
    """Identity of ONE generation: its payload token when present, else the file's
    ``mtime_ns:size``.

    The token is what distinguishes a successor whose file is otherwise indistinguishable
    (same byte length, same ``mtime_ns``); the pre-fix key re-derived the file's identity
    AT removal time, which compares the file with ITSELF and deletes whatever is there.
    """
    if isinstance(payload, dict):
        token = payload.get(GENERATION_FIELD)
        if isinstance(token, str) and token:
            return f"gen:{token}"
    return _file_key(path)


def _read_hold(path: Path) -> tuple[Optional[dict], Optional[datetime]]:
    """ONE read of the sentinel: its parsed body and, when parsable, its deadman.

    ``None`` body = absent/unreadable/corrupt → the caller holds (fail safe). ``None``
    deadline = no usable ``expires_at`` → the caller also holds. Both come from the same
    bytes, so the expiry DECISION and the generation IDENTITY always agree.
    """
    payload = _read_payload(path)
    if payload is None:
        return None, None
    return payload, _parse_stamp(payload.get("expires_at"))


def _is_expired(path: Path) -> bool:
    """True only for a sentinel whose body carries a PARSABLE ``expires_at`` in the past."""
    _, expires = _read_hold(path)
    return expires is not None and datetime.now(timezone.utc) >= expires


def _claim_path(path: Path) -> Path:
    """A same-directory sibling used to CLAIM one generation (a rename there is atomic)."""
    return path.with_name(f".{path.name}.retiring-{os.getpid()}-{uuid.uuid4().hex}")


def _restore_claim(claim: Path, path: Path) -> None:
    """Put a claimed generation back, never clobbering one that re-armed meanwhile."""
    try:
        os.link(claim, path)  # atomic: refuses when a newer sentinel already took the path
    except FileExistsError:
        pass  # a successor won the path; the claim is a stale duplicate of a DEAD generation
    except OSError:
        with suppress(OSError):
            if not path.exists():
                os.rename(claim, path)
                return
    with suppress(OSError):
        claim.unlink()


def _retire_expired(path: Path, examined_key: str) -> None:
    """Retire ONLY the generation whose key the caller examined as dead.

    The expiry decision and the removal are separated by an unbounded interval, so a
    successor re-arm can land in between. Deriving the key from the file AT removal time
    compares the file with ITSELF and deletes whatever is there — including the successor.
    So claim the file atomically (rename it aside), re-verify that the claimed bytes ARE
    the generation examined, and put them back when they are not. A successor therefore
    always survives, and the caller re-checks the path either way, so the brief claim
    interval never leaves an ACTIVE hold unseen beyond it. A failure to remove is
    swallowed: the pause is already lifted (``_is_expired`` stays True for the dead one).
    """
    claim = _claim_path(path)
    try:
        os.rename(path, claim)
    except OSError:
        return  # already gone (or unwritable): the pause is lifted either way
    if _engagement_key(_read_payload(claim), claim) != examined_key:
        _restore_claim(claim, path)  # a successor re-armed: it survives, this one was not ours
        return
    with _log_lock:
        first_report = _expired_logged.get(str(path)) != examined_key
        _expired_logged[str(path)] = examined_key
    removed = False
    try:
        claim.unlink()
        removed = True
    except OSError:
        pass
    if first_report:
        logger.warning(
            "Global emergency stop at %s EXPIRED at its deadman TTL — the pause has lifted and "
            "dispatch resumes. The sentinel %s; run `hermes pause --ttl <dur>` to re-arm.",
            path, "was removed" if removed else "could not be removed (it stays inert)",
        )


def is_engaged() -> bool:
    """True if ANY candidate sentinel exists and is NOT past its ``expires_at``; fail SAFE
    (True) on stat errors. An expired sentinel lifts the pause, logs one loud line and is
    retired — ONLY the generation examined, so a concurrent re-arm survives — so a crashed
    window job can never park the fleet."""
    saw_stat_error = False
    for path in _candidate_sentinel_paths():
        for _ in range(_RETIRE_PASSES):
            try:
                if not path.exists():
                    break
            except OSError:
                saw_stat_error = True
                break
            payload, expires = _read_hold(path)
            if expires is None or datetime.now(timezone.utc) < expires:
                return True  # active (or unreadable: fail safe)
            # Identity captured from the SAME read as the decision: a successor that re-arms
            # before the claim is therefore distinguishable, and will not be retired.
            _retire_expired(path, _engagement_key(payload, path))
            # Loop: retiring the dead generation may have revealed a successor at this path.
        else:
            return True  # bounded passes exhausted (re-arm loop): fail safe and hold
    return saw_stat_error


def _new_expiry(ttl: Any, expires_at: Any, previous: dict, now: datetime) -> Optional[datetime]:
    """The deadman for a (re-)engagement.

    An explicit ``expires_at`` wins. A SUPPLIED ``ttl`` replaces too — including with
    ``None`` when it is out of range, because an emergency stop must never REFUSE to arm
    (``hermes pause --ttl 999999999d`` arms WITHOUT a deadman rather than raising). An
    OMITTED ``ttl`` PRESERVES the standing sentinel's deadline, so a re-arm (in-band
    ``/pause <reason>``, or a no-flag ``hermes pause``) never silently strips the deadman
    the previous engagement armed.
    """
    if expires_at is not None:
        with suppress(Exception):
            return _resolve_expiry(expires_at)
        return None
    if ttl is None or not str(ttl).strip():
        kept = _parse_stamp((previous or {}).get("expires_at"))
        # A deadline that has already PASSED can never hold: preserving it would arm a
        # pause that is instantly lifted — a silent no-op re-arm. Drop it and say so.
        return None if kept is None or kept <= now else kept
    seconds = parse_duration(ttl)
    if not seconds:
        return None  # requested but unusable: engage without a deadman
    try:
        return now + timedelta(seconds=seconds)
    except (OverflowError, ValueError):
        # Out of range (past datetime.MAX): arm, but with no deadman at all — never an
        # absurd expiry, and never an exception out of a gate that must not fail open.
        return None


def engage(
    reason: Optional[str] = None,
    allow: Optional[dict] = None,
    ttl: Any = None,
    expires_at: Any = None,
) -> Path:
    """Create (or re-arm) the ESTOP sentinel. Idempotent; re-engaging rewrites the file.

    ``allow`` is ``{"user_ids": [...], "profiles": [...]}`` (``profiles`` NARROWS, never
    grants); pass ``allow=None`` to KEEP the standing allowlist. An omitted ``ttl`` and
    ``expires_at`` KEEP the standing deadman — a re-arm must not silently strip what the
    previous engagement armed; pass an explicit value to replace it. ``ttl`` accepts
    ``45m``/``90m``/``2h``/seconds; an out-of-range value arms WITHOUT a deadman (it never
    raises). The payload carries a fresh ``generation`` token identifying this engagement.
    """
    path = sentinel_path()
    now = datetime.now(timezone.utc)
    previous = _read_payload(path) or {}

    expiry = _new_expiry(ttl, expires_at, previous, now)
    normalized_allow = (
        _normalize_allow(previous.get("allow")) if allow is None else _normalize_allow(allow)
    )

    payload: dict = {
        GENERATION_FIELD: _new_generation(),
        "engaged_at": now.isoformat(),
        "reason": reason or None,
    }
    if expiry is not None:
        payload["expires_at"] = expiry.isoformat()
    if normalized_allow:
        payload["allow"] = normalized_allow
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        with suppress(OSError):  # Best effort: an empty/partial sentinel still pauses (fail safe).
            path.touch(exist_ok=True)
    return path


def disengage() -> bool:
    """Remove every visible sentinel (process-local and fleet-root)."""
    lifted = False
    for path in _candidate_sentinel_paths():
        try:
            path.unlink()
            lifted = True
        except (OSError, AttributeError):
            continue
        with _log_lock:
            _expired_logged.pop(str(path), None)
    return lifted


def get_state() -> Optional[dict]:
    """``{"reason", "engaged_at", "expires_at", "allow"}`` or None when not engaged; an
    unreadable/corrupt body still reports engaged with the fields None/{}. An expired
    sentinel is NOT engaged."""
    if not is_engaged():
        return None
    state = {"reason": None, "engaged_at": None, "expires_at": None, "allow": {}}
    found = False
    for path in _candidate_sentinel_paths():
        try:
            if not path.exists():
                continue
        except OSError:
            return state
        except AttributeError:
            continue
        found = True
        payload = _read_payload(path)
        if payload is not None:
            state = {
                "reason": payload.get("reason") or None,
                "engaged_at": payload.get("engaged_at") or None,
                "expires_at": payload.get("expires_at") or None,
                "allow": _normalize_allow(payload.get("allow")),
            }
            break
    return state if found else None


def _active_holds() -> list:
    """Payload of every ACTIVE hold on the candidate paths.

    ``None`` marks an active but unreadable/corrupt hold — it admits nobody (deny by
    intersection). A dead generation is retired (generation-safe) and does NOT count; a
    successor that re-armed at the same path does. No holds at all = nobody exempt.
    """
    holds: list = []
    for path in _candidate_sentinel_paths():
        try:
            if not path.exists():
                continue
        except OSError:
            holds.append(None)  # cannot tell: deny by intersection
            continue
        payload, expires = _read_hold(path)
        if expires is not None and datetime.now(timezone.utc) >= expires:
            _retire_expired(path, _engagement_key(payload, path))
            try:
                if not path.exists():
                    continue
            except OSError:
                holds.append(None)
                continue
            if _is_expired(path):
                continue  # a successor that is ALSO dead is not an active hold
            payload = _read_payload(path)
        holds.append(payload)
    return holds


def _hold_admits(hold: Optional[dict], user_id: Optional[str], profile: Optional[str]) -> bool:
    """Whether ONE hold admits this identity.

    Identity is the ONLY grant: ``user_ids`` must contain it. A present ``profiles`` list
    then NARROWS admission to those serving profiles — it never grants on its own, so a
    routing coordinate cannot escalate anyone past the authority they lack. An unreadable
    hold, or one with no allowlist, admits nobody.
    """
    if not isinstance(hold, dict):
        return False
    allow = _normalize_allow(hold.get("allow"))
    if user_id is None or str(user_id) not in (allow.get("user_ids") or []):
        return False
    profiles = allow.get("profiles") or []
    return not profiles or (profile is not None and str(profile) in profiles)


def is_allowed(user_id: Optional[str] = None, profile: Optional[str] = None) -> bool:
    """True when EVERY active hold admits this authenticated identity (deny wins).

    ``user_ids`` is the operator's authenticated id and the only grant; ``profile`` is a
    NARROWING key — a hold that names ``profiles`` admits only those serving profiles, and
    a profile alone never admits anyone. No active hold — or any active hold that is
    unreadable or lists no allowlist — admits nobody. Every hold composes, so a permissive
    sentinel cannot weaken a stricter one on another candidate path.
    """
    holds = _active_holds()
    if not holds:
        return False
    return all(_hold_admits(hold, user_id, profile) for hold in holds)


def paused_reply() -> Optional[str]:
    """Short user-facing notice for new gateway turns, or None if not paused."""
    state = get_state()
    if state is None:
        return None
    tag = f" ({state['reason']})" if state.get("reason") else ""
    until = f" Auto-resumes {state['expires_at']}." if state.get("expires_at") else ""
    return (
        f"⏸️ Hermes is paused{tag}. New work is on hold; run `hermes resume` to pick "
        f"things back up.{until}"
    )


def check_paused(component: str, logger: logging.Logger) -> bool:
    """Return True when engaged, logging once per engagement per component (re-armed after a resume)."""
    if not is_engaged():
        with _log_lock:
            _logged_components.discard(component)
        return False
    with _log_lock:
        first = component not in _logged_components
        _logged_components.add(component)
    if first:
        state = get_state() or {}
        reason = state.get("reason")
        suffix = f" (reason: {reason})" if reason else ""
        until = f" [auto-resumes {state.get('expires_at')}]" if state.get("expires_at") else ""
        logger.info(
            "%s dispatch paused by global emergency stop%s%s — remove with `hermes resume` (%s)",
            component, suffix, until, sentinel_path(),
        )
    return True
