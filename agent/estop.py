"""Global emergency stop (ESTOP) — a resumable pause for NEW work only.

``hermes pause`` writes a sentinel at ``$HERMES_HOME/ESTOP``; ``hermes resume``
removes it. While it exists the cron scheduler, kanban dispatcher and new gateway
turns skip work; in-flight work is never killed. The check is one or two uncached
``os.stat`` calls (process home + fleet root when they differ). The body is optional
JSON ``{"reason", "engaged_at", "expires_at", "allow"}``; a corrupt/empty file still
counts as engaged (fail safe, e.g. ``touch ~/.hermes/ESTOP``).

Two fields extend the primitive, both so an unattended hold cannot strand the fleet:

``expires_at``
    A DEADMAN (ISO-8601). Past that instant the sentinel is NOT engaged, so a window
    job that dies between arm and release cannot freeze the fleet forever. The lift is
    logged once, loudly, per engagement, and the dead file is retired (a re-arm starts
    a fresh engagement).
``allow``
    ``{"user_ids": [...], "profiles": [...]}`` — who keeps working THROUGH the pause.
    Identity first: ``user_ids`` is the operator's authenticated id. ``profiles`` is a
    secondary key for a maintenance lane, because a profile name is whatever the
    operator happens to be chatting through at 00:00. Absent allowlist = nobody exempt.

Ported from gastownhall/gastown estop.go (MIT).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# Same profile-aware / fleet-root resolvers the file-safety guards use (fail-open to ~/.hermes).
from agent.file_safety import _hermes_home_path as _hermes_home, _hermes_root_path as _canonical_root

SENTINEL_NAME = "ESTOP"

logger = logging.getLogger(__name__)

# Per-component "logged already for this engagement" flags: log once per engagement, not per tick.
_log_lock = threading.Lock()
_logged_components: set[str] = set()
# Sentinel path -> engagement key (mtime_ns:size) of the last logged EXPIRY, so the
# deadman's lift is one loud line per engagement rather than one per check.
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
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
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


def _engagement_key(path: Path) -> str:
    """Identity of the CURRENT engagement on disk: a re-engage rewrites the file and so
    earns a fresh key (and a fresh loud line)."""
    try:
        stat = path.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    except (OSError, AttributeError, TypeError, ValueError):
        return "unknown"


def _is_expired(path: Path) -> bool:
    """True only for a sentinel whose body carries a PARSABLE ``expires_at`` in the past."""
    payload = _read_payload(path)
    if payload is None:
        return False  # corrupt/unreadable body: still engaged (fail safe)
    expires = _parse_stamp(payload.get("expires_at"))
    if expires is None:
        return False
    return datetime.now(timezone.utc) >= expires


def _retire_expired(path: Path) -> None:
    """Report a deadman expiry once, then remove the dead sentinel.

    The removal is guarded by a re-check of the engagement key, so a re-arm that lands
    between the expiry read and the unlink loses nothing; any failure to remove is
    swallowed (the pause is already lifted — the leftover file must never re-hold it,
    because :func:`_is_expired` stays True).
    """
    key = _engagement_key(path)
    with _log_lock:
        first_report = _expired_logged.get(str(path)) != key
        _expired_logged[str(path)] = key
    if not first_report:
        return
    removed = False
    try:
        if _engagement_key(path) == key:
            path.unlink()
            removed = True
    except (OSError, AttributeError, TypeError, ValueError):
        pass
    logger.warning(
        "Global emergency stop at %s EXPIRED at its deadman TTL — the pause has lifted and "
        "dispatch resumes. The sentinel %s; run `hermes pause --ttl <dur>` to re-arm.",
        path, "was removed" if removed else "could not be removed (it stays inert)",
    )


def is_engaged() -> bool:
    """True if ANY candidate sentinel exists and is NOT past its ``expires_at``; fail SAFE
    (True) on stat errors. An expired sentinel lifts the pause, logs one loud line and is
    retired, so a crashed window job can never park the fleet."""
    saw_stat_error = False
    for path in _candidate_sentinel_paths():
        try:
            if not path.exists():
                continue
        except OSError:
            saw_stat_error = True
            continue
        if _is_expired(path):
            _retire_expired(path)
            continue
        return True
    return saw_stat_error


def engage(
    reason: Optional[str] = None,
    allow: Optional[dict] = None,
    ttl: Any = None,
    expires_at: Any = None,
) -> Path:
    """Create the ESTOP sentinel. Idempotent; re-engaging rewrites the file.

    ``allow`` is ``{"user_ids": [...], "profiles": [...]}`` (either key optional, values
    coerced to strings); ``ttl`` accepts ``45m``/``90m``/``2h``/seconds and sets the
    deadman, or pass ``expires_at`` (ISO-8601 / aware datetime) directly. An unusable ttl
    still engages — it never half-arms a pause — and is reported by the CLI.
    """
    path = sentinel_path()
    now = datetime.now(timezone.utc)
    expiry = _resolve_expiry(expires_at)
    if expiry is None:
        seconds = parse_duration(ttl)
        if seconds:
            expiry = now + timedelta(seconds=seconds)
    payload: dict = {"engaged_at": now.isoformat(), "reason": reason or None}
    if expiry is not None:
        payload["expires_at"] = expiry.isoformat()
    normalized_allow = _normalize_allow(allow)
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


def is_allowed(
    user_id: Optional[str] = None,
    profile: Optional[str] = None,
    state: Optional[dict] = None,
) -> bool:
    """True when the ACTIVE sentinel's allowlist admits this authenticated identity.

    PRIMARY key is ``user_id`` (identity survives a platform/profile change); ``profile``
    is the secondary key for a maintenance lane. No allowlist — or an unreadable sentinel —
    admits nobody. Pass ``state`` when the caller already read it, to save one stat+read.
    """
    if state is None:
        state = get_state()
    if not state:
        return False
    allow = state.get("allow") or {}
    if user_id is not None and str(user_id) in (allow.get("user_ids") or []):
        return True
    return bool(profile) and str(profile) in (allow.get("profiles") or [])


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


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import os  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
