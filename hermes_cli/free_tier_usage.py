"""Lifetime free-tier usage, beside shared auth and under its canonical lock.

Keep usage separate from rotating credentials: stale profile copies, token refreshes,
and signing out must not reset the allowance for the SAME anonymous identity. Only
opaque identity digests and counters are stored here; never credential values.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

TOOL_CALL_CAP = 10
LIMIT_REASON = "free_tier_limit"
_FAILED_IDENTITIES: set[str] = set()
LIMIT_NOTICE = (
    "You've reached the free tier's 10-tool-call limit. "
    "Sign in with /login to continue, or use /model to choose a local model or another provider."
)


def _usage_path() -> Path:
    from hermes_cli.auth_nous import _nous_shared_store_path
    return _nous_shared_store_path().with_name("free-tier-usage.json")


def current_identity() -> str | None:
    """Resolve the actual auth source; a completed shared sign-in supersedes a stale guest.

    Reading this never provisions, refreshes, or changes a profile's credentials.
    """
    from hermes_cli.anon_auth import _shared_identity_key, current_nous_state, is_guest_state
    from hermes_cli.auth import _provider_state_transaction
    from hermes_cli.auth_nous import _nous_shared_store_lock, _read_shared_nous_state

    if not is_guest_state(current_nous_state()):
        return None
    with _provider_state_transaction("nous") as (_, state, _source):
        if not is_guest_state(state):
            return None
        with _nous_shared_store_lock():
            shared = _read_shared_nous_state()
            if shared and not is_guest_state(shared):
                return None
            key = _shared_identity_key(state)
            return hashlib.sha256(key.encode()).hexdigest() if isinstance(key, str) and key else None


def _read_usage() -> dict:
    path = _usage_path()
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Free-tier usage store is invalid")
    return data


def _count(data: dict, identity: str | None) -> int:
    value = data.get(identity, 0) if identity else 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Free-tier usage counter is invalid")
    return value


def identity_status(identity: str | None) -> dict:
    try:
        used = _count(_read_usage(), identity) if identity else 0
    except (OSError, ValueError, RuntimeError):
        if identity:
            _FAILED_IDENTITIES.add(identity)
        used = 0
    return {"tool_calls_used": used, "tool_call_cap": TOOL_CALL_CAP,
            "capped": used >= TOOL_CALL_CAP or identity in _FAILED_IDENTITIES}


def status() -> dict:
    return identity_status(current_identity())


def record_completed_tool(identity: str) -> None:
    """Atomic increment, including completions after the cap within an admitted turn."""
    from hermes_cli.auth import _write_private_file_atomic
    from hermes_cli.auth_nous import _nous_shared_store_lock
    try:
        with _nous_shared_store_lock():
            data = _read_usage()
            data[identity] = _count(data, identity) + 1
            _write_private_file_atomic(_usage_path(), json.dumps(data, sort_keys=True), fsync_dir=True)
    except (OSError, ValueError, RuntimeError):
        # Finish the active turn without losing the real tool result. All subsequent
        # admissions in this process fail closed for this identity, not just this agent.
        _FAILED_IDENTITIES.add(identity)
