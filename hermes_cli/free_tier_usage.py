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
ONBOARDING_TURN_CAP = 20
TASK_CLARIFICATION_TURNS = 2
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
    if isinstance(value, dict):
        value = value.get("tool_calls_used", 0)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Free-tier usage counter is invalid")
    return value


def _onboarding_state(data: dict, identity: str) -> tuple[int, bool]:
    entry = data.get(identity, 0)
    _count(data, identity)  # Validate legacy counters before migrating them.
    if not isinstance(entry, dict):
        return 0, False
    turns, complete = entry.get("onboarding_turns_used", 0), entry.get("onboarding_complete", False)
    if not isinstance(turns, int) or isinstance(turns, bool) or turns < 0 or not isinstance(complete, bool):
        raise ValueError("Onboarding usage counter is invalid")
    if "onboarding_task_chosen_after_turn" in entry:
        chosen = entry["onboarding_task_chosen_after_turn"]
        completed = entry.get("onboarding_task_completed_turns")
        if (type(chosen) is not int or not 0 <= chosen <= turns
                or not isinstance(completed, list) or len(completed) > TASK_CLARIFICATION_TURNS
                or any(type(turn) is not int or not chosen < turn <= turns for turn in completed)
                or len(set(completed)) != len(completed)):
            raise ValueError("Onboarding task progress is invalid")
        complete = complete or len(completed) >= TASK_CLARIFICATION_TURNS
    return turns, complete or turns >= ONBOARDING_TURN_CAP


def onboarding_available(identity: str | None) -> bool:
    """Read-only eligibility, never a reservation or authorization on its own."""
    if not identity or identity in _FAILED_IDENTITIES:
        return False
    try:
        return not _onboarding_state(_read_usage(), identity)[1]
    except (OSError, ValueError, RuntimeError):
        _FAILED_IDENTITIES.add(identity)
        return False


def reserve_onboarding_turn(identity: str) -> int | None:
    """Return a durable admission ordinal, also the task-choice completion token.

    Choice snapshots the last ordinal under this same lock, so a response already
    in flight when the user chooses cannot count toward the two completed turns.
    """
    from hermes_cli.auth import _save_private_json
    from hermes_cli.auth_nous import _nous_shared_store_lock
    if identity in _FAILED_IDENTITIES:
        return None
    try:
        with _nous_shared_store_lock():
            data = _read_usage()
            turns, complete = _onboarding_state(data, identity)
            if complete:
                return None
            entry = _onboarding_entry(data, identity)
            entry.update(onboarding_turns_used=turns + 1,
                         onboarding_complete=turns + 1 >= ONBOARDING_TURN_CAP)
            _save_private_json(_usage_path(), data, sort_keys=True, fsync_dir=True)
        return turns + 1
    except (OSError, ValueError, RuntimeError):
        _FAILED_IDENTITIES.add(identity)
        return None


def _onboarding_entry(data: dict, identity: str) -> dict:
    """Migrate legacy tool-only counters without discarding task-choice progress."""
    if not isinstance(data.get(identity), dict):
        data[identity] = {"tool_calls_used": _count(data, identity)}
    return data[identity]


def choose_onboarding_task(identity: str | None) -> None:
    """First-write task marker; retries, reselects and profile rebuilds never renew grace."""
    from hermes_cli.auth import _save_private_json
    from hermes_cli.auth_nous import _nous_shared_store_lock
    if not identity:
        return
    with _nous_shared_store_lock():
        data = _read_usage()
        turns, complete = _onboarding_state(data, identity)
        entry = _onboarding_entry(data, identity)
        if complete or "onboarding_task_chosen_after_turn" in entry:
            return
        entry["onboarding_task_chosen_after_turn"] = turns
        entry["onboarding_task_completed_turns"] = []
        _save_private_json(_usage_path(), data, sort_keys=True, fsync_dir=True)


def complete_onboarding_task_turn(identity: str, admission: int) -> None:
    """Count a successful top-level response exactly once, even across process retries.

    The bounded list is both the completed-turn counter and deduplication receipt.
    Persist failures never discard the already-completed assistant response.
    """
    from hermes_cli.auth import _save_private_json
    from hermes_cli.auth_nous import _nous_shared_store_lock
    try:
        with _nous_shared_store_lock():
            data = _read_usage()
            _, complete = _onboarding_state(data, identity)
            entry = _onboarding_entry(data, identity)
            chosen_after = entry.get("onboarding_task_chosen_after_turn")
            if complete or chosen_after is None or admission <= chosen_after:
                return
            completed = entry["onboarding_task_completed_turns"]
            if admission in completed:
                return
            completed.append(admission)
            entry["onboarding_complete"] = len(completed) >= TASK_CLARIFICATION_TURNS
            _save_private_json(_usage_path(), data, sort_keys=True, fsync_dir=True)
    except (OSError, ValueError, RuntimeError):
        _FAILED_IDENTITIES.add(identity)


def finish_onboarding(identity: str | None) -> None:
    """One-way transition on real work admission; never reset tool usage.

    A failed write must stop admission rather than leave renewable grace behind.
    """
    from hermes_cli.auth import _save_private_json
    from hermes_cli.auth_nous import _nous_shared_store_lock
    if not identity:
        return
    with _nous_shared_store_lock():
        data = _read_usage()
        turns, complete = _onboarding_state(data, identity)
        if complete:
            return
        entry = _onboarding_entry(data, identity)
        entry.update(onboarding_turns_used=turns, onboarding_complete=True)
        _save_private_json(_usage_path(), data, sort_keys=True, fsync_dir=True)


def identity_status(identity: str | None, *, guide: bool = False) -> dict:
    complete = False
    try:
        data = _read_usage() if identity else {}
        used = _count(data, identity)
        if guide and identity:
            complete = _onboarding_state(data, identity)[1]
    except (OSError, ValueError, RuntimeError):
        if identity:
            _FAILED_IDENTITIES.add(identity)
        used = 0
    state = {"tool_calls_used": used, "tool_call_cap": TOOL_CALL_CAP,
             "capped": used >= TOOL_CALL_CAP or identity in _FAILED_IDENTITIES}
    if guide and identity and (complete or identity in _FAILED_IDENTITIES):
        state["onboarding_complete"] = True
    return state


def status() -> dict:
    from hermes_cli.onboarding_profile import is_onboarding_profile
    return identity_status(current_identity(), guide=is_onboarding_profile())


def record_completed_tool(identity: str) -> None:
    """Atomic increment, including completions after the cap within an admitted turn."""
    from hermes_cli.auth import _save_private_json
    from hermes_cli.auth_nous import _nous_shared_store_lock
    try:
        with _nous_shared_store_lock():
            data = _read_usage()
            used = _count(data, identity) + 1
            if isinstance(data.get(identity), dict):
                data[identity]["tool_calls_used"] = used
            else:
                data[identity] = used
            _save_private_json(_usage_path(), data, sort_keys=True, fsync_dir=True)
    except (OSError, ValueError, RuntimeError):
        # Finish the active turn without losing the real tool result. All subsequent
        # admissions in this process fail closed for this identity, not just this agent.
        _FAILED_IDENTITIES.add(identity)
