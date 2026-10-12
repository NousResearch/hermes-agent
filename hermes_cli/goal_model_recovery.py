"""Durable, vendor-neutral reservations for a same-session worker model change."""
from __future__ import annotations

import time
import uuid
from copy import deepcopy

from hermes_cli.goal_recovery import binding

MAX_ATTEMPTS = 3
RECOVERABLE_CAUSES = frozenset({"service", "dependency", "ambiguity"})


def _route(value):
    if not isinstance(value, dict) or set(value) != {"provider", "model"}:
        return None
    if any(not isinstance(value[k], str) or not value[k].strip() or len(value[k]) > 200 for k in value):
        return None
    return {k: value[k].strip() for k in ("provider", "model")}


def settings():
    from hermes_cli.config import load_config
    section = load_config().get("goal_recovery") or {}
    value = section.get("model_handoff") if isinstance(section, dict) else None
    return value if isinstance(value, dict) else {}


def enabled_for(session_id, config=None):
    cfg = settings() if config is None else config
    sessions = cfg.get("session_ids")
    return cfg.get("enabled") is True and isinstance(sessions, list) and session_id in sessions


def reserve(manager, assessment, option, prompt):
    """Called only after the existing safe-option checks; never executes or resets a budget."""
    from hermes_cli.goals import _decision
    cfg = settings()
    prior = _route(getattr(manager, "worker_model_route", None))
    sessions = cfg.get("session_ids")
    cause = assessment.get("classifications", {}).get("cause")
    cause = cause.get("label") if isinstance(cause, dict) else cause
    if (cfg.get("enabled") is not True or prior is None or not isinstance(sessions, list)
            or manager.session_id not in sessions or cause not in RECOVERABLE_CAUSES
            or option["route"] in {"wait", "scope_change", "block"}):
        return None
    limit = cfg.get("max_attempts", MAX_ATTEMPTS)
    if type(limit) is not int or not 1 <= limit <= MAX_ATTEMPTS:
        return None
    state = manager.state
    history = deepcopy(state.model_recovery or {"epoch": 0, "attempts": []})
    attempts = history.setdefault("attempts", [])
    epoch = history.get("epoch", 0)
    current = [row for row in attempts if row.get("epoch", 0) == epoch]
    if history.get("pending"):
        return pending_decision(manager)
    routes = cfg.get("routes")
    if not isinstance(routes, list):
        return None
    tried = [row.get("target") for row in current]
    target = next((route for value in routes if (route := _route(value)) and route != prior and route not in tried), None)
    if len(current) >= limit or target is None:
        return manager._pause_decision(
            f"blocked: {state.last_reason}", "blocked", state.last_reason,
            f"⏸ Model recovery held after {len(current)}/{limit} attempts. "
            f"Recommended: {option['plan']}\nVerify: {option['verification']}. "
            "Supply the missing input or explicitly resume; no budget was extended.")
    pending = {
        "id": uuid.uuid4().hex[:16], "session_id": manager.session_id, "epoch": epoch,
        "ordinal": len(current) + 1, "limit": limit, "prior": prior, "target": target,
        "binding": binding(manager), "created_at": time.time(), "expires_at": time.time() + 86400,
        "status": "reserved", "plan": option["plan"], "verification": option["verification"],
        "prompt": manager.next_continuation_prompt() + "\n\n" + prompt + "\nRecommended path: " + option["plan"],
    }
    attempts.append(deepcopy(pending))
    history["pending"] = pending
    state.model_recovery = history
    manager._save()
    return pending_decision(manager)


def pending_decision(manager):
    from hermes_cli.goals import _decision
    pending = (manager.state.model_recovery or {}).get("pending")
    if not pending:
        return None
    if binding(manager) != pending.get("binding") or time.time() >= pending.get("expires_at", 0):
        cancel_pending(manager.state)
        manager._save()
        return None
    result = _decision("active", False, None, "model_handoff", manager.state.last_reason,
                       f"↻ Model recovery {pending['ordinal']}/{pending['limit']}: "
                       f"{pending['prior']['model']} → {pending['target']['model']}. "
                       f"Recommended: {pending['plan']}")
    result["model_handoff"] = deepcopy(pending)
    return result


def current_pending(session_id, token=None):
    """Fresh durable state wins over a manager cached while network/model validation runs."""
    from hermes_cli.goals import GoalManager
    manager = GoalManager(session_id)
    if not manager.is_active():
        return None
    pending = (manager.state.model_recovery or {}).get("pending")
    if (not pending or (token is not None and pending.get("id") != token)
            or pending.get("session_id") != session_id or time.time() >= pending.get("expires_at", 0)
            or binding(manager) != pending.get("binding")):
        return None
    return manager


def settle(manager, token, outcome, *, error_type=None):
    """Update the reservation without resurrecting a changed/paused goal or replaying its turn."""
    fresh = current_pending(manager.session_id, token)
    if fresh is None:
        return None
    history = fresh.state.model_recovery
    pending = history["pending"]
    for row in history["attempts"]:
        if row["id"] == token:
            row.update(status=outcome, updated_at=time.time())
            if error_type:
                row["error_type"] = error_type
    if outcome == "committed":
        pending["status"] = outcome
    else:
        history.pop("pending", None)
    if outcome in {"failed", "cancelled", "superseded"}:
        fresh.state.status = "paused"
        fresh.state.paused_reason = f"Model handoff {outcome}; checkpoint preserved. Recommended: {pending['plan']}"
    fresh._save()
    manager._state = fresh.state
    return deepcopy(pending)


def qualify(manager, token):
    """Persist the successful tool-generation qualification before writing a target override."""
    fresh = current_pending(manager.session_id, token)
    if fresh is None:
        return None
    pending = fresh.state.model_recovery["pending"]
    proof = {**pending["target"], "at": time.time(), "probe": "tool_generation"}
    pending["qualification"] = proof
    for row in fresh.state.model_recovery["attempts"]:
        if row["id"] == token:
            row["qualification"] = proof
    fresh._save()
    manager._state = fresh.state
    return proof


def cancel_pending(state):
    if not state.model_recovery:
        return
    history = state.model_recovery
    token = (history.get("pending") or {}).get("id")
    for row in history.get("attempts", []):
        if row.get("id") == token:
            row["status"] = "cancelled"
    history.pop("pending", None)


def reset_episode(state):
    """Explicit user resume starts another episode but keeps the audit of previous attempts."""
    if not state.model_recovery:
        return
    history = state.model_recovery
    for row in history.get("attempts", []):
        if row.get("status") in {"reserved", "committed"}:
            row["status"] = "cancelled"
    history.pop("pending", None)
    history["epoch"] = history.get("epoch", 0) + 1
