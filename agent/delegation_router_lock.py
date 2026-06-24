"""Parent-session router/verifier lock for delegated work.

The lock is deliberately independent of prompt text.  Once the parent agent
successfully dispatches non-trivial delegated work, this module records a
session-scoped lock and exposes a pure allow/block decision that tool executors
must consult before running implementation tools.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from agent.tool_guardrails import IDEMPOTENT_TOOL_NAMES, MUTATING_TOOL_NAMES

LOCK_MODEL_CONFIG_KEY = "delegation_router_lock"
LOCK_VERSION = 1
STATUS_ACTIVE = "active"
STATUS_VERIFICATION_REQUIRED = "verification_required"

# Router/verifier mode is allow-list based.  These tools either route work,
# ask the user, inspect evidence, or monitor/cancel background activity.
_ALWAYS_ALLOWED_TOOLS = frozenset(
    {
        "delegate_task",
        "clarify",
        "read_terminal",
        "read_file",
        "search_files",
        "web_search",
        "web_extract",
        "session_search",
        "skills_list",
        "skill_view",
        "browser_snapshot",
        "browser_console",
        "browser_get_images",
        "vision_analyze",
    }
)

# ``process`` is allowed only for monitoring/cancellation.  Stdin writes can
# drive an already-running shell and therefore remain implementation authority.
_PROCESS_ALLOWED_ACTIONS = frozenset({"list", "poll", "log", "wait", "kill"})

# Tools that are clearly implementation authority even when they are not in the
# loop-guardrail mutating set (plugin-generated names, browser state changes,
# media generation, outbound posting, etc.).
_IMPLEMENTATION_TOOL_NAMES = frozenset(
    (MUTATING_TOOL_NAMES - {"delegate_task", "process"})
    | {
        "write_file",
        "patch",
        "terminal",
        "execute_code",
        "skill_manage",
        "memory",
        "todo",
        "cronjob",
        "image_generate",
        "image_gen",
        "video_gen",
        "video_generate",
        "tts",
        "send_message",
        "browser_navigate",
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_scroll",
        "browser_select",
        "browser_upload_file",
        "browser_evaluate",
        "mcp_filesystem_write_file",
        "mcp_filesystem_edit_file",
        "mcp_filesystem_create_directory",
        "mcp_filesystem_move_file",
    }
)

_READ_ONLY_NAME_RE = re.compile(
    r"(^|_)(read|search|list|get|show|describe|inspect|status|query|view|fetch|lookup)(_|$)"
)
_ASYNC_COMPLETE_RE = re.compile(
    r"\[ASYNC DELEGATION(?: BATCH)? COMPLETE\s+—\s+([^\]]+)\]"
)


@dataclass(frozen=True)
class RouterLockDecision:
    """Decision returned before a tool is allowed to execute."""

    allows_execution: bool
    reason: str = ""
    delegation_id: str = ""
    status: str = ""

    @property
    def blocks_execution(self) -> bool:
        return not self.allows_execution


def empty_state() -> dict[str, Any]:
    return {"version": LOCK_VERSION, "locks": []}


def normalize_state(value: Any) -> dict[str, Any]:
    """Return a canonical lock-state mapping from persisted or in-memory data."""

    if not isinstance(value, Mapping):
        return empty_state()
    raw_locks = value.get("locks")
    if not isinstance(raw_locks, list):
        raw_locks = []
    locks: list[dict[str, Any]] = []
    for raw in raw_locks:
        if not isinstance(raw, Mapping):
            continue
        status = str(raw.get("status") or STATUS_ACTIVE)
        if status not in {STATUS_ACTIVE, STATUS_VERIFICATION_REQUIRED}:
            continue
        delegation_id = str(raw.get("delegation_id") or "").strip()
        if not delegation_id:
            continue
        lock = dict(raw)
        lock["version"] = LOCK_VERSION
        lock["delegation_id"] = delegation_id
        lock["status"] = status
        locks.append(lock)
    return {"version": LOCK_VERSION, "locks": locks}


def load_state_from_model_config(model_config: Any) -> dict[str, Any]:
    if isinstance(model_config, str):
        try:
            model_config = json.loads(model_config) if model_config.strip() else {}
        except json.JSONDecodeError:
            model_config = {}
    if not isinstance(model_config, Mapping):
        return empty_state()
    return normalize_state(model_config.get(LOCK_MODEL_CONFIG_KEY))


def state_for_model_config(state: Any) -> dict[str, Any]:
    return normalize_state(state)


def restore_agent_state_from_session(agent: Any) -> None:
    """Restore persisted lock state for a freshly-created agent instance."""

    state = empty_state()
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if session_db is not None and session_id:
        try:
            row = session_db.get_session(session_id)
        except Exception:
            row = None
        if isinstance(row, Mapping):
            state = load_state_from_model_config(row.get("model_config"))
    setattr(agent, "_delegation_router_lock_state", state)
    try:
        init_cfg = getattr(agent, "_session_init_model_config", None)
        if isinstance(init_cfg, dict):
            init_cfg[LOCK_MODEL_CONFIG_KEY] = state_for_model_config(state)
    except Exception:
        pass


def _current_state(agent: Any) -> dict[str, Any]:
    state = normalize_state(getattr(agent, "_delegation_router_lock_state", None))
    setattr(agent, "_delegation_router_lock_state", state)
    return state


def persist_agent_state(agent: Any) -> None:
    """Persist lock state in the session model_config without clobbering peers."""

    state = _current_state(agent)
    try:
        init_cfg = getattr(agent, "_session_init_model_config", None)
        if isinstance(init_cfg, dict):
            init_cfg[LOCK_MODEL_CONFIG_KEY] = state_for_model_config(state)
    except Exception:
        pass

    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if session_db is None or not session_id:
        return

    model_config: dict[str, Any] = {}
    try:
        row = session_db.get_session(session_id)
        if isinstance(row, Mapping):
            raw = row.get("model_config")
            if isinstance(raw, str) and raw.strip():
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    model_config.update(parsed)
            elif isinstance(raw, Mapping):
                model_config.update(raw)
    except Exception:
        # Fall back to the init config; persistence failure should not unlock
        # the in-memory session.
        pass
    if not model_config:
        init_cfg = getattr(agent, "_session_init_model_config", None)
        if isinstance(init_cfg, Mapping):
            model_config.update(dict(init_cfg))
    model_config[LOCK_MODEL_CONFIG_KEY] = state_for_model_config(state)
    try:
        session_db.update_session_meta(session_id, json.dumps(model_config), None)
    except Exception:
        pass


def is_non_trivial_delegate_args(args: Any) -> bool:
    if not isinstance(args, Mapping):
        return False
    tasks = args.get("tasks")
    if isinstance(tasks, list):
        return any(isinstance(task, Mapping) and str(task.get("goal") or "").strip() for task in tasks)
    if isinstance(tasks, str) and tasks.strip():
        try:
            parsed = json.loads(tasks)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return any(isinstance(task, Mapping) and str(task.get("goal") or "").strip() for task in parsed)
    return bool(str(args.get("goal") or "").strip())


def delegate_success_payload(result: Any) -> tuple[bool, dict[str, Any]]:
    """Return (success, parsed_payload) for a delegate_task result."""

    payload: Any = result
    if isinstance(result, str):
        try:
            payload = json.loads(result) if result.strip() else {}
        except json.JSONDecodeError:
            return False, {}
    if not isinstance(payload, Mapping):
        return False, {}
    if payload.get("error"):
        return False, dict(payload)
    status = str(payload.get("status") or "").lower()
    if status in {"dispatched", "completed", "success"}:
        return True, dict(payload)
    if status in {"rejected", "error", "failed", "cancelled", "interrupted", "timeout"}:
        return False, dict(payload)
    results = payload.get("results")
    if isinstance(results, list) and results:
        for entry in results:
            if not isinstance(entry, Mapping):
                continue
            entry_status = str(entry.get("status") or "").lower()
            if entry_status in {"completed", "success"}:
                return True, dict(payload)
    return False, dict(payload)


def _goals_from_args(args: Mapping[str, Any]) -> list[str]:
    tasks = args.get("tasks")
    goals: list[str] = []
    if isinstance(tasks, str) and tasks.strip():
        try:
            parsed = json.loads(tasks)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            tasks = parsed
    if isinstance(tasks, list):
        for task in tasks:
            if isinstance(task, Mapping):
                goal = str(task.get("goal") or "").strip()
                if goal:
                    goals.append(goal)
    goal = str(args.get("goal") or "").strip()
    if goal:
        goals.insert(0, goal)
    return goals


def _scope_from_args(args: Mapping[str, Any]) -> dict[str, Any]:
    goals = _goals_from_args(args)
    context = str(args.get("context") or "").strip()
    digest_src = json.dumps({"goals": goals, "context": context}, sort_keys=True)
    digest = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()[:16]
    return {
        "digest": digest,
        "goals": goals,
        "context_preview": context[:500],
        # Scope matching is intentionally conservative in Task 1: when the
        # delegated scope cannot be safely reduced to paths, the parent loses
        # implementation authority for the delegated work as a whole.
        "match": "delegated_work",
    }


def _new_lock(args: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    delegation_id = str(payload.get("delegation_id") or "").strip()
    if not delegation_id:
        raw = json.dumps({"args": dict(args), "payload": dict(payload), "t": time.time()}, sort_keys=True, default=str)
        delegation_id = "deleg_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
    now = time.time()
    return {
        "version": LOCK_VERSION,
        "delegation_id": delegation_id,
        "status": STATUS_ACTIVE,
        "activated_at": now,
        "completed_at": None,
        "scope": _scope_from_args(args),
    }


def activate_after_delegate_result(agent: Any, args: Any, result: Any) -> bool:
    """Activate/persist a lock after a successful non-trivial delegation."""

    if not is_non_trivial_delegate_args(args):
        return False
    success, payload = delegate_success_payload(result)
    if not success:
        return False
    if not isinstance(args, Mapping):
        return False
    state = _current_state(agent)
    lock = _new_lock(args, payload)
    locks = [l for l in state["locks"] if l.get("delegation_id") != lock["delegation_id"]]
    locks.append(lock)
    state["locks"] = locks
    setattr(agent, "_delegation_router_lock_state", state)
    persist_agent_state(agent)
    return True


def active_locks(agent: Any) -> list[dict[str, Any]]:
    state = _current_state(agent)
    return [
        lock
        for lock in state.get("locks", [])
        if lock.get("status") in {STATUS_ACTIVE, STATUS_VERIFICATION_REQUIRED}
    ]


def mark_async_completion_from_text(agent: Any, text: Any) -> bool:
    if not isinstance(text, str):
        return False
    match = _ASYNC_COMPLETE_RE.search(text)
    if not match:
        return False
    delegation_id = match.group(1).strip()
    state = _current_state(agent)
    changed = False
    for lock in state.get("locks", []):
        if lock.get("delegation_id") == delegation_id:
            if lock.get("status") != STATUS_VERIFICATION_REQUIRED:
                lock["status"] = STATUS_VERIFICATION_REQUIRED
                lock["completed_at"] = time.time()
                changed = True
    if changed:
        setattr(agent, "_delegation_router_lock_state", state)
        persist_agent_state(agent)
    return changed


def mark_async_completion_event(agent: Any, event: Mapping[str, Any]) -> bool:
    delegation_id = str(event.get("delegation_id") or "").strip()
    if not delegation_id:
        return False
    state = _current_state(agent)
    changed = False
    for lock in state.get("locks", []):
        if lock.get("delegation_id") == delegation_id:
            if lock.get("status") != STATUS_VERIFICATION_REQUIRED:
                lock["status"] = STATUS_VERIFICATION_REQUIRED
                lock["completed_at"] = event.get("completed_at") or time.time()
                changed = True
    if changed:
        setattr(agent, "_delegation_router_lock_state", state)
        persist_agent_state(agent)
    return changed


def prepare_same_turn_lock(agent: Any, tool_calls: Iterable[Any]) -> None:
    """Record transient lock state for a mixed delegate+implementation batch."""

    scopes: list[dict[str, Any]] = []
    for tc in tool_calls or []:
        try:
            name = tc.function.name
            raw_args = tc.function.arguments
        except Exception:
            continue
        if name != "delegate_task":
            continue
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {}
        if isinstance(args, Mapping) and is_non_trivial_delegate_args(args):
            scopes.append(_scope_from_args(args))
    setattr(agent, "_delegation_router_lock_same_turn_scopes", scopes)


def clear_same_turn_lock(agent: Any) -> None:
    setattr(agent, "_delegation_router_lock_same_turn_scopes", [])


def _has_same_turn_lock(agent: Any) -> bool:
    scopes = getattr(agent, "_delegation_router_lock_same_turn_scopes", None)
    return isinstance(scopes, list) and bool(scopes)


def is_implementation_tool(tool_name: str, args: Optional[Mapping[str, Any]] = None) -> bool:
    """Return True when a tool would implement/modify rather than verify/route."""

    name = str(tool_name or "").strip()
    if not name:
        return True
    if name in _ALWAYS_ALLOWED_TOOLS or name in IDEMPOTENT_TOOL_NAMES:
        return False
    if name == "process":
        action = str((args or {}).get("action") or "").strip().lower()
        return action not in _PROCESS_ALLOWED_ACTIONS
    if name in _IMPLEMENTATION_TOOL_NAMES:
        return True
    if name.startswith("browser_"):
        return name not in {"browser_snapshot", "browser_console", "browser_get_images"}
    if name.startswith("mcp_filesystem_"):
        return not any(token in name for token in ("read", "list", "search", "get", "tree"))
    # Unknown tools are allowed only when their name advertises a read-only
    # verification operation.  Router/verifier mode is intentionally fail-closed.
    return _READ_ONLY_NAME_RE.search(name) is None


def should_block_tool(agent: Any, tool_name: str, args: Optional[Mapping[str, Any]] = None) -> RouterLockDecision:
    if not is_implementation_tool(tool_name, args):
        return RouterLockDecision(True)
    locks = active_locks(agent)
    if locks:
        lock = locks[-1]
        return RouterLockDecision(
            False,
            reason=(
                "Parent session is locked in router/verifier mode for delegated work. "
                "Route or re-dispatch, clarify, monitor/cancel, or verify returned evidence; "
                "do not execute implementation tools in the parent for this delegated scope."
            ),
            delegation_id=str(lock.get("delegation_id") or ""),
            status=str(lock.get("status") or STATUS_ACTIVE),
        )
    if _has_same_turn_lock(agent):
        return RouterLockDecision(
            False,
            reason=(
                "This same assistant tool batch delegates non-trivial work, so the parent "
                "cannot also execute implementation tools for that delegated scope."
            ),
            delegation_id="same_turn_delegate_task",
            status=STATUS_ACTIVE,
        )
    return RouterLockDecision(True)


def synthetic_block_result(decision: RouterLockDecision, tool_name: str) -> str:
    return json.dumps(
        {
            "error": decision.reason,
            "status": "blocked",
            "code": "delegation_router_lock",
            "tool": tool_name,
            "delegation_id": decision.delegation_id,
            "lock_status": decision.status,
        },
        ensure_ascii=False,
    )
