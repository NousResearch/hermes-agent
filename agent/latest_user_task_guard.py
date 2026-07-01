"""Latest-user-request authority guard for compacted or handed-off tasks."""

from __future__ import annotations

import re
from typing import Any, Mapping

from agent.runtime_evidence import current_runtime_evidence, update_runtime_evidence

_TASK_SENSITIVE_TERMINAL_PATTERNS = (
    re.compile(r"\bgh\s+(?:pr|run|workflow)\b", re.IGNORECASE),
    re.compile(r"\bgit\s+(?:push|status|diff|checkout|switch|merge|rebase|commit)\b", re.IGNORECASE),
    re.compile(r"\b(?:apply_patch|patch)\b", re.IGNORECASE),
)

_TASK_SENSITIVE_WRITE_HINTS = (
    re.compile(r">>", re.IGNORECASE),
    re.compile(r">", re.IGNORECASE),
    re.compile(r"\btee\b", re.IGNORECASE),
    re.compile(r"\bsed\s+-i\b", re.IGNORECASE),
    re.compile(r"\bperl\s+-pi\b", re.IGNORECASE),
    re.compile(r"\bcp\b", re.IGNORECASE),
    re.compile(r"\bmv\b", re.IGNORECASE),
)


def normalize_user_request(text: Any) -> str:
    return " ".join(str(text or "").split()).strip().lower()


def should_guard_latest_user_request(function_name: str, function_args: Mapping[str, Any] | None) -> bool:
    """Return True when the tool call is sensitive to stale task carryover."""
    args = function_args if isinstance(function_args, Mapping) else {}
    if function_name in {"write_file", "patch", "skill_manage"}:
        return True
    if function_name == "execute_code":
        code = str(args.get("code") or "").strip()
        return bool(code and _contains_write_intent(code))
    if function_name == "terminal":
        command = str(args.get("command") or args.get("cmd") or "").strip()
        return bool(command and _contains_task_sensitive_terminal_intent(command))
    return False


def enforce_latest_user_request_authority(
    *,
    function_name: str,
    function_args: Mapping[str, Any] | None,
    user_task: str | None,
) -> str | None:
    """Block stale task continuation after compaction/handoff if it diverges."""
    if not should_guard_latest_user_request(function_name, function_args):
        return None

    evidence = current_runtime_evidence() or {}
    latest_request = str(evidence.get("latest_user_request") or "").strip()
    if not latest_request:
        return None

    active_task = str(user_task or "").strip()
    if not active_task:
        return None

    latest_norm = normalize_user_request(latest_request)
    active_norm = normalize_user_request(active_task)
    if not latest_norm or not active_norm or latest_norm == active_norm:
        return None

    compaction_active = bool(
        evidence.get("compaction_summary_active")
        or evidence.get("compaction_handoff_active")
        or evidence.get("context_compacted")
    )
    if not compaction_active:
        update_runtime_evidence(active_user_task=active_task)
        return None

    update_runtime_evidence(
        active_user_task=active_task,
        stale_user_task_attempt=active_task,
    )
    return (
        "Blocked stale task continuation after compaction.\n"
        f"Latest real user request: {latest_request}\n"
        f"Stale/active task attempted: {active_task}\n"
        "Realign to the latest user request before continuing."
    )


def record_latest_user_request_from_dispatch(user_task: str | None) -> None:
    """Preserve the authoritative latest request while noting the active task."""
    active_task = str(user_task or "").strip()
    if not active_task:
        return
    evidence = current_runtime_evidence() or {}
    latest_request = str(evidence.get("latest_user_request") or "").strip()
    if latest_request:
        if normalize_user_request(latest_request) != normalize_user_request(active_task):
            update_runtime_evidence(active_user_task=active_task)
            return
        update_runtime_evidence(latest_user_request=active_task, active_user_task=active_task)
        return
    update_runtime_evidence(latest_user_request=active_task, active_user_task=active_task)


def _contains_task_sensitive_terminal_intent(command: str) -> bool:
    return any(pattern.search(command) for pattern in _TASK_SENSITIVE_TERMINAL_PATTERNS)


def _contains_write_intent(text: str) -> bool:
    return any(pattern.search(text) for pattern in _TASK_SENSITIVE_WRITE_HINTS)
