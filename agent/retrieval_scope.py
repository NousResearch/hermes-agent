"""Phase 6 retrieval-scope enforcement helpers.

This module is intentionally deterministic and side-effect free: it does not
call models, providers, retrievers, or mutate the session DB/transcripts. It
only derives a small allow/exclude scope from policy + task registry state and
returns decisions for callers that are about to retrieve context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


_SAFE_HOLD_RESPONSE = (
    "Context Health HOLD: retrieval scope could not be verified safely; "
    "no unscoped retrieval was executed."
)


@dataclass
class RetrievalScopeDecision:
    action: str
    reason: str = ""
    allowed_task_ids: list[str] = field(default_factory=list)
    excluded_task_ids: list[str] = field(default_factory=list)
    linked_task_ids: list[str] = field(default_factory=list)
    allowed_session_ids: list[str] = field(default_factory=list)
    excluded_session_ids: list[str] = field(default_factory=list)
    rewritten_args: dict[str, Any] | None = None
    hold_response: str | None = None
    fail_closed: bool = False


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [str(value)] if str(value) else []
    try:
        return [str(v) for v in value if v is not None and str(v)]
    except TypeError:
        return [str(value)] if str(value) else []


def retrieval_scope_enabled(policy: Any) -> bool:
    cfg = _as_dict(policy)
    if cfg.get("enabled") is False or cfg.get("runtime_behavior_enabled") is False:
        return False
    scope_cfg = _as_dict(cfg.get("retrieval_scope"))
    return bool(scope_cfg.get("enabled", False))


def _registry_tasks(registry_snapshot: Any) -> dict[str, Any]:
    snap = _as_dict(registry_snapshot)
    tasks = snap.get("tasks")
    return tasks if isinstance(tasks, dict) else {}


def _task_session_id(task_record: Any) -> str:
    value = _get(task_record, "session_id", "")
    return str(value or "")


def _derive_scope(
    *,
    current_task_id: str = "",
    registry_snapshot: Any = None,
    registry_decision: Any = None,
    explicit_continuation_refs: Any = None,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    tasks = _registry_tasks(registry_snapshot)
    linked_task_ids = _list(explicit_continuation_refs)
    linked = _get(registry_decision, "linked_task_id", None)
    if linked:
        linked_task_ids.append(str(linked))
    linked_task_ids = list(dict.fromkeys(linked_task_ids))

    allowed_task_ids: list[str] = []
    if current_task_id:
        allowed_task_ids.append(str(current_task_id))
    allowed_task_ids.extend(linked_task_ids)
    allowed_task_ids = list(dict.fromkeys(allowed_task_ids))

    excluded_task_ids: list[str] = []
    allowed_set = set(allowed_task_ids)
    for task_id, record in tasks.items():
        status = str(_get(record, "status", "")).lower()
        if status == "closed" and str(task_id) not in allowed_set:
            excluded_task_ids.append(str(task_id))

    allowed_session_ids: list[str] = []
    excluded_session_ids: list[str] = []
    for task_id in allowed_task_ids:
        sid = _task_session_id(tasks.get(task_id, {}))
        if sid:
            allowed_session_ids.append(sid)
    for task_id in excluded_task_ids:
        sid = _task_session_id(tasks.get(task_id, {}))
        if sid:
            excluded_session_ids.append(sid)

    return (
        list(dict.fromkeys(allowed_task_ids)),
        list(dict.fromkeys(excluded_task_ids)),
        list(dict.fromkeys(linked_task_ids)),
        list(dict.fromkeys(allowed_session_ids)),
        list(dict.fromkeys(excluded_session_ids)),
    )


def build_retrieval_scope(
    *,
    policy: Any,
    current_task_id: str = "",
    registry_snapshot: Any = None,
    registry_decision: Any = None,
    explicit_continuation_refs: Any = None,
) -> dict[str, Any]:
    enabled = retrieval_scope_enabled(policy)
    if not enabled:
        return {"enabled": False}
    allowed, excluded, linked, allowed_sessions, excluded_sessions = _derive_scope(
        current_task_id=current_task_id,
        registry_snapshot=registry_snapshot,
        registry_decision=registry_decision,
        explicit_continuation_refs=explicit_continuation_refs,
    )
    return {
        "enabled": True,
        "mode": "explicit_link" if linked else "current_task_only",
        "current_task_id": str(current_task_id or ""),
        "allowed_task_ids": allowed,
        "excluded_task_ids": excluded,
        "linked_task_ids": linked,
        "allowed_session_ids": allowed_sessions,
        "excluded_session_ids": excluded_sessions,
    }


def decision_from_scope(scope: Mapping[str, Any], *, action: str, reason: str = "") -> RetrievalScopeDecision:
    return RetrievalScopeDecision(
        action=action,
        reason=reason,
        allowed_task_ids=_list(scope.get("allowed_task_ids")),
        excluded_task_ids=_list(scope.get("excluded_task_ids")),
        linked_task_ids=_list(scope.get("linked_task_ids")),
        allowed_session_ids=_list(scope.get("allowed_session_ids")),
        excluded_session_ids=_list(scope.get("excluded_session_ids")),
    )


def enforce_retrieval_scope(
    *,
    policy: Any,
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
    current_task_id: str = "",
    registry_snapshot: Any = None,
    registry_decision: Any = None,
    explicit_continuation_refs: Any = None,
    user_message: str = "",
) -> RetrievalScopeDecision:
    if not retrieval_scope_enabled(policy):
        return RetrievalScopeDecision(action="use_original")

    scope = build_retrieval_scope(
        policy=policy,
        current_task_id=current_task_id,
        registry_snapshot=registry_snapshot,
        registry_decision=registry_decision,
        explicit_continuation_refs=explicit_continuation_refs,
    )
    relation = str(_get(registry_decision, "action", "") or "").lower()
    text = f"{user_message or ''} {(_as_dict(tool_args).get('query') or '')}".lower()
    ambiguous = relation == "hold" or "previous one" in text or "previous" == text.strip()
    if ambiguous:
        d = decision_from_scope(scope, action="hold", reason="ambiguous_prior_task_retrieval")
        d.hold_response = _SAFE_HOLD_RESPONSE
        return d

    session_id = str((_as_dict(tool_args).get("session_id") or "")).strip()
    excluded_sessions = set(_list(scope.get("excluded_session_ids")))
    allowed_sessions = set(_list(scope.get("allowed_session_ids")))
    if session_id and session_id in excluded_sessions and session_id not in allowed_sessions:
        d = decision_from_scope(scope, action="block", reason="closed_unlinked_session_excluded")
        d.hold_response = _SAFE_HOLD_RESPONSE
        return d

    action = "allow" if _list(scope.get("linked_task_ids")) else "rewrite_args"
    d = decision_from_scope(scope, action=action, reason="retrieval_scope_applied")
    d.rewritten_args = {}
    return d


def safe_retrieval_scope_failure(
    *,
    policy: Any,
    reason: str = "",
    current_task_id: str = "",
    registry_snapshot: Any = None,
    registry_decision: Any = None,
) -> RetrievalScopeDecision:
    scope = build_retrieval_scope(
        policy=policy,
        current_task_id=current_task_id,
        registry_snapshot=registry_snapshot,
        registry_decision=registry_decision,
    ) if retrieval_scope_enabled(policy) else {"enabled": False}
    d = decision_from_scope(scope, action="hold", reason="retrieval_scope_failure")
    d.fail_closed = True
    d.hold_response = _SAFE_HOLD_RESPONSE
    return d


def filter_text_by_scope(text: str, retrieval_scope: Mapping[str, Any] | None) -> str:
    """Conservative provider-payload quarantine for untrusted retrieval text.

    Phase 6 prevents/ scopes retrieval before execution where possible. For
    already-returned memory/plugin context we use a safe first implementation:
    when scope is enabled and there are excluded tasks/sessions, quarantine the
    untrusted retrieval text rather than trying to redact arbitrary raw content.
    """
    scope = _as_dict(retrieval_scope)
    if not scope.get("enabled"):
        return text or ""
    if _list(scope.get("excluded_task_ids")) or _list(scope.get("excluded_session_ids")):
        return ""
    return text or ""
