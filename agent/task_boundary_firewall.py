"""Phase 5 Task Boundary Firewall provider-payload enforcement.

This adapter is deliberately narrow. It classifies the current turn against the
Phase 4 task-registry snapshot and returns a provider-payload filtering decision.
It does not call models, providers, retrievers, session_search, memory prefetch,
or compact fallback code. It does not mutate transcripts or session DB rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TaskBoundaryFirewallDecision:
    action: str
    reason: str
    allowed_task_ids: tuple[str, ...] = ()
    excluded_task_ids: tuple[str, ...] = ()
    linked_task_ids: tuple[str, ...] = ()
    filtered_registry_snapshot: Mapping[str, Any] | None = None
    safe_linked_task_pointers: tuple[Mapping[str, str], ...] = ()
    hold_response: str | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)


_SENSITIVE_MARKERS = (
    "token",
    "password",
    "secret",
    "credential",
    "credentials",
    "api key",
    "private key",
    "connection string",
)

_SAFE_TASK_FIELDS = (
    "task_id",
    "status",
    "session_id",
    "latest_turn_id",
    "task_state_path",
    "workspec_path",
    "current_pin_path",
)

_EXPLICIT_CONTINUE_WORDS = (
    "continue",
    "continuation",
    "resume",
    "reopen",
    "link",
    "linked",
    "same task",
    "previous task",
    "prior task",
)

_AMBIGUOUS_REFERENCE_PHRASES = (
    "previous one",
    "the previous",
    "that one",
    "same one",
    "above task",
    "earlier task",
    "old task",
    "last task",
)


def enforce_task_boundary_firewall(
    *,
    policy: Any = None,
    registry_snapshot: Any = None,
    current_task_id: str | None = None,
    user_message: Any = None,
    explicit_continuation_refs: Sequence[Any] | None = None,
    registry_decision: Any = None,
    **_kwargs: Any,
) -> TaskBoundaryFirewallDecision:
    """Return a Phase 5 provider-payload firewall decision.

    Disabled mode is complete pass-through. Enabled mode narrows registry facts
    to the current task plus explicitly continued/linked task ids only. Ambiguous
    prior-task references HOLD before provider/model calls.
    """

    if not _firewall_enabled(policy):
        return TaskBoundaryFirewallDecision(
            action="use_original",
            reason="task_boundary_firewall_disabled",
            evidence={"phase": "phase5_provider_payload_firewall"},
        )

    snapshot = _safe_snapshot(registry_snapshot)
    current = _safe_text(current_task_id or snapshot.get("active_task_id"))
    explicit_refs = _normalize_task_ids(explicit_continuation_refs)
    explicit_refs = explicit_refs or _explicit_refs_from_registry_decision(registry_decision)
    user_text = _content_to_text(user_message)

    tasks = snapshot.get("tasks") if isinstance(snapshot.get("tasks"), Mapping) else {}
    closed_task_ids = tuple(
        _safe_text(task_id)
        for task_id, record in tasks.items()
        if isinstance(record, Mapping) and _safe_text(record.get("status")).lower() == "closed"
    )

    if _should_hold_for_ambiguous_reference(user_text, closed_task_ids, current, explicit_refs):
        return TaskBoundaryFirewallDecision(
            action="hold",
            reason="ambiguous_task_boundary_reference",
            allowed_task_ids=(),
            excluded_task_ids=tuple(task_id for task_id in closed_task_ids if task_id),
            linked_task_ids=(),
            filtered_registry_snapshot=_filter_snapshot(snapshot, allowed_task_ids=()),
            hold_response=_safe_hold_response("ambiguous_task_boundary_reference"),
            evidence={"phase": "phase5_provider_payload_firewall"},
        )

    allowed = _ordered_unique([current, *explicit_refs])
    filtered = _filter_snapshot(snapshot, allowed_task_ids=allowed)
    excluded = tuple(
        task_id for task_id in tasks.keys()
        if _safe_text(task_id) and _safe_text(task_id) not in set(allowed)
    )

    linked_pointers = _safe_linked_pointers(snapshot, explicit_refs)
    if explicit_refs:
        action = "allow_continue_task" if current in explicit_refs else "allow_linked_task_context"
        reason = "explicit_continuation_reference"
    else:
        action = "allow_new_task"
        reason = "default_new_task_without_clear_continuation"

    return TaskBoundaryFirewallDecision(
        action=action,
        reason=reason,
        allowed_task_ids=tuple(allowed),
        excluded_task_ids=excluded,
        linked_task_ids=tuple(explicit_refs),
        filtered_registry_snapshot=filtered,
        safe_linked_task_pointers=linked_pointers,
        evidence={"phase": "phase5_provider_payload_firewall"},
    )


def _firewall_enabled(policy: Any) -> bool:
    raw = _raw_context_health(policy)
    if not isinstance(raw, Mapping):
        return False
    if not raw.get("enabled") or not raw.get("runtime_behavior_enabled"):
        return False
    tbf = raw.get("task_boundary_firewall")
    if isinstance(tbf, Mapping):
        return bool(tbf.get("enabled"))
    return False


def _raw_context_health(policy: Any) -> Any:
    if isinstance(policy, Mapping):
        if "context_health" in policy and isinstance(policy.get("context_health"), Mapping):
            return policy.get("context_health")
        return policy
    return None


def _safe_snapshot(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping):
        return {"schema": "context_health_task_registry_v1", "active_task_id": None, "tasks": {}}
    tasks_raw = snapshot.get("tasks")
    tasks: dict[str, dict[str, Any]] = {}
    if isinstance(tasks_raw, Mapping):
        for task_id, record in tasks_raw.items():
            task_id_text = _safe_text(task_id)
            if not task_id_text or not isinstance(record, Mapping):
                continue
            safe_record: dict[str, Any] = {}
            for key in _SAFE_TASK_FIELDS:
                value = _safe_text(record.get(key))
                if value:
                    safe_record[key] = value
            linked = record.get("linked_task_ids")
            if isinstance(linked, Sequence) and not isinstance(linked, (str, bytes)):
                safe_links = [_safe_text(item) for item in linked]
                safe_record["linked_task_ids"] = [item for item in safe_links if item]
            tasks[task_id_text] = safe_record
    return {
        "schema": "context_health_task_registry_v1",
        "active_task_id": _safe_text(snapshot.get("active_task_id")) or None,
        "tasks": tasks,
    }


def _filter_snapshot(snapshot: Mapping[str, Any], *, allowed_task_ids: Sequence[str]) -> dict[str, Any]:
    allowed = set(_normalize_task_ids(allowed_task_ids))
    tasks_raw = snapshot.get("tasks")
    tasks: dict[str, Any] = {}
    if isinstance(tasks_raw, Mapping):
        for task_id, record in tasks_raw.items():
            task_id_text = _safe_text(task_id)
            if task_id_text in allowed and isinstance(record, Mapping):
                tasks[task_id_text] = dict(record)
    active = _safe_text(snapshot.get("active_task_id"))
    return {
        "schema": "context_health_task_registry_v1",
        "active_task_id": active if active in allowed else None,
        "tasks": tasks,
    }


def _explicit_refs_from_registry_decision(registry_decision: Any) -> tuple[str, ...]:
    if registry_decision is None:
        return ()
    action = _safe_text(getattr(registry_decision, "action", "")).lower()
    if action not in {"continue_task", "reopen_task", "link_task", "allow_continue_task", "allow_linked_task_context"}:
        return ()
    refs = [
        getattr(registry_decision, "effective_task_id", None),
        getattr(registry_decision, "linked_task_id", None),
    ]
    linked = getattr(registry_decision, "linked_task_ids", None)
    if isinstance(linked, Sequence) and not isinstance(linked, (str, bytes)):
        refs.extend(linked)
    return _normalize_task_ids(refs)


def _normalize_task_ids(values: Sequence[Any] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    return tuple(_ordered_unique(_safe_text(value) for value in values if _safe_text(value)))


def _ordered_unique(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = _safe_text(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _safe_linked_pointers(snapshot: Mapping[str, Any], refs: Sequence[str]) -> tuple[Mapping[str, str], ...]:
    tasks = snapshot.get("tasks")
    if not isinstance(tasks, Mapping):
        return ()
    pointers: list[Mapping[str, str]] = []
    for task_id in refs:
        record = tasks.get(task_id)
        if not isinstance(record, Mapping):
            continue
        item: dict[str, str] = {"task_id": task_id}
        for key in ("task_state_path", "workspec_path", "current_pin_path"):
            value = _safe_text(record.get(key))
            if value:
                item[key] = value
        pointers.append(item)
    return tuple(pointers)


def _should_hold_for_ambiguous_reference(
    user_text: str,
    closed_task_ids: Sequence[str],
    current_task_id: str,
    explicit_refs: Sequence[str],
) -> bool:
    lowered = user_text.lower()
    if explicit_refs:
        return False
    if any(phrase in lowered for phrase in _AMBIGUOUS_REFERENCE_PHRASES):
        return True
    closed = [task_id for task_id in closed_task_ids if task_id and task_id != current_task_id]
    if not closed:
        return False
    mentions_closed = any(task_id.lower() in lowered for task_id in closed)
    if not mentions_closed:
        return False
    has_clear_continue = any(word in lowered for word in _EXPLICIT_CONTINUE_WORDS)
    return not has_clear_continue


def _safe_hold_response(reason: str) -> str:
    return (
        "Context Health HOLD: task boundary relation is ambiguous, so Hermes "
        "did not send prior-task context to the provider. Reason: "
        f"{_safe_text(reason) or 'ambiguous_task_boundary_reference'}"
    )


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "text" in item:
                    parts.append(str(item.get("text", "")))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _safe_text(value: Any) -> str:
    text = str(value or "")
    lowered = text.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return "[REDACTED]"
    return text[:240]
