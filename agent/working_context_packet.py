"""Phase 3 Working Context Packet provider-payload enforcement.

This module is deliberately narrow: it constrains provider-visible
``api_messages`` only. It does not mutate persisted transcript messages, does
not implement active/closed task registry authority, and does not perform Task
Boundary Firewall or retrieval-scope enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class WorkingContextPacketDecision:
    action: str
    api_messages: list[dict[str, Any]] | None
    reason: str
    active_task_id: str | None = None
    included_sources: tuple[str, ...] = ()
    excluded_sources: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)
    hold_response: str | None = None


class WorkingContextPacketHold(Exception):
    """Raised by callers that choose exception-style WCP HOLD handling."""

    def __init__(self, reason: str, user_response: str):
        super().__init__(reason)
        self.reason = reason
        self.user_response = user_response


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


def enforce_working_context_packet(
    *,
    policy: Any = None,
    agent: Any = None,
    api_messages: Sequence[Mapping[str, Any]],
    messages: Sequence[Mapping[str, Any]] | None = None,
    original_user_message: Any = None,
    current_turn_user_idx: int | None = None,
    effective_task_id: str | None = None,
    turn_id: str | None = None,
    session_id: str | None = None,
    context_health_intake: Any = None,
    **_kwargs: Any,
) -> WorkingContextPacketDecision:
    """Return a provider-payload decision for Phase 3 WCP enforcement.

    Disabled mode is a pass-through. Enabled mode emits a default
    system-plus-one-user-message WCP, using only current-turn/safe intake
    pointers and never broad historical transcript content. If the current WCP
    source is unsafe, return a safe HOLD decision instead of falling back to full
    history.
    """

    original_api_messages = [dict(m) for m in api_messages]
    if not _wcp_enabled(policy, agent):
        return WorkingContextPacketDecision(
            action="use_original",
            api_messages=original_api_messages,
            reason="working_context_packet_disabled",
            evidence={"phase": "phase3_provider_payload"},
        )

    system_message = _first_system_message(original_api_messages)
    current_user_content = _current_user_content(
        original_api_messages,
        messages=messages,
        current_turn_user_idx=current_turn_user_idx,
        original_user_message=original_user_message,
    )

    intake_map = _mapping_from_obj(context_health_intake)
    registry_snapshot = _safe_registry_snapshot(getattr(agent, "_context_health_task_registry_snapshot", None))
    try:
        from agent.task_boundary_firewall import enforce_task_boundary_firewall
        firewall_decision = enforce_task_boundary_firewall(
            policy=policy if policy is not None else getattr(agent, "context_health", None),
            registry_snapshot=registry_snapshot,
            current_task_id=effective_task_id,
            user_message=current_user_content,
            registry_decision=getattr(agent, "_context_health_task_registry_decision", None),
        )
    except Exception:
        if _task_boundary_firewall_enabled(policy, agent):
            reason = "task_boundary_firewall_failure"
            return WorkingContextPacketDecision(
                action="hold",
                api_messages=None,
                reason=reason,
                active_task_id=effective_task_id,
                included_sources=(),
                excluded_sources=("full_conversation_history", "unfiltered_task_registry_snapshot"),
                evidence={
                    "phase": "phase5_provider_payload_firewall",
                    "session_id": session_id or "",
                    "turn_id": turn_id or "",
                    "fail_closed": True,
                },
                hold_response=_safe_task_boundary_firewall_failure_response(),
            )
        firewall_decision = None
    if firewall_decision is not None and getattr(firewall_decision, "action", None) == "hold":
        reason = str(getattr(firewall_decision, "reason", None) or "task_boundary_firewall_hold")
        return WorkingContextPacketDecision(
            action="hold",
            api_messages=None,
            reason=reason,
            active_task_id=effective_task_id,
            included_sources=(),
            excluded_sources=("full_conversation_history", "quarantined_closed_task_context"),
            evidence={
                "phase": "phase5_provider_payload_firewall",
                "session_id": session_id or "",
                "turn_id": turn_id or "",
            },
            hold_response=getattr(firewall_decision, "hold_response", None) or _safe_hold_response(reason),
        )
    filtered_registry = getattr(firewall_decision, "filtered_registry_snapshot", None) if firewall_decision is not None else None
    if isinstance(filtered_registry, Mapping):
        registry_snapshot = _safe_registry_snapshot(filtered_registry)
    if _intake_explicitly_unsafe(intake_map) or _current_content_unsafe(current_user_content):
        reason = str(intake_map.get("reason") or "working_context_packet_unsafe")
        return WorkingContextPacketDecision(
            action="hold",
            api_messages=None,
            reason=reason,
            active_task_id=effective_task_id,
            included_sources=(),
            excluded_sources=("full_conversation_history",),
            evidence={
                "phase": "phase3_provider_payload",
                "session_id": session_id or "",
                "turn_id": turn_id or "",
                "unsafe_current_content": True,
            },
            hold_response=_safe_hold_response(reason),
        )

    wcp_text = _build_wcp_text(
        current_user_content=current_user_content,
        effective_task_id=effective_task_id,
        turn_id=turn_id,
        session_id=session_id,
        intake=intake_map,
        registry_snapshot=registry_snapshot,
    )
    wcp_messages: list[dict[str, Any]] = []
    if system_message is not None:
        wcp_messages.append(system_message)
    wcp_messages.append({"role": "user", "content": wcp_text})

    return WorkingContextPacketDecision(
        action="replace_api_messages",
        api_messages=wcp_messages,
        reason="working_context_packet_applied",
        active_task_id=effective_task_id,
        included_sources=("system_prompt", "current_user_intent", "phase2_intake_pointer"),
        excluded_sources=("full_conversation_history", "unrelated_prior_tool_chains"),
        evidence={
            "phase": "phase3_provider_payload",
            "original_api_message_count": len(original_api_messages),
            "wcp_message_count": len(wcp_messages),
            "session_id": session_id or "",
            "turn_id": turn_id or "",
        },
    )


def _wcp_enabled(policy: Any, agent: Any) -> bool:
    raw = _raw_context_health(policy, agent)
    if isinstance(raw, Mapping):
        if not raw.get("enabled"):
            return False
        wcp = raw.get("working_context_packet")
        if isinstance(wcp, Mapping):
            return bool(raw.get("runtime_behavior_enabled") and wcp.get("enabled"))
        # Adapter unit tests may pass a minimal {'enabled': True} contract.
        return True

    # Fallback for tests or callers that provide the Phase 1 typed policy only.
    return bool(
        getattr(policy, "enabled", False)
        and getattr(policy, "runtime_behavior_enabled", False)
    )


def _raw_context_health(policy: Any, agent: Any) -> Any:
    if isinstance(policy, Mapping):
        if "context_health" in policy and isinstance(policy.get("context_health"), Mapping):
            return policy.get("context_health")
        return policy
    if agent is not None:
        raw = getattr(agent, "context_health", None)
        if isinstance(raw, Mapping):
            return raw
        cfg = getattr(agent, "config", None)
        if isinstance(cfg, Mapping):
            raw = cfg.get("context_health")
            if isinstance(raw, Mapping):
                return raw
    return None


def _task_boundary_firewall_enabled(policy: Any, agent: Any) -> bool:
    raw = _raw_context_health(policy, agent)
    if not isinstance(raw, Mapping):
        return False
    tbf = raw.get("task_boundary_firewall")
    return bool(
        raw.get("enabled")
        and raw.get("runtime_behavior_enabled")
        and isinstance(tbf, Mapping)
        and tbf.get("enabled")
    )


def _safe_task_boundary_firewall_failure_response() -> str:
    return (
        "Context Health HOLD: Task Boundary Firewall failed closed before "
        "provider call. Hermes did not send prior-task context to the provider."
    )


def _first_system_message(api_messages: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for msg in api_messages:
        if msg.get("role") == "system":
            return dict(msg)
    return None


def _current_user_content(
    api_messages: Sequence[Mapping[str, Any]],
    *,
    messages: Sequence[Mapping[str, Any]] | None,
    current_turn_user_idx: int | None,
    original_user_message: Any,
) -> str:
    if messages is not None and current_turn_user_idx is not None:
        try:
            msg = messages[current_turn_user_idx]
            if isinstance(msg, Mapping) and msg.get("role") == "user":
                return _content_to_text(msg.get("content", ""))
        except Exception:
            pass
    if original_user_message is not None:
        return _content_to_text(original_user_message)
    for msg in reversed(api_messages):
        if msg.get("role") == "user":
            return _content_to_text(msg.get("content", ""))
    return ""


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


def _mapping_from_obj(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, Mapping):
        return dict(obj)
    data: dict[str, Any] = {}
    for key in (
        "action",
        "reason",
        "packet_dir",
        "summary_path",
        "intake_path",
        "task_state_path",
        "replaced",
        "safe",
    ):
        if hasattr(obj, key):
            data[key] = getattr(obj, key)
    return data


def _intake_explicitly_unsafe(intake: Mapping[str, Any]) -> bool:
    if not intake:
        return False
    if intake.get("safe") is False:
        return True
    action = str(intake.get("action") or "").lower()
    return action in {"hold", "unsafe", "block"}


def _current_content_unsafe(content: str) -> bool:
    lowered = content.lower()
    return any(marker in lowered for marker in _SENSITIVE_MARKERS)


def _safe_hold_response(reason: str) -> str:
    return (
        "Context Health HOLD: Working Context Packet provider payload could not "
        "be built safely, so Hermes did not fall back to full conversation "
        f"history. Reason: {_safe_reason(reason)}"
    )


def _safe_reason(reason: str) -> str:
    text = str(reason or "unsafe_working_context_packet")
    for marker in _SENSITIVE_MARKERS:
        text = text.replace(marker, "[redacted-marker]")
        text = text.replace(marker.upper(), "[REDACTED-MARKER]")
    return text[:160]


def _safe_registry_snapshot(snapshot: Any) -> dict[str, Any]:
    if not isinstance(snapshot, Mapping):
        return {}
    tasks_raw = snapshot.get("tasks")
    tasks: dict[str, Any] = {}
    if isinstance(tasks_raw, Mapping):
        for task_id, record in tasks_raw.items():
            if not isinstance(record, Mapping):
                continue
            safe: dict[str, Any] = {}
            for key in (
                "task_id",
                "status",
                "session_id",
                "latest_turn_id",
                "task_state_path",
                "workspec_path",
                "current_pin_path",
            ):
                value = _safe_metadata_text(record.get(key))
                if value:
                    safe[key] = value
            linked = record.get("linked_task_ids")
            if isinstance(linked, Sequence) and not isinstance(linked, (str, bytes)):
                safe_links = [_safe_metadata_text(item) for item in linked]
                safe["linked_task_ids"] = [item for item in safe_links if item]
            tasks[_safe_metadata_text(task_id)] = safe
    active = _safe_metadata_text(snapshot.get("active_task_id"))
    return {
        "schema": "context_health_task_registry_v1",
        "active_task_id": active or None,
        "tasks": tasks,
    }


def _registry_lines(snapshot: Mapping[str, Any] | None) -> list[str]:
    if not snapshot:
        return []
    lines: list[str] = []
    active_task_id = _safe_metadata_text(snapshot.get("active_task_id"))
    if active_task_id:
        lines.append(f"- active_task_id: `{active_task_id}`")
    tasks = snapshot.get("tasks")
    if isinstance(tasks, Mapping):
        for task_id, record in tasks.items():
            if not isinstance(record, Mapping):
                continue
            status = _safe_metadata_text(record.get("status")) or "unknown"
            task_id_text = _safe_metadata_text(task_id)
            if task_id_text:
                lines.append(f"- task `{task_id_text}` status: `{status}`")
            for key in ("task_state_path", "workspec_path", "current_pin_path"):
                value = _safe_metadata_text(record.get(key))
                if value:
                    lines.append(f"  - {key}: `{value}`")
    return lines


def _safe_metadata_text(value: Any) -> str:
    text = str(value or "")
    lowered = text.lower()
    if any(marker in lowered for marker in _SENSITIVE_MARKERS):
        return "[REDACTED]"
    return text[:240]


def _build_wcp_text(
    *,
    current_user_content: str,
    effective_task_id: str | None,
    turn_id: str | None,
    session_id: str | None,
    intake: Mapping[str, Any],
    registry_snapshot: Mapping[str, Any] | None = None,
) -> str:
    lines = [
        "# Working Context Packet",
        "",
        "## Scope",
        "- phase: Phase 3 provider payload enforcement",
        "- boundary: provider-visible api_messages only",
        "- non-claim: not full A/B classification; Phase 4/5 remain separate",
        "",
        "## Coordinates",
        f"- session_id: `{session_id or 'none'}`",
        f"- task_id: `{effective_task_id or 'none'}`",
        f"- turn_id: `{turn_id or 'none'}`",
        "",
        "## Current User Intent",
        current_user_content,
    ]
    safe_paths = []
    for key in ("summary_path", "task_state_path", "packet_dir"):
        value = intake.get(key)
        if value:
            safe_paths.append(f"- {key}: `{value}`")
    if safe_paths:
        lines.extend(["", "## Phase 2 Safe Intake Pointers", *safe_paths])
    registry_lines = _registry_lines(registry_snapshot)
    if registry_lines:
        lines.extend(["", "## Phase 4 Task Registry", *registry_lines])
    lines.extend(
        [
            "",
            "## Provider Payload Exclusions",
            "- full conversation history",
            "- unrelated prior user/assistant/tool messages",
            "- raw intake.md body",
        ]
    )
    return "\n".join(lines).strip()
