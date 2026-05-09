"""Post-compression task-continuity guard.

Context compression can preserve todo snapshots, structured context injections, or
background-process notices as user-role messages. Those surfaces are useful, but
they are not fresh user intent. This module classifies those surfaces and blocks
tool execution when a heavily compressed/high-risk turn has conflicting task
vectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional


CONTINUITY_CHECK_PREFIX = "CONTINUITY CHECK:"
TASK_FRAME_LEDGER_SCHEMA_VERSION = 1

_HIGH_RISK_TERMS = (
    "system prompt",
    "policy file",
    "protected file",
    "security policy",
    "governance",
    "sudo",
    "credential",
    "secret",
    "token",
    "password",
    "private key",
    "api key",
    "production",
    "customer data",
    "client data",
    "deployment",
)


@dataclass
class CurrentTaskFrame:
    latest_real_user_message: str = ""
    preserved_task_list: str = ""
    last_assistant_intent: str = ""
    last_verified_tool_action: str = ""
    compression_count: int = 0
    protected_or_high_risk_active: bool = False
    current_todo_items: List[Dict[str, Any]] = field(default_factory=list)
    active_todo_ids: List[str] = field(default_factory=list)
    authorized_surfaces: List[str] = field(default_factory=list)
    last_tool_action: Dict[str, str] = field(default_factory=dict)
    synthetic_context_sources: List[str] = field(default_factory=list)
    risk_flags: List[str] = field(default_factory=list)


@dataclass
class TaskStateConflict:
    should_block_tools: bool
    reason: str = ""
    latest_real_user_message: str = ""
    preserved_task_list: str = ""
    last_assistant_intent: str = ""
    last_verified_tool_action: str = ""
    structured_context_injection: str = ""
    background_process_notification: str = ""
    compression_count: int = 0
    conflict_surfaces: List[str] = field(default_factory=list)


def _message_content(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _first_nonempty(*values: Any) -> Optional[str]:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def classify_message_type(message: Dict[str, Any]) -> str:
    """Classify an internal/persisted message into a continuity-aware type."""
    if not isinstance(message, dict):
        return "model"

    explicit = _first_nonempty(message.get("message_type"))
    metadata = message.get("metadata")
    if explicit:
        return explicit
    if isinstance(metadata, dict):
        meta_type = _first_nonempty(metadata.get("message_type"))
        if meta_type:
            return meta_type

    content = _message_content(message)
    stripped = content.lstrip()
    if stripped.startswith("Current todo list:"):
        return "preserved_task_list"
    if "[Your active task list was preserved across context compression]" in content:
        return "preserved_task_list"
    if stripped.startswith("[STRUCTURED CONTEXT]"):
        return "structured_context_injection"
    if stripped.startswith("[CONTEXT COMPACTION]") or stripped.startswith("[CONTEXT SUMMARY]"):
        return "structured_context_injection"
    if stripped.startswith("[IMPORTANT: Background process") or stripped.startswith("[Background process"):
        return "background_process_notification"

    role = message.get("role")
    if role == "user":
        return "real_user_prompt"
    if role == "assistant":
        return "model_assistant_content"
    if role == "tool":
        return "tool_result"
    return "model"


def _todo_text(items: Iterable[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).strip().lower()
        if status in {"completed", "cancelled"}:
            continue
        ident = str(item.get("id", "")).strip()
        content = str(item.get("content", "")).strip()
        if content:
            prefix = f"{ident}. " if ident else ""
            suffix = f" ({status})" if status else ""
            lines.append(f"- {prefix}{content}{suffix}")
    return "\n".join(lines)


def _unique_ordered(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values or []:
        norm = str(value or "").strip()
        if not norm:
            continue
        key = norm.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(norm)
    return result


def _active_todo_ids(items: Iterable[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).strip().lower()
        if status in {"completed", "cancelled"}:
            continue
        ident = str(item.get("id", "")).strip()
        if ident:
            ids.append(ident)
    return _unique_ordered(ids)


_SURFACE_RE = re.compile(
    r"(?<![\w.-])(?:"
    r"(?:[~./]?[\w.-]+/)+[\w.@%+=:,.-]+"
    r"|[\w.-]+\.(?:py|js|ts|tsx|jsx|json|md|yaml|yml|sh|toml|txt|css|html)"
    r")"
)


def _extract_authorized_surfaces(text: str) -> List[str]:
    return _unique_ordered(match.group(0).strip("`'\".,;:") for match in _SURFACE_RE.finditer(text or ""))


def _parse_list_field(text: str, field_name: str) -> List[str]:
    pattern = re.compile(rf"(?im)^\s*{re.escape(field_name)}\s*:\s*(.+?)\s*$")
    values: List[str] = []
    for match in pattern.finditer(text or ""):
        raw = match.group(1)
        for part in re.split(r"[, ]+", raw):
            cleaned = part.strip().strip("[]`'\"")
            if cleaned:
                values.append(cleaned)
    return _unique_ordered(values)


def _text_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _short_excerpt(text: str, limit: int = 240) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _last_tool_action(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        content = _message_content(msg).strip()
        target = ""
        surfaces = _extract_authorized_surfaces(content)
        if surfaces:
            target = surfaces[0]
        return {
            "tool": str(msg.get("name") or msg.get("tool_name") or "tool"),
            "target": target,
            "content_sha256": _text_sha256(content) if content else "",
        }
    return {}


def _is_high_risk_text(text: str) -> bool:
    low = (text or "").lower()
    return any(term in low for term in _HIGH_RISK_TERMS)


def _last_assistant_intent(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = _message_content(msg).strip()
        if content:
            return content[:1000]
        tool_calls = msg.get("tool_calls") or []
        if tool_calls:
            names: List[str] = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    name = fn.get("name") if isinstance(fn, dict) else None
                    if name:
                        names.append(str(name))
            if names:
                return "Assistant intended tool calls: " + ", ".join(names)
    return ""


def _last_verified_tool_action(messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        name = str(msg.get("name") or msg.get("tool_name") or "tool")
        content = _message_content(msg).strip()
        if content:
            return f"{name}: {content[:1000]}"
        return name
    return ""


def _last_by_type(messages: List[Dict[str, Any]], message_type: str) -> str:
    for msg in reversed(messages or []):
        if classify_message_type(msg) == message_type:
            content = _message_content(msg).strip()
            if content:
                return content[:2000]
    return ""


def capture_current_task_frame(
    messages: List[Dict[str, Any]],
    *,
    current_todo_items: Optional[List[Dict[str, Any]]] = None,
    compression_count: int = 0,
    latest_real_user_message: str = "",
) -> CurrentTaskFrame:
    """Capture the task vector immediately after compression."""
    todo_items = list(current_todo_items or [])
    preserved = _last_by_type(messages, "preserved_task_list") or _todo_text(todo_items)
    last_user = latest_real_user_message or _last_by_type(messages, "real_user_prompt")
    assistant_intent = _last_assistant_intent(messages)
    last_tool = _last_verified_tool_action(messages)
    high_risk_blob = "\n".join([last_user, preserved, assistant_intent, last_tool])
    authorized_surfaces = _unique_ordered(
        _extract_authorized_surfaces(last_user)
        + _extract_authorized_surfaces(preserved)
        + _extract_authorized_surfaces(last_tool)
    )
    active_ids = _active_todo_ids(todo_items)
    risk_flags: List[str] = []
    if _is_high_risk_text(high_risk_blob):
        risk_flags.append("protected-or-high-risk-text")
    return CurrentTaskFrame(
        latest_real_user_message=last_user,
        preserved_task_list=preserved,
        last_assistant_intent=assistant_intent,
        last_verified_tool_action=last_tool,
        compression_count=int(compression_count or 0),
        protected_or_high_risk_active=_is_high_risk_text(high_risk_blob),
        current_todo_items=todo_items,
        active_todo_ids=active_ids,
        authorized_surfaces=authorized_surfaces,
        last_tool_action=_last_tool_action(messages),
        synthetic_context_sources=[
            mt
            for mt in ("preserved_task_list", "structured_context_injection", "background_process_notification")
            if _last_by_type(messages, mt)
        ],
        risk_flags=risk_flags,
    )


def _surface_texts(
    frame: CurrentTaskFrame,
    *,
    structured_context_injection: str = "",
    background_process_notification: str = "",
) -> Dict[str, str]:
    surfaces = {
        "latest real user prompt": frame.latest_real_user_message,
        "preserved active task": frame.preserved_task_list,
        "last assistant intent": frame.last_assistant_intent,
        "last verified tool/action": frame.last_verified_tool_action,
        "structured context injection": structured_context_injection,
        "background process notification": background_process_notification,
    }
    return {k: v.strip() for k, v in surfaces.items() if isinstance(v, str) and v.strip()}


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def _structured_conflict_reasons(frame: CurrentTaskFrame, synthetic_text: str) -> List[str]:
    reasons: List[str] = []

    frame_surfaces = {s.lower() for s in frame.authorized_surfaces or []}
    synthetic_surfaces = {s.lower() for s in _extract_authorized_surfaces(synthetic_text)}
    synthetic_surfaces.update(s.lower() for s in _parse_list_field(synthetic_text, "authorized_surfaces"))
    if frame_surfaces and synthetic_surfaces:
        unknown_surfaces = synthetic_surfaces - frame_surfaces
        if unknown_surfaces:
            reasons.append("structured authorized surfaces mismatch")

    frame_todo_ids = {str(t).lower() for t in frame.active_todo_ids or []}
    synthetic_todo_ids = {s.lower() for s in _parse_list_field(synthetic_text, "active_todo_ids")}
    if frame_todo_ids and synthetic_todo_ids:
        unknown_todos = synthetic_todo_ids - frame_todo_ids
        if unknown_todos:
            reasons.append("structured active todo ids mismatch")

    return reasons


def _surfaces_conflict(surfaces: Dict[str, str]) -> List[str]:
    names = list(surfaces)
    if len(names) < 2:
        return []
    norms = {_norm(v) for v in surfaces.values() if _norm(v)}
    if len(norms) <= 1:
        return []

    # Conservative: require a preserved/synthetic surface plus another distinct
    # task vector, not just a verbose assistant/tool restatement of the same task.
    synthetic_names = {
        "preserved active task",
        "structured context injection",
        "background process notification",
    }
    if not any(name in synthetic_names for name in names):
        return []
    return names


def detect_task_state_conflict(
    frame: CurrentTaskFrame,
    *,
    structured_context_injection: str = "",
    background_process_notification: str = "",
    extra_structured_context: str = "",
) -> TaskStateConflict:
    """Detect conservative post-compression task-vector drift."""
    structured = "\n\n".join(
        part for part in (structured_context_injection, extra_structured_context) if part
    )
    surfaces = _surface_texts(
        frame,
        structured_context_injection=structured,
        background_process_notification=background_process_notification,
    )
    conflict_surfaces = _surfaces_conflict(surfaces)
    structured_reasons = _structured_conflict_reasons(frame, structured)
    risky_enough = frame.compression_count >= 5 or frame.protected_or_high_risk_active
    should_block = bool(structured_reasons or (risky_enough and len(conflict_surfaces) >= 2))
    reason = ""
    if structured_reasons:
        reason = "; ".join(structured_reasons)
    elif should_block:
        if frame.compression_count >= 5 and frame.protected_or_high_risk_active:
            reason = "high compression count and protected/high-risk task state"
        elif frame.compression_count >= 5:
            reason = "high compression count"
        else:
            reason = "protected/high-risk task state"

    return TaskStateConflict(
        should_block_tools=should_block,
        reason=reason,
        latest_real_user_message=frame.latest_real_user_message,
        preserved_task_list=frame.preserved_task_list,
        last_assistant_intent=frame.last_assistant_intent,
        last_verified_tool_action=frame.last_verified_tool_action,
        structured_context_injection=structured,
        background_process_notification=background_process_notification,
        compression_count=frame.compression_count,
        conflict_surfaces=conflict_surfaces or (["structured task frame"] if structured_reasons else []),
    )


def _section(label: str, value: str) -> str:
    value = _redact_sensitive_text((value or "").strip()) or "(none recorded)"
    return f"{label}:\n{value}"


_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|token|secret|credential|api[_-]?key|key)\s*[:=]\s*[^,\s;]+"
)


def _redact_sensitive_text(text: str) -> str:
    return _SENSITIVE_ASSIGNMENT_RE.sub(lambda m: f"{m.group(1)}=[redacted]", text or "")


def parse_continuity_resolution_choice(text: str) -> Optional[str]:
    normalized = " ".join((text or "").strip().lower().split())
    if normalized in {"a", "choose a", "option a"}:
        return "latest_real_user_prompt"
    if normalized in {"b", "choose b", "option b"}:
        return "preserved_active_task"
    if normalized in {"c", "choose c", "option c"}:
        return "explicit_new_target"
    return None


def build_task_frame_ledger_payload(
    frame: CurrentTaskFrame,
    *,
    turn_id: str = "",
    resolution_choice: str = "",
) -> Dict[str, Any]:
    latest = frame.latest_real_user_message or ""
    return {
        "schema_version": TASK_FRAME_LEDGER_SCHEMA_VERSION,
        "turn_id": turn_id,
        "compression_generation": int(frame.compression_count or 0),
        "latest_real_user_intent": {
            "excerpt": _short_excerpt(_redact_sensitive_text(latest)),
            "sha256": _text_sha256(latest),
        },
        "active_todo_ids": list(frame.active_todo_ids or []),
        "authorized_surfaces": list(frame.authorized_surfaces or []),
        "last_tool_action": dict(frame.last_tool_action or {}),
        "synthetic_context_sources": list(frame.synthetic_context_sources or []),
        "risk_flags": list(frame.risk_flags or []),
        "resolution_choice": resolution_choice,
    }


def format_continuity_check_response(conflict: TaskStateConflict) -> str:
    """Render a user-facing tool-block response for a detected conflict."""
    return "\n\n".join(
        [
            f"{CONTINUITY_CHECK_PREFIX} I have conflicting task state. Choose A/B/C.",
            "I stopped before executing tools because context was compressed and preserved/synthetic task state conflicts with the current task vector. Running tools now could act on the wrong task.",
            "A = continue latest real user prompt\nB = continue preserved active task\nC = ignore both and give explicit new target",
            f"Reason: {conflict.reason or 'conflicting post-compression task surfaces'} (compression_count={conflict.compression_count})",
            _section("Latest real user prompt", conflict.latest_real_user_message),
            _section("Preserved active task", conflict.preserved_task_list),
            _section("Last assistant intent", conflict.last_assistant_intent),
            _section("Last verified tool/action", conflict.last_verified_tool_action),
            _section("Structured context injection", conflict.structured_context_injection),
            _section("Background process notification", conflict.background_process_notification),
        ]
    )
