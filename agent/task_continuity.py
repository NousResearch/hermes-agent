"""Post-compression task-continuity guard.

This module keeps lossy context compaction from turning synthetic state
(preserved todo snapshots, structured context injections, background notices) into
fresh user intent.  It is deliberately heuristic and conservative: it only asks
for continuity arbitration when multiple task surfaces conflict after repeated
compression, instead of trying to choose a winner silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Optional


CONTINUITY_CHECK_PREFIX = "CONTINUITY CHECK:"

_SYNTHETIC_PREFIXES = (
    "Current todo list:",
    "[STRUCTURED CONTEXT]",
    "[CONTEXT COMPACTION",
    "[CONTEXT SUMMARY]",
    "[IMPORTANT: Background process",
    "[Background process",
)

_PROTECTED_HINTS = (
    "AGENTS.md",
    "CLAUDE.md",
    "SOUL.md",
    "config.yaml",
    "auth.json",
    ".env",
    "credential",
    "secret",
    "sudo",
    "protected-file",
)

_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "into", "your",
    "you", "are", "was", "were", "will", "shall", "task", "todo", "list",
    "current", "active", "in", "on", "to", "of", "a", "an", "is", "be",
    "as", "or", "by", "it", "its", "then", "now", "next",
}


@dataclass
class CurrentTaskFrame:
    task_title: str = ""
    objective: str = ""
    last_assistant_intent: str = ""
    last_verified_tool_or_action: str = ""
    files_or_surfaces: List[str] = field(default_factory=list)
    active_todo_ids: List[str] = field(default_factory=list)
    latest_user_surfaces: List[str] = field(default_factory=list)
    assistant_surfaces: List[str] = field(default_factory=list)
    pending_next_action: str = ""
    protected_or_high_risk: bool = False
    current_todo_list_snapshot: List[Dict[str, Any]] = field(default_factory=list)
    authorizing_user_message: str = ""
    compression_count: int = 0


@dataclass
class TaskStateConflict:
    should_block_tools: bool
    code: str = ""
    latest_real_user_message: str = ""
    preserved_active_task: str = ""
    structured_context: str = ""
    background_notifications: List[str] = field(default_factory=list)
    frame: CurrentTaskFrame = field(default_factory=CurrentTaskFrame)
    reasons: List[str] = field(default_factory=list)


def classify_message_type(message: Dict[str, Any]) -> str:
    """Classify a transcript message by intent source, not just OpenAI role."""
    explicit = message.get("message_type") or message.get("type")
    if explicit:
        return str(explicit)
    metadata = message.get("metadata")
    if isinstance(metadata, dict) and metadata.get("message_type"):
        return str(metadata["message_type"])

    role = message.get("role")
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = str(content)
    stripped = content.lstrip()

    if stripped.startswith("Current todo list:"):
        return "preserved_task_list"
    if stripped.startswith("[STRUCTURED CONTEXT]"):
        return "structured_context_injection"
    if stripped.startswith("[IMPORTANT: Background process") or stripped.startswith("[Background process"):
        return "background_process_notification"
    if stripped.startswith("[CONTEXT COMPACTION") or stripped.startswith("[CONTEXT SUMMARY]"):
        return "structured_context_injection"
    if role == "user":
        return "real_user_prompt"
    if role == "assistant":
        return "model_assistant_content"
    if role == "tool":
        return "tool_result"
    return "model"


def _content_text(message: Dict[str, Any]) -> str:
    content = message.get("content") or ""
    if isinstance(content, str):
        return content
    return str(content)


def _latest_message(messages: Iterable[Dict[str, Any]], *, kind: Optional[str] = None, role: Optional[str] = None) -> Optional[Dict[str, Any]]:
    for msg in reversed(list(messages)):
        if role is not None and msg.get("role") != role:
            continue
        if kind is not None and classify_message_type(msg) != kind:
            continue
        return msg
    return None


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9_./-]{2,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _overlap(a: str, b: str) -> float:
    ta = _tokens(a)
    tb = _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def _todo_active_text(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    active = []
    for item in items:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).lower()
        if status == "in_progress" or status == "active":
            active.append(" ".join(str(item.get(k, "")) for k in ("id", "content") if item.get(k)))
    return "\n".join(active)


def _active_todo_ids(items: Any) -> List[str]:
    if not isinstance(items, list):
        return []
    active_ids: List[str] = []
    seen = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "")).lower()
        todo_id = str(item.get("id", "")).strip()
        if status not in {"in_progress", "active"} or not todo_id:
            continue
        lowered = todo_id.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        active_ids.append(todo_id)
    return active_ids


def _extract_surfaces(messages: Iterable[Dict[str, Any]]) -> List[str]:
    surfaces: List[str] = []
    seen = set()
    surface_pattern = re.compile(
        r"(?:~?/[^\s`'\"]+|[A-Za-z0-9_./-]+\.(?:md|py|json|yaml|yml|toml))"
    )
    for msg in messages:
        text = _content_text(msg)
        for match in surface_pattern.findall(text):
            cleaned = match.rstrip(".,);]")
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                surfaces.append(cleaned)
    return surfaces[:20]


def _surfaces_from_text(text: str) -> List[str]:
    return _extract_surfaces([{"content": text or ""}])


def _normalized_values(values: Iterable[str]) -> set[str]:
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _mentions_any(text: str, values: Iterable[str]) -> bool:
    text_l = (text or "").lower()
    return any(value in text_l for value in _normalized_values(values))


def capture_current_task_frame(
    messages: List[Dict[str, Any]],
    *,
    current_todo_items: Optional[List[Dict[str, Any]]] = None,
    compression_count: int = 0,
    latest_real_user_message: Optional[str] = None,
) -> CurrentTaskFrame:
    latest_real = latest_real_user_message
    if not latest_real:
        latest_user_msg = _latest_message(messages, kind="real_user_prompt")
        latest_real = _content_text(latest_user_msg) if latest_user_msg else ""

    last_assistant = _latest_message(messages, role="assistant")
    last_tool = _latest_message(messages, role="tool")
    todo_snapshot = list(current_todo_items or [])
    active_text = _todo_active_text(todo_snapshot)
    active_ids = _active_todo_ids(todo_snapshot)
    surfaces = _extract_surfaces(messages)
    latest_surfaces = _surfaces_from_text(latest_real or "")
    assistant_surfaces = _surfaces_from_text(_content_text(last_assistant) if last_assistant else "")
    combined = "\n".join([latest_real or "", active_text, _content_text(last_assistant) if last_assistant else "", "\n".join(surfaces)])

    title = active_text.splitlines()[0] if active_text else (latest_real.splitlines()[0] if latest_real else "")
    return CurrentTaskFrame(
        task_title=title[:160],
        objective=(active_text or latest_real)[:500],
        last_assistant_intent=(_content_text(last_assistant) if last_assistant else "")[:500],
        last_verified_tool_or_action=(_content_text(last_tool) if last_tool else "")[:500],
        files_or_surfaces=surfaces,
        active_todo_ids=active_ids,
        latest_user_surfaces=latest_surfaces,
        assistant_surfaces=assistant_surfaces,
        pending_next_action=(active_text or (_content_text(last_assistant) if last_assistant else ""))[:500],
        protected_or_high_risk=any(h.lower() in combined.lower() for h in _PROTECTED_HINTS),
        current_todo_list_snapshot=todo_snapshot,
        authorizing_user_message=(latest_real or "")[:1000],
        compression_count=compression_count,
    )


def task_frame_to_dict(frame: CurrentTaskFrame) -> Dict[str, Any]:
    """Serialize the compact continuity ledger stored with a turn."""
    return {
        "task_title": frame.task_title,
        "objective": frame.objective,
        "files_or_surfaces": list(frame.files_or_surfaces),
        "active_todo_ids": list(frame.active_todo_ids),
        "latest_user_surfaces": list(frame.latest_user_surfaces),
        "assistant_surfaces": list(frame.assistant_surfaces),
        "pending_next_action": frame.pending_next_action,
        "protected_or_high_risk": frame.protected_or_high_risk,
        "authorizing_user_message": frame.authorizing_user_message,
        "compression_count": frame.compression_count,
    }


def detect_task_state_conflict(
    frame: CurrentTaskFrame,
    *,
    latest_real_user_message: str,
    preserved_active_task_list: Any = None,
    structured_context: str = "",
    background_notifications: Optional[List[str]] = None,
) -> TaskStateConflict:
    latest_real_user_message = latest_real_user_message or frame.authorizing_user_message or ""
    active_text = _todo_active_text(preserved_active_task_list)
    structured_context = structured_context or ""
    background_notifications = background_notifications or []
    assistant_intent = frame.last_assistant_intent or ""
    active_ids = _active_todo_ids(preserved_active_task_list) or list(frame.active_todo_ids)
    latest_surfaces = _surfaces_from_text(latest_real_user_message)
    frame_surfaces = list(frame.files_or_surfaces or []) + list(frame.assistant_surfaces or [])

    latest_explicitly_resumes_active_todo = _mentions_any(latest_real_user_message, active_ids)
    latest_explicitly_authorizes_known_surface = bool(
        _normalized_values(latest_surfaces) & _normalized_values(frame_surfaces)
    )
    if (
        latest_explicitly_resumes_active_todo
        or latest_explicitly_authorizes_known_surface
    ):
        return TaskStateConflict(
            should_block_tools=False,
            latest_real_user_message=latest_real_user_message[:500],
            preserved_active_task=active_text[:500],
            structured_context=structured_context[:500],
            background_notifications=[n[:500] for n in background_notifications],
            frame=frame,
            reasons=[],
        )

    reasons: List[str] = []
    latest_vs_active = _overlap(latest_real_user_message, active_text)
    latest_vs_assistant = _overlap(latest_real_user_message, assistant_intent)
    latest_vs_context = _overlap(latest_real_user_message, structured_context)
    active_vs_assistant = _overlap(active_text, assistant_intent)
    active_vs_context = _overlap(active_text, structured_context)

    if active_text and latest_real_user_message and latest_vs_active < 0.20:
        reasons.append("latest real user prompt conflicts with preserved active task list")
    if assistant_intent and latest_real_user_message and latest_vs_assistant < 0.20:
        reasons.append("last assistant intent conflicts with latest real user prompt")
    if structured_context and latest_real_user_message and latest_vs_context < 0.20:
        reasons.append("structured context conflicts with latest real user prompt")
    if active_text and assistant_intent and active_vs_assistant >= 0.20 and latest_vs_active < 0.20:
        reasons.append("assistant intent aligns with preserved task, not latest real user prompt")
    if active_text and structured_context and active_vs_context >= 0.20 and latest_vs_context < 0.20:
        reasons.append("structured context aligns with preserved task, not latest real user prompt")
    if background_notifications and latest_real_user_message:
        joined_bg = "\n".join(background_notifications)
        if _overlap(latest_real_user_message, joined_bg) < 0.20:
            reasons.append("background notification is not the latest real user prompt")

    # Conservative trigger: only block when compression has made state lossy or
    # protected/high-risk work is active, and at least two surfaces disagree.
    high_risk_window = frame.compression_count >= 5 or frame.protected_or_high_risk
    should_block = high_risk_window and len(set(reasons)) >= 2

    return TaskStateConflict(
        should_block_tools=should_block,
        code="task_vector_drift" if should_block else "",
        latest_real_user_message=latest_real_user_message[:500],
        preserved_active_task=active_text[:500],
        structured_context=structured_context[:500],
        background_notifications=[n[:500] for n in background_notifications],
        frame=frame,
        reasons=reasons,
    )


def format_continuity_check_response(conflict: TaskStateConflict) -> str:
    """Render the forced no-tool continuity arbitration response."""
    reasons = "\n".join(f"- {reason}" for reason in conflict.reasons) or "- conflicting task state"
    latest = conflict.latest_real_user_message or "(none detected)"
    active = conflict.preserved_active_task or "(none detected)"
    structured = conflict.structured_context or "(none detected)"
    last_intent = conflict.frame.last_assistant_intent or "(none detected)"
    last_tool = conflict.frame.last_verified_tool_or_action or "(none detected)"

    return (
        f"{CONTINUITY_CHECK_PREFIX} I have conflicting task state. Choose A/B/C.\n\n"
        f"A. Continue the latest real user prompt:\n{latest}\n\n"
        f"B. Continue the preserved active task:\n{active}\n\n"
        f"C. Ignore both and give me a new explicit target.\n\n"
        f"Why I stopped before tools:\n{reasons}\n\n"
        f"Last assistant intent:\n{last_intent}\n\n"
        f"Last verified tool/action:\n{last_tool}\n\n"
        f"Structured context injection:\n{structured}"
    )
