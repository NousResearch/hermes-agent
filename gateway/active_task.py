"""Deterministic active-task anchoring for gateway turns."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, Optional


LCM_ACTIVE_TASK = "ACTIVE_TASK"
LCM_BACKGROUND_TASK = "BACKGROUND_TASK"
LCM_PRIOR_INVESTIGATION = "PRIOR_INVESTIGATION"
LCM_CLOSED_FINDING = "CLOSED_FINDING"
LCM_USER_CORRECTION = "USER_CORRECTION"

LCM_LABELS = {
    LCM_ACTIVE_TASK,
    LCM_BACKGROUND_TASK,
    LCM_PRIOR_INVESTIGATION,
    LCM_CLOSED_FINDING,
    LCM_USER_CORRECTION,
}

_CORRECTION_RE = re.compile(
    r"""(?ix)
    ^\s*(?:
        no\b
        | nope\b
        | wrong\b
        | stop\b
        | not\s+(?:what|the\s+thing)\s+i\s+asked
        | that'?s\s+not\s+what\s+i\s+asked
        | you\s+(?:bailed|gave\s+up|repeated\s+yourself|missed|ignored)
        | repeated\s+yourself
        | you\s+didn'?t\s+(?:do|answer|fix|follow)
        | this\s+is\s+(?:wrong|not\s+right)
    )""",
)

_LOW_SIGNAL_RE = re.compile(r"^\s*(?:ok|okay|thanks|thank you|got it|cool|yes|yep|sure)\s*[.!]?\s*$", re.I)
_INSTRUCTION_VERB_RE = re.compile(
    r"^\s*(?:please\s+)?(?:fix|implement|add|update|remove|run|check|inspect|debug|route|send|write|create|make|keep|use|do|don'?t|stop|continue|resume|verify|test|deploy)\b",
    re.I,
)


@dataclass(frozen=True)
class ActiveTaskAnchor:
    source: str
    priority: int
    text: str
    task_id: Optional[str] = None
    lcm_label: str = LCM_ACTIVE_TASK
    correction_mode: bool = False
    chat_id: Optional[str] = None
    thread_id: Optional[str] = None
    source_message_id: Optional[str] = None
    reply_to_message_id: Optional[str] = None


def is_correction_message(text: str) -> bool:
    return bool(_CORRECTION_RE.search(text or ""))


def is_explicit_latest_instruction(text: str) -> bool:
    stripped = (text or "").strip()
    if not stripped or is_correction_message(stripped) or _LOW_SIGNAL_RE.match(stripped):
        return False
    return bool("?" in stripped or _INSTRUCTION_VERB_RE.search(stripped) or len(stripped.split()) >= 4)


def classify_lcm_summary(content: str, *, active: bool = False, correction: bool = False) -> str:
    text = (content or "").strip()
    lower = text.lower()
    if correction or is_correction_message(text):
        return LCM_USER_CORRECTION
    if active:
        return LCM_ACTIVE_TASK
    if any(token in lower for token in ("closed finding", "completed actions", "resolved questions", "task done", "done:")):
        return LCM_CLOSED_FINDING
    if any(token in lower for token in ("prior investigation", "previous investigation", "stale", "historical", "already addressed")):
        return LCM_PRIOR_INVESTIGATION
    return LCM_BACKGROUND_TASK


def _same_thread(task: Mapping[str, Any], *, chat_id: Optional[str], thread_id: Optional[str]) -> bool:
    if str(task.get("chat_id") or "") != str(chat_id or ""):
        return False
    return str(task.get("thread_id") or "") == str(thread_id or "")


def _task_text(task: Mapping[str, Any]) -> str:
    return str(task.get("content") or task.get("title") or "").strip()


def _latest_user_ask(history: Iterable[Mapping[str, Any]], current_text: str) -> Optional[str]:
    if is_explicit_latest_instruction(current_text):
        return current_text.strip()
    for msg in reversed(list(history or [])):
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str) and is_explicit_latest_instruction(content):
            return content.strip()
    return None


def _lcm_summaries(history: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for msg in history or []:
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        lower = content.lower()
        if "## active task" in lower or "summary" in lower or "prior investigation" in lower:
            label = str(msg.get("lcm_label") or classify_lcm_summary(content))
            summaries.append({"content": content.strip(), "lcm_label": label if label in LCM_LABELS else LCM_BACKGROUND_TASK})
    return summaries


def resolve_active_task_anchor(
    *,
    current_text: str,
    history: Iterable[Mapping[str, Any]] = (),
    open_tasks: Iterable[Mapping[str, Any]] = (),
    chat_id: Optional[str] = None,
    thread_id: Optional[str] = None,
    source_message_id: Optional[str] = None,
    reply_to_message_id: Optional[str] = None,
    reply_to_text: Optional[str] = None,
) -> ActiveTaskAnchor:
    """Resolve current task with priority: latest instruction, reply anchor,
    same-thread open task, latest ask, LCM summaries, other tasks.
    """

    correction_mode = is_correction_message(current_text)
    tasks = list(open_tasks or [])

    if not correction_mode and is_explicit_latest_instruction(current_text):
        return ActiveTaskAnchor(
            source="explicit_latest_instruction",
            priority=1,
            text=current_text.strip(),
            correction_mode=False,
            chat_id=chat_id,
            thread_id=thread_id,
            source_message_id=source_message_id,
            reply_to_message_id=reply_to_message_id,
        )

    if reply_to_message_id:
        for task in tasks:
            anchors = {
                str(task.get("source_message_id") or ""),
                str(task.get("reply_to_message_id") or ""),
            }
            if str(reply_to_message_id) in anchors and _same_thread(task, chat_id=chat_id, thread_id=thread_id):
                return ActiveTaskAnchor(
                    source="reply_to_task_anchor",
                    priority=2,
                    text=_task_text(task) or (reply_to_text or current_text or "").strip(),
                    task_id=str(task.get("task_id") or task.get("id") or ""),
                    lcm_label=str(task.get("lcm_label") or LCM_ACTIVE_TASK),
                    correction_mode=correction_mode,
                    chat_id=chat_id,
                    thread_id=thread_id,
                    source_message_id=str(task.get("source_message_id") or ""),
                    reply_to_message_id=str(reply_to_message_id),
                )

    for task in tasks:
        if _same_thread(task, chat_id=chat_id, thread_id=thread_id):
            return ActiveTaskAnchor(
                source="open_task_same_chat_thread",
                priority=3,
                text=_task_text(task),
                task_id=str(task.get("task_id") or task.get("id") or ""),
                lcm_label=str(task.get("lcm_label") or LCM_ACTIVE_TASK),
                correction_mode=correction_mode,
                chat_id=chat_id,
                thread_id=thread_id,
                source_message_id=str(task.get("source_message_id") or ""),
                reply_to_message_id=reply_to_message_id,
            )

    latest = _latest_user_ask(history, current_text)
    if latest:
        return ActiveTaskAnchor(
            source="latest_ask",
            priority=4,
            text=latest,
            lcm_label=LCM_BACKGROUND_TASK,
            correction_mode=correction_mode,
            chat_id=chat_id,
            thread_id=thread_id,
            source_message_id=source_message_id,
            reply_to_message_id=reply_to_message_id,
        )

    summaries = _lcm_summaries(history)
    if summaries:
        summary = summaries[-1]
        return ActiveTaskAnchor(
            source="lcm_summary",
            priority=5,
            text=summary["content"],
            lcm_label=summary["lcm_label"],
            correction_mode=correction_mode,
            chat_id=chat_id,
            thread_id=thread_id,
            source_message_id=source_message_id,
            reply_to_message_id=reply_to_message_id,
        )

    for task in tasks:
        return ActiveTaskAnchor(
            source="other_task",
            priority=6,
            text=_task_text(task),
            task_id=str(task.get("task_id") or task.get("id") or ""),
            lcm_label=str(task.get("lcm_label") or LCM_BACKGROUND_TASK),
            correction_mode=correction_mode,
            chat_id=str(task.get("chat_id") or ""),
            thread_id=str(task.get("thread_id") or ""),
            source_message_id=str(task.get("source_message_id") or ""),
            reply_to_message_id=reply_to_message_id,
        )

    return ActiveTaskAnchor(
        source="none",
        priority=99,
        text="",
        lcm_label=LCM_BACKGROUND_TASK,
        correction_mode=correction_mode,
        chat_id=chat_id,
        thread_id=thread_id,
        source_message_id=source_message_id,
        reply_to_message_id=reply_to_message_id,
    )


def format_active_task_anchor_block(anchor: ActiveTaskAnchor) -> str:
    lines = [
        "[Active Task Anchor]",
        f"Resolver source: {anchor.source}",
        f"Resolver priority: {anchor.priority}",
        f"LCM label: {anchor.lcm_label}",
        f"Correction mode: {'on' if anchor.correction_mode else 'off'}",
    ]
    if anchor.chat_id:
        lines.append(f"Chat: {anchor.chat_id}")
    if anchor.thread_id:
        lines.append(f"Thread/topic: {anchor.thread_id}")
    if anchor.source_message_id:
        lines.append(f"Source message: {anchor.source_message_id}")
    if anchor.reply_to_message_id:
        lines.append(f"Reply-to message: {anchor.reply_to_message_id}")
    if anchor.text:
        lines.append("Task:")
        lines.append(anchor.text[:2000])
    else:
        lines.append("Task: unresolved")
    lines.append(
        "Instruction: Resolve this turn from this active task before using broad memory, LCM summaries, or unrelated prior tasks."
    )
    return "\n".join(lines)
