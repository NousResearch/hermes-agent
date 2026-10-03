"""Client-facing projection helpers for model-only compaction carriers."""

from __future__ import annotations

from typing import Any, Dict, Optional

from agent.context_compressor import ContextCompressor, is_compaction_summary_message

try:
    from tools.todo_tool import TODO_INJECTION_HEADER
except Exception:  # tools tree unavailable in minimal imports
    TODO_INJECTION_HEADER = "[Your active task list was preserved across context compression]"


_COMPACTION_INTERNAL_FIELDS = (
    "tool_calls",
    "finish_reason",
    "reasoning",
    # Provider replay/metadata fields that ride the wire on every request but are invisible to
    # ``msg["content"]``/``msg["tool_calls"]`` accounting. Codex Responses sessions in particular carry
    # ``codex_reasoning_items`` blobs of ``encrypted_content`` that can dominate the serialized session (a
    # measured 214-turn session held ~115K tokens / 27% of its payload there — #55572).
    # ``reasoning_details`` is handled separately (see ``_reasoning_details_text_chars``): its signed/base64
    # envelope is excluded from the budget, mirroring the preflight estimator's exclusion in
    # ``model_metadata._estimate_message_tokens_without_images`` (#73298).
    # An assistant turn may carry only reasoning/thinking content with no visible text (extended-thinking
    # turns, thinking-only recovery responses). Such a turn is persisted with its reasoning fields and is
    # recallable from the transcript, but dropping it here as "empty" makes it vanish from the
    # resumed/reloaded session view while the desktop's reasoning disclosure has nothing to render. Keep it
    # when it carries reasoning so the "Thinking…" block still shows. (#44022)
    "reasoning_content",
    "reasoning_details",
    "codex_reasoning_items",
    "codex_message_items",
)


def _strip_todo_snapshot_content(content: Any) -> Any:
    """Remove a compaction TODO snapshot block (header to end) from content."""
    if isinstance(content, str):
        idx = content.find(TODO_INJECTION_HEADER)
        return content[:idx].rstrip() if idx != -1 else content
    if isinstance(content, list):
        cleaned: list = []
        for part in content:
            text = str(part.get("text") or "") if isinstance(part, dict) and part.get("type") == "text" else ""
            idx = text.find(TODO_INJECTION_HEADER) if text else -1
            if idx == -1:
                cleaned.append(part)
            elif stripped := text[:idx].rstrip():
                cleaned.append({**part, "text": stripped})
        return cleaned
    return content


def _todo_snapshot_is_only_content(content: Any, stripped: Any) -> bool:
    """Whether stripping the snapshot leaves no displayable content."""
    if isinstance(content, str) and isinstance(stripped, str):
        return not stripped.strip()
    if isinstance(content, list) and isinstance(stripped, list):
        return not stripped
    return False


def is_todo_snapshot_message(message: Any) -> bool:
    """True for model-only TODO continuity rows (flagged or header-bearing)."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    if message.get("_todo_snapshot_synthetic"):
        return True
    content = message.get("content")
    if isinstance(content, str):
        return TODO_INJECTION_HEADER in content
    if isinstance(content, list):
        return any(
            isinstance(p, dict) and p.get("type") == "text"
            and TODO_INJECTION_HEADER in str(p.get("text") or "")
            for p in content
        )
    return False


def project_compaction_message_for_display(message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return authentic transcript content, or ``None`` for a pure handoff.

    Model-facing recovery history retains the complete carrier. Display
    projections instead remove the handoff, inherited tool state, and internal
    reasoning while preserving any real prior-tail content or live user ask
    embedded in the carrier. A standalone TODO snapshot is pure scaffolding
    (``None``); a merged carrier keeps only its authentic human text.
    """
    if not isinstance(message, dict):
        return None
    if is_todo_snapshot_message(message):
        content = message.get("content")
        stripped = _strip_todo_snapshot_content(content)
        if _todo_snapshot_is_only_content(content, stripped):
            return None
        projected = message.copy()
        projected["content"] = stripped
        projected.pop("_todo_snapshot_synthetic", None)
        projected.pop("display_kind", None)
        for key in _COMPACTION_INTERNAL_FIELDS:
            projected.pop(key, None)
        return projected
    if not is_compaction_summary_message(message):
        return message.copy()

    projected = ContextCompressor._strip_context_summary_handoff_message(message)
    if projected is None:
        return None

    projected = projected.copy()
    for key in _COMPACTION_INTERNAL_FIELDS:
        projected.pop(key, None)
    projected.pop("display_kind", None)
    return projected
