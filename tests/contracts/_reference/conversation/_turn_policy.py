"""The turn and transcript policy the in-memory reference applies to its rows.

The store's projection and user-turn policy is the neutral ``session_rules.turns`` module every backend runs unchanged; this module is the one place the reference reaches it. Small pure helpers live here as private functions.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, MutableMapping, Optional

from session_rules import turns

#: ``display_metadata`` key of a row only the model sees.
MODEL_ONLY_KEY = turns.MODEL_ONLY_DISPLAY_METADATA_KEY


def sanitize_context(text: str) -> str:
    return str(turns.sanitize_context(text))


def loaded_view(role: Any, content: Any) -> Any:
    """Content as a replay load hands it out: user/assistant strings sanitized and stripped (``SessionMessagesMixin._loaded_view_content``)."""
    if role in {"user", "assistant"} and isinstance(content, str):
        return sanitize_context(content).strip()
    return content


def is_content_blank(content: Any) -> bool:
    return bool(turns.is_content_blank(content))


def coerce_epoch(value: Any) -> Optional[float]:
    from hermes_cli.timefmt import coerce_epoch as coerce

    result = coerce(value, field="message timestamp")
    return None if result is None else float(result)


def mint_uid() -> str:
    return str(turns.mint_uid())


def coalesce_tool_call_id(call: Any) -> Any:
    return turns.coalesce_tool_call_id(call)


def index_tool_call_uids(index: MutableMapping[str, str], assistant: Mapping[str, Any]) -> None:
    turns.index_tool_call_uids(index, assistant)


def resolve_tool_call_uid(index: MutableMapping[str, str], tool_call_id: Any) -> Optional[str]:
    uid = turns.resolve_tool_call_uid(index, tool_call_id)
    return uid if isinstance(uid, str) and uid else None


def split_user_originated_turn(message: Mapping[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    handoff, live = turns.split_user_originated_turn(message)
    return handoff, live


def user_originated_turn_view(message: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    result: Optional[dict[str, Any]] = turns.user_originated_turn_view(message)
    return result


def history_before_user_originated_turn(messages: list[dict[str, Any]], index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prefix, live = turns.history_before_user_originated_turn(messages, index)
    return prefix, live


def retryable_user_text(content: Any) -> str:
    return str(turns.retryable_user_text(content))


def composite_carrier(summary: str, ask: str) -> str:
    """A composite compaction carrier: the summary block and the live ask in one user row, as ``split_user_originated_turn`` parses it."""
    return f"{turns.SUMMARY_PREFIX}\n{turns.HISTORICAL_TASK_HEADING}\n{summary}\n\n{turns._SUMMARY_END_MARKER}\n\n{ask}"


def flatten_message_text(content: Any) -> str:
    return str(turns.flatten_message_text(content))


def is_ephemeral_scaffolding(message: Any) -> bool:
    return bool(turns.is_ephemeral_scaffolding(message))


def repair_message_sequence(messages: list[dict[str, Any]]) -> None:
    turns.repair_message_sequence(None, messages)


def project_compaction_message_for_display(message: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    result: Optional[dict[str, Any]] = turns.project_compaction_message_for_display(message)
    return result


def skill_preview_text(text: str) -> str:
    """A skill invocation as its description, else the text before the skill excerpt (the skill step of the listing preview)."""
    described = turns.describe_skill_invocation(text)
    return str(described) if described is not None else text.split(turns.SKILL_EXCERPT_JOINT)[0]


def durable_content(content: Any) -> Any:
    return turns.durable_content(content)


_REVIEW_HARNESS_PREFIXES = (
    "Review the conversation above and update the skill library",
    "Review the conversation above and consider saving to memory",
)

#: A bare protocol or tool-name marker such as ``[memory]``.
STALE_TOOL_CALL_MARKER_RE = turns.STALE_TOOL_CALL_MARKER_RE


def _is_review_harness(message: Mapping[str, Any]) -> bool:
    """A persisted background-review prompt."""
    content = message.get("content")
    return message.get("role") in {"user", "system"} and isinstance(content, str) and content.lstrip().startswith(_REVIEW_HARNESS_PREFIXES)


def strip_background_review_harness(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Harness prompts and the assistant reply right after each dropped."""
    out: list[dict[str, Any]] = []
    skip_next_assistant = previous_was_harness = False
    for message in messages:
        if _is_review_harness(message):
            skip_next_assistant = not previous_was_harness
            previous_was_harness = True
            continue
        if skip_next_assistant:
            skip_next_assistant = False
            if message.get("role") == "assistant":
                previous_was_harness = False
                continue
        previous_was_harness = False
        out.append(message)
    return out


def strip_stale_tool_call_markers(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Blank the content of tool-calling assistant turns that hold a bare marker."""
    for message in messages:
        content = message.get("content")
        if message.get("role") == "assistant" and message.get("tool_calls") and isinstance(content, str) and STALE_TOOL_CALL_MARKER_RE.fullmatch(content.strip()):
            message["content"] = ""
    return messages


def _canonical_replayed_user(message: Mapping[str, Any]) -> tuple[Any, bool]:
    """Canonical live content and whether the user row is a composite carrier (``SessionMessagesMixin._canonical_replayed_user_content``)."""
    if message.get("role") != "user":
        return None, False
    handoff, live = split_user_originated_turn(message)
    composite = handoff is not None and live is not None
    return (live.get("content") if composite and live is not None else message.get("content")), composite


def _exact_clone_key(timestamp: Any, content: Any) -> Optional[tuple[Any, str]]:
    """The key of a column-exact rotation clone (``SessionMessagesMixin._exact_replayed_user_clone_key``)."""
    if timestamp is None or content in (None, "", []):
        return None
    try:
        return timestamp, json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return None


def _adjacent_duplicate(messages: list[dict[str, Any]], message: Mapping[str, Any]) -> Optional[tuple[int, bool]]:
    """The adjacent replay duplicate ``(index, prefer_current)`` (``SessionMessagesMixin._find_duplicate_replayed_user_message``)."""
    if message.get("role") != "user":
        return None
    content, prefer_current = _canonical_replayed_user(message)
    if content in (None, "", []):
        return None
    for index in range(len(messages) - 1, -1, -1):
        previous = messages[index]
        if previous.get("role") == "user":
            previous_content, previous_composite = _canonical_replayed_user(previous)
            if previous_content == content and (prefer_current or previous_composite or isinstance(content, str)):
                return index, prefer_current
        elif previous.get("role") == "assistant" and (previous.get("content") or previous.get("tool_calls")):
            return None
    return None


def dedupe_replayed_user(messages: list[dict[str, Any]], message: dict[str, Any], clones: dict[Any, dict[str, Any]]) -> tuple[bool, Any]:
    """Ancestor-lineage dedupe of one replayed user row: ``(skip, exact_clone_key)`` (``SessionMessagesMixin._dedupe_replayed_user``)."""
    clone_key = _exact_clone_key(message.get("timestamp"), _canonical_replayed_user(message)[0])
    previous = clones.get(clone_key) if clone_key is not None else None
    duplicate = None
    if previous is not None:
        index = next((i for i, candidate in enumerate(messages) if candidate is previous), None)
        if index is not None:
            duplicate = (index, True)
    if duplicate is None:
        duplicate = _adjacent_duplicate(messages, message)
    if duplicate is None:
        return False, clone_key
    index, prefer_current = duplicate
    if prefer_current:
        messages.pop(index)
    return not prefer_current, clone_key


#: Positions of the user-originated turns, and content projected the way the durable row stores it.
user_indices = turns.user_indices
comparison_content = turns.comparison_content


_SYNTHETIC_PROMPT = re.compile(
    r"^\s*(?:\[IMPORTANT: Background process |\[ASYNC (?:DELEGATION )?(?:BATCH )?COMPLETE\b|"
    r"A background fan-out of \d+ subagent\(s\) you dispatched earlier has finished\.|"
    r"A background subagent you dispatched earlier has finished\.)",
    re.IGNORECASE,
)


def prompt_preview(content: Any, display_kind: Optional[str], summary: bool) -> str:
    """The timeline preview of a user row, ``""`` when it is no human prompt."""
    message = project_compaction_message_for_display({"role": "user", "content": content, "display_kind": display_kind, "_compressed_summary": summary})
    if message is None or user_originated_turn_view(message) is None:
        return ""
    text = message.get("content")
    if isinstance(text, list):
        text = " ".join(part if isinstance(part, str) else part.get("text", "") for part in text if isinstance(part, (str, dict)))
    if not isinstance(text, str):
        return ""
    text = " ".join(text.split())
    if not text or _SYNTHETIC_PROMPT.match(text):
        return ""
    return text if len(text) <= 120 else text[:119].rstrip() + "\u2026"


def flatten_search_text(decoded: Any) -> str:
    """Multimodal part list as joined text (or the placeholder); a string passes through."""
    if isinstance(decoded, list):
        parts = [p.get("text", "") for p in decoded if isinstance(p, dict) and p.get("type") == "text"]
        return " ".join(t for t in parts if t).strip() or "[multimodal content]"
    return decoded if isinstance(decoded, str) else ""
