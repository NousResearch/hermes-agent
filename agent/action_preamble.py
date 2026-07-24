"""Detect no-tool action preambles that should continue the same turn."""

from __future__ import annotations

import re

_ACTION_RE = re.compile(
    r"((?:let me|let's|i'?ll|i will|i'?m going to|i am going to|"
    r"now i|first,?\s+i|next,?\s+i|i need to|i should)\s+"
    r"(?:check|look|run|start|examine|search|read|list|create|write|edit|use|"
    r"inspect|open|review)\b|going to\s+"
    r"(?:check|look|run|start|examine|search|read|list|create|write|edit|use|"
    r"inspect|open|review)\b)",
    re.IGNORECASE,
)
_COMPLETION_RE = re.compile(
    r"(\bdone\b|\bcomplete(d)?\b|nothing to (do|save|change|report|fix)|"
    r"no changes?\b|no action\b|already (complete|done|finished)|\bfinished\b|"
    r"all set\b|no further\b|nothing left\b|here('?s| is| are)\b|"
    r"in summary\b|to summarize\b|the answer is\b)",
    re.IGNORECASE,
)
_NATURAL_END_CHARS = '.!?)"\']}。！？）】」』》^'
_MIN_INCOMPLETE_FINAL_CHARS = 80

ACTION_PREAMBLE_RECOVERY_PROMPT = (
    "Your previous assistant response announced the next action but did not "
    "include the tool call needed to perform it. Continue the same user request "
    "now by making the tool call immediately. Do not summarize or ask the user "
    "to type continue."
)


def looks_like_action_preamble_stall(
    content: str,
    finish_reason: str | None,
    has_tool_calls: bool,
    *,
    max_chars: int = 400,
) -> bool:
    """Return true for a no-tool response that reads like unfinished work."""
    if has_tool_calls or finish_reason not in {"stop", "length"}:
        return False

    text = _visible_text(content)
    if not text or _COMPLETION_RE.search(text):
        return False

    has_action_preamble = bool(_ACTION_RE.search(text))
    long_open_action = (
        len(text) > max_chars
        and len(text) <= max(max_chars * 2, 800)
        and has_action_preamble
        and text.endswith(":")
    )
    if len(text) > max_chars and not long_open_action:
        return False
    if has_action_preamble:
        return True
    if text.endswith(":"):
        return True
    if len(text) >= _MIN_INCOMPLETE_FINAL_CHARS and not _has_natural_ending(text):
        return True
    return False


def _visible_text(content: str) -> str:
    text = (content or "").strip()
    return re.sub(
        r"^<think>.*?</think>\s*",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def _has_natural_ending(content: str) -> bool:
    stripped = (content or "").rstrip()
    if not stripped:
        return False
    if stripped.endswith("```"):
        return True
    last = stripped[-1]
    if last in _NATURAL_END_CHARS:
        return True
    return ord(last) >= 0x1F300
