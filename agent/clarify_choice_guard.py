"""Hard gate for Markdown-only choice prompts that should use clarify.

The model already receives prompt/schema guidance telling it to call the
``clarify`` tool for final reports and next-action sections that ask the user
to choose.  This module is the deterministic backstop: if a final text answer
still contains an inert Markdown choice menu while the live session exposes the
clarify tool, the conversation loop nudges once for an actual tool call and then
fails closed instead of delivering dead choices.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple

_NONINTERACTIVE_PLATFORMS = {
    "batch",
    "cron",
    "curator",
    "no_agent",
}

_CHOICE_HEADING_RE = re.compile(
    r"(?im)^\s{0,3}#{1,6}\s*(?:"
    r"다음\s*추천\s*작업|"
    r"다음\s*작업|"
    r"승인(?:\s*필요|\s*게이트)?|"
    r"선택(?:\s*필요|\s*항목|\s*지)?|"
    r"범위(?:\s*선택)?|"
    r"next\s+(?:recommended\s+)?(?:actions?|steps?)|"
    r"recommended\s+next\s+(?:actions?|steps?)|"
    r"approval(?:\s+gate|\s+required)?|"
    r"choices?|"
    r"scope(?:\s+selection)?"
    r")\b.*$"
)

_ANY_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+\S")

_CHOICE_LINE_RE = re.compile(
    r"(?m)^\s{0,6}(?:[-*+]\s+|\d{1,2}[.)]\s+|[A-Ha-h][.)]\s+).+\S"
)

_SELECT_FENCE_RE = re.compile(r"(?im)^\s*```\s*select\b")

_APPROVAL_KEYWORDS = (
    "승인 실행",
    "승인",
    "수정",
    "보류",
    "금지",
    "approve",
    "modify",
    "defer",
    "forbid",
    "hold",
)


def _has_clarify(valid_tool_names: Iterable[str] | None) -> bool:
    return "clarify" in set(valid_tool_names or ())


def _is_interactive_platform(platform: str | None) -> bool:
    normalized = (platform or "").strip().lower()
    return normalized not in _NONINTERACTIVE_PLATFORMS


def _choice_sections(text: str) -> list[str]:
    sections: list[str] = []
    for match in _CHOICE_HEADING_RE.finditer(text or ""):
        start = match.start()
        next_heading = _ANY_HEADING_RE.search(text, match.end())
        end = next_heading.start() if next_heading else len(text)
        sections.append(text[start:end])
    return sections


def _looks_like_choice_menu(text: str) -> bool:
    if not text:
        return False
    if _SELECT_FENCE_RE.search(text):
        return True

    sections = _choice_sections(text)
    if not sections:
        return False

    for section in sections:
        choice_lines = _CHOICE_LINE_RE.findall(section)
        if len(choice_lines) >= 2:
            return True

        lowered = section.casefold()
        keyword_hits = sum(
            1 for keyword in _APPROVAL_KEYWORDS if keyword.casefold() in lowered
        )
        if keyword_hits >= 2:
            return True

    return False


def clarify_choice_guard_action(
    final_response: str,
    *,
    valid_tool_names: Iterable[str] | None,
    platform: str | None,
    attempts: int,
    max_attempts: int = 1,
) -> Tuple[str, Optional[str]]:
    """Return (action, payload) for the final-response clarify hard gate.

    Actions:
    - ``("allow", None)``: deliver the final text normally.
    - ``("nudge", message)``: append a synthetic user message and continue so the
      model can call the actual ``clarify`` tool.
    - ``("block", message)``: retry budget exhausted; replace the dead menu with
      a safe failure notice that does not ask the user to click inert Markdown.
    """
    if not _has_clarify(valid_tool_names):
        return "allow", None
    if not _is_interactive_platform(platform):
        return "allow", None
    if not _looks_like_choice_menu(final_response or ""):
        return "allow", None

    if attempts < max_attempts:
        return "nudge", (
            "[System: Your previous answer ended with Markdown-only next-action, "
            "approval, scope, or fenced select choices. That is not clickable UI. "
            "Do not send that final text yet. Call the `clarify` tool now, put "
            "selectable rows only in the `choices` array, and use "
            "`multi_select=true` when multiple choices can be selected. If no "
            "choice is genuinely required, answer again without a Markdown choice "
            "menu.]"
        )

    return "block", (
        "Clarify hard gate blocked this response: the model produced "
        "Markdown-only choice options after being told to call the `clarify` "
        "tool. No clickable choice UI was shown, and no action was selected. "
        "Please retry the request or ask for a plain recommendation without a "
        "choice menu."
    )


__all__ = ["clarify_choice_guard_action"]
