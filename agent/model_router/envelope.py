"""Turn-envelope classifier — port of pi-smart-router triage/turn-envelope.ts.

Derives turn_type from the message envelope using deterministic heuristics
(no neural inference). Classification priority (first match wins):

  1. tool_result — last message is role=tool
  2. planning    — planning/architecture signals in recent content
  3. subagent    — subagent/exploration context markers
  4. main_loop   — default agent loop turn (messages present)
  5. unknown     — no messages or empty envelope
"""
from __future__ import annotations

import re

from .types import (
    Message,
    TURN_MAIN_LOOP,
    TURN_PLANNING,
    TURN_SUBAGENT,
    TURN_TOOL_RESULT,
    TURN_UNKNOWN,
)

TOOL_RESULT_SIZE_THRESHOLD = 50_000

PLANNING_PATTERNS = tuple(
    re.compile(p, re.I | re.M)
    for p in (
        r"\b(?:plan|planning|architect(?:ure)?|design|refactor|migration)\b",
        r"\b(?:step\s*\d|phase\s*\d|breakdown|strategy|trade-?off)\b",
        r"^#+\s*(?:plan|design|architecture)",
        # Repo-hygiene / destructive-intent (SP-176) — escalate off main_loop
        r"\b(?:clean\s*up(?:\s+the)?\s+repo|repo\s+cleanup|cleanup\s+the\s+repo)\b",
        r"\b(?:clean\s*up|cleanup|mistakenly\s+added|accidentally\s+added|accidental\s+add)\b",
        r"\b(?:unstage|git\s+rm|rm\s+-rf|force\s+push|git\s+reset\s+--hard|destructive)\b",
    )
)

SUBAGENT_PATTERNS = tuple(
    re.compile(p, re.I)
    for p in (
        r"\b(?:subagent|sub-agent|exploration|explore|search|investigate)\b",
        r"\b(?:spawned|delegat(?:e|ed|ing)|parallel\s+agent)\b",
        r"\b(?:Task|Agent)\.(?:create|spawn|launch)\b",
    )
)


def classify_turn_envelope(messages) -> str:
    if not messages:
        return TURN_UNKNOWN
    last = messages[-1]
    if _is_tool_result(last):
        return TURN_TOOL_RESULT
    window = messages[-3:]
    if _matches_any(window, PLANNING_PATTERNS):
        return TURN_PLANNING
    if _matches_any(window, SUBAGENT_PATTERNS):
        return TURN_SUBAGENT
    return TURN_MAIN_LOOP


def _is_tool_result(message: Message) -> bool:
    if getattr(message, "role", None) != "tool":
        return False
    return len(getattr(message, "content", "") or "") <= TOOL_RESULT_SIZE_THRESHOLD


def _matches_any(messages, patterns) -> bool:
    for msg in messages:
        content = getattr(msg, "content", "") or ""
        for pattern in patterns:
            if pattern.search(content):
                return True
    return False
