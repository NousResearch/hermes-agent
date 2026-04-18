"""Importance-inference heuristics for sync_turn.

Decides whether a turn is worth logging to episodic memory and at what
importance. Cheap, regex-based. Will mis-score occasionally; that's the
tradeoff of staying outside an LLM call per turn.
"""

from __future__ import annotations

import re

_CORRECTION = re.compile(
    r"\b(no[,.]|actually|wrong|instead|not quite|incorrect|that's not|stop|"
    r"don't|doesn't work|not what i)\b",
    re.I,
)

# Strong failure signals: always indicate a failure if present.
_FAILURE_STRONG = re.compile(
    r"\b(traceback|failed|broken|exception|cannot|couldn't|timed out|"
    r"permission denied|not found|invalid)\b",
    re.I,
)

# "error" as a bare word. Requires a second check to exclude idioms like
# "trial and error" (Hermes's own post-turn skill-review nudge uses this
# phrase and was generating false-positive failure entries).
_ERROR_WORD = re.compile(r"\berror\b", re.I)
_TRIAL_AND_ERROR_IDIOM = re.compile(r"trial[-\s]and[-\s]error", re.I)

_SUCCESS = re.compile(r"\b(done|works|fixed|complete|shipped|merged|deployed)\b", re.I)

_CHITCHAT = re.compile(
    r"^(hi|hey|hello|thanks|thank you|ok|cool|nice|got it|good|sure|yes|no)\b",
    re.I,
)


def _has_failure(text: str) -> bool:
    """True if ``text`` contains a failure signal, discounting idioms.

    "trial and error" and its hyphenated forms must not count - those
    appear in Hermes's post-turn skill-review prompts and in responses
    explaining that no trial-and-error occurred.
    """
    if not text:
        return False
    if _FAILURE_STRONG.search(text):
        return True
    error_matches = list(_ERROR_WORD.finditer(text))
    if not error_matches:
        return False
    idiom_spans = [(m.start(), m.end()) for m in _TRIAL_AND_ERROR_IDIOM.finditer(text)]
    for m in error_matches:
        pos = m.start()
        in_idiom = any(s <= pos < e for s, e in idiom_spans)
        if not in_idiom:
            return True
    return False


def infer_importance(user: str, assistant: str) -> int:
    """Score 1-10. Below threshold in config, the turn is not logged."""
    u = (user or "")[:600]
    a = (assistant or "")[:2000]
    score = 4
    if _CORRECTION.search(u):
        score = max(score, 7)
    if _has_failure(a):
        score = max(score, 6)
    if _SUCCESS.search(a):
        score = max(score, 5)
    if len(a) > 1500:
        score = max(score, 5)
    if len(a) > 4000:
        score = max(score, 6)
    if len(u.strip()) < 25 and _CHITCHAT.match(u.strip()):
        score = min(score, 2)
    if len(u.strip()) < 40 and len(a.strip()) < 180:
        score = min(score, 3)
    return max(1, min(score, 10))


def infer_success(assistant: str) -> bool:
    """Heuristic: if the assistant's early text signals a failure, record as failure."""
    head = (assistant or "")[:600]
    return not _has_failure(head)


def infer_skill(platform: str, agent_context: str) -> str:
    """Best-effort skill name for the log entry.

    Falls back to a generic bucket since the plugin doesn't know which
    in-session skill actually fired. Cron and subagent writes get their
    own bucket so the dream cycle can cluster them apart.
    """
    if agent_context == "cron":
        return "cron"
    if agent_context == "subagent":
        return "subagent"
    if platform == "telegram":
        return "telegram-chat"
    if platform == "discord":
        return "discord-chat"
    return "chat"
