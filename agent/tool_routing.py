"""Conservative task-scoped toolset selection.

The router only removes optional toolsets when the current request has a clear
intent. Ambiguous follow-ups retain every safe configured toolset so an ongoing
task cannot lose capabilities. Browser is always denied because this deployment
uses browser-harness through the terminal instead of native browser schemas.
"""

from __future__ import annotations

import re
from typing import Iterable


SAFE_DENYLIST = frozenset({"browser"})
CORE_TOOLSETS = frozenset({
    "clarify", "code_execution", "file", "jev", "memory", "messaging",
    "session_search", "skills", "terminal", "todo",
})

_INTENT_PATTERNS = {
    "web": re.compile(
        r"\b(news|weather|price|(?:latest|current).{0,24}(?:versions?|releases?|news)|"
        r"search(?: the)? web|look up online|website|url|http[s]?)\b", re.I
    ),
    "homeassistant": re.compile(
        r"\b(home assistant|smart home|thermostat|lights?|garage door|climate|"
        r"temperature sensor)\b", re.I
    ),
    "spotify": re.compile(r"\b(spotify|play music|playlist|queue (?:a )?song)\b", re.I),
    "cronjob": re.compile(r"\b(remind|schedule|cron|every day|every week|tomorrow at)\b", re.I),
    "delegation": re.compile(r"\b(delegate|subagent|parallel agents?|swarm)\b", re.I),
    "image_gen": re.compile(r"\b(generate|create|draw|make)\b.{0,24}\b(image|picture|art|logo)\b", re.I),
    "vision": re.compile(r"\b(image|picture|photo|screenshot|diagram)\b", re.I),
    "tts": re.compile(r"\b(text[- ]to[- ]speech|tts|voice memo|speak this|audio narration)\b", re.I),
}

# These utterances do not carry enough task intent by themselves. Keep all safe
# capabilities unless recent user history provides a classifiable request.
_AMBIGUOUS = re.compile(
    r"^\s*(resume|continue|go on|do it|try again|yes|no|ok(?:ay)?|thanks?|check it|fix it)\s*[.!?]*\s*$",
    re.I,
)
_LOCAL_ACTION = re.compile(
    r"\b(code|python|javascript|typescript|test|repo(?:sitory)?|file|folder|"
    r"debug|implement|refactor|build|terminal|command|config|yaml|json|git)\b", re.I
)


def _recent_user_text(history: Iterable[dict] | None) -> str:
    if not history:
        return ""
    texts = []
    for msg in list(history)[-8:]:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                texts.append(content)
    return "\n".join(texts[-2:])


def select_task_toolsets(
    configured_toolsets: Iterable[str],
    user_message: str,
    *,
    history: Iterable[dict] | None = None,
    enabled: bool = True,
) -> set[str]:
    """Return the smallest conservative subset for this turn."""
    available = {str(item) for item in configured_toolsets} - SAFE_DENYLIST
    if not enabled:
        return available

    current = str(user_message or "")
    prior = _recent_user_text(history)
    if _AMBIGUOUS.match(current) and not prior:
        return available

    text = f"{prior}\n{current}" if prior else current
    matched = {name for name, pattern in _INTENT_PATTERNS.items() if pattern.search(text)}
    local_action = bool(_LOCAL_ACTION.search(text))

    # Unknown intent is deliberately a full-safe fallback. False negatives cost
    # an entire failed turn, while false positives only cost schema tokens.
    if not matched and not local_action:
        return available

    selected = (CORE_TOOLSETS | matched) & available
    # Image generation commonly needs vision for inspection of the result.
    if "image_gen" in selected and "vision" in available:
        selected.add("vision")
    return selected
