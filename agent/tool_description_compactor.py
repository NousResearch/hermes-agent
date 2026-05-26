"""Conservative prose compaction for tool descriptions.

Goal: shrink verbose natural-language descriptions without changing tool names,
URLs, paths, inline code, CLI flags, or numeric limits. This is intentionally
lighter than full caveman persona rewriting: preserve meaning, only trim prose.
"""

from __future__ import annotations

import copy
import re
from typing import Any

_URL_RE = re.compile(r"https?://[^\s)]+")
_INLINE_CODE_RE = re.compile(r"`[^`]+`")
_PATH_RE = re.compile(
    r"(?:\./|\.\./|/|~\/|[A-Za-z]:\\)[\w\-./\\]+|[\w\-.]+/[\w\-./]+"
)
_FLAG_RE = re.compile(r"--?[A-Za-z0-9][A-Za-z0-9_-]*")

_FILLER_PATTERNS = [
    (re.compile(r"\bplease\b", re.IGNORECASE), ""),
    (re.compile(r"\b(?:simply|basically|carefully|helpfully|clearly|really|just)\b", re.IGNORECASE), ""),
    (re.compile(r"\b(?:you can|you may)\b", re.IGNORECASE), ""),
    (re.compile(r"\b(?:used to|use this to)\b", re.IGNORECASE), "to"),
    (re.compile(r"\b(?:allows you to)\b", re.IGNORECASE), "lets you"),
    (re.compile(r"\b(?:in order to)\b", re.IGNORECASE), "to"),
    (re.compile(r"\b(?:for the purpose of)\b", re.IGNORECASE), "for"),
    (re.compile(r"\b(?:that you can)\b", re.IGNORECASE), "that can"),
    (re.compile(r"\b(?:there is|there are)\b", re.IGNORECASE), ""),
    (re.compile(r"\b(?:it is important to note that)\b", re.IGNORECASE), ""),
    (re.compile(r"\b(?:keep in mind that)\b", re.IGNORECASE), ""),
    (re.compile(r"\b(?:note that)\b", re.IGNORECASE), ""),
    (re.compile(r"\b(?:in the current page|on the current page)\b", re.IGNORECASE), "on the page"),
    (re.compile(r"\b(?:identified by its)\b", re.IGNORECASE), "by its"),
    (re.compile(r"\b(?:requires browser_navigate and browser_snapshot to be called first)\b", re.IGNORECASE), "Requires browser_navigate and browser_snapshot first"),
    (re.compile(r"\b(?:requires browser_navigate to be called first)\b", re.IGNORECASE), "Requires browser_navigate first"),
]

_SENTENCE_CLEANUPS = [
    (re.compile(r"\s+,", re.IGNORECASE), ","),
    (re.compile(r"\s+\.", re.IGNORECASE), "."),
    (re.compile(r"\(\s+", re.IGNORECASE), "("),
    (re.compile(r"\s+\)", re.IGNORECASE), ")"),
    (re.compile(r"\s{2,}"), " "),
]


def _protect(text: str) -> tuple[str, dict[str, str]]:
    protected: dict[str, str] = {}
    counter = 0

    def _sub(pattern: re.Pattern[str], src: str) -> str:
        nonlocal counter

        def repl(match: re.Match[str]) -> str:
            nonlocal counter
            key = f"__CPTK_{counter}__"
            counter += 1
            protected[key] = match.group(0)
            return key

        return pattern.sub(repl, src)

    out = text
    for pattern in (_URL_RE, _INLINE_CODE_RE, _PATH_RE, _FLAG_RE):
        out = _sub(pattern, out)
    return out, protected


def _restore(text: str, protected: dict[str, str]) -> str:
    out = text
    for key, value in protected.items():
        out = out.replace(key, value)
    return out


def compact_description(text: str) -> str:
    if not isinstance(text, str):
        return text
    original = text
    stripped = text.strip()
    if len(stripped) < 40:
        return original

    working, protected = _protect(original)

    for pattern, repl in _FILLER_PATTERNS:
        working = pattern.sub(repl, working)

    # Tighten a few verbose constructions while keeping meaning stable.
    working = re.sub(r"\b[Aa]sk the user a question when you need\b", "Ask the user when you need", working)
    working = re.sub(r"\bReturns up to ([0-9]+) results by default\b", r"Returns up to \1 results", working)
    working = re.sub(r"\bOptional:?\s+", "", working)
    working = re.sub(r"\bDefault:?\s+", "", working)

    for pattern, repl in _SENTENCE_CLEANUPS:
        working = pattern.sub(repl, working)

    # Clean repeated punctuation/space artifacts from aggressive substitutions.
    working = re.sub(r"\s*;\s*;", ";", working)
    working = re.sub(r"\s*\.\s*\.\s*", ". ", working)
    working = re.sub(r"\s+([:;!?])", r"\1", working)
    working = re.sub(r"([:;!?])(\w)", r"\1 \2", working)
    working = working.strip()

    out = _restore(working, protected)
    out = re.sub(r"\s{2,}", " ", out).strip()

    # Safety: never return empty, never expand a lot, and require a small win.
    if not out:
        return original
    if len(out) >= len(original) - 4:
        return original
    if len(out) < max(16, int(len(original) * 0.45)):
        # Too aggressive for a conservative v1.
        return original
    return out


def compact_tool_definitions(tool_defs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = copy.deepcopy(tool_defs)
    for tool in out:
        fn = tool.get("function")
        if not isinstance(fn, dict):
            continue
        desc = fn.get("description")
        if isinstance(desc, str) and desc:
            fn["description"] = compact_description(desc)
    return out
