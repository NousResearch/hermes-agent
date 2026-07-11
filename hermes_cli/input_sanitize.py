"""Sanitize user prompt text leaked from terminal / paste control sequences."""

from __future__ import annotations

import re

_BRACKETED_PASTE_BOUNDARY_START = re.compile(r"(^|[\s\n>:\]\)])\[200~")
_BRACKETED_PASTE_BOUNDARY_END = re.compile(r"\[201~(?=$|[\s\n<\[\(\):;.,!?])")
_BRACKETED_PASTE_DEGRADED_START = re.compile(r"(^|[\s\n>:\]\)])00~")
_BRACKETED_PASTE_DEGRADED_END = re.compile(r"01~(?=$|[\s\n<\[\(\):;.,!?])")

# Corruption signature from desktop bracketed-paste leaks (#62557).
_DESKTOP_PASTE_ARTIFACT = "~[[e"


def strip_leaked_bracketed_paste_wrappers(text: str) -> str:
    """Strip leaked bracketed-paste wrapper markers from user-visible text.

    Defensive normalization for cases where terminal/prompt_toolkit parsing
    fails and bracketed-paste markers end up in the buffer as literal text.

    Canonical wrappers are stripped unconditionally. Degraded visible forms like
    ``[200~`` / ``[201~`` and ``00~`` / ``01~`` are removed only at boundaries
    so embedded literals such as ``literal[200~tag`` stay intact.
    """
    if not text:
        return text

    text = (
        text.replace("\x1b[200~", "")
        .replace("\x1b[201~", "")
        .replace("^[[200~", "")
        .replace("^[[201~", "")
    )
    text = _BRACKETED_PASTE_BOUNDARY_START.sub(r"\1", text)
    text = _BRACKETED_PASTE_BOUNDARY_END.sub("", text)
    text = _BRACKETED_PASTE_DEGRADED_START.sub(r"\1", text)
    text = _BRACKETED_PASTE_DEGRADED_END.sub("", text)
    return text


def collapse_repeated_input_artifacts(text: str, min_repeats: int = 4) -> str:
    """Drop a trailing run of the desktop ~[[e corruption signature (#62557)."""
    if not text:
        return text

    marker = _DESKTOP_PASTE_ARTIFACT
    index = len(text)
    repeat_count = 0
    while index >= len(marker) and text[index - len(marker) : index] == marker:
        repeat_count += 1
        index -= len(marker)

    if repeat_count < min_repeats:
        return text

    start = index
    if start >= 2 and text[start - 2 : start] == "[e":
        start -= 2
    elif start >= 1 and text[start - 1] == "[":
        start -= 1
    return text[:start]


def sanitize_user_prompt_text(text: str) -> str:
    """Normalize user-authored prompt text before persistence or model input."""
    if not isinstance(text, str) or not text:
        return text
    cleaned = strip_leaked_bracketed_paste_wrappers(text)
    return collapse_repeated_input_artifacts(cleaned)
