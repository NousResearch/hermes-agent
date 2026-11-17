"""Sanitize user prompt text leaked from terminal / paste control sequences."""

from __future__ import annotations

import re

# Numeric ranges and degraded paste markers share the same spelling (#143444).
# Keep ambiguous number-adjacent forms; canonical escape wrappers are still removed.
_NO_NUMBER_AFTER = r"(?![ \t]*[0-9])"
_NO_NUMBER_BEFORE = r"(?<![0-9])(?<![0-9][-./:,])"
# A label may end in a colon; a digit, whitespace or line start may introduce minutes.
_DIGIT_OPENER_BOUNDARY = r"(^|[\s>\]\)]|(?<=[^\s0-9]):)"

# Degraded visible bracketed-paste forms, matched only at boundaries so embedded literals stay intact.
_BOUNDARY_SUBS = (
    (re.compile(r"(^|[\s\n>:\]\)])\[200~" + _NO_NUMBER_AFTER), r"\1"),
    (re.compile(r"\[201~(?=$|[\s\n<\[\(\):;.,!?])" + _NO_NUMBER_AFTER), ""),
    (re.compile(_DIGIT_OPENER_BOUNDARY + r"00~" + _NO_NUMBER_AFTER), r"\1"),
    (
        re.compile(
            _NO_NUMBER_BEFORE + r"01~(?=$|[\s\n<\[\(\):;.,!?])" + _NO_NUMBER_AFTER
        ),
        "",
    ),
)

# Corruption signature from desktop bracketed-paste leaks (#62557).
_DESKTOP_PASTE_ARTIFACT = "~[[e"


def strip_leaked_bracketed_paste_wrappers(text: str) -> str:
    """Strip leaked bracketed-paste wrapper markers: canonical wrappers unconditionally, degraded visible forms
    (``[200~`` / ``[201~`` and ``00~`` / ``01~``) only at nonnumeric boundaries so user-authored ranges stay intact."""
    if not text:
        return text
    for wrapper in ("\x1b[200~", "\x1b[201~", "^[[200~", "^[[201~"):
        text = text.replace(wrapper, "")
    for pattern, repl in _BOUNDARY_SUBS:
        text = pattern.sub(repl, text)
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
    if index >= 2 and text[index - 2 : index] == "[e":
        index -= 2
    elif index >= 1 and text[index - 1] == "[":
        index -= 1
    return text[:index]


def sanitize_user_prompt_text(text: str) -> str:
    """Normalize user-authored prompt text before persistence or model input."""
    if not isinstance(text, str) or not text:
        return text
    return collapse_repeated_input_artifacts(
        strip_leaked_bracketed_paste_wrappers(text)
    )
