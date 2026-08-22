"""Detect whether the user corrected the agent in a follow-up turn.

Heuristic-only signal: looks at the user's next message(s) within a
window after the skill ran. Triggers on correction-shaped phrases in
multiple languages. Does **not** require NLU — this is a cheap regex
pre-filter; downstream consumers can refine with an LLM judge if they
want a more accurate signal.

The detector is intentionally permissive (false positives are cheaper
than false negatives for a reward-hacking-style signal where the agent
might try to game the score). The reward-hack detection in
``ExperienceLedger._compute_summary`` already guards against systematic
mismatches between public and private scores.

Design:
- Pure function: ``detect(user_messages: list[str]) -> bool``.
- No I/O, no clock dependency — easy to unit-test.
- Configurable phrase lists so maintainers can extend per language.

Usage::

    from agent.signal_sources.user_correction_detector import detect
    corrected = detect([msg1.text, msg2.text, ...])
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence


# Correction-shaped phrases per language. Lower-case, matched against the
# normalized (lower-cased + whitespace-collapsed) user message.
#
# Phrases are anchored to the start of the message where natural, but not
# strictly — a phrase anywhere in the message indicates a correction in
# conversational contexts.
DEFAULT_CORRECTION_PATTERNS: List[str] = [
    # English
    "no,",
    "no.",
    "that's wrong",
    "that is wrong",
    "that's not",
    "that is not",
    "you're wrong",
    "you are wrong",
    "wrong",
    "incorrect",
    "redo",
    "redo it",
    "try again",
    "do it again",
    "not quite",
    "not right",
    "fix this",
    "fix it",
    "fix that",
    "that's incorrect",
    "that is incorrect",
    "this is wrong",
    "this isn't right",
    "this is not right",
    "wrong answer",
    "bad answer",
    "not what i",
    "not what i wanted",
    "not what i asked",
    # Chinese
    "不对",
    "错了",
    "错了",
    "不对的",
    "不对啊",
    "重新",
    "再试",
    "再试一次",
    "重新做",
    "再写",
    "再写一次",
    "改一下",
    "改改",
    "修改一下",
    "改成",
    "不是这个",
    "不是这样",
    "不是要",
    "不是要这个",
    "应该是",
    "应该不是",
    # Spanish
    "no,",
    "no.",
    "está mal",
    "esta mal",
    "incorrecto",
    "incorrecta",
    "vuelve a",
    "hazlo de nuevo",
    "otra vez",
    # French
    "non,",
    "c'est faux",
    "c est faux",
    "faux",
    "incorrect",
    "refais",
    "refaire",
    "encore",
    # German
    "falsch",
    "nein,",
    "nein.",
    "nochmal",
    "noch einmal",
    "korrigier",
    "korrigieren",
]


# Pre-compile patterns at import. Word boundaries for Latin scripts; for
# CJK we match on the substring directly since word boundaries don't
# apply.
_UNSET = object()  # sentinel distinguishing "no patterns set" from "empty list"
_compiled_state: List[re.Pattern[str]] | object = _UNSET


def _compile_patterns(patterns: Sequence[str]) -> List[re.Pattern[str]]:
    out: List[re.Pattern[str]] = []
    for p in patterns:
        # Latin phrases get word-boundary anchors; CJK phrases match as-is.
        if re.search(r"[\u4e00-\u9fff]", p):
            out.append(re.compile(re.escape(p), re.IGNORECASE))
        else:
            out.append(re.compile(r"\b" + re.escape(p) + r"\b", re.IGNORECASE))
    return out


def _default_compiled() -> List[re.Pattern[str]]:
    return _compile_patterns(DEFAULT_CORRECTION_PATTERNS)


def _current_compiled() -> List[re.Pattern[str]]:
    global _compiled_state
    if _compiled_state is _UNSET:
        _compiled_state = _default_compiled()
    # _compiled_state is now a concrete list (or an empty list after
    # reset_patterns([])). The ``is _UNSET`` check above guarantees it.
    return _compiled_state  # type: ignore[return-value]


def reset_patterns(patterns: Sequence[str]) -> None:
    """Replace the default patterns (e.g. for tests).

    Pass an empty list to disable detection (every input returns
    ``False``). Pass a non-empty list to install custom patterns.
    Pass a non-empty list of ``DEFAULT_CORRECTION_PATTERNS`` to
    restore the shipped defaults.
    """
    global _compiled_state
    if patterns:
        _compiled_state = _compile_patterns(patterns)
    else:
        # Empty list = explicitly disabled. UNSET means "use defaults".
        _compiled_state = []


def reset_to_defaults() -> None:
    """Forget any prior ``reset_patterns`` call so the next ``detect``
    call reloads the shipped default patterns.

    Tests should call this in their teardown to ensure no test leaks
    state into the next.
    """
    global _compiled_state
    _compiled_state = _UNSET


def detect(
    user_messages: Iterable[str],
    *,
    patterns: Sequence[str] | None = None,
) -> bool:
    """Return True if any user message looks like a correction.

    Args:
        user_messages: Iterable of user message texts. Typically the next
            N user messages after the skill ran; the caller decides the
            window size.
        patterns: Override the default correction patterns. ``None``
            means use the module default. Empty list means "no detection".

    Returns:
        True if any message contains a correction phrase; False otherwise.
        An empty input always returns False.
    """
    if patterns is None:
        compiled = _current_compiled()
    else:
        compiled = _compile_patterns(patterns) if patterns else []

    if not compiled:
        return False

    for raw in user_messages:
        if not raw:
            continue
        text = raw.strip().lower()
        if not text:
            continue
        # Collapse runs of whitespace so multi-line corrections match.
        text = re.sub(r"\s+", " ", text)
        for pat in compiled:
            if pat.search(text):
                return True
    return False


__all__ = [
    "DEFAULT_CORRECTION_PATTERNS",
    "detect",
    "reset_patterns",
    "reset_to_defaults",
]
