"""Bounded reset hints for sanitized provider failure replies."""

import re
from typing import Optional


# Captures the reset hint from a provider usage/session-limit envelope, e.g.
# "… session limit · resets 5:20pm (America/New_York)" → "resets 5:20pm
# (America/New_York)". Stops at line breaks and sentence-ending periods so we
# never leak a long raw body into chat, but keeps dotted time tokens whole:
# a period survives when the next char continues the token ("p.m", URLs) or
# when it closes a single-letter abbreviation ("p.m. ET" — period preceded by
# a lone word char). Input is already secret-redacted at the call site.
_GATEWAY_RESET_HINT_RE = re.compile(
    r"reset[s]?\b(?:[^\n.]|\.(?![\s])(?!$)|(?<=\b\w)\.){0,48}", re.IGNORECASE
)


def extract_gateway_reset_hint(text: str) -> Optional[str]:
    """Extract a short, whole-token reset hint from a provider error body.

    The regex character budget can land mid-token; surfacing "resets 5:20 p"
    or a dangling "(America/New_Yo" reads as a bug in chat. Snap the slice
    back to the last whole token and drop any unbalanced trailing
    parenthetical before handing the hint to the user-facing message.
    """
    match = _GATEWAY_RESET_HINT_RE.search(text)
    if not match:
        return None
    hint = match.group(0)
    end = match.end()
    if (
        end < len(text)
        and not text[end].isspace()
        and text[end] != "."
        and hint
        and not hint[-1].isspace()
    ):
        # The budget ran out mid-token: keep only whole tokens.
        parts = hint.rsplit(None, 1)
        if len(parts) == 2:
            hint = parts[0]
    hint = hint.strip().rstrip(" .·-")
    while hint.count("(") > hint.count(")"):
        # A parenthetical opened inside the budget but never closed
        # (long timezone tails); an unbalanced "(" must not reach chat.
        hint = hint[: hint.rindex("(")].rstrip(" .·-")
    return hint or None
