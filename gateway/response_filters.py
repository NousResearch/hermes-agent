"""Gateway response filtering helpers.

These helpers operate at the gateway boundary: they decide whether a completed
agent turn should be delivered to the chat, not what should be persisted in the
conversation history.
"""

from __future__ import annotations

import unicodedata
from typing import Any

# Canonical model-emitted control token for intentional silence.
SILENT_REPLY_TOKEN = "NO_REPLY"

# Exact whole-response markers that mean "the agent intentionally chose not to
# reply".  Keep this list small and explicit; arbitrary empty output remains an
# error/empty-response path, not silence.
LIVE_GATEWAY_SILENT_MARKERS = frozenset({
    "[SILENT]",
    "SILENT",
    "NO_REPLY",
    "NO REPLY",
})


def _canonical_silence_candidate(text: str) -> str:
    return " ".join(text.strip().upper().split())


def _strip_edge_punctuation(s: str) -> str:
    """Strip leading/trailing Unicode punctuation characters only.

    Preserves emoji and alphanumeric content to avoid false positives on
    prose like ``😄 NO_REPLY``.
    """
    # Strip leading punctuation
    i = 0
    while i < len(s) and unicodedata.category(s[i]).startswith("P"):
        i += 1
    s = s[i:]
    # Strip trailing punctuation
    j = len(s)
    while j > 0 and unicodedata.category(s[j - 1]).startswith("P"):
        j -= 1
    return s[:j]


def is_intentional_silence_response(response: Any) -> bool:
    """Return True only when ``response`` is exactly a silence marker.

    Substantive prose that merely mentions ``NO_REPLY`` or ``[SILENT]`` must be
    delivered normally.  A blank response is also not silence; blank output is
    handled by the empty-response failure path.

    Stray edge punctuation (``.``, ``*``, ``,`` etc.) around an otherwise bare
    marker is tolerated so that model formatting noise doesn't leak control
    tokens into user-visible chat surfaces (#58469).
    """
    if not isinstance(response, str):
        return False
    stripped = response.strip()
    if not stripped:
        return False
    if len(stripped) > 64:
        return False
    if _canonical_silence_candidate(stripped) in LIVE_GATEWAY_SILENT_MARKERS:
        return True
    # Tolerate edge punctuation around bare markers: ``.NO_REPLY``,
    # ``*[SILENT]*``, `` .NO_REPLY `` etc.
    depunct = _strip_edge_punctuation(stripped)
    if depunct and depunct != stripped:
        return _canonical_silence_candidate(depunct) in LIVE_GATEWAY_SILENT_MARKERS
    return False


def is_intentional_silence_agent_result(agent_result: dict | None, response: Any) -> bool:
    """Silence markers suppress delivery only for successful agent turns."""
    if not isinstance(agent_result, dict):
        return False
    if agent_result.get("failed"):
        return False
    return is_intentional_silence_response(response)


def is_partial_silence_marker(text: Any) -> bool:
    """Return True while ``text`` could still resolve to a silence marker.

    The streaming path accumulates the reply delta-by-delta and must decide,
    before the whole response is known, whether to show what it has so far.
    A buffer whose canonical form is a non-empty *prefix* of a silence marker
    (e.g. ``"NO"`` on the way to ``"NO_REPLY"``, or an exact marker that has
    not yet been terminated by stream-end) is held back so a raw marker is
    never edited onto the screen and then belatedly retracted.

    Anything that has already diverged from every marker (ordinary prose) —
    and anything longer than the marker cap — returns False so normal
    streaming resumes immediately.  This is the streaming counterpart to
    :func:`is_intentional_silence_response`, sharing the same marker set and
    canonicalization so the two never drift.
    """
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > 64:
        return False
    candidate = _canonical_silence_candidate(stripped)
    if not candidate:
        return False
    return any(marker.startswith(candidate) for marker in LIVE_GATEWAY_SILENT_MARKERS)
