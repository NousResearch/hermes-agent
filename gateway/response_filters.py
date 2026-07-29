"""Gateway response filtering helpers.

These helpers operate at the gateway boundary: they decide whether a completed
agent turn should be delivered to the chat, not what should be persisted in the
conversation history.
"""

from __future__ import annotations

import json
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

_MAX_SILENCE_MARKER_LENGTH = 64
_MAX_JSON_SILENCE_ENVELOPE_LENGTH = 256
_JSON_WHITESPACE = frozenset(" \t\r\n")


def _canonical_silence_candidate(text: str) -> str:
    return " ".join(text.strip().upper().split())


def _strip_edge_silence_punctuation(text: str) -> str:
    """Strip stray edge punctuation without erasing marker structure.

    Models sometimes emit ``.NO_REPLY`` or ``*NO_REPLY*`` instead of the exact
    marker. Keep square brackets structural so malformed ``[SILENT`` does not
    become ``SILENT``.
    """
    start = 0
    end = len(text)
    while start < end and text[start] not in "[]" and unicodedata.category(text[start]).startswith("P"):
        start += 1
    while end > start and text[end - 1] not in "[]" and unicodedata.category(text[end - 1]).startswith("P"):
        end -= 1
    return text[start:end].strip()


def _canonical_silence_candidates(text: str) -> tuple[str, ...]:
    exact = _canonical_silence_candidate(text)
    stripped = _strip_edge_silence_punctuation(text.strip())
    if stripped == text.strip():
        return (exact,)
    fallback = _canonical_silence_candidate(stripped)
    return (exact, fallback)


def _is_json_silence_envelope(text: str) -> bool:
    stripped = text.strip(" \t\r\n")
    if (
        not stripped.startswith("{")
        or not stripped.endswith("}")
        or len(text) > _MAX_JSON_SILENCE_ENVELOPE_LENGTH
    ):
        return False
    try:
        pairs = json.loads(stripped, object_pairs_hook=lambda items: items)
    except (TypeError, ValueError):
        return False
    return pairs == [("action", SILENT_REPLY_TOKEN)]


def _skip_json_whitespace(text: str, index: int) -> int:
    while index < len(text) and text[index] in _JSON_WHITESPACE:
        index += 1
    return index


def _match_json_string_prefix(
    text: str,
    index: int,
    expected: str,
) -> tuple[str, int]:
    """Match a JSON string that must decode to ``expected``."""
    if index >= len(text):
        return "partial", index
    if text[index] != '"':
        return "invalid", index
    index += 1
    expected_index = 0
    simple_escapes = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    while index < len(text):
        char = text[index]
        if char == '"':
            if expected_index != len(expected):
                return "invalid", index
            return "complete", index + 1
        if expected_index >= len(expected) or ord(char) < 0x20:
            return "invalid", index
        if char != "\\":
            if char != expected[expected_index]:
                return "invalid", index
            expected_index += 1
            index += 1
            continue

        if index + 1 >= len(text):
            return "partial", index
        escape = text[index + 1]
        if escape != "u":
            decoded = simple_escapes.get(escape)
            if decoded != expected[expected_index]:
                return "invalid", index
            expected_index += 1
            index += 2
            continue

        expected_hex = f"{ord(expected[expected_index]):04x}"
        available = text[index + 2 : index + 6]
        if any(char not in "0123456789abcdefABCDEF" for char in available):
            return "invalid", index
        if not expected_hex.startswith(available.lower()):
            return "invalid", index
        if len(available) < 4:
            return "partial", index
        expected_index += 1
        index += 6
    return "partial", index


def _could_be_json_silence_envelope(text: str) -> bool:
    """Return whether a bounded prefix can still become the exact envelope."""
    if len(text) > _MAX_JSON_SILENCE_ENVELOPE_LENGTH:
        return False
    index = _skip_json_whitespace(text, 0)
    if index >= len(text) or text[index] != "{":
        return False
    index = _skip_json_whitespace(text, index + 1)
    if index >= len(text):
        return True

    state, index = _match_json_string_prefix(text, index, "action")
    if state == "partial":
        return True
    if state == "invalid":
        return False
    index = _skip_json_whitespace(text, index)
    if index >= len(text):
        return True
    if text[index] != ":":
        return False
    index = _skip_json_whitespace(text, index + 1)
    if index >= len(text):
        return True

    state, index = _match_json_string_prefix(text, index, SILENT_REPLY_TOKEN)
    if state == "partial":
        return True
    if state == "invalid":
        return False
    index = _skip_json_whitespace(text, index)
    if index >= len(text):
        return True
    if text[index] != "}":
        return False
    return _skip_json_whitespace(text, index + 1) == len(text)


def is_intentional_silence_response(response: Any) -> bool:
    """Return True only when ``response`` is exactly a silence marker.

    Substantive prose that merely mentions ``NO_REPLY`` or ``[SILENT]`` must be
    delivered normally.  A blank response is also not silence; blank output is
    handled by the empty-response failure path.
    """
    if not isinstance(response, str):
        return False
    stripped = response.strip()
    if not stripped:
        return False
    if _is_json_silence_envelope(response):
        return True
    if len(stripped) > _MAX_SILENCE_MARKER_LENGTH:
        return False
    return any(
        candidate in LIVE_GATEWAY_SILENT_MARKERS
        for candidate in _canonical_silence_candidates(stripped)
    )


def is_autonomous_silence_response(response: Any) -> bool:
    """Loose silence matcher for autonomous lanes (cron, webhook).

    Autonomous lanes instruct the agent to emit ``[SILENT]`` when a tick
    produced nothing worth a human's attention, and models reliably bracket
    the marker with a short note explaining why they stayed quiet.  Unlike
    :func:`is_intentional_silence_response` (the interactive-chat rule, which
    demands the response be EXACTLY a marker), this suppresses when a marker
    is the whole response, sits on its own first or last line, or the
    bracketed sentinel opens the response (the documented
    ``[SILENT] No changes detected`` pattern).  A token buried mid-sentence
    in a genuine report is still delivered.

    Shares :data:`LIVE_GATEWAY_SILENT_MARKERS` so the interactive and
    autonomous marker sets can never drift apart.
    """
    if not isinstance(response, str):
        return False
    stripped = response.strip()
    if not stripped:
        return False

    if _is_json_silence_envelope(response):
        return True

    def _is_token(line: str) -> bool:
        return _canonical_silence_candidate(line) in LIVE_GATEWAY_SILENT_MARKERS

    # Whole response is exactly a token.
    if _is_token(stripped):
        return True
    # Marker on its own first or last line (leading/trailing note on a
    # separate line — e.g. "2 deals filtered\n\n[SILENT]").
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    if lines and (_is_token(lines[0]) or _is_token(lines[-1])):
        return True
    # Bracketed sentinel used as a same-line prefix — the documented pattern
    # "[SILENT] No changes detected".  Restricted to the bracketed form so a
    # bare word like "Silent retry succeeded" is NOT swallowed.
    if stripped.upper().startswith("[SILENT]"):
        return True
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
    if not stripped:
        return False
    if _could_be_json_silence_envelope(text):
        return True
    if len(stripped) > _MAX_SILENCE_MARKER_LENGTH:
        return False
    for candidate in _canonical_silence_candidates(stripped):
        if candidate and any(marker.startswith(candidate) for marker in LIVE_GATEWAY_SILENT_MARKERS):
            return True
    return False
