"""Gateway response filtering helpers.

These helpers operate at the gateway boundary: they decide whether a completed
agent turn should be delivered to the chat, not what should be persisted in the
conversation history.
"""

from __future__ import annotations

import re
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
    if len(stripped) > 64:
        return False
    return any(candidate in LIVE_GATEWAY_SILENT_MARKERS for candidate in _canonical_silence_candidates(stripped))


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
    if not stripped or len(stripped) > 64:
        return False
    for candidate in _canonical_silence_candidates(stripped):
        if candidate and any(marker.startswith(candidate) for marker in LIVE_GATEWAY_SILENT_MARKERS):
            return True
    return False


# ── DSML (DeepSeek markup language) control-tag leakage guard ────────────────
# Real DeepSeek/oMLX tool-call markers use the U+FF5C bar character:
#   <｜DSML｜tool_calls>, <｜DSML｜invoke …>, <｜DSML｜parameter …>
# and their closers (</｜DSML｜…>). They are literal UTF-8 markers (NOT special
# tokens), so skip_special_tokens / clean_special_tokens never remove them;
# only a successful parse_tool_calls() strips them. If the tool-call envelope
# is truncated (max_tokens cut-off mid-serialization), the server's content
# recovery emits the withheld bytes as visible text and the tags leak to the
# chat surface.
#
# This guard must be applied at EVERY chat-delivery chokepoint — not just the
# final assembled response but also the STREAMING path (GatewayStreamConsumer
# progressive edits), which previously leaked `</invoke>`/`</tool_calls>`
# before the final-response sanitizer could run. See DSML-leakage fix
# (2026-08-04, 2026-08-05). Marker literals from omlx
# patches/deepseek_v4/tool_parser_v4.py.
_DSML_MARKER = "\uFF5C" + "DSML" + "\uFF5C"  # ｜DSML｜

# Matches a full DSML tag (opening or closing), e.g. <｜DSML｜invoke name="f">,
# </｜DSML｜invoke>, <｜DSML｜parameter name="x">, </｜DSML｜tool_calls>.
_DSML_TAG_RE = re.compile(
    re.escape("</" + _DSML_MARKER) + r"[^>]*>?|" + re.escape("<" + _DSML_MARKER) + r"[^>]*>?"
)

# Matches a COMPLETE, balanced tool-call envelope — an opening <｜DSML｜tool_calls>
# through its matching </｜DSML｜tool_calls>, including every nested invoke/
# parameter tag and the serialized argument JSON in between. Everything inside
# is model-internal tool-call serialization, never user-facing prose, so the
# whole block is dropped. Non-greedy across the inner tags.
_DSML_TOOL_CALLS_ENVELOPE_RE = re.compile(
    re.escape("<" + _DSML_MARKER + "tool_calls>")
    + r".*?"
    + re.escape("</" + _DSML_MARKER + "tool_calls>"),
    re.DOTALL,
)


def sanitize_dsml_markers(text: str) -> str:
    """Strip DSML control-tag leakage from model text before chat delivery.

    Removes every <｜DSML｜…> tag (opening and closing) and drops the entire
    contents of any balanced <｜DSML｜tool_calls>…</｜DSML｜tool_calls> envelope
    (its inner invoke/parameter tags AND the serialized argument JSON — none
    of that is user-facing prose). If an orphaned <｜DSML｜tool_calls> opener is
    present with no matching closer, drops everything from that opener to the
    end of the string (the stream was cut off mid-envelope — the trailing
    content is a partial tool call, not user-facing prose).
    """
    if not text:
        return text
    # Drop complete balanced envelopes first (tags + inner content).
    text = _DSML_TOOL_CALLS_ENVELOPE_RE.sub("", text)
    # Drop any orphaned opener with no closer, through end-of-string.
    open_tag = "<" + _DSML_MARKER + "tool_calls>"
    close_tag = "</" + _DSML_MARKER + "tool_calls>"
    idx = text.find(open_tag)
    if idx != -1 and close_tag not in text[idx:]:
        text = text[:idx]
    # Strip any remaining standalone tags (a lone invoke/parameter leak).
    return _DSML_TAG_RE.sub("", text)
