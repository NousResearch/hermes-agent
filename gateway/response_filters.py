"""Gateway response filtering helpers.

These decide whether a completed agent turn should be delivered to the chat,
not what should be persisted in conversation history.
"""

from __future__ import annotations

import unicodedata
from typing import Any

# Exact whole-response markers meaning "the agent intentionally chose not to
# reply". Keep small and explicit; arbitrary empty output remains an
# error/empty-response path, not silence. A lane that does not think in English
# translates the sentinel rather than dropping it, and the whole control token
# then reaches the user as content, so the translated forms are carried here
# too. zh-Hans is the only non-English locale this project ships documentation
# for, which is where the list stops.
LIVE_GATEWAY_SILENT_MARKERS = frozenset({
    "[SILENT]", "SILENT", "NO_REPLY", "NO REPLY",
    "[静默]", "静默", "[沉默]", "沉默",
})

# Bracketed markers drive the autonomous lane's prefix rule ("[SILENT] nothing
# new this tick"). Derived from the set above so a new marker cannot be added to
# one rule and forgotten in the other.
_BRACKETED_SILENCE_MARKERS = tuple(
    sorted(m for m in LIVE_GATEWAY_SILENT_MARKERS if m.startswith("["))
)

# The persisted user-row kind of a self-injected MessageEvent(internal=True) turn — the only
# machinery kind the gateway produces. Shared conversation silence is separate policy below.
INTERNAL_NOTIFICATION_DISPLAY_KIND = "internal_notification"
MACHINERY_DISPLAY_KINDS = frozenset({INTERNAL_NOTIFICATION_DISPLAY_KIND})

# Longer than any marker could plausibly be, even with stray punctuation.
_MARKER_LENGTH_CAP = 64


def _canonical_silence_candidate(text: str) -> str:
    return " ".join(text.strip().upper().split())


def _is_edge_punctuation(ch: str) -> bool:
    # Square brackets stay structural so malformed ``[SILENT`` cannot become ``SILENT``.
    return ch not in "[]" and unicodedata.category(ch).startswith("P")


def _strip_edge_silence_punctuation(text: str) -> str:
    """Strip stray edge punctuation (``.NO_REPLY``, ``*NO_REPLY*``) without erasing marker structure."""
    start, end = 0, len(text)
    while start < end and _is_edge_punctuation(text[start]):
        start += 1
    while end > start and _is_edge_punctuation(text[end - 1]):
        end -= 1
    return text[start:end].strip()


def _canonical_silence_candidates(text: Any) -> tuple[str, ...]:
    """Canonical forms of a short marker-sized response; ``()`` when not a candidate at all."""
    stripped = text.strip() if isinstance(text, str) else ""
    if not 0 < len(stripped) <= _MARKER_LENGTH_CAP:
        return ()
    depunctuated = _strip_edge_silence_punctuation(stripped)
    forms = (stripped,) if depunctuated == stripped else (stripped, depunctuated)
    return tuple(_canonical_silence_candidate(f) for f in forms)


def is_intentional_silence_response(response: Any) -> bool:
    """True only when ``response`` is exactly a silence marker.

    Prose that merely mentions ``NO_REPLY`` must be delivered normally. A blank
    response is not silence either — that is the empty-response failure path.
    """
    return any(c in LIVE_GATEWAY_SILENT_MARKERS for c in _canonical_silence_candidates(response))


def is_autonomous_silence_response(response: Any) -> bool:
    """Loose silence matcher for autonomous lanes (cron, webhook).

    Models reliably bracket ``[SILENT]`` with a short note, so unlike the
    interactive EXACT rule this also suppresses when a marker sits on its own
    first/last line or the bracketed sentinel opens the response (``[SILENT] No
    changes detected``).  A token buried mid-sentence is still delivered.
    Shares :data:`LIVE_GATEWAY_SILENT_MARKERS` so the two sets cannot drift.
    """
    stripped = response.strip() if isinstance(response, str) else ""
    if not stripped:
        return False
    lines = [ln for ln in stripped.splitlines() if ln.strip()]
    # Bracketed form only for the prefix rule, so a bare "Silent retry succeeded" is NOT swallowed.
    # Same de-punctuating forms as the interactive rule, so ``【静默】`` / ``静默。`` cannot
    # be suppressed in chat yet delivered by cron.
    return stripped.upper().startswith(_BRACKETED_SILENCE_MARKERS) or any(
        is_intentional_silence_response(c) for c in (stripped, lines[0], lines[-1])
    )


def is_intentional_silence_agent_result(agent_result: dict | None, response: Any) -> bool:
    """Silence markers suppress delivery only for successful agent turns."""
    return isinstance(agent_result, dict) and not agent_result.get("failed") and is_intentional_silence_response(response)


def display_kind_for_event(event: Any) -> str | None:
    """The persisted user-row kind for a gateway turn: only self-injected events are machinery.

    A scheduled heartbeat prompt is self-injected too (``_heartbeat_session_id`` is stamped only
    by the gateway poller, never inferred from inbound text), but it deliberately stays
    non-internal so authorization and the emergency stop still apply to it.
    """
    if getattr(event, "internal", False) or getattr(event, "_heartbeat_session_id", None):
        return INTERNAL_NOTIFICATION_DISPLAY_KIND
    return None


def is_machinery_display_kind(display_kind: Any) -> bool:
    """Identify machinery independently of the surface's intentional-silence policy.

    The caller passes the current turn's persisted display kind instead of inferring it from the
    transcript: the inbound user row is not persisted yet, and a previous internal row must never
    authorize silence on a human turn.
    """
    return display_kind in MACHINERY_DISPLAY_KINDS


def allows_intentional_silence(source: Any, display_kind: Any) -> bool:
    """Shared Discord conversations can contain human turns requiring no bot reply.

    DMs and other interactive surfaces retain their visible failure fallback.
    This policy only applies after a successful, exact silence-marker result.
    """
    platform = getattr(source, "platform", None)
    return is_machinery_display_kind(display_kind) or (
        getattr(platform, "value", platform) == "discord"
        and getattr(source, "chat_type", None) in {"group", "thread"}
    )


def is_partial_silence_marker(text: Any) -> bool:
    """True while streamed ``text`` could still resolve to a silence marker.

    A buffer whose canonical form is a non-empty *prefix* of a marker (``"NO"`` on
    the way to ``"NO_REPLY"``, or an exact marker not yet terminated by stream-end)
    is held back so a raw marker is never shown and then retracted.  Divergence
    from every marker, or exceeding the cap, resumes normal streaming.
    """
    return any(
        c and any(marker.startswith(c) for marker in LIVE_GATEWAY_SILENT_MARKERS)
        for c in _canonical_silence_candidates(text)
    )


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

SILENT_REPLY_TOKEN = "NO_REPLY"
# ---- END PLUGIN-COMPAT ----
