"""Gateway response filtering helpers.

These decide whether a completed agent turn should be delivered to the chat,
not what should be persisted in conversation history.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional

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

# The persisted user-row kind of a self-injected MessageEvent(internal=True) turn; only these
# machinery kinds may vanish on a bare silence marker.
INTERNAL_NOTIFICATION_DISPLAY_KIND = "internal_notification"
# Async-result cards the in-process TUI persisted: a producer's internal event naming one (with a
# display_text) persists as that row kind instead of internal_notification.
PRODUCER_NOTICE_DISPLAY_KINDS = frozenset({"process_complete", "async_delegation_complete"})
MACHINERY_DISPLAY_KINDS = frozenset({INTERNAL_NOTIFICATION_DISPLAY_KIND, *PRODUCER_NOTICE_DISPLAY_KINDS})
# What a gateway producer's card may name (a liveness tick names internal_notification for its CLI
# receipt). Closed, and kept apart from the silence set above so a kind added there never becomes
# nameable; never a client-chosen presentation such as ``hidden``.
PRODUCER_CARD_KINDS = frozenset({INTERNAL_NOTIFICATION_DISPLAY_KIND, *PRODUCER_NOTICE_DISPLAY_KINDS})

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
    from gateway.session_display import admission_display
    if kind := admission_display(event).get("kind"):
        # A canonical admission committed hidden (Desktop widget intent): never a user bubble.
        return kind
    if getattr(event, "internal", False) and _producer_card(event):
        return event.metadata["display_kind"]
    if getattr(event, "internal", False) or getattr(event, "_heartbeat_session_id", None):
        return INTERNAL_NOTIFICATION_DISPLAY_KIND
    return None


def _producer_card(event: Any) -> dict:
    metadata = getattr(event, "metadata", None) or {}
    text = metadata.get("display_text")
    if (getattr(event, "internal", False) and metadata.get("display_kind") in PRODUCER_NOTICE_DISPLAY_KINDS
            and isinstance(text, str) and text.strip()):
        return {"display_text": text}
    return {}


def display_metadata_for_event(event: Any) -> dict:
    """``{"display_text": ...}`` for an internal event whose producer named an async-result card, plus the
    ``title_preview`` (titler-only) a canonical admission committed for this exact event."""
    from gateway.session_display import admission_display
    preview = admission_display(event).get("title_preview")
    return {**_producer_card(event), **({"title_preview": preview} if preview else {})}


def is_machinery_display_kind(display_kind: Any) -> bool:
    """Only a machinery turn may vanish on a bare silence marker; a human turn gets a visible fallback.

    The caller passes the current turn's persisted display kind instead of inferring it from the
    transcript: the inbound user row is not persisted yet, and a previous internal row must never
    authorize silence on a human turn.
    """
    return display_kind in MACHINERY_DISPLAY_KINDS


def silence_allowed(display_kind: Any, reply_expected: Optional[bool] = None) -> bool:
    """Whether a successful bare silence marker may remain silent for this turn."""
    return is_machinery_display_kind(display_kind) or reply_expected is False


def reply_expected_metadata(reply_expected: Optional[bool]) -> dict:
    """The persisted user row's ``reply_expected`` key, only when the adapter knew; crash recovery
    reads it back to judge a silence marker as the live turn did."""
    return {} if reply_expected is None else {"reply_expected": reply_expected}


# Kanban wake rows persisted before gateway turns were typed ``internal_notification`` (Oct 2026).
# Matched on the WHOLE ``gateway.kanban.wake.message`` template (every shipped locale): the
# ``[kanban] <word> t_<hex> ...`` head, then title / ``@assignee`` / board lines, then the guidance
# paragraph. A human pasting or quoting a one-line ping ("[kanban] Task t_x is stuck?") never matches.
_LEGACY_KANBAN_WAKE_HEAD_RE = re.compile(r"\[kanban\] .{0,40}?\bt_[0-9a-f]{6,64}\b")
_FIELD_LINE_RE = re.compile(r"[^:：]{1,24}[:：](.*)")


def _is_legacy_kanban_wake(text: str) -> bool:
    """Line-based (no backtracking over the whole body): head, three ``Label: value`` lines with an
    ``@assignee`` second, then a blank line and the guidance paragraph."""
    lines = text.lstrip()[:4000].split("\n", 6)
    if len(lines) < 6 or not _LEGACY_KANBAN_WAKE_HEAD_RE.match(lines[0]):
        return False
    title, assignee, board = (_FIELD_LINE_RE.fullmatch(line) for line in lines[1:4])
    return bool(title and board and assignee and assignee.group(1).strip().startswith("@")
                and lines[4] == "" and lines[5].strip())
HIDDEN_DISPLAY_KIND = "hidden"


def _display_row_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else str(p.get("text") or "") for p in content
                       if isinstance(p, (str, dict)))
    return content if isinstance(content, str) else ""


def is_machinery_prompt_row(message: Any) -> bool:
    """A self-injected wake row (Kanban status, internal notification): never words the user typed."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    kind = message.get("display_kind")
    if kind:
        return kind == INTERNAL_NOTIFICATION_DISPLAY_KIND
    return _is_legacy_kanban_wake(_display_row_text(message))


def is_silence_display_row(message: Any) -> bool:
    """A persisted assistant row that is exactly a silence marker (``NO_REPLY`` / ``[SILENT]``)."""
    return (isinstance(message, dict) and message.get("role") == "assistant" and not message.get("tool_calls")
            and is_intentional_silence_response(_display_row_text(message)))


def project_silent_machinery_turns(messages: list) -> list:
    """Display projection: plumbing stays in the store, out of the transcript.

    * A bare silence-marker assistant row is the agent deliberately saying nothing: hidden.
    * A machinery wake turn (Kanban status wake, internal notification) that produced no visible
      reply is hidden whole: its prompt row, tool rows and silence row. Nothing reached anyone, so
      the transcript shows nothing — not a user bubble and not a "no reply" error.
    * A machinery turn that did reply keeps its prompt row typed ``internal_notification`` (legacy
      untyped rows are typed here) so clients render it as a collapsible notice, never a user bubble.

    Rows are re-typed ``hidden`` rather than dropped so paged readers keep their row offsets.
    Persisted rows and model context are untouched. Returns a new list; input dicts are not mutated.
    """
    out = [({**m, "display_kind": HIDDEN_DISPLAY_KIND} if is_silence_display_row(m) else m) for m in messages]
    index, total = 0, len(out)
    while index < total:
        row = out[index]
        if not is_machinery_prompt_row(row):
            index += 1
            continue
        end = index + 1
        while end < total and not (isinstance(out[end], dict) and out[end].get("role") == "user"):
            end += 1
        replied = any(
            isinstance(m, dict) and m.get("role") == "assistant" and m.get("display_kind") != HIDDEN_DISPLAY_KIND
            and _display_row_text(m).strip()
            for m in out[index + 1:end])
        if not replied:
            for position in range(index, end):
                if isinstance(out[position], dict):
                    out[position] = {**out[position], "display_kind": HIDDEN_DISPLAY_KIND}
        else:
            out[index] = _wake_notice_card(row)
        index = end
    return out


_WAKE_HEAD_RE = re.compile(r"\A\[kanban\]\s*(?:\S+\s+)?(t_[0-9a-f]+)\s+(.+?)\.?\Z")


def wake_notice_label(text: str) -> str:
    """``[kanban] Task t_x completed.\\nTitle: Y`` -> ``Board update · t_x completed · Y``."""
    lines = [line.strip() for line in text.strip().splitlines()]
    head = _WAKE_HEAD_RE.match(lines[0]) if lines else None
    if not head:
        return "Background update" + (f" · {lines[0][:120]}" if lines and lines[0] else "")
    title = next((line.split(":", 1)[1].strip() for line in lines[1:3] if ":" in line), "")
    return " · ".join(part for part in ("Board update", f"{head[1]} {head[2]}", title[:120]) if part)


def _wake_notice_card(row: dict) -> dict:
    """A replied wake row as the collapsible async-result card every client already renders
    (label = ``display_text``, full wake text one click away) instead of a user bubble."""
    raw_metadata = row.get("display_metadata")
    metadata: dict = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
    return {**row, "display_kind": "async_delegation_complete",
            "display_metadata": {**metadata, "display_text": wake_notice_label(_display_row_text(row)),
                                 "wake_kind": INTERNAL_NOTIFICATION_DISPLAY_KIND}}


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
