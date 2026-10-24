"""The model-visible compression/elision marker and its matcher.

Tool-call arguments are no longer rewritten by the compressor, but legacy sessions may still
carry this marker inside replayed tool-call arguments. Dependency-free leaf shared by the
elision renderers (``agent.context_compressor``) and the dispatch-boundary detector
(``agent.tool_dispatch_helpers``), so the matcher is derived from the template instead of
re-typing its wording.
"""

from __future__ import annotations

import re

# The summary wire markers are kept in this dependency-free leaf so state
# persistence can build its SQL preview predicates without importing the
# compressor, whose normal runtime imports providers and credentials.
HISTORICAL_TASK_HEADING = "## Historical Task Snapshot"

SUMMARY_PREFIX = (
    # Jul 2026 (#65848 class): identical to the pre-#69619 prefix except it lacked the explicit "tools
    # remain fully active" clause — the strong REFERENCE ONLY framing bled into general tool-use suppression
    # (observed: 7 consecutive narration-only turns immediately after a compression event on a production
    # deployment).
    # Carveout era (#41607/#38364/#42812): "consistent → use as background" licensed stale-task resumption
    # on topic overlap.
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Respond ONLY to the latest user message that appears AFTER this "
    "summary — that message is the single source of truth for what to do "
    "right now. "
    "If no user message appears AFTER this summary, do nothing: do not "
    "resume, wrap up, or continue work from "
    f"'{HISTORICAL_TASK_HEADING}' or any other section, do not call tools, "
    "and wait for a new user message. This handoff must never become the "
    "active turn by itself. (Exception: if tool results or your own "
    "tool calls appear after this summary, you are mid-way through an "
    "in-flight exchange — continue that exchange normally.) "
    "Topic overlap with the summary does NOT mean you should resume its "
    "task: even on similar topics, the latest user message WINS. Treat ONLY "
    "the latest message as the active task and discard stale items from "
    f"'{HISTORICAL_TASK_HEADING}' entirely — do not 'wrap up' or "
    "'finish' work described there unless the latest message explicitly "
    "asks for it. "
    "Reverse signals in the latest message (e.g. 'stop', 'undo', 'roll "
    "back', 'just verify', 'don't do that anymore', 'never mind', a new "
    "topic) must immediately end any in-flight work described in the "
    "summary; do not re-surface it in later turns. "
    "IMPORTANT: Your persistent memory (MEMORY.md, USER.md) in the system "
    "prompt is ALWAYS authoritative and active — never ignore or deprioritize "
    "memory content due to this compaction note. "
    "None of the above restricts HOW you work: your tools remain fully "
    "active — keep calling them normally for the active task (edit files, "
    "run commands, search) instead of merely narrating what you would do. "
    "The current session state (files, config, etc.) may reflect work "
    "described here — avoid repeating it:"
)
LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"

# ``display_metadata`` flag on a row the model reads but nobody typed as one
# message (micro-compaction's merge of adjacent user turns). Its source rows
# stay in display history, so display projections skip it.
MODEL_ONLY_DISPLAY_METADATA_KEY = "model_only"

# Intrinsic marker stamped on a message dict once it has been written to the
# SQLite session store. The underscore prefix keeps it off strict provider
# request schemas.
_DB_PERSISTED_MARKER = "_db_persisted"

# Explicit end boundary: weak models otherwise read quoted headers as fresh
# user input or replay an assistant-role summary as their own output.
_SUMMARY_END_MARKER = "--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---"

# Merged-into-tail case: prior tail content is kept BEFORE the summary inside
# these delimiters, so the summary prefix is not at content start.
_MERGED_PRIOR_CONTEXT_HEADER = "[PRIOR CONTEXT — for reference only; not a new message]"
_MERGED_SUMMARY_DELIMITER = "[END OF PRIOR CONTEXT — COMPACTION SUMMARY BELOW]"

# #83714 — this text lands inside the model's OWN replayed tool call, so it must not read like
# something the model would write itself: the bare "...[truncated]" it replaced was imitated into
# new calls and written to disk. Non-prose delimiters, an explicit "not original content"
# disclaimer, and per-instance counts keep a copied marker visibly wrong; the counts also make a
# verbatim copy stale, which is why the marker must never be re-applied.
_COMPRESSION_MARKER_PREFIX = "⟪HERMES-CONTEXT-COMPRESSION:"
_COMPRESSION_MARKER_TEMPLATE = (
    _COMPRESSION_MARKER_PREFIX
    + " {omitted:,} of {total:,} chars omitted here by Hermes's context compressor. "
    "This is NOT part of the original tool call and must never be reproduced in new "
    "output — always write full, untruncated content.⟫"
)

# A rendered omitted/total count inside a minted marker.
_MARKER_COUNT_RE = r"\d[\d,]*"

# Full rendered first sentence of a minted marker; renderer tests assert this shape
# (the dispatch guard uses _COMPRESSION_MARKER_ARTIFACT_RE below).
_COMPRESSION_MARKER_RE = re.compile(
    re.escape(_COMPRESSION_MARKER_TEMPLATE.split(". ", 1)[0] + ".")
    .replace(re.escape("{omitted:,}"), _MARKER_COUNT_RE)
    .replace(re.escape("{total:,}"), _MARKER_COUNT_RE)
)

# A marker cut by a later boundary may never reach the fixed sentence above. The
# dispatch guard therefore recognizes a minted marker as soon as its first numeric
# count is present. Requiring that rendered count keeps the bare prefix/template
# editable in source and documentation.
_COMPRESSION_MARKER_ARTIFACT_RE = re.compile(
    re.escape(_COMPRESSION_MARKER_PREFIX) + r"\s+" + _MARKER_COUNT_RE
)

# #121548 — every OTHER model-visible elision (turn text, summaries, skill bodies, diagnostics)
# mints the args marker's first sentence only (so the dispatch-boundary artifact
# matcher catches a copy from any renderer) and fits small caps like clarify's 199.
_ELISION_MARKER_TEMPLATE = _COMPRESSION_MARKER_TEMPLATE.split(". ", 1)[0] + ".⟫"


def _elision_marker(omitted: int, total: int) -> str:
    """Render the non-imitable elision marker with per-instance byte counts."""
    return _ELISION_MARKER_TEMPLATE.format(omitted=omitted, total=total)


# Widest marker for any text under ~1 TB: callers with a small leftover budget skip the item
# when the budget cannot hold the marker plus content, instead of emitting a marker-only line.
ELISION_MARKER_MAX_LEN = len(_elision_marker(omitted=10**12 - 1, total=10**12 - 1))


def elide(text: str, limit: int) -> str:
    """Cap ``text`` at ``limit`` chars, marking the elision with ``_elision_marker``.

    Every model-visible renderer must truncate through here (#121548). ``text`` is
    returned unchanged when it already fits. Otherwise the result is ``head + marker``
    with accurate omitted/total counts and never exceeds ``limit`` (sized against the
    widest rendering the counts can take), unless ``limit`` cannot hold the marker at
    all, in which case the marker alone is returned.
    """
    if len(text) <= limit:
        return text
    total = len(text)
    head_len = limit - len(_elision_marker(omitted=total, total=total))
    if head_len <= 0:
        return _elision_marker(omitted=total, total=total)
    kept = text[:head_len].rstrip()
    return kept + _elision_marker(omitted=total - len(kept), total=total)


def elide_middle(text: str, head: int, tail: int) -> str:
    """Keep ``head`` chars from the start and ``tail`` from the end, eliding the middle."""
    if len(text) <= head + tail:
        return text
    marker = _elision_marker(omitted=len(text) - head - tail, total=len(text))
    return text[:head] + marker + text[len(text) - tail:]
