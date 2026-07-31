"""Regression coverage for #35344: a resumed session must not let a stale
historical task snapshot from an inherited compaction handoff hijack the reply to a
new, unrelated user message.

The failure mode (real report): a lineage was compacted, producing a handoff
whose historical task snapshot described task A. The lineage was resumed later and
the user asked about an unrelated task B. The model answered with A because
the handoff's resume directive outranked the fresh ask.

The structural fix lives in ``SUMMARY_PREFIX``: the handoff is framed as
reference-only and the latest user message explicitly *wins* on conflict, with
named reverse-signal verbs. Two invariants guard the resume path specifically:

  1. A handoff persisted under the OLD (conflicting) prefix is re-normalized to
     the CURRENT prefix when it is re-compacted on a resumed lineage — so a
     pre-fix stale handoff cannot keep its "resume exactly" directive forever.

  2. The current handoff prefix contains an unambiguous "latest message wins /
     discard stale historical task" rule, so an unrelated new ask is privileged over
     the inherited task snapshot.

These are content/structural assertions (no live model call) — they pin the
mechanism that makes the stale task historical rather than active.
"""

from agent.context_compressor import (
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    LEGACY_SUMMARY_PREFIX,
    ContextCompressor,
)


# The conflicting prefix that shipped before the #35344 fix. A handoff
# persisted in a resumed lineage could carry this verbatim.
_OLD_CONFLICTING_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. "
    "Your current task is identified in the '## Active Task' section of the "
    "summary — resume exactly from there. "
    "Respond ONLY to the latest user message "
    "that appears AFTER this summary. The current session state (files, "
    "config, etc.) may reflect work described here — avoid repeating it:"
)


def test_latest_message_wins_over_inherited_active_task():
    """The handoff must explicitly privilege the latest user message over a
    stale historical task snapshot — the core #35344 contract."""
    lower = SUMMARY_PREFIX.lower()
    assert "latest user message" in lower
    assert HISTORICAL_TASK_HEADING.lower() in lower
    # Conflict-resolution must be explicit, not implied.
    assert "wins" in lower or "supersede" in lower
    assert "discard" in lower
    # The "consistent -> use as background" carveout licensed stale-task
    # resumption on topic overlap (#41607, #38364) — it must stay gone.
    assert "you may use the summary as background" not in lower
    assert "topic overlap" in lower








def test_inherited_handoff_detected_in_resumed_protected_head():
    """On a resumed lineage the handoff commonly sits right after the system
    prompt (in the protected head). ``_find_latest_context_summary`` must
    detect it there so re-compaction rehydrates state from it rather than
    serializing it as a fresh user turn (which is what let the stale Active
    Task read as live intent)."""
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\nUser asked: 'task A'"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "Unrelated task B: what's the capital of France?"},
    ]
    # Search the whole post-system range.
    idx, body = ContextCompressor._find_latest_context_summary(
        messages, 1, len(messages)
    )
    assert idx == 1, "handoff in protected head must be found"
    assert "task A" in body
    # The detected body is stripped of the prefix (treated as state, not a
    # standalone instruction message).
    assert not body.startswith(SUMMARY_PREFIX)


