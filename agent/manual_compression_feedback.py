"""User-facing summaries for manual compression commands.

Two modes:

- **Classic** (CLI/TUI, and any caller that passes only the four positional
  args): before/after chat-message counts + a token line. Behavior is
  byte-identical to the original helper.

- **Enhanced** (gateway ``/compress``): the caller additionally reports the
  stored-transcript axis — how many non-chat rows (tool results / system /
  contentless turns) the transcript rewrite dropped, whether a rewrite
  actually happened, and the full stored-row count. The summary then
  reconciles BOTH axes so the headline can never contradict the token math
  (the "No changes: 179 messages" over a 453K→32K line bug, 2026-07-02).

Enhanced cases (only when ``non_chat_count > 0`` — pure-chat transcripts fall
back to the classic wording):

- CASE A — rewrite happened, chat untouched: the win came entirely from
  dropping stored tool/system rows. Headline says "Compacted stored
  transcript", the chat line says the chat was already compact.
- CASE B — rewrite happened, chat also compressed: both axes shrank.
- CASE C — no rewrite (abort / failure): the transcript — including its
  non-chat rows — is preserved verbatim, and the headline says so instead
  of implying a shrink.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence


def summarize_manual_compression(
    before_messages: Sequence[dict[str, Any]],
    after_messages: Sequence[dict[str, Any]],
    before_tokens: int,
    after_tokens: int,
    *,
    non_chat_count: Optional[int] = None,
    non_chat_tokens: Optional[int] = None,
    transcript_rewritten: Optional[bool] = None,
    full_before_count: Optional[int] = None,
) -> dict[str, Any]:
    """Return consistent user-facing feedback for manual compression.

    Args (enhanced mode, all-or-nothing — gateway /compress):
        non_chat_count: stored transcript rows EXCLUDED from the chat-only
            compression input (tool results, system rows, contentless turns).
            These are dropped by the transcript rewrite when it happens.
        non_chat_tokens: rough token estimate over those rows.
        transcript_rewritten: True when the stored transcript was actually
            rewritten (session rotation or in-place compaction); False when
            compression aborted/no-oped and the store is untouched.
        full_before_count: total stored rows before (chat + non-chat).
    """
    before_count = len(before_messages)
    after_count = len(after_messages)
    noop_chat = list(after_messages) == list(before_messages)

    enhanced = (
        non_chat_count is not None
        and non_chat_tokens is not None
        and transcript_rewritten is not None
        and full_before_count is not None
        and non_chat_count > 0
    )

    chat_line: Optional[str] = None
    dropped_line: Optional[str] = None

    if enhanced:
        _nc_tok = int(non_chat_tokens or 0)
        if not transcript_rewritten:
            # CASE C — true no-op: nothing was dropped, nothing rewritten.
            noop = True
            headline = (
                f"No changes: transcript preserved ({full_before_count} messages: "
                f"{before_count} chat + {non_chat_count} tool/system)"
            )
            chat_line = (
                f"Chat size: ~{before_tokens:,} tokens (unchanged; "
                f"excludes system, tools, tool results)"
            )
        elif noop_chat:
            # CASE A — chat already compact; the rewrite dropped stored
            # tool/system rows. This is a real, usually large, win — the
            # headline must NOT say "no changes".
            noop = False
            headline = (
                f"Compacted stored transcript: {full_before_count} → "
                f"{after_count} messages"
            )
            chat_line = (
                f"Chat: {before_count} messages (~{before_tokens:,} tokens) — "
                f"already compact, kept verbatim"
            )
            dropped_line = (
                f"Dropped: {non_chat_count} stored tool/system messages "
                f"(~{_nc_tok:,} tokens reclaimed)"
            )
        else:
            # CASE B — both axes shrank.
            noop = False
            headline = (
                f"Compressed: {full_before_count} → {after_count} stored messages"
            )
            chat_line = (
                f"Chat: {before_count} → {after_count} messages "
                f"(~{before_tokens:,} → ~{after_tokens:,} tokens; "
                f"older chat folded into a summary)"
            )
            dropped_line = (
                f"Dropped: {non_chat_count} stored tool/system messages "
                f"(~{_nc_tok:,} tokens reclaimed)"
            )
    else:
        # Classic mode — original behavior, byte-identical.
        noop = noop_chat
        if noop:
            headline = f"No changes from compression: {before_count} messages"
        else:
            headline = f"Compressed: {before_count} → {after_count} messages"

    # CASE C invariant (enhanced, no rewrite): the store is untouched, so the
    # next request resends the ORIGINAL context regardless of what the
    # compressor produced in memory. Force the unchanged token wording rather
    # than trusting every caller to pass after_tokens == before_tokens.
    _preserved = enhanced and not transcript_rewritten

    if noop_chat or _preserved:
        if after_tokens == before_tokens or _preserved:
            token_line = (
                f"Approx request size: ~{before_tokens:,} tokens (unchanged)"
            )
        else:
            token_line = (
                f"Approx request size: ~{before_tokens:,} → "
                f"~{after_tokens:,} tokens"
            )
    else:
        token_line = (
            f"Approx request size: ~{before_tokens:,} → "
            f"~{after_tokens:,} tokens"
        )

    note = None
    if (
        not _preserved
        and not noop_chat
        and after_count < before_count
        and after_tokens > before_tokens
    ):
        note = (
            "Note: fewer messages can still raise this estimate when "
            "compression rewrites the transcript into denser summaries."
        )

    return {
        "noop": noop,
        "headline": headline,
        "token_line": token_line,
        "note": note,
        "enhanced": enhanced,
        "chat_line": chat_line,
        "dropped_line": dropped_line,
    }
