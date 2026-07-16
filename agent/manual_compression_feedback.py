"""User-facing summaries for manual compression commands."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from agent.redact import redact_sensitive_text


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
    compression_state: Any = None,
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
        compression_state: the ContextCompressor (or compatible) instance;
            when provided, failure telemetry (_last_compress_aborted,
            _last_summary_fallback_used, _last_summary_error) surfaces
            aborted/fallback outcomes in the feedback (upstream 1e895f4c1/
            577beeb9b, merged 2026-07-16).
    """
    before_count = len(before_messages)
    after_count = len(after_messages)
    noop_chat = list(after_messages) == list(before_messages)

    # Failure telemetry (upstream): aborted/fallback outcomes take headline
    # precedence over both enhanced and classic wording.
    aborted = (
        compression_state is not None
        and getattr(compression_state, "_last_compress_aborted", False) is True
    )
    fallback_used = (
        compression_state is not None
        and getattr(compression_state, "_last_summary_fallback_used", False) is True
    )
    failure_reason = (
        getattr(compression_state, "_last_summary_error", None)
        if compression_state is not None
        else None
    )
    if not isinstance(failure_reason, str) or not failure_reason.strip():
        failure_reason = None

    enhanced = (
        non_chat_count is not None
        and non_chat_tokens is not None
        and transcript_rewritten is not None
        and full_before_count is not None
        and non_chat_count > 0
    )

    chat_line: Optional[str] = None
    dropped_line: Optional[str] = None

    if aborted:
        # Failure outcome (upstream): nothing was removed; say so regardless
        # of enhanced/classic mode.
        noop = True
        headline = f"Compression aborted: {before_count} messages preserved"
    elif fallback_used:
        noop = noop_chat
        headline = (
            f"Compressed with fallback: {before_count} → {after_count} messages"
        )
    elif enhanced:
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
    # An aborted compression preserves the store the same way.
    _preserved = (enhanced and not transcript_rewritten) or aborted

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
    if aborted:
        note = "Summary generation failed; no messages were removed."
    elif fallback_used:
        dropped_count = getattr(
            compression_state, "_last_summary_dropped_count", None
        )
        if not isinstance(dropped_count, int) or isinstance(dropped_count, bool):
            dropped_count = max(before_count - after_count, 0)
        note = (
            "Summary generation failed; Hermes used limited fallback context "
            f"and removed {dropped_count} message(s)."
        )
    elif (
        not _preserved
        and not noop_chat
        and after_count < before_count
        and after_tokens > before_tokens
    ):
        note = (
            "Note: fewer messages can still raise this estimate when "
            "compression rewrites the transcript into denser summaries."
        )

    if failure_reason and (aborted or fallback_used):
        # This text crosses a user-facing UI boundary.  Never let a disabled
        # global redaction preference expose credentials embedded in provider
        # exception text.
        safe_reason = redact_sensitive_text(failure_reason.strip(), force=True)
        note = f"{note} Reason: {safe_reason}"

    return {
        "noop": noop,
        "aborted": aborted,
        "fallback_used": fallback_used,
        "headline": headline,
        "token_line": token_line,
        "note": note,
        "enhanced": enhanced,
        "chat_line": chat_line,
        "dropped_line": dropped_line,
    }
