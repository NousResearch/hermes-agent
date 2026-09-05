"""Regression tests for iterative context-summary continuity."""

from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    SUMMARY_PREFIX,
    _MERGED_PRIOR_CONTEXT_HEADER,
    _MERGED_SUMMARY_DELIMITER,
    _RESTART_HANDOFF_PROBE_EXTRA_MESSAGES,
    _SUMMARY_END_MARKER,
)


def _compressor(protect_first_n: int = 1) -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=protect_first_n,
            protect_last_n=1,
            quiet_mode=True,
            # Pinned: these tests assert the stored summary ENDS with the raw
            # LLM text. Lean mode (the default since the tail-default flip)
            # appends the verbatim-user-quote appendix after it by design;
            # the continuity contract under test is mode-independent.
            tail_mode="legacy",
        )


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _messages_with_handoff(summary_body: str):
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{summary_body}"},
        {"role": "assistant", "content": "handoff acknowledged after resume"},
        {"role": "user", "content": "new user turn after resume"},
        {"role": "assistant", "content": "new assistant work after resume"},
        {"role": "user", "content": "more new work after resume"},
        {"role": "assistant", "content": "latest tail response"},
        {"role": "user", "content": "final active request stays in protected tail"},
    ]


def _messages_with_merged_handoff(summary_body: str, prior_tail: str):
    merged = {
        "role": "user",
        "content": (
            f"{_MERGED_PRIOR_CONTEXT_HEADER}\n{prior_tail}\n\n"
            f"{_MERGED_SUMMARY_DELIMITER}\n\n"
            f"{SUMMARY_PREFIX}\n{summary_body}\n\n{_SUMMARY_END_MARKER}"
        ),
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }
    messages = _messages_with_handoff(summary_body)
    messages[1] = merged
    return messages


def _messages_with_default_handoff(summary_body: str):
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "original task before first compaction"},
        {"role": "assistant", "content": "original answer before first compaction"},
        {"role": "user", "content": "original follow-up before first compaction"},
        {"role": "assistant", "content": f"{SUMMARY_PREFIX}\n{summary_body}"},
        {"role": "user", "content": "new user turn after restart"},
        {"role": "assistant", "content": "new assistant work after restart"},
        {"role": "user", "content": "more new work after restart"},
        {"role": "assistant", "content": "latest tail response"},
        {"role": "user", "content": "final active request stays in protected tail"},
    ]


def _messages_with_summary_at_index(summary_index: int):
    msgs = [{"role": "system", "content": "system prompt"}]
    for idx in range(1, summary_index):
        role = "user" if idx % 2 else "assistant"
        msgs.append({"role": role, "content": f"probe filler {idx}"})
    role = "user" if summary_index % 2 else "assistant"
    msgs.append({"role": role, "content": f"{SUMMARY_PREFIX}\nboundary summary"})
    msgs.extend([
        {"role": "assistant", "content": "new answer"},
        {"role": "user", "content": "tail request"},
    ])
    return msgs








def test_handoff_in_protected_head_is_replaced_not_duplicated():
    """Re-compaction must replace a protected old handoff with the updated one."""
    compressor = _compressor()
    old_summary = "OLD-PROTECTED-HANDOFF unique old summary body"

    with patch("agent.context_compressor.call_llm", return_value=_response("UPDATED summary body")):
        compressed = compressor.compress(_messages_with_handoff(old_summary))

    # The summary may be emitted standalone or merged into the first tail
    # message (alternation corner case), so detect it the same way the
    # compressor does rather than via a startswith(SUMMARY_PREFIX) check.
    summary_messages = [
        msg
        for msg in compressed
        if isinstance(msg, dict)
        and ContextCompressor._is_context_summary_content(msg.get("content"))
    ]
    assert len(summary_messages) == 1
    assert "UPDATED summary body" in str(summary_messages[0]["content"])
    assert old_summary not in str(summary_messages[0]["content"])
    assert old_summary not in "\n".join(str(msg.get("content") or "") for msg in compressed)






def test_recompression_of_current_merged_handoff_preserves_prior_tail_once():
    """Current merged handoffs lose only stale summary data on recompression.

    Composed contract after #57835 (restart head-protection decay): the
    merged handoff's genuine prior-tail content must be RECOVERED — either
    verbatim in the output (pre-decay head protection) or by entering the
    summarizer input so the fresh summary folds it in (post-decay). It must
    never be silently deleted, and the stale summary body must never be
    re-emitted verbatim.
    """
    compressor = _compressor()
    old_summary = "CURRENT-MERGED-OLD-SUMMARY unique continuity facts"
    prior_tail = "PRESERVED-PRIOR-TAIL real user content"

    seen_turns = []

    def _capture(turns, **kwargs):
        seen_turns.extend(turns)
        return ContextCompressor._with_summary_prefix(
            "fresh replacement summary"
        )

    with patch.object(
        compressor,
        "_generate_summary",
        side_effect=_capture,
    ):
        result = compressor.compress(
            _messages_with_merged_handoff(old_summary, prior_tail)
        )

    joined = "\n".join(str(message.get("content", "")) for message in result)
    summarizer_input = "\n".join(str(t.get("content", "")) for t in seen_turns)
    # Prior tail recovered: verbatim in output OR folded via summarizer input.
    assert prior_tail in joined or prior_tail in summarizer_input
    # Never duplicated in the output.
    assert joined.count(prior_tail) <= 1
    assert old_summary not in joined
    assert joined.count(SUMMARY_PREFIX) == 1
    assert "fresh replacement summary" in joined








def test_resume_handoff_after_default_protected_head_decays_initial_turns():
    """Default protect_first_n=3 should not fossilize old protected head turns."""
    compressor = _compressor(protect_first_n=3)
    old_summary = "DEFAULT-RESTART-SUMMARY durable facts from before restart"

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")) as mock_call:
        result = compressor.compress(_messages_with_default_handoff(old_summary))

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert prompt.count(old_summary) == 1
    assert "original task before first compaction" in prompt
    assert "original answer before first compaction" in prompt
    assert "original follow-up before first compaction" in prompt
    assert f"[ASSISTANT]: {SUMMARY_PREFIX}" not in prompt
    # Grounding (761a0b124e) may prepend a deterministic task-snapshot
    # section — pin the contract, not the exact stored string.
    stored_summary = compressor._previous_summary or ""
    assert stored_summary.endswith("fresh summary")
    assert old_summary not in stored_summary
    assert all(
        "original task before first compaction" not in str(msg.get("content", ""))
        for msg in result
    )
    assert all(
        "original answer before first compaction" not in str(msg.get("content", ""))
        for msg in result
    )
    assert all(
        old_summary not in str(msg.get("content", ""))
        for msg in result
    )


def test_restart_simulation_fresh_compressor_does_not_reprotect_head():
    """Gateway-restart simulation: a FRESH ContextCompressor (in-memory decay
    state reset — compression_count == 0, _previous_summary is None) over a
    transcript that contains a persisted handoff summary must NOT re-protect
    the head. compress_start must reflect decayed protection exactly as a
    live (non-restarted) process would compute it (#57814)."""
    # Live process: has already compacted once, decay is in-memory.
    live = _compressor(protect_first_n=3)
    live.compression_count = 1

    # Restarted process: brand-new compressor, all in-memory state fresh.
    restarted = _compressor(protect_first_n=3)
    assert restarted.compression_count == 0
    assert not restarted._previous_summary

    msgs = _messages_with_default_handoff(
        "PERSISTED-HANDOFF durable facts from before restart"
    )

    # The protected-head boundary the compressor uses for compress_start
    # must be identical for both: system prompt only (decayed protection).
    assert restarted._effective_protect_first_n(msgs) == 0
    assert restarted._protect_head_size(msgs) == live._protect_head_size(msgs) == 1
    restarted_start = restarted._align_boundary_forward(
        msgs, restarted._protect_head_size(msgs)
    )
    assert restarted_start == 1

    # End-to-end: the first post-restart compaction must not preserve the
    # pre-restart head turns or the old handoff verbatim.
    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")):
        result = restarted.compress(msgs)
    result_text = "\n".join(str(msg.get("content", "")) for msg in result)
    assert "PERSISTED-HANDOFF durable facts" not in result_text
    assert "original task before first compaction" not in result_text
    assert "original answer before first compaction" not in result_text








def test_zero_protect_first_n_still_folds_restart_fossil():
    """protect_first_n=0 should still self-heal restarted summaries."""
    compressor = _compressor(protect_first_n=0)
    old_summary = "OLD-SUMMARY-ZERO-PROTECT durable facts"
    msgs = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "task one"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "task two"},
        {"role": "assistant", "content": f"{SUMMARY_PREFIX}\n{old_summary}"},
        {"role": "user", "content": "active request"},
    ]

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")):
        result = compressor.compress(msgs)

    result_text = "\n".join(str(msg.get("content", "")) for msg in result)
    assert old_summary not in result_text
    assert result_text.index(_SUMMARY_END_MARKER) < result_text.index("active request")
    assert sum(
        1 for msg in result if ContextCompressor._is_context_summary_message(msg)
    ) == 1




def test_restart_fossil_survives_summary_abort_then_retry():
    """An aborted first compaction must not strand the rehydrated fossil.

    Regression for the abort/retry path. The first-compaction self-heal scan
    (``compression_count < 1``) populates ``_previous_summary`` from a fossil
    that drifted past the decay probe. If summary generation then aborts
    (auth / network / ``abort_on_summary_failure``) and returns the transcript
    unchanged, the aborted attempt must not leave that rehydrated state behind:
    otherwise the retry — still ``compression_count == 0`` but now with a
    truthy ``_previous_summary`` — takes the narrow rescan, misses the
    beyond-window fossil, and then discards the rehydrated summary as
    cross-session leakage, copying the fossil forward as a stacked summary.
    """
    compressor = _compressor(protect_first_n=1)
    compressor.abort_on_summary_failure = True
    old_summary = "ABORT-RETRY-OLD-SUMMARY durable facts"
    msgs = [{"role": "system", "content": "system prompt"}]
    msgs += [
        {
            "role": "user" if idx % 2 else "assistant",
            "content": f"filler {idx}",
        }
        for idx in range(1, 6)
    ]
    msgs += [
        {"role": "assistant", "content": f"{SUMMARY_PREFIX}\n{old_summary}"},
        {"role": "user", "content": "active request"},
    ]

    # First compaction aborts on a summary-generation failure. The transcript
    # is returned unchanged AND the self-heal state it rehydrated must be
    # rolled back, so a retry behaves like the original first compaction.
    with patch.object(compressor, "_generate_summary", return_value=None):
        aborted = compressor.compress([dict(m) for m in msgs])
    assert compressor._last_compress_aborted is True
    assert all(m["content"] for m in aborted)  # returned unchanged
    assert compressor.compression_count == 0
    assert compressor._previous_summary is None

    # Retry: the fossil beyond the narrow window is still folded, not copied
    # forward as a second stacked summary.
    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")):
        result = compressor.compress([dict(m) for m in msgs])

    assert all(old_summary not in str(msg.get("content", "")) for msg in result)
    assert sum(
        1 for msg in result if ContextCompressor._is_context_summary_message(msg)
    ) == 1


def test_degenerate_compress_end_keeps_same_session_previous_summary():
    """Handoff beyond a degenerate compress_end must not clear same-session state.

    Regression for #83248: with compression_count >= 1 and a truthy
    ``_previous_summary`` from this session's prior compaction, a scan bounded
    at ``compress_end`` can miss an in-window handoff sitting past the cut.
    The #57835 cross-session guard then discarded the valid summary and the
    from-scratch path replaced a rich handoff with a one-turn summary.
    """
    compressor = _compressor(protect_first_n=0)
    rich_summary = (
        "RICH-SAME-SESSION-HANDOFF\n"
        "1. completed action alpha\n"
        "2. completed action beta\n"
        "3. completed action gamma"
    )
    compressor.compression_count = 1
    compressor._previous_summary = rich_summary

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "early turn before cut"},
        {"role": "assistant", "content": "early answer"},
        {"role": "user", "content": "mid turn before handoff"},
        {"role": "assistant", "content": f"{SUMMARY_PREFIX}\n{rich_summary}"},
        {"role": "user", "content": "post handoff turn"},
        {"role": "assistant", "content": "post handoff answer"},
        {"role": "user", "content": "tail request"},
    ]
    # Handoff is at index 4; force compress_end=2 so a compress_end-bounded
    # scan would miss it while still leaving a non-empty middle window.
    assert ContextCompressor._is_context_summary_message(messages[4])

    previous_at_generate = []

    def _capture(turns, **kwargs):
        previous_at_generate.append(compressor._previous_summary)
        return ContextCompressor._with_summary_prefix("updated iterative summary")

    with (
        patch.object(compressor, "_find_tail_cut_by_tokens", return_value=2),
        patch.object(compressor, "_generate_summary", side_effect=_capture),
    ):
        result = compressor.compress(messages, current_tokens=90_000)

    assert previous_at_generate, "_generate_summary should have run"
    assert previous_at_generate[0] is not None, (
        "same-session _previous_summary must not be discarded when a handoff "
        "exists beyond a degenerate compress_end"
    )
    assert "RICH-SAME-SESSION-HANDOFF" in previous_at_generate[0]
    # Iterative state must remain available after the attempt (either the
    # prior rich handoff, or the updated summary if the compaction committed).
    assert compressor._previous_summary is not None
    stored = compressor._previous_summary or ""
    assert (
        "RICH-SAME-SESSION-HANDOFF" in stored
        or "updated iterative summary" in stored
    )
    assert sum(
        1 for msg in result if ContextCompressor._is_context_summary_message(msg)
    ) >= 1




def test_forced_leading_merged_summary_strips_live_tail_from_summary_body():
    """Rehydrating a forced-leading merged summary should ignore live tail."""
    merged = (
        f"{SUMMARY_PREFIX}\nSUMMARY_BODY\n\n"
        f"{_SUMMARY_END_MARKER}\n\n"
        "LIVE_TAIL_REQUEST"
    )

    assert ContextCompressor._is_context_summary_content(merged) is True
    assert ContextCompressor._strip_summary_prefix(merged) == "SUMMARY_BODY"










def test_empty_post_handoff_window_noops_without_summary_call():
    """A latest handoff that consumes the window must not trigger an empty summary.

    Regression test from PR #59526 (#59496), fixture adapted to current main:
    the standalone handoff sits alone in the compressible window, strips to
    None via _strip_context_summary_handoff_message, and leaves
    turns_to_summarize empty — the guard must skip _generate_summary
    entirely instead of wasting an aux LLM call on empty input.
    """
    compressor = _compressor()
    old_summary = "WINDOW-END-SUMMARY durable facts already captured"
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{old_summary}"},
        {"role": "assistant", "content": "recent tail response"},
        {"role": "user", "content": "tail request"},
        {"role": "assistant", "content": "tail answer"},
        {"role": "user", "content": "latest tail request"},
        {"role": "assistant", "content": "latest tail answer"},
    ]

    with (
        patch.object(compressor, "_find_tail_cut_by_tokens", return_value=2),
        patch.object(compressor, "_generate_summary") as mock_generate_summary,
    ):
        result = compressor.compress(messages, current_tokens=90_000)

    mock_generate_summary.assert_not_called()
    assert result == messages
    # The rehydrated summary state is deliberately kept: the handoff is
    # genuinely present in the returned (unchanged) transcript.
    assert compressor._previous_summary == old_summary
    assert compressor.compression_count == 0
    # Mirrors the sibling no-compressible-window guard, but as a structural
    # no-op (#93022): the window holds nothing eligible to compress, so the
    # compressor defers retries transiently instead of arming the permanent
    # anti-thrash breaker.
    assert compressor._ineffective_compression_count == 0
    assert compressor._structural_no_op_backoff_until > 0.0
    assert compressor._last_compression_savings_pct == 0.0
    assert compressor._last_summary_dropped_count == 0
    assert compressor._last_summary_fallback_used is False
    assert compressor._last_compress_aborted is False
    telemetry = compressor._last_compression_telemetry or {}
    assert telemetry.get("failure_class") == "empty_post_handoff_window"
