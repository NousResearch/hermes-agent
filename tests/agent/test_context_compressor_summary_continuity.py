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


def test_existing_previous_summary_is_not_serialized_again_as_new_turn():
    """Same-process iterative compression should not feed the old handoff twice."""
    compressor = _compressor()
    old_summary = "OLD-SUMMARY-BODY unique continuity facts"
    compressor._previous_summary = old_summary

    with patch("agent.context_compressor.call_llm", return_value=_response("updated summary")) as mock_call:
        compressor.compress(_messages_with_handoff(old_summary))

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert "NEW TURNS TO INCORPORATE:" in prompt
    assert prompt.count(old_summary) == 1
    assert f"[USER]: {SUMMARY_PREFIX}" not in prompt


def test_resume_rehydrates_previous_summary_from_handoff_message():
    """After restart/resume, the persisted handoff should regain summary identity."""
    compressor = _compressor()
    old_summary = "RESUMED-SUMMARY-BODY durable continuity facts"
    assert compressor._previous_summary is None

    with patch("agent.context_compressor.call_llm", return_value=_response("updated summary")) as mock_call:
        compressor.compress(_messages_with_handoff(old_summary))

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert "NEW TURNS TO INCORPORATE:" in prompt
    assert "TURNS TO SUMMARIZE:" not in prompt
    assert prompt.count(old_summary) == 1
    assert f"[USER]: {SUMMARY_PREFIX}" not in prompt


def test_handoff_in_protected_head_populates_previous_summary_before_update():
    """A resumed protected-head handoff should restore iterative-summary state."""
    compressor = _compressor()
    old_summary = "PROTECTED-HEAD-SUMMARY durable facts from before restart"
    seen_turns = []

    def fake_generate_summary(
        turns_to_summarize,
        focus_topic=None,
        memory_context="",
    ):
        seen_turns.extend(turns_to_summarize)
        return "new summary from resumed turns"

    with patch.object(compressor, "_generate_summary", side_effect=fake_generate_summary):
        compressor.compress(_messages_with_handoff(old_summary))

    assert compressor._previous_summary == old_summary
    assert seen_turns
    assert all(old_summary not in str(msg.get("content", "")) for msg in seen_turns)


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


def test_recompression_drops_prior_protected_handoff_from_output():
    """Repeated compression must not preserve stale handoff bubbles forever."""
    compressor = _compressor()
    old_summary = "DUPLICATE-HANDOFF-BODY unique old facts"

    with patch.object(
        compressor,
        "_generate_summary",
        return_value=ContextCompressor._with_summary_prefix(
            "updated summary with old facts folded in"
        ),
    ):
        result = compressor.compress(_messages_with_handoff(old_summary))

    joined = "\n".join(str(message.get("content", "")) for message in result)
    assert old_summary not in joined
    assert joined.count(SUMMARY_PREFIX) == 1
    assert "updated summary with old facts folded in" in joined


def test_legacy_string_merged_handoff_preserves_real_tail_text():
    """Pre-delimiter string handoffs still unwrap content after the end marker."""
    message = {
        "role": "user",
        "content": (
            f"{SUMMARY_PREFIX}\nold summary\n\n"
            f"{_SUMMARY_END_MARKER}\n\nreal tail message"
        ),
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }

    result = ContextCompressor._strip_context_summary_handoff_message(message)

    assert result == {"role": "user", "content": "real tail message"}


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


def test_current_multimodal_merged_handoff_preserves_original_blocks():
    """Unwrapping current list content must retain text and image blocks."""
    prior_text = {"type": "text", "text": "real multimodal tail"}
    prior_image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,AAAA"},
    }
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"{_MERGED_PRIOR_CONTEXT_HEADER}\n"},
            prior_text,
            prior_image,
            {
                "type": "text",
                "text": (
                    f"\n\n{_MERGED_SUMMARY_DELIMITER}\n\n"
                    f"{SUMMARY_PREFIX}\nstale summary\n\n{_SUMMARY_END_MARKER}"
                ),
            },
        ],
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }

    result = ContextCompressor._strip_context_summary_handoff_message(message)

    assert result == {
        "role": "user",
        "content": [prior_text, prior_image],
    }


def test_legacy_multimodal_merged_handoff_preserves_original_blocks():
    """Persisted pre-delimiter list handoffs must not lose their real tail."""
    prior_text = {"type": "text", "text": "legacy real tail"}
    prior_image = {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,BBBB"},
    }
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    f"{SUMMARY_PREFIX}\nlegacy stale summary\n\n"
                    f"{_SUMMARY_END_MARKER}\n\n"
                ),
            },
            prior_text,
            prior_image,
        ],
        COMPRESSED_SUMMARY_METADATA_KEY: True,
    }

    result = ContextCompressor._strip_context_summary_handoff_message(message)

    assert result == {
        "role": "user",
        "content": [prior_text, prior_image],
    }


def test_resume_handoff_in_protected_head_is_not_preserved_as_fossil():
    """After restart, a persisted handoff summary should decay head protection."""
    compressor = _compressor()
    old_summary = "RESTART-FOSSIL-SUMMARY durable facts from before restart"

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")):
        result = compressor.compress(_messages_with_handoff(old_summary))

    # Main's task-snapshot grounding (761a0b124e) prepends a deterministic
    # "## Historical Task Snapshot" section to the stored summary — pin the
    # contract (fresh body present, fossil absent), not the exact string.
    stored_summary = compressor._previous_summary or ""
    assert stored_summary.endswith("fresh summary")
    assert old_summary not in stored_summary
    summary_messages = [
        msg for msg in result
        if ContextCompressor._has_compressed_summary_metadata(msg)
        or ContextCompressor._is_context_summary_content(msg.get("content"))
    ]
    assert len(summary_messages) == 1
    assert all(
        old_summary not in str(msg.get("content", ""))
        for msg in result
    )


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


def test_tail_summary_marker_does_not_decay_first_compaction_head():
    """A live tail summary-looking message should not mimic a resumed handoff."""
    compressor = _compressor(protect_first_n=3)
    tail_summary = "TAIL-SUMMARY-LIKE message belongs to current protected tail"
    msgs = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "HEAD-ONE original request"},
        {"role": "assistant", "content": "HEAD-TWO original answer"},
        {"role": "user", "content": "HEAD-THREE original follow-up"},
        {"role": "assistant", "content": "middle answer one"},
        {"role": "user", "content": "middle request two"},
        {"role": "assistant", "content": "middle answer two"},
        {"role": "user", "content": "middle request three"},
        {"role": "assistant", "content": "middle answer three"},
        {"role": "user", "content": "middle request four"},
        {"role": "assistant", "content": f"{SUMMARY_PREFIX}\n{tail_summary}"},
        {"role": "user", "content": "final active request stays in protected tail"},
    ]

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")):
        result = compressor.compress(msgs)

    result_text = "\n".join(str(msg.get("content", "")) for msg in result)
    assert "HEAD-ONE original request" in result_text
    assert "HEAD-TWO original answer" in result_text
    assert "HEAD-THREE original follow-up" in result_text
    assert tail_summary not in result_text


def test_restart_handoff_in_protected_tail_is_folded_not_preserved():
    """Short resumed transcripts should not copy old summaries as tail."""
    compressor = _compressor(protect_first_n=3)
    old_summary = "TAIL-PROTECTED-OLD-SUMMARY durable facts"

    msgs = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": "original answer"},
        {"role": "user", "content": "original follow-up"},
        {"role": "assistant", "content": f"{SUMMARY_PREFIX}\n{old_summary}"},
        {"role": "user", "content": "active request"},
    ]

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")) as mock_call:
        result = compressor.compress(msgs)

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert prompt.count(old_summary) == 1
    result_text = "\n".join(str(msg.get("content", "")) for msg in result)
    assert old_summary not in result_text
    assert "active request" in result_text
    assert sum(
        1 for msg in result if ContextCompressor._is_context_summary_message(msg)
    ) == 1


def test_restart_handoff_fallback_preserves_rehydrated_summary_body():
    """Deterministic fallback should retain the rehydrated old summary."""
    compressor = _compressor(protect_first_n=3)
    old_summary = "FALLBACK-OLD-SUMMARY durable fact must survive"

    with patch.object(compressor, "_generate_summary", return_value=None):
        result = compressor.compress(_messages_with_default_handoff(old_summary))

    result_text = "\n".join(str(msg.get("content", "")) for msg in result)
    assert result_text.count(old_summary) == 1
    assert sum(
        1 for msg in result if ContextCompressor._is_context_summary_message(msg)
    ) == 1


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


def test_fossil_beyond_restart_probe_window_is_still_folded():
    """Self-heal should find summaries that drift past the decay probe."""
    compressor = _compressor(protect_first_n=1)
    old_summary = "OLD-SUMMARY-FAR-FROM-HEAD durable facts"
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

    assert compressor._effective_protect_first_n(msgs) == compressor.protect_first_n

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")):
        result = compressor.compress(msgs)

    assert all(old_summary not in str(msg.get("content", "")) for msg in result)


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


def test_tail_turns_before_late_handoff_are_not_lost():
    """Live tail turns before a late handoff should be summarized or kept."""
    compressor = _compressor(protect_first_n=3)
    old_summary = "LATE-TAIL-OLD-SUMMARY"
    msgs = [{"role": "system", "content": "system prompt"}]
    msgs += [
        {
            "role": "user" if idx % 2 else "assistant",
            "content": f"body {idx}",
        }
        for idx in range(1, 9)
    ]
    msgs += [
        {"role": "assistant", "content": "TAIL-BEFORE-SUMMARY-A"},
        {"role": "user", "content": "TAIL-BEFORE-SUMMARY-B"},
        {"role": "assistant", "content": f"{SUMMARY_PREFIX}\n{old_summary}"},
        {"role": "user", "content": "final active request"},
    ]

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")) as mock_call:
        result = compressor.compress(msgs)

    preserved = mock_call.call_args.kwargs["messages"][0]["content"] + "\n" + "\n".join(
        str(msg.get("content", "")) for msg in result
    )
    assert "TAIL-BEFORE-SUMMARY-A" in preserved
    assert "TAIL-BEFORE-SUMMARY-B" in preserved


def test_forced_leading_merged_summary_strips_live_tail_from_summary_body():
    """Rehydrating a forced-leading merged summary should ignore live tail."""
    merged = (
        f"{SUMMARY_PREFIX}\nSUMMARY_BODY\n\n"
        f"{_SUMMARY_END_MARKER}\n\n"
        "LIVE_TAIL_REQUEST"
    )

    assert ContextCompressor._is_context_summary_content(merged) is True
    assert ContextCompressor._strip_summary_prefix(merged) == "SUMMARY_BODY"


def test_restart_probe_boundary_summary_just_inside_window_decays():
    """A summary at the last restart-probe index should still decay."""
    compressor = _compressor(protect_first_n=3)
    first_non_system = 1
    last_probe_idx = (
        first_non_system
        + compressor.protect_first_n
        + _RESTART_HANDOFF_PROBE_EXTRA_MESSAGES
        - 1
    )

    assert (
        compressor._effective_protect_first_n(
            _messages_with_summary_at_index(last_probe_idx)
        )
        == 0
    )


def test_restart_probe_boundary_summary_just_outside_window_does_not_decay():
    """A summary past the restart-probe window should not decay."""
    compressor = _compressor(protect_first_n=3)
    first_non_system = 1
    first_outside_probe_idx = (
        first_non_system
        + compressor.protect_first_n
        + _RESTART_HANDOFF_PROBE_EXTRA_MESSAGES
    )

    assert (
        compressor._effective_protect_first_n(
            _messages_with_summary_at_index(first_outside_probe_idx)
        )
        == compressor.protect_first_n
    )


def test_restart_stacked_handoffs_fold_stray_head_and_collapse_to_single_summary():
    """Stacked restart summaries should keep stray head turns as new input."""
    compressor = _compressor(protect_first_n=3)
    old_summary = "OLD-ONLY facts from the first compaction"
    newer_summary = "NEW-ONLY facts from work after restart"

    msgs = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "FOSSIL-HEAD-TURN live detail before summary"},
        {"role": "assistant", "content": f"{SUMMARY_PREFIX}\n{old_summary}"},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\n{newer_summary}"},
        {"role": "assistant", "content": "work after restart"},
        {"role": "user", "content": "more work after restart"},
        {"role": "assistant", "content": "tail answer"},
        {"role": "user", "content": "active tail request"},
    ]

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")) as mock_call:
        result = compressor.compress(msgs)

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert prompt.count(old_summary) == 1
    assert prompt.count(newer_summary) == 1
    assert "FOSSIL-HEAD-TURN live detail before summary" in prompt
    assert f"[ASSISTANT]: {SUMMARY_PREFIX}" not in prompt
    assert f"[USER]: {SUMMARY_PREFIX}" not in prompt
    summary_messages = [
        msg for msg in result
        if ContextCompressor._is_context_summary_message(msg)
    ]
    assert len(summary_messages) == 1
    assert all(old_summary not in str(msg.get("content", "")) for msg in result)
    assert all(newer_summary not in str(msg.get("content", "")) for msg in result)
    # The stray head turn must be folded into the summary, not preserved as
    # its own verbatim message. Main's task-snapshot grounding (761a0b124e)
    # may legitimately QUOTE it inside the summary handoff as the
    # deterministic "User asked" anchor, so only non-summary messages are
    # checked for the verbatim fossil.
    assert all(
        "FOSSIL-HEAD-TURN" not in str(msg.get("content", ""))
        for msg in result
        if not ContextCompressor._is_context_summary_message(msg)
    )


def test_metadata_summary_decay_also_rehydrates_previous_summary():
    """Metadata-only in-process summaries should decay and rehydrate together."""
    compressor = _compressor(protect_first_n=3)

    msgs = [
        {"role": "system", "content": "system prompt"},
        {
            "role": "assistant",
            "content": "metadata-only prior summary",
            COMPRESSED_SUMMARY_METADATA_KEY: True,
        },
        {"role": "user", "content": "new work"},
        {"role": "assistant", "content": "new answer"},
        {"role": "user", "content": "tail request"},
        {"role": "assistant", "content": "tail answer"},
    ]

    with patch("agent.context_compressor.call_llm", return_value=_response("fresh summary")) as mock_call:
        compressor.compress(msgs)

    prompt = mock_call.call_args.kwargs["messages"][0]["content"]
    assert "PREVIOUS SUMMARY:" in prompt
    assert "metadata-only prior summary" in prompt
    # Grounding may prepend a task-snapshot section; pin the fresh body.
    assert (compressor._previous_summary or "").endswith("fresh summary")
