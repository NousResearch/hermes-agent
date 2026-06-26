"""Tests for the compression infinite-loop fix (#53008).

When the auxiliary compression model's context window is smaller than the
main model's compression threshold, ``check_compression_model_feasibility``
auto-lowers ``threshold_tokens`` to the aux model's context size.  If the
compression window (the middle turns sent to the summariser) then grows
beyond the aux model's context, the aux model cannot process it — the API
call errors out or is silently truncated, producing a near-useless summary.
Each pass is near-zero effective but the session remains above the threshold,
looping indefinitely (the reporter observed 12–16 consecutive compressions).

Two-layer fix:

1. **Proactive main-model fallback** (root fix): ``compress()`` estimates
   the token count of ``turns_to_summarize`` (the actual content sent to
   the summariser) and compares it with ``_aux_compression_context_length``.
   When the window exceeds the aux model's context, the aux model is
   temporarily swapped out for the main model for that pass.  The main
   model has enough context to summarise effectively, breaking the loop
   at the root.  The aux model is restored afterwards so future passes
   (on a smaller post-compression session) can use it again.

2. **Post-compression re-trigger guard** (safety net): ``should_compress()``
   records ``_post_compression_rough_tokens`` after every pass.  If the last
   pass left the session at or above the threshold AND not enough new content
   has arrived, compression is skipped.  This catches edge cases where even
   the main model cannot reduce the session below the threshold.
"""

from unittest.mock import patch

from agent.context_compressor import ContextCompressor, estimate_messages_tokens_rough


def _make_compressor(**kwargs) -> ContextCompressor:
    defaults = dict(
        model="test-model",
        threshold_percent=0.50,
        protect_first_n=2,
        protect_last_n=3,
        quiet_mode=True,
    )
    defaults.update(kwargs)
    with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
        return ContextCompressor(**defaults)


def _build_session(n_turns: int, chars_per_msg: int = 200) -> list:
    """Build a multi-turn conversation with controllable size."""
    base = " ".join(["x"] * chars_per_msg)
    messages = [{"role": "system", "content": "You are helpful."}]
    for i in range(n_turns):
        messages.append({"role": "user", "content": f"{base} turn {i}"})
        messages.append({"role": "assistant", "content": f"{base} reply {i}"})
    return messages


# ---------------------------------------------------------------------------
# Layer 1: proactive main-model fallback in compress()
# ---------------------------------------------------------------------------


class TestProactiveMainModelFallback:
    """compress() falls back to the main model when the compression window
    (middle turns sent to the summariser) exceeds the aux model's context."""

    def test_falls_back_when_window_exceeds_aux_context(self):
        """When the compression window > _aux_compression_context_length,
        compress() temporarily clears summary_model so _generate_summary
        uses the main model."""
        comp = _make_compressor()
        comp.summary_model = "aux-small-model"
        # Set aux context very low so even a small window exceeds it
        comp._aux_compression_context_length = 500

        messages = _build_session(30, chars_per_msg=200)

        captured_summary_model = []

        def _capture_summary(*args, **kwargs):
            captured_summary_model.append(comp.summary_model)
            return "Summary of conversation."

        with patch.object(comp, "_generate_summary", side_effect=_capture_summary):
            comp.compress(messages, current_tokens=50_000)

        assert captured_summary_model == [""], (
            "summary_model should be empty (main model) during _generate_summary "
            f"when window > aux_context, got {captured_summary_model}"
        )

    def test_restores_aux_model_after_fallback(self):
        """After the proactive fallback pass, summary_model is restored so
        future passes (on a smaller session) can use the aux model again."""
        comp = _make_compressor()
        comp.summary_model = "aux-small-model"
        comp._aux_compression_context_length = 500

        messages = _build_session(30, chars_per_msg=200)

        with patch.object(comp, "_generate_summary", return_value="Summary."):
            comp.compress(messages, current_tokens=50_000)

        assert comp.summary_model == "aux-small-model", (
            "summary_model should be restored after the fallback pass"
        )

    def test_no_fallback_when_window_within_aux_context(self):
        """When the compression window <= _aux_compression_context_length,
        the aux model is used as-is (no fallback)."""
        comp = _make_compressor()
        comp.summary_model = "aux-small-model"
        # Set aux context very high so the window fits
        comp._aux_compression_context_length = 10_000_000

        messages = _build_session(30, chars_per_msg=200)

        captured = []

        def _capture(*args, **kwargs):
            captured.append(comp.summary_model)
            return "Summary."

        with patch.object(comp, "_generate_summary", side_effect=_capture):
            comp.compress(messages, current_tokens=50_000)

        assert captured == ["aux-small-model"], (
            "aux model should be used when window fits within its context"
        )

    def test_no_fallback_when_no_aux_model_configured(self):
        """When summary_model is empty (no aux model), no fallback happens."""
        comp = _make_compressor()
        comp.summary_model = ""
        comp._aux_compression_context_length = 500

        messages = _build_session(30, chars_per_msg=200)

        with patch.object(comp, "_generate_summary", return_value="Summary."):
            comp.compress(messages, current_tokens=50_000)

        assert comp.summary_model == ""

    def test_no_fallback_when_aux_context_not_set(self):
        """When _aux_compression_context_length is 0 (feasibility check not
        run), no fallback happens — preserves existing behavior."""
        comp = _make_compressor()
        comp.summary_model = "aux-model"
        comp._aux_compression_context_length = 0

        messages = _build_session(30, chars_per_msg=200)

        with patch.object(comp, "_generate_summary", return_value="Summary."):
            comp.compress(messages, current_tokens=50_000)

        assert comp.summary_model == "aux-model"

    def test_no_fallback_when_session_large_but_window_small(self):
        """The comparison is against the compression window, NOT the full
        session.  A session can be much larger than the aux context while
        the window (middle turns) still fits — in that case the aux model
        is used, not the main model."""
        comp = _make_compressor()
        comp.summary_model = "aux-small-model"
        # Aux context is 50K — larger than the window but smaller than the
        # full session (which we claim is 200K via current_tokens)
        comp._aux_compression_context_length = 50_000

        messages = _build_session(30, chars_per_msg=200)

        captured = []

        def _capture(*args, **kwargs):
            captured.append(comp.summary_model)
            return "Summary."

        with patch.object(comp, "_generate_summary", side_effect=_capture):
            # current_tokens=200_000 (full session) but the actual window
            # is much smaller (only ~30 turns of ~200 chars each)
            comp.compress(messages, current_tokens=200_000)

        assert captured == ["aux-small-model"], (
            "aux model should be used — the window fits even though the "
            f"full session (200K) exceeds the aux context (50K). Got {captured}"
        )

    def test_overflow_warning_emitted_once(self):
        """The overflow warning is emitted at most once per session."""
        comp = _make_compressor(quiet_mode=False)
        comp.summary_model = "aux-small-model"
        comp._aux_compression_context_length = 500

        messages = _build_session(30, chars_per_msg=200)

        with patch.object(comp, "_generate_summary", return_value="Summary."):
            comp.compress(messages, current_tokens=50_000)
            assert comp._aux_context_overflow_warned is True
            # Second pass — warning flag already set, no duplicate
            comp.compress(messages, current_tokens=50_000)

        assert comp._aux_context_overflow_warned is True


# ---------------------------------------------------------------------------
# Layer 2: post-compression re-trigger guard in should_compress()
# ---------------------------------------------------------------------------


class TestPostCompressionRetriggerGuard:
    """should_compress() blocks re-triggering when the last pass left the
    session at or above the threshold with insufficient new content."""

    def test_blocks_when_last_pass_left_session_above_threshold(self):
        comp = _make_compressor()
        comp._post_compression_rough_tokens = 120_000
        assert not comp.should_compress(122_000), (
            "should_compress should return False — last pass left session "
            "above threshold and only 2K new tokens (< 20K floor)"
        )

    def test_allows_when_significant_growth_since_last_pass(self):
        comp = _make_compressor()
        comp._post_compression_rough_tokens = 120_000
        assert comp.should_compress(145_000), (
            "should_compress should return True — 25K new tokens >= 20K floor"
        )

    def test_allows_when_last_pass_brought_session_below_threshold(self):
        comp = _make_compressor()
        comp._post_compression_rough_tokens = 80_000  # below threshold 100,000
        assert comp.should_compress(120_000)

    def test_allows_when_no_prior_compression(self):
        comp = _make_compressor()
        comp._post_compression_rough_tokens = 0
        assert comp.should_compress(120_000)

    def test_growth_floor_is_twenty_percent_of_threshold(self):
        """Growth floor = max(threshold // 5, 4096). For 100K threshold → 20K."""
        comp = _make_compressor()
        comp._post_compression_rough_tokens = 120_000
        assert not comp.should_compress(139_999)  # 19,999 growth → blocked
        assert comp.should_compress(140_000)      # 20,000 growth → allowed

    def test_growth_floor_has_minimum_of_4096(self):
        """For very small thresholds, growth floor is at least 4096."""
        with patch("agent.context_compressor.get_model_context_length", return_value=10_000):
            comp = ContextCompressor(
                model="tiny-model",
                threshold_percent=0.50,
                protect_first_n=2,
                protect_last_n=3,
                quiet_mode=True,
            )
        threshold = comp.threshold_tokens
        comp._post_compression_rough_tokens = threshold + 1_000
        assert not comp.should_compress(threshold + 5_000)    # 4,000 growth → blocked
        assert comp.should_compress(threshold + 5_096)        # 4,096 growth → allowed

    def test_anti_thrashing_takes_precedence(self):
        """Anti-thrashing (ineffective_count >= 2) blocks regardless of guard."""
        comp = _make_compressor()
        comp._ineffective_compression_count = 2
        comp._post_compression_rough_tokens = 0
        assert not comp.should_compress(120_000)

    def test_guard_resets_after_session_reset(self):
        """on_session_reset clears _post_compression_rough_tokens."""
        comp = _make_compressor()
        comp._post_compression_rough_tokens = 120_000
        comp.on_session_reset()
        assert comp._post_compression_rough_tokens == 0
        assert comp.should_compress(120_000)


# ---------------------------------------------------------------------------
# compress(): _post_compression_rough_tokens is set after each pass
# ---------------------------------------------------------------------------


class TestPostCompressionTokensRecorded:
    """compress() records _post_compression_rough_tokens after every pass."""

    def test_compression_sets_post_compression_tokens(self):
        comp = _make_compressor()
        messages = _build_session(30, chars_per_msg=200)

        with patch.object(comp, "_generate_summary", return_value="Summary of conversation."):
            result = comp.compress(messages, current_tokens=50_000)

        assert comp._post_compression_rough_tokens > 0
        expected = estimate_messages_tokens_rough(result)
        assert comp._post_compression_rough_tokens == expected

    def test_noop_compression_sets_post_compression_tokens(self):
        """When compress_start >= compress_end (no-op), post-compression tokens
        are set to the pre-compression estimate so the guard fires."""
        comp = _make_compressor()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        comp.compress(messages, current_tokens=5_000)
        # Too few messages → early return before the no-op-window path.
        assert comp._post_compression_rough_tokens == 0

    def test_abort_sets_post_compression_tokens(self):
        """When compression is aborted (summary failure), post-compression
        tokens are set so the guard fires on the next turn."""
        comp = _make_compressor(abort_on_summary_failure=True)
        messages = _build_session(30, chars_per_msg=200)

        with patch.object(comp, "_generate_summary", return_value=None):
            comp.compress(messages, current_tokens=50_000)

        assert comp._post_compression_rough_tokens > 0


# ---------------------------------------------------------------------------
# Integration: the full #53008 scenario — proactive fallback breaks the loop
# ---------------------------------------------------------------------------


class TestInfiniteLoopPrevention:
    """Simulate the #53008 scenario end-to-end: threshold lowered below
    session size, proactive fallback uses the main model, compression is
    effective, no infinite loop."""

    def test_proactive_fallback_breaks_infinite_loop(self):
        """Simulate: aux context = 131K, threshold lowered to 131K.  Build
        a session large enough that the compression window exceeds 131K.
        The proactive fallback swaps to the main model for the summary,
        which has enough context to compress effectively.  After one pass
        the session is below threshold and should_compress returns False."""
        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            comp = ContextCompressor(
                model="big-main-model",
                threshold_percent=0.131,  # 131K — simulates auto-lowered
                protect_first_n=2,
                protect_last_n=3,
                quiet_mode=True,
            )
        comp.summary_model = "aux-small-model"
        comp._aux_compression_context_length = 131_000

        threshold = comp.threshold_tokens
        assert 130_000 <= threshold <= 132_000

        # Build a session large enough that the compression window > 131K.
        # tail_token_budget = 131K * 0.20 = 26K, soft_ceiling = 39K.
        # So the tail is ~26-39K tokens.  We need the middle window > 131K,
        # so the total session needs to be > 131K + 39K + head ≈ 175K.
        # At ~500 tokens per message (2000 chars), we need ~350 messages
        # in the middle, so ~360 turns total.
        messages = _build_session(360, chars_per_msg=2000)

        session_tokens = estimate_messages_tokens_rough(messages)
        assert session_tokens > 175_000, (
            f"Session must be > 175K for the window to exceed 131K, "
            f"got {session_tokens:,}"
        )

        # Verify the compression window actually exceeds the aux context
        compress_start = comp._protect_head_size(messages)
        compress_start = comp._align_boundary_forward(messages, compress_start)
        compress_end = comp._find_tail_cut_by_tokens(messages, compress_start)
        window = messages[compress_start:compress_end]
        window_tokens = estimate_messages_tokens_rough(window)
        assert window_tokens > 131_000, (
            f"Compression window ({window_tokens:,}) must exceed aux context "
            f"(131,000) for this test to be meaningful"
        )

        # First compression should trigger
        assert comp.should_compress(session_tokens)

        # Track which model was used for the summary
        used_models = []

        def _track_model(*args, **kwargs):
            used_models.append(comp.summary_model)
            return "Effective summary of the conversation. " * 100

        with patch.object(comp, "_generate_summary", side_effect=_track_model):
            result = comp.compress(messages, current_tokens=session_tokens)

        # The main model was used (proactive fallback), not the aux model
        assert used_models == [""], (
            f"Main model should have been used for the summary, got {used_models}"
        )

        # The aux model is restored after the pass
        assert comp.summary_model == "aux-small-model"

        # After effective compression, the session should be below threshold
        new_tokens = estimate_messages_tokens_rough(result)
        assert new_tokens < threshold, (
            f"Post-compression session ({new_tokens:,}) should be below "
            f"threshold ({threshold:,}) — effective compression"
        )
        assert not comp.should_compress(new_tokens), (
            "should_compress should return False — session now below threshold"
        )

    def test_guard_catches_edge_case_where_main_model_also_fails(self):
        """Edge case: if even the main model produces an ineffective
        compression (session stays above threshold), the re-trigger guard
        prevents the loop."""
        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            comp = ContextCompressor(
                model="big-main-model",
                threshold_percent=0.131,
                protect_first_n=2,
                protect_last_n=3,
                quiet_mode=True,
            )
        comp.summary_model = "aux-small-model"
        comp._aux_compression_context_length = 131_000

        # Simulate an ineffective compression: 205K → 204K
        comp._post_compression_rough_tokens = 204_000
        comp._ineffective_compression_count = 1

        assert not comp.should_compress(204_000), (
            "Guard should block — last pass left 204K >= threshold and "
            "no new content arrived"
        )

        # After significant growth (30K), compression re-triggers
        assert comp.should_compress(234_000), (
            "Guard should allow — 30K growth >= 26K floor"
        )


# ---------------------------------------------------------------------------
# Integration: simulate the turn_context.py multi-pass preflight loop
# (turn_context.py:383-410) — the exact code path that loops infinitely
# in #53008.  Up to 3 compression passes per turn, breaking when
# _compression_made_progress returns False or should_compress returns False.
# ---------------------------------------------------------------------------


class TestTurnContextMultiPassLoop:
    """Simulate the turn_context.py preflight multi-pass loop (up to 3
    passes per turn) and verify the proactive fallback + guard prevent
    the infinite loop across multiple turns."""

    def test_multi_pass_loop_terminates_with_proactive_fallback(self):
        """Simulate 5 turns, each with the preflight multi-pass loop.
        The proactive fallback ensures the main model is used when the
        compression window exceeds aux context, producing effective
        compression.  The loop terminates within 3 passes each turn."""
        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            comp = ContextCompressor(
                model="big-main-model",
                threshold_percent=0.131,  # 131K — simulates auto-lowered
                protect_first_n=2,
                protect_last_n=3,
                quiet_mode=True,
            )
        comp.summary_model = "aux-small-model"
        comp._aux_compression_context_length = 131_000

        # Build a session large enough that the window > 131K
        messages = _build_session(360, chars_per_msg=2000)
        current_tokens = estimate_messages_tokens_rough(messages)

        summary_calls = []

        def _track_and_summarise(*args, **kwargs):
            summary_calls.append(comp.summary_model)
            return "Concise summary of the conversation."

        with patch.object(comp, "_generate_summary", side_effect=_track_and_summarise):
            for turn in range(5):
                pass_count = 0
                while pass_count < 3:
                    if not comp.should_compress(current_tokens):
                        break
                    pass_count += 1
                    orig_len = len(messages)
                    orig_tokens = current_tokens
                    messages = comp.compress(messages, current_tokens=current_tokens)
                    current_tokens = estimate_messages_tokens_rough(messages)
                    # Check progress (same logic as turn_context.py)
                    if orig_len == len(messages) and orig_tokens > 0 and current_tokens >= orig_tokens * 0.95:
                        break  # No progress
                    if not comp.should_compress(current_tokens):
                        break

                assert pass_count <= 3, (
                    f"Turn {turn}: exceeded 3 passes — infinite loop"
                )

                # Add new content for the next turn
                messages.append({"role": "user", "content": " ".join(["x"] * 2000) + f" new turn {turn}"})
                messages.append({"role": "assistant", "content": " ".join(["x"] * 2000) + f" new reply {turn}"})
                current_tokens = estimate_messages_tokens_rough(messages)

        # The proactive fallback was used at least once (main model, not aux)
        assert "" in summary_calls, (
            f"Main model should have been used at least once, got {summary_calls}"
        )

        # The aux model was restored after each fallback pass
        assert comp.summary_model == "aux-small-model"

    def test_guard_prevents_loop_when_compression_always_ineffective(self):
        """Edge case: if compression is always ineffective, the re-trigger
        guard ensures the multi-pass loop terminates and does not re-trigger
        on the next turn."""
        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            comp = ContextCompressor(
                model="big-main-model",
                threshold_percent=0.131,
                protect_first_n=2,
                protect_last_n=3,
                quiet_mode=True,
            )
        comp.summary_model = "aux-small-model"
        comp._aux_compression_context_length = 131_000

        comp._post_compression_rough_tokens = 204_000
        comp._ineffective_compression_count = 1

        assert not comp.should_compress(204_000), (
            "Guard should block on turn 1 — last pass left session above "
            "threshold with no growth"
        )
        assert not comp.should_compress(204_500), (
            "Guard should block on turn 2 — only 500 tokens growth < 26K floor"
        )
        assert comp.should_compress(235_000), (
            "Guard should allow on turn 3 — 31K growth >= 26K floor"
        )
