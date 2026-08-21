"""The anti-thrashing guard must actually fire.

Two defects made it dead code:

1. Effectiveness was scored as ``(current_tokens - estimate(compressed)) / current_tokens``.
   ``current_tokens`` is the provider's FULL prompt (system prompt + tool schemas +
   messages); ``estimate(compressed)`` covers messages only. The mixed basis reported
   ~96% savings on every pass, so ``_ineffective_compression_count`` reset each time.

2. Even scored correctly, message shrinkage is the wrong yardstick. ``should_compress()``
   trips on the full prompt, but compaction can only shrink messages. When the
   incompressible floor alone meets the threshold, each pass can shrink messages by a
   healthy margin, reset the counter, and still leave the prompt over the line -- so the
   next turn compacts again, forever.

Together these made a mis-sized context window present as a hung CLI rather than a
warning. Effectiveness is now scored against the goal -- did the prompt get under the
threshold? -- judged in ``update_from_response()``, the one place that sees the
provider's real prompt count for the just-compacted conversation.

Two subtleties this pins:

* The verdict must NOT live in ``should_compress()``. That runs twice per turn with
  two different measures (a rough preflight estimate and the real post-response count,
  #36718); the rough one can dip below the threshold and reset the strike every turn,
  re-opening the loop. ``test_rough_preflight_reading_does_not_reopen_the_loop``.

* It must not be computed analytically as ``floor = current_tokens - rough_estimate``.
  That subtracts an estimate from a real count, absorbs the tokenizer skew into "floor",
  and disables compaction on a healthy session.
  ``test_no_false_positive_under_tokenizer_skew``.
"""
import pytest

from agent.context_compressor import ContextCompressor


def _compressor(threshold_tokens: int) -> ContextCompressor:
    cc = ContextCompressor(
        model="test-model",
        threshold_percent=0.75,
        protect_first_n=5,
        protect_last_n=20,
        quiet_mode=True,
        config_context_length=40960,
        provider="test",
    )
    cc.threshold_tokens = threshold_tokens  # pin; don't couple to window math
    cc._generate_summary = lambda *a, **k: "Summary of earlier turns."
    return cc


def _messages(n: int, size: int = 1500) -> list:
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"m{i} " + "z" * size})
    return msgs


def _turn(cc, msgs, real_prompt_tokens):
    """One agent turn as conversation_loop drives it.

    should_compress() gates on the real prompt count; if it fires, compress()
    runs and the provider then reports real usage for the shorter conversation.
    The anti-thrashing verdict is judged in update_from_response() against that
    real count -- not inside should_compress(), which is called twice per turn
    with two different measures. Returns (msgs, did_compact).
    """
    if not cc.should_compress(real_prompt_tokens):
        return msgs, False
    msgs = cc.compress(msgs, current_tokens=real_prompt_tokens)
    cc._verify_compaction_cleared_threshold = True
    cc.update_from_response({"prompt_tokens": real_prompt_tokens})
    return msgs, True


class TestSavingsBasis:
    def test_savings_does_not_depend_on_current_tokens(self):
        """Savings is a property of the messages, not of the caller's token count.

        Before the fix, passing the full-prompt count made an identical compaction
        report a wildly higher savings percentage.
        """
        msgs = _messages(14)

        a = _compressor(threshold_tokens=24_576)
        a.compress([m.copy() for m in msgs], current_tokens=100_000)
        with_full_prompt = a._last_compression_savings_pct

        b = _compressor(threshold_tokens=24_576)
        b.compress([m.copy() for m in msgs], current_tokens=None)
        messages_only = b._last_compression_savings_pct

        assert with_full_prompt == pytest.approx(messages_only, abs=0.01), (
            "savings must be scored on a messages-vs-messages basis; got "
            f"{with_full_prompt:.1f}% with a full-prompt count vs "
            f"{messages_only:.1f}% without"
        )

    def test_no_op_compaction_is_not_reported_as_huge_savings(self):
        cc = _compressor(threshold_tokens=24_576)
        msgs = _messages(4, size=10)  # below the minimum compressible size
        cc.compress(msgs, current_tokens=100_000)
        assert cc._last_compression_savings_pct < 50


class TestFutilityGuard:
    def test_stops_when_floor_alone_meets_threshold(self):
        """Incompressible floor >= threshold -> shrinking messages cannot help."""
        cc = _compressor(threshold_tokens=24_576)
        msgs = _messages(14)
        # Full prompt is far above threshold and dominated by system + tools,
        # so the real count never drops no matter how the message list shrinks.
        real_prompt_tokens = 33_564

        fired = 0
        for _ in range(6):
            msgs, did = _turn(cc, msgs, real_prompt_tokens)
            if did:
                fired += 1
                msgs.append({"role": "user", "content": "next " + "w" * 4000})

        assert cc._ineffective_compression_count >= 2
        assert not cc.should_compress(real_prompt_tokens), (
            "compaction that cannot clear the threshold must stop"
        )
        assert fired <= 3, f"expected the loop to break early, compacted {fired}x"


    def test_effective_compaction_still_resets_the_counter(self):
        """A compaction that gets the prompt under the threshold is not thrashing."""
        cc = _compressor(threshold_tokens=750_000)
        msgs = _messages(120, size=2000)
        before = len(msgs)
        out = cc.compress(msgs, current_tokens=None)

        assert len(out) < before, "a long transcript must still compact"
        assert cc._last_compression_savings_pct > 10
        assert cc._last_compression_made_progress is True
        assert cc._ineffective_compression_count == 0

    def test_no_false_positive_under_tokenizer_skew(self):
        """The provider's real count exceeds the rough estimate; that is not a floor.

        An analytic `floor = real_prompt - rough_estimate` misreads tokenizer skew
        as incompressible overhead and disables compaction on a healthy session.
        The check must compare the caller's own measure on both sides.
        """
        from agent.context_compressor import estimate_messages_tokens_rough

        skew, floor = 1.6, 30_000
        cc = _compressor(threshold_tokens=150_000)
        msgs = _messages(160, size=2500)

        real_prompt = floor + int(skew * estimate_messages_tokens_rough(msgs))
        assert cc.should_compress(real_prompt)
        msgs = cc.compress(msgs, current_tokens=real_prompt)
        real_after = floor + int(skew * estimate_messages_tokens_rough(msgs))
        cc._verify_compaction_cleared_threshold = True
        cc.update_from_response({"prompt_tokens": real_after})  # provider's real count

        assert real_after < cc.threshold_tokens, "compaction really did clear it"
        assert cc._ineffective_compression_count == 0, (
            "tokenizer skew must not be mistaken for an incompressible floor"
        )






    def test_model_switch_resets_and_persists_fallback_streak(self, tmp_path):
        from hermes_state import SessionDB

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s1", source="cli")
        cc = _compressor(threshold_tokens=24_576)
        cc.bind_session_state(db, "s1")
        cc.record_completed_compaction(used_fallback=True)

        cc.update_model("next-model", 100_000)

        assert cc._fallback_compression_streak == 0
        assert db.get_compression_fallback_streak("s1") == 0




class TestMinimumMessagesBranch:
    def test_too_few_messages_records_an_ineffective_pass(self):
        """Returning the transcript unchanged must move the anti-thrash state.

        Otherwise should_compress() keeps saying True about a transcript that can
        never shrink, and every turn re-enters a no-op compaction.
        """
        cc = _compressor(threshold_tokens=1)
        msgs = _messages(3, size=10)
        before = cc._ineffective_compression_count

        out = cc.compress(msgs, current_tokens=100_000)

        assert len(out) == len(msgs), "nothing should have been compressed"
        assert cc._last_compression_made_progress is False
        assert cc._ineffective_compression_count == before + 1


class TestRejectedCompactionStrike:
    """#88568 — a would-grow refusal must count as an ineffective strike.

    The anti-growth guard correctly keeps the original transcript, but the
    rejection used to leave ``_ineffective_compression_count`` untouched, so
    the breaker never latched and automatic compression retried the SAME
    unchanged transcript on every turn.
    """

    def test_rejected_compaction_increments_strike(self):
        cc = _compressor(threshold_tokens=1)
        assert cc._ineffective_compression_count == 0

        cc.record_rejected_compaction()

        assert cc._ineffective_compression_count == 1

    def test_two_rejections_stop_further_automatic_compression(self):
        cc = _compressor(threshold_tokens=1)
        cc.record_rejected_compaction()
        cc.record_rejected_compaction()

        # The latch consumers key on the counter itself (>= 2 blocks);
        # pin the counter and the recovery-clock arming side effect.
        assert cc._ineffective_compression_count >= 2

    def test_rejection_does_not_arm_real_usage_verification(self):
        """Nothing was committed, so the next response must not be scored
        against the pre-rejection transcript (that verdict belongs to
        committed compactions only)."""
        cc = _compressor(threshold_tokens=1)
        cc.record_rejected_compaction()

        assert cc._verify_compaction_cleared_threshold is False

    def test_rejection_leaves_fallback_streak_untouched(self):
        cc = _compressor(threshold_tokens=1)
        cc._fallback_compression_streak = 1

        cc.record_rejected_compaction()

        assert cc._fallback_compression_streak == 1
