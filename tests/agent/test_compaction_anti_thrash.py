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
threshold? -- and the check is made in ``should_compress()`` against the caller's own
token count on both sides.

That last detail matters. An analytic ``floor = current_tokens - rough_estimate(messages)``
looks equivalent and is not: it silently absorbs the skew between the rough estimator and
the provider's real tokenizer, reports it as incompressible overhead, and disables
compaction on a perfectly healthy session. ``test_no_false_positive_under_tokenizer_skew``
pins that.
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
        # so it never drops no matter how much the message list shrinks.
        real_prompt_tokens = 33_564

        fired = 0
        for _ in range(6):
            if cc.should_compress(real_prompt_tokens):
                fired += 1
                msgs = cc.compress(msgs, current_tokens=real_prompt_tokens)
                msgs.append({"role": "user", "content": "next " + "w" * 4000})

        assert cc._ineffective_compression_count >= 2
        assert not cc.should_compress(real_prompt_tokens), (
            "compaction that cannot clear the threshold must stop and warn"
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
        real_prompt = floor + int(skew * estimate_messages_tokens_rough(msgs))

        assert real_prompt < cc.threshold_tokens, "compaction really did clear it"
        assert not cc.should_compress(real_prompt)
        assert cc._ineffective_compression_count == 0, (
            "tokenizer skew must not be mistaken for an incompressible floor"
        )

    def test_counter_resets_once_the_prompt_fits_again(self):
        """One failed pass must not permanently disable compaction."""
        cc = _compressor(threshold_tokens=24_576)
        msgs = _messages(14)

        cc.should_compress(33_564)
        cc.compress(msgs, current_tokens=33_564)
        cc.should_compress(33_564)                 # still over -> 1 strike
        assert cc._ineffective_compression_count == 1

        cc.should_compress(1_000)                  # now under -> forgiven
        assert cc._ineffective_compression_count == 0


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
        assert cc._ineffective_compression_count == before + 1
