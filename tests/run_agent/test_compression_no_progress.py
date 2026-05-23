"""Tests for the per-turn compression no-progress guard.

Observed death-loop pattern:

    context compression done: messages=30->30 tokens=~66721->57363
    context compression done: messages=62->62 tokens=~65646->50434
    ...

Same message count in and out — the compressor had nothing to drop
because ``protect_first_n + protect_last_n >= total``, so each pass
just re-summarised inline, burning Haiku tokens and never crossing
back below the threshold. The next tool call would trigger the same
loop on the next iteration.

The guard: if compress returns the same message count AND saved less
than 15% of tokens, latch a turn-level flag and refuse further
compression attempts for the rest of that turn.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.conversation_compression import (
    compression_made_progress,
    should_skip_compression_for_turn,
)


# ── Predicate: did compress actually help? ───────────────────────────


class TestCompressionMadeProgress:
    def test_progress_when_message_count_drops(self):
        # 60 → 20 messages, even with no token savings: progress.
        assert compression_made_progress(
            pre_msg_count=60,
            post_msg_count=20,
            pre_tokens=70_000,
            post_tokens=70_000,
        ) is True

    def test_progress_when_tokens_drop_more_than_threshold(self):
        # Same message count, but 30% token reduction: real progress
        # (summary collapsed long tool results inline).
        assert compression_made_progress(
            pre_msg_count=30,
            post_msg_count=30,
            pre_tokens=66_721,
            post_tokens=46_700,  # ~30% drop
        ) is True

    def test_no_progress_when_same_count_and_small_token_change(self):
        # The forensic case: messages 30→30, tokens 66721→57363 (~14%)
        # Under 15% threshold with no message-count drop → no progress.
        assert compression_made_progress(
            pre_msg_count=30,
            post_msg_count=30,
            pre_tokens=66_721,
            post_tokens=57_363,
        ) is False

    def test_no_progress_when_tokens_grow(self):
        # Defensive: summary text added MORE than was removed.
        assert compression_made_progress(
            pre_msg_count=30,
            post_msg_count=30,
            pre_tokens=66_000,
            post_tokens=68_000,
        ) is False

    def test_progress_when_message_count_drops_even_one(self):
        # Strict "==" semantics: any drop counts as progress.
        assert compression_made_progress(
            pre_msg_count=30,
            post_msg_count=29,
            pre_tokens=66_000,
            post_tokens=66_000,
        ) is True

    def test_zero_token_baseline_is_treated_as_no_progress(self):
        # Defensive: 0 pre-tokens means we can't compute a ratio — must
        # not treat "any drop" as progress and loop forever.
        assert compression_made_progress(
            pre_msg_count=30,
            post_msg_count=30,
            pre_tokens=0,
            post_tokens=0,
        ) is False


# ── Turn-level flag: skip further compression once latched ──────────


class TestShouldSkipCompressionForTurn:
    def test_unset_flag_does_not_skip(self):
        agent = SimpleNamespace()
        assert should_skip_compression_for_turn(agent) is False

    def test_flag_false_does_not_skip(self):
        agent = SimpleNamespace(_compression_no_progress_turn=False)
        assert should_skip_compression_for_turn(agent) is False

    def test_flag_true_skips(self):
        agent = SimpleNamespace(_compression_no_progress_turn=True)
        assert should_skip_compression_for_turn(agent) is True


# ── End-to-end: preflight aborts after one no-progress pass ─────────


class TestPreflightAborts:
    """The preflight loop in conversation_loop.py runs up to 3 passes.
    After the no-progress guard latches, a second pass must NOT fire and
    `_compress_context` must NOT be called again."""

    def test_preflight_aborts_after_no_progress(self, monkeypatch):
        """Drive the preflight loop with an agent whose compressor always
        returns the same message list and reports nearly-identical tokens.
        The guard must abort after the first pass, set the flag, and skip
        the remaining 2 passes."""
        from agent.conversation_compression import _preflight_compress_with_guard

        messages = [{"role": "user", "content": "msg"}] * 30

        compress_calls = []

        def fake_compress(msgs, _system_message, *, approx_tokens, task_id):
            compress_calls.append(len(msgs))
            # Same length, only ~14% smaller — exactly the death-loop pattern.
            return list(msgs), "system"

        agent = SimpleNamespace(
            _compress_context=fake_compress,
            _compression_no_progress_turn=False,
        )

        # Fake token estimator: 66_721 before, 57_363 after every pass.
        token_states = iter([66_721, 57_363, 57_363, 57_363])

        def fake_estimate(_msgs, **_kwargs):
            try:
                return next(token_states)
            except StopIteration:
                return 57_363

        new_messages, _ = _preflight_compress_with_guard(
            agent,
            messages=messages,
            system_message="orig",
            active_system_prompt="sys",
            tools=None,
            preflight_tokens=66_721,
            max_passes=3,
            effective_task_id="t1",
            estimate_fn=fake_estimate,
        )

        # Exactly ONE compress attempt — the guard latched after the first.
        assert len(compress_calls) == 1
        # Turn flag is now set; any subsequent compress site must skip.
        assert agent._compression_no_progress_turn is True
        # Message list returned unchanged
        assert len(new_messages) == len(messages)

    def test_preflight_keeps_going_when_progress_is_real(self):
        """Sanity check: if compression really helps, the loop continues."""
        from agent.conversation_compression import _preflight_compress_with_guard

        # Start with 30 msgs; each pass halves it.
        state = {"msgs": [{"role": "user", "content": str(i)} for i in range(30)]}

        def fake_compress(msgs, _system_message, *, approx_tokens, task_id):
            state["msgs"] = msgs[: max(1, len(msgs) // 2)]
            return state["msgs"], "system"

        agent = SimpleNamespace(
            _compress_context=fake_compress,
            _compression_no_progress_turn=False,
        )

        # Tokens drop in step with message count: real progress
        token_states = iter([80_000, 40_000, 20_000, 10_000])

        def fake_estimate(_msgs, **_kwargs):
            return next(token_states, 5_000)

        new_messages, _ = _preflight_compress_with_guard(
            agent,
            messages=state["msgs"],
            system_message="orig",
            active_system_prompt="sys",
            tools=None,
            preflight_tokens=80_000,
            max_passes=3,
            effective_task_id="t1",
            estimate_fn=fake_estimate,
            threshold_tokens=15_000,
        )
        # Loop exited cleanly: flag NOT latched.
        assert agent._compression_no_progress_turn is False
        assert len(new_messages) < 30
