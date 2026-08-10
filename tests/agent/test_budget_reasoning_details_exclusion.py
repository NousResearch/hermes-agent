"""Regression tests for #73298 (second site): the tail-budget walk must not
charge the signed/base64 ``reasoning_details`` envelope at chars/4.

The preflight estimator (``model_metadata._estimate_message_tokens_without_images``)
excludes ``reasoning_details`` entirely; ``_estimate_msg_budget_tokens`` in the
compressor mirrors that for the envelope while — per the #51800 counter-argument —
still counting actual thinking TEXT when it is not already carried by
``reasoning``/``reasoning_content``.
"""

from agent.context_compressor import (
    _estimate_msg_budget_tokens,
    _reasoning_details_text_chars,
)


def _signed_thinking_block(thinking_chars: int, sig_chars: int) -> dict:
    return {
        "type": "thinking",
        "thinking": "t" * thinking_chars,
        "signature": "s" * sig_chars,
    }


class TestBudgetExcludesReasoningDetailsEnvelope:
    def test_signature_blob_not_charged(self):
        base = {"role": "assistant", "content": "hello there"}
        with_envelope = dict(
            base,
            reasoning_details=[_signed_thinking_block(0, 40_000)],
        )
        plain = _estimate_msg_budget_tokens(base)
        loaded = _estimate_msg_budget_tokens(with_envelope)
        # 40K chars of base64 signature would add ~10K phantom tokens at
        # chars/4. The envelope must be invisible to the budget.
        assert loaded - plain < 50, (
            f"signed envelope inflated the budget by {loaded - plain} tokens"
        )

    def test_thinking_text_still_counted(self):
        """#51800: real reasoning text must stay visible to the budget when
        it is not duplicated into reasoning/reasoning_content."""
        base = {"role": "assistant", "content": "hello there"}
        with_text = dict(
            base,
            reasoning_details=[_signed_thinking_block(8_000, 0)],
        )
        plain = _estimate_msg_budget_tokens(base)
        loaded = _estimate_msg_budget_tokens(with_text)
        # 8K chars of thinking ≈ 2K tokens at chars/4.
        assert loaded - plain > 1_000, (
            "thinking TEXT inside reasoning_details was dropped from the "
            f"budget (delta {loaded - plain}); only the envelope may be excluded"
        )

    def test_duplicated_thinking_text_not_double_charged(self):
        """Anthropic-wire sessions carry the prose byte-identical in both
        reasoning_content and reasoning_details; charge it once."""
        prose = "deep thought " * 1_000
        msg = {
            "role": "assistant",
            "content": "hi",
            "reasoning_content": prose,
            "reasoning_details": [
                {"type": "thinking", "thinking": prose, "signature": "s" * 5_000},
            ],
        }
        once = _estimate_msg_budget_tokens(
            {"role": "assistant", "content": "hi", "reasoning_content": prose}
        )
        both = _estimate_msg_budget_tokens(msg)
        assert both - once < 100, (
            f"thinking prose charged twice (delta {both - once} tokens)"
        )

    def test_codex_reasoning_items_still_charged(self):
        """Codex Responses genuinely replays these on every request (#55572);
        the exclusion is scoped to reasoning_details only."""
        base = {"role": "assistant", "content": "hi"}
        loaded = dict(
            base,
            codex_reasoning_items=[{"encrypted_content": "e" * 20_000}],
        )
        assert (
            _estimate_msg_budget_tokens(loaded) - _estimate_msg_budget_tokens(base)
            > 4_000
        )


class TestReasoningDetailsTextChars:
    def test_none_and_empty(self):
        assert _reasoning_details_text_chars(None) == 0
        assert _reasoning_details_text_chars([]) == 0
        assert _reasoning_details_text_chars("") == 0

    def test_string_counts_fully(self):
        assert _reasoning_details_text_chars("abcd" * 10) == 40

    def test_envelope_keys_skipped(self):
        blocks = [
            {"type": "thinking", "thinking": "abc", "signature": "S" * 999},
            {"type": "redacted_thinking", "data": "D" * 999},
            {"type": "text", "text": "xy"},
        ]
        assert _reasoning_details_text_chars(blocks) == 5


class TestPreflightExcludesReasoningDetails:
    """Regression coverage for the preflight half (#73306 shipped without a
    test): _estimate_message_tokens_without_images must not count the
    reasoning_details envelope."""

    def test_preflight_estimate_ignores_reasoning_details(self):
        from agent.model_metadata import _estimate_message_tokens_without_images

        base = {"role": "assistant", "content": "hello there"}
        loaded = dict(
            base,
            reasoning_details=[_signed_thinking_block(10_000, 40_000)],
        )
        plain = _estimate_message_tokens_without_images(base)
        with_details = _estimate_message_tokens_without_images(loaded)
        assert with_details - plain < 50, (
            "preflight estimator counted the reasoning_details payload "
            f"(delta {with_details - plain} tokens) — compression would fire "
            "at a fraction of real usage on thinking models (#73298)"
        )
