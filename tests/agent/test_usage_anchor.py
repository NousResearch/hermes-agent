"""Usage-anchored context accounting (agent/model_metadata.py).

Context-size checks anchor on the provider-reported ``usage.prompt_tokens``
of the last main-loop response and estimate ONLY the messages appended
since. These tests cover:

  * anchor + delta arithmetic (exact base, small estimated delta);
  * the image-heavy divergence the anchor eliminates (flat 1500/image
    heuristic vs provider truth);
  * fallback to full estimation when no anchor exists (first request,
    usage-less providers);
  * invalidation when compaction rewrites the transcript (structural
    id/index check fails closed) and on explicit reset sites;
  * the preflight consumer (_preflight_request_tokens) preferring the
    anchor, plus a sabotage check proving the anchored path (not the
    heuristic) produces the number.
"""

from types import SimpleNamespace

import pytest

from agent.model_metadata import (
    anchored_context_tokens,
    capture_usage_anchor,
    estimate_messages_tokens_rough,
)
from agent.turn_context import _preflight_request_tokens


def _msg(role, content):
    return {"role": role, "content": content}


def _image_msg():
    # ~40KB of fake base64 — the rough estimator charges a flat 1500
    # tokens per image part regardless of true provider accounting.
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": "look at this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + "A" * 40000},
            },
        ],
    }


def _history_with_images(n_images=10):
    msgs = [_msg("user", "start")]
    for i in range(n_images):
        msgs.append(_msg("assistant", f"taking screenshot {i}"))
        msgs.append(_image_msg())
    msgs.append(_msg("assistant", "done looking"))
    return msgs


class TestAnchorArithmetic:
    def test_anchor_plus_small_delta(self):
        messages = _history_with_images(10)
        anchor = capture_usage_anchor(50_000, 250, messages)
        assert anchor is not None
        assert anchor["prompt_tokens"] == 50_000
        assert anchor["base_count"] == len(messages)

        # Main loop appends the response's own assistant reply, then a tool
        # result / user follow-up.
        messages.append(_msg("assistant", "the anchored reply itself"))
        messages.append(_msg("user", "short follow-up"))

        anchored = anchored_context_tokens(messages, anchor)
        assert anchored is not None
        # Exact base + completion; the assistant reply at base_count is
        # covered by completion_tokens, so only the follow-up is estimated.
        delta_est = estimate_messages_tokens_rough([messages[-1]])
        assert anchored == 50_000 + 250 + delta_est
        assert delta_est < 50  # the estimated window is one small message

    def test_image_heavy_divergence_eliminated(self):
        messages = _history_with_images(10)
        # Provider ground truth: say the real prompt was 12,000 tokens
        # (providers often charge far less than 1500/image, or the images
        # were downscaled). The heuristic charges 10 * 1500 + text.
        anchor = capture_usage_anchor(12_000, 100, messages)
        messages.append(_msg("assistant", "reply"))
        messages.append(_msg("user", "ok"))

        rough = estimate_messages_tokens_rough(messages)
        anchored = anchored_context_tokens(messages, anchor)
        assert rough >= 15_000  # flat 1500 x 10 images dominates
        assert anchored is not None
        assert anchored < 12_200
        # The whole-history heuristic diverges by thousands of tokens;
        # the anchored figure is provider truth + a tiny delta.
        assert rough - anchored > 2_800

    def test_no_usage_returns_none(self):
        messages = [_msg("user", "hi")]
        assert capture_usage_anchor(0, 0, messages) is None
        assert capture_usage_anchor(None, None, messages) is None
        assert capture_usage_anchor("garbage", 1, messages) is None

    def test_missing_anchor_falls_back(self):
        messages = _history_with_images(2)
        assert anchored_context_tokens(messages, None) is None


class TestAnchorInvalidation:
    def test_compaction_rewrite_fails_closed(self):
        messages = _history_with_images(4)
        anchor = capture_usage_anchor(30_000, 50, messages)
        # Compaction: transcript rebuilt as a new, shorter list.
        compacted = [
            _msg("user", "summary handoff"),
            _msg("assistant", "[compressed summary]"),
        ]
        assert anchored_context_tokens(compacted, anchor) is None

    def test_middle_splice_shifts_base_and_fails_closed(self):
        messages = _history_with_images(4)
        anchor = capture_usage_anchor(30_000, 50, messages)
        # Micro-compact style splice: middle window replaced by one marker.
        spliced = messages[:1] + [_msg("assistant", "[marker]")] + messages[5:]
        assert anchored_context_tokens(spliced, anchor) is None

    def test_same_length_different_objects_fails_closed(self):
        messages = _history_with_images(4)
        anchor = capture_usage_anchor(30_000, 50, messages)
        rebuilt = [dict(m) for m in messages]  # fresh dicts, same values
        assert anchored_context_tokens(rebuilt, anchor) is None

    def test_explicit_invalidation_sites(self):
        """The compaction + session-reset sites null agent._usage_anchor."""
        import inspect

        import agent.conversation_compression as cc
        import agent.codex_runtime as cr
        import run_agent

        assert "agent._usage_anchor = None" in inspect.getsource(cc)
        assert "agent._usage_anchor = None" in inspect.getsource(cr)
        assert "self._usage_anchor = None" in inspect.getsource(
            run_agent.AIAgent.reset_session_state
        )


class TestPreflightConsumer:
    def _agent(self, anchor):
        return SimpleNamespace(
            _usage_anchor=anchor,
            tools=None,
            api_mode="",
            provider="openai",
        )

    def test_preflight_prefers_anchor(self):
        messages = _history_with_images(10)
        anchor = capture_usage_anchor(50_000, 250, messages)
        messages.append(_msg("assistant", "reply"))
        messages.append(_msg("user", "ok"))
        agent = self._agent(anchor)

        got = _preflight_request_tokens(agent, messages, "SYSTEM PROMPT " * 500)
        expected = anchored_context_tokens(messages, anchor)
        assert got == expected
        # The anchored figure ignores the (already-counted) system prompt
        # text passed in — provider usage includes the real one.
        assert 50_000 < got < 50_500

    def test_preflight_falls_back_without_anchor(self):
        messages = _history_with_images(3)
        agent = self._agent(None)
        got = _preflight_request_tokens(agent, messages, "sys")
        # Pure heuristic: flat image cost dominates.
        assert got >= 4_500

    def test_sabotage_disabling_anchor_changes_result(self):
        """Prove the anchored path produced the number: with the anchor
        removed (the sabotage), the same inputs yield the heuristic figure,
        which diverges by thousands of tokens on an image-heavy history."""
        messages = _history_with_images(10)
        anchor = capture_usage_anchor(12_000, 100, messages)
        messages.append(_msg("assistant", "reply"))
        messages.append(_msg("user", "ok"))

        anchored_result = _preflight_request_tokens(
            self._agent(anchor), messages, ""
        )
        sabotaged_result = _preflight_request_tokens(
            self._agent(None), messages, ""
        )
        assert sabotaged_result - anchored_result > 2_800


class TestCompressionTriggerUsesAnchor:
    def test_threshold_decision_flips_with_anchor(self):
        """An image-heavy history the heuristic pushes over a 15K threshold
        stays under it when the provider reports the real 12K prompt."""
        messages = _history_with_images(10)
        anchor = capture_usage_anchor(12_000, 100, messages)
        messages.append(_msg("assistant", "reply"))

        threshold = 15_000
        heuristic = estimate_messages_tokens_rough(messages)
        anchored = anchored_context_tokens(messages, anchor)
        assert heuristic >= threshold  # old behavior: spurious compression
        assert anchored is not None and anchored < threshold


class TestDeferBaselineAnchoredInteraction:
    def test_assemble_api_request_records_rough_scale_defer_baseline(self):
        """Regression test for #103391:
        When an anchor is present, assemble_api_request overrides request_pressure_tokens
        with the anchored real token count. However, note_request_rough_estimate must
        record the PRE-OVERRIDE rough estimate, so the defer baseline matches the scale
        of subsequent rough preflight estimates if the anchor is invalidated (e.g. edit-resend).
        """
        from agent.context_compressor import ContextCompressor
        from agent.turn_request_assembly import assemble_api_request

        messages = _history_with_images(10)
        # Real prompt usage is 12,000; rough whole-history estimate is ~15,000+
        anchor = capture_usage_anchor(12_000, 100, messages)
        messages.append(_msg("assistant", "reply"))
        messages.append(_msg("user", "edit follow up"))

        compressor = ContextCompressor(
            model="test/model",
            threshold_percent=0.8,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
        compressor.context_length = 20_000
        compressor.threshold_tokens = 16_000
        agent = SimpleNamespace(
            _usage_anchor=anchor,
            context_compressor=compressor,
            tools=[],
            prefill_messages=[],
            _use_prompt_caching=False,
            provider="openai",
            model="test-model",
            api_mode="chat",
            ephemeral_system_prompt="",
            _copy_reasoning_content_for_api=lambda msg, api_msg: None,
            _should_sanitize_tool_calls=lambda: False,
            _sanitize_api_messages=lambda msgs: msgs,
            _drop_thinking_only_and_merge_users=lambda msgs, **kw: msgs,
        )

        assembled = assemble_api_request(
            agent,
            messages=messages,
            current_turn_user_idx=len(messages) - 1,
            _ext_prefetch_cache=None,
            _plugin_user_context=None,
            moa_config=None,
            active_system_prompt="sys",
            original_user_message="edit follow up",
            pending_moa_prepared_request=None,
            request_logger=None,
        )

        # Assembled request pressure tokens use the anchored figure (~12k + delta)
        assert assembled.request_pressure_tokens < 13_000

        # But compressor._pending_request_rough_tokens must be on the rough scale (>= 15,000)
        assert compressor._pending_request_rough_tokens >= 15_000
        rough_baseline = compressor._pending_request_rough_tokens

        # Now simulate API response updating usage with real prompt tokens
        compressor.update_from_response({"prompt_tokens": 12_100})
        assert compressor.last_real_prompt_tokens == 12_100
        assert compressor.last_rough_tokens_when_real_prompt_fit == rough_baseline

        # Simulate user edit-resend: history rewinds/truncates, invalidating the anchor
        rewound_messages = messages[:5]
        # Anchor is invalidated because length < anchor base_count
        assert anchored_context_tokens(rewound_messages, anchor) is None

        # Rough estimate for rewound messages is ~7,500+
        rough_now = estimate_messages_tokens_rough(rewound_messages)

        # With threshold = 16,000 (80% of 20,000), a subsequent rough estimate of 18,000
        # (over threshold due to rough estimator overcharging images/CJK) should defer:
        # projected_real = 12_100 + max(0, 18_000 - 15_493) = 14_607 < 16_000 -> True
        # (Under the old bug where baseline was stored on the real scale ~12,000,
        # projected_real would be 12_100 + (18_000 - 12_000) = 18_100 >= 16_000 -> False,
        # causing spurious compaction!)
        assert compressor.should_defer_preflight_to_real_usage(18_000) is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


