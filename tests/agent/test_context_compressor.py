"""Tests for agent/context_compressor.py — compression logic, thresholds, truncation fallback."""

import json
import pytest
import time
from unittest.mock import patch, MagicMock

from agent.context_compressor import (
    ContextCompressor,
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    COMPRESSED_SUMMARY_METADATA_KEY,
    _summarize_tool_result,
    _is_summary_access_or_quota_error,
)
from hermes_state import SessionDB


class StubProviderError(Exception):
    def __init__(self, message, *, status_code=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


@pytest.fixture()
def compressor():
    """Create a ContextCompressor with mocked dependencies."""
    with patch("agent.context_compressor.get_model_context_length", return_value=100000):
        c = ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=2,
            protect_last_n=2,
            quiet_mode=True,
        )
        return c


class TestSummarizeToolResultWebExtract:
    """Pre-compression pruning must survive web_extract calls whose ``urls`` are
    web_search result dicts ({"url"/"href": ...}), which models routinely forward
    straight into web_extract.
    """

    CONTENT = "x" * 500  # >200 chars so the pruning pass actually summarizes

    def test_multiple_dict_urls_do_not_crash(self):
        # Two dict URLs previously hit ``dict + str`` -> TypeError, aborting
        # _prune_old_tool_results() (and thus compress()).
        args = json.dumps({
            "urls": [
                {"url": "https://example.com/a", "title": "A"},
                {"url": "https://example.org/b", "title": "B"},
            ]
        })
        summary = _summarize_tool_result("web_extract", args, self.CONTENT)
        assert summary == "[web_extract] https://example.com/a (+1 more) (500 chars)"

    def test_single_dict_url_is_unwrapped_not_stringified(self):
        args = json.dumps({"urls": [{"url": "https://example.com/a", "title": "A"}]})
        summary = _summarize_tool_result("web_extract", args, self.CONTENT)
        assert summary == "[web_extract] https://example.com/a (500 chars)"
        assert "{" not in summary  # no raw dict repr leaked into the summary

    def test_href_key_is_unwrapped(self):
        args = json.dumps({"urls": [{"href": "https://example.com/h"}]})
        summary = _summarize_tool_result("web_extract", args, self.CONTENT)
        assert summary == "[web_extract] https://example.com/h (500 chars)"

    def test_malformed_dict_falls_back_to_placeholder(self):
        args = json.dumps({"urls": [{"title": "no url here"}, {"title": "still none"}]})
        summary = _summarize_tool_result("web_extract", args, self.CONTENT)
        assert summary == "[web_extract] ? (+1 more) (500 chars)"

    def test_plain_string_urls_unchanged(self):
        # Regression guard: the normal (already-working) string path is intact.
        args = json.dumps({"urls": ["https://example.com/a", "https://example.org/b"]})
        summary = _summarize_tool_result("web_extract", args, self.CONTENT)
        assert summary == "[web_extract] https://example.com/a (+1 more) (500 chars)"


class TestShouldCompress:
    def test_below_threshold(self, compressor):
        compressor.last_prompt_tokens = 50000
        assert compressor.should_compress() is False

    def test_above_threshold(self, compressor):
        compressor.last_prompt_tokens = 90000
        assert compressor.should_compress() is True

    def test_exact_threshold(self, compressor):
        compressor.last_prompt_tokens = 85000
        assert compressor.should_compress() is True

    def test_explicit_tokens(self, compressor):
        assert compressor.should_compress(prompt_tokens=90000) is True
        assert compressor.should_compress(prompt_tokens=50000) is False



class TestUpdateFromResponse:
    def test_updates_fields(self, compressor):
        compressor.awaiting_real_usage_after_compression = True
        compressor.last_compression_rough_tokens = 90_000
        compressor.update_from_response({
            "prompt_tokens": 5000,
            "completion_tokens": 1000,
            "total_tokens": 6000,
        })
        assert compressor.last_prompt_tokens == 5000
        assert compressor.last_completion_tokens == 1000
        assert compressor.last_real_prompt_tokens == 5000
        assert compressor.last_rough_tokens_when_real_prompt_fit == 90_000
        assert compressor.awaiting_real_usage_after_compression is False

    def test_missing_fields_default_zero(self, compressor):
        compressor.update_from_response({})
        assert compressor.last_prompt_tokens == 0

class TestPreflightDeferral:
    def test_defers_when_recent_real_usage_fit_and_rough_growth_is_small(self, compressor):
        compressor.threshold_tokens = 85_000
        compressor.last_real_prompt_tokens = 50_000
        compressor.last_rough_tokens_when_real_prompt_fit = 90_000

        assert compressor.should_defer_preflight_to_real_usage(93_000) is True
        assert compressor.last_rough_tokens_when_real_prompt_fit == 93_000

    def test_does_not_defer_when_rough_growth_is_large(self, compressor):
        compressor.threshold_tokens = 85_000
        compressor.last_real_prompt_tokens = 50_000
        compressor.last_rough_tokens_when_real_prompt_fit = 90_000

        assert compressor.should_defer_preflight_to_real_usage(100_000) is False

    def test_does_not_defer_without_recent_real_usage(self, compressor):
        compressor.threshold_tokens = 85_000
        compressor.last_real_prompt_tokens = 0
        compressor.last_rough_tokens_when_real_prompt_fit = 90_000

        assert compressor.should_defer_preflight_to_real_usage(93_000) is False

    def test_defers_immediately_after_compaction_with_stale_real_prompt(self, compressor):
        """#36718: right after a compaction, last_real_prompt_tokens still holds
        the stale pre-compression value (above threshold). The awaiting flag
        must force deferral so preflight doesn't fire a SECOND compaction before
        real post-compaction usage arrives."""
        compressor.threshold_tokens = 85_000
        # Stale pre-compression value — would hit the `>= threshold => False`
        # short-circuit and defeat deferral without the flag guard.
        compressor.last_real_prompt_tokens = 120_000
        compressor.awaiting_real_usage_after_compression = True
        assert compressor.should_defer_preflight_to_real_usage(95_000) is True

    def test_resumes_normal_deferral_after_flag_cleared(self, compressor):
        """Once update_from_response() clears the flag, the normal baseline/
        growth deferral logic governs again (no permanent deferral)."""
        compressor.threshold_tokens = 85_000
        compressor.last_real_prompt_tokens = 120_000
        compressor.awaiting_real_usage_after_compression = False
        # Stale-high real prompt with the flag cleared => the >= threshold
        # short-circuit applies => no deferral.
        assert compressor.should_defer_preflight_to_real_usage(95_000) is False



class TestCompress:
    def _make_messages(self, n):
        return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(n)]

    def test_too_few_messages_returns_unchanged(self, compressor):
        msgs = self._make_messages(4)  # protect_first=2 + protect_last=2 + 1 = 5 needed
        result = compressor.compress(msgs)
        assert result == msgs

    def test_truncation_fallback_no_client(self, compressor):
        # Simulate "no summarizer available" explicitly. call_llm can otherwise
        # discover the developer's real auxiliary credentials from auth state.
        # The failed summary should use the deterministic fallback path.
        msgs = [{"role": "system", "content": "System prompt"}] + self._make_messages(10)
        with patch("agent.context_compressor.call_llm", side_effect=RuntimeError("no provider")):
            result = compressor.compress(msgs)
        assert len(result) < len(msgs)
        # Should keep system message and last N
        assert result[0]["role"] == "system"
        assert compressor.compression_count == 1
        # Abort flag must NOT fire under the default config.
        assert compressor._last_compress_aborted is False
        assert compressor._last_summary_fallback_used is True

    def test_summary_failure_uses_deterministic_fallback_with_recovered_context(self):
        """Regression: failed LLM summaries should not emit a content-free marker.

        The fallback should preserve locally recoverable continuity details so a
        future turn does not see only "messages were removed" after compaction.
        """
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test/model",
                protect_first_n=1,
                protect_last_n=2,
                quiet_mode=True,
            )

        msgs = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Please fix the compression summary failure"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"agent/context_compressor.py","offset":1}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "read agent/context_compressor.py and found static fallback marker",
            },
            {"role": "assistant", "content": "I found the issue."},
            {"role": "user", "content": "latest protected ask"},
            {"role": "assistant", "content": "ok"},
        ]

        with (
            patch.object(c, "_find_tail_cut_by_tokens", return_value=5),
            patch(
                "agent.context_compressor.call_llm",
                side_effect=RuntimeError("provider down"),
            ),
        ):
            result = c.compress(msgs)

        combined = "\n".join(str(m.get("content", "")) for m in result)
        assert HISTORICAL_TASK_HEADING in combined
        assert "Please fix the compression summary failure" in combined
        assert "read_file" in combined
        assert "agent/context_compressor.py" in combined
        assert "Summary generation was unavailable" in combined
        assert "removed to free context space but could not be summarized" not in combined
        assert c._last_summary_fallback_used is True
        assert c._last_summary_dropped_count == 3

    def test_fallback_summary_does_not_triplicate_latest_user_ask(self):
        """Regression for #49307: the deterministic fallback summary used to
        render the latest user ask verbatim under THREE headings (Task
        Snapshot, In-Progress, Pending Asks). The model then re-answered it
        and buried the genuinely-new post-compaction turn (answer repetition +
        new-instruction loss). The latest ask must appear ONCE, as historical
        context only — never re-presented as unfulfilled in-progress/pending
        work.
        """
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test/model", quiet_mode=True)

        unique_ask = "PLEASE_COMPUTE_THE_ARITHMETIC_CHAIN_XYZ"
        turns = [
            {"role": "user", "content": unique_ask},
            {"role": "assistant", "content": "working on it"},
        ]
        summary = c._build_static_fallback_summary(turns, reason="provider down")

        # The triplication bug rendered the SAME ``active_task`` line —
        # formatted as ``User asked: '<ask>'`` — verbatim under three
        # headings (Task Snapshot, In-Progress, Pending Asks), making the
        # model treat an already-handled ask as unresolved work and re-answer
        # it. That exact formatted line must now appear at most ONCE (only as
        # the historical Task Snapshot record). The raw ask text may still
        # appear elsewhere (e.g. the "Last Dropped Turns" verbatim transcript),
        # but never re-labeled as in-progress/pending work.
        active_task_line = f"User asked: {unique_ask!r}"
        count = summary.count(active_task_line)
        assert count <= 1, (
            f"active_task line should appear at most once (was triplicated in "
            f"#49307), found {count}x:\n{summary}"
        )

    def test_threshold_below_window_at_minimum_ctx(self):
        """Regression for #14690: at context_length == MINIMUM_CONTEXT_LENGTH
        the floored threshold used to equal the whole window, so
        auto-compression could never fire. It now triggers at 85% of the
        window — high enough not to waste the small budget, below 100% so it
        actually fires."""
        from agent.context_compressor import MINIMUM_CONTEXT_LENGTH
        t = ContextCompressor._compute_threshold_tokens(MINIMUM_CONTEXT_LENGTH, 0.50)
        assert t < MINIMUM_CONTEXT_LENGTH
        assert t == 54400  # 85% of 64000

    def test_threshold_below_window_for_small_ctx(self):
        # 32K model: the 64000 floor exceeds the window — trigger at 85%.
        t = ContextCompressor._compute_threshold_tokens(32000, 0.50)
        assert t == 27200  # 85% of 32000
        assert t < 32000

    def test_threshold_floored_for_large_ctx(self):
        from agent.context_compressor import MINIMUM_CONTEXT_LENGTH
        # 200K model at 50% = 100000 (above floor) — unchanged.
        assert ContextCompressor._compute_threshold_tokens(200000, 0.50) == 100000
        # 100K model at 50% = 50000 (below floor) — floored to MINIMUM.
        assert ContextCompressor._compute_threshold_tokens(100000, 0.50) == MINIMUM_CONTEXT_LENGTH

    def test_minimum_ctx_model_can_actually_compress(self):
        """End-to-end: a model at exactly the minimum context length must have
        should_compress() fire below its window (at the 85% trigger), not only
        at 100%."""
        with patch("agent.context_compressor.get_model_context_length", return_value=64000):
            c = ContextCompressor(model="small-64k", quiet_mode=True)
            c.context_length = 64000
            c.threshold_tokens = c._compute_threshold_tokens(64000, c.threshold_percent)
        assert c.threshold_tokens == 54400
        assert c.threshold_tokens < 64000
        # At 85%+ usage compaction fires; below it, it doesn't (no premature compact).
        assert c.should_compress(55000) is True
        assert c.should_compress(40000) is False

    def test_max_tokens_reservation_lowers_threshold(self):
        """#43547: the provider reserves max_tokens out of the window, so the
        threshold must be based on (context_length - max_tokens), not the full
        window. A 200K model reserving 65536 output tokens has a ~134K input
        budget; at 50% that's ~67K, NOT 100K."""
        # No reservation (provider default) → full-window behavior, unchanged.
        assert ContextCompressor._compute_threshold_tokens(200000, 0.50) == 100000
        assert ContextCompressor._compute_threshold_tokens(200000, 0.50, None) == 100000
        # 65536 reserved → effective input budget 134464; 50% = 67232.
        assert ContextCompressor._compute_threshold_tokens(200000, 0.50, 65536) == 67232

    def test_max_tokens_reservation_with_small_window_floors(self):
        """With a large reservation on a smaller window the effective budget
        can drop near/below the minimum floor — the degenerate-window guard
        then triggers at 85% of the EFFECTIVE budget, never the raw window."""
        # 128K window, 65536 reserved → effective 62464 (< MINIMUM 64000).
        # Floor (64000) >= effective window (62464) → 85% of effective.
        t = ContextCompressor._compute_threshold_tokens(128000, 0.50, 65536)
        assert t == int(62464 * 0.85)  # 53094
        assert t < 62464

    def test_max_tokens_exceeding_window_falls_back_to_full(self):
        """Pathological: max_tokens >= context_length would make the effective
        budget <= 0; fall back to the full window rather than produce a
        non-positive threshold."""
        t = ContextCompressor._compute_threshold_tokens(64000, 0.50, 70000)
        # effective_window <= 0 → fall back to full context (64000) → 85% guard.
        assert t == 54400  # 85% of 64000, same as no-reservation small-ctx case
        assert t > 0

    def test_max_tokens_coercion_treats_non_int_as_no_reservation(self):
        """A non-int / non-positive max_tokens must coerce safely so the
        threshold arithmetic never raises. Guards the path where a mocked
        parent agent forwards a MagicMock max_tokens into a child
        ContextCompressor (regression for the delegate-test TypeError:
        '<=' not supported between MagicMock and int)."""
        from unittest.mock import MagicMock
        assert ContextCompressor._coerce_max_tokens(None) is None
        assert ContextCompressor._coerce_max_tokens(0) is None
        assert ContextCompressor._coerce_max_tokens(-5) is None
        assert ContextCompressor._coerce_max_tokens("nope") is None
        assert ContextCompressor._coerce_max_tokens(65536) == 65536
        # The actual regression: building a compressor with a MagicMock
        # max_tokens must NOT raise (the unmocked code did `ctx - MagicMock`
        # then `MagicMock <= 0`). int(MagicMock()) returns 1, so coercion
        # yields a harmless positive int rather than crashing — the threshold
        # is computed cleanly with a 1-token reservation.
        with patch("agent.context_compressor.get_model_context_length", return_value=200000):
            c = ContextCompressor(model="m", quiet_mode=True, max_tokens=MagicMock())
        assert isinstance(c.max_tokens, int)
        assert isinstance(c.threshold_tokens, int)
        assert c.threshold_tokens > 0  # no crash, sane value

    def test_compression_increments_count(self, compressor):
        msgs = self._make_messages(10)
        # Default config (abort_on_summary_failure=False) — fallback path
        # increments the count even on summary failure.
        compressor.compress(msgs)
        assert compressor.compression_count == 1
        compressor.compress(msgs)
        assert compressor.compression_count == 2

    def test_protects_first_and_last(self, compressor):
        msgs = self._make_messages(10)
        result = compressor.compress(msgs)
        # First 2 messages should be preserved (protect_first_n=2)
        # Last 2 messages should be preserved (protect_last_n=2)
        assert result[-1]["content"] == msgs[-1]["content"]
        # The second-to-last tail message may have the summary merged
        # into it when a double-collision prevents a standalone summary
        # (head=assistant, tail=user in this fixture).  Verify the
        # original content is present in either case.
        assert msgs[-2]["content"] in result[-2]["content"]

    def test_compress_strips_db_persisted_from_assembled_messages(self, compressor):
        """Regression for #57491: shallow copies must not carry flush markers."""
        msgs = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}", "_db_persisted": True}
            for i in range(10)
        ]
        with patch("agent.context_compressor.call_llm", side_effect=RuntimeError("no provider")):
            result = compressor.compress(msgs)
        assert len(result) < len(msgs)
        assert all("_db_persisted" not in msg for msg in result)

    def test_compress_terminal_sweep_strips_markers_even_if_a_copy_site_leaks(self, compressor):
        """Regression for #57491, structural: even if a copy site fails to strip
        the marker (simulating a future refactor that adds/reintroduces a leaky
        copy), the single terminal sweep in compress() guarantees no compacted
        message leaves carrying `_db_persisted`. Neuter the per-site helper to a
        plain leaking copy and assert the invariant still holds."""
        import agent.context_compressor as _cc

        msgs = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}", "_db_persisted": True}
            for i in range(10)
        ]
        # Make the per-site helper leak the marker (dict.copy keeps it).
        with patch.object(_cc, "_fresh_compaction_message_copy", lambda m: m.copy()), \
             patch("agent.context_compressor.call_llm", side_effect=RuntimeError("no provider")):
            result = compressor.compress(msgs)
        assert len(result) < len(msgs)
        assert all("_db_persisted" not in msg for msg in result), (
            "terminal sweep must strip _db_persisted even when a copy site leaks"
        )

    def test_protect_first_n_decays_after_first_compression(self):
        """Regression for #11996: protect_first_n must protect early turns on
        the FIRST compaction but DECAY afterwards, so the same early user
        messages don't get re-copied verbatim into every child session and
        fossilize (grow immortal) across a long, repeatedly-compressed
        session. The system prompt is always protected separately."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=3)

        msgs = [{"role": "system", "content": "sys"}] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(10)
        ]

        # First compaction: protect system + first 3 non-system.
        assert c.compression_count == 0
        assert c._effective_protect_first_n() == 3
        assert c._protect_head_size(msgs) == 1 + 3

        # Simulate having compressed once — early turns now live in the summary.
        c.compression_count = 1
        assert c._effective_protect_first_n() == 0
        assert c._protect_head_size(msgs) == 1  # system prompt only

    def test_protect_first_n_decays_when_previous_summary_exists(self):
        """Even if compression_count was reset, an existing handoff summary
        means the early turns are already captured — decay still applies."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=3)
        c.compression_count = 0
        c._previous_summary = "[CONTEXT SUMMARY]: earlier work"
        assert c._effective_protect_first_n() == 0


class TestTailBudgetCodexReplayFields:
    def test_tail_cut_counts_codex_replay_and_reasoning_fields(self):
        """Tail protection must budget hidden replay fields sent back to providers.

        Codex Responses messages can have tiny visible content but large
        `codex_reasoning_items`, `codex_message_items`, or provider-native
        reasoning fields. Preflight compression counts these fields, so the
        tail-cut budget must count them too; otherwise compression preserves an
        oversized tail and immediately starts the next session near the limit.
        """
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test/model",
                protect_first_n=1,
                protect_last_n=1,
                quiet_mode=True,
            )

        big_replay = "x" * 5_000
        big_hidden_message = {
            "role": "assistant",
            "content": "ok",
            "reasoning": "summary " + big_replay,
            "reasoning_content": "scratchpad " + big_replay,
            "reasoning_details": [{"text": "details " + big_replay}],
            "codex_reasoning_items": [
                {"type": "reasoning", "encrypted_content": "enc_" + big_replay}
            ],
            "codex_message_items": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "reply " + big_replay}],
                }
            ],
        }
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "initial ask"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "older follow-up"},
            big_hidden_message,
        ]
        messages.extend(
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"tail visible message {i}",
            }
            for i in range(14)
        )

        cut_idx = c._find_tail_cut_by_tokens(messages, head_end=1, token_budget=150)

        assert cut_idx == 5
        assert messages[4]["codex_reasoning_items"][0]["encrypted_content"].startswith("enc_")
        assert messages[4]["codex_message_items"][0]["content"][0]["text"].startswith("reply ")

    @pytest.mark.parametrize(
        ("field_name", "field_value"),
        [
            ("reasoning", "x" * 5_000),
            ("reasoning_content", "x" * 5_000),
            ("reasoning_details", [{"text": "x" * 5_000}]),
            (
                "codex_reasoning_items",
                [{"type": "reasoning", "encrypted_content": "x" * 5_000}],
            ),
            (
                "codex_message_items",
                [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "x" * 5_000}],
                    }
                ],
            ),
        ],
    )
    def test_tail_cut_counts_each_hidden_replay_field(self, field_name, field_value):
        """Each provider replay/reasoning field should affect tail budgeting."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test/model",
                protect_first_n=1,
                protect_last_n=1,
                quiet_mode=True,
            )

        hidden_message = {"role": "assistant", "content": "ok", field_name: field_value}
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "initial ask"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "older follow-up"},
            hidden_message,
        ]
        messages.extend(
            {
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"tail visible message {i}",
            }
            for i in range(14)
        )

        assert c._find_tail_cut_by_tokens(messages, head_end=1, token_budget=150) == 5


class TestGenerateSummaryNoneContent:
    """Regression: content=None (from tool-call-only assistant messages) must not crash."""

    def test_none_content_does_not_crash(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: tool calls happened"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "search"}}
            ]},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": None},
            {"role": "user", "content": "thanks"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            summary = c._generate_summary(messages)
        assert isinstance(summary, str)
        assert summary.startswith(SUMMARY_PREFIX)

    def test_none_content_in_system_message_compress(self):
        """System message with content=None should not crash during compress."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        msgs = [{"role": "system", "content": None}] + [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(10)
        ]
        result = c.compress(msgs)
        assert len(result) < len(msgs)


class TestNonStringContent:
    """Regression: content as dict (e.g., llama.cpp tool calls) must not crash."""

    def test_dict_content_coerced_to_string(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = {"text": "some summary"}

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            summary = c._generate_summary(messages)
        assert isinstance(summary, str)
        assert summary.startswith(SUMMARY_PREFIX)

    def test_none_content_treated_as_failure_not_empty_summary(self):
        """Regression #11978/#11914: a well-formed response with ``content=None``
        (some OpenAI-compatible proxies, e.g. cmkey.cn, return HTTP 200 with
        null/empty content) must NOT be stored as a prefix-only summary that
        silently wipes the compacted turns. It is treated as a summary failure
        and routed through cooldown so the turns are dropped without a summary
        rather than replaced by an empty one."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            # summary_model == model here, so no fallback path: straight to cooldown.
            c = ContextCompressor(model="test", quiet_mode=True)

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            summary = c._generate_summary(messages)
        # Empty content → failure → None (drop turns), NOT a prefix-only summary.
        assert summary is None
        assert summary != SUMMARY_PREFIX
        # Transient cooldown engaged so we don't immediately retry the bad proxy.
        assert c._summary_failure_cooldown_until > 0

    def test_empty_string_content_treated_as_failure(self):
        """An empty-string (or whitespace-only) ``content`` is handled the same
        as ``None`` — failure, not an empty summary (#11978)."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "   \n  "

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            summary = c._generate_summary(messages)
        assert summary is None
        assert c._summary_failure_cooldown_until > 0

    def test_empty_content_falls_back_to_main_model(self):
        """When the auxiliary summary model returns empty content and a distinct
        main model is configured, compression falls back to the main model
        before entering cooldown (#11978 glm-5.1 → glm-5 path)."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="glm-5",
                summary_model_override="glm-5.1",
                quiet_mode=True,
            )

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response) as mock_call:
            summary = c._generate_summary(messages)
        # Two calls: aux model (glm-5.1) then fallback to main (glm-5).
        assert mock_call.call_count == 2
        assert c._summary_model_fallen_back is True
        assert summary is None
        assert c._summary_failure_cooldown_until > 0

    def test_string_message_coerced_to_summary_content(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = "plain summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            summary = c._generate_summary(messages)

        assert summary.startswith(f"{SUMMARY_PREFIX}\n{HISTORICAL_TASK_HEADING}\n")
        assert "do something" in summary
        assert summary.endswith("plain summary text")

    def test_summary_call_does_not_force_temperature(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response) as mock_call:
            c._generate_summary(messages)

        kwargs = mock_call.call_args.kwargs
        assert "temperature" not in kwargs

    def test_summary_prompt_avoids_filter_sensitive_handoff_framing(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response) as mock_call:
            c._generate_summary(messages)

        prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        assert "Your output will be injected" not in prompt
        assert "Do NOT respond" not in prompt
        assert "DIFFERENT assistant" not in prompt
        assert "different assistant" not in prompt
        assert "refactor the auth module" not in prompt
        assert "JWT instead of sessions" not in prompt
        assert "Treat the conversation turns below as source material" in prompt
        assert "structured checkpoint summary" in prompt

    def test_summary_task_snapshot_is_grounded_to_latest_user_turn(self):
        """Regression for a copied prompt example becoming the active task.

        A real #26 run showed the summarizer emitting the old template example
        "Now refactor the auth module to use JWT instead of sessions". The
        compacted turns did not contain that request, so the summary must
        replace it with the deterministic latest user turn before the handoff
        becomes live context.
        """
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """## Historical Task Snapshot
User asked: 'Now refactor the auth module to use JWT instead of sessions'

## Goal
Continue the task.

## Completed Actions
None.
"""

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        latest = "RUN_LONG_REAL_26_SCORE_V3B_WITH_FACTUALITY_SMOKE"
        messages = [
            {"role": "user", "content": latest},
            {"role": "assistant", "content": "I will inspect the loop harness."},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            summary = c._generate_summary(messages)

        assert "refactor the auth module" not in summary
        assert "JWT instead of sessions" not in summary
        assert latest in summary
        assert "deterministic, from compacted turns" in summary

    def test_task_snapshot_skips_synthetic_user_scaffolding(self):
        """Grounding must anchor on the human ask, not runtime scaffolding.

        Todo snapshots, truncation notices, and background-process reports
        are injected with role="user"; if the newest user turn is one of
        those, the deterministic snapshot must look past it to the real ask
        (and return None when no real ask exists at all).
        """
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        messages = [
            {"role": "user", "content": "fix the login bug on prod"},
            {"role": "assistant", "content": "on it"},
            {
                "role": "user",
                "content": "[Your active task list was preserved across context compression]\n- item",
                "_todo_snapshot_synthetic": True,
            },
        ]
        snapshot = c._latest_user_task_snapshot(messages)
        assert snapshot is not None
        assert "fix the login bug on prod" in snapshot
        assert "task list was preserved" not in snapshot

        only_synthetic = [
            {"role": "user", "content": "[System: Your previous response was truncated ...]"},
        ]
        assert c._latest_user_task_snapshot(only_synthetic) is None

    def test_grounding_preserves_following_sections_across_regrounding(self):
        """The snapshot rewrite must keep later headings intact — twice.

        A replacement that consumes the section's trailing newlines glues the
        next "## " heading mid-line; on the following iterative compaction the
        heading is no longer at line start, the regex matches through \\Z, and
        every later section is silently deleted.
        """
        import re as _re

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        summary = (
            f"{HISTORICAL_TASK_HEADING}\n"
            "User asked: stale example\n\n"
            "## Historical Remaining Work\n- keep me\n\n"
            "## Goal\nfinish"
        )
        turns = [{"role": "user", "content": "real ask"}]

        first = c._ground_historical_task_snapshot(summary, turns)
        assert _re.search(r"(?m)^## Historical Remaining Work$", first)
        assert "- keep me" in first and "## Goal" in first

        second = c._ground_historical_task_snapshot(first, turns)
        assert "- keep me" in second and "## Goal" in second
        assert second.count(HISTORICAL_TASK_HEADING) == 1

    def test_summary_call_passes_live_main_runtime(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="gpt-5.4",
                provider="openai-codex",
                base_url="https://chatgpt.com/backend-api/codex",
                api_key="codex-token",
                api_mode="codex_responses",
                quiet_mode=True,
            )

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response) as mock_call:
            c._generate_summary(messages)

        assert mock_call.call_args.kwargs["main_runtime"] == {
            "model": "gpt-5.4",
            "provider": "openai-codex",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_key": "codex-token",
            "api_mode": "codex_responses",
        }


class TestSummaryFailureCooldown:
    def test_summary_failure_enters_cooldown_and_skips_retry(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)

        messages = [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

        with patch("agent.context_compressor.call_llm", side_effect=Exception("boom")) as mock_call:
            first = c._generate_summary(messages)
            second = c._generate_summary(messages)

        assert first is None
        assert second is None
        assert mock_call.call_count == 1


class TestAuthFailureAborts:
    """A 401/403 on the summary call must ABORT compression (preserve the
    session unchanged) instead of rotating into a degraded child session
    with a placeholder summary — regardless of abort_on_summary_failure.

    Real incident: a nous token pointed at a stale staging inference URL
    401'd on every compression attempt, and because abort_on_summary_failure
    defaults False the session rotated anyway (messages N->N), stranding the
    user on a fresh-but-broken session that kept failing the same way.
    """

    def _msgs(self, n=10):
        return [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(n)
        ]

    def _auth_err(self, status=401):
        return StubProviderError(
            f"Error code: {status} - "
            "{'status': 401, 'message': 'Your API key is invalid, blocked or out of funds.'}",
            status_code=status,
        )

    @pytest.mark.parametrize(
        "message",
        [
            "insufficient_quota",
            "quota exceeded",
            "quota_exceeded",
            "out of funds",
            "out of credits",
            "out of credit",
            "out of extra usage",
        ],
    )
    def test_quota_classifier_accepts_explicit_provider_signals(self, message):
        assert _is_summary_access_or_quota_error(Exception(message)) is True

    def test_missing_provider_api_key_is_terminal_access_failure(self):
        err = RuntimeError(
            "Provider 'opencode-zen' is set in config.yaml but no API key was "
            "found. Set the OPENCODE-ZEN_API_KEY environment variable."
        )
        assert _is_summary_access_or_quota_error(err) is True

    @pytest.mark.parametrize(
        "message",
        [
            "billing portal is temporarily unavailable",
            "usage limit documentation could not be loaded",
            "API key documentation was not found",
            "rate limit exceeded; retry later",
            "quota exceeded, please retry after the window resets",
            "request timed out",
        ],
    )
    def test_quota_classifier_rejects_transient_or_ambiguous_messages(self, message):
        assert _is_summary_access_or_quota_error(Exception(message)) is False

    @pytest.mark.parametrize("status", [401, 402, 403])
    def test_access_classifier_accepts_non_retryable_http_statuses(self, status):
        err = StubProviderError(
            "provider rejected summary request",
            status_code=status,
        )
        assert _is_summary_access_or_quota_error(err) is True

    def test_classifier_reads_response_status_code(self):
        err = StubProviderError(
            "provider rejected summary request",
            response=MagicMock(status_code=402),
        )
        assert _is_summary_access_or_quota_error(err) is True

    def test_400_out_of_extra_usage_aborts_instead_of_dropping_context(self):
        """Quota exhaustion preserves the original messages for a later retry."""
        err = StubProviderError(
            "Error code: 400 - {'error': {'message': 'out of extra usage'}}",
            status_code=400,
        )
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=2,
                protect_last_n=2,
                abort_on_summary_failure=False,
            )
        msgs = self._msgs(12)
        with patch("agent.context_compressor.call_llm", side_effect=err):
            result = c.compress(msgs, current_tokens=999999, force=True)

        assert result == msgs
        assert c._last_summary_auth_failure is True
        assert c._last_compress_aborted is True
        assert c._last_summary_fallback_used is False

    def test_missing_provider_api_key_preserves_original_messages(self):
        """A configured auxiliary provider without a visible key preserves context."""
        err = RuntimeError(
            "Provider 'opencode-zen' is set in config.yaml but no API key was "
            "found. Set the OPENCODE-ZEN_API_KEY environment variable, or switch "
            "to a different provider with hermes model."
        )
        with patch(
            "agent.context_compressor.get_model_context_length", return_value=100000
        ):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=2,
                protect_last_n=2,
                abort_on_summary_failure=False,
            )
        msgs = self._msgs(12)
        with patch("agent.context_compressor.call_llm", side_effect=err):
            result = c.compress(msgs, current_tokens=999999, force=True)

        assert result == msgs
        assert c._last_summary_error == str(err)
        assert c._last_summary_auth_failure is True
        assert c._last_compress_aborted is True
        assert c._last_summary_fallback_used is False
        assert c._last_summary_dropped_count == 0

    def test_402_quota_with_retry_uses_existing_fallback(self):
        """A reset-window quota remains transient instead of aborting compression."""
        err = StubProviderError(
            "quota exceeded, please retry after the window resets",
            status_code=402,
        )
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=2,
                protect_last_n=2,
                abort_on_summary_failure=False,
            )
        msgs = self._msgs(12)
        with patch("agent.context_compressor.call_llm", side_effect=err):
            result = c.compress(msgs, current_tokens=999999, force=True)

        assert result != msgs
        assert c._last_summary_auth_failure is False
        assert c._last_compress_aborted is False
        assert c._last_summary_fallback_used is True

    def test_generate_summary_flags_auth_failure(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)
        with patch("agent.context_compressor.call_llm", side_effect=self._auth_err(401)):
            result = c._generate_summary(self._msgs())
        assert result is None
        assert c._last_summary_auth_failure is True

    def test_403_also_flags_auth_failure(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)
        with patch("agent.context_compressor.call_llm", side_effect=self._auth_err(403)):
            c._generate_summary(self._msgs())
        assert c._last_summary_auth_failure is True

    def test_compress_aborts_on_auth_failure_despite_flag_false(self):
        """abort_on_summary_failure=False (the default), but a 401 must still
        abort: messages returned unchanged, _last_compress_aborted=True."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=2,
                protect_last_n=2,
                abort_on_summary_failure=False,
            )
        msgs = self._msgs(12)
        with patch("agent.context_compressor.call_llm", side_effect=self._auth_err(401)):
            result = c.compress(msgs, current_tokens=999999, force=True)
        # Session must NOT be compressed/rotated — same messages back.
        assert result == msgs
        assert len(result) == len(msgs)
        assert c._last_compress_aborted is True
        assert c._last_summary_auth_failure is True
        # Did NOT fall through to the static-fallback (drop-the-middle) path.
        assert c._last_summary_fallback_used is False

    def test_non_auth_failure_still_uses_fallback_path(self):
        """A generic (non-auth) failure with abort_on_summary_failure=False
        keeps the historical behavior: insert a static fallback + drop the
        middle window (does NOT abort)."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=2,
                protect_last_n=2,
                abort_on_summary_failure=False,
            )
        msgs = self._msgs(12)
        with patch("agent.context_compressor.call_llm", side_effect=Exception("boom 500")):
            result = c.compress(msgs, current_tokens=999999, force=True)
        assert c._last_summary_auth_failure is False
        assert c._last_compress_aborted is False
        assert len(result) < len(msgs)  # middle window dropped

    def test_generate_summary_flags_network_failure(self):
        """A connection/network error on the summary call flags
        _last_summary_network_failure (#29559)."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)
        with patch(
            "agent.context_compressor.call_llm",
            side_effect=ConnectionError("Connection error."),
        ):
            result = c._generate_summary(self._msgs())
        assert result is None
        assert c._last_summary_network_failure is True
        assert c._last_summary_auth_failure is False

    def test_compress_aborts_on_network_failure_despite_flag_false(self):
        """#29559/#25585: abort_on_summary_failure=False (default), but a
        transient connection error must ABORT — messages returned unchanged,
        _last_compress_aborted=True — NOT drop the middle window. Retrying once
        the network recovers beats discarding context for a transient blip."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=2,
                protect_last_n=2,
                abort_on_summary_failure=False,
            )
        msgs = self._msgs(12)
        with patch(
            "agent.context_compressor.call_llm",
            side_effect=ConnectionError("Connection error."),
        ):
            result = c.compress(msgs, current_tokens=999999, force=True)
        # Session must NOT be compressed/rotated — same messages back.
        assert result == msgs
        assert len(result) == len(msgs)
        assert c._last_compress_aborted is True
        assert c._last_summary_network_failure is True
        # Did NOT fall through to the static-fallback (drop-the-middle) path.
        assert c._last_summary_fallback_used is False

    def test_aux_model_auth_failure_recovers_on_main_no_abort(self):
        """A 401 from a DISTINCT auxiliary summary_model retries on the main
        model; if main succeeds, the auth flag is cleared and compression is
        NOT aborted (the aux creds were the only broken thing)."""
        mock_ok = MagicMock()
        mock_ok.choices = [MagicMock()]
        mock_ok.choices[0].message.content = "summary via main model"
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                summary_model_override="broken-aux-model",
                quiet_mode=True,
            )
        with patch(
            "agent.context_compressor.call_llm",
            side_effect=[self._auth_err(401), mock_ok],
        ) as mock_call:
            result = c._generate_summary(self._msgs())
        assert mock_call.call_count == 2
        assert isinstance(result, str)
        assert c._last_summary_auth_failure is False  # cleared on success


class TestSummaryFallbackToMainModel:
    """When ``summary_model`` differs from the main model and the summary LLM
    call fails, the compressor should retry once on the main model before
    giving up — losing N turns of context is almost always worse than one
    extra summary attempt.  Covers both the fast-path (explicit
    model-not-found errors) and the unknown-error best-effort retry."""

    def _msgs(self):
        return [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

    def test_model_not_found_404_falls_back_to_main_and_succeeds(self):
        """Classic misconfiguration: ``auxiliary.compression.model`` points at
        a model the main provider doesn't serve → 404 → retry on main."""
        mock_ok = MagicMock()
        mock_ok.choices = [MagicMock()]
        mock_ok.choices[0].message.content = "summary via main model"

        err_404 = Exception("404 model_not_found: no such model")
        err_404.status_code = 404

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                summary_model_override="broken-aux-model",
                quiet_mode=True,
            )

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=[err_404, mock_ok],
        ) as mock_call:
            result = c._generate_summary(self._msgs())

        assert mock_call.call_count == 2
        # First call used the misconfigured aux model
        assert mock_call.call_args_list[0].kwargs.get("model") == "broken-aux-model"
        # Second call used the main model (no model kwarg → call_llm uses main)
        assert "model" not in mock_call.call_args_list[1].kwargs
        assert result is not None
        assert "summary via main model" in result
        # Aux-model failure is recorded even though retry succeeded — this is
        # how callers (gateway /compress, CLI warning) know to tell the user
        # their auxiliary.compression.model setting is broken.
        assert c._last_aux_model_failure_model == "broken-aux-model"
        assert c._last_aux_model_failure_error is not None
        assert "404" in c._last_aux_model_failure_error

    def test_unknown_error_falls_back_to_main_and_succeeds(self):
        """Errors that don't match the 404/503/model_not_found fast-path
        (400s, provider-specific 'no route', aggregator rejections) should
        ALSO trigger a best-effort retry on main before entering cooldown."""
        mock_ok = MagicMock()
        mock_ok.choices = [MagicMock()]
        mock_ok.choices[0].message.content = "summary via main model"

        # A 400 from OpenRouter / Nous portal with an opaque message — does
        # NOT match _is_model_not_found, but still an unrecoverable misconfig.
        err_400 = Exception("400 Bad Request: provider rejected model")
        err_400.status_code = 400

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                summary_model_override="broken-aux-model",
                quiet_mode=True,
            )

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=[err_400, mock_ok],
        ) as mock_call:
            result = c._generate_summary(self._msgs())

        assert mock_call.call_count == 2
        assert mock_call.call_args_list[0].kwargs.get("model") == "broken-aux-model"
        assert "model" not in mock_call.call_args_list[1].kwargs
        assert result is not None
        assert "summary via main model" in result
        # Aux-model failure recorded despite successful recovery
        assert c._last_aux_model_failure_model == "broken-aux-model"
        assert c._last_aux_model_failure_error is not None
        assert "400" in c._last_aux_model_failure_error

    def test_no_fallback_when_summary_model_equals_main_model(self):
        """If the aux model IS the main model, there's nowhere to fall back
        to — go straight to cooldown, don't loop retrying the same call."""
        err = Exception("500 internal error")

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                summary_model_override="main-model",  # same as main
                quiet_mode=True,
            )

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=err,
        ) as mock_call:
            result = c._generate_summary(self._msgs())

        # Only one attempt — retry gate blocks fallback when models match
        assert mock_call.call_count == 1
        assert result is None
        # Not flagged as fallen back — the retry condition was never met
        assert getattr(c, "_summary_model_fallen_back", False) is False

    def test_fallback_only_happens_once_per_compressor(self):
        """If the retry-on-main ALSO fails, don't loop forever — enter
        cooldown like the normal failure path."""
        err1 = Exception("400 aux model rejected")
        err2 = Exception("500 main model also exploded")

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                summary_model_override="broken-aux-model",
                quiet_mode=True,
            )

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=[err1, err2],
        ) as mock_call:
            result = c._generate_summary(self._msgs())

        # Exactly 2 calls: initial + one retry on main.  No further retries.
        assert mock_call.call_count == 2
        assert result is None
        assert c._summary_model_fallen_back is True

    def test_json_decode_error_falls_back_to_main_and_succeeds(self):
        """JSONDecodeError from the OpenAI SDK's ``response.json()`` (raised
        when a misconfigured proxy returns HTML/plain-text with
        ``Content-Type: application/json``) should trigger the same
        retry-on-main path as 404/timeout.  Issue #22244."""
        import json as _json

        mock_ok = MagicMock()
        mock_ok.choices = [MagicMock()]
        mock_ok.choices[0].message.content = "summary via main model"

        # Simulate the SDK raising a raw JSONDecodeError with a realistic
        # error message ("Expecting value: line X column Y char Z").
        err_json = _json.JSONDecodeError(
            "Expecting value", "<!DOCTYPE html><html>...</html>", 0
        )

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                summary_model_override="aux-via-broken-proxy",
                quiet_mode=True,
            )

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=[err_json, mock_ok],
        ) as mock_call:
            result = c._generate_summary(self._msgs())

        assert mock_call.call_count == 2
        assert mock_call.call_args_list[0].kwargs.get("model") == "aux-via-broken-proxy"
        assert "model" not in mock_call.call_args_list[1].kwargs
        assert result is not None
        assert "summary via main model" in result
        # Aux-model failure recorded so /usage / gateway warnings can surface it
        assert c._last_aux_model_failure_model == "aux-via-broken-proxy"
        assert c._last_aux_model_failure_error is not None
        # The 220-char cap is shared with other fallback branches
        assert len(c._last_aux_model_failure_error) <= 220

    def test_json_decode_error_substring_match_in_wrapped_exception(self):
        """When the OpenAI SDK wraps the raw JSONDecodeError inside its own
        ``APIResponseValidationError`` (or similar), ``isinstance`` no longer
        matches but the substring "expecting value" still appears in
        ``str(e)``.  We detect this case by string match and fall back the
        same way."""
        mock_ok = MagicMock()
        mock_ok.choices = [MagicMock()]
        mock_ok.choices[0].message.content = "summary via main model"

        # A plain Exception with the canonical JSON decode error text — what
        # the SDK's APIResponseValidationError looks like at str() time.
        err_wrapped = Exception("Expecting value: line 1 column 1 (char 0)")

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                summary_model_override="aux-model",
                quiet_mode=True,
            )

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=[err_wrapped, mock_ok],
        ) as mock_call:
            result = c._generate_summary(self._msgs())

        assert mock_call.call_count == 2
        assert result is not None
        assert "summary via main model" in result

    def test_json_decode_error_on_main_uses_short_cooldown(self):
        """When already on the main model (no separate summary_model, or
        fallback already happened), a JSONDecodeError should set the short
        30s cooldown, not the default 60s — provider bodies tend to
        recover quickly when an upstream proxy comes back online."""
        import json as _json

        err_json = _json.JSONDecodeError("Expecting value", "<html/>", 0)

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                # No summary_model_override → already on main, no fallback path.
                quiet_mode=True,
            )

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=err_json,
        ), patch("agent.context_compressor.time.monotonic", return_value=1000.0):
            result = c._generate_summary(self._msgs())

        assert result is None
        # Short JSON-decode cooldown is 30s, not the default 60s.
        assert c._summary_failure_cooldown_until == 1030.0


class TestStreamingClosedFallback:
    """httpcore / httpx streaming premature-close errors must be classified the
    same as timeouts so the compressor retries on the main model instead of
    entering a 60-second cooldown.  Issue #18458.

    ``_is_connection_error`` is patched here because the test venv may not
    have ``openai`` installed (the real function does ``from openai import ...``
    inside its body).  We test the *wiring* — that `_generate_summary` calls
    ``_is_connection_error`` and acts on its result — not the classifier itself
    (that's covered in ``test_auxiliary_client.py::TestIsConnectionError``).
    """

    def _msgs(self):
        return [
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "ok"},
        ]

    def test_incomplete_chunked_read_falls_back_to_main(self):
        """``httpcore.RemoteProtocolError: incomplete chunked read`` triggers
        the retry-on-main path when ``_is_connection_error`` returns True."""
        mock_ok = MagicMock()
        mock_ok.choices = [MagicMock()]
        mock_ok.choices[0].message.content = "summary via main model"

        err = Exception("RemoteProtocolError: incomplete chunked read")

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                summary_model_override="aux-stream-model",
                quiet_mode=True,
            )

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=[err, mock_ok],
        ) as mock_call, patch(
            "agent.context_compressor._is_connection_error",
            return_value=True,
        ):
            result = c._generate_summary(self._msgs())

        assert mock_call.call_count == 2
        assert mock_call.call_args_list[0].kwargs.get("model") == "aux-stream-model"
        assert "model" not in mock_call.call_args_list[1].kwargs
        assert result is not None
        assert "summary via main model" in result

    def test_peer_closed_connection_falls_back_to_main(self):
        """``peer closed connection`` triggers the retry-on-main path."""
        mock_ok = MagicMock()
        mock_ok.choices = [MagicMock()]
        mock_ok.choices[0].message.content = "summary ok"

        err = Exception("peer closed connection without sending complete message body")

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                summary_model_override="aux-model",
                quiet_mode=True,
            )

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=[err, mock_ok],
        ) as mock_call, patch(
            "agent.context_compressor._is_connection_error",
            return_value=True,
        ):
            result = c._generate_summary(self._msgs())

        assert mock_call.call_count == 2
        assert result is not None

    def test_streaming_closed_on_main_uses_short_cooldown(self):
        """When already on the main model, a streaming-closed error should use
        the 30s cooldown, not the default 60s — these errors are transient."""
        err = Exception("RemoteProtocolError: response ended prematurely")

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                # No summary_model_override → no fallback path.
                quiet_mode=True,
            )

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=err,
        ), patch(
            "agent.context_compressor._is_connection_error",
            return_value=True,
        ), patch("agent.context_compressor.time.monotonic", return_value=1000.0):
            result = c._generate_summary(self._msgs())

        assert result is None
        # Streaming-closed should use the 30s short cooldown.
        assert c._summary_failure_cooldown_until == 1030.0

    def test_non_streaming_unknown_error_still_uses_long_cooldown(self):
        """Unclassified errors should retain the 60s default cooldown to
        prevent hammering a broken provider."""
        err = Exception("Internal Server Error: something unexpected happened")

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                quiet_mode=True,
            )

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=err,
        ), patch(
            "agent.context_compressor._is_connection_error",
            return_value=False,
        ), patch("agent.context_compressor.time.monotonic", return_value=1000.0):
            result = c._generate_summary(self._msgs())

        assert result is None
        assert c._summary_failure_cooldown_until == 1060.0


class TestAuxModelFallbackSurfacedToCallers:
    """When summary_model fails but retry-on-main succeeds, compress() must
    expose the aux-model failure via _last_aux_model_failure_{model,error}
    so gateway /compress and CLI callers can warn the user about their
    broken auxiliary.compression.model config — silent recovery would hide
    a misconfiguration only the user can fix."""

    def _make_msgs(self):
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
            {"role": "assistant", "content": "msg 6"},
            {"role": "user", "content": "msg 7"},
        ]

    def test_compress_exposes_aux_failure_fields_after_successful_fallback(self):
        mock_ok = MagicMock()
        mock_ok.choices = [MagicMock()]
        mock_ok.choices[0].message.content = "summary via main"
        err_400 = Exception("400 provider rejected configured model")
        err_400.status_code = 400

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                summary_model_override="broken-aux-model",
                quiet_mode=True,
                protect_first_n=2,
                protect_last_n=2,
            )

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=[err_400, mock_ok],
        ):
            result = c.compress(self._make_msgs())

        # Recovery succeeded → no fallback placeholder
        assert c._last_summary_fallback_used is False
        # But aux-model failure IS recorded for the gateway/CLI warning
        assert c._last_aux_model_failure_model == "broken-aux-model"
        assert c._last_aux_model_failure_error is not None
        assert "400" in c._last_aux_model_failure_error
        # Result is well-formed with a real summary, not a placeholder
        assert any(
            isinstance(m.get("content"), str) and "summary via main" in m["content"]
            for m in result
        )

    def test_compress_clears_aux_failure_fields_at_start_of_next_call(self):
        """A subsequent successful compression must clear the aux-failure
        fields so the warning doesn't persist forever."""
        mock_ok = MagicMock()
        mock_ok.choices = [MagicMock()]
        mock_ok.choices[0].message.content = "summary via main"
        err_400 = Exception("400 aux model busted")
        err_400.status_code = 400

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                summary_model_override="broken-aux-model",
                quiet_mode=True,
                protect_first_n=2,
                protect_last_n=2,
            )

        # Call 1: aux fails, retry-on-main succeeds
        with patch(
            "agent.context_compressor.call_llm",
            side_effect=[err_400, mock_ok],
        ):
            c.compress(self._make_msgs())
        assert c._last_aux_model_failure_model == "broken-aux-model"

        # Call 2: clean run on main (summary_model was cleared to "" after
        # first fallback).  Aux-failure fields MUST reset at compress() start
        # so the old warning state doesn't leak into this call.
        with patch(
            "agent.context_compressor.call_llm",
            return_value=mock_ok,
        ):
            c.compress(self._make_msgs())
        assert c._last_aux_model_failure_model is None
        assert c._last_aux_model_failure_error is None


class TestSummaryFailureTrackingForGatewayWarning:
    """Default behavior (compression.abort_on_summary_failure=False):
    summary-generation failure inserts a static fallback placeholder and
    records dropped count + fallback flag so gateway hygiene & /compress
    can surface a visible warning."""

    def test_compress_records_fallback_and_dropped_count_on_summary_failure(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
            {"role": "assistant", "content": "msg 6"},
            {"role": "user", "content": "msg 7"},
        ]

        with patch("agent.context_compressor.call_llm", side_effect=Exception("404 model not found")):
            result = c.compress(msgs)

        assert c._last_summary_fallback_used is True
        assert c._last_summary_dropped_count > 0
        assert c._last_summary_error is not None
        # Default mode: abort flag must NOT fire.
        assert c._last_compress_aborted is False
        assert any(
            isinstance(m.get("content"), str) and "Summary generation was unavailable" in m["content"]
            for m in result
        )

    def test_summary_failure_fallback_preserves_tool_paths_and_redacts_secret_context(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=1, protect_last_n=1)

        secret = "ghp_" + ("a" * 36)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": f"Fix /tmp/project/app.py and never leak {secret}"},
            {
                "role": "assistant",
                "content": "I will inspect it.",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"/tmp/project/app.py"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": f"read /tmp/project/app.py with token {secret}"},
            {"role": "assistant", "content": "Found the bug in /tmp/project/app.py"},
            {"role": "user", "content": "Patch it after this"},
            {"role": "assistant", "content": "Ready to patch"},
            {"role": "user", "content": "current live request should stay in tail"},
        ]

        with patch("agent.context_compressor.call_llm", side_effect=Exception("timeout")):
            result = c.compress(msgs)

        fallback = next(m["content"] for m in result if "Summary generation was unavailable" in m.get("content", ""))
        assert "Called tool(s): read_file" in fallback
        assert "/tmp/project/app.py" in fallback
        assert secret not in fallback
        assert "ghp_" not in fallback

    def test_summary_failure_fallback_supports_object_tool_calls_and_content_path_mentions(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=1, protect_last_n=1)

        tool_call = MagicMock()
        tool_call.id = "call-object"
        tool_call.function.name = "terminal"
        tool_call.function.arguments = '{"command":"python /repo/scripts/fix.py", "workdir":"/repo"}'
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Review ~/src/pkg/module.py before editing"},
            {"role": "assistant", "content": "Running command", "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": "call-object", "content": "Traceback in /repo/src/pkg/module.py: boom"},
            {"role": "assistant", "content": "Need to update C:\\work\\pkg\\module.py too"},
            {"role": "user", "content": "Patch ~/src/pkg/module.py after checking those files"},
            {"role": "assistant", "content": "Ready to patch"},
            {"role": "user", "content": "tail task"},
        ]

        with patch("agent.context_compressor.call_llm", side_effect=Exception("timeout")):
            result = c.compress(msgs)

        fallback = next(m["content"] for m in result if "Summary generation was unavailable" in m.get("content", ""))
        assert "Called tool(s): terminal" in fallback
        assert "/repo/scripts/fix.py" in fallback
        assert "/repo" in fallback
        assert "/repo/src/pkg/module.py" in fallback
        assert "C:\\work\\pkg\\module.py" in fallback
        assert "Traceback" in fallback
        assert "## Last Dropped Turns" in fallback
        assert "TOOL: Traceback in /repo/src/pkg/module.py: boom" in fallback

    def test_summary_failure_fallback_preserves_last_dropped_turns_without_tail(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=1, protect_last_n=1)

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Investigate dropped-window request in /tmp/active.py"},
            {"role": "assistant", "content": "I inspected /tmp/active.py and found the failing branch"},
            {"role": "tool", "tool_call_id": "call-old", "content": "ValueError: boom in /tmp/active.py"},
            {"role": "assistant", "content": "Next step is patching /tmp/active.py"},
            {"role": "user", "content": "Confirm regression coverage for /tmp/active.py"},
            {"role": "assistant", "content": "Regression note is ready"},
            {"role": "user", "content": "protected tail request must not be copied from dropped window"},
        ]

        with patch("agent.context_compressor.call_llm", side_effect=Exception("timeout")):
            result = c.compress(msgs)

        fallback = next(m["content"] for m in result if "Summary generation was unavailable" in m.get("content", ""))
        assert "## Last Dropped Turns" in fallback
        assert "ASSISTANT: I inspected /tmp/active.py and found the failing branch" in fallback
        assert "TOOL: ValueError: boom in /tmp/active.py" in fallback
        assert "protected tail request must not be copied" not in fallback

    def test_summary_failure_fallback_is_bounded(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=1, protect_last_n=1)

        long_text = "important detail " * 2000
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "head user"},
            {"role": "assistant", "content": "head assistant"},
            {"role": "user", "content": long_text},
            {"role": "assistant", "content": long_text},
            {"role": "user", "content": long_text},
            {"role": "assistant", "content": long_text},
            {"role": "user", "content": "tail"},
        ]

        with patch("agent.context_compressor.call_llm", side_effect=Exception("timeout")):
            result = c.compress(msgs)

        fallback = next(m["content"] for m in result if "Summary generation was unavailable" in m.get("content", ""))
        assert len(fallback) <= 8300
        assert "deterministic fallback" in fallback
        assert "important detail" in fallback

    def test_compress_clears_fallback_flag_on_subsequent_success(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
            {"role": "assistant", "content": "msg 6"},
            {"role": "user", "content": "msg 7"},
        ]

        with patch("agent.context_compressor.call_llm", side_effect=Exception("boom")):
            c.compress(msgs)
        assert c._last_summary_fallback_used is True

        c._summary_failure_cooldown_until = 0.0
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            c.compress(msgs)
        assert c._last_summary_fallback_used is False
        assert c._last_summary_dropped_count == 0


class TestAbortOnSummaryFailure:
    """Opt-in behavior (compression.abort_on_summary_failure=True):
    summary-generation failure ABORTS compression entirely — returns the
    original messages unchanged and sets _last_compress_aborted=True so
    gateway hygiene & /compress can surface a visible warning."""

    def _make_msgs(self):
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
            {"role": "assistant", "content": "msg 6"},
            {"role": "user", "content": "msg 7"},
        ]

    def _make_compressor(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            return ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=2,
                protect_last_n=2,
                abort_on_summary_failure=True,
            )

    def test_compress_aborts_and_preserves_messages_on_summary_failure(self):
        c = self._make_compressor()
        msgs = self._make_msgs()
        with patch("agent.context_compressor.call_llm", side_effect=Exception("404 model not found")):
            result = c.compress(msgs)

        assert c._last_compress_aborted is True
        assert c._last_summary_error is not None
        # No fallback inserted, no messages dropped
        assert c._last_summary_fallback_used is False
        assert c._last_summary_dropped_count == 0
        # Original messages preserved byte-for-byte.
        assert result == msgs
        # No "Summary generation was unavailable" placeholder leaked in.
        assert not any(
            isinstance(m.get("content"), str) and "Summary generation was unavailable" in m["content"]
            for m in result
        )

    def test_compress_clears_abort_flag_on_subsequent_success(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        c = self._make_compressor()
        msgs = self._make_msgs()

        with patch("agent.context_compressor.call_llm", side_effect=Exception("boom")):
            c.compress(msgs)
        assert c._last_compress_aborted is True

        c._summary_failure_cooldown_until = 0.0
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            c.compress(msgs)
        assert c._last_compress_aborted is False
        assert c._last_summary_fallback_used is False
        assert c._last_summary_dropped_count == 0

    def test_force_true_bypasses_failure_cooldown(self):
        """Manual /compress passes force=True so it can retry immediately
        after an auto-compress abort instead of waiting out the 30-60s
        cooldown."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        c = self._make_compressor()
        msgs = self._make_msgs()

        import time as _time
        c._summary_failure_cooldown_until = _time.monotonic() + 999.0

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs, force=True)

        assert c._last_compress_aborted is False
        assert c._summary_failure_cooldown_until == 0.0
        assert len(result) < len(msgs)

    def test_force_true_bypasses_persisted_session_cooldown(self, tmp_path):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s1", "cli")
        db.record_compression_failure_cooldown("s1", time.time() + 999.0, "timeout")

        c = self._make_compressor()
        c.bind_session_state(db, "s1")
        msgs = self._make_msgs()

        with patch("agent.context_compressor.call_llm", return_value=mock_response) as mock_llm:
            result = c.compress(msgs, current_tokens=999999, force=True)

        mock_llm.assert_called()
        assert c._last_compress_aborted is False
        assert len(result) < len(msgs)
        assert db.get_compression_failure_cooldown("s1") is None

    def test_aux_fallback_clears_persisted_session_cooldown_before_retry(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s1", "cli")
        db.record_compression_failure_cooldown("s1", time.time() + 999.0, "timeout")

        c = self._make_compressor()
        c.bind_session_state(db, "s1")
        c.summary_model = "aux/model"

        c._fallback_to_main_for_compression(Exception("provider down"), "failed")

        assert c.summary_model == ""
        assert c._summary_failure_cooldown_until == 0.0
        assert db.get_compression_failure_cooldown("s1") is None

    def test_success_clears_persisted_session_cooldown(self, tmp_path):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s1", "cli")
        db.record_compression_failure_cooldown("s1", time.time() + 999.0, "timeout")

        c = self._make_compressor()
        c.bind_session_state(db, "s1")
        c._summary_failure_cooldown_until = 0.0
        msgs = self._make_msgs()

        with patch("agent.context_compressor.call_llm", return_value=mock_response) as mock_llm:
            result = c.compress(msgs, current_tokens=999999)

        mock_llm.assert_called()
        assert c._last_compress_aborted is False
        assert len(result) < len(msgs)
        assert db.get_compression_failure_cooldown("s1") is None

    def test_session_end_does_not_clear_persisted_session_cooldown(self, tmp_path):
        db = SessionDB(db_path=tmp_path / "state.db")
        db.create_session("s1", "cli")
        db.record_compression_failure_cooldown("s1", time.time() + 999.0, "timeout")

        c = self._make_compressor()
        c.bind_session_state(db, "s1")
        c.on_session_end("s1", [])

        assert db.get_compression_failure_cooldown("s1") is not None


class TestSummaryPrefixNormalization:
    def test_legacy_prefix_is_replaced(self):
        summary = ContextCompressor._with_summary_prefix("[CONTEXT SUMMARY]: did work")
        assert summary == f"{SUMMARY_PREFIX}\ndid work"

    def test_existing_new_prefix_is_not_duplicated(self):
        summary = ContextCompressor._with_summary_prefix(f"{SUMMARY_PREFIX}\ndid work")
        assert summary == f"{SUMMARY_PREFIX}\ndid work"


class TestCompressWithClient:
    def test_system_content_list_gets_compression_note_without_crashing(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "system prompt"}]},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
            {"role": "assistant", "content": "msg 6"},
            {"role": "user", "content": "msg 7"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        assert isinstance(result[0]["content"], list)
        assert any(
            isinstance(block, dict)
            and "compacted into a handoff summary" in block.get("text", "")
            for block in result[0]["content"]
        )

    def test_summarization_path(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: stuff happened"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(10)]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        # Should have summary message in the middle
        contents = [m.get("content", "") for m in result]
        assert any(c.startswith(SUMMARY_PREFIX) for c in contents)
        assert len(result) < len(msgs)

    def test_summarization_does_not_split_tool_call_pairs(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: compressed middle"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=3,
                protect_last_n=4,
            )

        msgs = [
            {"role": "user", "content": "Could you address the reviewer comments in PR#71"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_a", "type": "function", "function": {"name": "skill_view", "arguments": "{}"}},
                    {"id": "call_b", "type": "function", "function": {"name": "skill_view", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "output a"},
            {"role": "tool", "tool_call_id": "call_b", "content": "output b"},
            {"role": "user", "content": "later 1"},
            {"role": "assistant", "content": "later 2"},
            {"role": "tool", "tool_call_id": "call_x", "content": "later output"},
            {"role": "assistant", "content": "later 3"},
            {"role": "user", "content": "later 4"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        answered_ids = {
            msg.get("tool_call_id")
            for msg in result
            if msg.get("role") == "tool" and msg.get("tool_call_id")
        }
        for msg in result:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    assert tc["id"] in answered_ids

    def test_sanitizer_matches_responses_call_id_when_id_differs(self, compressor):
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "fc_123",
                        "call_id": "call_123",
                        "response_item_id": "fc_123",
                        "type": "function",
                        "function": {"name": "search_files", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_123", "content": "result"},
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        assert [m.get("tool_call_id") for m in sanitized if m.get("role") == "tool"] == [
            "call_123"
        ]

    def test_user_role_summary_carries_end_marker(self):
        """When the summary lands as standalone role='user' (e.g. head ends
        with assistant/tool), the message body must include the explicit
        '--- END OF CONTEXT SUMMARY ---' marker. Without it, weak models
        read the verbatim past user request quoted in the historical task
        snapshot as
        fresh input (#11475, #14521).
        """
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        # head_last=assistant, tail_first=assistant (same shape as the
        # existing consecutive-user test) → role resolves to "user".
        msgs = [
            {"role": "user", "content": "msg 0"},
            {"role": "assistant", "content": "msg 1"},
            {"role": "user", "content": "msg 2"},
            {"role": "assistant", "content": "msg 3"},
            {"role": "user", "content": "msg 4"},
            {"role": "assistant", "content": "msg 5"},
            {"role": "user", "content": "msg 6"},
            {"role": "assistant", "content": "msg 7"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        summary_msg = next(
            m for m in result if (m.get("content") or "").startswith(SUMMARY_PREFIX)
        )
        assert summary_msg["role"] == "user"
        assert "END OF CONTEXT SUMMARY" in summary_msg["content"]
        assert summary_msg["content"].rstrip().endswith(
            "respond to the message below, not the summary above ---"
        )

    def test_assistant_role_summary_carries_end_marker(self):
        """When the summary lands as standalone role='assistant' (head ends
        with user), the message body must include the explicit
        '--- END OF CONTEXT SUMMARY ---' marker. Without it, models may
        regurgitate the summary text as their own output (#33256).
        """
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: stuff happened"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        # head_last=user → summary_role="assistant" (same setup as
        # test_summary_role_avoids_consecutive_user_when_head_ends_with_user).
        # With min_tail=3, tail = last 3 messages (indices 5-7).
        # head_last=user, tail_first=user → the assistant-role summary does
        # not collide with either neighbor and should be inserted standalone.
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "msg 1"},
            {"role": "user", "content": "msg 2"},  # last head — user
            {"role": "assistant", "content": "msg 3"},
            {"role": "user", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
            {"role": "assistant", "content": "msg 6"},
            {"role": "user", "content": "msg 7"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        summary_msg = next(
            m for m in result if (m.get("content") or "").startswith(SUMMARY_PREFIX)
        )
        assert summary_msg["role"] == "assistant"
        assert "END OF CONTEXT SUMMARY" in summary_msg["content"]
        assert summary_msg["content"].rstrip().endswith(
            "respond to the message below, not the summary above ---"
        )

    def test_summary_role_avoids_consecutive_user_messages(self):
        """Summary role should alternate with the last head message to avoid consecutive same-role messages."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: stuff happened"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        # Last head message (index 1) is "assistant" → summary should be "user".
        # With min_tail=3, tail = last 3 messages (indices 5-7).
        # head_last=assistant, tail_first=assistant → summary_role="user", no collision.
        # Need 8 messages: min_for_compress = 2+3+1 = 6, must have > 6.
        msgs = [
            {"role": "user", "content": "msg 0"},
            {"role": "assistant", "content": "msg 1"},
            {"role": "user", "content": "msg 2"},
            {"role": "assistant", "content": "msg 3"},
            {"role": "user", "content": "msg 4"},
            {"role": "assistant", "content": "msg 5"},
            {"role": "user", "content": "msg 6"},
            {"role": "assistant", "content": "msg 7"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)
        summary_msg = [
            m for m in result if (m.get("content") or "").startswith(SUMMARY_PREFIX)
        ]
        assert len(summary_msg) == 1
        assert summary_msg[0]["role"] == "user"

    def test_summary_role_avoids_consecutive_user_when_head_ends_with_user(self):
        """When last head message is 'user', summary must be 'assistant' to avoid two consecutive user messages."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: stuff happened"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        # Last head message (index 2) is "user" → summary should be "assistant"
        # NOTE: protect_first_n=2 preserves 2 non-system messages in addition to
        # the system prompt (always implicitly protected), yielding head [system,
        # user, user] with last head = user.
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "msg 1"},
            {"role": "user", "content": "msg 2"},  # last head — user
            {"role": "assistant", "content": "msg 3"},
            {"role": "user", "content": "msg 4"},
            {"role": "assistant", "content": "msg 5"},
            {"role": "user", "content": "msg 6"},
            {"role": "assistant", "content": "msg 7"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)
        summary_msg = [
            m for m in result if m.get(COMPRESSED_SUMMARY_METADATA_KEY)
        ]
        assert len(summary_msg) == 1
        assert summary_msg[0]["role"] == "assistant"

    def test_summary_role_flips_to_avoid_tail_collision(self):
        """When summary role collides with the first tail message but flipping
        doesn't collide with head, the role should be flipped."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        # Head ends with tool (index 1), tail starts with user (index 6).
        # Default: tool → summary_role="user" → collides with tail.
        # Flip to "assistant" → tool→assistant is fine.
        msgs = [
            {"role": "user", "content": "msg 0"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "t", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result 1"},
            {"role": "assistant", "content": "msg 3"},
            {"role": "user", "content": "msg 4"},
            {"role": "assistant", "content": "msg 5"},
            {"role": "user", "content": "msg 6"},
            {"role": "assistant", "content": "msg 7"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)
        # Verify no consecutive user or assistant messages
        for i in range(1, len(result)):
            r1 = result[i - 1].get("role")
            r2 = result[i].get("role")
            if r1 in {"user", "assistant"} and r2 in {"user", "assistant"}:
                assert r1 != r2, f"consecutive {r1} at indices {i-1},{i}"

    def test_double_collision_merges_summary_into_tail(self):
        """When neither role avoids collision with both neighbors, the summary
        should be merged into the first tail message rather than creating a
        standalone message that breaks role alternation.

        Common scenario: head ends with 'assistant', tail starts with 'user'.
        summary='user' collides with tail, summary='assistant' collides with head.
        """
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=3)

        # Head: [system, user, assistant]  →  last head = assistant
        # Tail: [user, assistant, user]    →  first tail = user
        # summary_role="user" collides with tail, "assistant" collides with head → merge
        # NOTE: protect_first_n=2 preserves 2 non-system messages in addition to
        # the system prompt (always implicitly protected).
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},      # compressed
            {"role": "assistant", "content": "msg 4"},  # compressed
            {"role": "user", "content": "msg 5"},       # compressed
            {"role": "user", "content": "msg 6"},       # tail start
            {"role": "assistant", "content": "msg 7"},
            {"role": "user", "content": "msg 8"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        # Verify no consecutive user or assistant messages
        for i in range(1, len(result)):
            r1 = result[i - 1].get("role")
            r2 = result[i].get("role")
            if r1 in {"user", "assistant"} and r2 in {"user", "assistant"}:
                assert r1 != r2, f"consecutive {r1} at indices {i-1},{i}"

        # The summary text should be merged into the first tail message
        first_tail = [m for m in result if "msg 6" in (m.get("content") or "")]
        assert len(first_tail) == 1
        assert "summary text" in first_tail[0]["content"]

    def test_double_collision_merges_summary_into_list_tail_content(self):
        """Structured tail content should accept a merged summary without TypeError."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=3)

        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
            {"role": "user", "content": [{"type": "text", "text": "msg 6"}]},
            {"role": "assistant", "content": "msg 7"},
            {"role": "user", "content": "msg 8"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        merged_tail = next(
            m for m in result
            if m.get("role") == "user" and isinstance(m.get("content"), list)
        )
        assert isinstance(merged_tail["content"], list)
        # With the fixed merge format, summary text is in the last text block
        # (after PRIOR CONTEXT and END OF PRIOR CONTEXT delimiters),
        # not necessarily in block [0].
        assert any(
            "summary text" in (block.get("text") or "")
            for block in merged_tail["content"]
            if isinstance(block, dict)
        )
        assert any(
            isinstance(block, dict) and block.get("text") == "msg 6"
            for block in merged_tail["content"]
        )

    def test_merge_into_tail_end_marker_is_last(self):
        """Regression for #56372: in a merge-into-tail summary, the END MARKER
        must come AFTER the preserved prior tail content, not before it.

        The old format was SUMMARY + END_MARKER + OLD_CONTENT, so the preserved
        tail content landed after the marker and the model could read it as a
        fresh message. The fix wraps old content in [PRIOR CONTEXT] delimiters
        and always places the END MARKER last.

        Mirrors test_double_collision_merges_summary_into_list_tail_content so
        the merged tail message genuinely carries preserved content ("msg 6").
        """
        from agent.context_compressor import _SUMMARY_END_MARKER

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "SUMMARY_BODY"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=3)

        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
            {"role": "user", "content": [{"type": "text", "text": "PRESERVED_TAIL_CONTENT"}]},
            {"role": "assistant", "content": "msg 7"},
            {"role": "user", "content": "msg 8"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        merged = next(m for m in result if m.get(COMPRESSED_SUMMARY_METADATA_KEY))
        content = merged["content"]
        text = (
            content if isinstance(content, str)
            else " ".join(
                b.get("text", "") for b in content if isinstance(b, dict)
            )
        )
        end = _SUMMARY_END_MARKER.strip()
        # All three fragments present.
        assert "PRESERVED_TAIL_CONTENT" in text
        assert "SUMMARY_BODY" in text
        assert end in text
        # Ordering invariant: prior content BEFORE summary BEFORE end marker,
        # and the end marker is the very last fragment.
        assert text.index("PRESERVED_TAIL_CONTENT") < text.index("SUMMARY_BODY")
        assert text.index("SUMMARY_BODY") < text.index(end)
        assert text.rstrip().endswith(end)

    def test_merged_tail_summary_still_detected_and_stripped(self):
        """Regression for #56372 salvage: the merge-into-tail reorder moves the
        summary prefix AFTER the [PRIOR CONTEXT] wrapper, so content-prefix
        detection (_is_context_summary_content) and body extraction
        (_strip_summary_prefix) must look past the delimiter. Otherwise a merged
        summary is mistaken for a real user turn (breaking the last-real-user
        anchor and carry-forward summary find) and the wrapper + stale tail
        content leaks into the next summarizer prompt.
        """
        from agent.context_compressor import (
            SUMMARY_PREFIX,
            _SUMMARY_END_MARKER,
            _MERGED_PRIOR_CONTEXT_HEADER,
            _MERGED_SUMMARY_DELIMITER,
        )

        merged = (
            _MERGED_PRIOR_CONTEXT_HEADER + "\n"
            "old tail content here\n\n"
            + _MERGED_SUMMARY_DELIMITER + "\n\n"
            + SUMMARY_PREFIX + "\nTHE_SUMMARY_BODY\n\n"
            + _SUMMARY_END_MARKER
        )

        # Detected as a summary despite the prefix not being at the start.
        assert ContextCompressor._is_context_summary_content(merged) is True
        # Stripping yields only the real summary body — no wrapper, no stale
        # tail content, no prefix, no end marker.
        body = ContextCompressor._strip_summary_prefix(merged)
        assert body == "THE_SUMMARY_BODY"

        # Standalone (non-merged) summaries still work unchanged.
        standalone = SUMMARY_PREFIX + "\nSTANDALONE_BODY\n\n" + _SUMMARY_END_MARKER
        assert ContextCompressor._is_context_summary_content(standalone) is True
        assert ContextCompressor._strip_summary_prefix(standalone) == "STANDALONE_BODY"

    def test_double_collision_user_head_assistant_tail(self):
        """Reverse double collision: head ends with 'user', tail starts with 'assistant'.
        summary='assistant' collides with tail, 'user' collides with head → merge."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=1, protect_last_n=2)

        # Head: [system, user]        → last head = user
        # Tail: [assistant, user, assistant] → first tail = assistant
        # summary_role="assistant" collides with tail, "user" collides with head → merge
        # NOTE: protect_first_n=1 preserves 1 non-system message in addition to
        # the system prompt (always implicitly protected).
        # With min_tail=3, tail = last 3 messages (indices 5-7).
        # Need 8 messages: _min_for_compress = head(2) + 3 + 1 = 6, must have > 6.
        msgs = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},   # compressed
            {"role": "user", "content": "msg 3"},        # compressed
            {"role": "assistant", "content": "msg 4"},   # compressed
            {"role": "assistant", "content": "msg 5"},   # tail start
            {"role": "user", "content": "msg 6"},
            {"role": "assistant", "content": "msg 7"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        # Verify no consecutive user or assistant messages
        for i in range(1, len(result)):
            r1 = result[i - 1].get("role")
            r2 = result[i].get("role")
            if r1 in {"user", "assistant"} and r2 in {"user", "assistant"}:
                assert r1 != r2, f"consecutive {r1} at indices {i-1},{i}"

        # The summary should be merged into the first tail message (assistant at index 5)
        first_tail = [m for m in result if "msg 5" in (m.get("content") or "")]
        assert len(first_tail) == 1
        assert "summary text" in first_tail[0]["content"]

    def test_no_collision_scenarios_still_work(self):
        """Verify that the common no-collision cases (head=assistant/tail=assistant,
        head=user/tail=user) still produce a standalone summary message."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary text"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2)

        # Head=assistant, Tail=assistant → summary_role="user", no collision.
        # With min_tail=3, tail = last 3 messages (indices 5-7).
        # Need 8 messages: min_for_compress = 2+3+1 = 6, must have > 6.
        msgs = [
            {"role": "user", "content": "msg 0"},
            {"role": "assistant", "content": "msg 1"},
            {"role": "user", "content": "msg 2"},
            {"role": "assistant", "content": "msg 3"},
            {"role": "user", "content": "msg 4"},
            {"role": "assistant", "content": "msg 5"},
            {"role": "user", "content": "msg 6"},
            {"role": "assistant", "content": "msg 7"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)
        summary_msgs = [m for m in result if (m.get("content") or "").startswith(SUMMARY_PREFIX)]
        assert len(summary_msgs) == 1, "should have a standalone summary message"
        assert summary_msgs[0]["role"] == "user"

    def test_summarization_does_not_start_tail_with_tool_outputs(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "[CONTEXT SUMMARY]: compressed middle"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=2,
                protect_last_n=3,
            )

        msgs = [
            {"role": "user", "content": "earlier 1"},
            {"role": "assistant", "content": "earlier 2"},
            {"role": "user", "content": "earlier 3"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_c", "type": "function", "function": {"name": "search_files", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_c", "content": "output c"},
            {"role": "user", "content": "latest user"},
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        called_ids = {
            tc["id"]
            for msg in result
            if msg.get("role") == "assistant" and msg.get("tool_calls")
            for tc in msg["tool_calls"]
        }
        for msg in result:
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                assert msg["tool_call_id"] in called_ids


class TestSummaryTargetRatio:
    """Verify that summary_target_ratio properly scales budgets with context window."""

    def test_tail_budget_scales_with_context(self):
        """Tail token budget should be threshold_tokens * summary_target_ratio."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c = ContextCompressor(model="test", quiet_mode=True, summary_target_ratio=0.40)
        # 200K < 512K → threshold floored at 75%: 150K * 0.40 ratio = 60K
        assert c.tail_token_budget == 60_000

        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            c = ContextCompressor(model="test", quiet_mode=True, summary_target_ratio=0.40)
        # 1M * 0.50 threshold * 0.40 ratio = 200K
        assert c.tail_token_budget == 200_000

    def test_summary_cap_scales_with_context(self):
        """Max summary tokens should be 5% of context, capped at 10K."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c = ContextCompressor(model="test", quiet_mode=True)
        assert c.max_summary_tokens == 10_000  # 200K * 0.05

        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            c = ContextCompressor(model="test", quiet_mode=True)
        assert c.max_summary_tokens == 10_000  # capped at 10K ceiling

    def test_ratio_clamped(self):
        """Ratio should be clamped to [0.10, 0.80]."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(model="test", quiet_mode=True, summary_target_ratio=0.05)
        assert c.summary_target_ratio == 0.10

        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(model="test", quiet_mode=True, summary_target_ratio=0.95)
        assert c.summary_target_ratio == 0.80

    def test_default_threshold_floored_at_75_percent_below_512k(self):
        """Sub-512K models get the 75% small-context threshold floor."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(model="test", quiet_mode=True)
        assert c.threshold_percent == 0.75
        # 75% of 100K = 75K, above the 64K minimum floor
        assert c.threshold_tokens == 75_000

    def test_configured_threshold_used_at_512k_and_above(self):
        """At 512K+ the configured (default 50%) percentage is used directly."""
        with patch("agent.context_compressor.get_model_context_length", return_value=512_000):
            c = ContextCompressor(model="test", quiet_mode=True)
        assert c.threshold_percent == 0.50
        assert c.threshold_tokens == 256_000

    def test_default_protect_last_n_is_20(self):
        """Default protect_last_n should be 20."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(model="test", quiet_mode=True)
        assert c.protect_last_n == 20

    def test_default_protect_first_n_is_3(self):
        """Default protect_first_n is 3 (system + 3 extra non-system messages =
        4 protected messages total when a system prompt is present). With the
        new semantics, the constructor default is 3 — the system prompt is
        always implicitly protected ON TOP OF protect_first_n non-system
        messages.
        """
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(model="test", quiet_mode=True)
        assert c.protect_first_n == 3

    def test_protect_first_n_override(self):
        """protect_first_n=0 should be honoured — for users who rely on rolling
        compaction and want NOTHING pinned at head except the system prompt
        (always implicitly protected)."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(model="test", quiet_mode=True, protect_first_n=0)
        assert c.protect_first_n == 0

    def test_protect_first_n_0_preserves_only_system_prompt(self):
        """End-to-end: when protect_first_n=0, compression should treat only
        the system prompt as head.  All user/assistant messages between the
        system prompt and the protected tail become summarization candidates.

        This is the cleanest configuration for long-running rolling-compaction
        sessions — no user/assistant turn gets pinned verbatim forever just
        because it happened to be early in the session."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=0,
                protect_last_n=2,
            )
        msgs = (
            [{"role": "system", "content": "System prompt"}]
            + [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
               for i in range(8)]
        )
        result = c.compress(msgs)
        # System prompt (msg[0]) survives as head
        assert result[0]["role"] == "system"
        assert result[0]["content"].startswith("System prompt")
        # The first user/assistant exchange (msg 0, msg 1) should NOT be pinned
        # as head verbatim — those would have been summarized or absorbed.
        # Under default protect_first_n=3, result[1..3] would be the literal
        # "msg 0" / "msg 1" / "msg 2"; with protect_first_n=0 they aren't.
        assert result[1].get("content") != "msg 0"
        # Last 2 messages are tail-protected under protect_last_n=2
        assert result[-1]["content"] == msgs[-1]["content"]

    def test_protect_first_n_semantics_stable_without_system_prompt(self):
        """Regression: gateway /compress handler strips the system prompt
        before calling compress().  protect_first_n must mean the same thing
        in both paths — "N non-system head messages" — so configuring
        protect_first_n=0 preserves NOTHING at the head regardless of whether
        the system prompt is in the messages list.

        Bug this covers: under the old semantics, protect_first_n counted
        literally from messages[0].  In the gateway path (no system prompt)
        that meant protect_first_n=1 would pin the first user turn of the
        session forever — a user-reported complaint that a week-old
        resolved question kept getting reinserted into every compaction
        summary."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=0,
                protect_last_n=2,
            )
        # No system prompt — this is what the gateway passes to compress().
        msgs = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(10)
        ]
        head_size = c._protect_head_size(msgs)
        # With no system prompt and protect_first_n=0 → head is empty.
        # The first user message is NOT pinned as head.
        assert head_size == 0

        # And with protect_first_n=3 on the same no-system-prompt list →
        # head size is 3 (the three earliest non-system messages).
        c.protect_first_n = 3
        assert c._protect_head_size(msgs) == 3


class TestTokenBudgetTailProtection:
    """Tests for token-budget-based tail protection (PR #6240).

    The core change: tail protection is now based on a token budget rather
    than a fixed message count.  This prevents large tool outputs from
    blocking compaction.
    """

    @pytest.fixture()
    def budget_compressor(self):
        """Compressor with known token budget for tail protection tests."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c = ContextCompressor(
                model="test/model",
                threshold_percent=0.50,  # 100K threshold
                protect_first_n=2,
                protect_last_n=20,
                quiet_mode=True,
            )
            return c

    def test_large_tool_outputs_no_longer_block_compaction(self, budget_compressor):
        """The motivating scenario: 20 messages with large tool outputs should
        NOT prevent compaction.  With message-count tail protection they would
        all be protected, leaving nothing to summarize."""
        c = budget_compressor
        messages = [
            {"role": "user", "content": "Start task"},
            {"role": "assistant", "content": "On it"},
        ]
        # Add 20 messages with large tool outputs (~5K chars each ≈ 1250 tokens)
        for i in range(10):
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{"function": {"name": f"tool_{i}", "arguments": "{}"}}],
            })
            messages.append({
                "role": "tool", "content": "x" * 5000,
                "tool_call_id": f"call_{i}",
            })
        # Add 3 recent small messages
        messages.append({"role": "user", "content": "What's the status?"})
        messages.append({"role": "assistant", "content": "Here's what I found..."})
        messages.append({"role": "user", "content": "Continue"})

        # The tail cut should NOT protect all 20 tool messages
        head_end = c.protect_first_n
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        tail_size = len(messages) - cut
        # With token budget, the tail should be much smaller than 20+
        assert tail_size < 20, f"Tail {tail_size} messages — large tool outputs are blocking compaction"
        # But at least 3 (hard minimum)
        assert tail_size >= 3

    def test_min_tail_always_3_messages(self, budget_compressor):
        """Even with a tiny token budget, at least 3 messages are protected."""
        c = budget_compressor
        # Override to a tiny budget
        c.tail_token_budget = 10
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "do something"},
            {"role": "assistant", "content": "working on it"},
            {"role": "user", "content": "more work"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "thanks"},
        ]
        head_end = 2
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        tail_size = len(messages) - cut
        assert tail_size >= 3, f"Tail is only {tail_size} messages, min should be 3"

    def test_tiny_budget_preserves_bounded_recent_turns(self, budget_compressor):
        """A token-exhausted tail must preserve more than just the latest ask.

        Regression for #9413: the previous hard-coded 3-message floor could
        leave the latest user message live while summarizing the assistant/tool
        context immediately before it, which made the post-compression turn feel
        like a fresh conversation.
        """
        c = budget_compressor
        c.tail_token_budget = 10
        c.protect_last_n = 20
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "old start"},
            {"role": "assistant", "content": "old ack"},
            {"role": "user", "content": "middle work"},
            {"role": "assistant", "content": "middle ack"},
            {"role": "user", "content": "middle ask 2"},
            {"role": "assistant", "content": "middle answer 2"},
            {"role": "user", "content": "middle ask 3"},
            {"role": "assistant", "content": "middle answer 3"},
            {"role": "user", "content": "recent ask 1"},
            {"role": "assistant", "content": "recent answer 1"},
            {"role": "user", "content": "recent ask 2"},
            {"role": "assistant", "content": "recent answer 2"},
            {"role": "user", "content": "latest ask"},
        ]

        cut = c._find_tail_cut_by_tokens(messages, head_end=1)

        assert len(messages) - cut >= 8
        assert messages[cut]["content"] == "middle answer 2"
        assert messages[-1]["content"] == "latest ask"

    def test_soft_ceiling_allows_oversized_message(self, budget_compressor):
        """The 1.5x soft ceiling allows an oversized message to be included
        rather than splitting it."""
        c = budget_compressor
        # Set a small budget — 500 tokens
        c.tail_token_budget = 500
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "read the file"},
            # This message is ~600 tokens (> budget of 500, but < 1.5x = 750)
            {"role": "assistant", "content": "a" * 2400},
            {"role": "user", "content": "short"},
            {"role": "assistant", "content": "short reply"},
            {"role": "user", "content": "continue"},
        ]
        head_end = 2
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        # The oversized message at index 3 should NOT be the cut point
        # because 1.5x ceiling = 750 tokens and accumulated would be ~610
        # (short msgs + oversized msg) which is < 750
        tail_size = len(messages) - cut
        assert tail_size >= 3

    def test_small_conversation_still_compresses(self, budget_compressor):
        """With the new min of 8 messages (head=2 + 3 + 1 guard + 2 middle),
        a small but compressible conversation should still compress."""
        c = budget_compressor
        # 9 messages: head(2) + 4 middle + 3 tail = compressible
        messages = []
        for i in range(9):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": f"Message {i}"})

        # Should not early-return (needs > protect_first_n + 3 + 1 = 6)
        # Mock the summary generation to avoid real API call
        with patch.object(c, "_generate_summary", return_value="Summary of conversation"):
            result = c.compress(messages, current_tokens=90_000)
        # Should have compressed (fewer messages than original)
        assert len(result) < len(messages)

    def test_prune_with_token_budget(self, budget_compressor):
        """_prune_old_tool_results with protect_tail_tokens respects the budget."""
        c = budget_compressor
        messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "big.txt"}'}}]},
            {"role": "tool", "content": "x" * 10000, "tool_call_id": "c1"},  # ~2500 tokens
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "read_file", "arguments": '{"path": "small.txt"}'}}]},
            {"role": "tool", "content": "y" * 10000, "tool_call_id": "c2"},  # ~2500 tokens
            {"role": "user", "content": "short recent message"},
            {"role": "assistant", "content": "short reply"},
        ]
        # With a 1000-token budget, only the last couple messages should be protected
        result, pruned = c._prune_old_tool_results(
            messages, protect_tail_count=2, protect_tail_tokens=1000,
        )
        # At least one old tool result should have been pruned
        assert pruned >= 1

    def test_prune_short_conv_protects_entire_tail(self, budget_compressor):
        """Regression guard for PR #17025.

        When ``len(messages) <= protect_tail_count`` and a token budget is
        also set, every message must be protected. The previous code used
        ``min(protect_tail_count, len(result) - 1)`` which capped the floor
        one below the full length, leaving the oldest message eligible for
        pruning.
        """
        c = budget_compressor
        # 4 messages, protect_tail_count=4 -- nothing should be pruned.
        # Oldest message is a large tool result; on the buggy path it falls
        # outside the protected window and gets summarized.
        messages = [
            {"role": "tool", "content": "x" * 5000, "tool_call_id": "c0"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "recent"},
            {"role": "assistant", "content": "reply"},
        ]
        result, pruned = c._prune_old_tool_results(
            messages,
            protect_tail_count=4,
            protect_tail_tokens=1_000_000,  # budget large enough to protect all
        )
        assert pruned == 0
        # Tool result at index 0 must be preserved verbatim
        assert result[0]["content"] == "x" * 5000

    def test_prune_without_token_budget_uses_message_count(self, budget_compressor):
        """Without protect_tail_tokens, falls back to message-count behavior."""
        c = budget_compressor
        messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "tool", "arguments": "{}"}}]},
            {"role": "tool", "content": "x" * 5000, "tool_call_id": "c1"},
            {"role": "user", "content": "recent"},
            {"role": "assistant", "content": "reply"},
        ]
        # protect_tail_count=3 means last 3 messages protected
        result, pruned = c._prune_old_tool_results(
            messages, protect_tail_count=3,
        )
        # Tool at index 2 is outside the protected tail (last 3 = indices 2,3,4)
        # so it might or might not be pruned depending on boundary
        assert isinstance(pruned, int)

    def test_multimodal_message_accumulates_text_chars_not_block_count(self, budget_compressor):
        """_find_tail_cut_by_tokens must use text char count, not list length,
        for multimodal content. Regression guard for #16087.

        Setup: 6 messages, budget=80 (soft_ceiling=120).  The multimodal message
        at index 1 has 500 chars of text → 135 tokens (correct) or 10 tokens (bug).

        Fixed path: walk stops at the multimodal (44+135=179 > 120), cut stays at 2,
        tail = messages[2:] = 4 messages.

        Bug path: walk counts only 10 tokens for the multimodal, exhausts to head_end,
        the head_end safeguard forces cut = n - min_tail = 3, tail = only 3 messages.
        """
        c = budget_compressor
        # 500 chars → 500//4 + 10 = 135 tokens; len([text, image]) // 4 + 10 = 10 (bug)
        big_text = "x" * 500
        multimodal_content = [
            {"type": "text", "text": big_text},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ]
        messages = [
            {"role": "user", "content": "head1"},               # 0
            {"role": "user", "content": multimodal_content},    # 1: BIG (index under test)
            {"role": "assistant", "content": "tail1"},           # 2
            {"role": "user", "content": "tail2"},                # 3
            {"role": "assistant", "content": "tail3"},           # 4
            {"role": "user", "content": "tail4"},                # 5
        ]
        c.tail_token_budget = 80  # soft_ceiling = 120
        head_end = 0
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        # With the fix: cut=2, tail has 4 messages (soft_ceiling not exceeded by tail1-4).
        # With the bug: head_end safeguard fires → cut = n - min_tail = 3, only 3 in tail.
        assert len(messages) - cut >= 4, (
            f"Expected ≥4 messages in tail (got {len(messages) - cut}, cut={cut}). "
            "The multimodal message was underestimated — len(list) used instead of text chars."
        )

    def test_plain_string_content_unchanged(self, budget_compressor):
        """Plain string content must still be estimated correctly after the fix."""
        c = budget_compressor
        # Same layout as the multimodal test but with a plain 500-char string.
        # Both buggy and fixed code count plain strings the same way (len(str)).
        # With 135 tokens the plain string also exceeds soft_ceiling=120, so
        # the walk stops at index 1 and tail has 4 messages — same as the fix path.
        big_plain = "x" * 500
        messages = [
            {"role": "user", "content": "head1"},
            {"role": "user", "content": big_plain},   # 1: 135 tokens, plain string
            {"role": "assistant", "content": "tail1"},
            {"role": "user", "content": "tail2"},
            {"role": "assistant", "content": "tail3"},
            {"role": "user", "content": "tail4"},
        ]
        c.tail_token_budget = 80
        head_end = 0
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        assert len(messages) - cut >= 4, (
            f"Plain string regression: expected ≥4 messages in tail, got {len(messages) - cut}"
        )

    def test_image_only_block_contributes_zero_text_chars(self, budget_compressor):
        """Image-only content blocks (no 'text' key) contribute 0 chars + base overhead."""
        c = budget_compressor
        c.tail_token_budget = 500
        image_only = [{"type": "image_url", "image_url": {"url": "https://example.com/x.jpg"}}]
        messages = [
            {"role": "user", "content": "a" * 4000},
            {"role": "user", "content": image_only},   # 0 text chars → 10 tokens overhead
            {"role": "assistant", "content": "ok"},
        ]
        head_end = 0
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        assert isinstance(cut, int)
        assert 0 <= cut <= len(messages)

    def test_mixed_list_with_bare_strings_does_not_crash(self, budget_compressor):
        """Content list may contain bare strings (not dicts) — must not raise AttributeError."""
        c = budget_compressor
        c.tail_token_budget = 500
        # Bare string item alongside a dict item — normalisation elsewhere allows this.
        mixed_content = ["Hello, world!", {"type": "text", "text": "extra text"}]
        messages = [
            {"role": "user", "content": mixed_content},
            {"role": "assistant", "content": "ok"},
        ]
        head_end = 0
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        assert isinstance(cut, int)
        assert 0 <= cut <= len(messages)

    def test_generous_budget_protects_everything_floor_does_not_override(
        self, budget_compressor
    ):
        """A budget that covers the whole transcript must prune nothing —
        ``protect_tail_count`` is a minimum floor, not a ceiling."""
        c = budget_compressor

        # 100 alternating assistant/tool messages.  Each tool result has
        # *unique* content so the dedup pass (Pass 1, which is independent
        # of prune_boundary) is a no-op and we isolate the boundary logic.
        messages = []
        for i in range(50):
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": f"c{i}",
                    "type": "function",
                    "function": {"name": "noop", "arguments": "{}"},
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": f"c{i}",
                "content": f"unique-tool-output-{i:03d}-" + ("x" * 250),
            })

        # Budget large enough to cover the whole transcript many times over,
        # so the budget walk completes without hitting its break condition
        # and the boundary lands at 0 ("protect everything").
        _, pruned = c._prune_old_tool_results(
            messages,
            protect_tail_count=20,
            protect_tail_tokens=10_000_000,
        )

        assert pruned == 0, (
            "budget said protect everything, but the floor still pruned "
            f"{pruned} messages — protect_tail_count is acting as a ceiling, "
            "not a minimum floor"
        )


class TestUpdateModelBudgets:
    """Regression: update_model() must recalculate token budgets."""

    def test_tail_budget_recalculated(self):
        """tail_token_budget must change after switching to a different context length."""
        from unittest.mock import patch
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            comp = ContextCompressor("model-a", threshold_percent=0.50, quiet_mode=True)
        old_tail = comp.tail_token_budget
        old_max_summary = comp.max_summary_tokens

        comp.update_model("model-b", context_length=32_000)
        assert comp.tail_token_budget != old_tail, "tail_token_budget should change"
        assert comp.tail_token_budget < old_tail, "smaller context → smaller budget"
        assert comp.max_summary_tokens != old_max_summary, "max_summary_tokens should change"

    def test_budgets_proportional(self):
        """Budgets should be proportional to context_length after update."""
        from unittest.mock import patch
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            comp = ContextCompressor("model-a", threshold_percent=0.50, quiet_mode=True)
        comp.update_model("model-b", context_length=10_000)
        assert comp.tail_token_budget == int(comp.threshold_tokens * comp.summary_target_ratio)
        assert comp.max_summary_tokens == min(int(10_000 * 0.05), 4000)


class TestUpdateModelResetsCalibration:
    """#23767: update_model() must clear stale cross-call calibration state.

    Old-model real-usage / defer baselines must not suppress a preflight
    compression the new (smaller) model actually needs.
    """

    def _comp(self):
        from unittest.mock import patch
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            return ContextCompressor("big-model", threshold_percent=0.50, quiet_mode=True)

    def test_real_usage_state_cleared(self):
        comp = self._comp()
        # Simulate a large-model session that proved a prompt fit.
        comp.last_prompt_tokens = 120_000
        comp.last_real_prompt_tokens = 120_000
        comp.last_rough_tokens_when_real_prompt_fit = 130_000
        comp.last_compression_rough_tokens = 130_000
        comp.awaiting_real_usage_after_compression = True
        comp._ineffective_compression_count = 2

        comp.update_model("small-model", context_length=65_536)

        assert comp.last_prompt_tokens == 0
        assert comp.last_real_prompt_tokens == 0
        assert comp.last_rough_tokens_when_real_prompt_fit == 0
        assert comp.last_compression_rough_tokens == 0
        assert comp.awaiting_real_usage_after_compression is False
        assert comp._ineffective_compression_count == 0

    def test_defer_no_longer_suppresses_after_switch(self):
        """The exact #23767 failure: old model's 'it fit' must not defer
        preflight on the new smaller model."""
        comp = self._comp()
        comp.last_real_prompt_tokens = 50_000
        comp.last_rough_tokens_when_real_prompt_fit = 90_000
        # Before switch, a modest rough growth would defer.
        comp.threshold_tokens = 85_000
        assert comp.should_defer_preflight_to_real_usage(93_000) is True

        # After switching to a 65K model, the stale state is gone, so a rough
        # estimate over the new threshold is NOT deferred — preflight will run.
        comp.update_model("small-model", context_length=65_536)
        assert comp.should_defer_preflight_to_real_usage(comp.threshold_tokens + 5_000) is False


    def test_summary_failure_cooldown_cleared(self):
        """Stale summary-failure cooldown from the old model must not block
        the new model from generating summaries after a switch."""
        import time
        comp = self._comp()
        # Simulate a 600-second cooldown set because the old model had no
        # provider configured for summarization.
        comp._summary_failure_cooldown_until = time.monotonic() + 600

        comp.update_model("new-model", context_length=128_000)

        assert comp._summary_failure_cooldown_until == 0.0

    def test_summary_failure_cooldown_survives_same_runtime_refresh(self):
        """Refreshing metadata for the same runtime must not defeat backoff."""
        import time
        comp = self._comp()
        cooldown_until = time.monotonic() + 600
        comp._summary_failure_cooldown_until = cooldown_until

        comp.update_model("big-model", context_length=128_000)

        assert comp._summary_failure_cooldown_until == cooldown_until


class TestTruncateToolCallArgsJson:
    """Regression tests for #11762.

    The previous implementation produced invalid JSON by slicing
    ``function.arguments`` mid-string, which caused non-retryable 400s from
    strict providers (observed on MiniMax) and stuck long sessions in a
    re-send loop. The helper here must always emit parseable JSON whose
    shape matches the original — shrunken, not corrupted.
    """

    def _helper(self):
        from agent.context_compressor import _truncate_tool_call_args_json
        return _truncate_tool_call_args_json

    def test_shrunken_args_remain_valid_json(self):
        import json as _json
        shrink = self._helper()
        original = _json.dumps({
            "path": "~/.hermes/skills/shopping/browser-setup-notes.md",
            "content": "# Shopping Browser Setup Notes\n\n" + "abc " * 400,
        })
        assert len(original) > 500
        shrunk = shrink(original)
        parsed = _json.loads(shrunk)  # must not raise
        assert parsed["path"] == "~/.hermes/skills/shopping/browser-setup-notes.md"
        assert parsed["content"].endswith("...[truncated]")
        assert len(shrunk) < len(original)

    def test_non_json_arguments_pass_through(self):
        shrink = self._helper()
        not_json = "this is not json at all, " * 50
        assert shrink(not_json) == not_json

    def test_short_string_leaves_unchanged(self):
        import json as _json
        shrink = self._helper()
        payload = _json.dumps({"command": "ls -la", "cwd": "/tmp"})
        assert _json.loads(shrink(payload)) == {"command": "ls -la", "cwd": "/tmp"}

    def test_nested_structures_are_walked(self):
        import json as _json
        shrink = self._helper()
        payload = _json.dumps({
            "messages": [
                {"role": "user", "content": "x" * 500},
                {"role": "assistant", "content": "ok"},
            ],
            "meta": {"note": "y" * 500},
        })
        parsed = _json.loads(shrink(payload))
        assert parsed["messages"][0]["content"].endswith("...[truncated]")
        assert parsed["messages"][1]["content"] == "ok"
        assert parsed["meta"]["note"].endswith("...[truncated]")

    def test_non_string_leaves_preserved(self):
        import json as _json
        shrink = self._helper()
        payload = _json.dumps({
            "retries": 3,
            "enabled": True,
            "timeout": None,
            "items": [1, 2, 3],
            "note": "z" * 500,
        })
        parsed = _json.loads(shrink(payload))
        assert parsed["retries"] == 3
        assert parsed["enabled"] is True
        assert parsed["timeout"] is None
        assert parsed["items"] == [1, 2, 3]
        assert parsed["note"].endswith("...[truncated]")

    def test_scalar_json_string_gets_shrunk(self):
        import json as _json
        shrink = self._helper()
        payload = _json.dumps("q" * 500)
        parsed = _json.loads(shrink(payload))
        assert isinstance(parsed, str)
        assert parsed.endswith("...[truncated]")

    def test_unicode_preserved(self):
        import json as _json
        shrink = self._helper()
        payload = _json.dumps({"content": "非德满" + ("a" * 500)})
        out = shrink(payload)
        # ensure_ascii=False keeps CJK intact rather than emitting \uXXXX
        assert "非德满" in out

    def test_pass3_emits_valid_json_for_downstream_provider(self):
        """End-to-end: Pass 3 must never produce the exact failure payload
        that caused the 400 loop (unterminated string, missing brace)."""
        import json as _json
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test/model",
                threshold_percent=0.85,
                protect_first_n=1,
                protect_last_n=1,
                quiet_mode=True,
            )
        huge_content = "# Shopping Browser Setup Notes\n\n## Overview\n" + "x " * 400
        args_payload = _json.dumps({
            "path": "~/.hermes/skills/shopping/browser-setup-notes.md",
            "content": huge_content,
        })
        assert len(args_payload) > 500  # triggers the Pass-3 shrink
        messages = [
            {"role": "user", "content": "please write two files"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "write_file", "arguments": args_payload}},
            ]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": '{"bytes_written": 727}'},
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "done"},
        ]
        result, _ = c._prune_old_tool_results(messages, protect_tail_count=2)
        shrunk = result[1]["tool_calls"][0]["function"]["arguments"]
        # Must parse — otherwise downstream provider returns 400
        parsed = _json.loads(shrunk)
        assert parsed["path"] == "~/.hermes/skills/shopping/browser-setup-notes.md"
        assert parsed["content"].endswith("...[truncated]")


class TestPreflightSentinelGuard:
    """Regression for #36718: the preflight token-display seed in
    run_conversation must NOT overwrite the -1 sentinel that
    compress_context() sets immediately after compression.

    The old guard `_preflight_tokens > (last_prompt_tokens or 0)` evaluated
    `(-1 or 0)` -> -1 (truthy), so any positive preflight estimate was > -1
    and clobbered the sentinel with a schema-inflated rough count, re-firing
    compression on the next turn. The fix treats any negative value as
    "no real usage yet" and skips the seed.
    """

    def _seed(self, last_prompt_tokens, preflight_tokens):
        # Mirror the exact guard in agent/conversation_loop.py run_conversation.
        _last = last_prompt_tokens
        if _last >= 0 and preflight_tokens > _last:
            return preflight_tokens  # would overwrite
        return last_prompt_tokens   # preserved

    def test_sentinel_preserved_after_compression(self, compressor):
        compressor.last_prompt_tokens = -1
        # A large schema-inflated preflight estimate must NOT overwrite -1.
        result = self._seed(compressor.last_prompt_tokens, 250_000)
        assert result == -1

    def test_real_value_still_revises_upward(self, compressor):
        compressor.last_prompt_tokens = 10_000
        result = self._seed(compressor.last_prompt_tokens, 50_000)
        assert result == 50_000

    def test_real_value_not_revised_downward(self, compressor):
        compressor.last_prompt_tokens = 50_000
        result = self._seed(compressor.last_prompt_tokens, 10_000)
        assert result == 50_000


class TestTurnPairPreservation:
    """Causal Coupling guard (#22523): compaction must never orphan a user turn.

    ``_ensure_last_user_message_in_tail`` pulls the cut back to keep the last
    user message in the tail (fixes #10896).  But its final
    ``max(last_user_idx, head_end + 1)`` clamp pushes the cut *past* the user
    when the user sits at ``head_end`` (the first compressible index) — the
    only case where ``head_end + 1 > last_user_idx``.  The user then lands in
    the compressed region without its assistant reply; the summariser marks it
    as a pending ask and the next session re-executes the completed task.

    The guard detects that split and pushes the cut forward to ``pair_end`` so
    the complete (user -> assistant [-> tool results]) pair is summarised as a
    finished unit.
    """

    @pytest.fixture
    def compressor(self):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=0,
            quiet_mode=True,
        )

    # ------------------------------------------------------------------
    # _find_turn_pair_end unit tests
    # ------------------------------------------------------------------

    def test_pair_end_user_only(self, compressor):
        """User at end of list — no reply yet — pair_end is user+1."""
        msgs = [{"role": "user", "content": "hello"}]
        assert compressor._find_turn_pair_end(msgs, 0) == 1

    def test_pair_end_user_with_assistant_reply(self, compressor):
        """User + assistant — pair_end skips both."""
        msgs = [
            {"role": "user", "content": "do x"},
            {"role": "assistant", "content": "done"},
        ]
        assert compressor._find_turn_pair_end(msgs, 0) == 2

    def test_pair_end_user_assistant_with_tools(self, compressor):
        """User + assistant + tool results — pair_end skips the whole group."""
        msgs = [
            {"role": "user", "content": "run it"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "exec", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
            {"role": "tool", "tool_call_id": "c2", "content": "ok"},
        ]
        assert compressor._find_turn_pair_end(msgs, 0) == 4

    def test_pair_end_stops_at_next_user(self, compressor):
        """pair_end must not cross into the next user turn."""
        msgs = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "second"},
        ]
        assert compressor._find_turn_pair_end(msgs, 0) == 2

    # ------------------------------------------------------------------
    # _ensure_last_user_message_in_tail unit tests
    # ------------------------------------------------------------------

    def test_user_already_in_tail_unchanged(self, compressor):
        """When the user message is already past cut_idx, nothing changes."""
        msgs = [
            {"role": "user", "content": "head"},
            {"role": "assistant", "content": "head reply"},
            {"role": "user", "content": "last user"},
            {"role": "assistant", "content": "last reply"},
        ]
        result = compressor._ensure_last_user_message_in_tail(msgs, cut_idx=2, head_end=1)
        assert result == 2

    def test_user_in_compressed_region_pulled_back(self, compressor):
        """User in the middle (not at head_end) is pulled into the tail (#10896)."""
        msgs = [
            {"role": "user", "content": "head"},       # 0
            {"role": "assistant", "content": "hi"},     # 1
            {"role": "user", "content": "do thing"},   # 2  <- last user
            {"role": "assistant", "content": "done"},  # 3
        ]
        # head_end=0, so head_end+1=1 <= last_user_idx=2: the #10896 pullback
        # applies and the user stays in the tail (no forward push).
        result = compressor._ensure_last_user_message_in_tail(msgs, cut_idx=3, head_end=0)
        assert result <= 2

    def test_orphan_prevented_user_at_head_end(self, compressor):
        """Causal Coupling: user at head_end pushes the WHOLE pair into the summary.

        This is the #22523 case: last_user_idx == head_end, so the clamp would
        return head_end+1 and orphan the user.  The guard instead pushes the
        cut forward to pair_end so user + reply + tool results are summarised
        together and the tail never starts with a dangling user ask.
        """
        msgs = [
            {"role": "user", "content": "first exchange"},   # 0 head
            {"role": "user", "content": "THE ACTIVE ASK"},   # 1 = head_end, last user
            {"role": "assistant", "content": "done"},        # 2 reply
            {"role": "tool", "tool_call_id": "c1", "content": "toolout"},  # 3
            {"role": "assistant", "content": "final reply"}, # 4
        ]
        head_end = 1
        result = compressor._ensure_last_user_message_in_tail(msgs, cut_idx=3, head_end=head_end)
        # Whole pair (indices 1..3) lands in the compressed region; tail starts at 4.
        assert result == 4
        tail = msgs[result:]
        assert tail and tail[0]["role"] == "assistant"

    def test_no_orphan_after_full_compaction_cycle(self, compressor):
        """End-to-end: after _find_tail_cut_by_tokens, the tail never starts
        with an unanswered user message."""
        msgs = [
            {"role": "user", "content": "initial"},
            {"role": "assistant", "content": "ok"},
        ]
        for i in range(5):
            msgs.append({"role": "user", "content": f"step {i}"})
            msgs.append({"role": "assistant", "content": f"done {i}"})
        msgs.append({"role": "user", "content": "lights off please"})
        msgs.append({"role": "assistant", "content": "lights are off"})

        head_end = compressor.protect_first_n
        cut = compressor._find_tail_cut_by_tokens(msgs, head_end)
        tail = msgs[cut:]

        if tail and tail[0].get("role") == "user":
            assert len(tail) >= 2 and tail[1].get("role") == "assistant", (
                f"Orphan user turn at tail start: {tail[0]['content']!r} — "
                f"next role is {tail[1].get('role') if len(tail) > 1 else 'nothing'}"
            )


class TestSanitizerStripsOrphanedToolCalls:
    """PR #51218 (salvaged from #51225): orphaned tool_calls are stripped from
    assistant messages instead of having stub tool results inserted, avoiding
    the call_id != id mismatch that let downstream repair_message_sequence drop
    the stubs and re-expose orphans."""

    def test_sanitizer_strips_orphaned_tool_calls(self, compressor):
        """Orphaned tool_calls (no matching tool result) are stripped from
        assistant messages instead of having stubs inserted.  #51218"""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_orphan", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            {"role": "user", "content": "never mind"},
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        # Orphaned tool_call should be stripped, not stub-inserted
        asst = next(m for m in sanitized if m.get("role") == "assistant")
        assert not asst.get("tool_calls"), "orphaned tool_calls should be stripped"
        # No stub tool messages should be added
        assert not any(m.get("role") == "tool" for m in sanitized)
        # Empty assistant should get placeholder content
        assert asst.get("content") == "(tool call removed)"

    def test_sanitizer_strips_orphaned_keeps_valid(self, compressor):
        """When an assistant has both valid and orphaned tool_calls, only
        the orphans are stripped.  #51218"""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "tc_valid", "function": {"name": "read_file", "arguments": "{}"}},
                    {"id": "tc_orphan", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc_valid", "content": "file content"},
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        asst = next(m for m in sanitized if m.get("role") == "assistant")
        assert len(asst["tool_calls"]) == 1
        assert asst["tool_calls"][0]["id"] == "tc_valid"
        # Valid tool result preserved
        tool_msgs = [m for m in sanitized if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0]["tool_call_id"] == "tc_valid"

    def test_sanitizer_strips_orphaned_preserves_text_content(self, compressor):
        """When an assistant has text content AND orphaned tool_calls,
        the text is preserved and only tool_calls are stripped.  #51218"""
        msgs = [
            {
                "role": "assistant",
                "content": "Let me search for that.",
                "tool_calls": [
                    {"id": "tc_orphan", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            {"role": "user", "content": "thanks"},
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        asst = next(m for m in sanitized if m.get("role") == "assistant")
        assert asst["content"] == "Let me search for that."
        assert not asst.get("tool_calls")
        # The placeholder must NOT overwrite existing text content.
        assert asst["content"] != "(tool call removed)"

    def test_sanitizer_strips_orphaned_with_call_id_mismatch(self, compressor):
        """Stubs with call_id != id used to be dropped by downstream
        repair_message_sequence, re-exposing orphans.  Stripping avoids
        this entirely.  #51218"""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "fc_abc",
                        "call_id": "call_abc",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                ],
            },
            # No tool result for call_abc — orphaned
            {"role": "user", "content": "next"},
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        asst = next(m for m in sanitized if m.get("role") == "assistant")
        assert not asst.get("tool_calls")
        # No stub tool messages (which would have call_id != id mismatch)


class TestCooldownReentryAbort:
    """Regression: a second compress() call during the failure cooldown must
    still abort when the original failure was a network/auth error.

    Before the fix, compress() unconditionally reset _last_summary_network_failure
    and _last_summary_auth_failure at the top of every call.  When
    _generate_summary() returned None from the cooldown early-return (without
    re-setting the flags), the abort guard saw False and fell through to the
    destructive static-fallback path — reproducing the data-loss scenario from
    #29559 / #25585 that PR #51881 originally fixed.
    """

    def _msgs(self, n=12):
        return [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"}
            for i in range(n)
        ]

    def test_network_failure_cooldown_reentry_still_aborts(self):
        """ConnectionError → first compress aborts (PR #51881).  Second
        compress within the 30s cooldown must ALSO abort — not drop the
        middle window via the static-fallback path."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=2,
                protect_last_n=2,
                abort_on_summary_failure=False,
            )
        msgs = self._msgs(12)

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=ConnectionError("Connection error."),
        ):
            first = c.compress(msgs, current_tokens=999999, force=True)
        assert first == msgs
        assert c._last_compress_aborted is True
        assert c._last_summary_network_failure is True

        second = c.compress(msgs, current_tokens=999999)
        assert second == msgs, (
            "Second compress during cooldown must abort (preserve messages), "
            "not drop the middle window via static-fallback"
        )
        assert c._last_compress_aborted is True
        assert c._last_summary_fallback_used is False

    def test_auth_failure_cooldown_reentry_still_aborts(self):
        """Same re-entry hole for auth failures: a 401 sets the flag, cooldown
        returns None, second compress must still abort."""
        err = Exception("Error code: 401 - invalid api key")
        err.status_code = 401
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test",
                quiet_mode=True,
                protect_first_n=2,
                protect_last_n=2,
                abort_on_summary_failure=False,
            )
        msgs = self._msgs(12)

        with patch("agent.context_compressor.call_llm", side_effect=err):
            first = c.compress(msgs, current_tokens=999999, force=True)
        assert first == msgs
        assert c._last_compress_aborted is True
        assert c._last_summary_auth_failure is True

        second = c.compress(msgs, current_tokens=999999)
        assert second == msgs, (
            "Second compress during cooldown must abort (preserve messages), "
            "not drop the middle window via static-fallback"
        )
        assert c._last_compress_aborted is True
        assert c._last_summary_fallback_used is False


class TestDoubleCompactionSummaryRole:
    """PR #52160 (salvaged from #52167): when only the system prompt is
    protected, the summary must lead with role=user (Anthropic/Bedrock send
    system as a separate param, so the summary is the first visible message)."""

    def test_double_compaction_summary_must_be_user_when_only_system_protected(self):
        """After the first compression, protect_first_n decays to 0.

        On the second compression the only protected head message is the
        system prompt (role=system).  The summary becomes the first
        *visible* message in the API request because adapters like
        Anthropic and Bedrock send the system prompt as a separate
        ``system`` parameter.  The summary MUST be role=user or the
        provider rejects with HTTP 400 (#52160).
        """
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary of earlier turns"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2,
            )
        # Simulate second compression: protect_first_n decays to 0.
        c.compression_count = 1

        # compress_start will be 1 (system only), last_head_role = "system".
        # Without the fix, summary_role would be "assistant".
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},
            {"role": "assistant", "content": "msg 6"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        # The system message must still be at index 0.
        assert result[0]["role"] == "system"
        # The summary (first non-system message) must be role=user.
        non_system = [m for m in result if m.get("role") != "system"]
        assert non_system, "expected at least one non-system message"
        assert non_system[0]["role"] == "user", (
            f"first non-system message must be role=user for Anthropic "
            f"compatibility, got role={non_system[0]['role']!r}"
        )

    def test_double_compaction_user_tail_merges_into_tail(self):
        """When the summary is forced to role=user (system-only head) and
        the first tail message is also user, the summary must merge into
        the tail rather than flipping back to assistant (#52160).
        """
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary of earlier turns"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test", quiet_mode=True, protect_first_n=2, protect_last_n=2,
            )
        c.compression_count = 1  # decay protect_first_n

        # tail starts with user → would collide with forced summary_role=user.
        # The fix should merge into tail instead of flipping to assistant.
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "msg 1"},
            {"role": "assistant", "content": "msg 2"},
            {"role": "user", "content": "msg 3"},
            {"role": "assistant", "content": "msg 4"},
            {"role": "user", "content": "msg 5"},       # tail start (user)
            {"role": "assistant", "content": "msg 6"},
            {"role": "user", "content": "msg 7"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        # No standalone summary message should exist (merged into tail).
        summary_msgs = [
            m for m in result
            if m.get("_compressed_summary") and "msg 5" not in (m.get("content") or "")
        ]
        assert len(summary_msgs) == 0, (
            "summary should be merged into tail, not standalone"
        )
        # The first non-system message must be role=user.
        non_system = [m for m in result if m.get("role") != "system"]
        assert non_system[0]["role"] == "user"
        # The merged tail should contain the summary text.
        assert any(
            "summary of earlier turns" in (m.get("content") or "")
            for m in result
        )
