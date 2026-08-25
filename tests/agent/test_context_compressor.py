"""Tests for agent/context_compressor.py — compression logic, thresholds, truncation fallback."""

import json
import sqlite3
import pytest
import time
from unittest.mock import patch, MagicMock

from agent.context_compressor import (
    ContextCompressor,
    HISTORICAL_TASK_HEADING,
    SUMMARY_PREFIX,
    COMPRESSED_SUMMARY_METADATA_KEY,
    _PRUNE_MIN_CHARS,
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
        # Resolve context_length while the mock is still active so the
        # fixture returns a fully-initialized compressor.
        _ = c.context_length
        return c


class TestSummarizeToolResultWebExtract:
    """Pre-compression pruning must survive web_extract calls whose ``urls`` are
    web_search result dicts ({"url"/"href": ...}), which models routinely forward
    straight into web_extract.
    """

    CONTENT = "x" * 500  # >200 chars so the pruning pass actually summarizes


    def test_single_dict_url_is_unwrapped_not_stringified(self):
        args = json.dumps({"urls": [{"url": "https://example.com/a", "title": "A"}]})
        summary = _summarize_tool_result("web_extract", args, self.CONTENT)
        assert summary == "[web_extract] https://example.com/a (500 chars)"
        assert "{" not in summary  # no raw dict repr leaked into the summary

    def test_href_key_is_unwrapped(self):
        args = json.dumps({"urls": [{"href": "https://example.com/h"}]})
        summary = _summarize_tool_result("web_extract", args, self.CONTENT)
        assert summary == "[web_extract] https://example.com/h (500 chars)"


class TestSummarizeToolResultClarify:
    def test_preserves_resolved_user_response_without_metadata(self):
        content = json.dumps({
            "question": "When should I deploy?",
            "choices_offered": ["Friday", "Monday"],
            "user_response": "Friday",
        })

        summary = _summarize_tool_result("clarify", "{}", content)

        assert summary == '[clarify] user responded: "Friday"'

    def test_preserves_multi_select_user_response(self):
        content = json.dumps({
            "question": "Which checks should I run?",
            "choices_offered": ["lint", "tests", "types"],
            "user_response": ["lint", "tests"],
        })

        summary = _summarize_tool_result("clarify", "{}", content)

        assert summary == '[clarify] user responded: ["lint", "tests"]'

    def test_long_response_is_bounded_and_prefixed_text_is_not_trusted(self):
        content = json.dumps({
            "question": "Describe the deployment constraints",
            "choices_offered": None,
            "user_response": "A" * 1_000,
        })

        summary = _summarize_tool_result("clarify", "{}", content)

        # Strictly below the prune floor so a later prune pass can never
        # re-summarize the preserved answer away (idempotency below).
        assert len(summary) == _PRUNE_MIN_CHARS - 1
        assert summary.startswith('[clarify] user responded: "AAA')
        assert summary.endswith("...[truncated]")
        assert (
            _summarize_tool_result("clarify", "{}", summary)
            == "[clarify] asked user a question"
        )

    def test_forged_response_prefix_does_not_expose_internal_content(self):
        forged = "[clarify] user responded: internal error: secret diagnostic"

        summary = _summarize_tool_result("clarify", "{}", forged)

        assert summary == "[clarify] asked user a question"
        assert "secret diagnostic" not in summary

    def test_prefixed_lone_surrogate_is_rejected_and_sqlite_safe(self):
        forged = "[clarify] user responded: " + "\ud83d" * 1_000

        summary = _summarize_tool_result("clarify", "{}", forged)

        assert summary == "[clarify] asked user a question"
        assert summary.encode("utf-8")
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE messages (content TEXT)")
            connection.execute("INSERT INTO messages VALUES (?)", (summary,))
            assert connection.execute("SELECT content FROM messages").fetchone()[0] == summary

    def test_unpaired_surrogates_are_safe_through_pruning_and_sqlite(self, compressor):
        content = json.dumps({"user_response": "Привет 😀" + "\ud83d" * 1_000})
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "clarify-1",
                        "type": "function",
                        "function": {"name": "clarify", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "clarify-1", "content": content},
            {"role": "user", "content": "recent request"},
            {"role": "assistant", "content": "recent response"},
        ]

        pruned_messages, pruned_count = compressor._prune_old_tool_results(
            messages, protect_tail_count=2
        )
        summary = pruned_messages[1]["content"]

        assert pruned_count == 1
        assert len(summary) <= _PRUNE_MIN_CHARS
        assert summary.encode("utf-8")
        assert "Привет 😀" in summary
        assert "\\ud83d" in summary
        with sqlite3.connect(":memory:") as connection:
            connection.execute("CREATE TABLE messages (content TEXT)")
            connection.execute("INSERT INTO messages VALUES (?)", (summary,))
            assert connection.execute("SELECT content FROM messages").fetchone()[0] == summary

        pruned_again, _ = compressor._prune_old_tool_results(
            pruned_messages, protect_tail_count=2
        )
        assert pruned_again[1]["content"] == summary

    @pytest.mark.parametrize(
        "content",
        [
            json.dumps({"error": "Failed to get user input: internal details"}),
            json.dumps({"question": "Q?", "user_response": ""}),
            json.dumps({"question": "Q?", "user_response": {"internal": "value"}}),
            "not json",
        ],
    )
    def test_does_not_expose_unresolved_or_internal_content(self, content):
        summary = _summarize_tool_result("clarify", "{}", content)

        assert summary == "[clarify] asked user a question"

    @pytest.mark.parametrize(
        "sentinel",
        [
            # cli.py clarify timeout callback
            "The user did not provide a response within the time limit. "
            "Use your best judgement to make the choice and proceed.",
            # gateway/run.py timeout + delivery-failure paths
            "[user did not respond within 15m]",
            "[clarify prompt could not be delivered]",
            # hermes_cli/oneshot.py no-user callback
            "[oneshot mode: no user available. Pick the best option from "
            "['a', 'b'] using your own judgment and continue.]",
        ],
    )
    def test_non_response_sentinels_are_not_attributed_to_user(self, sentinel):
        """Timeout/no-user sentinel prose must not be quoted as a user answer."""
        content = json.dumps({
            "question": "Deploy when?",
            "choices_offered": ["Friday", "Monday"],
            "user_response": sentinel,
        })

        summary = _summarize_tool_result("clarify", "{}", content)

        assert summary == "[clarify] asked user a question"

    def test_multi_select_containing_sentinel_stays_generic(self):
        content = json.dumps({
            "user_response": ["lint", "[user did not respond within 15m]"],
        })

        summary = _summarize_tool_result("clarify", "{}", content)

        assert summary == "[clarify] asked user a question"

    def test_live_oneshot_producer_is_recognized_as_sentinel(self):
        """Producer→recognizer drift guard: run the REAL oneshot no-user
        callback and assert its output is filtered. If the producer's wording
        drifts away from _CLARIFY_NON_RESPONSE_PREFIXES, this fails."""
        from hermes_cli.oneshot import _oneshot_clarify_callback

        sentinels = (
            _oneshot_clarify_callback("Deploy when?", choices=["a", "b"]),
            _oneshot_clarify_callback(
                "Deploy when?", choices=["a", "b"], multi_select=True
            ),
            _oneshot_clarify_callback("Deploy when?"),
        )
        for sentinel in sentinels:
            content = json.dumps({"user_response": sentinel})

            summary = _summarize_tool_result("clarify", "{}", content)

            assert summary == "[clarify] asked user a question", sentinel


class TestShouldCompress:
    def test_below_threshold(self, compressor):
        compressor.last_prompt_tokens = 50000
        assert compressor.should_compress() is False

    def test_above_threshold(self, compressor):
        compressor.last_prompt_tokens = 90000
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

    def test_pairs_noted_rough_estimate_with_fitting_real_usage(self, compressor):
        """note_request_rough_estimate() + a fitting response must anchor the
        defer baseline even when no compression ever ran — this is what gives
        fresh sessions a (rough, real) pair before their first compaction."""
        compressor.note_request_rough_estimate(120_000)
        compressor.update_from_response({"prompt_tokens": 60_000})

        assert compressor.last_real_prompt_tokens == 60_000
        assert compressor.last_rough_tokens_when_real_prompt_fit == 120_000
        # Consumed: a later usage-bearing response without a fresh note keeps
        # the previous pair instead of re-pairing against a stale estimate.
        assert compressor._pending_request_rough_tokens == 0

    def test_post_compression_pairing_wins_over_noted_estimate(self, compressor):
        """Right after a compaction the post-compression rough count is the
        authoritative baseline (#36718); a stale pre-compression note must not
        displace it."""
        compressor.note_request_rough_estimate(120_000)
        compressor.awaiting_real_usage_after_compression = True
        compressor.last_compression_rough_tokens = 40_000
        compressor.update_from_response({"prompt_tokens": 30_000})

        assert compressor.last_rough_tokens_when_real_prompt_fit == 40_000

    def test_usage_less_response_preserves_pending_note(self, compressor):
        """Transports that report usage separately send usage-less responses
        first; the pending pair must survive until real usage arrives."""
        compressor.note_request_rough_estimate(120_000)
        compressor.update_from_response({})

        assert compressor._pending_request_rough_tokens == 120_000

    def test_over_threshold_real_usage_clears_pending_note(self, compressor):
        compressor.note_request_rough_estimate(120_000)
        compressor.update_from_response({"prompt_tokens": 90_000})

        assert compressor.last_rough_tokens_when_real_prompt_fit == 0
        assert compressor._pending_request_rough_tokens == 0

class TestPreflightDeferral:

    def test_defers_while_projected_real_usage_fits(self, compressor):
        """Large rough growth alone must not trigger compaction: with real
        usage at 50K and 10K of rough growth since that reading, projected
        real usage is 60K — far under the 85K threshold. The old fixed 5%
        growth tolerance compacted here at ~59% of the real window (CJK /
        replay-blob overcount churn)."""
        compressor.threshold_tokens = 85_000
        compressor.last_real_prompt_tokens = 50_000
        compressor.last_rough_tokens_when_real_prompt_fit = 90_000

        assert compressor.should_defer_preflight_to_real_usage(100_000) is True

    def test_does_not_defer_when_projected_real_usage_crosses_threshold(self, compressor):
        """Projection = last real + rough growth. 80K real + 6K growth = 86K
        >= 85K threshold: compression must run."""
        compressor.threshold_tokens = 85_000
        compressor.last_real_prompt_tokens = 80_000
        compressor.last_rough_tokens_when_real_prompt_fit = 90_000

        assert compressor.should_defer_preflight_to_real_usage(96_000) is False

    def test_does_not_defer_without_a_baseline(self, compressor):
        """No synchronized (rough, real) pair yet — fall back to trusting the
        rough estimate (conservative: compress)."""
        compressor.threshold_tokens = 85_000
        compressor.last_real_prompt_tokens = 50_000
        compressor.last_rough_tokens_when_real_prompt_fit = 0
        compressor.last_compression_rough_tokens = 0

        assert compressor.should_defer_preflight_to_real_usage(100_000) is False

    def test_defer_does_not_ratchet_baseline(self, compressor):
        """The baseline is refreshed only by update_from_response() pairing.
        Deferring must not advance it: without a fresh real reading, a
        ratcheted baseline would shrink apparent growth and defer on stale
        data."""
        compressor.threshold_tokens = 85_000
        compressor.last_real_prompt_tokens = 50_000
        compressor.last_rough_tokens_when_real_prompt_fit = 90_000

        assert compressor.should_defer_preflight_to_real_usage(100_000) is True
        assert compressor.last_rough_tokens_when_real_prompt_fit == 90_000


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




class TestCompress:
    def _make_messages(self, n):
        return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(n)]




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
        with patch("agent.context_compressor.call_llm", side_effect=RuntimeError("no provider")):
            result = c.compress(msgs)
        assert len(result) < len(msgs)


class TestNonStringContent:
    """Regression: content as dict (e.g., llama.cpp tool calls) must not crash."""


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


    def test_missing_provider_api_key_is_terminal_access_failure(self):
        err = RuntimeError(
            "Provider 'opencode-zen' is set in config.yaml but no API key was "
            "found. Set the OPENCODE-ZEN_API_KEY environment variable."
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


    def test_403_also_flags_auth_failure(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)
        with patch("agent.context_compressor.call_llm", side_effect=self._auth_err(403)):
            c._generate_summary(self._msgs())
        assert c._last_summary_auth_failure is True



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

    def test_generate_summary_flags_empty_content_failure(self):
        """An empty-content response on the summary call flags
        _last_summary_empty_content_failure (#94448)."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(model="test", quiet_mode=True)
        with patch(
            "agent.context_compressor.call_llm",
            return_value={"choices": [{"message": {"content": "   "}}]},
        ):
            result = c._generate_summary(self._msgs())
        assert result is None
        assert c._last_summary_empty_content_failure is True
        assert c._last_summary_auth_failure is False
        assert c._last_summary_network_failure is False

    def test_empty_content_summary_aborts_compression_and_preserves_messages(self):
        """Empty-content response from degraded provider aborts compression and
        preserves original messages without dropping context (#94448)."""
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
            return_value={"choices": [{"message": {"content": ""}}]},
        ):
            result = c.compress(msgs, current_tokens=999999, force=True)

        assert result == msgs
        assert c._last_summary_empty_content_failure is True
        assert c._last_compress_aborted is True
        assert c._last_summary_fallback_used is False
        assert c._last_summary_dropped_count == 0

        # Cooldown re-entry must keep aborting, same as network/auth —
        # _generate_summary() returns None from the cooldown early-return
        # without re-asserting the flag, so compress() must still see it.
        second = c.compress(msgs, current_tokens=999999)
        assert second == msgs
        assert c._last_compress_aborted is True
        assert c._last_summary_fallback_used is False

    def test_auxiliary_none_response_aborts_compression(self):
        """Sibling shape (#94459, from #7264): the auxiliary boundary's own
        terminal "None response" error is the same degraded-provider class
        and must ABORT, not fall through to the destructive fallback."""
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
            side_effect=RuntimeError("Auxiliary compression: LLM returned None response"),
        ):
            result = c.compress(msgs, current_tokens=999999, force=True)
        assert result == msgs
        assert c._last_compress_aborted is True
        assert c._last_summary_empty_content_failure is True

    def test_auxiliary_invalid_response_aborts_compression(self):
        """Sibling shape (#94459, from #7264): malformed/missing
        choices[0].message terminal error must ABORT the same way."""
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
            side_effect=RuntimeError(
                "Auxiliary compression: LLM returned invalid response "
                "(type=str): 'oops'. Expected object with .choices[0].message "
                "— check provider adapter or custom endpoint compatibility."
            ),
        ):
            result = c.compress(msgs, current_tokens=999999, force=True)
        assert result == msgs
        assert c._last_compress_aborted is True
        assert c._last_summary_empty_content_failure is True


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

    def test_empty_content_falls_back_to_main_and_succeeds(self):
        """Aux model returns empty content -> falls back to main model -> succeeds (#94448)."""
        mock_ok = MagicMock()
        mock_ok.choices = [MagicMock()]
        mock_ok.choices[0].message.content = "summary via main model after empty aux"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="main-model",
                summary_model_override="flaky-aux-model",
                quiet_mode=True,
            )

        with patch(
            "agent.context_compressor.call_llm",
            side_effect=[
                {"choices": [{"message": {"content": "   "}}]},
                mock_ok,
            ],
        ) as mock_call:
            result = c._generate_summary(self._msgs())

        assert mock_call.call_count == 2
        assert mock_call.call_args_list[0].kwargs.get("model") == "flaky-aux-model"
        assert "model" not in mock_call.call_args_list[1].kwargs
        assert result is not None
        assert "summary via main model after empty aux" in result
        assert c._last_aux_model_failure_model == "flaky-aux-model"
        assert "empty content" in (c._last_aux_model_failure_error or "").lower()

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



class TestSummaryPrefixNormalization:
    def test_legacy_prefix_is_replaced(self):
        summary = ContextCompressor._with_summary_prefix("[CONTEXT SUMMARY]: did work")
        assert summary == f"{SUMMARY_PREFIX}\ndid work"

    def test_existing_new_prefix_is_not_duplicated(self):
        summary = ContextCompressor._with_summary_prefix(f"{SUMMARY_PREFIX}\ndid work")
        assert summary == f"{SUMMARY_PREFIX}\ndid work"


class TestCompressWithClient:



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





class TestSummaryTargetRatio:
    """Verify that summary_target_ratio properly scales budgets with context window."""




    def test_default_threshold_floored_at_75_percent_below_512k(self):
        """Sub-512K models get the 75% small-context threshold floor."""
        with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
            c = ContextCompressor(model="test", quiet_mode=True)
            _ = c.context_length
        assert c.threshold_percent == 0.75
        # 75% of 100K = 75K, above the 64K minimum floor
        assert c.threshold_tokens == 75_000





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
        with patch("agent.context_compressor.call_llm", side_effect=RuntimeError("no provider")):
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





class TestThresholdTokensCap:
    """Tests for the absolute token cap (compression.threshold_tokens).

    The cap takes the lower of the ratio-based threshold and the absolute
    count. It must survive model switches (update_model re-applies it)
    and be clamped to the model's context length.
    """



    def test_no_cap_uses_ratio_only(self):
        """Without a cap, the ratio-based threshold is used."""
        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            comp = ContextCompressor(
                "model-a", threshold_percent=0.50, quiet_mode=True,
            )
            _ = comp.context_length
        assert comp.threshold_tokens == 500_000
        assert comp.threshold_tokens_cap is None






    def test_invalid_cap_treated_as_none(self):
        """Non-numeric, zero, or negative cap values are treated as None."""
        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            comp0 = ContextCompressor(
                "model-a", threshold_percent=0.50, quiet_mode=True,
                threshold_tokens_cap=0,
            )
            assert comp0.threshold_tokens_cap is None
            assert comp0.threshold_tokens == 500_000

            comp_neg = ContextCompressor(
                "model-a", threshold_percent=0.50, quiet_mode=True,
                threshold_tokens_cap=-100,
            )
            assert comp_neg.threshold_tokens_cap is None

            comp_str = ContextCompressor(
                "model-a", threshold_percent=0.50, quiet_mode=True,
                threshold_tokens_cap="not-a-number",
            )
            assert comp_str.threshold_tokens_cap is None

    def test_should_compress_fires_at_cap_below_ratio_threshold(self):
        """Behavioral: with a cap below the ratio-based threshold,
        should_compress() fires once usage crosses the cap — even though
        the percentage threshold has not been reached (first-fires-wins)."""
        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            comp = ContextCompressor(
                "model-a", threshold_percent=0.50, quiet_mode=True,
                threshold_tokens_cap=200_000,
            )
        # Ratio-based would be 500K; cap pulls the trigger down to 200K.
        assert comp.should_compress(150_000) is False   # below cap
        assert comp.should_compress(200_000) is True    # at cap (below 500K pct)
        assert comp.should_compress(250_000) is True    # above cap

    def test_default_config_disabled_and_no_behavior_change(self):
        """DEFAULT_CONFIG ships threshold_tokens=None (disabled) and both
        None and 0 leave the ratio-based trigger byte-identical."""
        from hermes_cli.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG["compression"]["threshold_tokens"] is None

        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            baseline = ContextCompressor(
                "model-a", threshold_percent=0.50, quiet_mode=True,
            )
            comp_none = ContextCompressor(
                "model-a", threshold_percent=0.50, quiet_mode=True,
                threshold_tokens_cap=None,
            )
            comp_zero = ContextCompressor(
                "model-a", threshold_percent=0.50, quiet_mode=True,
                threshold_tokens_cap=0,
            )
        assert comp_none.threshold_tokens == baseline.threshold_tokens
        assert comp_zero.threshold_tokens == baseline.threshold_tokens
        # And after a model switch, still identical to baseline.
        baseline.update_model("model-b", context_length=200_000)
        comp_none.update_model("model-b", context_length=200_000)
        comp_zero.update_model("model-b", context_length=200_000)
        assert comp_none.threshold_tokens == baseline.threshold_tokens
        assert comp_zero.threshold_tokens == baseline.threshold_tokens



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


class TestLazyContextResolution:
    """Verify that ContextCompressor defers get_model_context_length until
    context_length is first accessed, so construction never blocks on network
    I/O or blocks startup when the model metadata service is slow."""


    def test_init_does_not_probe_when_not_quiet(self, caplog):
        """quiet_mode=False must ALSO stay non-blocking in __init__.

        Regression for the lazy-init defect: the "Context compressor
        initialized" log reads context_length/threshold_tokens/tail_token_budget,
        so emitting it in __init__ forced the deferred get_model_context_length()
        probe to run during construction whenever quiet_mode was False (the
        interactive CLI path) — silently re-introducing the #32221 blocking that
        the original PR set out to remove. The informative line must instead be
        emitted once, on first context-length resolution.
        """
        import logging

        with patch(
            "agent.context_compressor.get_model_context_length",
            return_value=200_000,
        ) as mock_get:
            c = ContextCompressor(model="test/model", quiet_mode=False)
            # No probe, and no init log, at construction time.
            mock_get.assert_not_called()

            with caplog.at_level(logging.INFO, logger="agent.context_compressor"):
                _ = c.context_length
            mock_get.assert_called_once()
            init_lines = [
                r for r in caplog.records
                if "Context compressor initialized" in r.getMessage()
            ]
            assert len(init_lines) == 1, (
                f"expected exactly one init log on first access, got {len(init_lines)}"
            )

            # Subsequent access must not re-probe or re-log.
            with caplog.at_level(logging.INFO, logger="agent.context_compressor"):
                _ = c.context_length
                _ = c.threshold_tokens
            mock_get.assert_called_once()
            again = [
                r for r in caplog.records
                if "Context compressor initialized" in r.getMessage()
            ]
            assert len(again) == 1, "init log fired more than once"


    def test_config_context_length_skips_network_probe(self):
        """When config_context_length is provided, the resolver must use it
        as the cached value and not make a network call."""
        with patch(
            "agent.context_compressor.get_model_context_length",
            side_effect=lambda model, **kwargs: kwargs.get("config_context_length"),
        ) as mock_get:
            c = ContextCompressor(
                model="test/model",
                quiet_mode=True,
                config_context_length=200_000,
            )
            result = c.context_length
            assert result == 200_000


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





    # ------------------------------------------------------------------
    # _ensure_last_user_message_in_tail unit tests
    # ------------------------------------------------------------------


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
        """When a MID-LIST assistant has both valid and orphaned tool_calls,
        only the orphans are stripped.  #51218

        The shape must sit mid-list: the same shape at the TAIL is
        indistinguishable from a partial multi-call batch whose remaining
        results are still in flight, and the sanitizer now presumes in-flight
        there (#79278) — preserving is safe because the pre-API chokepoint
        injects stub results for genuinely unanswered calls, while stripping
        a live call silently loses its late result.
        """
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
            # Later turn: the chain above is settled history, not in flight.
            {"role": "assistant", "content": "done"},
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

    def test_sanitizer_keeps_valid_pair_matching_on_id_not_call_id(self, compressor):
        """A genuinely matching Codex-format pair must survive when the
        result's tool_call_id matches ``id`` rather than ``call_id`` (#58168
        class). ``_get_tool_call_id``'s ``call_id || id`` precedence picks
        ``call_id`` first, so building the known-id set from a single value
        per tool_call misclassified this valid pair as orphaned on BOTH
        sides — dropping the result AND stripping the tool_call, even though
        neither was actually orphaned."""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "fc_777",
                        "call_id": "call_777",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "fc_777", "content": "result"},
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        asst = next(m for m in sanitized if m.get("role") == "assistant")
        assert asst.get("tool_calls"), "valid tool_call must not be stripped"
        assert asst["tool_calls"][0]["id"] == "fc_777"
        tool_msgs = [m for m in sanitized if m.get("role") == "tool"]
        assert len(tool_msgs) == 1, "valid tool result must not be dropped as orphaned"
        assert tool_msgs[0]["tool_call_id"] == "fc_777"

    def test_sanitizer_still_drops_genuine_orphan_with_dual_ids(self, compressor):
        """Negative control: registering both id and call_id must not
        over-relax orphan detection. A genuinely orphaned tool_call (no
        result matching either id variant) is still stripped, while a valid
        dual-id pair in the same window survives. A trailing user turn keeps
        the assistant message out of the in-flight protection window (#79278),
        which intentionally preserves a still-pending trailing call."""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "fc_1",
                        "call_id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                    {
                        "id": "fc_2",
                        "call_id": "call_2",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "user", "content": "next question"},
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        asst = next(m for m in sanitized if m.get("role") == "assistant")
        surviving_ids = {tc["id"] for tc in asst.get("tool_calls") or []}
        assert surviving_ids == {"fc_1"}


class TestSanitizerPreservesInFlightToolChain:
    """Issue #79278: an in-flight tool call chain must survive compression.

    When compression fires mid-chain — after the model emitted
    ``assistant(tool_calls)`` but before tool_executor.py appended the matching
    ``role="tool"`` result — the final message is a *pending* tool call, not an
    orphan.  Stripping it (or replacing content with ``(tool call removed)``)
    destroys the chain: when the executor later appends the real result,
    repair_message_sequence drops it as an unmatched orphan and the completed
    side effect and final synthesis are lost.  _sanitize_tool_pairs must exempt
    the trailing in-flight call.
    """

    def test_trailing_inflight_tool_call_preserved(self, compressor):
        """A trailing assistant tool_call with no result yet is pending, not
        orphaned — preserve it verbatim.  #79278"""
        msgs = [
            {"role": "user", "content": "summarize"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "summarize", "arguments": "{}"}},
                ],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_2", "function": {"name": "finalize", "arguments": "{}"}},
                ],
            },
            # in-flight: call_2 has been emitted but its tool result is not
            # yet appended by tool_executor.py
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        # The trailing in-flight call must survive intact.
        assert sanitized[-1]["role"] == "assistant"
        assert len(sanitized[-1]["tool_calls"]) == 1
        assert sanitized[-1]["tool_calls"][0]["id"] == "call_2"
        assert sanitized[-1]["content"] != "(tool call removed)"

    def test_inflight_result_arrives_after_compress(self, compressor):
        """After compress() preserves the pending call, appending its tool
        result leaves a well-formed chain — the side effect's result reaches
        the model.  #79278"""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_2", "function": {"name": "finalize", "arguments": "{}"}},
                ],
            },
        ]
        sanitized = compressor._sanitize_tool_pairs(msgs)
        assert sanitized[-1]["tool_calls"][0]["id"] == "call_2"

        # The executor appends the result after compress() returns.
        sanitized.append(
            {"role": "tool", "tool_call_id": "call_2", "content": "side effect done"}
        )

        # Chain now settled: the assistant call is matched, result survives.
        surviving = {tc["id"] for m in sanitized if m["role"] == "assistant"
                     for tc in (m.get("tool_calls") or [])}
        assert "call_2" in surviving
        results = [m for m in sanitized if m["role"] == "tool"]
        assert len(results) == 1 and results[0]["tool_call_id"] == "call_2"

    def test_inflight_preserved_while_true_orphan_still_stripped(self, compressor):
        """Preserving the trailing in-flight call must not weaken the existing
        orphan-stripping behavior for genuinely orphaned calls in the discarded
        region.  #79278"""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_orphan", "function": {"name": "search", "arguments": "{}"}},
                ],
            },
            {"role": "user", "content": "interim"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_pending", "function": {"name": "write", "arguments": "{}"}},
                ],
            },
            # trailing in-flight call_pending
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        middle = [m for m in sanitized if m["role"] == "assistant" and m.get("tool_calls")]
        # Only the trailing in-flight call survives.
        assert len(middle) == 1
        assert middle[0]["tool_calls"][0]["id"] == "call_pending"
        # The genuine orphan at the head was still stripped.
        assert not any(m.get("tool_calls") and m["tool_calls"][0]["id"] == "call_orphan"
                       for m in sanitized)

    def test_side_effect_final_result_returned_end_to_end(self, compressor):
        """Issue #79278 end-to-end: an in-flight tool chain that triggers
        compression must still deliver the completed side effect's result.

        Reproduces the full executor flow:
          1. the model emits ``assistant(tool_calls=call_2)`` while call_2 is
             still IN-FLIGHT — tool_executor.py has not yet appended its
             ``role="tool"`` result;
          2. context compression fires and ``_sanitize_tool_pairs()`` runs on
             a history whose tail is that pending call;
          3. the side effect completes and the executor appends the ``tool``
             result for call_2;
          4. the next pre-call pass, ``repair_message_sequence()``, runs and
             must NOT drop the completed result as an unmatched orphan.

        On the old code, compression stripped the pending call_2 as an
        "orphan"; when the result then arrived, repair_message_sequence
        dropped it as unmatched — so the completed side effect's result and
        the final synthesis built on it were both lost.  The fix preserves the
        trailing in-flight call, so the chain closes and the result survives.
        """
        from agent.agent_runtime_helpers import repair_message_sequence

        history = [
            {"role": "user", "content": "Do the work and use the tools."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "run_side_effect", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1",
             "content": "side effect completed"},
            # in-flight: call_2 emitted, its result not yet appended
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_2", "type": "function",
                 "function": {"name": "gather_final_result", "arguments": "{}"}}]},
        ]

        # Step 1: compression fires mid-chain.
        compressed = compressor._sanitize_tool_pairs([dict(m) for m in history])

        # The in-flight call must survive the compression pass (fix), not be
        # stripped as an orphan (bug) — otherwise the chain is already broken.
        assert compressed[-1]["role"] == "assistant"
        assert [tc["id"] for tc in compressed[-1].get("tool_calls", [])] == ["call_2"]
        assert compressed[-1].get("content") != "(tool call removed)"

        # Step 2: the side effect completes; the executor appends its result.
        messages = [dict(m) for m in compressed] + [
            {"role": "tool", "tool_call_id": "call_2",
             "content": "final computed result: 42"}
        ]

        # Step 3: the next pre-call sanitizer runs on the settled chain.
        repair_message_sequence(None, messages)

        # Step 4: the final result must still be returned — the assistant call
        # and its completed tool result both survive, still paired.
        assistant_ids = {
            tc["id"]
            for m in messages if m["role"] == "assistant"
            for tc in (m.get("tool_calls") or [])
        }
        assert "call_2" in assistant_ids

        results = [m for m in messages
                   if m["role"] == "tool" and m.get("tool_call_id") == "call_2"]
        assert len(results) == 1
        assert "42" in results[0]["content"]

    def test_partial_batch_inflight_calls_preserved(self, compressor):
        """Multi-call batch snapshotted BETWEEN result appends: the executor
        has appended tool(c1) but not yet tool(c2)/tool(c3), so the last
        message is a tool result while c2/c3 are still pending.  The walk-back
        must find the assistant behind the trailing results and preserve the
        whole batch — stripping c2/c3 there loses their late results exactly
        like the tail-is-assistant shape.  #79278 follow-up."""
        msgs = [
            {"role": "user", "content": "run the batch"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "function": {"name": "a", "arguments": "{}"}},
                {"id": "c2", "function": {"name": "b", "arguments": "{}"}},
                {"id": "c3", "function": {"name": "c", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "done 1"},
            # snapshot taken here: c2/c3 results not yet appended
        ]

        sanitized = compressor._sanitize_tool_pairs(msgs)

        batch = [m for m in sanitized if m.get("role") == "assistant"][-1]
        assert [tc["id"] for tc in batch["tool_calls"]] == ["c1", "c2", "c3"]
        # And the already-arrived result survives too.
        assert any(
            m.get("role") == "tool" and m.get("tool_call_id") == "c1"
            for m in sanitized
        )


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

    def test_restart_handoff_without_system_still_starts_with_user(self):
        """When decayed head protection leaves no head, the visible transcript
        must still begin with a user role for Anthropic/Bedrock compatibility.
        """
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "summary of resumed turns"

        with patch("agent.context_compressor.get_model_context_length", return_value=100000):
            c = ContextCompressor(
                model="test", quiet_mode=True, protect_first_n=3, protect_last_n=2,
            )

        msgs = [
            {"role": "user", "content": f"{SUMMARY_PREFIX}\nold persisted summary"},
            {"role": "assistant", "content": "handoff acknowledged"},
            {"role": "user", "content": "new work after restart"},
            {"role": "assistant", "content": "new answer after restart"},
            {"role": "user", "content": "more new work after restart"},
            {"role": "assistant", "content": "more new answer after restart"},
            {"role": "user", "content": "tail request"},
            {"role": "assistant", "content": "tail answer"},
        ]
        with patch("agent.context_compressor.call_llm", return_value=mock_response):
            result = c.compress(msgs)

        assert result[0]["role"] == "user"
        assert "summary of resumed turns" in (result[0].get("content") or "")

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


class TestSummaryPromptBounding:




    def test_iterative_update_path_is_bounded(self):
        """The iterative prompt (previous summary + new turns) must be bounded
        too — a pathological rehydrated handoff must not blow up the prompt."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "updated summary"

        with patch("agent.context_compressor.get_model_context_length", return_value=272000):
            c = ContextCompressor(model="test", quiet_mode=True)
        cap = c._SUMMARY_INPUT_MAX_CHARS
        c._previous_summary = "PREV_HEAD " + ("p" * (cap * 2)) + " PREV_TAIL"

        messages = [
            {"role": "user", "content": f"turn-{i}-" + ("x" * 6000)}
            for i in range(80)
        ]

        with patch("agent.context_compressor.call_llm", return_value=mock_response) as mock_call:
            summary = c._generate_summary(messages)

        prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        assert summary.startswith(SUMMARY_PREFIX)
        # previous summary block + new-turns block each capped, plus the
        # fixed template: well under 3x the cap (unbounded would be ~800K).
        assert len(prompt) < 2 * cap + 30_000
        assert "PREV_HEAD" in prompt
        assert "PREV_TAIL" in prompt
        assert "summary input truncated" in prompt

    def test_marker_does_not_collide_with_summary_classifier(self):
        """The omitted-middle marker must never make bounded content classify
        as a compaction handoff (SUMMARY_PREFIX / merged-handoff patterns)."""
        cap = ContextCompressor._SUMMARY_INPUT_MAX_CHARS
        bounded = ContextCompressor._bound_summary_input("z" * (cap * 2))
        assert "summary input truncated" in bounded
        assert ContextCompressor.classify_summary_content(bounded) is None
        # Marker alone (worst case: lands at the start of a message) is not a
        # handoff prefix either.
        marker_only = bounded[bounded.index("\n\n...[summary input truncated"):]
        assert ContextCompressor.classify_summary_content(marker_only.lstrip()) is None


class TestMinTailUserMessages:
    """COMPRESS-01,02,07,08: N-user-message tail protection.

    Tests the ``_ensure_last_n_user_messages_in_tail`` method and its
    integration through ``_find_tail_cut_by_tokens``.
    """


    def test_n3_tool_group_integrity(self):
        """COMPRESS-02: When the 3rd-to-last user message is preceded by
        assistant(tool_calls) + tool results, the boundary is set at the
        user message (a clean boundary).  The preceding tool group stays
        together — it is either entirely in the compressed region or
        entirely in the tail, never split across the boundary."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c = ContextCompressor(
                model="test/model",
                threshold_percent=0.50,
                protect_first_n=1,
                quiet_mode=True,
                min_tail_user_messages=3,
            )
        c.tail_token_budget = 200
        messages = [
            {"role": "user", "content": "start"},
            {"role": "assistant", "content": None,
             "tool_calls": [{"function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "content": "result content",
             "tool_call_id": "call_1"},
            {"role": "user", "content": "user 3rd last"},
            {"role": "assistant", "content": "reply 3rd last"},
            {"role": "user", "content": "user 2nd last"},
            {"role": "assistant", "content": "reply 2nd last"},
            {"role": "user", "content": "user last"},
            {"role": "assistant", "content": "reply last"},
        ]
        head_end = c.protect_first_n
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        compressed = messages[:cut]
        tool_positions = [
            i for i, m in enumerate(compressed)
            if m.get("role") in ("assistant", "tool")
            and (m.get("tool_calls") or m.get("tool_call_id"))
        ]
        if len(tool_positions) >= 2:
            assert tool_positions[-1] - tool_positions[0] == len(tool_positions) - 1, (
                "Tool group must stay contiguous across the boundary"
            )
        tail_messages = messages[cut:]
        tail_users = [m["content"] for m in tail_messages if m["role"] == "user"]
        assert "user 3rd last" in tail_users
        assert "user 2nd last" in tail_users
        assert "user last" in tail_users
        assert cut >= head_end + 1






    def test_default_is_behavior_preserving(self):
        """Default min_tail_user_messages=1 leaves the tail cut byte-identical
        to the historical single-anchor pipeline.

        A default of 3 was measured to CHANGE the cut on transcripts whose
        tail budget covers only the last turn, so the default is gated to 1
        (= the existing single last-user anchor) and N>1 is opt-in.
        """
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c_default = ContextCompressor(
                model="test/model",
                threshold_percent=0.50,
                protect_first_n=2,
                quiet_mode=True,
            )
            c_explicit = ContextCompressor(
                model="test/model",
                threshold_percent=0.50,
                protect_first_n=2,
                quiet_mode=True,
                min_tail_user_messages=1,
            )
        assert c_default.min_tail_user_messages == 1
        c_default.tail_token_budget = 200
        c_explicit.tail_token_budget = 200
        messages = [
            {"role": "user", "content": "head msg"},
            {"role": "assistant", "content": "head reply"},
        ]
        for i in range(3):
            messages.append({"role": "user", "content": f"user {i}"})
            messages.append({"role": "assistant", "content": "X" * 4000})
        messages.append({"role": "user", "content": "final user"})
        messages.append({"role": "assistant", "content": "final reply"})
        head_end = c_default.protect_first_n
        assert (
            c_default._find_tail_cut_by_tokens(messages, head_end)
            == c_explicit._find_tail_cut_by_tokens(messages, head_end)
        )

    def test_blank_echo_does_not_count_toward_n(self):
        """A blank platform echo (empty user row) must not consume one of the
        N slots — otherwise the guarantee silently degrades to N-1 real turns
        (the #69291 bug class the single anchor already fixed)."""
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c = ContextCompressor(
                model="test/model",
                threshold_percent=0.50,
                protect_first_n=1,
                quiet_mode=True,
                min_tail_user_messages=3,
            )
        messages = [
            {"role": "user", "content": "head"},                 # 0 (head)
            {"role": "assistant", "content": "head reply"},      # 1
            {"role": "user", "content": "real oldest"},          # 2  <- 3rd real user
            {"role": "assistant", "content": "reply oldest"},    # 3
            {"role": "user", "content": "real middle"},          # 4
            {"role": "assistant", "content": "reply middle"},    # 5
            {"role": "user", "content": ""},                     # 6  blank echo
            {"role": "assistant", "content": "reply to echo"},   # 7
            {"role": "user", "content": "   "},                  # 8  whitespace echo
            {"role": "assistant", "content": "another reply"},   # 9
            {"role": "user", "content": "real latest"},          # 10
            {"role": "assistant", "content": "final reply"},     # 11
        ]
        head_end = c.protect_first_n  # = 1
        # cut_idx=10 → only "real latest" in tail; N=3 must walk back to
        # index 2 ("real oldest"), NOT stop at a blank echo (6/8).
        result = c._ensure_last_n_user_messages_in_tail(
            messages, cut_idx=10, head_end=head_end, n=3
        )
        assert result == 2, (
            f"3rd real user is at index 2, got cut {result} — blank echoes "
            "must not count toward N"
        )
        tail_users = [
            m["content"] for m in messages[result:]
            if m["role"] == "user" and m["content"].strip()
        ]
        assert {"real oldest", "real middle", "real latest"} <= set(tail_users)



    def test_n_guarantee_wins_over_tail_token_budget_and_floor(self):
        """Interaction contract: the N-user guarantee WINS over both
        tail_token_budget and _MAX_TAIL_MESSAGE_FLOOR.

        The budget walk (and its bounded message floor) computes the initial
        cut; the N-anchor then only ever pulls the cut BACKWARD (tail can
        grow, never shrink), exactly like the existing single-user and
        assistant anchors. So a tiny budget cannot roll real users 2..N into
        the summary, and the floor remains a lower bound, not a cap, on the
        anchored tail.
        """
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c = ContextCompressor(
                model="test/model",
                threshold_percent=0.50,
                protect_first_n=1,
                quiet_mode=True,
                min_tail_user_messages=3,
            )
        # Budget covers roughly one bulky turn — without the N-anchor the
        # cut lands after users 2..3.
        c.tail_token_budget = 150
        messages = [{"role": "user", "content": "head"},
                    {"role": "assistant", "content": "head reply"}]
        for i in (3, 2, 1):
            messages.append({"role": "user", "content": f"real {i}"})
            messages.append({"role": "assistant", "content": "B" * 6000})
        head_end = c.protect_first_n
        cut = c._find_tail_cut_by_tokens(messages, head_end)
        tail = messages[cut:]
        tail_users = [m["content"] for m in tail if m["role"] == "user"]
        assert {"real 1", "real 2", "real 3"} <= set(tail_users), (
            f"N-guarantee must win over the token budget; tail users: {tail_users}"
        )
        # The anchored tail legitimately exceeds the budget (and the 8-message
        # floor is a minimum, not a cap): the guarantee is the stronger
        # invariant by design.
        from agent.context_compressor import _estimate_msg_budget_tokens
        accumulated = sum(_estimate_msg_budget_tokens(m) for m in tail)
        assert accumulated > c.tail_token_budget



class TestContextLengthSetterCoherence:
    """The context_length setter must (a) not wipe runtime corrections on
    no-op re-assignment of the same window (codex app-server usage callback
    re-reports it every response), and (b) re-apply the small-context
    threshold floor for a genuinely new window so percent and tokens derive
    from the same window."""

    def test_same_value_reassignment_preserves_threshold_override(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
            c = ContextCompressor(model="test", quiet_mode=True)
            _ = c.context_length
        # Runtime correction (aux-context threshold sync pattern).
        c.threshold_tokens = 42_000
        c.tail_token_budget = 8_400
        # Codex usage callback re-reports the same window every response.
        c.context_length = 200_000
        assert c.threshold_tokens == 42_000
        assert c.tail_token_budget == 8_400

    def test_new_value_assignment_refloors_and_invalidates(self):
        with patch("agent.context_compressor.get_model_context_length", return_value=1_000_000):
            c = ContextCompressor(model="test", quiet_mode=True)
            _ = c.context_length
        assert c.threshold_percent == 0.50  # 1M >= 512K: configured value
        # Switch to a small window via direct assignment (codex path).
        c.context_length = 200_000
        # Floor re-applied for the new window...
        assert c.threshold_percent == 0.75
        # ...and budgets recompute from the same window+percent.
        assert c.threshold_tokens == 150_000



class TestPreLlmFeasibilityCheck:
    """Tests for the pre-LLM feasibility skip in compress().

    When the middle section is < 10% of threshold tokens AND at least one
    prior real-usage ineffectiveness strike has been recorded, the expensive
    LLM summarization call is skipped and the deterministic fallback is used
    instead. The skip must NOT increment _ineffective_compression_count (the
    strike counter that latches at >=2) and must NOT trip the abort branch
    via stale _last_summary_auth_failure / _last_summary_network_failure flags.
    """

    def _make_messages(self, n_pairs=10, content="short reply"):
        """Build a message list large enough to survive compress()'s early exits.

        compress() bails if n_messages <= protect_head_size + 4, and again
        if compress_start >= compress_end (tail budget covers everything).
        With protect_first_n=2 → head=2, protect_last_n=2, we need enough
        middle turns to produce compress_start < compress_end.  10 pairs
        (21 messages including system) is comfortably past both gates.
        """
        msgs = [{"role": "system", "content": "system prompt"}]
        for i in range(n_pairs):
            msgs.append({"role": "user", "content": f"question {i}"})
            msgs.append({"role": "assistant", "content": content})
        return msgs

    def test_skip_does_not_increment_strike_counter(self, compressor):
        """Feasibility skip must use _prellm_skip_count, not _ineffective_compression_count."""
        compressor._ineffective_compression_count = 1  # one prior real strike
        msgs = self._make_messages()

        with patch.object(compressor, "_generate_summary") as mock_gen:
            # Middle section is tiny → feasibility skip fires
            compressor.compress(msgs, force=False)

        # The strike counter must NOT have been incremented by the skip
        assert compressor._ineffective_compression_count == 1
        # The pre-LLM skip counter should have been incremented
        assert compressor._prellm_skip_count >= 1
        # _generate_summary must NOT have been called
        mock_gen.assert_not_called()

    def test_skip_does_not_trip_abort_on_stale_auth_failure(self, compressor):
        """A feasibility skip must not trip the abort branch even if
        _last_summary_auth_failure is stale True from a prior cycle."""
        compressor._ineffective_compression_count = 1
        compressor._last_summary_auth_failure = True  # stale from prior failure
        msgs = self._make_messages()

        with patch.object(compressor, "_generate_summary") as mock_gen:
            result = compressor.compress(msgs, force=False)

        # Should NOT abort — should produce compressed output with fallback
        assert result is not None
        assert len(result) > 0
        mock_gen.assert_not_called()

    def test_skip_does_not_trip_abort_on_stale_network_failure(self, compressor):
        """Same as above but for _last_summary_network_failure."""
        compressor._ineffective_compression_count = 1
        compressor._last_summary_network_failure = True  # stale
        msgs = self._make_messages()

        with patch.object(compressor, "_generate_summary") as mock_gen:
            result = compressor.compress(msgs, force=False)

        assert result is not None
        assert len(result) > 0
        mock_gen.assert_not_called()

    def test_no_skip_when_force_true(self, compressor):
        """force=True (manual /compress) must bypass the feasibility check."""
        compressor._ineffective_compression_count = 1
        msgs = self._make_messages()

        with patch.object(compressor, "_generate_summary", return_value="LLM summary") as mock_gen:
            compressor.compress(msgs, force=True)

        # _generate_summary MUST have been called despite tiny middle section
        mock_gen.assert_called_once()
        assert compressor._prellm_skip_count == 0

    def test_no_skip_when_no_prior_strikes(self, compressor):
        """No prior ineffectiveness strikes → feasibility check doesn't fire."""
        compressor._ineffective_compression_count = 0
        msgs = self._make_messages()

        with patch.object(compressor, "_generate_summary", return_value="LLM summary") as mock_gen:
            compressor.compress(msgs, force=False)

        mock_gen.assert_called_once()
        assert compressor._prellm_skip_count == 0

    def test_skip_count_resets_on_session_reset(self, compressor):
        """_prellm_skip_count must reset alongside _ineffective_compression_count."""
        compressor._prellm_skip_count = 5
        compressor._ineffective_compression_count = 2

        # on_session_reset() resets all per-session counters
        compressor.on_session_reset()

        assert compressor._prellm_skip_count == 0
        assert compressor._ineffective_compression_count == 0

    def test_skip_count_resets_on_bind_session_state(self, compressor):
        """Rebinding to a session row must clear the per-session skip counter
        like every other per-session guard (stale carry-over from a previous
        binding must not inflate the new session's observability count)."""
        compressor._prellm_skip_count = 5

        compressor.bind_session_state(None, "s-new")

        assert compressor._prellm_skip_count == 0

    def test_skip_fires_on_fat_tail_small_middle(self, compressor):
        """The target scenario from #60451: a tool-heavy transcript whose
        protected tail already holds most of the tokens, leaving a tiny
        middle window. The skip must fire and _generate_summary must not
        be called."""
        compressor._ineffective_compression_count = 1
        msgs = [{"role": "system", "content": "system prompt"}]
        # Small middle: a few lightweight early exchanges.
        for i in range(6):
            msgs.append({"role": "user", "content": f"early question {i}"})
            msgs.append({"role": "assistant", "content": "brief answer"})
        # Fat tail: recent turns carrying big tool-style payloads.
        for i in range(4):
            msgs.append({"role": "user", "content": f"recent request {i}"})
            msgs.append({"role": "assistant", "content": "big result " + "x" * 20000})

        with patch.object(compressor, "_generate_summary") as mock_gen:
            result = compressor.compress(msgs, force=False)

        mock_gen.assert_not_called()
        assert compressor._prellm_skip_count == 1
        assert compressor._last_feasibility_skip is True
        # Deterministic dropping still made progress.
        assert len(result) < len(msgs)

    def test_boundary_accounting_skip_does_not_feed_fallback_streak(self, compressor):
        """The interaction teknium's sweeper review flagged on #68334: the
        skip path sets _last_summary_fallback_used, which the boundary
        wrapper (conversation_compression.py) records via
        record_completed_compaction(used_fallback=True) — incrementing
        _fallback_compression_streak, whose second occurrence blocks
        automatic compression. Two deliberate skips must NOT trip that
        breaker."""
        compressor._ineffective_compression_count = 1
        msgs = self._make_messages()

        for _ in range(2):
            with patch.object(compressor, "_generate_summary") as mock_gen:
                compressor.compress(list(msgs), force=False)
            mock_gen.assert_not_called()
            # Mirror the boundary wrapper's bookkeeping
            # (agent/conversation_compression.py: record_completed_compaction
            # call after a made-progress boundary).
            assert compressor._last_compression_made_progress is True
            compressor.record_completed_compaction(
                used_fallback=compressor._last_summary_fallback_used,
                feasibility_skip=compressor._last_feasibility_skip,
            )

        assert compressor._prellm_skip_count == 2
        assert compressor._fallback_compression_streak == 0
        assert not compressor._automatic_compression_blocked_locally(), (
            "two deliberate feasibility skips must not disable automatic "
            "compression via the fallback-streak breaker"
        )

    def test_boundary_accounting_skip_does_not_reset_fallback_streak(self, compressor):
        """A skip proves nothing about the summary model's health: an
        existing real-fallback streak must survive a skip boundary (neither
        incremented nor reset)."""
        compressor.record_completed_compaction(used_fallback=True)
        assert compressor._fallback_compression_streak == 1

        compressor.record_completed_compaction(
            used_fallback=True, feasibility_skip=True,
        )

        assert compressor._fallback_compression_streak == 1

    def test_real_fallback_still_feeds_streak(self, compressor):
        """Negative control: a genuine summary-failure fallback boundary
        (no feasibility skip) must keep incrementing the streak breaker."""
        compressor._ineffective_compression_count = 1
        msgs = self._make_messages(content="filler " * 3000)  # fat middle → no skip

        with patch.object(compressor, "_generate_summary", return_value=None):
            compressor.compress(list(msgs), force=False)

        assert compressor._last_feasibility_skip is False
        assert compressor._last_summary_fallback_used is True
        compressor.record_completed_compaction(
            used_fallback=compressor._last_summary_fallback_used,
            feasibility_skip=compressor._last_feasibility_skip,
        )
        assert compressor._fallback_compression_streak == 1
