"""Sequential summary fold: chunked summarization for routed summary models.

When a separate (typically small/fast) summary model is configured and the
compression window exceeds what a single call can take, the window is split
into chunks and folded — ``S = summarize(chunk_1); S = update(S, chunk_2)`` —
via the existing iterative-update prompt, instead of silently dropping the
middle of the serialized input. Main-model summarization keeps the
single-call path (its per-call latency would multiply).
"""

from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    _SUMMARY_FOLD_MAX_CHUNKS,
    ContextCompressor,
)


def _compressor(aux_window: int = 0, summary_model: str = "small-model") -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=400_000,
    ):
        c = ContextCompressor(
            model="test/big-model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )
        _ = c.context_length
    c.summary_model = summary_model
    c.summary_model_context_length = aux_window
    return c


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _big_cjk_turns(n: int = 100, chars: int = 5_000) -> list:
    # 100 x 5K CJK chars ≈ 500K serialized chars — far over the 160K
    # aggregate cap, so a fold is required to read everything.
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "測" * chars}
        for i in range(n)
    ]


class TestPlanSummaryFold:
    def test_small_window_returns_single_chunk(self):
        c = _compressor()
        turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        assert c._plan_summary_fold(turns) == [turns]

    def test_no_fold_without_routed_summary_model(self):
        c = _compressor(summary_model="")
        turns = _big_cjk_turns()
        assert c._plan_summary_fold(turns) == [turns]

    def test_oversized_window_splits_into_bounded_chunks(self):
        c = _compressor(aux_window=128_000)
        turns = _big_cjk_turns()

        chunks = c._plan_summary_fold(turns)

        assert 1 < len(chunks) <= _SUMMARY_FOLD_MAX_CHUNKS
        assert [t for chunk in chunks for t in chunk] == turns  # order + coverage
        # Every chunk except a possibly-overflowing final one stays under the
        # aggregate char cap once serialized.
        for chunk in chunks[:-1]:
            assert len(c._serialize_for_summary(chunk)) <= c._SUMMARY_INPUT_MAX_CHARS

    def test_tool_results_stay_with_their_assistant_turn(self):
        c = _compressor(aux_window=128_000)
        turns = []
        for i in range(60):
            turns.append({"role": "user", "content": "問" * 2_000})
            turns.append(
                {
                    "role": "assistant",
                    "content": "查" * 2_000,
                    "tool_calls": [{"id": f"call_{i}", "function": {"name": "t", "arguments": "{}"}}],
                }
            )
            turns.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "果" * 2_000})

        chunks = c._plan_summary_fold(turns)

        assert len(chunks) > 1
        for chunk in chunks:
            # A chunk never starts with a dangling tool result.
            assert chunk[0].get("role") in ("user", "assistant")

    def test_fold_guard_prevents_replanning(self):
        c = _compressor(aux_window=128_000)
        c._in_summary_fold = True
        turns = _big_cjk_turns()
        assert c._plan_summary_fold(turns) == [turns]


class TestGenerateSummaryFolded:
    def test_fold_threads_intermediate_summary_between_chunks(self):
        c = _compressor(aux_window=128_000)
        turns = _big_cjk_turns()
        expected_chunks = len(c._plan_summary_fold(turns))
        assert expected_chunks > 1
        prompts = []

        def _capture(**kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            return _response(f"S{len(prompts)} chunk summary")

        with patch("agent.context_compressor.call_llm", side_effect=_capture):
            result = c._generate_summary(turns)

        assert len(prompts) == expected_chunks
        # Chunk 1: fresh summary; later chunks: iterative updates carrying
        # the previous chunk's intermediate summary.
        assert "PREVIOUS SUMMARY:" not in prompts[0]
        for i in range(1, expected_chunks):
            assert "PREVIOUS SUMMARY:" in prompts[i]
            assert f"S{i} chunk summary" in prompts[i]
        # The final summary is what the fold returns and stores.
        final = f"S{expected_chunks} chunk summary"
        assert result is not None and final in result
        # Stored for the next compaction's iterative update (the grounding
        # post-pass may append a historical-task section around it).
        assert final in (c._previous_summary or "")

    def test_memory_context_only_in_final_chunk(self):
        c = _compressor(aux_window=128_000)
        turns = _big_cjk_turns()
        prompts = []

        def _capture(**kwargs):
            prompts.append(kwargs["messages"][0]["content"])
            return _response(f"S{len(prompts)}")

        with patch("agent.context_compressor.call_llm", side_effect=_capture):
            c._generate_summary(turns, memory_context="remember-me-block")

        assert len(prompts) > 1
        for early in prompts[:-1]:
            assert "remember-me-block" not in early
        assert "remember-me-block" in prompts[-1]

    def test_mid_fold_failure_restores_previous_summary(self):
        c = _compressor(aux_window=128_000)
        c._previous_summary = "pre-fold summary"
        turns = _big_cjk_turns()
        calls = {"n": 0}

        def _fail_second(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _response("S1")
            raise TimeoutError("summary model timed out")

        with patch("agent.context_compressor.call_llm", side_effect=_fail_second):
            result = c._generate_summary(turns)

        assert result is None
        # The intermediate S1 must not leak into iterative state: a failed
        # fold looks exactly like a failed single call.
        assert c._previous_summary == "pre-fold summary"
        assert c._in_summary_fold is False

    def test_telemetry_records_chunk_count(self):
        c = _compressor(aux_window=128_000)
        telemetry = {"chunking": False, "chunk_count": 0}
        c._active_compression_telemetry = telemetry
        turns = _big_cjk_turns()
        expected_chunks = len(c._plan_summary_fold(turns))
        calls = {"n": 0}

        def _reply(**kwargs):
            calls["n"] += 1
            return _response(f"S{calls['n']}")

        with patch("agent.context_compressor.call_llm", side_effect=_reply):
            c._generate_summary(turns)

        assert telemetry["chunking"] is True
        assert telemetry["chunk_count"] == expected_chunks > 1

    def test_overflow_beyond_max_chunks_merges_into_final_chunk(self):
        c = _compressor(aux_window=128_000)
        turns = _big_cjk_turns(n=400)  # ~2M chars — far beyond 4 chunks' worth

        chunks = c._plan_summary_fold(turns)

        assert len(chunks) == _SUMMARY_FOLD_MAX_CHUNKS
        assert [t for chunk in chunks for t in chunk] == turns
