"""Per-call summarizer-input fit for small-context auxiliary models.

A summary model whose context window is smaller than the session threshold
must NOT shrink the session threshold (the old auto-lower halved the main
model's usable window). Instead:

* ``check_compression_model_feasibility`` stashes the aux window on the
  compressor and leaves the threshold alone; and
* ``_generate_summary`` shrinks the serialized-turns block per call until
  the assembled prompt plus the output budget fits that window.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    _SUMMARY_FIT_MARGIN_TOKENS,
    ContextCompressor,
)
from agent.conversation_compression import check_compression_model_feasibility
from agent.model_metadata import estimate_tokens_rough


def _compressor(context_length: int = 200_000) -> ContextCompressor:
    with patch(
        "agent.context_compressor.get_model_context_length",
        return_value=context_length,
    ):
        c = ContextCompressor(
            model="test/big-model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=1,
            quiet_mode=True,
        )
        _ = c.context_length
        return c


def _response(content: str):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = content
    return mock_response


def _cjk_turns(n: int = 30, chars: int = 5_000) -> list:
    # CJK text is the worst case for the rough estimator (1 token/char),
    # so it exercises the fit loop with realistic dense content.
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "測" * chars}
        for i in range(n)
    ]


class TestFeasibilityStashesWindowInsteadOfLoweringThreshold:
    def _agent(self, compressor: ContextCompressor) -> SimpleNamespace:
        return SimpleNamespace(
            compression_enabled=True,
            context_compressor=compressor,
            _current_main_runtime=lambda: {},
            _custom_providers={},
            _aux_compression_context_length_config=None,
            _emit_status=lambda _msg: None,
            model="test/big-model",
            provider="test",
        )

    def _run_check(self, agent, aux_context: int) -> None:
        client = SimpleNamespace(base_url="https://aux.invalid/v1", api_key="k")
        with (
            patch(
                "agent.auxiliary_client.get_text_auxiliary_client",
                return_value=(client, "small-model"),
            ),
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("small-provider", "small-model", "", "", ""),
            ),
            patch(
                "agent.model_metadata.get_model_context_length",
                return_value=aux_context,
            ),
        ):
            check_compression_model_feasibility(agent)

    def test_small_aux_window_keeps_threshold_and_stashes_window(self):
        compressor = _compressor()
        threshold_before = compressor.threshold_tokens
        percent_before = compressor.threshold_percent
        agent = self._agent(compressor)
        assert 128_000 < threshold_before  # premise of the scenario

        self._run_check(agent, aux_context=128_000)

        assert compressor.threshold_tokens == threshold_before
        assert compressor.threshold_percent == percent_before
        assert compressor.summary_model_context_length == 128_000
        # No user-facing warning: the per-call bound makes the model work.
        assert getattr(agent, "_compression_warning", None) is None

    def test_large_aux_window_stashes_without_side_effects(self):
        compressor = _compressor()
        threshold_before = compressor.threshold_tokens
        agent = self._agent(compressor)

        self._run_check(agent, aux_context=1_000_000)

        assert compressor.threshold_tokens == threshold_before
        assert compressor.summary_model_context_length == 1_000_000


class TestGenerateSummaryWindowFit:
    def test_prompt_trimmed_to_fit_small_window(self):
        """The per-call fit is the backstop below fold granularity: a window
        the fold planner cannot split further (here: forced single-chunk, as
        for one giant turn group) must still be trimmed to the aux window."""
        c = _compressor()
        c.summary_model = "small-model"
        c.summary_model_context_length = 70_000
        turns = _cjk_turns()
        # Oversized CJK content normally folds into chunks; pin the
        # single-call path so the trim backstop itself is under test.
        c._in_summary_fold = True

        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response("summary body"),
        ) as mock_call:
            summary = c._generate_summary(turns)

        assert summary  # the call still succeeds end-to-end
        prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        budget = c._compute_summary_budget(turns)
        fit_budget = 70_000 - budget - _SUMMARY_FIT_MARGIN_TOKENS
        assert estimate_tokens_rough(prompt) <= fit_budget
        assert "summary input truncated" in prompt

    def test_no_trim_without_stashed_window(self):
        c = _compressor()
        c.summary_model = "small-model"
        c.summary_model_context_length = 0
        turns = _cjk_turns()

        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response("summary body"),
        ) as mock_call:
            c._generate_summary(turns)

        prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        # Content is CJK-dense: without the per-call fit the prompt far
        # exceeds the small window the previous test squeezed into.
        assert estimate_tokens_rough(prompt) > 70_000

    def test_no_trim_when_summary_model_is_main(self):
        # summary_model == "" means the main model summarizes; the per-call
        # fit must stay inactive even if a stale window value is present.
        c = _compressor()
        c.summary_model = ""
        c.summary_model_context_length = 70_000
        turns = _cjk_turns()

        with patch(
            "agent.context_compressor.call_llm",
            return_value=_response("summary body"),
        ) as mock_call:
            c._generate_summary(turns)

        prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        assert estimate_tokens_rough(prompt) > 70_000

    def test_bound_summary_input_honors_max_chars_override(self):
        content = "x" * 50_000
        bounded = ContextCompressor._bound_summary_input(content, max_chars=10_000)
        assert len(bounded) <= 10_000
        assert "summary input truncated" in bounded
        # Default cap leaves short content untouched.
        assert ContextCompressor._bound_summary_input(content) == content
