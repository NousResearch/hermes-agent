"""Summary-input budgeting: the prompt bound is derived from the summariser's window.

Split out of ``tests/agent/test_context_compressor.py`` (also over its line cap, which may only go
down): the tests that pin the window-derived budget and its two consumers live here.
"""

from unittest.mock import MagicMock, patch

from agent.context_compressor import (
    ContextCompressor,
    SUMMARY_PREFIX,
    _CHARS_PER_TOKEN,
    _SUMMARY_INPUT_MAX_CHARS,
)
from agent.context_compressor_summary_input import (
    _SUMMARY_INPUT_MIN_CHARS,
    _SUMMARY_INPUT_WINDOW_FRACTION,
)


class TestSummaryInputBudget:
    """One budget for the assembled prompt, derived from the window actually being called."""

    def test_iterative_update_path_is_bounded(self):
        """The iterative prompt (previous summary + new turns) must be bounded
        too — a pathological rehydrated handoff must not blow up the prompt."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "updated summary"

        with patch("agent.context_compressor.get_model_context_length", return_value=272000):
            c = ContextCompressor(model="test", quiet_mode=True)
        cap = c._SUMMARY_INPUT_MAX_CHARS
        budget = c._summary_input_budget_chars()
        assert budget <= cap
        c._previous_summary = "PREV_HEAD " + ("p" * (cap * 2)) + " PREV_TAIL"

        messages = [
            {"role": "user", "content": f"turn-{i}-" + ("x" * 6000)}
            for i in range(80)
        ]

        # Stub the synthetic-turn probe: it lazily imports agent.conversation_loop, which trips the
        # test harness's real-home I/O guard before any bounding logic runs.
        with patch.object(
            ContextCompressor, "_is_synthetic_compression_user_turn", return_value=False
        ), patch("agent.context_compressor.call_llm", return_value=mock_response) as mock_call:
            summary = c._generate_summary(messages)

        prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        assert summary.startswith(SUMMARY_PREFIX)
        # The previous-summary block and the new-turns block draw from ONE window-derived
        # budget rather than each filling the static cap independently, so the assembled
        # prompt stays within the budget plus the fixed template (unbounded would be ~800K).
        assert len(prompt) < budget + 30_000
        assert "PREV_HEAD" in prompt
        assert "PREV_TAIL" in prompt

    def test_iterative_prompt_fits_a_small_aux_window(self):
        """A compaction route pointed at a small-window model must not overflow that window.

        Regression for the 104241-token / 65536-window summariser 400: the two variable blocks
        used to be bounded by the static cap *independently* (~2x160K chars = ~80K tokens),
        which cannot fit a 64K window however small each block individually is.
        """
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "updated summary"

        aux_window = 65536
        with patch("agent.context_compressor.get_model_context_length", return_value=272000):
            c = ContextCompressor(model="test", quiet_mode=True)
        c._aux_context_ceiling = aux_window
        budget = c._summary_input_budget_chars()
        assert budget < c._SUMMARY_INPUT_MAX_CHARS
        # One bounded block must not consume the whole window on its own.
        assert budget // _CHARS_PER_TOKEN < aux_window
        c._previous_summary = "PREV_HEAD " + ("p" * 400_000) + " PREV_TAIL"

        messages = [
            {"role": "user", "content": f"turn-{i}-" + ("x" * 6000)}
            for i in range(80)
        ]

        with patch.object(
            ContextCompressor, "_is_synthetic_compression_user_turn", return_value=False
        ), patch("agent.context_compressor.call_llm", return_value=mock_response) as mock_call:
            c._generate_summary(messages)

        prompt = mock_call.call_args.kwargs["messages"][0]["content"]
        prompt_tokens = len(prompt) // _CHARS_PER_TOKEN
        assert prompt_tokens < aux_window, (
            f"prompt {prompt_tokens} tokens exceeds the {aux_window}-token aux window"
        )
        # Room must be left for the generated summary itself.
        assert prompt_tokens <= int(aux_window * _SUMMARY_INPUT_WINDOW_FRACTION) + 4_000
        # Both blocks still contribute: neither is silently dropped.
        assert "PREV_HEAD" in prompt and "turn-79" in prompt

    def test_summary_input_budget_is_window_derived(self):
        """The static cap is a ceiling, not the budget: the budget follows the summariser's window."""
        with patch("agent.context_compressor.get_model_context_length", return_value=272000):
            c = ContextCompressor(model="test", quiet_mode=True)
        # A dedicated aux route wins over the (larger) main window.
        c._aux_context_ceiling = 65536
        assert c._summary_input_budget_chars() == (
            int(65536 * _SUMMARY_INPUT_WINDOW_FRACTION) * _CHARS_PER_TOKEN
        )
        # With no aux ceiling the main window is used, still capped by the static ceiling.
        c._aux_context_ceiling = None
        assert c._summary_input_budget_chars() == _SUMMARY_INPUT_MAX_CHARS
        # A tiny window floors rather than collapsing the transcript block.
        c._aux_context_ceiling = 8000
        assert c._summary_input_budget_chars() == _SUMMARY_INPUT_MIN_CHARS
        # No window information at all degrades to the old behaviour, never crashes.
        c._aux_context_ceiling = None
        c.context_length = 0
        assert c._summary_input_budget_chars() == _SUMMARY_INPUT_MAX_CHARS
