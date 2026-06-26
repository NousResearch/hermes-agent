"""Regression tests for issue #53008 auxiliary-model compression loops.

Two issue-specific safeguards remain necessary on top of the provider-token
anti-thrash verification already present on current main:

1. The auto-lowered threshold uses 80% of the auxiliary model context so the
   summary and protected tail have headroom.
2. If the actual compression window exceeds the auxiliary model context,
   compression temporarily uses the main model and always restores the
   configured auxiliary model afterwards.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from agent.context_compressor import ContextCompressor, estimate_messages_tokens_rough

Message = dict[str, Any]


def _make_compressor(
    *,
    model: str = "test-model",
    threshold_percent: float = 0.50,
    protect_first_n: int = 2,
    protect_last_n: int = 3,
    quiet_mode: bool = True,
    abort_on_summary_failure: bool = False,
) -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=200_000):
        return ContextCompressor(
            model=model,
            threshold_percent=threshold_percent,
            protect_first_n=protect_first_n,
            protect_last_n=protect_last_n,
            quiet_mode=quiet_mode,
            abort_on_summary_failure=abort_on_summary_failure,
        )


def _build_session(n_turns: int, chars_per_msg: int = 200) -> list[Message]:
    """Build a multi-turn conversation with controllable size."""
    base = " ".join(["x"] * chars_per_msg)
    messages: list[Message] = [{"role": "system", "content": "You are helpful."}]
    for i in range(n_turns):
        messages.append({"role": "user", "content": f"{base} turn {i}"})
        messages.append({"role": "assistant", "content": f"{base} reply {i}"})
    return messages


# ---------------------------------------------------------------------------
# Layer 1: safety margin (tested via conversation_compression integration)
# The 80% margin is applied in check_compression_model_feasibility, which
# requires a full agent mock.  The unit test for the margin is in
# test_compression_feasibility.py.  Here we test the compressor-side fixes.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Layer 2: proactive main-model fallback in compress()
# ---------------------------------------------------------------------------


class TestProactiveMainModelFallback:
    """compress() falls back to the main model when the compression window
    (middle turns sent to the summariser) exceeds the aux model's context."""

    def test_falls_back_when_window_exceeds_aux_context(self):
        comp = _make_compressor()
        comp.summary_model = "aux-small-model"
        comp._aux_compression_context_length = 500

        messages = _build_session(30, chars_per_msg=200)

        captured = []

        def _capture(*args, **kwargs):
            captured.append(comp.summary_model)
            return "Summary."

        with patch.object(comp, "_generate_summary", side_effect=_capture):
            comp.compress(messages, current_tokens=50_000)

        assert captured == [""], (
            f"summary_model should be empty (main model) when window > aux_context, got {captured}"
        )

    def test_restores_aux_model_after_fallback(self):
        comp = _make_compressor()
        comp.summary_model = "aux-small-model"
        comp._aux_compression_context_length = 500

        messages = _build_session(30, chars_per_msg=200)

        with patch.object(comp, "_generate_summary", return_value="Summary."):
            comp.compress(messages, current_tokens=50_000)

        assert comp.summary_model == "aux-small-model"

    def test_fallback_explicitly_bypasses_auxiliary_task_config(self):
        """The real call boundary must receive the main runtime explicitly."""
        comp = _make_compressor()
        comp.provider = "test"
        comp.base_url = "http://test"
        comp.api_key = "test-key"
        comp.summary_model = "aux-small-model"
        comp._aux_compression_context_length = 500
        messages = _build_session(30, chars_per_msg=200)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Summary."))]
        )

        with patch("agent.context_compressor.call_llm", return_value=response) as call:
            comp.compress(messages, current_tokens=50_000)

        kwargs = call.call_args.kwargs
        assert kwargs["provider"] == "test"
        assert kwargs["model"] == "test-model"
        assert kwargs["base_url"] == "http://test"
        assert kwargs["api_key"] == "test-key"
        assert "aux-small-model" not in kwargs.values()

    def test_no_fallback_when_window_within_aux_context(self):
        comp = _make_compressor()
        comp.summary_model = "aux-small-model"
        comp._aux_compression_context_length = 10_000_000

        messages = _build_session(30, chars_per_msg=200)

        captured = []

        def _capture(*args, **kwargs):
            captured.append(comp.summary_model)
            return "Summary."

        with patch.object(comp, "_generate_summary", side_effect=_capture):
            comp.compress(messages, current_tokens=50_000)

        assert captured == ["aux-small-model"]

    def test_no_fallback_when_no_aux_model(self):
        comp = _make_compressor()
        comp.summary_model = ""
        comp._aux_compression_context_length = 500

        messages = _build_session(30, chars_per_msg=200)

        with patch.object(comp, "_generate_summary", return_value="Summary."):
            comp.compress(messages, current_tokens=50_000)

        assert comp.summary_model == ""

    def test_no_fallback_when_aux_context_not_set(self):
        comp = _make_compressor()
        comp.summary_model = "aux-model"
        comp._aux_compression_context_length = 0

        messages = _build_session(30, chars_per_msg=200)

        with patch.object(comp, "_generate_summary", return_value="Summary."):
            comp.compress(messages, current_tokens=50_000)

        assert comp.summary_model == "aux-model"

    def test_no_fallback_when_session_large_but_window_small(self):
        """The comparison is against the compression window, NOT the full
        session.  A session can be much larger than the aux context while
        the window (middle turns) still fits — in that case the aux model
        is used, not the main model."""
        comp = _make_compressor()
        comp.summary_model = "aux-small-model"
        comp._aux_compression_context_length = 50_000

        messages = _build_session(30, chars_per_msg=200)

        captured = []

        def _capture(*args, **kwargs):
            captured.append(comp.summary_model)
            return "Summary."

        with patch.object(comp, "_generate_summary", side_effect=_capture):
            comp.compress(messages, current_tokens=200_000)

        assert captured == ["aux-small-model"], (
            "aux model should be used — the window fits even though the "
            f"full session (200K) exceeds the aux context (50K). Got {captured}"
        )

    def test_aux_model_restored_even_when_summary_raises(self):
        """If _generate_summary raises an unexpected exception, the aux
        model is still restored via the finally block."""
        comp = _make_compressor()
        comp.summary_model = "aux-small-model"
        comp._aux_compression_context_length = 500

        messages = _build_session(30, chars_per_msg=200)

        with patch.object(comp, "_generate_summary", side_effect=RuntimeError("boom")):
            try:
                comp.compress(messages, current_tokens=50_000)
            except RuntimeError:
                pass

        assert comp.summary_model == "aux-small-model", (
            f"summary_model should be restored even after an exception, got '{comp.summary_model}'"
        )

    def test_aux_model_restored_when_summary_returns_none(self):
        """When _generate_summary returns None (summary failure), the aux
        model is restored before the abort/fallback path runs."""
        comp = _make_compressor(abort_on_summary_failure=True)
        comp.summary_model = "aux-small-model"
        comp._aux_compression_context_length = 500

        messages = _build_session(30, chars_per_msg=200)

        with patch.object(comp, "_generate_summary", return_value=None):
            comp.compress(messages, current_tokens=50_000)

        assert comp.summary_model == "aux-small-model"

    def test_overflow_warning_emitted_once(self):
        comp = _make_compressor(quiet_mode=False)
        comp.summary_model = "aux-small-model"
        comp._aux_compression_context_length = 500

        messages = _build_session(30, chars_per_msg=200)

        with patch.object(comp, "_generate_summary", return_value="Summary."):
            comp.compress(messages, current_tokens=50_000)
            assert comp._aux_context_overflow_warned is True
            comp.compress(messages, current_tokens=50_000)

        assert comp._aux_context_overflow_warned is True
