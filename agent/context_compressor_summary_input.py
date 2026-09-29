"""Window-derived budget for the summary prompt's variable blocks.

Split out of ``agent.context_compressor`` (a god-file whose line count may only go down): the budget,
the head/tail bounding it feeds, and the coverage the sampler reports are one topic, and keeping
them here is what lets the compressor keep its size. The mixin is composed into
``ContextCompressor``; ``_SUMMARY_INPUT_MAX_CHARS`` and its class alias stay in the compressor module,
so this module reads the ceiling through the class and imports nothing back from it.
"""

from __future__ import annotations

from typing import Optional

from agent.model_metadata import CHARS_PER_TOKEN

# Share of the summariser's window the prompt may occupy, in tokens; the remainder covers the fixed
# template, the focus block, and the generated summary. 0.55 of 64K = 35.2K tokens.
_SUMMARY_INPUT_WINDOW_FRACTION = 0.55
# Floor for the derived budget: a tiny window must still get a usable transcript block.
_SUMMARY_INPUT_MIN_CHARS = 24_000
# Share of the budget reserved for the previous-summary block during an iterative update. Only
# *reserved* when a previous summary is actually present and that large; the transcript block gets
# the remainder, so the two together stay within the budget.
_SUMMARY_PREVIOUS_SHARE = 0.45


class SummaryInputBudgetMixin:
    """Bound the summary prompt to the window of the model actually being called."""

    def _summary_input_window_tokens(self) -> Optional[int]:
        """The summariser's own window: the aux ceiling when one is installed, else the main window.

        ``_aux_context_ceiling`` is set by the feasibility probe exactly when the compression route
        points at a model with a smaller window than the main model (#114707) — the same condition
        that clamps the trigger. Reading it here is what keeps the prompt bound tied to the window
        actually being called.
        """
        for value in (getattr(self, "_aux_context_ceiling", None), self.context_length):
            if isinstance(value, int) and value > 0:
                return value
        return None

    def _summary_input_budget_chars(self) -> int:
        """Total char budget for the prompt's variable blocks (transcript + previous summary + extras).

        The static ``_SUMMARY_INPUT_MAX_CHARS`` is a ceiling, not the budget: on a small aux window
        (e.g. 64K) it would let the transcript and the previous summary *each* fill their own cap,
        which overflows the window in the single assembled request. Derive the real budget from the
        summariser's window instead, and never exceed the ceiling.
        """
        window = self._summary_input_window_tokens()
        if not window:
            return self._SUMMARY_INPUT_MAX_CHARS
        derived = int(window * _SUMMARY_INPUT_WINDOW_FRACTION) * CHARS_PER_TOKEN
        return max(_SUMMARY_INPUT_MIN_CHARS, min(self._SUMMARY_INPUT_MAX_CHARS, derived))

    def _previous_summary_input_budget(self, total_budget: int) -> Optional[int]:
        """The share of *total_budget* reserved for the previous-summary block, or ``None``.

        Iterative updates reserve this so a large rehydrated handoff cannot crowd out the new turns;
        the transcript block takes the remainder, so the two blocks stay inside one budget.
        """
        if not self._previous_summary:
            return None
        return max(_SUMMARY_INPUT_MIN_CHARS, int(total_budget * _SUMMARY_PREVIOUS_SHARE))

    @classmethod
    def _bound_summary_input(cls, content: str, max_chars: Optional[int] = None) -> str:
        """Cap summarizer input, keeping head and tail and marking the omitted middle.

        ``max_chars`` is the *total* budget for the assembled prompt's variable blocks, from
        ``_summary_input_budget_chars``; it defaults to the static ceiling for callers with no
        window context (tests, direct calls).
        """
        budget = cls._SUMMARY_INPUT_MAX_CHARS if max_chars is None else max(0, int(max_chars))
        if len(content) <= budget:
            return content

        marker_template = (
            "\n\n...[summary input truncated: omitted "
            "{omitted:,} chars from the middle to keep compression prompt bounded]...\n\n"
        )
        # Marker width can change with the omitted count; estimate, then rebuild once.
        omitted = len(content)
        for _ in range(2):
            marker = marker_template.format(omitted=omitted)
            remaining = max(budget - len(marker), 0)
            head_chars = int(remaining * 0.45)
            tail_chars = remaining - head_chars
            omitted = max(len(content) - head_chars - tail_chars, 0)
        tail = content[-tail_chars:].lstrip() if tail_chars else ""
        return content[:head_chars].rstrip() + marker + tail

    def _record_summary_input_coverage(self, coverage: dict[str, int]) -> None:
        """Expose lean sampling coverage without including transcript content in telemetry."""
        telemetry = getattr(self, "_active_compression_telemetry", None)
        if not isinstance(telemetry, dict):
            return
        telemetry.update({
            "summary_input_chars": coverage["input_chars"],
            "summary_input_sampled_chars": coverage["sampled_chars"],
            "summary_input_omitted_chars": coverage["omitted_chars"],
            "summary_input_record_count": coverage["record_count"],
            "summary_input_sampled_record_count": coverage["sampled_record_count"],
            "summary_input_elided_record_count": coverage["elided_record_count"],
        })
