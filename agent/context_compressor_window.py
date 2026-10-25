"""Compaction window planning for ContextCompressor: the head/middle/tail boundaries that compress() and
the opt-in prepared-compaction pass (agent/prepared_compaction.py) both plan, and the splice of a prepared
summary into that window."""

from __future__ import annotations

from typing import Any, Optional

from agent.prepared_compaction import PreparedCompaction, PreparedSummary


class CompactionWindowMixin:
    """Window planning shared by compress() and the background pass."""

    # Set by agent_init when compression.prepare_ahead is on; None leaves compress() inline as before.
    prepared_compaction: Optional[PreparedCompaction] = None

    def _compress_window(self, messages: list[dict[str, Any]]) -> tuple[int, int]:
        """Return ``(compress_start, compress_end)`` for the summarizable middle."""
        compress_start = self._align_boundary_forward(messages, self._protect_head_size(messages))
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)
        # A role collision can merge the summary into the first tail row; keep an actionable user
        # event out of that slot by retaining an older assistant/tool bridge.
        latest_actionable_idx = self._find_last_user_message_idx(messages, 0)
        if compress_end == latest_actionable_idx:
            bridge_idx = latest_actionable_idx - 1
            bridge_role = messages[bridge_idx].get("role") if bridge_idx >= 0 else None
            if bridge_role == "tool":
                bridge_idx = self._align_boundary_backward(messages, latest_actionable_idx)
            elif bridge_role != "assistant":
                bridge_idx = -1
            if bridge_idx > compress_start:
                compress_end = bridge_idx
        return compress_start, compress_end

    def _plan_compaction_window(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int, int]:
        """Phases 1-2 of compress(): ``(working, pruned_count, compress_start, compress_end)``.
        ``working`` is a pruned copy without blank echoes; the boundaries index into it."""
        working, pruned_count = self._prune_old_tool_results(
            messages, protect_tail_count=self.protect_last_n, protect_tail_tokens=self.tail_token_budget,
        )
        working = self._drop_blank_echoes(working)
        compress_start, compress_end = self._compress_window(working)
        return working, pruned_count, compress_start, compress_end

    def _take_prepared_summary(
        self, messages: list[dict[str, Any]], compress_start: int, compress_end: int, *, fresh: bool,
    ) -> Optional[PreparedSummary]:
        """The summary prepared in the background for this window, which an automatic compaction splices at
        ITS boundary; newer messages (at most one more tail budget) stay outside the summary, in the tail.
        ``fresh`` compactions (manual, focus, overflow recovery, memory-provider context) summarize anew."""
        if self.prepared_compaction is None:
            return None
        return self.prepared_compaction.take(
            messages, self._session_id, compress_start, compress_end, eligible=not fresh,
            max_gap_tokens=self.tail_token_budget,
        )
