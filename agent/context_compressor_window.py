"""Compaction window planning for ContextCompressor: the head/middle/tail boundaries that compress() and
the opt-in prepared-compaction pass (agent/prepared_compaction.py) both plan, and the splice of a prepared
summary into that window."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Optional

from agent.context_compressor_tail import tail_cut_progress_floor
from agent.model_metadata import estimate_messages_tokens_rough
from agent.prepared_compaction import PreparedCompaction, PreparedSummary


@dataclass(frozen=True)
class CompactionWindow:
    messages: list[dict[str, Any]]
    pruned_count: int
    start: int
    end: int
    protect_active: bool = False
    active_start: Optional[int] = None
    protected_start: Optional[int] = None
    failure_class: Optional[str] = None


class CompactionWindowMixin:
    """Window planning shared by compress() and the background pass."""

    # Set by agent_init when compression.prepare_ahead is on; None leaves compress() inline as before.
    prepared_compaction: Optional[PreparedCompaction] = None

    def _compress_window(self, messages: list[dict[str, Any]]) -> tuple[int, int]:
        """Return ``(compress_start, compress_end)`` for the summarizable middle."""
        compress_start = self._align_boundary_forward(messages, self._protect_head_size(messages))
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)
        # A role collision can merge the summary into the first tail row; keep an actionable user
        # event out of that slot by retaining an older assistant/tool bridge. The bridge never moves
        # the cut below the progress floor: a bridge that leaves only handoff rows in the window
        # would no-op with ``empty_post_handoff_window`` at every trigger.
        latest_actionable_idx = self._find_last_user_message_idx(messages, 0)
        if compress_end == latest_actionable_idx:
            bridge_idx = latest_actionable_idx - 1
            bridge_role = messages[bridge_idx].get("role") if bridge_idx >= 0 else None
            if bridge_role == "tool":
                bridge_idx = self._align_boundary_backward(messages, latest_actionable_idx)
            elif bridge_role != "assistant":
                bridge_idx = -1
            if bridge_idx >= tail_cut_progress_floor(self, messages, compress_start, bridge_idx):
                compress_end = bridge_idx
        return compress_start, compress_end

    def _plan_compaction_window(
        self, messages: list[dict[str, Any]], current_tokens: Optional[int] = None,
    ) -> CompactionWindow:
        """Resolve one working copy and its boundaries for the inline and prepared passes."""
        canonical = self._drop_blank_echoes(messages)
        bounds = self._window_bounds(canonical, current_tokens)
        if bounds.protect_active:
            active = bounds.active_start
            historical, pruned = self._prune_old_tool_results(canonical[:active], protect_tail_count=0)
            return replace(bounds, messages=historical + copy.deepcopy(canonical[active:]), pruned_count=pruned)
        if bounds.start == bounds.end:
            return bounds
        working, pruned_count = self._prune_old_tool_results(
            canonical, protect_tail_count=self.protect_last_n, protect_tail_tokens=self.tail_token_budget,
        )
        working = self._drop_blank_echoes(working)
        # Legacy pruning may demote active tool payloads. Keep the selected mode: a fresh
        # pressure decision here could falsely promise that those payloads stayed verbatim.
        start, end = self._compress_window(working)
        return replace(bounds, messages=working, start=start, end=end, pruned_count=pruned_count)

    def _window_bounds(
        self, canonical: list[dict[str, Any]], current_tokens: Optional[int] = None,
    ) -> CompactionWindow:
        """Observational eligibility: no pruning, demotion, summary or transcript writes."""
        preferred_head = self._protect_head_size(canonical)
        pressure = max(current_tokens or 0, estimate_messages_tokens_rough(canonical)) >= self.threshold_tokens
        active = self._find_last_user_message_idx(canonical, 0)
        system = 1 if canonical and canonical[0].get("role") == "system" else 0
        price = self._tail_row_pricer(canonical)
        head_tokens = sum(price(i) for i in range(system, min(preferred_head, max(system, active))))
        historical_tokens = sum(price(i) for i in range(system, max(system, active)))
        user_boundary, multiple_users = self._user_tail_boundary(canonical, system)
        protected = user_boundary if multiple_users else None
        if len(canonical) <= preferred_head + 4:
            normal = CompactionWindow(canonical, 0, preferred_head, preferred_head,
                                      protected_start=protected, failure_class="insufficient_messages")
        else:
            start, end = self._compress_window(canonical)
            normal = CompactionWindow(canonical, 0, start, end, protected_start=protected)
        # Count protection is a continuity preference, not proof that a short history is small.
        # The active user and its whole tool transaction stay hard even when history gives way.
        blocked = normal.start >= normal.end
        retained_history = sum(price(i) for i in range(max(system, normal.end), max(system, active)))
        # Head continuity and the ordinary tail are separate protections. A full but valid
        # tail must not consume the head's allowance or displace a productive normal middle.
        # Only an oversized historical head, an over-ceiling historical anchor, or a blocked
        # preferred window needs recovery; retain the existing tail's soft-ceiling policy.
        preferred_blocks_reclaim = (blocked or head_tokens > self.tail_token_budget
                                    or retained_history > self._tail_soft_ceiling(self.tail_token_budget))
        if (pressure and active > system and historical_tokens > self.tail_token_budget
            and preferred_blocks_reclaim):
            end = min(active, self._find_tail_cut_by_tokens(canonical, system, allow_split_turn=False))
            end = min(end, user_boundary) if user_boundary is not None else system
            if end > system:
                return CompactionWindow(canonical, 0, system, end, True, active,
                                        protected if protected is not None else active)
        return normal

    def _user_tail_boundary(self, messages: list[dict[str, Any]], system: int) -> tuple[Optional[int], bool]:
        """The configured hard user anchor, including one live user when the setting is absent."""
        users = self._real_user_indices_desc(messages, system)
        count = getattr(self, "min_tail_user_messages", 1)
        count = max(1, count) if isinstance(count, int) and not isinstance(count, bool) else 1
        return (users[min(count, len(users)) - 1] if users else None), count > 1

    def _take_prepared_summary(
        self, messages: list[dict[str, Any]], compress_start: int, *, fresh: bool, provider_context: bool,
        compress_end: Optional[int] = None, protect_active: bool = False,
    ) -> Optional[PreparedSummary]:
        """The summary prepared in the background for this window, which an automatic compaction splices at
        ITS boundary; every newer message stays outside the summary, in the tail. ``fresh`` compactions
        (manual, focus, overflow recovery, memory-provider context) summarize anew, and once a memory
        provider supplies context no further summaries are prepared (each would be thrown away)."""
        if self.prepared_compaction is None:
            return None
        if provider_context:
            self.prepared_compaction.stop_for_provider_context()
        return self.prepared_compaction.take(
            messages, self._session_id, compress_start, eligible=not (fresh or provider_context),
            end=compress_end, protect_active=protect_active,
        )

    def _adopt_prepared_summary(self, prepared: PreparedSummary) -> str:
        """Adopt a prepared summary as an inline summary call would have left it; returns the handoff text."""
        from agent.prepared_compaction import adopt

        adopt(self, prepared)
        return prepared.summary
