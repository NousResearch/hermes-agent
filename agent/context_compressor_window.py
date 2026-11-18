"""Compaction window planning for ContextCompressor: the head/middle/tail boundaries that compress() and
the opt-in prepared-compaction pass (agent/prepared_compaction.py) both plan, and the splice of a prepared
summary into that window."""

from __future__ import annotations

import logging
from typing import Any, Optional

from agent.context_compressor_tail import tail_cut_progress_floor
from agent.prepared_compaction import PreparedCompaction, PreparedSummary

logger = logging.getLogger(__name__)


def _restart_probe_extra() -> int:
    from agent.context_compressor import _RESTART_HANDOFF_PROBE_EXTRA_MESSAGES
    return _RESTART_HANDOFF_PROBE_EXTRA_MESSAGES


#: Tools whose pruned form is a correctness filter, not a size cut: a clarify one-liner keeps only real user
#: answers and drops legacy headless notices ("... using your own judgment") the summarizer would read as one.
_SUMMARY_KEEPS_PRUNED_FORM = frozenset({"clarify"})


def summary_view(
    plan: list[dict[str, Any]], originals: list[dict[str, Any]], start: int, end: int,
) -> list[dict[str, Any]]:
    """*plan* with the window ``[start, end)`` put back to the original tool results: what the summarizer
    reads. The pre-summary prune shapes only the boundaries and the kept head/tail; results being summarized
    are never pointers, stubs or dedup references (the summarizer's own input budget bounds them)."""
    if len(originals) != len(plan):
        return plan
    from agent.context_compressor import _tool_calls_by_id

    calls = _tool_calls_by_id(originals)
    window = [
        original if (original.get("role") == "tool"
                     and calls.get(original.get("tool_call_id") or "", ("",))[0] not in _SUMMARY_KEEPS_PRUNED_FORM)
        else pruned
        for pruned, original in zip(plan[start:end], originals[start:end])
    ]
    return [*plan[:start], *window, *plan[end:]]


class CompactionWindowMixin:
    """Window planning shared by compress() and the background pass."""

    # Set by agent_init when compression.prepare_ahead is on; None leaves compress() inline as before.
    prepared_compaction: Optional[PreparedCompaction] = None
    # Set on delegated children (tools/delegate_tool.py): their first user turn is the whole brief.
    pin_task_turn: bool = False

    def _restart_handoff_probe_bounds(self, messages: list[dict[str, Any]]) -> tuple[int, int]:
        """Return the bounded transcript region that can indicate restart decay."""
        if not messages or self.protect_first_n <= 0:
            return 0, 0
        first_non_system = 1 if messages[0].get("role") == "system" else 0
        return first_non_system, min(len(messages), first_non_system + self.protect_first_n + _restart_probe_extra())

    def _effective_protect_first_n(self, messages: Optional[list[dict[str, Any]]] = None) -> int:
        """``protect_first_n``, decayed to 0 once the session has been compressed so early turns don't fossilize.
        After a restart the decayed state is inferred from handoff summaries in the resumed head.
        A delegated child's first user turn is its whole brief (goal, context, output contract), so it
        never decays: a steer would otherwise become the turn compaction restates, and the brief goes
        into the summary."""
        if self.pin_task_turn:
            return 1
        if self.compression_count >= 1 or self._previous_summary:
            return 0
        if messages and self.protect_first_n > 0:
            # Probe only the early resumed-handoff shape; summary-like tail content must not decay protection.
            probe_start, probe_end = self._restart_handoff_probe_bounds(messages)
            if any(map(self._is_context_summary_message, messages[probe_start:probe_end])):
                return 0
        return self.protect_first_n

    def _protect_head_size(self, messages: list[dict[str, Any]]) -> int:
        """Head messages to protect: the system prompt (if present) plus the decaying ``protect_first_n`` extra rows.

        The ``protect_first_n`` portion DECAYS after the first compression (see _effective_protect_first_n)
        so early user turns don't fossilize across repeated compactions (#11996).
        """
        head = 1 if messages and messages[0].get("role") == "system" else 0
        return head + self._effective_protect_first_n(messages)

    def _compress_window(self, messages: list[dict[str, Any]], *, protect_last_n: int | None = None) -> tuple[int, int]:
        """Return ``(compress_start, compress_end)`` for the summarizable middle."""
        compress_start = self._align_boundary_forward(messages, self._protect_head_size(messages))
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start, protect_last_n=protect_last_n)
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
        self, messages: list[dict[str, Any]], *, protect_last_n: int | None = None,
    ) -> tuple[list[dict[str, Any]], int, int, int]:
        """Phases 1-2 of compress(): ``(working, pruned_count, compress_start, compress_end)``.
        ``working`` is a pruned copy without blank echoes; the boundaries index into it. Write-free: the
        prune only decides where the window ends. The summarizer reads the window's ORIGINAL rows
        (:func:`summary_view`), and archiving happens once, after a summary exists (:meth:`_commit_window`).
        ``protect_last_n`` is a caller-chosen verbatim boundary (``/compress here N``): when set, it
        replaces the configured tail window through pruning and the boundary cut alike (#144423)."""
        effective_protect_last_n = self.protect_last_n if protect_last_n is None else protect_last_n
        working, pruned_count = self._prune_old_tool_results(
            messages, protect_tail_count=effective_protect_last_n, protect_tail_tokens=self.tail_token_budget,
        )
        plan = self._drop_blank_echoes(working)
        compress_start, compress_end = self._compress_window(plan, protect_last_n=protect_last_n)
        return plan, pruned_count, compress_start, compress_end

    def _commit_window(
        self, originals: list[dict[str, Any]], plan: list[dict[str, Any]], pruned_count: int, start: int, end: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """After the summary: save, byte-exact, every result *plan* shortens and every whole result the summary
        replaces, each stub naming its file, and (lean mode) demote the stale tail results the same way. Returns
        ``(rows to assemble from, pruned_count)``. *originals* is the unpruned transcript without blank echoes
        (index-aligned with *plan*). The boundaries stay as summarized even when a failed write keeps an
        original whole."""
        from tools.tool_result_storage import archive_compaction_drops

        session_id = getattr(self, "_session_id", "") or ""
        if pruned_count:
            plan, _written, kept = archive_compaction_drops(originals, plan, session_id)
            pruned_count -= kept
        if start < end:
            plan, offloaded, _written, failed = self._offload_window_results(plan, start, end)
            if failed:
                # The summary (which read the original) still replaces the window; refusing would re-pay the
                # summary on every later turn while the disk stays full. Say so: no byte-exact copy exists.
                logger.warning("compaction: %d summarized tool result(s) could not be archived (disk full or "
                               "read-only home); only the summary and the session store keep them", failed)
            pruned_count += offloaded
        if getattr(self, "tail_mode", "lean") == "lean":
            plan = self._demote_stale_tail_tools(plan, end)
        return plan, pruned_count

    def _assemble_with_archive_list(
        self, messages: list[dict[str, Any]], start: int, end: int, scan: Any, summary: str,
    ) -> list[dict[str, Any]]:
        """``_assemble_compressed`` with every archive a replaced row named, that neither the kept head and tail
        nor the generated summary still name, listed in the summary row: the summarizer never saw the pointers
        (it read the originals), and the list keeps the model's path back and the reference the retention
        sweep keeps the file for."""
        from tools.tool_result_storage import archive_list_for_summary, pruned_archive_refs

        named = pruned_archive_refs(messages)
        kept = [*messages[:start], *messages[max(end, scan.tail_start):], {"content": summary}]
        return self._assemble_compressed(messages, start, end, scan, summary + archive_list_for_summary(named, kept))

    def _offload_window_results(
        self, plan: list[dict[str, Any]], start: int, end: int,
    ) -> tuple[list[dict[str, Any]], int, list[str], int]:
        """The tool results the summary replaced, saved after it was generated: each one the prune left whole
        inside ``plan[start:end]`` becomes its one-line stub naming the byte-exact file, so the model can still
        open the exact output. Local backend only. Returns ``(rows, stubbed, written paths, failed writes)``."""
        from agent.context_compressor import _PRUNE_MIN_CHARS, _rewritten, _tool_calls_by_id
        from tools.tool_result_storage import _backend_reads_host_files, archive_pruned_originals, has_archive_pointer

        if not _backend_reads_host_files():
            return plan, 0, [], 0
        demoted, calls = list(plan), _tool_calls_by_id(plan)
        changed = 0
        for i in range(start, end):
            if self._demote_tool_result_at(demoted, i, calls, _PRUNE_MIN_CHARS):
                changed += 1
            elif (plan[i].get("role") == "tool" and plan[i].get("content")
                  and (isinstance(plan[i]["content"], (list, dict))
                       or isinstance(plan[i]["content"], str) and not has_archive_pointer(plan[i]["content"]))):
                # The summary replaces small strings too, even when the size prune left them alone.
                demoted[i] = _rewritten(plan[i], "[Tool output archived]")
                changed += 1
        if not changed:
            return plan, 0, [], 0
        rows, written, kept = archive_pruned_originals(plan, demoted, getattr(self, "_session_id", "") or "")
        return rows, changed - kept, written, kept

    def _take_prepared_summary(
        self, messages: list[dict[str, Any]], compress_start: int, *, fresh: bool, provider_context: bool,
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
        )

    def _adopt_prepared_summary(self, prepared: PreparedSummary) -> str:
        """Adopt a prepared summary as an inline summary call would have left it; returns the handoff text."""
        from agent.prepared_compaction import adopt

        adopt(self, prepared)
        return prepared.summary
