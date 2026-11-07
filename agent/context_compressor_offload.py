"""Eviction offload + post-summary audit wiring for the compressor (A1).

Two compaction-safety seams, kept out of the compressor body:

* ``_maybe_offload_evicted`` writes the exact-to-summarize region to a per-session
  searchable JSONL BEFORE any summarizer call (``agent/compaction_offload``), and
  ``_recovery_footer`` points the summary at that file beside ``session_search``.
* ``_audit_summary_boundary`` diffs the finished summary against the region's exact
  identifiers and appends a bounded verbatim repair for genuinely dropped ones
  (``agent/compaction_summary_audit``); LLM-free, so the one-summary-call-per-attempt
  budget (#96603) is unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

from agent.compaction_summary_audit import EXTRA_AUDIT_PATTERNS


def harvest_region_text(turns: list[dict[str, Any]]) -> str:
    """Join the plain text of a turn list for mechanical identifier harvesting.

    Shared with the lean anchor index (``_build_anchor_index``) so the audit sees
    exactly the bytes the anchor index saw.
    """
    from agent.context_compressor import _typed_user_text

    contents = (
        _typed_user_text(m["content"])
        if m.get("role") == "user" and isinstance(m.get("content"), str)
        else m.get("content")
        for m in turns
    )
    return "\n".join(c for c in contents if isinstance(c, str) and c)


class EvictionOffloadMixin:
    """Offload-before-summarize + summary-boundary audit hooks for ContextCompressor."""

    def _maybe_offload_evicted(self, turns: list[dict[str, Any]]) -> Optional[str]:
        """Persist the compacted region verbatim before the summary call; never raises.

        Returns the file path, also stashed on ``_last_offload_path`` (reset per
        attempt) so the recovery footer can name it. Gated by
        ``compression.offload_evicted_history`` (default on).
        """
        self._last_offload_path = None
        if not getattr(self, "offload_evicted", True) or not self._session_id or not turns:
            return None
        from agent.compaction_offload import offload_evicted_region

        path = offload_evicted_region(self._session_id, turns)
        self._last_offload_path = path
        if path:
            telemetry = getattr(self, "_active_compression_telemetry", None)
            if isinstance(telemetry, dict):
                telemetry["offload_path"] = path
        return path

    def _audit_summary_boundary(self, summary: str, turns_to_summarize: list[dict[str, Any]]) -> str:
        """Diff the finished summary against the region; append the bounded repair."""
        from agent.compaction_summary_audit import audit_and_repair_summary
        from agent.context_compressor import _ANCHOR_PATTERNS

        summary, audit_telemetry = audit_and_repair_summary(
            summary,
            harvest_region_text(turns_to_summarize),
            [*_ANCHOR_PATTERNS, *EXTRA_AUDIT_PATTERNS],
            pending_ask=self._latest_user_task_snapshot(turns_to_summarize) or "",
        )
        telemetry = getattr(self, "_active_compression_telemetry", None)
        if isinstance(telemetry, dict):
            telemetry.update(audit_telemetry)
        return summary

    def _recovery_footer(self, session_id: str, region_len: int) -> str:
        """Deterministic pointer to the compacted region in session history.
        state.db keeps every pre-compaction message; naming the session_search re-access path lets the model
        treat compaction as deferred retrieval, not loss. When the pre-summarize offload file
        exists it is named beside the tool pointer as an independent grep target.
        """
        if not session_id:
            return ""
        footer = (
            "\n\n" + _LEAN_RECOVERY_HEADING + "\n"
            f"The {region_len} compacted message(s) remain fully preserved in "
            "session history. If you need any detail this summary does not carry "
            "(exact command output, file contents, error text, earlier "
            "reasoning), recover it with: "
            f"session_search(query='<keywords>', session_id='{session_id}') — "
            "do not guess at lost specifics when you can look them up."
        )
        offload_path = getattr(self, "_last_offload_path", None)
        if offload_path:
            footer += f"\nA verbatim copy of the compacted turns is also on disk: {offload_path}"
        return footer


_LEAN_RECOVERY_HEADING = "## Context Recovery"
