"""The proactive tool-result prune: deterministic passes plus the in-place row commit (#124102).

Split out of :mod:`agent.context_compressor` (FILE_LINES ratchet): the prune is a self-contained
reclamation feature. The origin module keeps the ``ContextCompressor`` class; this sibling owns
``prune_tool_results_only`` and its in-place commit helper. Log name parity with the origin
module, and every origin-module symbol is read through ``_cc()`` at call time so tests that patch
``agent.context_compressor.X`` stay effective.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("agent.context_compressor")


def _cc():
    """The origin module, resolved lazily: avoids the import cycle and keeps tests that patch
    ``agent.context_compressor.X`` effective (attributes are read at call time)."""
    from agent import context_compressor
    return context_compressor


class ProactivePruneMixin:
    """``ContextCompressor`` mixin: the deterministic, no-LLM tool-result prune and its commit."""

    def prune_tool_results_only(
        self, messages: list[dict[str, Any]], current_tokens: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Deterministic, no-LLM tool-result prune gated on ``proactive_prune_tokens``.

        Protects the tail by message COUNT only. A commit breaks the prompt cache, so it requires
        ``proactive_prune_min_reclaim_tokens`` and a full regrowth runway; otherwise returns the INPUT
        object as ``(messages, 0)``. The rearm gate is measured on message bodies only, so it is
        bypassed (never the reclaim gate) when a provider-billed ``current_tokens`` reading already
        puts the request over ``threshold_tokens`` (#101889); every no-op taken while over threshold
        is logged once per distinct reason.

        ``_prune_old_tool_results`` runs all deterministic passes: (1) dedup byte-identical tool results —
        keeps the newest full copy and back-references older exact duplicates ANYWHERE in the list
        (including the protected tail), so no unique content is ever lost; (2) summarize non-tail tool
        results larger than ``min_prune_chars``; (3) truncate oversized tool_call arguments on non-tail
        assistant messages; (3.5) retire image payloads on all but the newest ``_MAX_KEEP_TOOL_IMAGES``
        image-bearing tool results, except a pending round that fits the hard share (#92699). Only pass (2)'s floor is
        raised by ``proactive_prune_min_result_chars``; passes (1) and (3) keep their own fixed floors. The
        recent-tail protection applies to passes (2) and (3); pass (1) is tail-agnostic by design because
        dedup is lossless.
        """
        if self.proactive_prune_tokens <= 0 or (
            current_tokens is not None and current_tokens < self.proactive_prune_tokens
        ):
            return messages, 0
        if len(messages) <= self.protect_last_n + self._protect_head_size(messages) + 1:
            self._warn_reclamation_no_op("prune:tail_only", current_tokens)
            return messages, 0
        _estimate_msg_budget_tokens = _cc()._estimate_msg_budget_tokens
        before = sum(_estimate_msg_budget_tokens(m) for m in messages)
        # Under-threshold runway skip is ordinary hysteresis (silent); above it the lockout is the bug.
        if before < self._proactive_prune_rearm_tokens and not self._billed_basis_over_threshold(current_tokens):
            return messages, 0
        # Capability gate first: a store without archive_and_compact makes every prune a no-op.
        session_db = getattr(self, "_session_db", None)
        session_id = getattr(self, "_session_id", "")
        if session_db and session_id and not callable(getattr(session_db, "archive_and_compact", None)):
            self._warn_reclamation_no_op("prune:store_cannot_persist", current_tokens)
            return messages, 0
        pruned_msgs, pruned_count = self._prune_old_tool_results(
            messages, protect_tail_count=self.protect_last_n, protect_tail_tokens=None, min_prune_chars=self.proactive_prune_min_result_chars,
        )
        if not pruned_count:
            # No-op contract: return the INPUT object so callers can gate on `result is not input`.
            self._warn_reclamation_no_op("prune:nothing_eligible", current_tokens)
            return messages, 0
        # Prompt-cache hysteresis: commit only when the reclaim is meaningful.
        after = sum(_estimate_msg_budget_tokens(m) for m in pruned_msgs)
        reclaimed = max(0, before - after)
        if reclaimed < self.proactive_prune_min_reclaim_tokens:
            self._warn_reclamation_no_op("prune:reclaim_below_minimum", current_tokens, before=before)
            return messages, 0
        # Require a full trigger-sized regrowth before the next cache-breaking rewrite.
        runway = max(reclaimed, self.proactive_prune_tokens, self.proactive_prune_min_reclaim_tokens)
        next_rearm_tokens = after + runway
        if session_db and session_id:
            _cc_ = _cc()
            from hermes_state_errors import HeldRowRewrittenError
            model_config_patch = {_cc_.PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY: next_rearm_tokens}
            StaleHeldHistory = _cc_.StaleHeldHistory
            _DB_PERSISTED_MARKER = _cc_._DB_PERSISTED_MARKER
            rewritten = None
            try:
                # Raises StaleHeldHistory when another compaction already won, for both commit paths.
                watermark = _cc_._archive_watermark_for(session_db, session_id, messages)
                rewritten = self._rewrite_pruned_rows_in_place(
                    session_db, session_id, messages, pruned_msgs, model_config_patch)
                if rewritten is None:
                    from agent.conversation_compression_archive import coverage_for_commit
                    covered_ids, unresolved_held = coverage_for_commit(session_db, session_id, messages)
                    session_db.archive_and_compact(
                        session_id, pruned_msgs,
                        model_config_patch=model_config_patch,
                        watermark=watermark,
                        covered_ids=covered_ids, unresolved_held=unresolved_held,
                        held_for_cas=messages)
            except StaleHeldHistory:
                # Another compaction already committed this session's history; a lease-less prune of the
                # generation this process holds would publish beside the winner. Leave the input alone.
                logger.info("Proactive tool-result prune skipped: another compaction already committed this session")
                self._warn_reclamation_no_op("prune:stale_generation", current_tokens, before=before)
                return messages, 0
            except HeldRowRewrittenError:
                # The full writer's in-transaction CAS fence tripped: a held row changed between the
                # precheck and the commit. Same disposition as StaleHeldHistory — nothing was written.
                logger.info("Proactive tool-result prune skipped: a held row was rewritten in place before the commit")
                self._warn_reclamation_no_op("prune:stale_generation", current_tokens, before=before)
                return messages, 0
            except Exception as exc:  # health: allow BLE001 -- the commit boundary keeps the original transcript on any store failure (#124102)
                logger.warning("Proactive tool-result prune DB commit failed; keeping the original transcript: %s", exc)
                return messages, 0
            # Shared post-commit stamp site with the in-place commit and micro-compaction sync.
            # See #98450. The in-place rewrite wrote no new rows, so only already-durable messages
            # carry the marker; unwritten ones reach the DB through the next flush.
            stamp_db_persisted_markers = _cc_.stamp_db_persisted_markers
            if rewritten is None:
                stamp_db_persisted_markers(pruned_msgs)
            else:
                stamp_db_persisted_markers([
                    replacement for original, replacement in zip(messages, pruned_msgs)
                    if original.get(_DB_PERSISTED_MARKER)])
        self._proactive_prune_rearm_tokens = next_rearm_tokens
        # Reclamation just ran: let a future lockout warn again.
        self._last_reclaim_block_warn = None
        return pruned_msgs, pruned_count

    def _rewrite_pruned_rows_in_place(
        self, session_db: Any, session_id: str, messages: list[dict[str, Any]],
        pruned_msgs: list[dict[str, Any]], model_config_patch: dict[str, Any],
    ) -> Optional[int]:
        """Commit a prune by rewriting only its changed rows (#124102); ``None`` means use the full rewrite.

        A prune edits the content of a few existing messages and never adds, drops or reorders one, so the
        durable rows it touched can be updated where they are. That needs a store with
        ``rewrite_pruned_rows`` and unwritten messages only at the end (the next flush appends those in
        order; one sitting before durable rows would land out of order). A row the store cannot name
        exactly also falls back: ``archive_and_compact`` handles every shape. A row it *can* name whose
        stored body changed is not ambiguity but a lost race, and raises ``StaleHeldHistory`` so the
        caller skips the pass rather than republishing the held generation.
        """
        rewrite = getattr(session_db, "rewrite_pruned_rows", None)
        if not callable(rewrite) or len(messages) != len(pruned_msgs):
            return None
        from hermes_state_errors import PruneRowStaleError, PruneRowUnresolvedError

        _DB_PERSISTED_MARKER = _cc()._DB_PERSISTED_MARKER
        ignored = {_DB_PERSISTED_MARKER, "_row_id"}
        changes: list[tuple] = []
        unwritten_seen = False
        for original, replacement in zip(messages, pruned_msgs):
            if not isinstance(original, dict) or not isinstance(replacement, dict):
                return None
            if not original.get(_DB_PERSISTED_MARKER):
                unwritten_seen = True
                continue
            if unwritten_seen:
                return None
            if ({k: v for k, v in original.items() if k not in ignored}
                    != {k: v for k, v in replacement.items() if k not in ignored}):
                changes.append((original, replacement))
        try:
            return rewrite(session_id, changes, model_config_patch=model_config_patch)
        except PruneRowStaleError as exc:
            # A named row whose stored body changed is another writer's generation, which is exactly
            # what StaleHeldHistory exists to catch — the in-place prune preserved the row ids, so
            # _archive_watermark_for could not see it. Falling back would hand this stale transcript to
            # archive_and_compact, which archives rows it never compared and republishes the held text
            # over the newer one. Fail closed; the caller skips the pass.
            raise _cc().StaleHeldHistory(str(exc)) from exc
        except PruneRowUnresolvedError as exc:
            logger.info("Proactive prune: in-place rewrite refused (%s); using the full rewrite", exc)
            return None
