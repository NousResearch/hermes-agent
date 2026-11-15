"""The proactive tool-result prune: deterministic passes plus the in-place row commit (#124102).

Split out of :mod:`agent.context_compressor` (FILE_LINES ratchet): the prune is a self-contained
reclamation feature. The origin module keeps the ``ContextCompressor`` class; this sibling owns
``prune_tool_results_only`` and its in-place commit helper. Log name parity with the origin
module, and every origin-module symbol is read through ``_cc()`` at call time so tests that patch
``agent.context_compressor.X`` stay effective.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger("agent.context_compressor")
# Reclaim gate on an already-expired provider cache: there is no cache break to amortize.
_COLD_CACHE_MIN_RECLAIM_TOKENS = 16_000


def _cc():
    """The origin module, resolved lazily: avoids the import cycle and keeps tests that patch
    ``agent.context_compressor.X`` effective (attributes are read at call time)."""
    from agent import context_compressor
    return context_compressor


def terminal_stub(args: dict, content: str, line_count: int) -> str:
    """One-line terminal stub. Lines are counted on the decoded ``output`` (the JSON envelope has no
    literal newlines), and a failed command keeps its last non-empty output line. The whole stub stays
    under ``_PRUNE_MIN_CHARS``, so a later pass never takes it for a result to summarise again."""
    cmd = " ".join(_cc()._str_arg(args, "command").splitlines())
    cmd = cmd if len(cmd) <= 80 else cmd[:77] + "..."
    exit_code = m.group(1) if (m := re.search(r'"exit_code"\s*:\s*(-?\d+)', content)) else "?"
    output = _cc()._json_dict(content).get("output")
    lines = output.splitlines() if isinstance(output, str) else []
    if isinstance(output, str):
        line_count = len(lines)
    stub = f"[terminal] ran `{cmd}` -> exit {exit_code}, {line_count} {'line' if line_count == 1 else 'lines'} output"
    last = next((line.strip() for line in reversed(lines) if line.strip()), "")
    room = min(120, _cc()._PRUNE_MIN_CHARS - 1 - len(stub) - len(" | last: "))
    if exit_code not in ("?", "0") and last and room >= 20:
        stub += f" | last: {last[:room]}"
    return stub


def delegate_stub(args: dict, content: str, content_len: int) -> str:
    tasks = args.get("tasks")
    batch = isinstance(tasks, list) and bool(tasks)  # the tool runs the batch whenever tasks is non-empty
    first = tasks[0] if batch and isinstance(tasks[0], dict) else {}
    goal = " ".join(_cc()._str_arg(first if batch else args, "goal").splitlines())
    goal = goal if len(goal) <= 60 else goal[:57] + "..."
    detail = f"{len(tasks)} tasks, first: '{goal}'" if batch else f"'{goal}'"
    return f"[delegate_task] {detail} ({content_len:,} chars result)" + _cc()._result_failure_suffix(content)


def stub_arguments(args: dict) -> str:
    result = ""
    for key, value in args.items():
        preview = " ".join(str(value).splitlines())[:40]
        result += f" {key}={preview}"
        if len(result) >= 120:
            return result[:120]
    return result


def _changed_rows(messages: list[dict[str, Any]], pruned: list[dict[str, Any]]) -> int:
    return sum(original is not replacement and original != replacement for original, replacement in zip(messages, pruned))


def _keep_running_turn(
    messages: list[dict[str, Any]], pruned: list[dict[str, Any]], turn_start: int | None,
) -> tuple[list[dict[str, Any]], int] | None:
    """*pruned* with the turn in progress (from its user row at *turn_start*) put back whole, and how many
    shortened rows that restored; ``None`` when *turn_start* names no user row (skip this round).

    The post-tool site runs between tool rounds of a live turn; a count-only tail let a long tool loop
    stub its own earlier calls, so the model re-ran them or answered blind. ``None`` means the list holds
    no running turn (the turn-start site prunes only the history before it). An index that is not a user
    row (``reanchor_current_turn_user_idx`` answers -1 when no user row survived a compaction) cannot say
    where the running turn starts, so the round fails closed instead of falling back to the count."""
    if turn_start is None:
        return pruned, 0
    if not (isinstance(turn_start, int) and 0 <= turn_start < len(messages)
            and isinstance(messages[turn_start], dict) and messages[turn_start].get("role") == "user"):
        return None
    return [*pruned[:turn_start], *messages[turn_start:]], _changed_rows(messages[turn_start:], pruned[turn_start:])


class ProactivePruneMixin:
    """``ContextCompressor`` mixin: the deterministic, no-LLM tool-result prune and its commit."""

    # The host skips this prune for a session with no store row (batch runs): their originals live nowhere else.
    prune_needs_session_store = True

    def would_proactively_prune(self, current_tokens: int | None = None) -> bool:
        """Configured on AND above its trigger: the two state-free preconditions
        ``prune_tool_results_only`` checks first, published so the host can tell a withheld prune
        from a no-op. Not a prediction that a prune would commit (tail, runway and reclaim gate
        need the scan)."""
        return self.proactive_prune_tokens > 0 and (
            current_tokens is None or current_tokens >= self.proactive_prune_tokens
        )

    def prune_tool_results_only(
        self, messages: list[dict[str, Any]], current_tokens: int | None = None, *, cold_cache: bool = False,
        turn_start: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Deterministic, no-LLM tool-result prune: on an already-expired provider cache (``cold_cache``,
        on by default), else gated on ``proactive_prune_tokens`` (off by default).

        ``turn_start`` indexes the user row of the turn in progress: nothing from it on is shortened, and an
        index that names no user row skips the round. The protected tail never starts inside one assistant
        step's parallel results (its boundary moves back to the step); each older result is selected on its
        own. Every result that is shortened has its full original written to ``cache/pruned/`` first and its stub names the file; a
        result whose write fails is left whole, so no committed prune loses an original (unflushed rows
        included: the store never needs to have seen the body).

        Protects the tail by message COUNT only. A warm commit breaks the prompt cache, so it requires
        ``proactive_prune_min_reclaim_tokens`` (less what cold prunes freed since the last warm one) and a
        full regrowth runway; otherwise returns the INPUT object as ``(messages, 0)``. ``cold_cache`` (the
        provider cache has already expired) has nothing to break: no token trigger, the gate is
        ``_COLD_CACHE_MIN_RECLAIM_TOKENS`` (lower only when the warm prune is on with a lower gate) and the
        runway is neither checked nor moved. The rearm gate is
        measured on message bodies only, so it is bypassed (never the reclaim gate) when a provider-billed ``current_tokens`` reading already
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
        # The cold path needs no trigger: an already-expired cache has no prefix left to break.
        if not (cold_cache or self.would_proactively_prune(current_tokens)):
            return messages, 0
        if len(messages) <= self.protect_last_n + self._protect_head_size(messages) + 1:
            self._warn_reclamation_no_op("prune:tail_only", current_tokens)
            return messages, 0
        _estimate_msg_budget_tokens = _cc()._estimate_msg_budget_tokens
        before = sum(_estimate_msg_budget_tokens(m) for m in messages)
        # Under-threshold runway skip is ordinary hysteresis (silent); above it the lockout is the bug.
        if (not cold_cache and before < self._proactive_prune_rearm_tokens
                and not self._billed_basis_over_threshold(current_tokens)):
            return messages, 0
        # Capability gate first: a store that cannot archive (no archive_and_compact, or no bound row) loses the originals.
        session_db, session_id = getattr(self, "_session_db", None), getattr(self, "_session_id", "")
        if session_db and not (session_id and callable(getattr(session_db, "archive_and_compact", None))):
            self._warn_reclamation_no_op("prune:store_cannot_persist", current_tokens)
            return messages, 0
        # The tail never starts inside one step's parallel results (a count-only tail split the step).
        # Only a boundary ON a tool row splits one; a group that ends right before it is already whole.
        boundary = len(messages) - self.protect_last_n
        if 0 < boundary < len(messages) and messages[boundary].get("role") == "tool":
            boundary = self._align_boundary_backward(messages, boundary)
        protect_tail = len(messages) - boundary
        pruned_msgs, pruned_count = self._prune_old_tool_results(
            messages, protect_tail_count=protect_tail, protect_tail_tokens=None, min_prune_chars=self.proactive_prune_min_result_chars,
        )
        kept = _keep_running_turn(messages, pruned_msgs, turn_start)
        if kept is None:
            self._warn_reclamation_no_op("prune:running_turn_unknown", current_tokens)
            return messages, 0
        pruned_msgs, restored = kept
        pruned_count -= restored
        if not pruned_count:
            # No-op contract: return the INPUT object so callers can gate on `result is not input`.
            self._warn_reclamation_no_op("prune:nothing_eligible", current_tokens)
            return messages, 0
        # Prompt-cache hysteresis: commit only when the reclaim is meaningful.
        gate = self.proactive_prune_min_reclaim_tokens
        # The warm gate lowers the cold one only when the warm prune is on (it is off by default).
        cold_gate = min(gate, _COLD_CACHE_MIN_RECLAIM_TOKENS) if self.proactive_prune_tokens > 0 else _COLD_CACHE_MIN_RECLAIM_TOKENS
        min_reclaim = cold_gate if cold_cache else max(cold_gate, gate - self._proactive_prune_cold_credit)
        if before - sum(_estimate_msg_budget_tokens(m) for m in pruned_msgs) < min_reclaim:
            self._warn_reclamation_no_op("prune:reclaim_below_minimum", current_tokens, before=before)
            return messages, 0
        # No information loss: every shortened result's original goes to disk first and its stub names the
        # file; a result whose write fails stays whole, so the gate is re-checked on what is left.
        from agent.context_compressor_prune_archive import archive_pruned_originals, discard_archives
        from agent.context_compressor_tail import with_pruned_row_skill_safety_rule
        pruned_msgs, archives, unarchived = archive_pruned_originals(messages, pruned_msgs, session_id)
        pruned_msgs = with_pruned_row_skill_safety_rule(messages, pruned_msgs)
        pruned_count -= unarchived
        after = sum(_estimate_msg_budget_tokens(m) for m in pruned_msgs)
        reclaimed = max(0, before - after)
        if not pruned_count or reclaimed < min_reclaim:
            discard_archives(archives)
            self._warn_reclamation_no_op("prune:archive_write_failed" if unarchived else "prune:reclaim_below_minimum", current_tokens, before=before)
            return messages, 0
        if cold_cache:  # Nothing to protect: keep the warm runway and credit the reclaim to the next warm gate.
            next_rearm_tokens, cold_credit = self._proactive_prune_rearm_tokens, self._proactive_prune_cold_credit + reclaimed
        else:  # Require a full trigger-sized regrowth before the next cache-breaking rewrite.
            next_rearm_tokens, cold_credit = after + max(reclaimed, self.proactive_prune_tokens, gate), 0
        if session_db and session_id:
            _cc_ = _cc()
            from hermes_state_errors import HeldRowRewrittenError
            model_config_patch = {_cc_.PROACTIVE_PRUNE_REARM_MODEL_CONFIG_KEY: next_rearm_tokens,
                                  _cc_.PROACTIVE_PRUNE_COLD_CREDIT_MODEL_CONFIG_KEY: cold_credit}
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
                discard_archives(archives)
                self._warn_reclamation_no_op("prune:stale_generation", current_tokens, before=before)
                return messages, 0
            except HeldRowRewrittenError:
                # The full writer's in-transaction CAS fence tripped: a held row changed between the
                # precheck and the commit. Same disposition as StaleHeldHistory — nothing was written.
                logger.info("Proactive tool-result prune skipped: a held row was rewritten in place before the commit")
                discard_archives(archives)
                self._warn_reclamation_no_op("prune:stale_generation", current_tokens, before=before)
                return messages, 0
            except Exception as exc:  # health: allow BLE001 -- the commit boundary keeps the original transcript on any store failure (#124102)
                # Uncertain outcome (a worker may have committed and lost the reply): the transcript may already
                # point at these archives, so they stay; the reference-aware retention sweep reclaims unused ones.
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
        self._proactive_prune_rearm_tokens, self._proactive_prune_cold_credit = next_rearm_tokens, cold_credit
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
