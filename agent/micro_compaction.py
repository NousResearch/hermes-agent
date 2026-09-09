"""Micro-compaction mixin for ContextCompressor.

Rolling per-exchange summarization that folds old user/assistant exchanges into a single
summary marker between turns. OFF by default: every pass rewrites the prompt prefix and
breaks the provider prompt cache.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from agent.model_metadata import estimate_messages_tokens_rough, estimate_tokens_rough

# Log name parity with the origin module.
logger = logging.getLogger("agent.context_compressor")


def _cc():
    """The origin module, resolved lazily: avoids the import cycle and keeps tests that patch
    ``agent.context_compressor.X`` effective (attributes are read at call time)."""
    from agent import context_compressor
    return context_compressor


def _is_summary_marker(entry: Any) -> bool:
    return isinstance(entry, dict) and bool(entry.get(_cc().COMPRESSED_SUMMARY_METADATA_KEY))


def _is_micro_marker(entry: Any) -> bool:
    """True for a summary marker provably absorbed into the rolling summary (micro, not batch)."""
    return _is_summary_marker(entry) and bool(entry.get(_cc().MICRO_COMPACT_MARKER_KEY))


class MicroCompactionMixin:
    """Rolling micro-compaction; host must be a ``ContextCompressor``."""

    def _resolve_compact_cursor(self, messages: List[Dict[str, Any]], head_end: int, tail_start: int) -> int:
        """Index of the first message not yet absorbed into the rolling summary: the in-memory
        cursor when valid, else just past the last summary marker."""
        if head_end < self._micro_compact_cursor < tail_start:
            return self._micro_compact_cursor
        summaries = (i for i in range(head_end, tail_start) if self._is_context_summary_message(messages[i]))
        last = max(summaries, default=-1)
        cursor = last + 1 if last >= head_end else head_end
        if last >= head_end:
            # Resumed session: rehydrate the rolling summary from the surviving marker so the next
            # pass merges, not replaces.
            recovered = "" if self._micro_compact_rolling_summary.strip() else (
                self._rolling_summary_from_marker(messages[last].get("content"))
            )
            if recovered:
                self._micro_compact_rolling_summary = recovered
                # Rehydration proves containment: this marker (batch or micro) becomes
                # supersede/defrag-eligible; unabsorbed markers never get the key.
                messages[last][_cc().MICRO_COMPACT_MARKER_KEY] = True
                logger.info("Micro-compaction: recovered rolling summary from transcript (%d chars)", len(recovered))
        self._micro_compact_cursor = cursor
        return cursor

    def _find_one_exchange(
        self, messages: List[Dict[str, Any]], start: int, tail_start: int,
    ) -> Optional[tuple[int, int]]:
        """Find the next complete exchange (full agent turn) starting at *start*; returns
        ``(exchange_start, exchange_end)`` or ``None``. Spans assistant+tool rows up to the next
        user message; user turns are never absorbed (alternation safety, verbatim user text)."""
        limit = min(tail_start, len(messages))

        def _turn_row(idx: int, roles: tuple) -> bool:
            return messages[idx].get("role") in roles and not self._is_context_summary_message(messages[idx])

        # Skip user messages and (assistant-role) summary markers to reach a real assistant message;
        # otherwise a rehydrated cursor could absorb the marker itself.
        idx = start
        while idx < limit and not _turn_row(idx, ("assistant",)):
            idx += 1
        if idx >= limit:
            return None
        exchange_start = idx
        idx += 1
        while idx < limit and _turn_row(idx, ("assistant", "tool")):
            idx += 1

        # Boundary must close the turn: a mid-turn stop at tail_start would put the assistant marker
        # beside assistant/tool rows. Any other role is a safe splice (avoids wedging the cursor).
        boundary = messages[idx] if idx < len(messages) else None
        if not isinstance(boundary, dict) or boundary.get("role") in ("assistant", "tool"):
            return None
        return (exchange_start, idx)

    def _build_micro_summary_prompt(self, existing_summary: str, exchange_text: str) -> List[Dict[str, str]]:
        """Build the prompt messages for a single-exchange micro-summary."""
        summary_block = existing_summary if existing_summary.strip() else "(No previous summary yet.)"
        user_prompt = (
            "You are a summarization agent creating a compact record of an "
            "ongoing conversation.  You are given a running summary and the "
            "next exchange from the conversation.  Merge the exchange's key "
            "decisions, requirements, file paths, and open questions into the "
            "summary.  Preserve the summary's structure.  Drop resolved details "
            "that are no longer relevant.  Add new decisions, file paths, and "
            "open questions.\n\n"
            "NEVER include API keys, tokens, passwords, secrets, credentials, "
            "or connection strings in the summary \u2014 replace any that appear "
            f"with [REDACTED].\n\n"
            f"## Current Running Summary\n{summary_block}\n\n"
            f"## Next Exchange to Merge\n{exchange_text}\n\n"
            "Return ONLY the updated summary text, no preamble or explanation. "
            "Do not include this instruction block in your output."
        )
        return [
            {"role": "system", "content": "You are a conversation summarization assistant."},
            {"role": "user", "content": user_prompt},
        ]

    def _micro_summarize_one(self, exchange_text: str) -> Optional[str]:
        """Micro-summarize one exchange into the rolling summary via the aux LLM (None on failure)."""
        from agent.auxiliary_client import aux_interrupt_protection, call_llm

        call_kwargs = {
            "task": "compression",
            "messages": self._build_micro_summary_prompt(self._micro_compact_rolling_summary, exchange_text),
            "max_tokens": min(1500, self.max_summary_tokens or 1500),
            "temperature": 0.1,
        }
        if self.summary_model:
            call_kwargs["model"] = self.summary_model
        if self.model:
            call_kwargs.setdefault("main_runtime", {
                "model": self.model, "provider": self.provider or "", "base_url": self.base_url or "",
                "api_key": self.api_key or "", "api_mode": getattr(self, "api_mode", "") or "",
            })

        try:
            with aux_interrupt_protection():
                response = call_llm(**call_kwargs)
        except Exception as exc:
            logger.info("micro-summarization call failed: %s", exc)
            return None

        # A length stop means a partial merge; leave the exchange unabsorbed so a later pass retries.
        if _cc()._response_finish_reason(response) == "length":
            logger.warning(
                "micro-summarization output hit the token cap (finish_reason=length) — discarding partial summary",
            )
            return None

        message = response.choices[0].message
        content = message.get("content") if isinstance(message, dict) else getattr(message, "content", message)
        content = (content if isinstance(content, str) else str(content) if content else "").strip()
        if not content:
            logger.info("micro-summarization returned empty content")
            return None

        from agent.agent_runtime_helpers import strip_think_blocks
        return strip_think_blocks(None, content).strip() or None

    def _needs_defrag(self) -> bool:
        """Return True when the rolling summary is large enough to defrag."""
        return estimate_tokens_rough(self._micro_compact_rolling_summary) >= self._micro_compact_defrag_threshold_tokens

    def _defrag_rolling_summary(self, messages: List[Dict[str, Any]]) -> bool:
        """Re-summarize the rolling summary text and rewrite the marker in place.
        Transcript-shape-neutral (no splice, no cursor move). Returns True when it rewrote."""
        old_summary = self._micro_compact_rolling_summary
        if not old_summary.strip():
            return False
        # Empty base turns the merge prompt into a rewrite-compactly instruction.
        self._micro_compact_rolling_summary = ""
        fresh_summary = self._micro_summarize_one(old_summary)
        self._micro_compact_rolling_summary = fresh_summary or old_summary
        if not fresh_summary:
            return False
        # Rewrite only the newest MICRO marker (resume rehydrates from it); a batch marker holds
        # history we lack.
        entry = next((e for e in reversed(messages) if _is_micro_marker(e)), None)
        if entry is not None:
            entry["content"] = self._render_micro_marker_content(fresh_summary)
            # Content changed: clear the persisted stamp so the DB sync rewrites the row. An
            # in-place pop on a live dict would be identity-skipped by the bounded flush scan;
            # flag the finalizer.
            entry.pop(_cc()._DB_PERSISTED_MARKER, None)
            # Sibling of the finalize_turn pop site (#75170): this pop also strips the marker from a LIVE
            # dict in place, so the bounded flush-scan cursor would identity-skip the rewritten marker and
            # the defragged summary would never reach state.db. The compressor holds no agent reference, so
            # raise a flag the finalizer consumes to invalidate agent._db_flush_scan_prefix. (The pop sites
            # at module scope — fresh copies in strip-marker helpers — break identity and need no flag.)
            self._flush_scan_cursor_invalidated = True
        logger.info(
            "Micro-compaction defrag: rolling summary re-summarized (%d -> %d chars)",
            len(old_summary), len(fresh_summary),
        )
        return True

    def _reset_micro_failure_tracking(self) -> None:
        self._micro_compact_consecutive_failures = 0
        self._micro_compact_last_failure_cursor = -1

    def _micro_compact(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run one round of micro-compaction (entry point from ``finalize_turn()``). Returns the
        (possibly modified) list and syncs the session DB via ``archive_and_compact`` (the
        append-only flush alone would double-load on resume)."""
        if not self._micro_compact_enabled:
            return messages

        # Cadence gate: each pass breaks prompt cache once. Counted per invocation so a no-op turn
        # can't wedge it.
        every_n = max(1, int(self._micro_compact_every_n_turns or 1))
        if every_n > 1:
            self._micro_compact_turns_since_pass += 1
            if self._micro_compact_turns_since_pass < every_n:
                return messages
            self._micro_compact_turns_since_pass = 0

        n_messages = len(messages)
        exchange = self._next_exchange(messages) if n_messages >= 4 else None
        if exchange is None:
            return messages
        exchange_start, exchange_end = exchange

        # Telemetry baseline; taken only once an exchange exists so no-op turns don't pay.
        _started_at = time.monotonic()
        _tokens_before = estimate_messages_tokens_rough(messages)

        def _telemetry(outcome: str, result: List[Dict[str, Any]], **extra: Any) -> None:
            self._emit_micro_compaction_telemetry(
                outcome=outcome, messages_before=n_messages, messages_after=len(result),
                tokens_before=_tokens_before, duration_ms=int((time.monotonic() - _started_at) * 1000), **extra,
            )

        # Defrag rewrites summary text/marker in place (no splice, no cursor move) instead of
        # absorbing this turn.
        if self._needs_defrag():
            defragged = self._defrag_rolling_summary(messages)
            if defragged:
                self._sync_micro_compact_to_db(messages)
                self._reset_micro_failure_tracking()
            outcome = "defrag" if defragged else "defrag_failed"
            _telemetry(outcome, messages, tokens_after=estimate_messages_tokens_rough(messages))
            return messages

        # Cumulative iff it subsumes an earlier marker; captured before summarizing.
        _cumulative = bool(self._micro_compact_rolling_summary.strip())

        exchange_text = self._serialize_for_summary(messages[exchange_start:exchange_end])
        _exchange_tokens = estimate_tokens_rough(exchange_text)
        updated_summary = self._micro_summarize_one(exchange_text)
        if updated_summary is None:
            _outcome = self._record_micro_failure(exchange_start, exchange_end)
            _telemetry(_outcome, messages, tokens_after=_tokens_before, exchange_tokens=_exchange_tokens)
            return messages

        self._micro_compact_rolling_summary = updated_summary
        self._micro_compact_cursor = exchange_end
        self._reset_micro_failure_tracking()

        result = self._splice_micro_compact_result(messages, exchange_start, exchange_end, supersede=_cumulative)
        self._micro_compact_cursor = self._cursor_after_splice(result, exchange_start + 1)
        self._sync_micro_compact_to_db(result)
        _telemetry(
            "absorbed", result, tokens_after=estimate_messages_tokens_rough(result), exchange_tokens=_exchange_tokens,
        )
        return result

    def _record_micro_failure(self, exchange_start: int, exchange_end: int) -> str:
        """Count a summarize failure at this cursor; skip the exchange after too many in a row."""
        # Track consecutive failures at the same cursor to avoid busy-looping every turn.
        same_cursor = exchange_start == self._micro_compact_last_failure_cursor
        self._micro_compact_consecutive_failures = self._micro_compact_consecutive_failures + 1 if same_cursor else 1
        self._micro_compact_last_failure_cursor = exchange_start
        if self._micro_compact_consecutive_failures < _cc()._MICRO_COMPACT_MAX_CONSECUTIVE_FAILURES:
            return "summarize_failed"
        logger.info(
            "Micro-compaction: skipping exchange at cursor %d after %d consecutive failures",
            exchange_start, self._micro_compact_consecutive_failures,
        )
        # Skip the stuck exchange; it stays in the transcript for batch compression/defrag.
        self._micro_compact_cursor = exchange_end
        self._reset_micro_failure_tracking()
        return "exchange_skipped"

    def _next_exchange(self, messages: List[Dict[str, Any]]) -> Optional[tuple[int, int]]:
        """The next un-absorbed exchange inside the compressible window, or None."""
        compress_start = self._align_boundary_forward(messages, self._protect_head_size(messages))
        compress_end = self._find_tail_cut_by_tokens(messages, compress_start)
        if compress_start >= compress_end:
            return None
        cursor = self._resolve_compact_cursor(messages, compress_start, compress_end)
        return None if cursor >= compress_end else self._find_one_exchange(messages, cursor, compress_end)

    @staticmethod
    def _rolling_summary_from_marker(content: Any) -> str:
        """Recover the rolling-summary text from a summary marker (resume rehydration)."""
        cc = _cc()
        if not isinstance(content, str) or not content.strip():
            return ""
        # rfind: SUMMARY_PREFIX itself mentions the heading, so the first hit is in the preamble.
        idx = content.rfind(cc.HISTORICAL_TASK_HEADING)
        body = content[idx + len(cc.HISTORICAL_TASK_HEADING):] if idx != -1 else content
        end = body.find(cc._SUMMARY_END_MARKER)
        return (body[:end] if end != -1 else body).strip()

    def _cursor_after_splice(self, result: List[Dict[str, Any]], fallback: int) -> int:
        """Cursor position just past the summary marker in *result*. Must derive from the SPLICED
        list: a splice collapses several rows into one marker (and may drop a superseded one), so
        pre-splice indices land inside a later exchange and silently skip it."""
        return next((idx + 1 for idx in range(len(result) - 1, -1, -1) if _is_summary_marker(result[idx])), fallback)

    def _emit_micro_compaction_telemetry(
        self, *, outcome: str, messages_before: int, messages_after: int, tokens_before: int | None,
        tokens_after: int | None, exchange_tokens: int | None = None, duration_ms: int | None = None,
    ) -> None:
        """Emit one content-free JSON log line for a micro-compaction pass.
        ``tokens_delta`` < 0 means the pass shrank the transcript; ``*_total`` fields accumulate."""
        _safe_int = _cc()._safe_int
        try:
            delta = tokens_after - tokens_before if tokens_before is not None and tokens_after is not None else None
            self._micro_compact_tokens_saved_total -= delta or 0
            self._micro_compact_passes += 1
            # Cached reads only: the lazy properties can fire a synchronous /models probe.
            # The ``threshold_tokens`` / ``context_length`` properties resolve lazily and can fire a
            # synchronous /models probe on first access (#32221) — telemetry must never be the thing that
            # blocks a turn. Unresolved simply reports null.
            threshold = self._threshold_tokens
            has_occupancy = threshold and tokens_after is not None and threshold > 0
            occupancy = round(tokens_after / threshold * 100, 1) if has_occupancy else None
            payload = {
                "event": "micro_compaction", "session_id": getattr(self, "_session_id", "") or "", "outcome": outcome,
                "messages_before": messages_before, "messages_after": messages_after,
                "tokens_before": _safe_int(tokens_before), "tokens_after": _safe_int(tokens_after),
                "tokens_delta": _safe_int(delta), "exchange_tokens": _safe_int(exchange_tokens),
                "rolling_summary_tokens": estimate_tokens_rough(self._micro_compact_rolling_summary),
                "cursor": _safe_int(self._micro_compact_cursor), "passes_total": self._micro_compact_passes,
                "tokens_saved_total": self._micro_compact_tokens_saved_total, "duration_ms": _safe_int(duration_ms),
                # Headroom: how full the window is being kept.
                "threshold_tokens": _safe_int(threshold), "context_limit": _safe_int(self._resolved_context_length),
                "occupancy_pct": occupancy, "main_model": self.model or "", "aux_model": self.summary_model or "",
            }
            logger.info("micro compaction telemetry: %s", json.dumps(payload, sort_keys=True, separators=(",", ":")))
        except Exception as exc:
            logger.debug("failed to emit micro-compaction telemetry: %s", exc)

    def _sync_micro_compact_to_db(self, compacted_messages: List[Dict[str, Any]]) -> None:
        """Persist the micro-compacted set to the session DB atomically and stamp rows persisted.
        Without this the old exchange rows stay ``active=1`` and a resume double-loads both the
        summary and the originals."""
        session_db, session_id = getattr(self, "_session_db", None), getattr(self, "_session_id", "")
        if not session_db or not session_id:
            return
        try:
            # Every row except the marker is a carried-forward original: archive rewind-style.
            session_db.archive_and_compact(session_id, compacted_messages, tail_count=max(0, len(compacted_messages) - 1))
            # Shared post-commit stamp site with batch commit and proactive prune.
            # See #98450.
            _cc().stamp_db_persisted_markers(compacted_messages)
        except Exception:
            logger.info(
                "Micro-compaction DB sync failed — resume will double-load "
                "compacted messages until the next batch compression"
            )

    def _splice_micro_compact_result(
        self, messages: List[Dict[str, Any]], splice_start: int, splice_end: int, supersede: bool = True,
    ) -> List[Dict[str, Any]]:
        """Replace *messages[splice_start:splice_end]* with an assistant-role summary marker.
        Merges user turns left adjacent by a superseded marker so the result is alternation-valid."""
        cc = _cc()
        summary_text = self._micro_compact_rolling_summary
        if not summary_text.strip():
            return messages

        summary_msg = {
            "role": "assistant", "content": self._render_micro_marker_content(summary_text),
            cc.COMPRESSED_SUMMARY_METADATA_KEY: True,
            # Micro marker: eligible for supersede/defrag; batch markers never carry this key.
            cc.MICRO_COMPACT_MARKER_KEY: True,
            # Micro markers absorb only assistant/tool content; user turns stay in the transcript.
            cc.COMPRESSED_SUMMARY_HAS_USER_TURN_KEY: False,
        }
        result = messages[:splice_start] + [summary_msg] + messages[splice_end:]

        # Cumulative summary: keep only the newest marker. Drop an older one only if supersede AND
        # it has MICRO_COMPACT_MARKER_KEY (provably absorbed); a batch marker holds MORE history.
        stale = [i for i, m in enumerate(result) if _is_micro_marker(m)][:-1] if supersede else []
        if stale:
            result = self._merge_adjacent_user_turns([m for i, m in enumerate(result) if i not in stale])

        # Deliberately no _strip_persistence_markers: micro archives in place under the same session
        # id, so stamps stay accurate and a failed archive keeps the append-only flush idempotent.
        return result

    @staticmethod
    def _render_micro_marker_content(summary_text: str) -> str:
        """Assemble the marker content wrapper around *summary_text*."""
        cc = _cc()
        return f"{cc.SUMMARY_PREFIX}\n\n{cc.HISTORICAL_TASK_HEADING}\n{summary_text.strip()}\n\n{cc._SUMMARY_END_MARKER}"

    @staticmethod
    def _merge_adjacent_user_turns(result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge consecutive plain-text real user turns left by a supersede. Same ``\\n\\n`` join as
        ``repair_message_sequence`` pass 2, done here so the marker and cursor are never collateral
        damage of the downstream repair. Lists untouched."""
        from agent.turn_context import drop_stale_api_content

        def _plain_user(m: Any) -> bool:
            return (
                isinstance(m, dict) and m.get("role") == "user" and not _is_summary_marker(m)
                and isinstance(m.get("content"), str)
            )

        merged: List[Dict[str, Any]] = []
        for msg in result:
            prev = merged[-1] if merged else None
            if _plain_user(msg) and _plain_user(prev):
                prev["content"] = "\n\n".join(c for c in (prev["content"], msg["content"]) if c)
                drop_stale_api_content(prev)  # merged content invalidates the api_content sidecar
            else:
                merged.append(msg)
        return merged
