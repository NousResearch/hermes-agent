"""keep_history — Codex-style context engine: compact background context
(tool logs, terminal dumps, file arrays) in the REQUEST only, never in the
persisted transcript.

Why this engine exists
----------------------
The built-in ``compressor`` engine rewrites the live transcript on
compaction (``archive_and_compact`` soft-archives every active row and
re-inserts head + summary + tail). Desktop/gateway UIs render only active
rows, so a compaction visibly collapses the whole chat into a summary card
plus the recent tail. ``keep_history`` never rewrites the transcript: it
prunes old tool-result payloads from the per-request message list via the
``select_context()`` hook (request-only by contract — the session DB is
never touched), so the model always sees a bounded, compacted context while
the UI keeps the full conversation scrollable. This mirrors Codex Desktop's
behavior (thread stays intact; background tool context is compacted).

Behavior
--------
- ``should_compress()`` -> False: the destructive full-compression path
  (LLM summary + ``archive_and_compact``) is never auto-triggered.
- ``select_context()``: when the assembled request crosses the token
  trigger, run the deterministic 3-pass tool-result prune (dedup, one-line
  summaries for large outputs, tool_call-arg truncation) on a COPY of the
  request. Returns the pruned list; persisted history is untouched. A
  rearm watermark (like the built-in proactive prune) keeps cache-breaking
  prunes episodic, and a hard-ceiling trim guarantees the request still
  fits the window even in pathological sessions — note that this trim CAN
  drop mid-conversation messages from the request the model sees
  (request-level drop only; the persisted transcript — and therefore the
  visible chat history — still keeps every turn).
- ``compress()`` (manual ``/compress`` / gateway hygiene): deterministic
  prune-only compaction — every user/assistant message is preserved
  verbatim; only old tool-result bodies are shortened. Never produces a
  summary card and never drops chat turns.
- ``prune_tool_results_only()`` -> no-op: the loop-level proactive prune
  commits via ``archive_and_compact`` (a transcript rewrite); this engine
  does its pruning request-only in ``select_context`` instead.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from agent.context_compressor import (
    ContextCompressor,
    _estimate_msg_budget_tokens,
    resolve_model_threshold,
)

logger = logging.getLogger(__name__)

# Request-prune trigger as a fraction of the context window when
# ``proactive_prune_tokens`` is not configured.
_REQUEST_PRUNE_TRIGGER_RATIO = 0.55

# Safety-valve window when neither the model's context length nor the
# request's ``budget_tokens`` is known (call sites that skip the budget):
# request compaction must never silently disable itself and let requests
# grow past the provider window unbounded.
_FALLBACK_CONTEXT_TOKENS = 128_000


def _load_compression_config() -> Dict[str, Any]:
    """Read the ``compression`` block via the standard config loader.

    Plugin engines are constructed with no arguments (the loader calls
    ``Engine()``), so config values that the built-in engine receives via
    its constructor must be resolved here. Going through
    ``hermes_cli.config.load_config`` (instead of a direct YAML read)
    keeps env overrides, managed scope and profile switches consistent
    with the rest of the app. Falls back to built-in-safe defaults when
    the config subsystem is unavailable.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        block = cfg.get("compression") if isinstance(cfg, dict) else None
        return block if isinstance(block, dict) else {}
    except Exception as exc:
        logger.debug("Could not load compression config: %s", exc)
        return {}


class KeepHistoryContextEngine(ContextCompressor):
    """Request-only compaction engine that preserves the full transcript.

    The persisted conversation history (and therefore the visible chat
    timeline in the desktop app) is never rewritten by this engine. All
    compaction happens on the per-request message list.
    """

    @property
    def name(self) -> str:
        return "keep_history"

    def __init__(self, *args, **kwargs):
        _cfg = _load_compression_config()

        def _f(key, default):
            try:
                v = _cfg.get(key)
                return default if v is None else v
            except Exception:
                return default

        kwargs.setdefault("model", "keep_history")
        kwargs.setdefault("threshold_percent", float(_f("threshold", 0.85)))
        kwargs.setdefault("protect_first_n", int(_f("protect_first_n", 3)))
        kwargs.setdefault("protect_last_n", int(_f("protect_last_n", 60)))
        kwargs.setdefault("summary_target_ratio", float(_f("target_ratio", 0.20)))
        kwargs.setdefault("quiet_mode", True)
        kwargs.setdefault(
            "proactive_prune_tokens", int(_f("proactive_prune_tokens", 0))
        )
        kwargs.setdefault(
            "proactive_prune_min_result_chars",
            int(_f("proactive_prune_min_result_chars", 2000)),
        )
        kwargs.setdefault(
            "proactive_prune_min_reclaim_tokens",
            int(_f("proactive_prune_min_reclaim_tokens", 4096)),
        )
        kwargs.setdefault(
            "min_tail_user_messages", int(_f("min_tail_user_messages", 5))
        )
        super().__init__(*args, **kwargs)
        # Watermark: after a committed request-prune, do not prune again
        # until the transcript regrows a full trigger-sized runway, so
        # prompt-cache breaks stay episodic (same contract as the built-in
        # proactive prune's rearm gate).
        self._select_rearm_tokens: int = 0
        # Last measured savings from a request prune (for status/logging).
        self._last_select_prune_savings_tokens: int = 0

    # -- never auto-trigger the destructive full compression ----------------

    def should_compress(self, prompt_tokens: Optional[int] = None) -> bool:
        """Never auto-fire the LLM-summary + transcript-rewrite path."""
        return False

    def should_compress_info(
        self, prompt_tokens: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        return False, None

    def should_compress_preflight(self, messages: List[Dict[str, Any]]) -> bool:
        # The request is compacted every turn via select_context(); there is
        # no preflight path that should force a transcript rewrite.
        return False

    def prune_tool_results_only(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """No-op: the loop-level proactive prune commits to the session DB
        (``archive_and_compact``), which this engine never does. Request
        pruning happens in ``select_context`` instead."""
        return messages, 0

    # -- request-only compaction -------------------------------------------

    def select_context(
        self,
        request_messages: List[Dict[str, Any]],
        *,
        conversation_messages: Optional[List[Dict[str, Any]]] = None,
        incoming_message: Optional[Dict[str, Any]] = None,
        budget_tokens: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        """Prune old tool-result payloads from THIS request only.

        Runs every turn, independent of ``should_compress()``. Returns the
        compacted request list when the request crossed the token trigger
        and the prune reclaimed meaningful tokens; ``None`` (no-op) keeps
        the request and the provider-side prompt cache untouched.
        """
        if not request_messages:
            return None
        ctx = int(self.context_length or budget_tokens or 0)
        if ctx <= 0:
            # Safety valve: compaction must never silently disable itself
            # just because the call site didn't pass a budget.
            ctx = _FALLBACK_CONTEXT_TOKENS

        trigger = (
            int(self.proactive_prune_tokens)
            if self.proactive_prune_tokens and self.proactive_prune_tokens > 0
            else int(ctx * _REQUEST_PRUNE_TRIGGER_RATIO)
        )
        est = sum(_estimate_msg_budget_tokens(m) for m in request_messages)
        if est < trigger:
            return None
        if est < self._select_rearm_tokens:
            return None

        pruned, n = self._prune_old_tool_results(
            request_messages,
            protect_tail_count=self.protect_last_n,
            protect_tail_tokens=None,
            min_prune_chars=self.proactive_prune_min_result_chars,
        )
        if not n:
            return None

        after = sum(_estimate_msg_budget_tokens(m) for m in pruned)
        reclaimed = max(0, est - after)
        if reclaimed < self.proactive_prune_min_reclaim_tokens:
            return None

        # Hard ceiling: even after tool-result pruning the request would
        # still overflow the window (pathological: giant user/assistant
        # text). Drop the oldest non-head/non-tail messages — request-only,
        # the persisted transcript is untouched.
        if after > int(ctx * 0.95):
            pruned = self._trim_request_to_budget(pruned, int(ctx * 0.85))
            after = sum(_estimate_msg_budget_tokens(m) for m in pruned)
            reclaimed = max(0, est - after)

        runway = max(reclaimed, trigger, self.proactive_prune_min_reclaim_tokens)
        self._select_rearm_tokens = after + runway
        self._last_select_prune_savings_tokens = reclaimed
        return pruned

    def _trim_request_to_budget(
        self, messages: List[Dict[str, Any]], budget_tokens: int
    ) -> List[Dict[str, Any]]:
        """Drop the oldest non-head messages until the request fits ``budget_tokens``.

        Request-only fallback for pathological sessions. Never drops the
        head (system prompt + first exchange) or the recent tail. Orphaned
        tool results from the cut are cleaned by ``_sanitize_tool_pairs``.
        """
        if not messages:
            return messages
        head = self._protect_head_size(messages)
        tail = min(self.protect_last_n, max(3, len(messages) - head))
        if head + tail >= len(messages):
            return messages
        end_kept = len(messages) - tail
        cut = head
        total = sum(_estimate_msg_budget_tokens(m) for m in messages)
        while cut < end_kept and total > budget_tokens:
            total -= _estimate_msg_budget_tokens(messages[cut])
            cut += 1
        if cut <= head:
            return messages
        kept = messages[:head] + messages[cut:]
        return self._sanitize_tool_pairs(kept)

    # -- non-destructive compaction for /compress and hygiene ---------------

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Deterministic prune-only compaction.

        All user and assistant messages are preserved verbatim; only old
        tool-result bodies are shortened/deduped and oversized tool_call
        arguments truncated. No LLM summary is produced and no chat turn is
        dropped, so even a manual ``/compress`` never collapses the visible
        history into a summary card.
        """
        if not messages:
            return messages
        pruned, _n = self._prune_old_tool_results(
            messages,
            protect_tail_count=self.protect_last_n,
            protect_tail_tokens=None,
            min_prune_chars=self.proactive_prune_min_result_chars,
        )
        pruned = self._sanitize_tool_pairs(pruned)
        self.compression_count += 1
        return pruned


def register(ctx) -> None:
    """Plugin-style registration hook (also handled by the loader's class
    scan; kept for general-plugin-system compatibility)."""
    ctx.register_context_engine(KeepHistoryContextEngine())
