"""CLI-side compaction edges of ``ClaudeAgentSdkSession``: the PreCompact hook,
the compact_boundary completion edge and the context-usage probe. Extracted from
``claude_agent_sdk_session.py``; every method resolves through
``ClaudeAgentSdkSession``'s MRO unchanged.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.transports.claude_agent_sdk_session")


class ClaudeSdkCompactionMixin:
    """Compaction hooks, boundary handling and context usage (see module docstring)."""

    # ---------- compaction ----------

    def _build_compaction_hooks(self) -> Optional[dict]:
        """PreCompact -> watchdog suspension, and on_compaction(trigger).

        ``trigger`` is the SDK's own literal: "auto" for the CLI's automatic
        compaction (the one that silently stalls a turn) or "manual" for an
        explicit /compact. The callback is best-effort: it must never fail the
        hook, because refusing a hook can block the compaction itself.

        Wired even when ``_on_compaction`` is None. The hook's primary job is
        no longer the status notice but telling _TurnWatch that the coming
        silence is legitimate; skipping it when only the status callback is
        unset would leave the turn killable mid-compaction for no benefit.
        """
        try:
            from claude_agent_sdk import HookMatcher
        except Exception:  # pragma: no cover - SDK predates hooks
            logger.debug("claude-agent-sdk: HookMatcher unavailable", exc_info=True)
            return None

        async def _on_pre_compact(input_data, tool_use_id, context):
            # FIRST and unconditionally, outside the try: if a status callback
            # raises, the suspension must still be in place. This ordering is
            # the fix -- everything below is the pre-existing status notice.
            #
            # getattr, not attribute access: a hook that raises AttributeError
            # would break the very turn it exists to protect, and the attribute
            # is genuinely absent on sessions built without __init__.
            watch = getattr(self, "_turn_watch", None)
            if watch is not None:
                watch.compaction_begin()
            try:
                trigger = ""
                if isinstance(input_data, dict):
                    trigger = str(input_data.get("trigger") or "")
                if self._on_compaction is not None:
                    self._on_compaction(trigger or "auto")
            except Exception:
                logger.debug("compaction status callback failed", exc_info=True)
            return {}

        return {"PreCompact": [HookMatcher(hooks=[_on_pre_compact])]}

    def _handle_compact_boundary(self, message: Any) -> None:
        """compact_boundary -> on_compact_boundary(trigger): compaction FINISHED.

        The SDK exposes ``PreCompact`` as a hook but has no post-side
        counterpart, which is why the completion edge was originally deferred to
        the end of the turn. That was wrong: the CLI *does* announce completion,
        as a plain ``system`` message with ``subtype="compact_boundary"``, and
        the SDK's parser passes unknown subtypes through its generic fallback,
        so it arrives on the normal message stream mid-turn.

        The distinction is not cosmetic. Emitting at turn end put the notice in
        the one place it could never be seen: non-durable statuses are deleted
        by end-of-turn progress cleanup, so it was created and destroyed in the
        same instant (measured 2026-08-16 -- boundary at 07:41:16, deferred emit
        at 07:42:17, 61s late and invisible). Firing here restores the native
        contract every other provider already follows -- notice right after
        compaction, cleaned up with the rest of the turn's progress -- and makes
        COMPACTION_DONE_STATUS's "continuing turn" literally true again.

        Best-effort by construction: a raising callback must not break the turn
        that is still streaming.
        """
        if (
            type(message).__name__ != "SystemMessage"
            or getattr(message, "subtype", "") != "compact_boundary"
        ):
            return
        # Lift the suspension before anything else -- including the unwired-
        # callback early return further down. Leaving it armed would hold the
        # gate open until the bounded ceiling on every compacting turn.
        # getattr for the same reason as the PreCompact side: this runs on the
        # message drain, where an AttributeError would break a streaming turn.
        watch = getattr(self, "_turn_watch", None)
        if watch is not None:
            watch.compaction_end()
        data = getattr(message, "data", None)
        metadata = None
        if isinstance(data, dict):
            # The SDK hands this over snake_cased ("compact_metadata", observed
            # in production 2026-08-16); the CLI's own transcript writes the
            # camelCase original. Accept both so the trigger stays accurate if
            # the normalization changes -- an unreadable trigger is not fatal
            # (it falls back to "auto"), just less honest.
            metadata = data.get("compact_metadata") or data.get("compactMetadata")
        trigger = ""
        if isinstance(metadata, dict):
            trigger = str(metadata.get("trigger") or "")
        logger.info(
            "claude-agent-sdk compact_boundary: session=%s trigger=%s",
            getattr(message, "session_id", None) or self._session_id or "none",
            trigger or "unknown",
        )
        if self._on_compact_boundary is None:
            return
        try:
            self._on_compact_boundary(trigger or "auto")
        except Exception:
            logger.debug("compact_boundary callback failed", exc_info=True)

    # ---------- context usage ----------

    def context_usage(self) -> Optional[dict]:
        """Live context usage reported by the CLI, or None if unavailable.

        Ground truth, unlike Hermes' own estimate on this lane: api_messages
        holds FULL tool payloads that are never sent to the CLI, so the local
        estimate over-reports by an order of magnitude (1.5-2.4M tokens for a
        transcript whose real size was ~111k). Callers that need a real number
        -- status lines, compaction heuristics -- must use this instead.

        Returns the SDK's ContextUsageResponse mapping: totalTokens, maxTokens
        (already reduced by the autocompact buffer), contextWindow, the
        percentage used, model, and isAutoCompactEnabled.

        Best-effort by design: a disconnected session, an older SDK without the
        method, or a query failure all yield None rather than raising into a
        status path.
        """
        client = self._client
        if client is None or self._loop is None:
            return None
        getter = getattr(client, "get_context_usage", None)
        if not callable(getter):
            return None  # SDK predates get_context_usage()
        try:
            usage = self._run_coro(getter(), timeout=10.0)
        except Exception:
            logger.debug(
                "claude-agent-sdk context-usage query failed", exc_info=True
            )
            return None
        return usage if isinstance(usage, dict) else None
