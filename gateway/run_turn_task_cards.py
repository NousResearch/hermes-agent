"""Opt-in rich Slack task cards (``display.tool_progress_native``) layered on the stock card rail.

The stock native-card rail (``_TaskCardState``, ``_task_card_publish``,
``_send_native_task_card_progress``) lives in ``GatewayTurnProgressMixin`` and is untouched. This
mixin sits in front of it in ``TurnRunner``'s MRO and changes behaviour only for a turn whose
``ctx._rich_slack_task_cards`` is true; every override otherwise delegates to ``super()``, so a turn
without the flag runs the stock code path unchanged.

The rich rail reuses the stock destination preflight, egress-decline suppression and editable text
fallback. What it adds is a second publisher (``gateway.slack_task_stream.RichTaskCardSession``) that
receives the same progress queue plus reasoning and subagent events.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import queue
from typing import Any

from gateway.platforms.base import SendResult
from gateway.run_turn_progress import GatewayTurnProgressMixin

logger = logging.getLogger("gateway.run")

_TOOL_EVENTS = frozenset({"tool.started", "tool.completed"})
_SUBAGENT_EVENTS = frozenset({"subagent.start", "subagent.tool", "subagent.complete"})
_RICH_EVENTS = _TOOL_EVENTS | _SUBAGENT_EVENTS | {"reasoning.delta"}


def rich_cards_enabled(ctx: Any) -> bool:
    """Rich cards refine the stock native card: both must be on, and only Slack ever sets the former."""
    resolve = getattr(ctx, "resolve_display_setting", None)
    if not (getattr(ctx, "_native_slack_task_cards", False) and callable(resolve)):
        return False
    return bool(resolve(ctx.user_config, "slack", "tool_progress_native"))


def card_reasoning_callback(runner: Any):
    """The per-turn ``agent.reasoning_callback``: a reasoning sink only while rich cards are on."""
    ctx = runner._ctx
    if not (getattr(ctx, "_rich_slack_task_cards", False) and ctx.progress_queue):
        return None

    def _card_reasoning(text):
        if text and runner._native_card_gate():
            ctx.progress_queue.put({"type": "reasoning.delta", "text": text})
    return _card_reasoning


class TaskCardMixin:
    """Rich-card extension points for ``TurnRunner``; see the module docstring."""

    def _rich_cards_on(self) -> bool:
        return bool(getattr(self._ctx, "_rich_slack_task_cards", False))

    # ── queue producers (agent thread) ──────────────────────────────────────────────────────

    def _native_card_gate(self) -> bool:
        ctx = self._ctx
        if not super()._native_card_gate():
            return False
        # Closed once cleanup starts; a muted notification turn never opens a rich card.
        return not (self._rich_cards_on() and (ctx._task_cards_closed or ctx.mute_notification_reply))

    def progress_callback(self, event_type, tool_name=None, preview=None, args=None, **kwargs):
        super().progress_callback(event_type, tool_name, preview, args, **kwargs)
        if event_type in _SUBAGENT_EVENTS and self._rich_cards_on() and self._native_card_gate():
            self._ctx.progress_queue.put({**kwargs, "type": event_type, "tool_name": tool_name,
                                          "preview": preview, "args": args})

    def native_tool_start_callback(self, call_id, tool_name, args):
        if not self._rich_cards_on():
            return super().native_tool_start_callback(call_id, tool_name, args)
        if not self._native_card_gate():
            return None
        from agent.display import build_tool_preview
        name = str(tool_name or "tool")
        self._ctx.progress_queue.put({
            "type": "tool.started", "tool_call_id": str(call_id or ""), "tool_name": name,
            "preview": build_tool_preview(name, args or {}, max_len=64) or "", "args": args or {},
        })
        return None

    def native_tool_complete_callback(self, call_id, tool_name, args, result):
        if not self._rich_cards_on():
            return super().native_tool_complete_callback(call_id, tool_name, args, result)
        if not self._native_card_gate():
            return None
        from agent.display import _detect_tool_failure
        name = str(tool_name or "tool")
        is_error, _ = _detect_tool_failure(name, result)
        self._ctx.progress_queue.put({
            "type": "tool.completed", "tool_call_id": str(call_id or ""), "tool_name": name,
            "args": args or {}, "result": result, "is_error": bool(is_error),
        })
        return None

    # ── publisher state and rail ─────────────────────────────────────────────────────────────

    @dataclasses.dataclass
    class _RichTaskCardState(GatewayTurnProgressMixin._TaskCardState):
        """Stock card state plus the rich publisher and the events it has not published yet."""
        rich: Any = None
        pending: list = dataclasses.field(default_factory=list)

        def apply_event(self, raw: Any) -> bool:
            event_type = raw.get("type") if isinstance(raw, dict) else None
            if event_type not in _RICH_EVENTS:
                return super().apply_event(raw)
            if event_type != "reasoning.delta":
                from gateway.slack_task_stream import _redact_card_event
                raw = _redact_card_event(raw)
            if not self.native_failed:
                self.pending.append(raw)
                if event_type not in _TOOL_EVENTS:
                    return True
            # After a native failure only tool rows still feed the stock text fallback.
            return super().apply_event(raw)

    def _rich_card_session(self, adapter):
        """A rich publisher for this turn, or None when the adapter cannot host one directly."""
        from gateway.slack_task_stream import RichTaskCardSession
        try:
            candidate = RichTaskCardSession(adapter, self._ctx)
        except Exception:
            logger.warning("Rich card setup failed; using the adapter's native rail", exc_info=True)
            return None
        # Non-direct connectors keep THEIR egress guard and the stock basic card.
        return candidate if candidate.main is not None else None

    async def _task_card_native_publish(self, st):
        rich = getattr(st, "rich", None)
        if rich is None:
            return await super()._task_card_native_publish(st)
        pending, st.pending = st.pending, []
        if not rich.client.is_live():
            return SendResult(success=True)  # a dead turn is not a transport failure
        result = await rich.publish(pending)
        if not rich.client.is_live():
            return SendResult(success=True)  # turn death during I/O is not a transport failure either
        return result

    async def _task_card_send_or_edit_fallback(self, st) -> None:
        rich = getattr(st, "rich", None)
        if rich is not None and not rich.client.is_live():
            return
        await super()._task_card_send_or_edit_fallback(st)

    async def _send_native_task_card_progress(self, adapter) -> None:
        """The stock drain, plus: publications shielded from cleanup, idle flush, ordered close."""
        ctx = self._ctx
        rich = self._rich_card_session(adapter) if self._rich_cards_on() else None
        if rich is None:
            return await super()._send_native_task_card_progress(adapter)
        st = self._RichTaskCardState(adapter, rich=rich)
        publication = None
        try:
            while ctx._run_still_current():
                try:
                    raw = ctx.progress_queue.get_nowait()
                except queue.Empty:
                    if not st.native_failed:
                        await rich.flush_pending()
                    await asyncio.sleep(0.1)
                    continue
                if not self._agent_interrupted() and st.apply_event(raw):
                    # Cleanup cancels this consumer, not the only owner of the popped append-only
                    # event. Never replay an append whose acceptance is still unknown.
                    publication = asyncio.create_task(self._task_card_publish(st))
                    await asyncio.shield(publication)
                    publication = None
        except asyncio.CancelledError:
            ctx._task_cards_closed = True
            try:
                if publication is not None:
                    await asyncio.wait_for(publication, timeout=5.0)
                if self._task_card_drain(st) and ctx._run_still_current() and not self._agent_interrupted():
                    await asyncio.wait_for(self._task_card_publish(st), timeout=5.0)
            except TimeoutError:
                # Unknown acceptance is terminal for this rail: no replay, fallback or reasoning flush.
                st.native_failed = True
                st.publication_suppressed = True
                self._drain_progress_queue()
                logger.warning("Timed out joining task-card publication during cleanup")
        finally:
            ctx._task_cards_closed = True
            await rich.stop(flush_reasoning=(
                ctx._run_still_current() and not self._agent_interrupted()
                and not st.native_failed and not st.publication_suppressed
            ))
