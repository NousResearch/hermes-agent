"""How a gateway turn's reply reaches the editor: ``messageId`` grouping, exactly-once final text, tool and
permission bubbles closed as the in-process agent closed them. Mixed into ``GatewayACPAgent``, whose state
(``_conn``, ``_streamed``, ``_segments``, ``_tool_args``, ``_message_ids``) it reads and updates."""
from __future__ import annotations

import logging

import acp

logger = logging.getLogger(__name__)


def _text(payload):
    """Reply text of a message frame; ``text`` may be present and null (a suppressed delivery)."""
    text = payload.get("text")
    return text if isinstance(text, str) else ""


class GatewayReplyMixin:
    async def _project_tool(self, kind, sid, editor_id, aid, payload):
        """``tool.start`` / ``tool.complete`` as the in-process callbacks projected them (#134889 ACP-15)."""
        from acp_adapter.tools import build_tool_complete, build_tool_start, coerce_tool_args
        # In-process turns publish ``tool_name``; managed workers publish ``name``.
        tool_name = payload.get("tool_name") or payload.get("name") or "tool"
        call_id = payload["tool_call_id"]
        if kind == "tool.start":
            # Deltas before a tool call are commentary the final reply does not repeat: close
            # that segment so ``message.complete`` measures only the final's own stream, and end
            # that assistant message so the text after the tool opens a new one (a new ``messageId``).
            self._close_segment(aid)
            self._end_message(sid)
            args = coerce_tool_args(payload.get("args"))
            self._tool_args[(sid, call_id)] = (tool_name, args, aid)
            if self._conn:
                await self._conn.session_update(session_id=editor_id, update=build_tool_start(call_id, tool_name, args))
            return
        name, args, _ = self._tool_args.pop((sid, call_id), (tool_name, {}, aid))
        result = payload.get("result")
        if self._conn:
            # ``is_error`` is the executor's verdict: a cancelled or failed tool may return plain text the
            # result heuristic reads as success.
            await self._conn.session_update(session_id=editor_id, update=build_tool_complete(
                call_id, name, result=result if isinstance(result, str) else None, function_args=args,
                is_error=payload.get("is_error") is True))

    async def _finish_reply(self, sid, editor_id, aid, payload):
        """Show the settled reply's unseen remainder, then end the turn: its reply's id closes, and a tool
        call it left open (blocked by a guard, an approval denied) ends ``failed``, as the in-process agent
        flushed them at turn end. Shared by ``message.complete`` and a retry's terminal receipt."""
        remainder = self._final_remainder(aid, payload)
        if remainder and self._conn:
            await self._conn.session_update(session_id=editor_id, update=self._chunk(sid, remainder))
        self._end_message(sid)
        await self._flush_open_tools(sid, editor_id, aid)

    async def _flush_open_tools(self, sid, editor_id, aid):
        """Close every call admission *aid* started and never completed (``failed``: it produced no result)."""
        from acp_adapter.tools import build_tool_abandoned
        for key in [key for key, held in self._tool_args.items() if key[0] == sid and held[2] == aid]:
            name, _, _ = self._tool_args.pop(key)
            if self._conn:
                await self._conn.session_update(session_id=editor_id, update=build_tool_abandoned(key[1], name))

    def _chunk(self, sid, text, *, thought=False):
        """An assistant (or thought) chunk carrying the session's current ``messageId``: ACP clients group
        chunks into one reply by it and start a new bubble when it changes."""
        from acp_adapter.events import AssistantMessageIdAllocator
        update = (acp.update_agent_thought_text if thought else acp.update_agent_message_text)(text)
        update.message_id = self._message_ids.setdefault(sid, AssistantMessageIdAllocator()).current()
        return update

    def _end_message(self, sid):
        if (ids := self._message_ids.get(sid)) is not None:
            ids.close()

    async def _project_side_update(self, kind, sid, editor_id, payload):
        """Turn events outside the reply text, as the in-process server projected them.

        ``reasoning.delta``: the model's reasoning as thought chunks. It never enters the reply's segments,
        so ``message.complete``'s final remainder is unchanged.
        ``todo.updated``: the owner's full todo snapshot (published for a LOCAL session's turn, which an ACP
        session is) as the native ``plan`` update Zed renders as its task panel (#134889 ACP-5)."""
        if kind == "reasoning.delta":
            text = _text(payload)
            update = self._chunk(sid, text, thought=True) if text else None
        else:
            import json
            from acp_adapter.events import _build_plan_update_from_todo_result
            todos = payload.get("todos")
            update = _build_plan_update_from_todo_result(json.dumps({"todos": todos})) \
                if isinstance(todos, list) else None
        if update is not None and self._conn:
            await self._conn.session_update(session_id=editor_id, update=update)

    def _close_segment(self, aid):
        streamed = self._streamed.pop(aid, "")
        if streamed.strip():
            self._segments.setdefault(aid, []).append(streamed)

    def _final_remainder(self, aid, payload):
        """The part of the settled reply the editor has not been shown yet (exactly-once text)."""
        from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX

        text = _text(payload)
        streamed = self._streamed.pop(aid, "")
        shown = self._segments.pop(aid, [])
        # Local interrupt status is metadata; ACP carries it in stop_reason.
        if not text or (payload.get("outcome") == "cancelled"
                        and text.startswith(INTERRUPT_WAITING_FOR_MODEL_PREFIX)):
            return ""
        if payload.get("response_reused") is True and (streamed or shown):
            # The owner names the final as the reply already painted (never inferred from text).
            return ""
        if text.startswith(streamed):
            remainder = text[len(streamed):]
            # An unstreamed final the owner already published as an interim segment.
            return "" if not streamed and text.strip() in {s.strip() for s in shown} else remainder
        if text.strip() == streamed.strip():
            return ""
        return "\n\n" + text

    async def _close_card(self, editor_id, tool_call, options, response):
        """The editor shows a permission request's ``tool_call`` as a pending bubble; once it is answered,
        close it as the in-process agent did (``permissions.await_permission``): ``completed`` for an allow,
        ``failed`` for a reject or a dismissal (#134889 ACP-15)."""
        outcome = getattr(response, "outcome", None)
        allowed = getattr(outcome, "outcome", None) == "selected" and any(
            option.option_id == getattr(outcome, "option_id", None) and option.kind.startswith("allow")
            for option in options)
        try:
            await self._conn.session_update(session_id=editor_id, update=acp.update_tool_call(
                tool_call.tool_call_id, status="completed" if allowed else "failed"))
        except Exception:
            # Boundary: display only; a failed close must never cost the user's answer its delivery.
            logger.debug("ACP permission card %s not closed", tool_call.tool_call_id, exc_info=True)
