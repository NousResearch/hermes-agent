"""Adapter-driven dispatch of structured stream events to a delivery sink.

``GatewayEventDispatcher`` is the seam Tobi asked for: the agent emits typed
events (gateway/stream_events.py), and the *adapter* decides how each one is
delivered.  The dispatcher holds an adapter + the stream consumer (sink) + the
resolved per-channel presentation settings (tool-progress mode, preview length)
and routes each event through the adapter's render hooks.

Message/commentary/segment events flow into the consumer (native draft on
Telegram DMs, edit-in-place elsewhere).  Tool events are formatted by the
adapter — which may return None to *eat* the event on platforms that can't
render tool chrome — and the rendered line is enqueued onto the same tool
progress queue the gateway already drains, so the two no longer race through
independent code paths.

This module deliberately has no platform knowledge and no asyncio: it is a thin
synchronous router callable from the agent's worker thread, exactly like the
callbacks it replaces.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from gateway.stream_events import (
    Commentary,
    GatewayNotice,
    LongToolHint,
    MessageChunk,
    MessageStop,
    StreamEvent,
    ToolCallChunk,
    ToolCallFinished,
)
from gateway.progress_summary import ToolProgressSummary

logger = logging.getLogger("gateway.stream_events")


class GatewayEventDispatcher:
    """Route typed stream events through an adapter onto a delivery sink.

    Parameters
    ----------
    adapter:
        The platform adapter.  Provides ``render_message_event`` and
        ``format_tool_event`` (BasePlatformAdapter defaults reproduce today's
        behavior; adapters may override for native rendering).
    sink:
        The GatewayStreamConsumer for assistant-text delivery.  May be None
        when streaming is disabled, in which case message events are dropped
        (the final response still goes out via the normal send path).
    enqueue_tool_line:
        Callback that places a rendered tool-progress line onto the gateway's
        progress queue (the same queue ``send_progress_messages`` drains).  May
        be None when tool progress is disabled for this channel.
    tool_mode:
        Resolved tool-progress mode for this channel ("all" / "new" /
        "summary" / "verbose" / "off").
    preview_max_len:
        Resolved ``tool_preview_length`` (0 = no cap in verbose mode).
    on_long_tool / on_notice:
        Optional hooks for LongToolHint / GatewayNotice events, letting the
        gateway own the "should I surface this here?" decision.
    """

    def __init__(
        self,
        adapter: Any,
        sink: Any = None,
        *,
        enqueue_tool_line: Optional[Callable[[Any], None]] = None,
        tool_mode: str = "all",
        preview_max_len: int = 40,
        on_long_tool: Optional[Callable[[LongToolHint], None]] = None,
        on_notice: Optional[Callable[[GatewayNotice], None]] = None,
    ) -> None:
        self.adapter = adapter
        self.sink = sink
        self._enqueue_tool_line = enqueue_tool_line
        self.tool_mode = tool_mode or "all"
        self.preview_max_len = preview_max_len
        self._on_long_tool = on_long_tool
        self._on_notice = on_notice
        # "new" mode dedup — only report when the tool changes.
        self._last_tool: Optional[str] = None
        self._summary = ToolProgressSummary() if self.tool_mode == "summary" else None

    def dispatch(self, event: StreamEvent) -> None:
        """Route a single event.  Never raises into the agent's worker thread."""
        try:
            self._dispatch(event)
        except Exception:  # presentation must never break the agent loop
            logger.debug("stream-event dispatch error", exc_info=True)

    def _dispatch(self, event: StreamEvent) -> None:
        if isinstance(event, (MessageChunk, MessageStop, Commentary)):
            if self.sink is not None:
                self.adapter.render_message_event(event, self.sink)
            return

        if isinstance(event, ToolCallChunk):
            if self.tool_mode == "off" or self._enqueue_tool_line is None:
                return
            if self._summary is not None:
                rendered = self._summary.update(
                    "tool.started", event.tool_name, event.preview, event.args
                )
                if rendered:
                    self._enqueue_tool_line(("__replace__", rendered))
                return
            # "new" mode: only emit when the tool changes.
            if self.tool_mode == "new" and event.tool_name == self._last_tool:
                return
            self._last_tool = event.tool_name
            line = self.adapter.format_tool_event(
                event, mode=self.tool_mode, preview_max_len=self.preview_max_len,
            )
            # None == adapter chose to eat this event (can't render tool chrome).
            if line:
                self._enqueue_tool_line(line)
            return

        if isinstance(event, ToolCallFinished):
            if self._summary is not None and self._enqueue_tool_line is not None:
                rendered = self._summary.update(
                    "tool.completed",
                    event.tool_name,
                    None,
                    None,
                    duration=event.duration,
                    is_error=not event.ok,
                )
                if rendered:
                    self._enqueue_tool_line(("__replace__", rendered))
            # Default non-summary path: no chrome on completion (matches today —
            # the gateway only rendered "started" events). Completion also
            # drives onboarding hints through LongToolHint.
            return

        if isinstance(event, LongToolHint):
            if self._on_long_tool is not None:
                self._on_long_tool(event)
            return

        if isinstance(event, GatewayNotice):
            if self._on_notice is not None:
                self._on_notice(event)
            return


__all__ = ["GatewayEventDispatcher"]
