"""Structured stream-event rendering (gateway/stream_events.py) for platform adapters.

Presentation-only: nothing rendered here is persisted, so what an adapter "eats" never changes
history. Mixed into ``BasePlatformAdapter``; no imports from it (base imports this module).
"""

from typing import TYPE_CHECKING, Any, Optional

from agent.i18n import t

if TYPE_CHECKING:
    from agent.display import ToolPreview


class StreamEventRenderMixin:
    """Render stream events into adapter-native text; rich-text adapters override the hooks.
    Host contract: ``supports_code_blocks`` (declared on ``BasePlatformAdapter``)."""

    def render_message_event(self, event: Any, sink: Any) -> None:
        """Render a MessageChunk / MessageStop / Commentary onto the sink (a
        GatewayStreamConsumer), mapping 1:1 onto its existing primitives."""
        from gateway.stream_events import MessageChunk, MessageStop, Commentary
        if isinstance(event, MessageChunk) and event.text:
            sink.on_delta(event.text)
        elif isinstance(event, MessageStop) and not event.final:
            # Intermediate stop (text → tool → text) = segment break; the terminal stop is finish().
            sink.on_segment_break()
        elif isinstance(event, Commentary) and event.text:
            sink.on_commentary(event.text)

    def format_tool_event(self, event: Any, *, mode: str = "all", preview_max_len: int = 40) -> Optional[str]:
        """Rendered chrome for a ToolCallChunk, or None to eat it (adapters without editing/rich
        text override to None). ``mode``: tool-progress mode ("all"/"new"/"verbose");
        ``preview_max_len`` mirrors ``tool_preview_length`` (0 = no cap in verbose)."""
        from gateway.stream_events import ToolCallChunk
        if not isinstance(event, ToolCallChunk):
            return None
        from agent.display import get_tool_emoji, prepare_tool_preview, verbose_tool_args_text
        emoji, tool = get_tool_emoji(event.tool_name, default='⚙️'), event.tool_name
        if mode == "verbose" and event.args:
            args_str = verbose_tool_args_text(event.args, preview_max_len, self.supports_code_blocks)
            return t("gateway.progress.tool_verbose", emoji=emoji, tool=tool, keys=list(event.args.keys()), args=args_str)
        if not event.preview:
            return t("gateway.progress.tool_pending", emoji=emoji, tool=tool)
        if mode == "verbose":
            return t("gateway.progress.tool_preview", emoji=emoji, tool=tool, preview=event.preview)
        # "all" / "new": short capped preview (default 40; progress bubbles persist as messages).
        cap = preview_max_len if preview_max_len > 0 else 40
        prepared = prepare_tool_preview(
            event.tool_name, event.args, fallback=event.preview, max_len=cap)
        return t("gateway.progress.tool_preview", emoji=emoji, tool=tool, preview=self.format_tool_preview(prepared))

    def format_tool_preview(self, preview: "ToolPreview") -> str:
        """Platform-native formatting of a compact tool preview; rich-text adapters may use
        the preview's metadata (e.g. a URL shortened for display)."""
        return preview.text
