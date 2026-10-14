"""Tool lifecycle dispatch into chat, logs and single-message activity overlays."""
from __future__ import annotations

from datetime import datetime
from agent.i18n import t


class TurnProgressMixin:
    def progress_callback(self, event_type: str, tool_name: str | None = None, preview: str | None = None, args: dict | None = None, **kwargs):
        """Callback invoked by agent on tool lifecycle events."""
        ctx = self._ctx
        # Failed subagent → one clean user-facing notice, handled FIRST, before every progress-queue
        # gate: platforms with tool_progress off must still hear about a dead delegation.
        if event_type == "subagent.complete":
            self._progress_subagent_notice(preview, kwargs)
            return
        self._progress_live_status(event_type, tool_name, args)
        # "log" mode: append tool.started lines to the log queue, silent in chat. Handled before
        # the progress_queue guard because log mode runs without a chat progress queue.
        if ctx.log_queue is not None and event_type == "tool.started" and tool_name and tool_name != "_thinking":
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            preview_str = f' "{preview}"' if preview else ""
            ctx.log_queue.put(f"{ts}  {tool_name}:{preview_str}".rstrip())
        # Single-message mode keeps tool_progress quiet, but the evolving preview can carry
        # a transient overlay while it runs: tool-start lines and (opt-in) thinking snippets.
        # Placed before the progress-queue guard — single mode usually runs without a queue.
        sc = self._stream_consumer()
        if sc is not None and getattr(sc, "single_message_mode", False) is True:
            if (event_type == "tool.started" and tool_name and tool_name not in {"clarify", "_thinking"}
                    and ctx._run_still_current() and not self._agent_interrupted()
                    and getattr(sc, "accepts_tool_progress", False) is True):
                msg = self._progress_build_message(tool_name, preview, args)
                if msg:
                    self._progress_emit(msg)
                return
            if ((event_type == "_thinking" or tool_name == "_thinking")
                    and getattr(sc, "accepts_thinking_progress", False) is True):
                thinking_text = (preview if tool_name == "_thinking" else tool_name)
                snippet = " ".join(str(thinking_text or "").split())[:160]
                if snippet and ctx._run_still_current() and not self._agent_interrupted():
                    sc.on_tool_progress(f"💭 {snippet}")
                return
        if not ctx.progress_queue or not ctx._run_still_current():
            return
        if event_type == "tool.completed" and not ctx.long_tool_hint_fired[0]:
            self._progress_onboarding_hint(kwargs)
            return
        # "_thinking" is assistant scratch text between tool calls, never ordinary tool progress:
        # only relayed when the platform explicitly opted into thinking_progress.
        if event_type == "_thinking" or tool_name == "_thinking":
            thinking_text = (preview if tool_name == "_thinking" else tool_name) if ctx._thinking_enabled else None
            if thinking_text:
                ctx.progress_queue.put(t("gateway.progress.thinking_prefix", text=thinking_text))
            return
        # Native task cards consume the ID-bearing tool_start/tool_complete callbacks instead;
        # name-correlated text events would duplicate cards and mispair concurrent same-tool calls.
        if ctx._native_slack_task_cards and event_type in {"tool.started", "tool.completed"}:
            return
        # tool_progress off → only _thinking passes (above). Only tool.started renders. clarify:
        # send_clarify IS the user-facing rendering (a bubble would duplicate it, and verbose mode
        # would dump the raw args JSON right under the prompt). Post-`stop`: N parallel tool calls
        # fire N tool.started events before the interrupt check, so a late stop must not render them.
        if (
            not ctx.tool_progress_enabled
            or event_type != "tool.started"
            # The adapter's send_clarify IS the user-facing rendering (interactive buttons or the
            # numbered-text fallback), so a progress bubble is pure duplication — and in verbose mode it
            # dumps the raw tool-call args JSON into the chat. Because
            # the progress queue drains on a background task, that raw JSON typically lands right underneath
            # the rendered prompt (#52374).
            or tool_name == "clarify"
            or self._agent_interrupted()
        ):
            return
        # "new" mode: only report when tool changes
        if ctx.progress_mode == "new" and tool_name == ctx.last_tool[0]:
            return
        ctx.last_tool[0] = tool_name
        msg = self._progress_build_message(tool_name, preview, args)
        if msg is not None:
            self._progress_emit(msg)
