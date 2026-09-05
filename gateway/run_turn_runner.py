"""Per-turn callback runner (progress/status/voice/run_sync) for the gateway agent turn.

``TurnRunner`` owns the per-turn callbacks ``GatewayRunner._run_agent_inner`` binds. ``gateway.run``
internals are imported lazily inside method bodies (import cycle), so ``patch("gateway.run.X")``
keeps intercepting them at call time.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import queue
import re
import threading
import time
from contextlib import suppress
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent.interrupt_compat import _accepts_keyword
from agent.replay_cleanup import strip_stale_dangerous_confirmations
from gateway.config import Platform
from gateway.media_repair import repair_explicit_computer_use_media_paths
from gateway.platforms.base import BasePlatformAdapter
from gateway.turn_context import TurnContext
from hermes_cli.config import cfg_get
from utils import is_truthy_value

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class TurnRunner:
    """Per-turn collaborator carrying ``GatewayRunner._run_agent_inner``'s tool-progress callbacks."""

    def __init__(self, runner: "GatewayRunner", ctx: TurnContext) -> None:
        self._runner = runner
        self._ctx = ctx

    # ── shared thread→loop plumbing ─────────────────────────────────────────────────────────

    def _schedule(self, coro, log_message: str, loop=None):
        """Hop a coroutine from the agent's sync worker thread onto the gateway loop."""
        from gateway.run import safe_schedule_threadsafe
        return safe_schedule_threadsafe(
            coro, self._ctx._loop_for_step if loop is None else loop, logger=logger, log_message=log_message,
        )

    def _agent_interrupted(self) -> bool:
        """True once the user sent `stop` (agent_holder[0] is the shared agent handle)."""
        try:
            agent = self._ctx.agent_holder[0] if self._ctx.agent_holder else None
            return bool(agent is not None and getattr(agent, "is_interrupted", False))
        except Exception:
            return False

    def _stream_consumer(self):
        holder = self._ctx.stream_consumer_holder
        return holder[0] if holder else None

    def _drain_progress_queue(self) -> None:
        q = self._ctx.progress_queue
        with suppress(Exception):
            while not q.empty():
                q.get_nowait()

    def _track_progress_result(self, result) -> None:
        """Remember a delivered progress/status message id for end-of-turn cleanup."""
        ctx = self._ctx
        if ctx._cleanup_progress and getattr(result, "success", False) and getattr(result, "message_id", None):
            ctx._cleanup_msg_ids.append(str(result.message_id))

    def _track_future_cleanup_id(self, fut) -> None:
        try:
            res = fut.result()
        except Exception:
            return
        self._track_progress_result(res)

    # ── progress_callback (agent thread → progress queue) ───────────────────────────────────

    def progress_callback(self, event_type: str, tool_name: str = None, preview: str = None, args: dict = None, **kwargs):
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
                ctx.progress_queue.put(f"💬 {thinking_text}")
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
            # dumps the raw tool-call args JSON ({"question": ..., "choices": [...]}) into the chat. Because
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

    def _progress_subagent_notice(self, preview, kwargs: dict) -> None:
        """Only terminal failure statuses render (same notice rail as credit warnings)."""
        ctx = self._ctx
        status = kwargs.get("status")
        try:
            from tools.delegate_tool import SUBAGENT_FAILURE_STATUSES, format_subagent_failure_line
            if status in SUBAGENT_FAILURE_STATUSES and ctx._run_still_current():
                line = format_subagent_failure_line(
                    kwargs.get("goal"), status, error=kwargs.get("summary") or preview,
                    duration_seconds=kwargs.get("duration_seconds"),
                )
                self._schedule(self._runner._deliver_platform_notice(ctx.source, line), "subagent failure notice scheduling error")
        except Exception:
            logger.debug("subagent failure notice failed", exc_info=True)

    def _progress_live_status(self, event_type: str, tool_name, args) -> None:
        """Live status line (Slack assistant status): stash the tool phrase on the adapter; the
        _keep_typing refresh renders it. Plain dict write, safe from the sync worker thread."""
        ctx = self._ctx
        adapter = ctx._live_status_adapter
        if adapter is None or ctx._live_status_mode == "off" or tool_name == "_thinking":
            return
        try:
            if event_type == "tool.started" and tool_name and ctx._run_still_current():
                from agent.display import build_status_phrase
                adapter.set_status_text(ctx.source.chat_id, build_status_phrase(tool_name, args if ctx._live_status_mode == "full" else None))
            elif event_type == "tool.completed":
                # Between tools the model is genuinely "thinking" again — revert to the static default.
                adapter.set_status_text(ctx.source.chat_id, None)
        except Exception as err:
            logger.debug("live status update failed: %s", err)

    def _progress_onboarding_hint(self, kwargs: dict) -> None:
        """First-touch onboarding: the first time a tool exceeds _LONG_TOOL_THRESHOLD_S while
        streaming every tool (progress_mode == "all"), append a one-time /verbose hint."""
        from gateway.run import _hermes_home, _load_gateway_config
        ctx = self._ctx
        try:
            if (kwargs.get("duration") or 0) >= ctx._LONG_TOOL_THRESHOLD_S and ctx.progress_mode == "all":
                from agent.onboarding import TOOL_PROGRESS_FLAG, is_seen, mark_seen, tool_progress_hint_gateway
                cfg = _load_gateway_config()
                gate_on = is_truthy_value(cfg_get(cfg, "display", "tool_progress_command"), default=False)
                if gate_on and not is_seen(cfg, TOOL_PROGRESS_FLAG):
                    ctx.long_tool_hint_fired[0] = True
                    ctx.progress_queue.put(tool_progress_hint_gateway())
                    mark_seen(_hermes_home / "config.yaml", TOOL_PROGRESS_FLAG)
        except Exception as err:
            logger.debug("tool-progress onboarding hint failed: %s", err)

    @staticmethod
    def _preview_cap() -> int:
        """tool_preview_length (default 40): the one-line preview budget for "all"/"new" modes."""
        from agent.display import get_tool_preview_max_len
        pl = get_tool_preview_max_len()
        return pl if pl > 0 else 40

    def _progress_terminal_blocks(self, adapter, tool_name, args, emoji):
        """(full, short) fenced blocks for a terminal command on markdown platforms, else (None, None).

        No language tag: Slack mrkdwn renders it as a literal first code line. Verbose shows the FULL
        command; "all"/"new" truncate to one line capped at ``tool_preview_length``. Consecutive
        terminal calls drop the repeated header so back-to-back commands render as adjacent blocks.
        """
        from agent.tool_identity import canonical_tool_args, canonical_tool_name

        canonical_name = canonical_tool_name(tool_name)
        canonical_args = canonical_tool_args(tool_name, args)
        if not (
            getattr(adapter, "supports_code_blocks", False)
            and canonical_name == "terminal"
            and isinstance(canonical_args.get("command"), str)
            and canonical_args["command"].strip()
        ):
            return None, None
        cmd_full = canonical_args["command"].rstrip()
        header = "" if self._ctx.last_was_terminal_block[0] else f"{emoji} {canonical_name}\n"
        cap = self._preview_cap()
        from agent.display import build_terminal_preview_line

        cmd_short = build_terminal_preview_line(cmd_full, cap)
        return f"{header}```\n{cmd_full}\n```", f"{header}```\n{cmd_short}\n```"

    def _progress_build_message(self, tool_name, preview, args) -> Optional[str]:
        """Render the progress line. Verbose mode queues directly (no dedup) and returns None."""
        ctx = self._ctx
        from agent.display import get_tool_emoji
        from agent.tool_identity import display_tool_label

        emoji = get_tool_emoji(tool_name, default="⚙️")
        display_label = display_tool_label(tool_name)
        try:
            adapter = self._runner._adapter_for_source(ctx.source)
        except Exception:
            adapter = None
        code_full, code_short = self._progress_terminal_blocks(adapter, tool_name, args, emoji)
        verbose = ctx.progress_mode == "verbose"
        code = code_full if verbose else code_short
        ctx.last_was_terminal_block[0] = code is not None
        if verbose:
            if code is None and args:
                from agent.display import get_tool_preview_max_len
                pl = get_tool_preview_max_len()
                args_str = json.dumps(args, ensure_ascii=False, default=str)
                # tool_preview_length 0 (default) = no truncation in verbose mode; the user asked
                # for full detail and platform message-length limits handle the rest.
                if pl > 0 and len(args_str) > pl:
                    args_str = args_str[:pl - 3] + "..."
                code = f"{emoji} {tool_name}({list(args.keys())})\n{args_str}"
            elif code is None:
                code = f"{emoji} {tool_name}: \"{preview}\"" if preview else f"{emoji} {tool_name}..."
            ctx.progress_queue.put(code)
            return None
        if code is not None:
            return code
        if not preview:
            return f"{emoji} {display_label}..."
        from agent.display import get_tool_verb, prepare_tool_preview, tool_verb_connector, verb_drops_preview
        prepared = prepare_tool_preview(tool_name, args, fallback=preview, max_len=self._preview_cap())
        preview = adapter.format_tool_preview(prepared) if adapter is not None else prepared.text
        # Friendly labels: human-phrased line for built-in tools ("🔍 Searching the web for ...")
        # by prefixing the verb onto the computed preview, so the command/url/query is kept.
        verb = get_tool_verb(tool_name)
        if not verb:
            # No curated verb, so the tool's own name is what the user sees — the one path where a raw
            # wire name leaks. Strip the MCP scaffolding; a third-party server stays visible as
            # "linear · get_issue". A tool with no preview rule yields no preview and the caller's
            # fallback is the tool name itself, which rendered as `name: "name"` — say it once.
            if preview in (tool_name, display_label):
                return f"{emoji} {display_label}"
            return f"{emoji} {display_label}: \"{preview}\""
        return f"{emoji} {verb}" if verb_drops_preview(tool_name) else f"{emoji} {verb}{tool_verb_connector(tool_name)}{preview}"

    def _progress_emit(self, msg: str) -> None:
        """Dedup consecutive identical lines (execute_code boilerplate), then route to the native
        stream bubble when the consumer accepts tool progress, else the progress queue."""
        ctx = self._ctx
        sc = self._stream_consumer()
        native = sc is not None and getattr(sc, "accepts_tool_progress", False)
        if msg == ctx.last_progress_msg[0]:
            ctx.repeat_count[0] += 1
            if native:
                sc.on_tool_progress(f"{msg} (×{ctx.repeat_count[0] + 1})")
            else:
                ctx.progress_queue.put(("__dedup__", msg, ctx.repeat_count[0]))
            return
        ctx.last_progress_msg[0], ctx.repeat_count[0] = msg, 0
        if native:
            sc.on_tool_progress(msg)
        else:
            ctx.progress_queue.put(msg)

    # ── Slack-native task cards (progress-queue drain) ──────────────────────────────────────

    @dataclasses.dataclass
    class _TaskCardState:
        """Task-card rail state for ``_send_native_task_card_progress``."""
        adapter: Any
        tasks: Dict[str, Dict[str, str]] = dataclasses.field(default_factory=dict)
        task_order: List[str] = dataclasses.field(default_factory=list)
        fallback_msg_id: Optional[str] = None
        native_failed: bool = False
        anonymous_seq: int = 0

        @staticmethod
        def _compact(value: Any, limit: int = 120) -> str:
            text = re.sub(r"\s+", " ", str(value or "")).strip()
            return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."

        def visible_tasks(self) -> List[Dict[str, str]]:
            return [self.tasks[task_id] for task_id in self.task_order[-8:]]

        def fallback_text(self) -> str:
            labels = {"in_progress": "running", "complete": "complete", "error": "error"}
            lines = [f"- {t['title']} - {labels.get(t['status'], t['status'])}" for t in self.visible_tasks()]
            return "Hermes is working\n" + "\n".join(lines)

        def _upsert(self, call_id: str, title: str) -> Dict[str, str]:
            if call_id not in self.tasks:
                self.task_order.append(call_id)
            self.tasks[call_id] = {"id": call_id, "title": self._compact(title), "status": "in_progress"}
            return self.tasks[call_id]

        def apply_event(self, raw: Any) -> bool:
            event_type = raw.get("type") if isinstance(raw, dict) else None
            if event_type not in {"tool.started", "tool.completed"}:
                return False
            call_id = str(raw.get("tool_call_id") or "")
            if not call_id:
                self.anonymous_seq += 1
                call_id = f"anonymous_{self.anonymous_seq}"
            tool_name = str(raw.get("tool_name") or "tool")
            if event_type == "tool.started":
                preview = self._compact(raw.get("preview"), 64)
                self._upsert(call_id, f"{tool_name} - {preview}" if preview else tool_name)
                return True
            # Completion-only events are rare but valid on some runtimes; keep their real ID instead
            # of guessing a same-name pending call.
            task = self.tasks.get(call_id) or self._upsert(call_id, tool_name)
            task["status"] = "error" if raw.get("is_error") else "complete"
            return True

    async def _task_card_send_or_edit_fallback(self, st) -> None:
        ctx = self._ctx
        text = st.fallback_text()
        if st.fallback_msg_id:
            result = await st.adapter.edit_message(
                chat_id=ctx.source.chat_id, message_id=st.fallback_msg_id, content=text, metadata=ctx._progress_metadata,
            )
            if getattr(result, "success", False):
                return
        result = await self._send_progress_text(st, text)
        if getattr(result, "success", False) and getattr(result, "message_id", None):
            st.fallback_msg_id = str(result.message_id)

    async def _task_card_publish(self, st) -> None:
        ctx = self._ctx
        if not st.tasks:
            return
        if not st.native_failed:
            result = await st.adapter.send_native_task_card_progress(
                chat_id=ctx.source.chat_id, tasks=st.visible_tasks(), title="Hermes is working",
                reply_to=ctx._progress_reply_to, metadata=ctx._progress_metadata, fallback_text=st.fallback_text(),
            )
            if getattr(result, "success", False):
                return
            st.native_failed = True
            logger.warning(
                "Slack native task-card progress failed; falling back "
                "to an editable text update: %s", getattr(result, "error", "unknown error"),
            )
        # Once the native rail fails, every later lifecycle event edits the same fallback message.
        await self._task_card_send_or_edit_fallback(st)

    def _task_card_drain(self, st) -> bool:
        changed = False
        try:
            while True:
                changed = st.apply_event(self._ctx.progress_queue.get_nowait()) or changed
        except queue.Empty:
            pass
        except Exception:
            logger.debug("Slack native progress queue drain failed", exc_info=True)
        return changed

    async def _send_native_task_card_progress(self, adapter) -> None:
        """Drain the progress queue into Slack-native plan/task cards; on any native failure, fall
        back to an editable in-thread message so progress stays live.

        See #29483.
        """
        ctx = self._ctx
        st = self._TaskCardState(adapter)
        try:
            while ctx._run_still_current():
                try:
                    raw = ctx.progress_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue
                if not self._agent_interrupted() and st.apply_event(raw):
                    await self._task_card_publish(st)
        except asyncio.CancelledError:
            if self._task_card_drain(st) and ctx._run_still_current() and not self._agent_interrupted():
                await self._task_card_publish(st)
        finally:
            if hasattr(adapter, "stop_native_task_card_progress"):
                # Best-effort on the turn-cleanup path: an escaping transport exception would skip
                # final-delivery logic (cleanup awaits catch only CancelledError).
                try:
                    await adapter.stop_native_task_card_progress(
                        ctx.source.chat_id, reply_to=ctx._progress_reply_to, metadata=ctx._progress_metadata,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug("task-card stop failed during turn cleanup", exc_info=True)

    # ── editable progress bubbles (progress-queue drain) ────────────────────────────────────

    @dataclasses.dataclass
    class _ProgressEditState:
        """Mutable editable-bubble state shared by ``send_progress_messages`` and its helpers."""
        adapter: Any
        progress_lines: list
        progress_msg_id: Any
        can_edit: bool
        _progress_len_fn: Any
        _PROGRESS_TEXT_LIMIT: int
        _edit_accepts_metadata: bool

    def _progress_edit_state(self, adapter) -> "TurnRunner._ProgressEditState":
        ctx = self._ctx
        len_fn = adapter.message_len_fn if isinstance(adapter, BasePlatformAdapter) else len
        try:
            raw_limit = int(getattr(adapter, "MAX_MESSAGE_LENGTH", 4000) or 4000)
        except Exception:
            raw_limit = 4000
        # Per-chat resolution (relay adapter fronting N platforms): cap and length unit follow the
        # chat's underlying platform; native adapters return their scalar/property unchanged.
        if isinstance(adapter, BasePlatformAdapter):
            with suppress(Exception):
                raw_limit = int(adapter.max_message_length_for_chat(ctx.source.chat_id) or 4000)
                len_fn = adapter.message_len_fn_for_chat(ctx.source.chat_id)
        return self._ProgressEditState(
            adapter=adapter, progress_lines=[], progress_msg_id=None,
            # "separate" = one message per tool (pre-v0.9 behavior)
            can_edit=ctx.progress_grouping != "separate",
            _progress_len_fn=len_fn,
            # Leave room for platform quirks / formatting; tiny test adapters keep a usable limit.
            _PROGRESS_TEXT_LIMIT=max(1, raw_limit - (64 if raw_limit > 128 else 0)),
            # Overflow edits pass metadata (Telegram topic/thread routing) only when edit_message takes it.
            _edit_accepts_metadata=bool(ctx._progress_metadata) and _accepts_keyword(adapter.edit_message, "metadata"),
        )

    async def _edit_progress_message(self, st, message_id: str, content: str):
        ctx = self._ctx
        kwargs = {"chat_id": ctx.source.chat_id, "message_id": message_id, "content": content}
        if getattr(st.adapter, "REQUIRES_EDIT_FINALIZE", False):
            kwargs["finalize"] = True
        if st._edit_accepts_metadata:
            kwargs["metadata"] = ctx._progress_metadata
        return await st.adapter.edit_message(**kwargs)

    @staticmethod
    def _progress_text(lines: list) -> str:
        return "\n".join(str(line) for line in lines)

    def _split_progress_groups(self, st, lines: list) -> list[list]:
        """Partition progress lines into platform-sized editable bubbles."""
        groups: list[list] = []
        current: list = []
        for line in lines:
            candidate = current + [line]
            if current and st._progress_len_fn(self._progress_text(candidate)) > st._PROGRESS_TEXT_LIMIT:
                groups.append(current)
                candidate = [line]
            current = candidate
        return groups + ([current] if current else [])

    async def _send_progress_text(self, st, text: str):
        ctx = self._ctx
        result = await st.adapter.send(
            chat_id=ctx.source.chat_id, content=text, reply_to=ctx._progress_reply_to, metadata=ctx._progress_metadata,
        )
        self._track_progress_result(result)
        return result

    async def _roll_progress_overflow_if_needed(self, st) -> bool:
        """Start fresh editable progress bubbles before a bubble exceeds limit.

        Returns True when it delivered/split the buffer or a transient edit failure left it
        intact for retry — either way the caller skips the normal send/edit path this tick.
        """
        if not st.progress_lines or not st.can_edit:
            return False
        groups = self._split_progress_groups(st, st.progress_lines)
        if len(groups) <= 1:
            return False
        if st.progress_msg_id is not None:
            result = await self._edit_progress_message(st, st.progress_msg_id, self._progress_text(groups[0]))
            if not result.success:
                if getattr(result, "retryable", False):
                    logger.debug("[%s] Transient overflow edit failure — keeping can_edit=True", st.adapter.name)
                    return True
                st.can_edit = False
                # Fall back to the existing non-edit behavior.
                return False
            groups = groups[1:]
        for group in groups:
            result = await self._send_progress_text(st, self._progress_text(group))
            if result.success and result.message_id:
                st.progress_msg_id = result.message_id
        # The newest continuation is the only mutable bubble: keep just its lines so later
        # edits update it instead of replaying the full transcript into new messages.
        st.progress_lines = groups[-1]
        return True

    @staticmethod
    def _is_reset_marker(raw) -> bool:
        return isinstance(raw, tuple) and len(raw) >= 1 and raw[0] == "__reset__"

    def _reset_progress_bubble(self, st) -> None:
        """Content bubble landed — close the tool-progress bubble so the next tool starts fresh
        below it; else tool edits hit the ORIGINAL message above (out of order)."""
        st.progress_msg_id, st.progress_lines = None, []
        self._ctx.last_progress_msg[0], self._ctx.repeat_count[0] = None, 0

    def _progress_absorb(self, st, raw) -> Any:
        """Fold a queue item into the bubble buffer; returns the line to render this tick."""
        if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
            _, base_msg, count = raw
            if not st.progress_lines:
                return base_msg
            st.progress_lines[-1] = f"{base_msg} (×{count + 1})"
            return st.progress_lines[-1]
        st.progress_lines.append(raw)
        return raw

    async def _flush_progress_edit(self, st) -> None:
        if st.can_edit and st.progress_lines and st.progress_msg_id:
            with suppress(Exception):
                await self._edit_progress_message(st, st.progress_msg_id, self._progress_text(st.progress_lines))

    async def _drain_progress_on_cancel(self, st) -> None:
        ctx = self._ctx
        with suppress(Exception):
            while not ctx.progress_queue.empty():
                raw = ctx.progress_queue.get_nowait()
                if self._is_reset_marker(raw):
                    # Content-bubble marker during drain: close the current progress bubble
                    # and start a fresh one for tool lines that arrived after.
                    await self._roll_progress_overflow_if_needed(st)
                    await self._flush_progress_edit(st)
                    self._reset_progress_bubble(st)
                else:
                    self._progress_absorb(st, raw)
                    await self._roll_progress_overflow_if_needed(st)
        # Final edit with all remaining tools (only if editing works)
        if st.can_edit and st.progress_lines and st.progress_msg_id:
            await self._roll_progress_overflow_if_needed(st)
        await self._flush_progress_edit(st)

    async def _progress_restore_typing(self, st) -> None:
        ctx = self._ctx
        await asyncio.sleep(0.3)
        if ctx._run_still_current():
            await st.adapter.send_typing(ctx.source.chat_id, metadata=ctx._progress_metadata)

    async def _progress_send_or_edit(self, st, msg) -> bool:
        """Deliver this tick's bubble. Returns False on a transient edit failure (retry next tick).

        Transient network errors (ConnectError, timeouts) must not disable editing; only permanent
        failures (not found, permissions) set can_edit=False. Flood control backs off but keeps editing.
        """
        if st.can_edit and st.progress_msg_id is not None:
            result = await self._edit_progress_message(st, st.progress_msg_id, "\n".join(st.progress_lines))
            if result.success:
                return True
            if getattr(result, "retryable", False):
                logger.debug("[%s] Transient edit failure — keeping can_edit=True", st.adapter.name)
                return False
            if any(w in (getattr(result, "error", "") or "").lower() for w in ("flood", "retry after")):
                logger.info("[%s] Progress edit flood control, backing off", st.adapter.name)
            else:
                st.can_edit = False
            await self._send_progress_text(st, msg)
            return True
        # First tool: send all accumulated text as a new message; editing unsupported: just this line.
        result = await self._send_progress_text(st, "\n".join(st.progress_lines) if st.can_edit else msg)
        if result.success and result.message_id:
            st.progress_msg_id = result.message_id
        return True

    async def send_progress_messages(self):
        ctx = self._ctx
        adapter = self._runner._adapter_for_source(ctx.source) if ctx.progress_queue else None
        if not adapter:
            return
        if ctx._native_slack_task_cards and hasattr(adapter, "send_native_task_card_progress"):
            await self._send_native_task_card_progress(adapter)
            return
        # Skip tool progress for platforms that can't edit messages (e.g. iMessage/BlueBubbles):
        # each update would be a separate bubble. getattr, not attribute access: duck-typed
        # adapters (test fakes, minimal plugins) may lack edit_message — treated as "can't edit".
        adapter_edit = getattr(type(adapter), "edit_message", None)
        if adapter_edit is None or adapter_edit is BasePlatformAdapter.edit_message:
            self._drain_progress_queue()
            return
        st = self._progress_edit_state(adapter)
        last_edit_ts = 0.0
        EDIT_INTERVAL = 1.5  # Minimum seconds between edits (Telegram flood control)
        while True:
            try:
                if not ctx._run_still_current():
                    self._drain_progress_queue()
                    return
                raw = ctx.progress_queue.get_nowait()
                # Drain silently when interrupted: events queued in the window between tool parse
                # and interrupt processing should not render as bubbles.
                if self._agent_interrupted():
                    await asyncio.sleep(0)
                    continue
                if self._is_reset_marker(raw):
                    self._reset_progress_bubble(st)
                    continue
                msg = self._progress_absorb(st, raw)
                if not await self._roll_progress_overflow_if_needed(st):
                    # Throttle edits: batch rapid tool updates into fewer API calls (grammY pattern:
                    # proactively rate-limit rather than react to 429s). Loop back to drain further
                    # queued messages before sending a single batched edit.
                    remaining = EDIT_INTERVAL - (time.monotonic() - last_edit_ts)
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                        continue
                    if not ctx._run_still_current():
                        return
                    if not await self._progress_send_or_edit(st, msg):
                        continue
                last_edit_ts = time.monotonic()
                await self._progress_restore_typing(st)
            except queue.Empty:
                await asyncio.sleep(0.3)
            except asyncio.CancelledError:
                await self._drain_progress_on_cancel(st)
                return
            except Exception as e:
                logger.error("Progress message error: %s", e)
                await asyncio.sleep(1)

    # ── ID-bearing lifecycle callbacks (agent thread) ───────────────────────────────────────

    def voice_ack_callback(self, call_id, tool_name, args):
        """tool_start_callback: speak a one-time ack in the voice channel."""
        ctx = self._ctx
        if ctx._voice_ack_fired[0] or ctx._voice_ack_guild[0] is None or not ctx._run_still_current():
            return
        ctx._voice_ack_fired[0] = True
        adapter = self._runner.adapters.get(Platform.DISCORD)
        if adapter is None or not hasattr(adapter, "play_ack_in_voice"):
            return
        try:
            self._schedule(
                adapter.play_ack_in_voice(ctx._voice_ack_guild[0]), "voice ack scheduling error", loop=ctx._voice_ack_loop,
            )
        except Exception as err:
            logger.debug("voice ack schedule failed: %s", err)

    # Slack-native task cards ride agent.tool_start_callback / tool_complete_callback so start and
    # completion correlate by the REAL tool-call id; name-correlated progress_callback text events
    # would duplicate cards and mispair concurrent calls.

    def _native_card_gate(self) -> bool:
        ctx = self._ctx
        return bool(ctx.progress_queue) and ctx._run_still_current() and not self._agent_interrupted()

    # ── Slack-native task cards: ID-bearing lifecycle callbacks (#29483) ── These ride
    # agent.tool_start_callback / agent.tool_complete_callback so start/completion events correlate by the
    # REAL tool-call id — the name-correlated text events in progress_callback would duplicate cards and
    # mispair concurrent calls to the same tool.
    def native_tool_start_callback(self, call_id, tool_name, args):
        """Queue an ID-correlated native progress start from the agent thread."""
        if not self._native_card_gate():
            return
        from agent.display import build_tool_preview
        name = str(tool_name or "tool")
        self._ctx.progress_queue.put({
            "type": "tool.started", "tool_call_id": str(call_id or ""), "tool_name": name,
            "preview": build_tool_preview(name, args or {}, max_len=64) or "",
        })

    def native_tool_complete_callback(self, call_id, tool_name, args, result):
        """Queue the matching native completion using the real tool-call ID."""
        if not self._native_card_gate():
            return
        from agent.display import _detect_tool_failure
        name = str(tool_name or "tool")
        is_error, _ = _detect_tool_failure(name, result)
        self._ctx.progress_queue.put({
            "type": "tool.completed", "tool_call_id": str(call_id or ""), "tool_name": name, "is_error": bool(is_error),
        })

    def combined_tool_start_callback(self, call_id, tool_name, args):
        """Compose the voice ack + native task-card start consumers."""
        if self._ctx._voice_ack_guild[0] is not None:
            self.voice_ack_callback(call_id, tool_name, args)
        if self._ctx._native_slack_task_cards:
            self.native_tool_start_callback(call_id, tool_name, args)

    # ── hook / status bridges (agent thread → gateway loop) ────────────────────────────────

    def _step_callback_sync(self, iteration: int, prev_tools: list) -> None:
        ctx = self._ctx
        if not ctx._run_still_current():
            return
        # prev_tools may be list[str] or list[dict] with "name"/"result" keys. Normalise so
        # "tool_names" stays backward-compatible for user hooks that do ', '.join(tool_names).
        names = [(t.get("name") or "") if isinstance(t, dict) else str(t) for t in (prev_tools or [])]
        self._schedule(
            ctx._hooks_ref.emit("agent:step", {
                "platform": ctx.source.platform.value if ctx.source.platform else "",
                "user_id": ctx.source.user_id, "session_id": ctx.session_id,
                "iteration": iteration, "tool_names": names, "tools": prev_tools,
            }),
            "agent:step hook scheduling error",
        )

    def _event_callback_sync(self, event_type: str, context: dict) -> None:
        ctx = self._ctx
        try:
            asyncio.run_coroutine_threadsafe(ctx._hooks_ref.emit(event_type, context), ctx._loop_for_step)
        except Exception as e:
            logger.debug("event_callback hook error: %s", e)

    def _status_live(self) -> bool:
        """Status adapter present and this run is still the current generation."""
        return bool(self._ctx._status_adapter) and self._ctx._run_still_current()

    def _send_status_text(self, text: str, metadata, log_message: str) -> None:
        ctx = self._ctx
        self._schedule(ctx._status_adapter.send(ctx._status_chat_id, text, metadata=metadata), log_message)

    def _attach_session_title_callback(self, agent, ctx) -> None:
        """Wire the platform thread-rename lane onto the agent as `_on_session_title`.

        The titler runs in the turn prologue, so attach before the run, not after it.
        """
        try:
            # Gateway auto-title failures are not user-actionable, so never surface them as messages;
            # overriding the failure sink keeps CLI on _emit_auxiliary_failure while gateway logs debug.
            agent._title_failure_callback = lambda task, exc: logger.debug(
                "Gateway auto-title failure suppressed (not user-visible): %s: %s", task, exc,
            )
            session_id = getattr(agent, "session_id", None)
            source = ctx.source
            runner = self._runner
            # Both lanes spend a rate-limited platform call per title, so they use the model's title
            # only (TitleCallback); renaming twice burns Discord's 2-per-10-min budget on a throwaway.
            # Relay Discord predicate is shape-only: whether the connector auto-threaded our reply is
            # only knowable AFTER delivery, so register eagerly and let the rename lane look up the
            # cache at fire time — gating registration on the cache read meant it never registered.
            if runner._is_telegram_topic_lane(source):
                lane = "_schedule_telegram_topic_title_rename"
            elif runner._is_discord_auto_thread_lane(source) or runner._is_relay_discord_channel_lane(source):
                lane = "_schedule_discord_semantic_thread_rename"
            else:
                return
            agent._on_session_title = lambda title, title_source: (
                title_source == "llm" and getattr(runner, lane)(source, session_id, title)
            )
        except Exception:
            logger.debug("Failed to attach session title callback", exc_info=True)

    def _status_callback_sync(self, event_type: str, message: str) -> None:
        from gateway.run import _prepare_gateway_status_message, _redact_gateway_user_facing_secrets, _send_or_update_status_coro
        ctx = self._ctx
        if not self._status_live():
            return
        prepared = _prepare_gateway_status_message(ctx.source.platform, event_type, message)
        if prepared is None:
            logger.debug(
                "status_callback suppressed for %s/%s: %s",
                ctx.source.platform.value if ctx.source.platform else "unknown", event_type,
                _redact_gateway_user_facing_secrets(str(message or ""))[:160],
            )
            return
        fut = self._schedule(
            _send_or_update_status_coro(ctx._status_adapter, ctx._status_chat_id, event_type, prepared, ctx._status_thread_metadata),
            f"status_callback ({event_type}) scheduling error",
        )
        if fut is not None and ctx._cleanup_progress:
            fut.add_done_callback(self._track_future_cleanup_id)

    # ── stream consumer / interim commentary wiring ─────────────────────────────────────────

    def _setup_stream_consumer(self, platform_key):
        ctx = self._ctx
        stream_consumer = None
        # The streaming-TTS consumer is created on the outer loop thread before run_sync launches;
        # run_sync only reads it via the holder for delta-callback wiring.
        stts = ctx.streaming_tts_consumer_holder[0]
        scfg = getattr(getattr(self._runner, 'config', None), 'streaming', None)
        if scfg is None:
            from gateway.config import StreamingConfig
            scfg = StreamingConfig()
        # display.platforms.<plat>.streaming may disable streaming per platform; None = follow global.
        plat_streaming = ctx.resolve_display_setting(ctx.user_config, platform_key, "streaming")
        want_stream_deltas = (
            scfg.enabled and scfg.transport != "off" if plat_streaming is None else bool(plat_streaming)
        )
        want_interim_messages = ctx.interim_assistant_messages_enabled
        if want_stream_deltas or want_interim_messages:
            try:
                from gateway.stream_consumer import GatewayStreamConsumer
                adapter = self._runner._adapter_for_source(ctx.source)
                if adapter:
                    consumer_cfg, pause_typing_before_finalize = self._runner._build_stream_consumer_config(
                        ctx.source, scfg, adapter, on_missing_cursor="raise",
                    )
                    stream_consumer = GatewayStreamConsumer(
                        adapter=adapter, chat_id=ctx.source.chat_id, config=consumer_cfg,
                        metadata=ctx._status_thread_metadata,
                        on_new_message=(
                            (lambda: ctx.progress_queue.put(("__reset__",))) if ctx.progress_queue is not None else None
                        ),
                        on_before_finalize=pause_typing_before_finalize,
                        initial_reply_to_id=ctx.event_message_id, run_still_current=ctx._run_still_current,
                    )
                    ctx.stream_consumer_holder[0] = stream_consumer
            except Exception as err:
                logger.debug("Could not set up stream consumer: %s", err)
        # Deltas tee to the stream consumer (when text streaming is on) and to streaming TTS.
        delta_sinks = [sc for sc in ((stream_consumer if want_stream_deltas else None), stts) if sc is not None]
        stream_delta_cb = None
        if delta_sinks:
            def stream_delta_cb(text: str) -> None:
                if ctx._run_still_current():
                    for sink in delta_sinks:
                        sink.on_delta(text)

        def interim_assistant_cb(text: str, *, already_streamed: bool = False) -> None:
            if not ctx._run_still_current():
                return
            if stream_consumer is not None:
                stream_consumer.on_segment_break() if already_streamed else stream_consumer.on_commentary(text)
            elif not already_streamed and ctx._status_adapter and str(text or "").strip():
                self._send_status_text(text, ctx._status_thread_metadata, "interim_assistant_callback scheduling error")

        return stream_consumer, stream_delta_cb, interim_assistant_cb, want_interim_messages

    # ── agent resolution (cache reuse vs fresh build) ───────────────────────────────────────

    @dataclasses.dataclass
    class _CachedAgentLookup:
        agent: Any = None
        reused: bool = False
        evicted: Any = None  # agent evicted under the lock; released off-lock on a daemon thread

    def _skip_context_files(self, platform_key) -> bool:
        """gateway.platforms.<plat>.skip_context_files: messaging platforms may opt out of
        filesystem-heavy context-file discovery (SOUL.md, AGENTS.md, .cursorrules)."""
        platforms_cfg = (self._ctx.user_config.get("gateway") or {}).get("platforms") or {}
        # ``hermes gateway setup`` writes ``gateway.platforms`` as a LIST of enabled platform names,
        # not a dict; treat any non-dict shape as "no per-platform overrides" rather than crashing.
        if not isinstance(platforms_cfg, dict):
            return False
        return bool((platforms_cfg.get(platform_key) or {}).get("skip_context_files"))

    def _cached_sid_is_dead(self, cache_lock, cache) -> tuple:
        """(peeked cached session_id, is_dead) — checked OUTSIDE the cache lock. "cached sid != current
        sid" normally means an intentional switch (reuse), but the routing-key self-heal yields the same
        shape with an agent bound to a DEAD session; reusing it re-binds the dead sid and loops."""
        ctx = self._ctx
        peek_sid = None
        if cache_lock and cache is not None:
            with cache_lock:
                entry = cache.get(ctx.session_key)
            if entry and len(entry) > 3:
                peek_sid = entry[3]
        dead = False
        if peek_sid is not None and ctx.session_id is not None and peek_sid != ctx.session_id:
            with suppress(Exception):
                dead = self._runner.session_store._is_session_ended_in_db(peek_sid)
        return peek_sid, dead

    def _current_message_count(self):
        """Cross-process write guard input: the session's current DB message_count (or None)."""
        ctx = self._ctx
        if self._runner._session_db is None or not ctx.session_id:
            return None
        count = None
        with suppress(Exception):
            # run_sync is off-loop (executor); sync DB is fine.
            row = self._runner._session_db._db.get_session(ctx.session_id)
            if row:
                count = row.get("message_count", 0)
        return count

    def _pop_cached_agent_for_eviction(self):
        """Evict under the lock but DEFER release (release_clients can block on memory-provider /
        socket teardown while the idle sweeper waits on this lock). The turn rebuilds a fresh agent, so
        the caller does a SOFT release that keeps sandbox / browser / bg processes."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        evicted = self._runner._agent_cache.pop(self._ctx.session_key, None)
        agent = evicted[0] if isinstance(evicted, tuple) and evicted else None
        return agent if agent and agent is not _AGENT_PENDING_SENTINEL else None

    def _lookup_cached_agent(self, sig, cache_lock, cache, max_iterations, peek_sid, dead, msg_count):
        ctx = self._ctx
        out = self._CachedAgentLookup()
        if not (cache_lock and cache is not None):
            return out
        with cache_lock:
            cached = cache.get(ctx.session_key)
            if not (cached and cached[1] == sig):
                return out
            # cached[2] = message_count at cache time (stale when a second process appended rows);
            # cached[3] = the session_id the snapshot was taken for.
            cached_mc = cached[2] if len(cached) > 2 else None
            cached_sid = cached[3] if len(cached) > 3 else None
            # Same session_key, other conversation: the counts track DIFFERENT DB rows, so the
            # comparison is meaningless — REUSE rather than bust the prompt cache on every switch.
            sid_mismatch = cached_sid is not None and ctx.session_id is not None and cached_sid != ctx.session_id
            # Re-validate the outside-lock dead-session peek against the tuple read under THIS lock:
            # a stale "dead" verdict must never be applied to a different (possibly live) agent.
            if sid_mismatch and dead and cached_sid == peek_sid:
                logger.info(
                    "Agent cache invalidated for session %s: "
                    "cached agent's session_id %s is ended in "
                    "state.db (stale self-heal artifact, "
                    "#54878 x #54947) — discarding instead of "
                    "reusing across the routing recovery", ctx.session_key, cached_sid,
                )
            elif not sid_mismatch and cached_mc is not None and msg_count is not None and msg_count != cached_mc:
                logger.info(
                    "Agent cache invalidated for session %s: "
                    "message_count changed (%s -> %s), "
                    "possible cross-process write", ctx.session_key, cached_mc, msg_count,
                )
            else:
                out.agent = cached[0]
                # Refresh LRU order so cap enforcement evicts truly-oldest entries.
                if hasattr(cache, "move_to_end"):
                    with suppress(KeyError):
                        cache.move_to_end(ctx.session_key)
                self._runner._init_cached_agent_for_turn(out.agent, ctx._interrupt_depth)
                # Cached agent may have been created with old config.
                out.agent.max_iterations = max_iterations
                logger.debug("Reusing cached agent for session %s", ctx.session_key)
                out.reused = True
                return out
            out.evicted = self._pop_cached_agent_for_eviction()
        return out

    def _release_evicted_agent(self, agent) -> None:
        """Off-lock soft release on a daemon thread so teardown never blocks the gateway loop."""
        self._runner._spawn_release_thread(
            self._runner._release_evicted_agent_soft, (agent,), f"agent-xproc-evict-{str(self._ctx.session_key)[:24]}",
            inline_fallback=True,
        )

    def _build_fresh_agent(self, turn_route, platform_key, combined_ephemeral, max_iterations,
                           reasoning_config, pr, skip_context_files):
        from gateway.run import _checkpoint_agent_kwargs
        ctx = self._ctx
        runner = self._runner
        src = ctx.source
        return ctx.AIAgent(
            model=turn_route["model"], **turn_route["runtime"], **_checkpoint_agent_kwargs(ctx.user_config),
            max_iterations=max_iterations, quiet_mode=True, verbose_logging=False,
            enabled_toolsets=ctx.enabled_toolsets, disabled_toolsets=ctx.disabled_toolsets,
            ephemeral_system_prompt=combined_ephemeral or None,
            prefill_messages=runner._prefill_messages or None,
            reasoning_config=reasoning_config, service_tier=runner._service_tier,
            request_overrides=turn_route.get("request_overrides"),
            providers_allowed=pr.get("only"), providers_ignored=pr.get("ignore"), providers_order=pr.get("order"),
            provider_sort=pr.get("sort"), provider_require_parameters=pr.get("require_parameters", False),
            provider_data_collection=pr.get("data_collection"),
            session_id=ctx.session_id, platform=platform_key,
            user_id=src.user_id, user_id_alt=src.user_id_alt, user_name=src.user_name,
            chat_id=src.chat_id, chat_name=src.chat_name, chat_type=src.chat_type, thread_id=src.thread_id,
            gateway_session_key=ctx.session_key,
            session_db=getattr(runner._session_db, "_db", runner._session_db),
            # Reload from disk — do not reuse the startup snapshot.
            # See #60955.
            fallback_model=self._runner._refresh_fallback_model(),
            skip_context_files=skip_context_files,
            # Keep the persona even with minimal context: soul identity is one small file.
            load_soul_identity=True,
        )

    def _resolve_turn_agent(self, turn_route, platform_key, combined_ephemeral, max_iterations, reasoning_config, pr):
        """Reuse this session's cached AIAgent (frozen system prompt + tool schemas → prompt cache
        hits) or build a fresh one. Returns (agent, reused_cached_agent)."""
        ctx = self._ctx
        runner = self._runner
        skip_context_files = self._skip_context_files(platform_key)
        sig = runner._agent_config_signature(
            turn_route["model"], turn_route["runtime"], ctx.enabled_toolsets, combined_ephemeral,
            cache_keys=runner._extract_cache_busting_config(ctx.user_config),
            user_id=getattr(ctx.source, "user_id", None),
            user_id_alt=getattr(ctx.source, "user_id_alt", None),
            skip_context_files=skip_context_files,
        )
        cache_lock = getattr(runner, "_agent_cache_lock", None)
        cache = getattr(runner, "_agent_cache", None)
        peek_sid, dead = self._cached_sid_is_dead(cache_lock, cache)
        msg_count = self._current_message_count()
        found = self._lookup_cached_agent(sig, cache_lock, cache, max_iterations, peek_sid, dead, msg_count)
        agent = found.agent
        # Lock released — refresh the reused agent's fallback chain from disk OUTSIDE the cache lock
        # (disk I/O under the lock stalls the idle-sweep watcher and Discord heartbeats). A chain
        # configured after caching must reach the next turn; per-session serialization keeps it safe.
        if found.reused and agent is not None:
            self._runner._apply_fallback_chain_to_agent(agent, runner._refresh_fallback_model())
        if found.evicted is not None:
            self._release_evicted_agent(found.evicted)
        if agent is None:
            agent = self._build_fresh_agent(
                turn_route, platform_key, combined_ephemeral, max_iterations, reasoning_config, pr, skip_context_files,
            )
            displaced = None
            if cache_lock and cache is not None:
                with cache_lock:
                    # A fresh agent may DISPLACE a cached one under the same key (signature change,
                    # session switch, memory-pressure rebuild). The eviction paths release what they
                    # pop; a plain overwrite released nothing, so the displaced agent's provider
                    # session — on the claude-agent-sdk lane its Claude Code CLI child — was dropped
                    # to GC, which never reaps a live subprocess. Capture it, release it below.
                    prev_entry = cache.get(ctx.session_key)
                    prev = prev_entry[0] if isinstance(prev_entry, tuple) and prev_entry else None
                    if prev is not None and prev is not agent:
                        displaced = prev
                    # Record the snapshot's session_id with message_count so the cross-process guard
                    # can skip the meaningless count comparison if the active session_id switches.
                    cache[ctx.session_key] = (agent, sig, msg_count, ctx.session_id)
                    runner._enforce_agent_cache_cap()
            if displaced is not None:
                runner._release_displaced_agent(displaced, ctx.session_key)
            logger.debug("Created new agent for session %s (sig=%s)", ctx.session_key, sig)
        return agent, found.reused

    # ── per-turn agent wiring ───────────────────────────────────────────────────────────────

    def _notice_callback_sync(self, notice) -> None:
        """Credits / out-of-band notices (usage bands, depletion, restored) fire from the agent's
        sync worker thread; hop onto the gateway loop. Fired-once latch lives on the cached agent."""
        from gateway.run import render_notice_line
        if not self._status_live():
            return
        try:
            line = render_notice_line(notice)
        except Exception:
            logger.debug("render_notice_line failed", exc_info=True)
            return
        if line:
            self._schedule(self._runner._deliver_platform_notice(self._ctx.source, line), "notice_callback delivery scheduling error")

    def _make_bg_review_callbacks(self):
        """(send, release): background-review messages ("💾 Memory updated") are held until the
        adapter's post-delivery hook releases them after the main response lands."""
        from gateway.run import _interim_metadata, _non_conversational_metadata
        ctx = self._ctx
        release_evt = threading.Event()
        pending: list[str] = []
        pending_lock = threading.Lock()

        def deliver(message: str) -> None:
            if self._status_live():
                self._send_status_text(
                    message,
                    _interim_metadata(_non_conversational_metadata(ctx._status_thread_metadata, platform=ctx.source.platform)),
                    "background_review_callback scheduling error",
                )

        def release() -> None:
            release_evt.set()
            with pending_lock:
                queued = list(pending)
                pending.clear()
            for message in queued:
                deliver(message)

        def send(message: str) -> None:
            if not self._status_live():
                return
            if not release_evt.is_set():
                with pending_lock:
                    if not release_evt.is_set():
                        pending.append(message)
                        return
            deliver(message)

        return send, release

    @staticmethod
    def _merge_turn_request_overrides(agent, turn_route) -> None:
        """Merge, never overwrite: init-time request overrides (e.g. a custom provider's extra_body)
        must survive every reused-agent turn. Drop only the PREVIOUS turn's routing overrides before
        layering this turn's, so stale per-turn values never linger."""
        overrides = dict(getattr(agent, "request_overrides", {}) or {})
        for key, value in (getattr(agent, "_gateway_turn_request_overrides", {}) or {}).items():
            if overrides.get(key) == value:
                overrides.pop(key, None)
        turn_overrides = dict(turn_route.get("request_overrides") or {})
        overrides.update(turn_overrides)
        agent.request_overrides = overrides
        agent._gateway_turn_request_overrides = turn_overrides

    def _wire_turn_agent_callbacks(self, agent, turn_route, reasoning_config,
                                   stream_delta_cb, interim_assistant_cb, want_interim_messages):
        """Per-message state — callbacks and reasoning config change every turn, so they aren't
        baked into the cached agent."""
        ctx = self._ctx
        runner = self._runner
        # ALWAYS attached (never gated to None): its body gates each event class, and subagent-
        # failure notices must fire even with tool_progress/thinking off.
        agent.tool_progress_callback = ctx.progress_callback
        # Discord's one-time voice ack and Slack's task cards both ride the authoritative start
        # callback, so neither infers identity from tool names.
        agent.tool_start_callback = (
            (ctx.native_tool_start_callback or ctx.voice_ack_callback)
            if (ctx._voice_ack_guild[0] is not None or ctx._native_slack_task_cards) else None
        )
        agent.tool_complete_callback = ctx.native_tool_complete_callback if ctx._native_slack_task_cards else None
        agent.step_callback = ctx._step_callback_sync if ctx._hooks_ref.loaded_hooks else None
        agent.stream_delta_callback = stream_delta_cb
        agent.interim_assistant_callback = interim_assistant_cb if want_interim_messages else None
        agent.status_callback, agent.notice_callback = ctx._status_callback_sync, self._notice_callback_sync
        agent.notice_clear_callback = None  # sends can't be retracted
        agent.event_callback = ctx._event_callback_sync
        agent.reasoning_config, agent.service_tier = reasoning_config, runner._service_tier
        self._merge_turn_request_overrides(agent, turn_route)
        # Must-deliver notes for THIS turn ride the current user message (api_content sidecar), never
        # the system prompt. Assigned unconditionally so a reused agent never replays a stale note.
        agent._gateway_turn_context_notes = "\n\n".join(runner._consume_pending_turn_sidecar_notes(ctx.session_key))
        agent.background_review_callback, bg_release = self._make_bg_review_callbacks()
        # Register the release hook on the adapter so base.py's finally block fires it after the
        # main response is delivered.
        if ctx._status_adapter and ctx.session_key:
            if getattr(type(ctx._status_adapter), "register_post_delivery_callback", None) is not None:
                ctx._status_adapter.register_post_delivery_callback(ctx.session_key, bg_release, generation=ctx.run_generation)
            else:
                pdc = getattr(ctx._status_adapter, "_post_delivery_callbacks", None)
                if pdc is not None:
                    pdc[ctx.session_key] = bg_release
        # display.memory_notifications: off | on (generic "💾 Memory updated", default) | verbose.
        mem_notif = ctx.user_config.get("display", {}).get("memory_notifications")
        if isinstance(mem_notif, bool):
            mem_notif = "on" if mem_notif else "off"
        agent.memory_notifications = str(mem_notif).lower() if mem_notif else "on"
        agent.clarify_callback = self._clarify_callback_sync
        # Thinking between tool calls is independent of tool_progress mode (Mattermost opts in
        # per platform so global scratch-text doesn't leak into threads).
        agent.thinking_progress = ctx._thinking_enabled
        ctx.agent_holder[0] = agent  # interrupt support
        # The titler fires from the turn prologue, so attach the rename lane before the run.
        self._attach_session_title_callback(agent, ctx)
        # Publish turn ownership for /stop, /new, disconnect and shutdown interrupts; older session
        # processes are outside this baseline and remain alive.
        agent._gateway_turn_process_task_id, agent._gateway_turn_process_baseline = ctx.process_task_id, ctx.process_baseline
        ctx.tools_holder[0] = getattr(agent, "tools", None)  # transcript logging

    # ── blocking prompts from the agent thread (approval / clarify) ─────────────────────────

    def _close_native_stream_boundary(self, reason: str, placeholder: str | None = None, reopen: bool = False) -> bool:
        """Native-streaming platforms (e.g. WeCom): an interrupting interaction (approval or clarify
        prompt) must finalize the current stream first, or post-interaction output keeps updating the
        OLD bubble above the prompt. Runs on the agent thread; the consumer serializes via its queue."""
        sc = self._stream_consumer()
        if not (sc and getattr(sc, "_use_native_streaming", False)):
            return True
        cancelled_flag = None
        try:
            boundary = sc.close_for_approval_prompt(placeholder, reason=reason, reopen=reopen)
            # Returns (future, cancelled_flag) or just a future.
            if isinstance(boundary, tuple):
                boundary, cancelled_flag = boundary
            if not hasattr(boundary, "result"):
                return True
            ok = boundary.result(timeout=10)
            if not ok:
                logger.warning(
                    "%s boundary failed to close stream properly — "
                    "prompt may still appear in typing bubble", reason,
                )
            return bool(ok)
        except (TimeoutError, Exception) as err:
            if cancelled_flag is not None:
                cancelled_flag["cancelled"] = True
            logger.warning("%s boundary timed out or failed: %s", reason, err)
            return False

    def _clarify_callback_sync(self, question: str, choices, multi_select: bool = False) -> str:
        """Present a clarify prompt and block on a response (clarify_tool's synchronous contract):
        schedule send_clarify on the gateway loop, block on the primitive's threading.Event with a
        timeout. Returns the response string, or a sentinel when none arrived."""
        from gateway.run import _clarify_send_then_wait
        from tools import clarify_gateway as clarify_mod
        import uuid
        ctx = self._ctx
        if not ctx._status_adapter:
            return ""
        session_key = ctx.session_key or ""
        clarify_id = uuid.uuid4().hex[:10]
        choices = list(choices) if choices else None
        clarify_mod.register(
            clarify_id=clarify_id, session_key=session_key, question=question, choices=choices,
            multi_select=bool(multi_select),
        )
        # Unlike approval, clarify passes reopen=True so the continuation re-opens a native stream
        # below the question; if the re-seed fails the consumer degrades to send() automatically.
        self._close_native_stream_boundary("Clarify", "💬 等待你的选择...", reopen=True)
        # Pause typing: a "thinking..." status must not obscure the prompt or block an "Other" reply
        # on platforms that disable input while typing (Slack Assistant).
        with suppress(Exception):
            ctx._status_adapter.pause_typing_for_chat(ctx._status_chat_id)
        # Ordering barrier: flush buffered assistant prose BEFORE the poll, which goes out on a
        # separate agent-thread-blocking path and would otherwise render ABOVE its own explanation.
        # Best-effort + short timeout so the agent thread never hangs if the consumer isn't running.
        flush = getattr(self._stream_consumer(), "flush_pending_sync", None)
        try:
            if callable(flush):
                flush(timeout=3.0)
        except Exception:
            logger.debug("Stream-consumer flush before clarify prompt failed", exc_info=True)
        fut = self._schedule(
            ctx._status_adapter.send_clarify(
                chat_id=ctx._status_chat_id, question=question, choices=choices, clarify_id=clarify_id,
                session_key=session_key, metadata=ctx._status_thread_metadata,
            ),
            "Clarify send failed to schedule",
        )
        # Boundary rule (see _approval_send_outcome): a send timeout is AMBIGUOUS — the card may
        # have posted with a late ack. Only a definitive failure tears down the registration;
        # ambiguous falls through to the bounded wait so a late reply resolves.
        response = _clarify_send_then_wait(fut, clarify_id=clarify_id, session_key=session_key, clarify_mod=clarify_mod)
        # Only re-arm typing when the user actually answered — the undeliverable sentinel and the
        # timeout/cancellation strings start with '[' and must pass through untouched.
        if not (isinstance(response, str) and response.startswith("[")):
            # Reopen typing IMMEDIATELY, not on the LLM's first post-answer token (native streaming
            # otherwise re-seeds lazily on the first delta: ~48s of dead air). request_reopen_seed is
            # a no-op outside the reopen-pending native state.
            sc = self._stream_consumer()
            if sc is not None:
                try:
                    sc.request_reopen_seed()
                except Exception:
                    logger.debug("request_reopen_seed after clarify answer failed", exc_info=True)
            try:
                ctx._status_adapter.resume_typing_for_chat(ctx._status_chat_id)
            except Exception:
                logger.debug("resume_typing_for_chat after clarify answer failed", exc_info=True)
        return response

    def _approval_notify_sync(self, approval_data: dict) -> None:
        """Send the approval request from the agent thread: the adapter's interactive button
        approvals (``send_exec_approval``) when available, else plain text with ``/approve`` steps."""
        from gateway.run import _approval_send_outcome, _format_exec_approval_fallback, _interim_metadata, _redact_approval_command
        ctx = self._ctx
        adapter = ctx._status_adapter
        # Slack's assistant_threads_setStatus disables the compose box, so the user can't type
        # /approve while "is thinking..." shows. Pausing stops _keep_typing re-setting it; resumed
        # in approve/deny.
        adapter.pause_typing_for_chat(ctx._status_chat_id)
        self._close_native_stream_boundary("Approval")
        # Redact credentials before display: Tirith's findings are already redacted, but the raw
        # command string still leaks secrets. Both the button and plain-text paths use this value.
        cmd = _redact_approval_command(approval_data.get("command", ""))
        desc = approval_data.get("description", "dangerous command")
        flags = {k: approval_data.get(k, d) for k, d in (("allow_permanent", True), ("allow_session", True), ("smart_denied", False))}
        # Check the *class*, not the instance — MagicMock auto-creates attributes in tests.
        if getattr(type(adapter), "send_exec_approval", None) is not None:
            try:
                # The SDK prompt correlator rides on the card so a tap resolves the MATCHING prompt
                # instead of queue[0]; only SDK approvals carry it (tools.approval sets it for
                # pattern_key "claude_sdk_tool").
                tool_use_id = approval_data.get("tool_use_id") or ""
                send_flags = {**flags, "tool_use_id": tool_use_id} if tool_use_id else flags
                fut = self._schedule(
                    adapter.send_exec_approval(
                        chat_id=ctx._status_chat_id, command=cmd, session_key=ctx.session_key or "",
                        description=desc, metadata=ctx._status_thread_metadata, **send_flags,
                    ),
                    "send_exec_approval scheduling error",
                )
                if fut is None:
                    raise RuntimeError("send_exec_approval: loop unavailable")
                outcome = _approval_send_outcome(fut, timeout=15)
                if outcome == "sent":
                    return
                if outcome == "ambiguous":
                    # Timeout ≠ failure: the card may have posted with a late ack. The prompt
                    # registration stays alive so a tap still resolves; re-sending made duplicate
                    # cards + orphaned "/approve: nothing pending".
                    logger.warning(
                        "Button-based approval send timed out — treating "
                        "as possibly-delivered (no re-send; the prompt "
                        "stays armed for a late tap)"
                    )
                    return
                logger.warning("Button-based approval failed (send returned error), falling back to text")
            except Exception as e:
                logger.warning("Button-based approval failed, falling back to text: %s", e)
        # Plain-text prompt with the adapter's typed prefix (e.g. `!approve`): typed "/" is blocked
        # in Slack threads and reserved by Matrix clients.
        msg = _format_exec_approval_fallback(cmd, desc, getattr(adapter, "typed_command_prefix", "/"), **flags)
        try:
            # Mark as approval prompt so WeCom routes through the control lane.
            metadata = {**(ctx._status_thread_metadata or {}), "is_approval_prompt": True}
            fut = self._schedule(
                adapter.send(ctx._status_chat_id, msg, metadata=_interim_metadata(metadata)), "Approval text-send scheduling error",
            )
            if fut is not None:
                fut.result(timeout=15)
        except Exception as e:
            logger.error("Failed to send approval request: %s", e)

    # ── run_sync phases ─────────────────────────────────────────────────────────────────────

    def _load_turn_history(self, agent, reused_cached_agent):
        from gateway.run import (
            _build_gateway_agent_history, _collect_history_media_paths, _message_timestamps_enabled,
            _select_cached_agent_history,
        )
        ctx = self._ctx
        # Transcript rows ({role, content, timestamp}) lose timestamps; interrupt-path agent messages
        # (tool_calls/tool_call_id/reasoning) pass through intact so the API sees valid assistant→tool
        # sequences. Telegram observed=True rows are withheld from replayable history and attached to
        # the current addressed message as API-only context.
        agent_history, observed_group_context = _build_gateway_agent_history(
            ctx.history, channel_prompt=ctx.channel_prompt, inject_timestamps=_message_timestamps_enabled(ctx.user_config),
        )
        # FTS write-corruption guard: if persistence failed silently the reloaded transcript is stale
        # while the SAME cached agent still holds the live conversation (same-session amnesia). Only
        # for a reused agent bound to this exact session_id.
        # Replacing the live transcript with that shorter copy causes immediate same-session amnesia. See
        # #50502.
        if reused_cached_agent and getattr(agent, "session_id", None) == ctx.session_id:
            selected = _select_cached_agent_history(agent_history, getattr(agent, "_session_messages", None))
            if selected is not agent_history:
                logger.warning(
                    "Persisted transcript lagged live cached history for "
                    "session %s (disk=%d, memory=%d); preserving live "
                    "conversation context (possible FTS write corruption)",
                    ctx.session_key, len(agent_history), len(selected),
                )
                # The live history bypassed _build_gateway_agent_history's cleanup — re-apply the
                # stale-confirmation expiry so a dangerous confirmation can't slip through.
                agent_history = strip_stale_dangerous_confirmations(selected, now=time.time())
        # MEDIA paths already in history are excluded from this turn's extraction (compression-safe).
        return agent_history, observed_group_context, _collect_history_media_paths(agent_history)

    def _prepend_pending_note(self, attr: str) -> None:
        """Consume a one-shot per-session note (model switch, /reload-skills) into the NEXT user
        message. Nothing hits the transcript out-of-band, so alternation stays intact."""
        ctx = self._ctx
        notes = getattr(self._runner, attr, None)
        note = notes.pop(ctx.session_key, None) if notes and ctx.session_key and ctx.session_key in notes else None
        if note:
            ctx.message = note + "\n\n" + ctx.message

    def _resume_note_interactive(self) -> bool:
        """Interactive platforms report the restore and ask what next; event platforms (webhook,
        API server) continue the work — nobody is present to answer."""
        return bool(getattr(self._runner._adapter_for_source(self._ctx.source), "interactive_resume", True))

    def _prepare_turn_message(self, agent_history):
        """Prepend recovery/notice guidance to ``ctx.message``.

        Returns (persist_user_message_override, persist_user_timestamp_override): real user text is
        kept separate from API-only recovery guidance so stale guidance never replays as user text.
        """
        from gateway.run import (
            _auto_continue_freshness_window, _is_fresh_gateway_interruption,
            _last_transcript_timestamp, _prepare_resume_pending_message, build_resume_recovery_note,
        )
        ctx = self._ctx
        persist_override: Optional[Any] = ctx.persist_user_message
        self._prepend_pending_note("_pending_model_notes")
        # Auto-continue: history ending with a tool result means the previous turn was cut off
        # (restart, crash, SIGTERM). Session-level resume_pending (drain-timeout shutdown) uses
        # stronger reason-aware wording that subsumes this case. Both gate on the age of
        # ``history[-1]`` (not agent_history, which stripped tool-row timestamps); no stamp = fresh.
        window = _auto_continue_freshness_window()
        interruption_is_fresh = _is_fresh_gateway_interruption(_last_transcript_timestamp(ctx.history), window_secs=window)
        entry = None
        if ctx.session_key:
            with suppress(Exception):
                entry = self._runner.session_store._entries.get(ctx.session_key)
        resume_pending = entry is not None and getattr(entry, "resume_pending", False)
        resume_reason = (getattr(entry, "resume_reason", None) or "restart_timeout") if resume_pending else None
        # resume_pending freshness ALSO uses the restart watchdog's ``last_resume_marked_at`` (the true
        # interruption stamp): the transcript clock can be hours older for an active thread, and the
        # startup auto-resume turn has empty text, so gating on it alone yields a blank user message.
        mark_is_fresh = resume_pending and _is_fresh_gateway_interruption(
            getattr(entry, "last_resume_marked_at", None), window_secs=window,
        )
        if resume_pending and (interruption_is_fresh or mark_is_fresh):
            # Empty message = the startup auto-resume turn; there is no NEW user message.
            ctx.message, persist_override = _prepare_resume_pending_message(
                resume_reason, ctx.message, interactive=self._resume_note_interactive(),
            )
        elif agent_history and agent_history[-1].get("role") == "tool" and interruption_is_fresh:
            persist_override = ctx.message
            ctx.message = (
                "[System note: A new message has arrived. The conversation "
                "history contains pending tool outputs from an interrupted turn. "
                "IGNORE those pending results. Address the user's NEW message "
                "below FIRST. Do NOT re-execute old tool calls from the history.]\n\n"
                + ctx.message
            )
        self._prepend_pending_note("_pending_skills_reload_notes")
        # Safety net: a startup auto-resume event carries empty text; if the resume_pending branch
        # did not fire (freshness signals disagreed, marker cleared) we must NOT hand the model a blank
        # user turn. Restricted to resume_pending sessions so caption-less image turns are untouched.
        if isinstance(ctx.message, str) and not ctx.message.strip() and resume_pending:
            ctx.message = build_resume_recovery_note(resume_reason, "", interactive=self._resume_note_interactive())
        return persist_override, ctx.persist_user_timestamp

    def _native_image_run_message(self):
        """Wrap the user turn as an OpenAI-style multimodal content list when
        _prepare_inbound_message_text buffered image paths; consume-and-clear so later turns on the
        same runner never re-attach stale images. Falls back to plain text when nothing is readable."""
        ctx = self._ctx
        native_imgs = self._runner._consume_pending_native_image_paths(ctx.session_key)
        if not native_imgs:
            return ctx.message
        try:
            from agent.image_routing import build_native_content_parts
            parts, skipped = build_native_content_parts(ctx.message, native_imgs)
            if skipped:
                logger.warning("Native image attachment: skipped %d unreadable path(s): %s", len(skipped), skipped)
            if any(p.get("type") == "image_url" for p in parts):
                return parts
        except Exception as exc:
            logger.warning("Native image attachment failed, falling back to text: %s", exc)
        return ctx.message

    def _run_conversation_with_approval(self, agent, agent_history, observed_group_context,
                                        persist_user_message_override, persist_user_timestamp_override):
        """Run the turn with the per-session gateway approval callback registered: dangerous-command
        approval blocks the agent thread (mirrors CLI input()); the callback bridges sync→async."""
        from gateway.run import _wrap_current_message_with_observed_context
        from tools.approval import register_gateway_notify, unregister_gateway_notify
        from tools.approval_session_notify import register_session_notify
        from tools.approval_context import reset_current_session_key, set_current_session_key
        ctx = self._ctx
        session_key = ctx.session_key or ""
        token = set_current_session_key(session_key)
        register_gateway_notify(session_key, self._approval_notify_sync)
        # The turn callback is removed below, but SDK background work may ask
        # for permission between Hermes turns.  This session-scoped callback
        # remains valid until the conversation boundary or gateway shutdown.
        register_session_notify(session_key, self._approval_notify_sync)
        try:
            api_message = _wrap_current_message_with_observed_context(self._native_image_run_message(), observed_group_context)
            kwargs = {"conversation_history": agent_history, "task_id": ctx.session_id}
            if persist_user_message_override is not None:
                kwargs["persist_user_message"] = persist_user_message_override
            elif observed_group_context:
                kwargs["persist_user_message"] = ctx.message
            if ctx.persist_user_display_kind:
                # Internal self-injected turn: type the persisted user row so UIs render it as a
                # timeline notice, not a user bubble (stripped from provider payloads downstream).
                kwargs["persist_user_display_kind"] = ctx.persist_user_display_kind
            if ctx.moa_config is not None:
                kwargs["moa_config"] = ctx.moa_config
            if persist_user_timestamp_override is not None:
                kwargs["persist_user_timestamp"] = persist_user_timestamp_override
            # The RAW inbound id (not event_message_id, the reply anchor) rides the persisted user
            # turn so a restart-interrupted turn is recorded WITH its id for drain-window dedup.
            if ctx.inbound_message_id is not None:
                kwargs["persist_user_platform_id"] = str(ctx.inbound_message_id)
            return agent.run_conversation(api_message, **kwargs)
        finally:
            unregister_gateway_notify(session_key)
            # Cancel pending clarify entries so blocked agent threads don't hang past the end of the
            # run (interrupt, completion, gateway shutdown). Idempotent.
            with suppress(Exception):
                from tools.clarify_gateway import clear_session
                clear_session(session_key)
            reset_current_session_key(token)

    def _finish_stream_consumer(self, result, agent_history, stream_consumer):
        ctx = self._ctx
        # Canonicalize a model-emitted computer-use screenshot path at the common result boundary so
        # the streaming finalizer and the non-streaming delivery path see the same response.
        if isinstance(result, dict) and isinstance(result.get("final_response"), str):
            result["final_response"] = repair_explicit_computer_use_media_paths(
                result["final_response"], result.get("messages", []), history_offset=len(agent_history),
            )
        ctx.result_holder[0] = result
        if stream_consumer is None:
            return
        # Pass final_response as the authoritative finalize payload: it includes post-stream
        # augmentation (verifier footer, explainer) the accumulator never saw. Adopt ONLY a genuinely
        # completed final: interrupt paths return {interrupted: True, completed: False} with a
        # DIAGNOSTIC final_response — adopting it would seal the partial answer over with the
        # diagnostic AND suppress the gateway's own error delivery.
        _final_for_stream = None
        if (
            isinstance(result, dict) and not result.get("failed") and not result.get("interrupted")
            and result.get("completed") is not False
        ):
            fr = result.get("final_response")
            if isinstance(fr, str) and fr.strip() and fr != "(empty)":
                _final_for_stream = fr
        if _final_for_stream is None:
            stream_consumer.finish()
            return
        # Duck-type safe: test doubles / older consumers may expose a zero-arg finish().
        try:
            stream_consumer.finish(_final_for_stream)
        except TypeError:
            stream_consumer.finish()

    def _restore_telegram_thread_id_after_split(self, agent_session_id) -> None:
        """Telegram DM whose source.thread_id was lost in the session split (synthetic/recovered
        event): restore it from the binding so _thread_metadata_for_source yields the right
        message_thread_id instead of the General thread (non-fatal)."""
        ctx = self._ctx
        try:
            # run_sync is off-loop (executor); sync DB is fine.
            binding = self._runner._session_db._db.get_telegram_topic_binding_by_session(session_id=agent_session_id)
            if binding and binding.get("thread_id"):
                ctx.source.thread_id = str(binding["thread_id"])
                logger.debug(
                    "Restored source.thread_id=%s from binding after session split %s → %s",
                    ctx.source.thread_id, ctx.session_id, agent_session_id,
                )
        except Exception:
            logger.debug("Failed to restore thread_id from binding after session split", exc_info=True)

    def _sync_session_after_run(self, agent_history):
        """Sync session_id right after run_conversation(): compression can rotate before a follow-up
        model call fails, and the failure return must still point at the compressed child.
        Returns (compacted_in_place, effective_session_id, effective_history_offset)."""
        ctx = self._ctx
        runner = self._runner
        agent = ctx.agent_holder[0]
        # In-place compaction compacts the transcript WITHOUT rotating the id, so the id-change diff
        # can't see it; compress_context() sets this flag and the gateway re-baselines as for a split.
        compacted_in_place = bool(getattr(agent, "_last_compaction_in_place", False)) if agent else False
        agent_session_id = getattr(agent, 'session_id', ctx.session_id) if agent else ctx.session_id
        session_was_split = bool(agent and ctx.session_key and agent_session_id != ctx.session_id)
        if session_was_split:
            logger.info("Session split detected: %s → %s (compression)", ctx.session_id, agent_session_id)
            entry = runner.session_store._entries.get(ctx.session_key)
            persisted = False
            if entry:
                entry_session_id = getattr(entry, "session_id", None)
                if not ctx._run_still_current():
                    logger.info(
                        "Skipping session split sync for stale run %s — "
                        "generation %s is no longer current",
                        ctx.session_key or "?", ctx.run_generation,
                    )
                elif entry_session_id == agent_session_id:
                    persisted = True
                elif entry_session_id != ctx.session_id:
                    logger.info(
                        "Skipping session split sync for %s because the "
                        "session binding moved from %s to %s before "
                        "compression finished",
                        ctx.session_key or "?", ctx.session_id, entry_session_id,
                    )
                else:
                    entry.session_id = agent_session_id
                    runner.session_store._save()
                    runner.session_store._record_gateway_session_peer(agent_session_id, ctx.session_key, ctx.source)
                    persisted = True
            # Only after this run published its split — a stale /stop→/new predecessor must not
            # mutate routing state.
            if persisted:
                src = ctx.source
                if (
                    getattr(src, "platform", None) == Platform.TELEGRAM and getattr(src, "chat_type", None) == "dm"
                    and getattr(src, "thread_id", None) is None and runner._session_db is not None
                ):
                    self._restore_telegram_thread_id_after_split(agent_session_id)
                runner._sync_telegram_topic_binding(src, entry, reason="agent-run-compression")
        runner._sync_session_model_from_agent(agent_session_id, agent)
        # history_offset=0 whenever the agent's message list lost the original history prefix
        # (split OR in-place compaction): the returned `messages` is the compacted set, persist all
        # of it; slicing past the pre-compaction length would drop everything.
        offset = 0 if (session_was_split or compacted_in_place) else len(agent_history)
        return compacted_in_place, agent_session_id, offset

    def _combined_ephemeral_prompt(self) -> str:
        """Platform context + YAML channel_prompts hint + channel_overrides system_prompt (or global
        ephemeral) + the gateway ephemeral prompt."""
        ctx = self._ctx
        combined = ctx.context_prompt or ""
        for extra in (
            (ctx.channel_prompt or "").strip(),
            self._runner._get_system_prompt_for_channel(
                ctx.source.platform, ctx.source.chat_id or "", thread_id=getattr(ctx.source, "thread_id", None),
                parent_id=getattr(ctx.source, "parent_chat_id", None),
            ),
        ):
            if extra:
                combined = (combined + "\n\n" + extra).strip()
        return combined

    def _append_auto_media_tags(self, final_response: str, result, agent_history, history_media_paths) -> str:
        """Append MEDIA:<path> tags from tool results (e.g. TTS) that the model's final text omits, so
        extract_media() delivers each file once. Scoped to THIS turn (slice at len(agent_history)) so
        a stale MEDIA: path from an earlier turn never rides a later reply; the history-path dedup is
        the secondary guard — and the sole one when mid-run compression shrank the list."""
        from gateway.run import _collect_auto_append_media_tags
        if "MEDIA:" in final_response:
            return final_response
        # Scan tool results for MEDIA:<path> tags that need to be delivered as native audio/file
        # attachments. The TTS tool embeds MEDIA: tags in its JSON response, but the model's final text
        # reply usually doesn't include them. We collect unique tags from tool results and append any that
        # aren't already present in the final response, so the adapter's extract_media() can find and
        # deliver the files exactly once. Scope the scan to THIS turn's tool results only. ``agent_history``
        # was passed into run_conversation as ``conversation_history``, so the agent's returned ``messages``
        # list is ``agent_history`` followed by the messages produced this turn. Slicing at
        # ``len(agent_history)`` isolates the current turn precisely, so a stale MEDIA: path emitted by a
        # tool several turns earlier (still present in the full message list) can never leak onto a later
        # text-only reply. (Fixes #34608) Path-based deduplication against _history_media_paths (collected
        # before run_conversation) is retained as a secondary guard. It is also the sole guard on the
        # fallback branch taken when mid-run context compression shrinks the message list below the original
        # history length, preserving the compression-safe behaviour of #160.
        media_tags, has_voice_directive = _collect_auto_append_media_tags(
            result.get("messages", []), history_offset=len(agent_history), history_media_paths=history_media_paths,
        )
        if not media_tags:
            return final_response
        unique_tags = (["[[audio_as_voice]]"] if has_voice_directive else []) + list(dict.fromkeys(media_tags))
        return final_response + "\n" + "\n".join(unique_tags)

    def run_sync(self):
        """Executor-thread body of the turn; returns the gateway result dict.

        The turn message lives on the shared TurnContext (``ctx.message``) so ``_run_agent_inner`` sees
        every rebind. session_key propagates via contextvars (_set_session_env / set_current_session_key)
        — never os.environ["HERMES_SESSION_KEY"], which would misroute approvals across sessions.
        """
        from gateway.run import _current_max_iterations, _normalize_empty_agent_response, _sanitize_gateway_final_response
        ctx = self._ctx
        runner = self._runner
        # Platform.LOCAL ("local") maps to the "cli" hint key the agent understands.
        # session_key is propagated via contextvars in _set_session_env() (_SESSION_KEY) and via
        # set_current_session_key() (_approval_session_key) below — both concurrency-safe and inherited by
        # tool worker threads. We deliberately do NOT write os.environ["HERMES_SESSION_KEY"] here:
        # os.environ is process-global, so concurrent gateway sessions (e.g. two Discord threads) would
        # clobber each other's value, and a tool thread whose contextvar is unset would fall back to
        # os.environ and read the wrong session key — misrouting command-approval prompts to the wrong
        # thread (#24100). The non-gateway surfaces don't depend on this write: CLI and cron bind the
        # session via contextvars (set_current_session_key / session context), and only the TUI slash-worker
        # *subprocess* exports HERMES_SESSION_KEY (from its own --session-key argv, a separate process) — so
        # removing this in-process gateway write does not affect any of them.
        platform_key = "cli" if ctx.source.platform == Platform.LOCAL else ctx.source.platform.value
        combined_ephemeral = self._combined_ephemeral_prompt()
        max_iterations = _current_max_iterations()
        try:
            model, runtime_kwargs = runner._resolve_session_agent_runtime(
                source=ctx.source, session_key=ctx.session_key, user_config=ctx.user_config,
            )
            logger.debug(
                "run_agent resolved: model=%s provider=%s session=%s",
                model, runtime_kwargs.get("provider"), ctx.session_key or "",
            )
        except Exception as exc:
            return {"final_response": f"⚠️ Provider authentication failed: {exc}", "messages": [], "api_calls": 0, "tools": []}
        pr = runner._provider_routing
        reasoning_config = runner._resolve_session_reasoning_config(source=ctx.source, session_key=ctx.session_key, model=model)
        runner._reasoning_config = reasoning_config
        runner._service_tier = runner._resolve_session_service_tier(source=ctx.source, session_key=ctx.session_key)
        stream_consumer, stream_delta_cb, interim_cb, want_interim = self._setup_stream_consumer(platform_key)
        turn_route = runner._resolve_turn_agent_config(ctx.message, model, runtime_kwargs)
        agent, reused_cached_agent = self._resolve_turn_agent(
            turn_route, platform_key, combined_ephemeral, max_iterations, reasoning_config, pr,
        )
        self._wire_turn_agent_callbacks(agent, turn_route, reasoning_config, stream_delta_cb, interim_cb, want_interim)
        agent_history, observed_group_context, history_media_paths = self._load_turn_history(agent, reused_cached_agent)
        persist_msg, persist_ts = self._prepare_turn_message(agent_history)
        result = self._run_conversation_with_approval(agent, agent_history, observed_group_context, persist_msg, persist_ts)
        self._finish_stream_consumer(result, agent_history, stream_consumer)
        # The streaming-TTS consumer's finish() runs on the outer loop thread after the executor
        # returns, so early run_sync returns are also finalised.
        # See the outer finally/completion section below. See #60671.
        final_response = result.get("final_response")
        # Actual token counts from the agent instance used for this run.
        agent = ctx.agent_holder[0]
        has_comp = bool(agent) and hasattr(agent, "context_compressor")
        comp = agent.context_compressor if has_comp else None
        usage = {
            "last_prompt_tokens": getattr(comp, "last_prompt_tokens", 0) if has_comp else 0,
            "input_tokens": getattr(agent, "session_prompt_tokens", 0) if has_comp else 0,
            "output_tokens": getattr(agent, "session_completion_tokens", 0) if has_comp else 0,
            "model": getattr(agent, "model", None) if agent else None,
            "context_length": (getattr(comp, "context_length", 0) or 0) if has_comp else 0,
        }
        compacted_in_place, effective_session_id, history_offset = self._sync_session_after_run(agent_history)
        # failure_reason must survive the empty-response path too (TUI billing, transient-failure
        # persistence). compression_deferred (soft lock-contention defer) is distinct from
        # compression_exhausted so the gateway never auto-resets a session a concurrent compressor is
        # about to shrink.
        common = {
            "messages": result.get("messages", []), "api_calls": result.get("api_calls", 0),
            "failed": result.get("failed", False), "failure_reason": result.get("failure_reason"),
            "partial": result.get("partial", False), "completed": result.get("completed"),
            "interrupted": result.get("interrupted", False), "interrupt_message": result.get("interrupt_message"),
            "error": result.get("error"),
            "compression_exhausted": result.get("compression_exhausted", False),
            "compression_deferred": result.get("compression_deferred", False),
            "tools": ctx.tools_holder[0] or [],
            "history_offset": history_offset, "compacted_in_place": compacted_in_place, "session_id": effective_session_id,
            **usage,
        }
        if not final_response:
            final_response = _normalize_empty_agent_response(result, final_response or "", history_len=len(agent_history))
            final_response = _sanitize_gateway_final_response(ctx.source.platform, final_response)
            if not final_response:
                final_response = f"⚠️ {result['error']}" if result.get("error") else ""
            # NOTE: deliberately omits agent_persisted/last_reasoning/response_* — the caller
            # defaults agent_persisted differently when the key is absent.
            return {"final_response": final_response, **common}
        final_response = self._append_auto_media_tags(final_response, result, agent_history, history_media_paths)
        # Auto-titling runs at TURN START (agent/turn_context.py) from the user's message alone, so a
        # failed/interrupted turn is still titled.
        return {
            "final_response": final_response, "last_reasoning": result.get("last_reasoning"), **common,
            "response_previewed": result.get("response_previewed", False),
            "response_transformed": result.get("response_transformed", False),
            # Lets the persistence block tell whether the codex app-server path self-persisted (it
            # didn't — see codex_runtime.py); default True keeps skip-db for the standard runtime.
            "agent_persisted": result.get("agent_persisted", True),
        }
