"""Extracted gateway run_turn_progress responsibility; consumed by TurnRunner."""
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


class GatewayTurnProgressMixin:
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
        if not (
            getattr(adapter, "supports_code_blocks", False) and tool_name == "terminal" and isinstance(args, dict)
            and isinstance(args.get("command"), str) and args["command"].strip()
        ):
            return None, None
        cmd_full = args["command"].rstrip()
        header = "" if self._ctx.last_was_terminal_block[0] else f"{emoji} {tool_name}\n"
        cap = self._preview_cap()
        lines = cmd_full.splitlines()
        cmd_short = lines[0] if lines else cmd_full
        if len(cmd_short) > cap:
            cmd_short = cmd_short[:cap - 3] + "..."
        elif len(lines) > 1:
            cmd_short += " ..."
        return f"{header}```\n{cmd_full}\n```", f"{header}```\n{cmd_short}\n```"

    def _progress_build_message(self, tool_name, preview, args) -> Optional[str]:
        """Render the progress line. Verbose mode queues directly (no dedup) and returns None."""
        ctx = self._ctx
        from agent.display import get_tool_emoji
        emoji = get_tool_emoji(tool_name, default="⚙️")
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
            return f"{emoji} {tool_name}..."
        from agent.display import get_tool_verb, prepare_tool_preview, tool_verb_connector, verb_drops_preview
        prepared = prepare_tool_preview(tool_name, args, fallback=preview, max_len=self._preview_cap())
        preview = adapter.format_tool_preview(prepared) if adapter is not None else prepared.text
        # Friendly labels: human-phrased line for built-in tools ("🔍 Searching the web for ...")
        # by prefixing the verb onto the computed preview, so the command/url/query is kept.
        verb = get_tool_verb(tool_name)
        if not verb:
            return f"{emoji} {tool_name}: \"{preview}\""
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
        # TERMINAL authorization refusal, distinct from native_failed: the
        # connector refused this destination, so no later publication in this
        # turn may re-deliver the task text through the text fallback. Declared
        # rather than set dynamically so the state is visible where it lives.
        egress_declined: bool = False
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
        from gateway.relay.egress import declined_send

        if getattr(st, "egress_declined", False):
            return
        if st.fallback_msg_id:
            result = await st.adapter.edit_message(
                chat_id=ctx.source.chat_id, message_id=st.fallback_msg_id, content=text, metadata=ctx._progress_metadata,
            )
            if getattr(result, "success", False):
                return
            # P5(b): R5-4 made a declined native CARD terminal but left this
            # editable-text fallback: a declined edit fell through to
            # _send_progress_text and re-sent the same task text to the refused
            # chat. The decline must set the terminal state here too.
            if declined_send(result):
                logger.warning(
                    "Task-card fallback edit DECLINED by the connector's egress "
                    "guard; suppressing progress delivery for the rest of this "
                    "turn (the destination is not approved)"
                )
                st.egress_declined = True
                return
        result = await self._send_progress_text(st, text)
        if getattr(result, "success", False) and getattr(result, "message_id", None):
            st.fallback_msg_id = str(result.message_id)

    async def _task_card_publish(self, st) -> None:
        ctx = self._ctx
        if not st.tasks:
            return
        if getattr(st, "egress_declined", False):
            # The connector refused this destination earlier in the turn; every
            # later publication would re-deliver the same task text there.
            return
        if not st.native_failed:
            result = await st.adapter.send_native_task_card_progress(
                chat_id=ctx.source.chat_id, tasks=st.visible_tasks(), title="Hermes is working",
                reply_to=ctx._progress_reply_to, metadata=ctx._progress_metadata, fallback_text=st.fallback_text(),
            )
            if getattr(result, "success", False):
                return
            # P5(b): an AUTHORIZATION decline is not a broken card lane. The
            # fallback below sends the same task text to the same chat, which
            # turns a refused card into delivered plain text. Stop the lane
            # without re-delivering; the refusal is already logged.
            from gateway.relay.egress import declined_send

            if declined_send(result):
                # TERMINAL, and stored SEPARATELY from native_failed. Reusing
                # native_failed suppressed exactly ONE update: the next progress
                # event skipped this branch (the lane is already "failed") and
                # went straight to the text fallback. A refusal does not expire
                # after one tick.
                st.egress_declined = True
                st.native_failed = True
                logger.warning(
                    "Slack native task-card progress DECLINED by the connector's "
                    "egress guard — suppressing the text fallback for the rest "
                    "of this turn (the destination is not approved)"
                )
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
        self._publish_execution("tool.start", {
            "tool_call_id": str(call_id or ""), "tool_name": str(tool_name or "tool"),
            "args": args if isinstance(args, dict) else {}})
        self._publish_api_tool("tool.start", call_id, tool_name, args)
        if self._ctx._voice_ack_guild[0] is not None:
            self.voice_ack_callback(call_id, tool_name, args)
        if self._ctx._native_slack_task_cards:
            self.native_tool_start_callback(call_id, tool_name, args)

    def combined_tool_complete_callback(self, call_id, tool_name, args, result):
        from agent.display import _detect_tool_failure
        is_error, _ = _detect_tool_failure(tool_name, result)
        self._publish_execution("tool.complete", {
            "tool_call_id": str(call_id or ""), "tool_name": str(tool_name or "tool"),
            "is_error": bool(is_error), "args": args if isinstance(args, dict) else {},
            "result": result if isinstance(result, str) else str(result)})
        self._publish_api_tool("tool.complete", call_id, tool_name, args, result)
        if self._ctx._native_slack_task_cards:
            self.native_tool_complete_callback(call_id, tool_name, args, result)

    # ── hook / status bridges (agent thread → gateway loop) ────────────────────────────────

    def _step_callback_sync(self, iteration: int, prev_tools: list) -> None:
        ctx = self._ctx
        if not ctx._run_still_current():
            return
        self._publish_execution("agent.step", {"iteration": iteration})
        if not ctx._hooks_ref.loaded_hooks:
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
