"""Live Discord card for delegated children that outlives the turn that dispatched them (#128008).

``delegate_task`` children relay ``subagent.start`` / ``subagent.tool`` / ``subagent.complete`` into the
dispatching turn's ``TurnRunner.progress_callback``. The turn's own progress task only renders the parent's
``tool.started`` lines and is cancelled when the turn ends, while background children keep running. This owner
is created by that callback on the first child start and captures the turn's adapter, chat, thread metadata and
display mode at that moment, so later events never re-resolve against a newer turn.

One mention-free message per dispatching turn, edited in place at a bounded rate: a block per child (status
glyph, model, goal, tool count, recent tool lines) ending in a terminal state. Child reasoning and streamed
reply text are never shown. Presentation failures are swallowed; an ambiguous first send is never repeated.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("gateway.run")

_RUNNING, _DONE, _INTERRUPTED, _TIMEOUT, _FAILED = "⏳", "✅", "⏹️", "⏱️", "❌"
_STATUS_GLYPHS = {"completed": _DONE, "interrupted": _INTERRUPTED, "timeout": _TIMEOUT,
                  "failed": _FAILED, "error": _FAILED}


def _no_mentions(text: str) -> str:
    """Break Discord mention syntax (users/roles/channels, @everyone/@here) with a zero-width space."""
    return (str(text).replace("<@", "<\u200b@").replace("<#", "<\u200b#")
            .replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere"))


def _redact(text: str) -> str:
    from agent.redact import redact_sensitive_text
    return redact_sensitive_text(text)


class _Child:
    __slots__ = ("key", "batch", "index", "count", "model", "goal", "tools", "tool_count", "status", "reason",
                 "duration", "started")

    def __init__(self, key: Tuple[str, str], batch: str, kw: Dict[str, Any], goal: str) -> None:
        self.key, self.batch, self.goal = key, batch, goal
        self.index, self.count = int(kw.get("task_index") or 0), int(kw.get("task_count") or 1)
        self.model = str(kw.get("model") or "")
        self.tools: List[Tuple[str, Optional[str]]] = []  # (compact line, verbose block or None)
        self.tool_count, self.reason, self.duration = 0, "", None
        self.status: Optional[str] = None
        self.started = time.monotonic()


class DelegatedChildProgress:
    """Per-dispatching-turn card for its delegated children; ``on_event`` is safe from any thread."""

    EDIT_INTERVAL = 2.0
    RECENT_TOOLS = 3
    MAX_EDIT_FAILURES = 5

    def __init__(self, *, adapter: Any, loop: Any, chat_id: str, metadata: Optional[dict], reply_to: Any,
                 verbose: bool, preview_cap: int, verbose_cap: int, retain: Any = None) -> None:
        self.adapter, self.loop, self.chat_id = adapter, loop, chat_id
        self.metadata = dict(metadata) if metadata else None
        self.reply_to, self.verbose, self.preview_cap, self.verbose_cap = reply_to, verbose, preview_cap, verbose_cap
        self._retain = retain
        self._lock = threading.Lock()
        self._children: Dict[Tuple[str, str], _Child] = {}
        self._batches: List[str] = []
        self._wake: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task] = None
        self._msg_id: Optional[str] = None
        self._last_text: Optional[str] = None
        self._last_publish = 0.0
        self._edit_failures = 0
        self._dead = False  # ambiguous/refused delivery: never send again, never edit an unverified id

    # ── agent/child threads ────────────────────────────────────────────────────────────────

    def on_event(self, event_type: str, tool_name: Optional[str], preview: Any, args: Any, kw: Dict[str, Any]) -> None:
        if self._dead or kw.get("depth") not in (None, 0):
            return
        key = (str(kw.get("delegation_id") or ""), str(kw.get("subagent_id") or kw.get("task_index") or ""))
        with self._lock:
            child = self._children.get(key)
            if event_type == "subagent.start" and child is None:
                if key[0] not in self._batches:
                    self._batches.append(key[0])
                goal = " ".join(str(kw.get("goal") or preview or "").split())
                child = self._children[key] = _Child(key, key[0], kw, goal)
            if child is None:
                return
            if event_type == "subagent.start":
                pass  # the new row itself is the update
            elif event_type == "subagent.tool" and child.status is None and tool_name and tool_name != "_thinking":
                child.tool_count = max(child.tool_count + 1, int(kw.get("tool_count") or 0))
                child.tools = (child.tools + [self._tool_lines(tool_name, preview, args)])[-self.RECENT_TOOLS:]
            elif event_type == "subagent.complete" and child.status is None:
                status = str(kw.get("status") or "completed")
                child.status = status
                duration = kw.get("duration_seconds")  # frozen, so a settled card renders identically
                child.duration = duration if isinstance(duration, (int, float)) else time.monotonic() - child.started
                if status != "completed" and status != "interrupted":
                    from tools.delegate_tool_progress import describe_subagent_failure
                    child.reason = describe_subagent_failure(kw.get("failure_reason"), kw.get("summary") or preview)
            else:
                return
        try:
            self.loop.call_soon_threadsafe(self._kick)
        except RuntimeError:  # loop closed (shutdown) — nothing left to present on
            self._dead = True

    def _tool_lines(self, tool_name: str, preview: Any, args: Any) -> Tuple[str, Optional[str]]:
        from agent.display import (get_tool_emoji, get_tool_verb, prepare_tool_preview, tool_verb_connector,
                                   verb_drops_preview)
        emoji, verb = get_tool_emoji(tool_name, default="⚙️"), get_tool_verb(tool_name)
        fallback = " ".join(str(preview or "").split())
        prepared = prepare_tool_preview(tool_name, args if isinstance(args, dict) else None, fallback=fallback,
                                        max_len=self.verbose_cap if self.verbose else self.preview_cap)
        shown = self.adapter.format_tool_preview(prepared) if hasattr(self.adapter, "format_tool_preview") else prepared.text
        if verb and verb_drops_preview(tool_name):
            line = f"{emoji} {verb}"
        elif not shown:
            line = f"{emoji} {verb or tool_name}"
        else:
            line = f"{emoji} {verb}{tool_verb_connector(tool_name)}{shown}" if verb else f"{emoji} {tool_name}: {shown}"
        block = None
        command = args.get("command") if isinstance(args, dict) and tool_name == "terminal" else None
        if self.verbose and isinstance(command, str) and command.strip() and "```" not in command:
            # Verbose keeps the complete command, like the parent's own verbose terminal blocks.
            block = f"{emoji} {verb or tool_name}\n```\n{_no_mentions(_redact(command.rstrip()))}\n```"
        return _no_mentions(_redact(line)), block

    # ── gateway loop ───────────────────────────────────────────────────────────────────────

    def _kick(self) -> None:
        if self._wake is None:
            self._wake = asyncio.Event()
        self._wake.set()
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._publish_loop())
            if callable(self._retain):  # registered so gateway shutdown cancels it
                self._retain(self._task)

    def _settled(self) -> bool:
        with self._lock:
            return bool(self._children) and all(c.status is not None for c in self._children.values())

    async def _publish_loop(self) -> None:
        wake = self._wake
        assert wake is not None  # _kick creates it before the task
        try:
            while not self._dead:
                await wake.wait()
                delay = self.EDIT_INTERVAL - (time.monotonic() - self._last_publish)
                if delay > 0:
                    await asyncio.sleep(delay)
                wake.clear()  # everything up to now is in this render
                await self._deliver(self.render())
                self._last_publish = time.monotonic()
                if self._settled() and not wake.is_set() and self._last_text == self.render():
                    return
        except asyncio.CancelledError:
            # Gateway shutdown: mark still-running children stopped, one bounded best-effort edit.
            with self._lock:
                for child in self._children.values():
                    if child.status is None:
                        child.status, child.duration = "interrupted", time.monotonic() - child.started
            if self._msg_id and not self._dead:
                try:
                    await asyncio.wait_for(self._deliver(self.render()), 3)
                except BaseException:  # noqa: BLE001 — presentation only; the cancel still propagates
                    pass
            raise
        except Exception:
            logger.debug("delegated child progress publisher failed", exc_info=True)
            self._dead = True

    async def _deliver(self, text: str) -> None:
        if self._dead or not text or text == self._last_text:
            return
        if self._msg_id is None:
            try:
                result = await self.adapter.send(chat_id=self.chat_id, content=text, reply_to=self.reply_to,
                                                 metadata=self.metadata)
            except Exception:
                logger.debug("delegated child progress send failed", exc_info=True)
                result = None
            message_id = getattr(result, "message_id", None) if getattr(result, "success", False) else None
            if message_id:
                self._msg_id, self._last_text = str(message_id), text
            else:  # refused, failed or delivered without an id: cannot reconcile, so never re-send
                self._dead = True
            return
        kwargs: Dict[str, Any] = {"chat_id": self.chat_id, "message_id": self._msg_id, "content": text}
        if self.metadata:
            from agent.interrupt_compat import _accepts_keyword
            if _accepts_keyword(self.adapter.edit_message, "metadata"):
                kwargs["metadata"] = self.metadata
        try:
            result = await self.adapter.edit_message(**kwargs)
        except Exception:
            logger.debug("delegated child progress edit failed", exc_info=True)
            result = None
        if getattr(result, "success", False):
            self._last_text, self._edit_failures = text, 0
            return
        self._edit_failures += 1
        if result is not None and not getattr(result, "retryable", False) and not any(
                w in (getattr(result, "error", "") or "").lower() for w in ("flood", "retry after", "rate")):
            self._dead = True  # message gone / no permission: stop, never post a second card
        elif self._edit_failures >= self.MAX_EDIT_FAILURES:
            self._dead = True
        elif self._wake is not None:
            self._wake.set()  # retry on the next paced tick (edits are idempotent)

    # ── rendering ──────────────────────────────────────────────────────────────────────────

    def render(self) -> str:
        with self._lock:
            children = sorted(self._children.values(), key=lambda c: (self._batches.index(c.batch), c.index))
        try:
            limit = int(self.adapter.max_message_length_for_chat(self.chat_id))
            len_fn = self.adapter.message_len_fn_for_chat(self.chat_id)
        except Exception:
            limit, len_fn = int(getattr(self.adapter, "MAX_MESSAGE_LENGTH", 2000) or 2000), len
        limit = max(1, limit - (64 if limit > 128 else 0))
        text = ""
        for recent, blocks in ((self.RECENT_TOOLS, True), (self.RECENT_TOOLS, False), (1, False), (0, False)):
            text = "\n".join(self._child_block(c, recent, blocks) for c in children)
            if len_fn(text) <= limit:
                return text
        return text[: max(1, limit - 1)] + "…"

    def _child_block(self, child: _Child, recent: int, blocks: bool) -> str:
        from tools.delegate_tool_progress import _format_duration
        tags = []
        if len(self._batches) > 1:
            tags.append(f"set {self._batches.index(child.batch) + 1}")
        if child.count > 1:
            tags.append(f"{child.index + 1}/{child.count}")
        glyph = _STATUS_GLYPHS.get(child.status, _FAILED) if child.status else _RUNNING
        head = [f"{glyph} **{_no_mentions(child.model) or 'subagent'}**"]
        if tags:
            head.append(" · ".join(tags))
        if child.goal:
            goal = child.goal if len(child.goal) <= 80 else child.goal[:77] + "..."
            head.append(_no_mentions(_redact(goal)))
        head.append(f"🛠 {child.tool_count}")
        if child.status:
            if _format_duration(child.duration):
                head.append(_format_duration(child.duration))
        lines = [" · ".join(head)]
        shown = child.tools[-recent:] if recent else []
        for i, (line, block) in enumerate(shown):
            last = i == len(shown) - 1
            lines.append(block if (blocks and last and block and child.status is None) else f"└ {line}")
        if child.reason:
            reason = child.reason if len(child.reason) <= 160 else child.reason[:157] + "..."
            lines.append(f"└ ⚠️ {_no_mentions(_redact(reason))}")
        return "\n".join(lines)
