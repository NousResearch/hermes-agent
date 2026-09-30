"""Lossless, paced Discord activity owned by the dispatching turn.

The turn's progress-queue consumer publishes parent and child records together. At
turn cleanup it releases this same acknowledged cursor to the retained publisher.
Only the first bubble is editable until overflow; all later chunks are sealed.
A failed POST is ambiguous and stops publication, never replaying its payload.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("gateway.run")
_RUNNING, _DONE, _INTERRUPTED, _TIMEOUT, _FAILED = "⏳", "✅", "⏹️", "⏱️", "❌"
_STATUS_GLYPHS = {"completed": _DONE, "interrupted": _INTERRUPTED, "timeout": _TIMEOUT,
                  "failed": _FAILED, "error": _FAILED}
_RETRYABLE_KINDS = frozenset({"transient", "rate_limited"})
_RETRY_DELAYS = (0.5, 2.0)
_Block = Tuple[str, str]
_Unit = Any


def _no_mentions(text: str) -> str:
    # Labels are prose; command bodies are preserved and protected on the wire.
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
        self.tool_count, self.duration = 0, None
        self.status: Optional[str] = None
        self.started = time.monotonic()


class DelegatedChildProgress:
    EDIT_INTERVAL = 2.0
    MAX_TRANSPORT_RETRIES = 2

    def __init__(self, *, adapter, loop, chat_id, metadata, reply_to, verbose,
                 preview_cap, verbose_cap, retain=None, native_sink=None,
                 on_result=None):
        self.adapter, self.loop, self.chat_id = adapter, loop, chat_id
        self.metadata = {**(metadata or {}), "progress": True, "non_conversational": True}
        self.reply_to, self.verbose = reply_to, verbose
        self.preview_cap, self.verbose_cap = preview_cap, verbose_cap
        self._retain, self._on_result = retain, on_result
        self._native_sink = native_sink
        self._lock = threading.Lock()
        self._children, self._batches = {}, []
        # Parts are immutable in event order. Only acknowledged parts advance the cursor.
        self._parts = []
        self._cursor = 0
        self._sealed = False
        self._wake = None
        self._task = None
        self._msg_id = None
        self._chunk_ids, self._chunk_text = [], []
        self._last_publish = 0.0
        self._transport_retries, self._retry_at = 0, 0.0
        self._dead = False

    def _append(self, unit):
        limit, len_fn = self._limits()
        self._parts.extend(self._unit_parts(unit, limit, len_fn))

    def add_activity(self, text):
        """Parent progress uses the same bounded, acknowledged activity lane."""
        if self._dead:
            return
        with self._lock:
            if "```\n" in text and text.endswith("\n```"):
                header, body = text.split("```\n", 1)
                self._append((header.rstrip(), body[:-4]))
            else:
                self._append(text)

    def on_event(self, event_type, tool_name, preview, args, kw):
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
                self._append(self._head_line(child))
            elif child is None:
                return
            elif event_type == "subagent.tool" and child.status is None and tool_name and tool_name != "_thinking":
                child.tool_count = max(child.tool_count + 1, int(kw.get("tool_count") or 0))
                line, block = self._tool_lines(tool_name, preview, args)
                prefix = f"[{_no_mentions(key[1]) or 'child'}] "
                self._append((prefix + block[0], block[1]) if block else prefix + line)
            elif event_type == "subagent.complete" and child.status is None:
                child.status = str(kw.get("status") or "completed")
                duration = kw.get("duration_seconds")
                child.duration = duration if isinstance(duration, (int, float)) else time.monotonic() - child.started
                self._append(self._head_line(child))
                if child.status not in {"completed", "interrupted"}:
                    from tools.delegate_tool_progress import describe_subagent_failure
                    reason = describe_subagent_failure(kw.get("failure_reason"), kw.get("summary") or preview)
                    self._append("⚠️ " + _no_mentions(_redact(reason)))
            else:
                return
        try:
            self.loop.call_soon_threadsafe(self._kick)
        except RuntimeError:
            self._dead = True

    def _tool_lines(self, tool_name: str, preview: Any, args: Any) -> Tuple[str, Optional[_Block]]:
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
        if self.verbose and isinstance(command, str) and command.strip():
            # Verbose keeps the complete command, like the parent's own verbose terminal blocks.
            block = (f"{emoji} {verb or tool_name}", _redact(command))
        elif self.verbose and isinstance(args, dict) and args:
            line = f"{emoji} {verb or tool_name}\n" + json.dumps(args, ensure_ascii=False, indent=2, default=str)
        return _redact(line), block

    def _native_live(self):
        return self._native_sink is not None

    def end_turn(self):
        """Explicit queue-consumer handover, even when generation remains current."""
        self._native_sink = None
        if self._cursor < len(self._parts) or (self._children and not self._settled()):
            self._kick()

    def _kick(self):
        if self._dead:
            return
        if self._native_live():
            self._native_sink(("__child_progress__",))
            return
        if self._wake is None:
            self._wake = asyncio.Event()
        self._wake.set()
        if self._task is None or self._task.done():
            self._task = self.loop.create_task(self._publish_loop())
            if callable(self._retain):
                self._retain(self._task)

    def _settled(self):
        with self._lock:
            return all(c.status is not None for c in self._children.values())

    async def _publish_loop(self):
        try:
            while not self._dead:
                await self._wake.wait()
                self._wake.clear()
                await self._deliver_chunks()
                if self._settled() and self._cursor == len(self._parts):
                    return
        except asyncio.CancelledError:
            # Shutdown never creates new messages. Only the verified head gets a final mark.
            if self._msg_id and not self._dead and not self._settled():
                text = self._chunk_text[0].replace(_RUNNING, _INTERRUPTED)
                try:
                    await asyncio.wait_for(self.adapter.edit_message(
                        chat_id=self.metadata.get("thread_id") or self.chat_id,
                        message_id=self._msg_id, content=text, metadata=self.metadata), 3)
                except Exception:
                    logger.debug("child shutdown edit failed", exc_info=True)
            self._dead = True
            raise
        except Exception:
            logger.debug("child activity publisher failed", exc_info=True)
            self._dead = True

    async def _deliver_chunks(self):
        """Seal successful chunks, never recompute their indices from a mutable render."""
        while not self._dead:
            with self._lock:
                end = self._cursor
                if end == len(self._parts):
                    return
                editable = self._msg_id is not None and not self._sealed
                text = self._chunk_text[0] if editable else ""
                limit, len_fn = self._limits()
                while end < len(self._parts):
                    candidate = text + ("\n" if text else "") + self._parts[end]
                    if text and len_fn(candidate) > limit:
                        break
                    text = candidate
                    end += 1
                if end == self._cursor:
                    self._sealed = True
                    continue
            delay = max(self._last_publish + self.EDIT_INTERVAL, self._retry_at) - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                if editable:
                    result = await self.adapter.edit_message(
                        chat_id=self.metadata.get("thread_id") or self.chat_id,
                        message_id=self._msg_id, content=text, metadata=self.metadata)
                else:
                    result = await self.adapter.send(chat_id=self.chat_id, content=text,
                        reply_to=self._chunk_ids[-1] if self._chunk_ids else self.reply_to, metadata=self.metadata)
            except Exception:
                result = None
            self._last_publish = time.monotonic()
            if not getattr(result, "success", False) or (not editable and not getattr(result, "message_id", None)):
                # A transport exception says nothing about whether a POST landed. Never retry it.
                kind = getattr(result, "error_kind", None)
                if editable and kind in _RETRYABLE_KINDS and self._transport_retries < self.MAX_TRANSPORT_RETRIES:
                    delay = _RETRY_DELAYS[self._transport_retries]
                    retry_after = getattr(result, "retry_after", None)
                    if isinstance(retry_after, (int, float)):
                        delay = max(delay, min(30.0, max(0.0, retry_after)))
                    self._transport_retries += 1
                    self._retry_at = time.monotonic() + delay
                    continue
                self._dead = True
                return
            self._transport_retries, self._retry_at = 0, 0.0
            if self._on_result:
                self._on_result(result)
            if editable:
                self._chunk_text[0] = text
            elif self._msg_id is None:
                self._msg_id = str(result.message_id)
                self._chunk_text.append(text)
            else:
                self._chunk_ids.append(str(result.message_id))
                self._chunk_text.append(text)
                self._sealed = True
            self._cursor = end
            if self._native_live():
                return  # let the queue consumer observe cleanup between bounded chunks

    def _limits(self) -> Tuple[int, Callable[[str], int]]:
        try:
            limit = int(self.adapter.max_message_length_for_chat(self.chat_id))
            len_fn = self.adapter.message_len_fn_for_chat(self.chat_id)
        except Exception:
            limit, len_fn = int(getattr(self.adapter, "MAX_MESSAGE_LENGTH", 2000) or 2000), len
        return max(1, limit - (64 if limit > 128 else 0)), len_fn

    def _head_line(self, child: _Child) -> str:
        """One child's head line: status glyph, model, batch slot, goal, tool count, duration."""
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
        if child.status and _format_duration(child.duration):
            head.append(_format_duration(child.duration))
        return " · ".join(head)

    def _split_block(self, header: str, command: str, limit: int, len_fn: Callable[[str], int]) -> List[str]:
        """Split one verbose command into self-contained fenced parts.

        Parts break on whitespace, keep the space at the end of the part they cut, and carry no chunk
        indicators, so the parts' bodies concatenate back to the command character for character — the
        whole command reaches the chat even when it is several times the platform cap.
        """
        parts: List[str] = []
        rest, first = command, True
        fence = "`" * max(3, max((len(m[0]) + 1 for m in re.finditer(r"`+", command)), default=0))
        while True:
            head = f"{header}\n{fence}\n" if first else f"{fence}\n"
            budget = limit - len_fn(head) - len_fn("\n" + fence)
            if budget < 1:
                budget = max(1, limit // 2)
            if len_fn(rest) <= budget:
                parts.append(f"{head}{rest}\n{fence}")
                return parts
            window = rest[:budget]
            while len_fn(window) > budget:
                window = window[:-1]
            cut = window.rfind(" ")
            cut = cut + 1 if cut > 0 else len(window)
            parts.append(f"{head}{rest[:cut]}\n{fence}")
            rest, first = rest[cut:], False

    def _unit_parts(self, unit: _Unit, limit: int, len_fn: Callable[[str], int]) -> List[str]:
        """One unit, or its parts when the unit alone is past the cap.

        A verbose block is split by :meth:`_split_block` (lossless, whitespace-aligned).
        Plain lines split directly, without the display splitter's truncation or adornments.
        """
        if isinstance(unit, tuple):
            return self._split_block(unit[0], unit[1], limit, len_fn)
        if len_fn(unit) <= limit:
            return [unit]
        parts = []
        while unit:
            end = min(len(unit), limit)
            while len_fn(unit[:end]) > limit:
                end -= 1
            parts.append(unit[:end])
            unit = unit[end:]
        return parts

    def _pack(self, units: List[_Unit], limit: int, len_fn: Callable[[str], int]) -> List[str]:
        """Greedily pack units into bounded chunks; unit text is never discarded."""
        chunks: List[str] = []
        current: List[str] = []
        size = 0
        for unit in units:
            for part in self._unit_parts(unit, limit, len_fn):
                width = len_fn(part) + 1
                if current and size + width > limit:
                    chunks.append("\n".join(current))
                    current, size = [], 0
                current.append(part)
                size += width
        if current:
            chunks.append("\n".join(current))
        return chunks

    def render_chunks(self):
        with self._lock:
            return self._pack(list(self._parts), *self._limits())

    def render(self):
        return "\n".join(self.render_chunks())
