"""Live Discord progress for delegated children that outlives the turn that dispatched them (#128008).

``delegate_task`` children relay ``subagent.start`` / ``subagent.tool`` / ``subagent.complete`` into the
dispatching turn's ``TurnRunner.progress_callback``. The turn's own progress task only renders the parent's
``tool.started`` lines and is cancelled when the turn ends, while background children keep running. This owner
is created by that callback on the first child start and captures the turn's adapter, chat, thread metadata and
display mode at that moment, so later events never re-resolve against a newer turn.

Two presentation lanes, in preference order:

* **Native activity stream** (``native_sink`` + ``turn_current``): while the turn that dispatched the children
  is still live, child activity is rendered by the platform's own tool-activity presentation — the same lane
  the parent's tool lines use. Nothing else is posted, so a live turn never shows the children twice.
* **Standalone card** (fallback): once that lane is gone (turn ended, no stream consumer, another platform) the
  card takes over, so children stay visible while they outlive the turn. One mention-free message per
  dispatching turn, edited in place at a bounded rate: a block per child (status glyph, model, goal, tool
  count, recent tool lines) ending in a terminal state.

Long content is never dropped: :meth:`DelegatedChildProgress.render_chunks` splits the full render into bounded
chunks — a pinned, editable head plus append-only continuations — so a verbose tool block keeps its complete
text. Child reasoning and streamed reply text are never shown. Presentation failures are swallowed; an
ambiguous first send is never repeated, and only typed transient transport failures are retried.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("gateway.run")

_RUNNING, _DONE, _INTERRUPTED, _TIMEOUT, _FAILED = "⏳", "✅", "⏹️", "⏱️", "❌"
_STATUS_GLYPHS = {"completed": _DONE, "interrupted": _INTERRUPTED, "timeout": _TIMEOUT,
                  "failed": _FAILED, "error": _FAILED}

# Failure kinds we may retry: both mean the request demonstrably did not land, so re-issuing an edit
# (idempotent — identical text) or an unlanded first send cannot duplicate visible content. Everything
# else — timeouts (may have landed), formatting, permissions, a deleted message — is terminal here.
_RETRYABLE_KINDS = frozenset({"transient", "rate_limited"})
# Bounded inline backoff: attempt N sleeps _RETRY_DELAYS[N-1]; once the budget is spent, the card stops.
_RETRY_DELAYS = (0.5, 2.0)


def _no_mentions(text: str) -> str:
    """Break Discord mention syntax (users/roles/channels, @everyone/@here) with a zero-width space."""
    return (str(text).replace("<@", "<\u200b@").replace("<#", "<\u200b#")
            .replace("@everyone", "@\u200beveryone").replace("@here", "@\u200bhere"))


def _redact(text: str) -> str:
    from agent.redact import redact_sensitive_text
    return redact_sensitive_text(text)


# A verbose terminal unit: (header line, complete command). Kept structured so a command past the
# platform cap can be split into self-contained fenced parts that still reassemble exactly.
_Block = Tuple[str, str]
_Unit = Any  # str for a line, _Block for a verbose block


def _strip_fences(part: str) -> str:
    """The command text of one fenced part, without its header or fence scaffolding."""
    body = part.split("```\n", 1)[1] if "```\n" in part else part
    return body[: body.rfind("\n```")] if body.endswith("\n```") else body


class _Child:
    __slots__ = ("key", "batch", "index", "count", "model", "goal", "tools", "tool_count", "status", "reason",
                 "duration", "started")

    def __init__(self, key: Tuple[str, str], batch: str, kw: Dict[str, Any], goal: str) -> None:
        self.key, self.batch, self.goal = key, batch, goal
        self.index, self.count = int(kw.get("task_index") or 0), int(kw.get("task_count") or 1)
        self.model = str(kw.get("model") or "")
        self.tools: List[Tuple[str, Optional[_Block]]] = []  # (compact line, verbose block or None)
        self.tool_count, self.reason, self.duration = 0, "", None
        self.status: Optional[str] = None
        self.started = time.monotonic()


class DelegatedChildProgress:
    """Per-dispatching-turn progress for its delegated children; ``on_event`` is safe from any thread."""

    EDIT_INTERVAL = 2.0
    RECENT_TOOLS = 3
    MAX_TRANSPORT_RETRIES = 2
    # Bounded continuation messages for lossless overflow. A verbose command far past the platform cap
    # spills into continuations; past this many, the last one carries an explicit, visible count.
    MAX_CHUNKS = 8

    def __init__(self, *, adapter: Any, loop: Any, chat_id: str, metadata: Optional[dict], reply_to: Any,
                 verbose: bool, preview_cap: int, verbose_cap: int, retain: Any = None,
                 native_sink: Optional[Callable[[str], None]] = None,
                 turn_current: Optional[Callable[[], bool]] = None) -> None:
        self.adapter, self.loop, self.chat_id = adapter, loop, chat_id
        self.metadata = dict(metadata) if metadata else None
        self.reply_to, self.verbose, self.preview_cap, self.verbose_cap = reply_to, verbose, preview_cap, verbose_cap
        self._retain = retain
        # Native activity lane: a callable relaying rendered text into the platform's own tool-activity
        # presentation, plus the predicate saying that lane is still alive. Absent -> card only.
        self._native_sink, self._turn_current = native_sink, turn_current
        # Lines awaiting relay into the native lane, in event order. Draining a queue (rather than
        # re-rendering) is what keeps a live turn from showing the children twice.
        self._native_queue: List[str] = []
        self._lock = threading.Lock()
        self._children: Dict[Tuple[str, str], _Child] = {}
        self._batches: List[str] = []
        self._wake: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task] = None
        self._msg_id: Optional[str] = None
        self._last_text: Optional[str] = None
        self._last_publish = 0.0
        self._transport_retries = 0
        self._retry_at = 0.0
        self._dead = False  # ambiguous/refused delivery: never send again, never edit an unverified id
        # Lossless chunking state: chunk 0 is pinned to the activity units it held when the first
        # continuation was created (so later edits never re-flow text already delivered), and every
        # continuation is written exactly once.
        self._chunk0_units: Optional[int] = None
        self._chunk_ids: List[str] = []
        self._chunk_text: List[str] = []

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
                recorded = self._tool_lines(tool_name, preview, args)
                child.tools = (child.tools + [recorded])[-self.RECENT_TOOLS:]
                self._native_queue.append(recorded[0])
            elif event_type == "subagent.complete" and child.status is None:
                status = str(kw.get("status") or "completed")
                child.status = status
                duration = kw.get("duration_seconds")  # frozen, so a settled card renders identically
                child.duration = duration if isinstance(duration, (int, float)) else time.monotonic() - child.started
                self._native_queue.append(self._head_line(child))
                if status != "completed" and status != "interrupted":
                    from tools.delegate_tool_progress import describe_subagent_failure
                    child.reason = describe_subagent_failure(kw.get("failure_reason"), kw.get("summary") or preview)
            else:
                return
        try:
            self.loop.call_soon_threadsafe(self._kick)
        except RuntimeError:  # loop closed (shutdown) — nothing left to present on
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
        if self.verbose and isinstance(command, str) and command.strip() and "```" not in command:
            # Verbose keeps the complete command, like the parent's own verbose terminal blocks.
            block = (f"{emoji} {verb or tool_name}", _no_mentions(_redact(command.rstrip())))
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

    def _native_live(self) -> bool:
        """True while the platform's own activity presentation can carry these children."""
        if self._native_sink is None:
            return False
        if self._turn_current is None:
            return True
        try:
            return bool(self._turn_current())
        except Exception:
            return False

    async def _publish_loop(self) -> None:
        wake = self._wake
        assert wake is not None  # _kick creates it before the task
        try:
            while not self._dead:
                await wake.wait()
                delay = self.EDIT_INTERVAL - (time.monotonic() - self._last_publish)
                if delay > 0:
                    await asyncio.sleep(delay)
                if self._retry_at > time.monotonic():  # a spent retry budget still honors its backoff
                    await asyncio.sleep(min(1.0, self._retry_at - time.monotonic()))
                wake.clear()  # everything up to now is in this render
                if self._native_live():
                    # The native lane owns the presentation while the turn is live: relay what it has
                    # not seen and post nothing of our own, so the children never appear twice.
                    sink = self._native_sink
                    while sink is not None and self._native_queue:
                        sink(self._native_queue.pop(0))
                    self._last_text = self.render()
                    self._last_publish = time.monotonic()
                    if self._settled() and not wake.is_set() and not self._native_queue:
                        return
                    continue
                else:
                    await self._deliver_chunks()
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
                    await asyncio.wait_for(self._deliver_chunks(), 3)
                except BaseException:  # noqa: BLE001 — presentation only; the cancel still propagates
                    pass
            raise
        except Exception:
            logger.debug("delegated child progress publisher failed", exc_info=True)
            self._dead = True

    @staticmethod
    def _failure_kind(result: Any) -> str:
        """Typed failure kind for a SendResult: the adapter's own classification, else one classifier."""
        kind = getattr(result, "error_kind", None)
        if isinstance(kind, str) and kind:
            return kind
        from gateway.platforms.base import classify_send_error
        return classify_send_error(None, str(getattr(result, "error", "") or ""))

    def _retry_decision(self, result: Any, *, editable: bool) -> str:
        """``"retry"`` or ``"stop"`` for a failed call.

        Only typed transient transport failures retry, within a bounded budget. An *edit* re-issues
        identical text, so a retry can never duplicate visible content. A *first send* may have landed
        even when it failed, so it retries only when the retryable kind proves nothing reached the wire.
        """
        if result is None:
            return "stop"
        kind = self._failure_kind(result)
        if not editable and getattr(result, "retryable", False) is not True:
            # An untyped failure on a send is ambiguous: never risk a duplicate card.
            return "stop"
        if kind in _RETRYABLE_KINDS and self._transport_retries < self.MAX_TRANSPORT_RETRIES:
            return "retry"
        return "stop"

    def _schedule_retry(self) -> None:
        self._transport_retries += 1
        delay = _RETRY_DELAYS[min(self._transport_retries - 1, len(_RETRY_DELAYS) - 1)]
        self._retry_at = time.monotonic() + delay
        if self._wake is not None:
            self._wake.set()  # paced retry (edits are idempotent)

    async def _deliver_chunks(self) -> None:
        """Publish the render: edit/create the pinned head, then append any new continuation once."""
        if self._dead:
            return
        chunks = self.render_chunks()
        if not chunks:
            return
        head = chunks[0]
        if self._msg_id is None:
            try:
                result = await self.adapter.send(chat_id=self.chat_id, content=head, reply_to=self.reply_to,
                                                 metadata=self.metadata)
            except Exception:
                logger.debug("delegated child progress send failed", exc_info=True)
                result = None
            message_id = getattr(result, "message_id", None) if getattr(result, "success", False) else None
            if not message_id:
                if self._retry_decision(result, editable=False) == "retry":
                    self._schedule_retry()
                    return
                # Refused, failed or delivered without an id: cannot reconcile, so never re-send.
                self._dead = True
                return
            self._msg_id = str(message_id)
            self._chunk_text = [head]
            self._last_text = head
        elif head != (self._chunk_text[0] if self._chunk_text else None):
            kwargs: Dict[str, Any] = {"chat_id": self.chat_id, "message_id": self._msg_id, "content": head}
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
                self._transport_retries = 0
                self._chunk_text[0] = head
                self._last_text = head
            else:
                if self._retry_decision(result, editable=True) == "retry":
                    self._schedule_retry()
                else:
                    # Not retryable: the message is gone or we may not edit it. Stop — never post a
                    # second card.
                    self._dead = True
                return
        else:
            self._last_text = head
        # Append-only continuations: written once, never rewritten, so a growing head cannot duplicate text.
        if len(chunks) > 1:
            await self._append_continuations(chunks[1:])

    async def _append_continuations(self, chunks: List[str]) -> bool:
        """Send continuations that have not been delivered yet. False when a send failed terminally."""
        for offset, text in enumerate(chunks, start=1):
            if offset < len(self._chunk_text):  # already on screen; the unit list is append-only
                continue
            if offset > len(self._chunk_text):
                break  # a gap (an earlier continuation failed) — never deliver out of order
            previous = self._chunk_ids[-1] if self._chunk_ids else self._msg_id
            try:
                result = await self.adapter.send(chat_id=self.chat_id, content=text, reply_to=previous,
                                                 metadata=self.metadata)
            except Exception:
                logger.debug("delegated child progress continuation failed", exc_info=True)
                return False
            if not getattr(result, "success", False) or not getattr(result, "message_id", None):
                return False
            self._chunk_ids.append(str(result.message_id))
            self._chunk_text.append(text)
        return True

    # ── rendering ──────────────────────────────────────────────────────────────────────────

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

    def _child_units(self, children: List[_Child]) -> Tuple[List[str], List[str]]:
        """``(head units, activity units)`` for the current children.

        Activity units are append-only by construction: a tool line is recorded once and never rewritten,
        so a continuation already on screen can never disagree with a later render.
        """
        heads = [self._head_line(child) for child in children]
        activity: List[_Unit] = []
        for child in children:
            for i, (line, block) in enumerate(child.tools):
                last = i == len(child.tools) - 1
                keep_block = bool(block) and last and child.status is None
                activity.append(block if keep_block else f"└ {line}")
            if child.reason:
                reason = child.reason if len(child.reason) <= 160 else child.reason[:157] + "..."
                activity.append(f"└ ⚠️ {_no_mentions(_redact(reason))}")
        return heads, activity

    def _split_block(self, header: str, command: str, limit: int, len_fn: Callable[[str], int]) -> List[str]:
        """Split one verbose command into self-contained fenced parts.

        Parts break on whitespace, keep the space at the end of the part they cut, and carry no chunk
        indicators, so the parts' bodies concatenate back to the command character for character — the
        whole command reaches the chat even when it is several times the platform cap.
        """
        parts: List[str] = []
        rest, first = command, True
        while True:
            head = f"{header}\n```\n" if first else "```\n"
            budget = limit - len_fn(head) - len("\n```")
            if budget < 1:
                budget = max(1, limit // 2)
            if len_fn(rest) <= budget:
                parts.append(f"{head}{rest}\n```")
                return parts
            window = rest[:budget]
            cut = window.rfind(" ")
            cut = cut + 1 if cut > 0 else len(window)
            parts.append(f"{head}{rest[:cut]}\n```")
            rest, first = rest[cut:], False

    def _unit_parts(self, unit: _Unit, limit: int, len_fn: Callable[[str], int]) -> List[str]:
        """One unit, or its parts when the unit alone is past the cap.

        A verbose block is split by :meth:`_split_block` (lossless, whitespace-aligned). A plain line
        falls back to the adapter's canonical splitter, which closes and reopens fences.
        """
        if isinstance(unit, tuple):
            return self._split_block(unit[0], unit[1], limit, len_fn)
        if len_fn(unit) <= limit:
            return [unit]
        splitter: Optional[Callable[[str, int], List[str]]] = getattr(self.adapter, "truncate_message", None)
        if callable(splitter):
            try:
                parts = [p for p in splitter(unit, limit) if p]
                if parts:
                    return parts
            except Exception:
                logger.debug("delegated child progress split failed", exc_info=True)
        # No usable splitter: cut on line boundaries so the head still renders; the caller's explicit
        # marker accounts for anything beyond the chunk budget.
        return [part for part in unit.splitlines() if part] or [unit]

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

    def render_chunks(self) -> List[str]:
        """The full render as bounded chunks: a pinned editable head plus append-only continuations."""
        with self._lock:
            children = sorted(self._children.values(), key=lambda c: (self._batches.index(c.batch), c.index))
            heads, activity = self._child_units(children)
        limit, len_fn = self._limits()
        if self._chunk0_units is None:
            if len(self._pack(heads + activity, limit, len_fn)) <= 1:
                return self._pack(heads + activity, limit, len_fn)
            # Pin the head chunk to the longest activity prefix that still fits; everything after it
            # belongs to continuations and must never re-flow into the edited head.
            keep = 0
            while keep < len(activity) and len(self._pack(heads + activity[: keep + 1], limit, len_fn)) <= 1:
                keep += 1
            self._chunk0_units = keep
        keep = self._chunk0_units
        head_chunks = self._pack(heads + activity[:keep], limit, len_fn)
        chunks = head_chunks[:1] or [""]
        rest = head_chunks[1:] + (self._pack(activity[keep:], limit, len_fn) if len(activity) > keep else [])
        if len(rest) + 1 > self.MAX_CHUNKS:
            omitted = len(rest) - (self.MAX_CHUNKS - 2)
            rest = rest[: self.MAX_CHUNKS - 2] + [f"…(+{omitted} more chunk(s) not shown)"]
        return chunks + rest

    def render(self) -> str:
        """The complete render across every chunk — never a silently truncated preview."""
        return "\n".join(c for c in self.render_chunks() if c)
