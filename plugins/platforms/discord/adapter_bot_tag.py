"""Discord bot-tag handoffs: one tag plus its unmentioned continuation chunks is one handoff.

A bot that tags this one (``<@self> brief``) often sends the rest of the handoff as unmentioned
messages: a long text split at Discord's 2000-char cap, or one file per message. A short window
per (channel, author) admits those chunks. Each live chunk waits until its tag is handled, then
follows the tag's routing: in the tag's auto-thread when it opened one (one thread, one session),
inline when the tag ran inline, and merged into the tag's pending batch, files included, so the
whole handoff is one turn."""
from __future__ import annotations

import asyncio
import enum
import logging
import time
from typing import Any, Optional

from gateway.platforms.event import MessageEvent, MessageType

logger = logging.getLogger(__name__)

# A merged batch is a text turn: an audio or voice chunk would be transcribed, not attached.
_UNBATCHED_TYPES = frozenset({MessageType.COMMAND, MessageType.AUDIO, MessageType.VOICE})


class ThreadRoute(enum.Enum):
    """Where a bot tag ran, as its chunks follow it."""

    UNDECIDED = "undecided"  # the tag never reached the auto-thread decision (dropped, in a thread)
    INLINE = "inline"  # the channel auto-threads, but the tag ran inline (a reply)
    THREAD = "thread"  # the tag runs in ``BotTagWindow.thread``
    FAILED = "failed"  # creating the tag's thread failed or was cancelled; the tag said so


class BotTagWindow:
    """One continuation window: when it closes, where its tag ran, and the chunks on their way.

    The tag's thread lives on the window, so it expires with it. A window does not close before
    its tag is handled (thread creation can retry for longer than the window). ``inflight`` counts
    the chunks not yet in their batch; a batch they feed is held until the last one is in."""

    __slots__ = ("flush_key", "handled", "inflight", "route", "thread", "until")

    def __init__(self, until: float) -> None:
        self.until = until
        self.handled = asyncio.Event()
        self.route = ThreadRoute.UNDECIDED
        self.thread: Any = None
        self.inflight = 0
        self.flush_key: Optional[str] = None

    def is_closed(self, now: float) -> bool:
        return self.until <= now and self.handled.is_set()

    def rearm(self, until: float) -> None:
        self.until = max(self.until, until)

    def mark_handled(self, until: float) -> None:
        self.until = until
        self.handled.set()

    def publish(self, route: ThreadRoute, thread: Any = None) -> None:
        self.route, self.thread = route, thread

    def hold(self, key: str) -> bool:
        """Hold the batch ``key`` while a chunk of this window is on its way in."""
        if not self.inflight:
            return False
        self.flush_key = key
        return True

    def leave(self) -> Optional[str]:
        """One chunk is done; return the batch key to flush once the last chunk of a held batch is."""
        self.inflight -= 1
        if self.inflight or self.flush_key is None:
            return None
        key, self.flush_key = self.flush_key, None
        return key


class BotTagHandoff:
    """The live message's part in a bot handoff: the tag that opened ``window``, or a chunk of it."""

    __slots__ = ("is_tag", "window")

    def __init__(self, window: BotTagWindow, *, is_tag: bool) -> None:
        self.window = window
        self.is_tag = is_tag


class DiscordBotTagMixin:
    name: str
    _EMPTY_MESSAGE_TEXT: str
    _text_batch_delay_seconds: float
    _text_batch_split_delay_seconds: float
    _bot_tag_windows: dict[str, BotTagWindow]
    _pending_text_batches: dict
    _pending_text_batch_tasks: dict
    _dedup: Any
    _threads: Any
    _self_is_explicitly_mentioned: Any
    _text_batch_key: Any
    _flush_text_batch: Any
    _handle_message: Any
    _auto_create_thread: Any
    _notify_auto_thread_failed: Any

    @staticmethod
    def _bot_tag_debounce_key(message: Any) -> str:
        return (
            f"{getattr(getattr(message, 'channel', None), 'id', '')}:"
            f"{getattr(getattr(message, 'author', None), 'id', '')}"
        )

    def _bot_tag_window_seconds(self) -> float:
        return max(self._text_batch_delay_seconds, self._text_batch_split_delay_seconds)

    def _record_bot_tag_debounce(self, message: Any) -> Optional[BotTagWindow]:
        """Open a short continuation window after a bot-authored tag; return it, else None."""
        if (
            self._text_batch_delay_seconds <= 0
            or not getattr(message.author, "bot", False)
            or not self._self_is_explicitly_mentioned(message)
        ):
            return None
        now = time.monotonic()
        for key in [key for key, window in self._bot_tag_windows.items() if window.is_closed(now)]:
            del self._bot_tag_windows[key]
        window = BotTagWindow(now + self._bot_tag_window_seconds())
        self._bot_tag_windows[self._bot_tag_debounce_key(message)] = window
        return window

    def _live_bot_tag_window(self, message: Any) -> Optional[BotTagWindow]:
        """The open window of ``message``'s author in its channel, re-armed; None once it closed.

        A hit re-arms the window: Discord paces a bot's sends at roughly one per
        second, so chunk N of a long handoff lands well after the tag itself; each
        admitted chunk therefore vouches for the next one. The gateway bot loop
        guard bounds a bot that never stops talking."""
        if self._text_batch_delay_seconds <= 0 or not getattr(message.author, "bot", False):
            return None
        key = self._bot_tag_debounce_key(message)
        window = self._bot_tag_windows.get(key)
        now = time.monotonic()
        if window is None or window.is_closed(now):
            self._bot_tag_windows.pop(key, None)
            return None
        window.rearm(now + self._bot_tag_window_seconds())
        return window

    def _is_bot_tag_debounce_continuation(self, message: Any) -> bool:
        """Return whether an unmentioned chunk belongs to a recent bot tag (re-arms the window)."""
        return self._live_bot_tag_window(message) is not None

    def _bot_tag_handoff(self, message: Any) -> Optional[BotTagHandoff]:
        """A live message's part in a bot handoff: a tag opens a window, a chunk continues one."""
        window = self._record_bot_tag_debounce(message)
        if window is not None:
            return BotTagHandoff(window, is_tag=True)
        if self._self_is_explicitly_mentioned(message):
            return None
        window = self._live_bot_tag_window(message)
        return BotTagHandoff(window, is_tag=False) if window is not None else None

    async def _handle_live_message(self, message: Any, *, role_authorized: bool) -> bool:
        """``_handle_message`` for a live event. A bot tag marks its window handled once its own
        event is queued; each chunk of the window waits for that, so it follows the tag's routing
        and merges behind it, never ahead."""
        handoff = self._bot_tag_handoff(message)
        chunk_window = handoff.window if handoff is not None and not handoff.is_tag else None
        if chunk_window is not None:
            chunk_window.inflight += 1
        try:
            if chunk_window is not None:
                await chunk_window.handled.wait()
            return await self._handle_message(message, role_authorized=role_authorized, bot_tag=handoff)
        finally:
            if chunk_window is not None:
                self._release_bot_tag_chunk(chunk_window)
            elif handoff is not None:
                handoff.window.mark_handled(time.monotonic() + self._bot_tag_window_seconds())

    def _release_bot_tag_chunk(self, window: BotTagWindow) -> None:
        """The chunk is in its batch (or went elsewhere): flush a batch it held back."""
        key = window.leave()
        if key is None or key not in self._pending_text_batches:
            return
        task = self._pending_text_batch_tasks.get(key)
        if task is None or task.done():
            self._pending_text_batch_tasks[key] = asyncio.create_task(self._flush_text_batch(key))

    def _bot_tag_auto_threads(self, handoff: Optional[BotTagHandoff], wants_thread: bool) -> bool:
        """Whether a top-level message runs the auto-thread step. A tag records that it ran inline;
        a chunk follows its tag's choice, and decides alone only when the tag never made one."""
        if handoff is None:
            return wants_thread
        window = handoff.window
        if handoff.is_tag:
            if not wants_thread:
                window.publish(ThreadRoute.INLINE)
            return wants_thread
        if window.route is ThreadRoute.UNDECIDED:
            return wants_thread
        return window.route is not ThreadRoute.INLINE

    async def _open_auto_thread(
        self, message: Any, handoff: Optional[BotTagHandoff],
    ) -> tuple[Optional[Any], bool]:
        """``(thread, created)`` for a top-level message that auto-threads. A bot-tag chunk runs in
        its tag's thread (``created`` False). ``(None, False)`` skips the run: on a failure, never
        fall back to an inline parent-channel reply (dumps the task into a shared channel); the
        user gets a short visible error to retry once Discord recovers (#20243)."""
        window = handoff.window if handoff is not None else None
        if window is not None and not handoff.is_tag and window.route is not ThreadRoute.UNDECIDED:
            if window.route is ThreadRoute.FAILED:
                logger.info("[%s] Dropping bot-tag chunk %s: its tag's auto-thread failed",
                            self.name, getattr(message, "id", "?"))
            return window.thread, False
        if window is not None and handoff.is_tag:
            window.publish(ThreadRoute.FAILED)  # until it succeeds: a cancelled creation stays failed
        thread = await self._auto_create_thread(message)
        if not thread:
            await self._notify_auto_thread_failed(message)
            return None, False
        if window is not None and handoff.is_tag:
            window.publish(ThreadRoute.THREAD, thread)
        # Pre-seed dedup: message.create_thread() fires a second MESSAGE_CREATE for the starter
        # (id == thread.id, maybe type=default); mark it so it can't trigger a rerun. Must run
        # before the first await below: mark_async yields to the loop, and the echo's
        # _discord_message_admission would otherwise claim the id first.
        self._dedup.is_duplicate(str(thread.id))
        await self._threads.mark_async(str(thread.id))
        return thread, True

    def _mark_bot_tag_event(self, event: MessageEvent, message: Any, handoff: Optional[BotTagHandoff]) -> None:
        """Flag an event of a bot handoff for the batcher (split-delay wait, held flush)."""
        if handoff is None and not (
            getattr(getattr(message, "author", None), "bot", False)
            and self._is_bot_tag_debounce_continuation(message)
        ):
            return
        event._bot_tag_debounce = True  # type: ignore[attr-defined]
        event._bot_tag_windows = {handoff.window} if handoff is not None else set()  # type: ignore[attr-defined]

    def _batches_bot_tag_event(self, event: MessageEvent, handoff: Optional[BotTagHandoff]) -> bool:
        """Whether a bot-handoff event that is not plain text still goes through the text batcher:
        a tag opens a batch so its chunks merge into one turn, and a chunk's file joins a pending
        batch, without the empty-message placeholder (the merged turn already has text)."""
        if handoff is None or event.message_type in _UNBATCHED_TYPES:
            return False
        if handoff.is_tag:
            return True
        if self._text_batch_key(event) not in self._pending_text_batches:
            return False
        if event.text == self._EMPTY_MESSAGE_TEXT:
            event.text = ""
        return True

    def _enqueue_text_event(self, event: MessageEvent) -> None:
        """Queue as the base does; a batch fed by several handoffs is held for each of them."""
        windows = getattr(event, "_bot_tag_windows", None)
        super()._enqueue_text_event(event)  # type: ignore[misc]
        if not windows:
            return
        pending = self._pending_text_batches.get(self._text_batch_key(event))
        if pending is not None and pending is not event:
            pending._bot_tag_windows = getattr(pending, "_bot_tag_windows", set()) | windows

    def _pop_text_batch(self, key: str) -> Optional[MessageEvent]:
        """Remove and return the pending batch for ``key``, unless a bot-tag chunk that feeds it is
        still on its way in (a slow file read): then keep it pending and return None; the last
        such chunk restarts the flush."""
        windows = getattr(self._pending_text_batches.get(key), "_bot_tag_windows", ())
        if [window for window in windows if window.hold(key)]:
            return None
        return super()._pop_text_batch(key)  # type: ignore[misc]

    def _text_batch_delay_for(self, pending: Optional[MessageEvent]) -> float:
        """A bot handoff's continuation chunks arrive at Discord's send rate (~1/s), so a
        batch opened by a bot tag waits the split delay regardless of chunk length."""
        if getattr(pending, "_bot_tag_debounce", False):
            return self._text_batch_split_delay_seconds
        return super()._text_batch_delay_for(pending)  # type: ignore[misc]
