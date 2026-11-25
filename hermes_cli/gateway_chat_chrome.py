"""Plugin-supplied terminal chrome for attached chats: an animated tab title and a status-bar segment.

The classic CLI runs no plugin code (plugins live in the gateway), so a plugin that wants to show
something in the user's terminal pushes ``plugin.<plugin_id>.terminal`` to the session's viewers with
``ctx.emit_to_session(session_id, "terminal", payload)``. Every payload field is optional; the latest
event from a plugin replaces its previous one, and ``SessionState`` keeps it for viewers that attach
later, so a replayed stream settles on the current state:

    frames       list[str]  title prefixes shown in turn (at most 16, each at most 8 characters)
    interval_ms  int        time per frame; 0 or a single frame is static (100 ms to 60 s)
    label        str        text after the frame (at most 60 characters)
    status       str        status-bar segment (at most 60 characters)
    tone         str        status-bar style: strong | good | warn | dim (the skin's ``status_bar_*``)
    clear        bool       drop this plugin's chrome

The payload is untrusted plugin text: each string is cut to a bounded length, escape sequences are
removed and every other non-printable character folds to a space before anything reaches the terminal.
Each plugin animates on its own clock. Events may arrive on any thread; the title is written only from
the prompt_toolkit application's own event loop (an OSC title sequence through ``app.output``, like
every other write the app makes). When the chat ends the title is restored on Windows, where the
console can report it; elsewhere it is cleared, since terminals do not report their title. Consumer:
the classic CLI (``GatewayAgent`` + ``CLIStatusBarMixin``).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from contextlib import suppress

from tools.ansi_strip import strip_ansi

logger = logging.getLogger(__name__)

EVENT_SUFFIX = ".terminal"
_PREFIX = "plugin."
_PLUGIN_ID_RE = re.compile(r"[a-z0-9_-]{1,64}")
MAX_FRAMES, MAX_FRAME_CHARS, MAX_TEXT_CHARS = 16, 8, 60
MIN_INTERVAL_MS, MAX_INTERVAL_MS = 100, 60_000
MAX_RAW_CHARS = 512  # how much of each untrusted string is read, before escapes are stripped
TONES = frozenset({"strong", "good", "warn", "dim"})


def plugin_of(kind) -> str | None:
    """The plugin id of a ``plugin.<id>.terminal`` event kind, else None."""
    if not isinstance(kind, str) or not kind.startswith(_PREFIX) or not kind.endswith(EVENT_SUFFIX):
        return None
    plugin_id = kind[len(_PREFIX):-len(EVENT_SUFFIX)]
    return plugin_id if _PLUGIN_ID_RE.fullmatch(plugin_id) else None


def _clean(text, limit: int, *, keep_edges: bool = False) -> str:
    if not isinstance(text, str):
        return ""
    folded = "".join(ch if ch.isprintable() else " " for ch in strip_ansi(text[:MAX_RAW_CHARS]))
    line = folded if keep_edges else " ".join(folded.split())
    return line[:limit - 1] + "…" if len(line) > limit else line


def sanitize(payload) -> dict | None:
    """The renderable form of one payload; None when it clears the plugin's chrome."""
    if not isinstance(payload, dict) or payload.get("clear") is True:
        return None
    raw_frames = payload.get("frames")
    frames = [f for f in (_clean(f, MAX_FRAME_CHARS, keep_edges=True)
                          for f in (raw_frames if isinstance(raw_frames, list) else [])[:MAX_FRAMES]) if f.strip()]
    interval = payload.get("interval_ms")
    interval = interval if isinstance(interval, int) and not isinstance(interval, bool) else 0
    tone = payload.get("tone") if payload.get("tone") in TONES else ""
    interval = min(max(MIN_INTERVAL_MS, interval), MAX_INTERVAL_MS) if interval > 0 else 0
    return {"frames": frames, "interval_ms": interval,
            "label": _clean(payload.get("label"), MAX_TEXT_CHARS),
            "status": _clean(payload.get("status"), MAX_TEXT_CHARS), "tone": tone}


def _console_title() -> str | None:
    """The console's current title on Windows (prompt_toolkit can set but not read it), else None."""
    if os.name != "nt":
        return None
    with suppress(Exception):
        import ctypes
        buf = ctypes.create_unicode_buffer(1024)
        if ctypes.windll.kernel32.GetConsoleTitleW(buf, len(buf)):
            return buf.value
    return None


class TerminalChrome:
    """Each plugin's latest chrome; once started on an application's loop, it animates the title."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}  # insertion order = update order; the newest owns the title
        self._loop = None
        self._changed = None
        self._task = None
        self._output = None
        self._app = None
        self._title_set = False
        self._original_title = None

    # -- state (any thread) --------------------------------------------------------------------
    def update(self, plugin_id: str, payload) -> None:
        entry = sanitize(payload)
        with self._lock:
            self._entries.pop(plugin_id, None)
            if entry is not None and (entry["frames"] or entry["label"] or entry["status"]):
                entry["since"] = self._clock()
                self._entries[plugin_id] = entry
        self._wake()

    def reset(self) -> None:
        """Forget every plugin's chrome (the view moved to another session)."""
        with self._lock:
            self._entries.clear()
        self._wake()

    def _wake(self) -> None:
        loop, changed = self._loop, self._changed
        if loop is not None and changed is not None:
            with suppress(RuntimeError):  # the loop already closed: nothing left to wake
                loop.call_soon_threadsafe(changed.set)

    @staticmethod
    def _period(entry) -> float | None:
        """Seconds per frame of an animated entry, else None."""
        return entry["interval_ms"] / 1000 if len(entry["frames"]) > 1 and entry["interval_ms"] else None

    def _frame(self, entry, now: float) -> str:
        frames, period = entry["frames"], self._period(entry)
        if not frames:
            return ""
        return frames[int((now - entry["since"]) // period) % len(frames)] if period else frames[0]

    def title(self) -> str | None:
        now = self._clock()
        with self._lock:
            titled = [e for e in self._entries.values() if e["frames"] or e["label"]]
            if not titled:
                return None
            return f"{self._frame(titled[-1], now)} {titled[-1]['label']}".strip() or None

    def segments(self) -> list[tuple[str, str]]:
        """``(tone, text)`` for each plugin's status-bar segment, oldest first."""
        now = self._clock()
        with self._lock:
            return [(e["tone"], f"{self._frame(e, now)} {e['status']}".strip())
                    for e in self._entries.values() if e["status"]]

    def next_change(self) -> float | None:
        """Seconds until any plugin's next frame, or None when nothing animates."""
        now = self._clock()
        with self._lock:
            periods = [(e["since"], p) for e in self._entries.values() if (p := self._period(e))]
        return min(p - (now - since) % p for since, p in periods) if periods else None

    # -- rendering (the application's loop) ----------------------------------------------------
    def start(self, output, app=None) -> None:
        """Render through ``output`` from the running loop; ``app`` is invalidated for each frame."""
        if self._task is not None:
            return
        self._output, self._app = output, app
        self._original_title = _console_title()
        self._loop = asyncio.get_running_loop()
        self._changed = asyncio.Event()
        self._task = asyncio.ensure_future(self._run())

    async def stop(self) -> None:
        """Stop animating and restore the title (from the loop that started it)."""
        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        self.restore()

    def restore(self) -> None:
        """Hand the title back if chrome ever replaced it: the original on Windows, cleared elsewhere.
        Safe once the loop is gone."""
        self._loop = None
        if self._title_set:
            self._write_title(self._original_title)
            self._title_set = False

    def _write_title(self, title: str | None) -> None:
        try:
            if title is None:
                self._output.clear_title()
            else:
                self._output.set_title(title)
            self._output.flush()
        except Exception:  # health: allow BLE001 -- a terminal that refuses a title must never break the chat
            logger.debug("terminal title write failed", exc_info=True)

    def _invalidate(self) -> None:
        app = self._app
        if app is not None and getattr(app, "is_running", False):
            try:
                app.invalidate()
            except Exception:  # health: allow BLE001 -- a repaint request must never break the chat
                logger.debug("status bar repaint failed", exc_info=True)

    def _paint(self, last: str | None) -> str | None:
        title = self.title()
        if title is not None and title != last:
            self._write_title(title)
            self._title_set = True
        elif title is None and self._title_set:
            self._write_title(self._original_title)
            self._title_set = False
        self._invalidate()
        return title

    async def _run(self) -> None:
        last = None
        while True:
            self._changed.clear()  # before reading state: an update from here on wakes the next pass
            last = self._paint(last)
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._changed.wait(), self.next_change())
