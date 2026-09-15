"""Per-turn observability board for live structural subagent status (Telegram).

``_ChildProgressRelay`` (tools/delegate_tool_progress.py) already threads
``task_index``/``task_count``/``subagent_id``/``tool_count``/``depth`` into
every ``subagent.*`` event it relays to the parent's progress callback. The
gateway only ever rendered ``subagent.complete`` failures from that stream;
every other child event fell through the ``tool.started``-only progress gate,
so a messaging surface saw nothing until a delegation died.

``SubagentActivityBoard`` turns that stream into one message per parent turn:
one ``send`` when the first child appears, then ``edit_message`` on that id
only. Rules that keep it safe for children that outlive their turn:

* Structural state only — ordinals, phases, elapsed time, tool counts. Never a goal, preview,
  tool argument, model name or error text: a detached child keeps reporting
  after the foreground turn has moved on, so nothing it says is trusted.
* Owned by exactly one ``TurnContext`` and bound to that turn's adapter, chat
  and thread metadata. The adapter's normal routing fallbacks still apply to the
  initial send; after an id returns, updates edit that owned message only.
* Direct children only (``depth == 0``); a nested orchestrator's grandchildren
  are its business.
* One publisher task per board, paced by ``_MIN_EDIT_INTERVAL``: a burst of
  tool events collapses to the latest snapshot per interval instead of a queue
  of edits, and failed edits get bounded retries against the latest state
  rather than falling back to a second message.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Callable, Optional

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter

logger = logging.getLogger(__name__)

# First events can arrive concurrently from child worker threads. Serialize lazy
# board creation so one parent turn cannot orphan multiple initial bubbles.
_BOARD_LOCK = threading.Lock()

_DONE = frozenset({"ok", "completed", "success"})
_TIMEOUT = frozenset({"timeout"})
_STOPPED = frozenset({"interrupted", "cancelled", "canceled", "stalled"})
# Anything else on subagent.complete (error, failed, unknown, novel) is a failure: the child
# ended and did not report success.
_MARKS = {
    "spawned": "🚀",
    "thinking": "💭",
    "working": "⚙️",
    "done": "✅",
    "failed": "❌",
    "timeout": "⏱",
    "stopped": "⛔",
}
_TERMINAL = frozenset({"done", "failed", "timeout", "stopped"})
_EVENTS = frozenset({
    "subagent.start", "subagent.thinking", "subagent.tool", "subagent.heartbeat", "subagent.complete",
})
_MAX_ROWS = 8            # rendered rows; the tail collapses into one "…and N more" line
_MAX_TOOLS = 9999
_MIN_EDIT_INTERVAL = 3.0  # seconds between edits of one bubble (Telegram group budget is ~20/min)
_MAX_MISSES = 5          # consecutive delivery misses before waiting for a later state change
_MAX_RETRY_DELAY = 300.0  # do not let a hostile/buggy retry_after park the publisher indefinitely


def _phase_for(status: Any) -> str:
    status = str(status or "").strip().lower()
    if status in _DONE:
        return "done"
    if status in _TIMEOUT:
        return "timeout"
    if status in _STOPPED:
        return "stopped"
    return "failed"


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _format_elapsed(seconds: Any, *, terminal: bool = False) -> str:
    """Show sampled running age approximately and authoritative completion precisely."""
    try:
        total = max(0, int(float(seconds)))
    except (TypeError, ValueError, OverflowError):
        total = 0
    minutes, secs = divmod(total, 60)
    if terminal:
        if total < 60:
            return f"{total}s"
        if minutes < 60:
            return f"{minutes}m{secs:02d}s"
        hours, minutes = divmod(minutes, 60)
        if hours < 24:
            return f"{hours}h{minutes:02d}m"
        days, hours = divmod(hours, 24)
        return f"{days}d{hours:02d}h"

    if total < 30:
        return "<30s"
    if total < 45:
        return "<1m"

    rounded_minutes = (total + 30) // 60
    if rounded_minutes < 60:
        return f"~{rounded_minutes}m"
    hours, minutes = divmod(rounded_minutes, 60)
    if hours < 24:
        return f"~{hours}h{minutes:02d}m"
    rounded_hours = (total + 1800) // 3600
    days, hours = divmod(rounded_hours, 24)
    return f"~{days}d{hours:02d}h"


class SubagentActivityBoard:
    """Structural, privacy-safe status board owned by exactly one parent turn."""

    def __init__(
        self, adapter: Any, chat_id: str, metadata: Optional[dict], *,
        min_edit_interval: float = _MIN_EDIT_INTERVAL, clock=time.monotonic, sleep=asyncio.sleep,
    ) -> None:
        self._adapter = adapter
        self._chat_id = str(chat_id)
        self._metadata = dict(metadata or {})
        # Status is an interim gateway send. Relay adapters use this marker to avoid
        # sealing an open answer stream with the board's text.
        self._metadata["_interim_send"] = True
        self._min_edit_interval, self._clock, self._sleep = min_edit_interval, clock, sleep
        self._lock = threading.Lock()  # observe() runs on the agent's worker thread
        self._children: dict[str, dict[str, Any]] = {}
        self._expected_by_wave: dict[str, int] = {}
        self._fallback_wave = 0
        self._fallback_indices: set[int] = set()
        self._revision = 0
        self._published_revision = 0
        self._publisher_running = False
        self._delivery_abandoned = False
        self._message_id: Optional[str] = None
        self._last_edit_at: Optional[float] = None
        self._last_text: Optional[str] = None

    # ── agent worker thread ──────────────────────────────────────────────────────────────

    def observe(self, event_type: str, payload: dict[str, Any]) -> bool:
        """Apply one child event. True when a publisher must be started to flush the change."""
        if event_type not in _EVENTS or _as_int(payload.get("depth"), 0) != 0:
            return False
        task_index = max(0, _as_int(payload.get("task_index"), 0))
        key = str(payload.get("subagent_id") or f"task-{task_index}")
        with self._lock:
            child = self._children.get(key)
            if child is None:
                delegation_id = payload.get("delegation_id")
                if delegation_id:
                    wave_key = f"delegation:{delegation_id}"
                else:
                    # Older/synthetic producers have no delegation_id. A repeated task_index is
                    # then the best available boundary between sequential waves.
                    if task_index in self._fallback_indices:
                        self._fallback_wave += 1
                        self._fallback_indices.clear()
                    self._fallback_indices.add(task_index)
                    wave_key = f"fallback:{self._fallback_wave}"
                self._expected_by_wave[wave_key] = max(
                    self._expected_by_wave.get(wave_key, 0), _as_int(payload.get("task_count"), 0),
                )
                child = self._children[key] = {
                    "ordinal": len(self._children) + 1,
                    "phase": "spawned",
                    "tools": 0,
                    "started_at": self._clock(),
                    "elapsed": 0.0,
                }
                changed = True
            else:
                changed = False
            if child["phase"] not in _TERMINAL:
                observed_elapsed = max(0.0, self._clock() - child["started_at"])
                changed |= _format_elapsed(child["elapsed"]) != _format_elapsed(observed_elapsed)
                child["elapsed"] = observed_elapsed
                if event_type == "subagent.thinking":
                    changed |= child["phase"] != "thinking"
                    child["phase"] = "thinking"
                elif event_type == "subagent.tool":
                    tools = min(_MAX_TOOLS, max(child["tools"], _as_int(payload.get("tool_count"), child["tools"] + 1)))
                    changed |= (child["phase"], child["tools"]) != ("working", tools)
                    child["phase"], child["tools"] = "working", tools
                elif event_type == "subagent.complete":
                    child["phase"] = _phase_for(payload.get("status"))
                    duration = payload.get("duration_seconds")
                    child["elapsed"] = (
                        max(0.0, float(duration))
                        if isinstance(duration, (int, float))
                        else max(0.0, self._clock() - child["started_at"])
                    )
                    changed = True
                elif event_type == "subagent.heartbeat":
                    # The delegation heartbeat already runs every 30s. It gives a quiet child enough
                    # board revisions to keep elapsed time useful without adding another timer.
                    pass
            if not changed:
                return False
            self._revision += 1
            if self._publisher_running or self._delivery_abandoned:
                return False
            self._publisher_running = True
            return True

    def publisher_not_started(self) -> None:
        """The caller could not schedule ``run()``; let the next event try again."""
        with self._lock:
            self._publisher_running = False

    # ── gateway loop ─────────────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Publisher: edit the bubble to the latest snapshot until nothing is pending."""
        try:
            while True:
                with self._lock:
                    if self._revision == self._published_revision:
                        self._publisher_running = False
                        return
                    revision, text = self._revision, self._render()
                if self._last_edit_at is not None:
                    await self._sleep(max(0.0, self._last_edit_at + self._min_edit_interval - self._clock()))
                    with self._lock:  # a newer snapshot may have landed while pacing; publish that instead
                        revision, text = self._revision, self._render()
                misses = 0
                while True:
                    editing = self._message_id is not None
                    result = await self._deliver(text)
                    if result is True:
                        break
                    # A nominally successful initial send without an id is still unowned. It may
                    # have posted, so retrying could create a duplicate bubble.
                    if not editing and getattr(result, "success", False):
                        with self._lock:
                            self._delivery_abandoned = True
                            self._publisher_running = False
                        return
                    # A send exception or an explicit permanent/ambiguous failure may have posted
                    # without returning an id. Never risk a duplicate. Edits are idempotent, so an
                    # exception while editing the owned message remains safe to retry.
                    retryable = (editing and result is None) or bool(
                        getattr(result, "retryable", False)
                    )
                    if not retryable:
                        with self._lock:
                            self._delivery_abandoned = True
                            self._publisher_running = False
                        return
                    misses += 1
                    if misses >= _MAX_MISSES:
                        break
                    # Retry against the LATEST state, never a queued second message. Repeated misses
                    # are common during Telegram flood waits, so use bounded progressive backoff.
                    retry_delay = max(
                        self._min_edit_interval * misses,
                        getattr(result, "retry_after", None) or 0.0,
                    )
                    await self._sleep(min(_MAX_RETRY_DELAY, retry_delay))
                    with self._lock:
                        revision, text = self._revision, self._render()
                with self._lock:
                    if result is True:
                        self._published_revision = max(self._published_revision, revision)
                    else:
                        # Keep the failed revision pending. A later state change can start a new
                        # bounded publisher instead of silently treating stale text as delivered.
                        self._publisher_running = False
                        return
        except asyncio.CancelledError:
            # Shutdown can cancel an in-flight initial send after Telegram accepted it but before
            # its id came back. Abandon this board so a late child event cannot create a duplicate.
            with self._lock:
                self._delivery_abandoned = True
                self._publisher_running = False
            raise
        except Exception:
            logger.debug("subagent activity board publisher failed", exc_info=True)
            with self._lock:
                self._publisher_running = False

    async def _deliver(self, text: str):
        """True on success; otherwise the failed SendResult (or None) for its ``retry_after``."""
        if text == self._last_text:
            return True  # an invisible change (e.g. inside the collapsed tail) owes no edit
        try:
            initial_send = self._message_id is None
            if initial_send:
                result = await self._adapter.send(self._chat_id, text, metadata=self._metadata)
                if getattr(result, "success", False) and getattr(result, "message_id", None):
                    self._message_id = str(result.message_id)
            else:
                result = await self._adapter.edit_message(
                    self._chat_id, self._message_id, text, metadata=self._metadata,
                )
        except Exception:
            logger.debug("subagent activity board delivery failed", exc_info=True)
            return None
        self._last_edit_at = self._clock()
        if getattr(result, "success", False):
            if initial_send and self._message_id is None:
                return result
            self._last_text = text
            return True
        return result

    def _render(self) -> str:
        children = sorted(self._children.values(), key=lambda item: item["ordinal"])
        total = max(len(children), sum(self._expected_by_wave.values()))
        finished = sum(item["phase"] in _TERMINAL for item in children)
        lines = [f"🔀 Subagents · {finished}/{total} done"]
        overflow = len(children) - _MAX_ROWS
        shown = children[:_MAX_ROWS]
        for index, item in enumerate(shown):
            connector = "└" if overflow <= 0 and index == len(shown) - 1 else "├"
            line = (
                f"{connector} #{item['ordinal']} {_MARKS[item['phase']]} {item['phase']}"
                f" · {_format_elapsed(item['elapsed'], terminal=item['phase'] in _TERMINAL)}"
            )
            if item["tools"]:
                line += f" · {item['tools']} {'tool' if item['tools'] == 1 else 'tools'}"
            lines.append(line)
        if overflow > 0:
            lines.append(f"└ …and {overflow} more")
        return "\n".join(lines)


async def _publish_board(board: SubagentActivityBoard, runner: Any) -> None:
    """Own one publisher task on the gateway loop, unless shutdown has begun."""
    shutdown_event = getattr(runner, "_shutdown_event", None)
    if getattr(runner, "_running", True) is False or (
        shutdown_event is not None and shutdown_event.is_set()
    ):
        board.publisher_not_started()
        return
    retain = getattr(runner, "_retain_background_task", None)
    if callable(retain):
        retain(asyncio.current_task())
    await board.run()


def observe_subagent_activity(
    ctx: Any,
    runner: Any,
    schedule: Callable[[Any, str], Any],
    event_type: Any,
    payload: dict[str, Any],
) -> None:
    """Project one direct-subagent event into its originating turn's Telegram board.

    Called on agent worker threads before the foreground-generation gate so detached
    subagents can finish updating the board after their parent response has returned.
    """
    if not isinstance(event_type, str) or not event_type.startswith("subagent."):
        return
    if not ctx.subagent_activity_board_enabled:
        return
    platform = getattr(ctx.source, "platform", None)
    if getattr(platform, "value", platform) != Platform.TELEGRAM.value:
        return
    adapter = ctx._status_adapter
    adapter_edit = getattr(type(adapter), "edit_message", None) if adapter is not None else None
    if adapter_edit is None or adapter_edit is BasePlatformAdapter.edit_message:
        return
    # RelayAdapter implements edit_message generically; its destination descriptor
    # remains authoritative about whether this chat can actually edit.
    descriptor_for_chat = getattr(adapter, "_descriptor_for_chat", None)
    if callable(descriptor_for_chat):
        try:
            descriptor = descriptor_for_chat(str(ctx._status_chat_id))
            if not descriptor.supports_edit or not descriptor.supports_op("edit"):
                return
        except Exception:
            return
    try:
        board = ctx._subagent_activity_board
        if board is None:
            with _BOARD_LOCK:
                board = ctx._subagent_activity_board
                if board is None:
                    board = ctx._subagent_activity_board = SubagentActivityBoard(
                        adapter, ctx._status_chat_id, ctx._status_thread_metadata,
                    )
        if board.observe(event_type, payload):
            future = schedule(_publish_board(board, runner), "subagent activity board scheduling error")
            if future is None:
                board.publisher_not_started()
    except Exception:
        logger.debug("subagent activity board failed", exc_info=True)
