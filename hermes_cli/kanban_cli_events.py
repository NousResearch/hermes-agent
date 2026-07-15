"""Kanban event bridge for the interactive CLI.

Connects the kanban event stream to the CLI lifecycle:
- Polls task_events from the kanban DB on a background thread.
- Renders concise notices while the agent is idle.
- Queues events while a model turn is active (no fake user messages).
- Recovers missed events on session resume via persisted cursor.
- Persists cursor on clean shutdown.

Non-goals: no autonomous continuation, no policy routing.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from hermes_cli.kanban_db import Event, kanban_db_path

_log = logging.getLogger(__name__)

# Event kinds surfaced to the operator.
NOTABLE_KINDS = frozenset({
    "completed",
    "blocked",
    "failed",
    "timed_out",
    "crashed",
})

DEFAULT_POLL_INTERVAL = 5.0
MAX_EVENTS_PER_TICK = 10
MAX_CATCH_UP_EVENTS = 200


@dataclass
class KanbanEventBridge:
    """Background kanban event watcher for the interactive CLI.

    Args:
        on_render:          Called with a formatted event line string.
                            Signature: ``callback(line: str) -> None``.
        is_agent_running:   Callable returning True while a model turn is active.
        on_queue_nonempty:  Optional callback when events are queued during an
                            active turn (so the CLI can show a badge/count).
        board:              Kanban board slug. None means auto-resolve.
        poll_interval:      Seconds between DB polls.
    """

    on_render: Callable[[str], None] = lambda line: print(line, flush=True)
    is_agent_running: Callable[[], bool] = lambda: False
    on_queue_nonempty: Optional[Callable[[], None]] = None
    board: Optional[str] = None
    poll_interval: float = DEFAULT_POLL_INTERVAL
    task_ids: Optional[set[str]] = None

    # Internal mutable state
    _cursor: int = field(default=0, repr=False)
    _queue: list[Event] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _watcher_thread: Optional[threading.Thread] = field(default=None, repr=False)
    _db_path: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self._db_path = kanban_db_path(self.board)
        self._cursor = self._load_cursor()

    # ------------------------------------------------------------------
    # Cursor persistence
    # ------------------------------------------------------------------

    def _cursor_file(self) -> Path:
        """Per-board cursor file."""
        root = self._resolve_root()
        if self.board:
            return root / "kanban" / "boards" / self.board / "cli_event_cursor"
        return root / "kanban" / "cli_event_cursor"

    def _resolve_root(self) -> Path:
        hermes_home = os.environ.get("HERMES_HOME")
        if hermes_home:
            return Path(hermes_home)
        return Path.home() / ".hermes"

    def _load_cursor(self) -> int:
        cf = self._cursor_file()
        try:
            if cf.exists():
                return int(cf.read_text().strip())
        except (ValueError, OSError):
            pass
        return 0

    def _save_cursor(self) -> None:
        cf = self._cursor_file()
        try:
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_text(str(self._cursor))
        except OSError as exc:
            _log.debug("failed to persist kanban event cursor: %s", exc)

    # ------------------------------------------------------------------
    # Event polling
    # ------------------------------------------------------------------

    def _query_events(self, after_id: int, limit: int) -> list[Event]:
        """Query notable task events after a cursor, optionally by task."""
        if not self._db_path or not Path(self._db_path).exists():
            return []

        try:
            conn = sqlite3.connect(self._db_path, timeout=5)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                kind_placeholders = ", ".join("?" for _ in NOTABLE_KINDS)
                params: list[object] = [after_id, *sorted(NOTABLE_KINDS)]
                task_clause = ""
                if self.task_ids:
                    task_placeholders = ", ".join("?" for _ in self.task_ids)
                    task_clause = f" AND task_id IN ({task_placeholders})"
                    params.extend(sorted(self.task_ids))
                params.append(limit)
                rows = conn.execute(
                    "SELECT id, task_id, kind, payload, created_at, run_id "
                    "FROM task_events "
                    f"WHERE id > ? AND kind IN ({kind_placeholders}){task_clause} "
                    "ORDER BY id ASC LIMIT ?",
                    params,
                ).fetchall()
            finally:
                conn.close()
        except Exception as exc:
            _log.debug("kanban event query failed: %s", exc)
            return []

        events: list[Event] = []
        for r in rows:
            try:
                payload = json.loads(r["payload"]) if r["payload"] else None
            except Exception:
                payload = None
            run_id = int(r["run_id"] if r["run_id"] is not None else None)
            events.append(Event(
                id=r["id"],
                task_id=r["task_id"],
                kind=r["kind"],
                payload=payload,
                created_at=r["created_at"],
                run_id=run_id,
            ))
        return events

    def subscribe(self, task_ids: str | list[str] | set[str]) -> None:
        """Restrict delivery to the supplied task IDs."""
        if isinstance(task_ids, str):
            task_ids = [task_ids]
        self.task_ids = {task_id.strip() for task_id in task_ids if task_id.strip()}

    def unsubscribe(self, task_ids: str | list[str] | set[str]) -> None:
        """Stop delivering the supplied task IDs."""
        if isinstance(task_ids, str):
            task_ids = [task_ids]
        if self.task_ids is None:
            return
        self.task_ids.difference_update(task_ids)

    def poll(self) -> list[Event]:
        """Poll for new events since _cursor. Advances and persists cursor."""
        events = self._query_events(self._cursor, MAX_EVENTS_PER_TICK)
        if events:
            self._cursor = max(self._cursor, max(e.id for e in events))
            self._save_cursor()
        return events

    def catch_up(self) -> list[Event]:
        """Fetch all missed events since last cursor (session resume)."""
        events = self._query_events(self._cursor, MAX_CATCH_UP_EVENTS)
        if events:
            self._cursor = max(self._cursor, max(e.id for e in events))
            self._save_cursor()
        return events

    # ------------------------------------------------------------------
    # Event formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_event(event: Event) -> str:
        """Format a kanban event as a concise CLI status line."""
        kind = event.kind
        task_id = event.task_id

        if kind == "completed":
            icon, color = "✓", "#8FAF6A"
            detail = ""
            if event.payload:
                s = event.payload.get("summary", "")
                if s:
                    detail = f": {s[:60]}"
        elif kind == "blocked":
            icon, color = "⛔", "#D4A26A"
            detail = ""
            if event.payload:
                r = event.payload.get("reason", "")
                if r:
                    detail = f": {r[:60]}"
        elif kind in ("failed", "crashed"):
            icon, color = "✗", "#D46A6A"
            detail = ""
            if event.payload:
                e = event.payload.get("error", "")
                if e:
                    detail = f": {e[:60]}"
        elif kind == "timed_out":
            icon, color = "⏱", "#D4A26A"
            detail = ""
        else:
            icon, color = "•", "#888888"
            detail = ""

        return f"[dim {color}][{icon}] {task_id} {kind}{detail}[/]"

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------

    def deliver_events(self, events: list[Event]) -> None:
        """Render if idle, queue if agent is running."""
        if not events:
            return

        with self._lock:
            if self.is_agent_running():
                self._queue.extend(events)
                if self.on_queue_nonempty:
                    self.on_queue_nonempty()
            else:
                for event in events:
                    self.on_render(self.format_event(event))
                # Flush previously queued events too
                if self._queue:
                    for event in self._queue:
                        self.on_render(self.format_event(event))
                    self._queue.clear()

    def drain_queue(self) -> list[Event]:
        """Return and clear queued events (called after turn ends."""
        with self._lock:
            events = list(self._queue)
            self._queue.clear()
        return events

    # ------------------------------------------------------------------
    # Background watcher
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._watcher_thread and self._watcher_thread.is_alive():
            return
        self._stop_event.clear()
        self._watcher_thread = threading.Thread(
            target=self._watch_loop,
            name="kanban-event-bridge",
            daemon=True,
        )
        self._watcher_thread.start()
        _log.debug("kanban event bridge started (poll=%ss)", self.poll_interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._watcher_thread:
            self._watcher_thread.join(timeout=10)
        self._save_cursor()
        _log.debug("kanban event bridge stopped")

    def _watch_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                events = self.poll()
                if events:
                    self.deliver_events(events)
            except Exception as exc:
                _log.debug("kanban event bridge tick error: %s", exc)
            deadline = time.monotonic() + self.poll_interval
            while not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._stop_event.wait(min(0.5, remaining))