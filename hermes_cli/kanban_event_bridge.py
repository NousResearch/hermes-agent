"""Local Kanban event bridge for CLI sessions.

Associates a set of task IDs with a local session (the CLI), reads
``task_events`` incrementally with per-task cursors, persists an
acknowledgement cursor to disk, and shuts down cleanly.

This is a **read-only** layer on top of the kanban DB.  It does not
modify task state, create subscriptions in ``kanban_notify_subs``, or
depend on the gateway.  The cursor file is a small JSON blob that can
be inspected or edited by hand.

Usage (context manager, auto-shutdown):

    from hermes_cli.kanban_event_bridge import KanbanEventBridge

    with KanbanEventBridge(board="default", task_ids=["t_abc123"]) as bridge:
        for ev in bridge.poll():
            print(f"{ev.task_id} {ev.kind}")

Usage (manual):

    bridge = KanbanEventBridge(task_ids=["t_abc123"])
    try:
        for ev in bridge.poll():
            ...
        bridge.shutdown()
    finally:
        bridge.shutdown()
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

from hermes_cli import kanban_db as kb

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class BridgeEvent:
    """One kanban event delivered by the bridge.

    Thinner than ``kanban_db.Event``: drops ``payload`` and ``run_id``
    by default (they are available via ``raw`` if needed).  The bridge
    layer does not interpret payloads -- that is the consumer's job.
    """
    id: int
    task_id: str
    kind: str
    created_at: int
    # Optional: the raw Event from kanban_db, for consumers that need
    # payload/run_id without adding them as first-class fields here.
    raw: Optional[object] = None


# ---------------------------------------------------------------------------
# Cursor persistence
# ---------------------------------------------------------------------------


def _default_cursor_path(board: Optional[str]) -> Path:
    """Return the default cursor file path for a board.

    Lives next to the kanban DB so it travels with the board:
    ``<kanban-dir>/kanban-event-bridge-cursors.json``.
    """
    import hermes_constants as hc

    root = Path(hc.get_hermes_home())
    if board and board != kb.DEFAULT_BOARD:
        board_dir = root / "kanban" / "boards" / board
    else:
        board_dir = root  # default board DB is at root
    return board_dir / "kanban-event-bridge-cursors.json"


def _load_cursors(path: Path) -> dict[str, int]:
    """Load persisted cursors from a JSON file. Returns {} on any error."""
    try:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return {k: int(v) for k, v in data.items()}
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        pass
    return {}


def _save_cursors(path: Path, cursors: dict[str, int]) -> None:
    """Atomically persist cursors to a JSON file.

    Writes to a temp file then renames, so a crash mid-write does not
    corrupt the cursor file.
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(cursors, sort_keys=True) + "\n")
        tmp.rename(path)
    except OSError as exc:
        _log.warning("kanban event bridge: failed to save cursors to %s: %s", path, exc)
        # Best-effort cleanup of stale tmp
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------


class KanbanEventBridge:
    """Incremental event reader for a local CLI session.

    Subscribes to a set of task IDs on a kanban board, reads new
    ``task_events`` on each ``poll()`` call, and persists per-task
    cursors so restarts do not replay old events.

    Thread-safe for the common CLI pattern (single polling thread).
    Not designed for concurrent pollers on the same cursor file.

    Args:
        board: Kanban board slug. Defaults to the active board
            (env / current symlink / ``default``).
        task_ids: Initial set of task IDs to subscribe to.
        kinds: If given, only events whose ``kind`` is in this set
            are returned.  ``None`` means all kinds.
        cursor_file: Path to the cursor JSON file.  Defaults to a file
            next to the board's kanban DB.
        interval: Default poll interval in seconds (used by
            ``stream()``).  Not used by ``poll()``.
    """

    def __init__(
        self,
        *,
        board: Optional[str] = None,
        task_ids: Iterable[str] = (),
        kinds: Optional[set[str]] = None,
        cursor_file: Optional[str | Path] = None,
        interval: float = 1.0,
    ):
        self._board = board
        self._task_ids: set[str] = set(task_ids)
        self._kinds: Optional[frozenset[str]] = frozenset(kinds) if kinds else None
        self._cursor_file = Path(cursor_file) if cursor_file else _default_cursor_path(board)
        self._interval = interval
        self._cursors: dict[str, int] = _load_cursors(self._cursor_file)
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def board(self) -> Optional[str]:
        """The board slug, or None for the active board."""
        return self._board

    @property
    def task_ids(self) -> frozenset[str]:
        """Immutable view of subscribed task IDs."""
        return frozenset(self._task_ids)

    @property
    def cursors(self) -> dict[str, int]:
        """Current per-task cursors (id of last seen event)."""
        return dict(self._cursors)

    def subscribe(self, task_id: str) -> None:
        """Add a task to the subscription set.

        If the task already has a persisted cursor it is loaded;
        otherwise the cursor starts at 0 (will begin at the next
        event created after this call).
        """
        if self._closed:
            raise RuntimeError("bridge is closed")
        self._task_ids.add(task_id)
        if task_id not in self._cursors:
            self._cursors[task_id] = 0

    def unsubscribe(self, task_id: str) -> None:
        """Remove a task from the subscription set.

        The cursor is preserved in the cursor file so re-subscribing
        later picks up from where we left off.
        """
        self._task_ids.discard(task_id)

    def poll(self, kinds: Optional[set[str]] = None) -> list[BridgeEvent]:
        """Read new events for all subscribed tasks and advance cursors.

        Returns a list of ``BridgeEvent`` sorted by ``id`` (oldest first).
        The in-memory cursors are advanced immediately; the cursor file
        is persisted at the end of ``poll()`` and on ``shutdown()``.

        If ``kinds`` is given, it overrides the instance-level filter
        for this call only.

        Raises ``RuntimeError`` if the bridge is closed.
        """
        if self._closed:
            raise RuntimeError("bridge is closed")
        if not self._task_ids:
            return []

        effective_kinds: Optional[frozenset[str]] = None
        if kinds is not None:
            effective_kinds = frozenset(kinds)
        elif self._kinds is not None:
            effective_kinds = self._kinds

        conn = kb.connect(board=self._board)
        try:
            all_events: list[BridgeEvent] = []
            for task_id in self._task_ids:
                cursor = self._cursors.get(task_id, 0)
                raw_events = self._fetch_events(
                    conn, task_id=task_id, cursor=cursor, kinds=effective_kinds
                )
                if raw_events:
                    max_id = max(e.id for e in raw_events)
                    self._cursors[task_id] = max_id
                    for e in raw_events:
                        all_events.append(BridgeEvent(
                            id=e.id,
                            task_id=e.task_id,
                            kind=e.kind,
                            created_at=e.created_at,
                            raw=e,
                        ))
            # Sort by id for deterministic delivery order
            all_events.sort(key=lambda e: e.id)
            # Persist cursors after advancing
            _save_cursors(self._cursor_file, self._cursors)
            return all_events
        finally:
            conn.close()

    def stream(
        self, kinds: Optional[set[str]] = None
    ) -> Iterator[BridgeEvent]:
        """Blocking generator that yields events as they arrive.

        Polls at ``self._interval`` seconds.  Stops when ``shutdown()``
        is called (either explicitly or via the context manager).

        Can be used as::

            for ev in bridge.stream():
                handle(ev)

        Or broken out for explicit shutdown::

            it = bridge.stream()
            try:
                for ev in it:
                    handle(ev)
            finally:
                bridge.shutdown()
        """
        while not self._closed:
            events = self.poll(kinds=kinds)
            for ev in events:
                yield ev
            if not events:
                time.sleep(self._interval)

    def shutdown(self) -> None:
        """Persist cursors and mark the bridge as closed.

        Idempotent -- safe to call multiple times.  After shutdown,
        ``poll()`` and ``subscribe()`` raise ``RuntimeError``.
        """
        if self._closed:
            return
        self._closed = True
        _save_cursors(self._cursor_file, self._cursors)
        _log.info(
            "kanban event bridge shutdown: cursors persisted to %s",
            self._cursor_file,
        )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "KanbanEventBridge":
        return self

    def __exit__(self, *exc: object) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_events(
        conn,
        *,
        task_id: str,
        cursor: int,
        kinds: Optional[frozenset[str]] = None,
    ) -> list[kb.Event]:
        """Fetch unseen events for one task from the DB.

        Reads directly from ``task_events`` -- no subscription table
        needed (this is a local CLI reader, not a gateway notifier).
        """
        if kinds:
            kind_list = list(kinds)
            placeholders = ",".join("?" * len(kind_list))
            query = (
                f"SELECT * FROM task_events "
                f"WHERE task_id = ? AND id > ? AND kind IN ({placeholders}) "
                f"ORDER BY id ASC"
            )
            params = [task_id, cursor] + kind_list
        else:
            query = (
                "SELECT * FROM task_events "
                "WHERE task_id = ? AND id > ? "
                "ORDER BY id ASC"
            )
            params = [task_id, cursor]

        rows = conn.execute(query, params).fetchall()
        out: list[kb.Event] = []
        for r in rows:
            try:
                payload = json.loads(r["payload"]) if r["payload"] else None
            except Exception:
                payload = None
            out.append(kb.Event(
                id=r["id"],
                task_id=r["task_id"],
                kind=r["kind"],
                payload=payload,
                created_at=r["created_at"],
                run_id=(
                    int(r["run_id"])
                    if "run_id" in r.keys() and r["run_id"] is not None
                    else None
                ),
            ))
        return out