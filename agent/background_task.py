"""
Async detach + wake mechanism for long-running background tasks.

Usage (from a tool handler):

    from agent.background_task import background_tasks, current_session_origin

    origin = current_session_origin()
    handle = background_tasks.create(
        coro=_generate(prompt),    # coroutine that returns a wake_text string
        session_key=origin.session_key,
        origin=origin,
        label="music generation",
    )
    if handle is None:
        # No active session (CLI) or a task already running — fall back to sync.
        return None

    return {"status": "started", "task_id": handle.task_id}

The coroutine should return a string that will be injected as the wake message.
Any unhandled exception is caught by the gateway wrapper and converted to an
error wake message automatically.

The gateway drains ``background_tasks`` after each agent turn via
``background_tasks.drain_pending()``, schedules each entry as an asyncio
task via ``_run_background_task``, which runs the coroutine and injects
the result back into the originating session.

Query active task (for duplicate guard / status action):

    handle = background_tasks.get_active(session_key)
    if handle:
        return {"status": handle.status, "task_id": handle.task_id}
"""

import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Coroutine, Optional

logger = logging.getLogger(__name__)


@dataclass
class SessionOrigin:
    """Snapshot of session context variables captured at task-creation time."""
    session_key: str
    platform: str
    chat_id: str
    thread_id: str = ""
    user_id: str = ""
    user_name: str = ""


@dataclass
class BackgroundTaskHandle:
    task_id: str
    session_key: str
    origin: SessionOrigin
    label: str
    status: str = "running"  # running | succeeded | failed


@dataclass
class _PendingEntry:
    coro: Coroutine
    handle: BackgroundTaskHandle


class BackgroundTaskRegistry:
    """
    Registry for async background tasks.

    Thread-safe. Shared between tool threads and the gateway asyncio loop.
    Follows the same pending-list + gateway-drain pattern as
    ``process_registry.pending_watchers``.
    """

    def __init__(self) -> None:
        # Tasks queued for the gateway to schedule (asyncio.create_task).
        # Drained atomically via drain_pending() — do not access directly.
        self._pending: list[_PendingEntry] = []

        # One active task per session_key (duplicate guard + status queries).
        self._active: dict[str, BackgroundTaskHandle] = {}

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Tool-facing API
    # ------------------------------------------------------------------

    def create(
        self,
        *,
        coro: Coroutine,
        session_key: str,
        origin: SessionOrigin,
        label: str = "background task",
    ) -> Optional[BackgroundTaskHandle]:
        """Register a background task and queue it for gateway scheduling.

        Returns ``None`` if:
        - ``session_key`` is empty (no gateway session — caller runs sync), or
        - a task is already active for this session (duplicate guard).
        """
        if not session_key:
            return None
        with self._lock:
            if session_key in self._active:
                return None
            handle = BackgroundTaskHandle(
                task_id=str(uuid.uuid4()),
                session_key=session_key,
                origin=origin,
                label=label,
            )
            self._active[session_key] = handle
            self._pending.append(_PendingEntry(coro=coro, handle=handle))
        logger.debug(
            "background_task: created %s (%s) for session %.20s",
            handle.task_id, label, session_key,
        )
        return handle

    def get_active(self, session_key: str) -> Optional[BackgroundTaskHandle]:
        """Return the active task for *session_key*, or ``None``."""
        with self._lock:
            return self._active.get(session_key)

    def drain_pending(self) -> "list[_PendingEntry]":
        """Atomically remove and return all queued pending entries.

        Safe to call from any thread or the asyncio event loop. Use this
        instead of accessing ``_pending`` directly to avoid races between
        tool threads appending and drain sites consuming.
        """
        with self._lock:
            entries, self._pending = self._pending, []
        return entries

    # ------------------------------------------------------------------
    # Gateway-facing API (called from _run_background_task)
    # ------------------------------------------------------------------

    def _mark_done(self, handle: BackgroundTaskHandle, status: str) -> None:
        with self._lock:
            handle.status = status
            self._active.pop(handle.session_key, None)


# Module-level singleton — import this in tools and gateway.
background_tasks = BackgroundTaskRegistry()


def current_session_origin() -> SessionOrigin:
    """Read current session context variables and return a ``SessionOrigin``.

    Call this at the start of a tool handler, before any ``await``, so the
    contextvars are still set for the current task.
    """
    from gateway.session_context import get_session_env
    return SessionOrigin(
        session_key=get_session_env("HERMES_SESSION_KEY"),
        platform=get_session_env("HERMES_SESSION_PLATFORM"),
        chat_id=get_session_env("HERMES_SESSION_CHAT_ID"),
        thread_id=get_session_env("HERMES_SESSION_THREAD_ID"),
        user_id=get_session_env("HERMES_SESSION_USER_ID"),
        user_name=get_session_env("HERMES_SESSION_USER_NAME"),
    )
