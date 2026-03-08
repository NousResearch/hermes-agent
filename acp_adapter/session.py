"""SessionManager — maps ACP session IDs to AIAgent instances.

Each ACP session gets its own ``AIAgent`` configured for non-interactive use
(``quiet_mode=True``, ``platform="acp"``), plus a ``ToolBridge`` for
delegating file/terminal operations back to the editor.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """State for a single ACP session."""

    agent: Any  # AIAgent instance
    bridge: Any  # ToolBridge instance
    history: list[dict[str, Any]] = field(default_factory=list)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    cwd: str = ""


class SessionManager:
    """Thread-safe registry of ACP sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = threading.Lock()
        self._next_id = 0

    def generate_id(self) -> str:
        with self._lock:
            sid = str(self._next_id)
            self._next_id += 1
            return sid

    def create(
        self,
        session_id: str,
        conn: Any,
        loop: asyncio.AbstractEventLoop,
        cwd: str = "",
    ) -> SessionState:
        """Create a new session with an AIAgent and ToolBridge.

        Args:
            session_id: ACP session identifier.
            conn: ACP client connection for editor interaction.
            loop: asyncio event loop that owns the ACP connection.
            cwd: Working directory for the session.

        Returns:
            The new ``SessionState``.
        """
        import sys
        from pathlib import Path

        project_root = Path(__file__).resolve().parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from run_agent import AIAgent
        from acp_adapter.tool_bridge import ToolBridge

        bridge = ToolBridge(conn=conn, session_id=session_id, loop=loop)

        agent = AIAgent(
            quiet_mode=True,
            platform="acp",
        )
        # Attach the bridge so _execute_tool_calls can delegate file/terminal tools.
        agent._acp_tool_bridge = bridge

        state = SessionState(
            agent=agent,
            bridge=bridge,
            history=[],
            cwd=cwd,
        )

        with self._lock:
            self._sessions[session_id] = state

        logger.info("Created ACP session %s (cwd=%s)", session_id, cwd)
        return state

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.get(session_id)

    def list_all(self) -> list[tuple[str, SessionState]]:
        """Return all (session_id, state) pairs."""
        with self._lock:
            return list(self._sessions.items())

    def remove(self, session_id: str) -> None:
        with self._lock:
            state = self._sessions.pop(session_id, None)
        if state is not None:
            try:
                state.agent._cleanup_task_resources()
            except Exception:
                pass

    def cleanup_all(self) -> None:
        """Remove all sessions and clean up their resources."""
        with self._lock:
            sids = list(self._sessions.keys())
        for sid in sids:
            self.remove(sid)

    def load(
        self,
        session_id: str,
        conn: Any,
        loop: asyncio.AbstractEventLoop,
        cwd: str = "",
    ) -> SessionState:
        """Load an existing Hermes session by ID.

        If the session is already tracked, returns it directly.
        Otherwise creates a fresh ``SessionState`` and attempts to
        hydrate conversation history from the Hermes SQLite database.
        """
        existing = self.get(session_id)
        if existing is not None:
            if cwd:
                existing.cwd = cwd
            return existing

        state = self.create(session_id, conn, loop, cwd)

        # Try to restore conversation history from the session DB.
        try:
            from hermes_state import SessionDB

            db = SessionDB()
            messages = db.get_messages(session_id)
            db.close()
            if messages:
                state.history = messages
                logger.info(
                    "Loaded %d messages for session %s from DB",
                    len(messages),
                    session_id,
                )
        except Exception as exc:
            logger.debug("Could not restore session %s from DB: %s", session_id, exc)

        return state
