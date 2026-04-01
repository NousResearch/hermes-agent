"""
Session capture for action recording.

Provides a global singleton that intercepts tool calls during an agent
session and records them for later replay.
"""

import logging
import threading
from typing import Optional

from recording.store import add_step, create_recording, get_recording

logger = logging.getLogger(__name__)

# Thread-safe global singleton
_lock = threading.Lock()
_active_session: Optional["RecordingSession"] = None


def get_active_session() -> Optional["RecordingSession"]:
    """Return the active recording session, or None if not recording.

    This is called from the agent's tool execution loop. It must be
    fast and safe to call on every tool invocation.
    """
    return _active_session


class RecordingSession:
    """Captures tool calls during an agent session for later replay.

    Usage::

        session = RecordingSession("my-recording")
        session.start()
        # ... agent runs, tool calls are captured via get_active_session() ...
        session.stop()
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._active = False
        self._step_count = 0

    def start(self) -> None:
        """Begin recording. Creates the recording in the store and registers
        this session as the global active session."""
        global _active_session
        with _lock:
            if _active_session is not None:
                raise RuntimeError(
                    f"Another recording is already active: {_active_session.name}"
                )
            # Create the recording file (raises ValueError if exists)
            existing = get_recording(self.name)
            if existing is None:
                create_recording(self.name, self.description)
            self._active = True
            _active_session = self
        logger.info("Recording started: %s", self.name)

    def stop(self) -> dict:
        """Stop recording and deregister from global state.

        Returns:
            Summary dict with name, step_count.
        """
        global _active_session
        with _lock:
            self._active = False
            if _active_session is self:
                _active_session = None
        logger.info("Recording stopped: %s (%d steps)", self.name, self._step_count)
        return {"name": self.name, "step_count": self._step_count}

    def capture_tool_call(
        self, tool_name: str, arguments: dict, result: str, success: bool
    ) -> None:
        """Record a single tool call. Called by the agent loop hook.

        Safe to call from any thread. If the session is not active, this
        is a no-op.
        """
        if not self._active:
            return
        try:
            add_step(self.name, tool_name, arguments, result, success)
            self._step_count += 1
        except Exception as e:
            logger.warning("Failed to capture tool call %s: %s", tool_name, e)

    @property
    def is_active(self) -> bool:
        return self._active
