"""Adapter message-task ownership through final delivery and task completion."""

import logging
from typing import Any

logger = logging.getLogger("gateway.platforms.base")


class BaseSessionWorkMixin:
    def _track_session_task(self, session_key: str, task: Any) -> bool:
        """Track the owner for shutdown; False for unhashable create_task test sentinels."""
        self._session_tasks[session_key] = task
        try:
            self._background_tasks.add(task)
        except TypeError:
            return False
        if hasattr(task, "add_done_callback"):
            task.add_done_callback(self._background_tasks.discard)
            task.add_done_callback(self._expected_cancelled_tasks.discard)
            task.add_done_callback(self._message_work_finished)
        return True

    def _message_work_finished(self, _task: Any) -> None:
        # The runner releases its slot before final delivery. Only a done callback can
        # publish idle after that owner (including cancellation before entry) is gone.
        if self.gateway_runner is not None:
            self.gateway_runner._persist_active_agents()

    def active_message_work_keys(self) -> set[str]:
        """Session keys with live owners, including final delivery; unreadable owners stay active."""
        active = set()
        for session_key, task in list(self._session_tasks.items()):
            try:
                done = task.done()
            except Exception:
                logger.debug("[%s] Non-Task owner in _session_tasks[%s]; counting as active", self.name, session_key, exc_info=True)
                done = False
            if not done:
                active.add(session_key)
        return active
