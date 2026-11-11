"""Session activity shared by platform adapters, independent of transport."""


class SessionActivityMixin:
    def has_background_work(self, session_key: str) -> bool:
        """Whether this canonical session key owns live completion-producing work.

        Read in-memory state only: no process probing or profile/config lookup.
        Foreground turns and silent servers are not included. This query does
        not transfer ownership or interrupt work.
        """
        if not session_key:
            return False
        from tools.async_delegation import has_live_for_session
        from tools.process_registry import process_registry

        return (has_live_for_session(session_key=session_key)
                or process_registry.has_completion_work_for_session(session_key))

    async def interrupt_session_activity(self, session_key: str, chat_id: str, metadata=None) -> None:
        """Signal the active session loop to stop and clear typing immediately."""
        if session_key and session_key in self._active_sessions:
            self._active_sessions[session_key].set()
        await self._stop_typing_quietly(chat_id, metadata)
