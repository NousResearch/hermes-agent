# gateway/daimon/discord_hooks.py
"""Discord adapter integration hooks for Daimon.

These functions are called by the Discord adapter at specific lifecycle points.
They encapsulate all Daimon logic so the adapter changes are minimal (just calls to these).
"""
from __future__ import annotations

import logging
from typing import Optional, Any

from gateway.daimon.session_manager import DaimonSessionManager, SessionStartResult
from gateway.daimon.admin_commands import handle_daimon_command, CommandResult
from gateway.daimon.window_buffer import WindowBuffer, BufferedMessage, format_window_context

logger = logging.getLogger(__name__)


class DaimonDiscordHooks:
    """Lifecycle hooks for Daimon integration with Discord adapter.

    Instantiated once by the adapter. Provides methods called at each lifecycle point.
    """

    def __init__(self, raw_config: dict) -> None:
        self._manager: DaimonSessionManager | None = None
        self._banned: set[str] = set()
        self._queued: dict[str, Any] = {}  # thread_id → thread object (for promotion notification)
        self._window_buffer = WindowBuffer()

        try:
            self._manager = DaimonSessionManager(raw_config)
            if not self._manager.is_active:
                self._manager = None
                logger.debug("[Daimon] Inactive — no admin_users configured")
            else:
                # Configure buffer size from config
                self._window_buffer = WindowBuffer(
                    max_per_thread=self._manager.config.max_buffer_per_thread
                    if hasattr(self._manager.config, 'max_buffer_per_thread')
                    else 50
                )
                logger.info("[Daimon] Active with %d admin(s)", len(self._manager.config.admin_users))
                # Recover bans from DB
                try:
                    self._banned = self._manager.db.get_all_bans()
                except Exception:
                    pass
        except Exception as e:
            logger.warning("[Daimon] Init failed: %s", e)
            self._manager = None

    @property
    def active(self) -> bool:
        """Whether Daimon access control is active."""
        return self._manager is not None

    @property
    def manager(self) -> DaimonSessionManager | None:
        return self._manager

    def is_banned(self, user_id: str) -> bool:
        """Check if a user is banned."""
        return user_id in self._banned

    def buffer_message(self, thread_id: str, author_name: str, author_id: str, content: str, has_attachments: bool = False, message_id: str = "") -> None:
        """Buffer a non-mention message for later context flush."""
        from datetime import datetime, timezone
        if message_id and self._window_buffer.has_seen(thread_id, message_id):
            return  # dedup
        if message_id:
            self._window_buffer.mark_seen(thread_id, message_id)
        msg = BufferedMessage(
            author_name=author_name,
            author_id=author_id,
            content=content,
            timestamp=datetime.now(timezone.utc),
            has_attachments=has_attachments,
        )
        self._window_buffer.append(thread_id, msg)

    def flush_window(self, thread_id: str) -> str:
        """Flush the window buffer and return formatted context string.

        Returns empty string if no messages buffered.
        """
        buffered = self._window_buffer.flush(thread_id)
        return format_window_context(buffered)

    def clear_buffer(self, thread_id: str) -> None:
        """Clear buffer for a thread (cleanup on close)."""
        self._window_buffer.clear(thread_id)

    def is_duplicate_trigger(self, thread_id: str, message_id: str) -> bool:
        """Check if an @mention trigger message is a duplicate (dedup)."""
        if self._window_buffer.has_seen(thread_id, message_id):
            return True
        self._window_buffer.mark_seen(thread_id, message_id)
        return False

    def should_process_in_thread(self, author_id: str, thread_id: str, role_ids: Optional[list[str]] = None) -> tuple[bool, str]:
        """Check if a message should be processed (thread ownership + turn cap).

        Returns (allowed, denial_reason):
        - (True, "") — process the message
        - (False, "") — silent ignore (ownership/role)
        - (False, "reason") — deny with message (turn cap hit)
        """
        if not self._manager:
            return True, ""
        return self._manager.should_process_message(author_id, thread_id, role_ids=role_ids)

    def on_thread_created(
        self, thread_id: str, creator_id: str, raw_config: dict
    ) -> SessionStartResult:
        """Called when a new thread is created for a user.

        Returns SessionStartResult indicating if session started, queued, or denied.
        """
        if not self._manager:
            return SessionStartResult(allowed=True)

        # Check ban first
        if creator_id in self._banned:
            return SessionStartResult(
                allowed=False,
                denial_reason="You have been banned from using Daimon.",
            )

        return self._manager.start_session(thread_id, creator_id, raw_config)

    def on_thread_closed(self, thread_id: str) -> Optional[str]:
        """Called when a thread is archived/closed.

        Cleans up session resources. Returns promoted thread_id if any.
        """
        if not self._manager:
            return None

        # Remove from queued tracking
        self._queued.pop(thread_id, None)

        return self._manager.end_session(thread_id)

    def queue_thread(self, thread_id: str, thread_obj: Any) -> None:
        """Store a thread object for later promotion notification."""
        self._queued[thread_id] = thread_obj

    def pop_queued(self, thread_id: str) -> Any | None:
        """Pop and return a queued thread object for promotion."""
        return self._queued.pop(thread_id, None)

    def handle_admin_command(self, subcommand: str, args: str) -> CommandResult:
        """Handle a /daimon admin subcommand."""
        if not self._manager:
            return CommandResult(success=False, message="Daimon is not active.")
        return handle_daimon_command(subcommand, args, self._manager, self._banned)

    def redact(self, text: str) -> str:
        """Apply output redaction for user sessions."""
        if not self._manager:
            return text
        return self._manager.redact(text)

    async def recover_thread_ownership(self, client) -> int:
        """Recover thread ownership from Discord API on gateway restart.

        Queries all active threads the bot is in, registers their creators.
        Called once after Discord connect.

        Args:
            client: The discord.py Client/Bot instance

        Returns:
            Number of threads recovered.
        """
        if not self._manager:
            return 0

        recovered = 0
        try:
            for guild in client.guilds:
                # Fetch active threads in this guild
                threads = await guild.fetch_active_threads() if hasattr(guild, 'fetch_active_threads') else None
                if not threads:
                    continue
                for thread in (threads.threads if hasattr(threads, 'threads') else threads):
                    owner_id = str(thread.owner_id) if thread.owner_id else None
                    if owner_id:
                        self._manager._threads.register(str(thread.id), owner_id)
                        recovered += 1
        except Exception as e:
            logger.debug("Thread recovery error: %s", e)

        return recovered
