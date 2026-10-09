"""Session finalize dispatch and delivery of plugin ``on_session_finalize`` messages for GatewayRunner.

Split out of ``gateway/run_shutdown.py``; bound onto ``GatewayRunner`` via ``GatewayShutdownMixin``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, List, Optional

logger = logging.getLogger("gateway.run")


class GatewaySessionEndMixin:
    """Bounded off-loop ``finalize_session`` plus out-of-band delivery of what plugins return."""

    async def _deliver_session_end_messages(
        self, messages: list[str], *, source: Any = None, session_key: Optional[str] = None,
    ) -> None:
        """Send plugin ``on_session_finalize`` text to the chat that owns the session — a plain adapter send,
        never an agent turn. No resolvable chat (or no adapter) → logged and dropped."""
        if not messages:
            return
        if source is None and session_key:
            target = await self._shutdown_notification_target(session_key)
            source = target[0] if target else None
        adapter = self._delivery_adapter_for(source) if source is not None else None
        if adapter is None:
            logger.info("Session-end plugin message(s) dropped: no chat for session %s", session_key)
            return
        metadata = self._thread_metadata_for_source(source)
        for message in messages:
            await self._send_notice_logged(
                adapter, str(source.chat_id), message, source.platform.value if source.platform else "",
                "Failed to send session-end plugin message to %s:%s: %s", metadata=metadata)

    async def _finalize_session_off_loop(
        self, *, session_id: Any, platform: str, reason: str, session_key: Optional[str] = None, **extra: Any,
    ) -> list[str]:
        """Run hermes_cli.lifecycle.finalize_session off-loop, bounded; on timeout the worker is left alone.
        ``session_key`` lets an unscoped caller (shutdown) enter the owning profile's scope: plugin
        ``on_session_finalize`` observers and the Relay coordinator (``current_profile_key``) resolve
        profile state at call time. Returns the plugins' user-facing messages (empty on timeout/error)."""
        messages: list[str] = []

        def _call() -> None:
            from hermes_cli.lifecycle import finalize_session, session_end_messages
            messages.extend(session_end_messages(
                finalize_session(session_id=session_id, platform=platform, reason=reason, **extra)))

        try:
            await asyncio.wait_for(
                self._run_housekeeping_in_executor(self._run_release_in_profile_scope, _call, (), session_key),
                timeout=self._FINALIZE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Session finalize hooks (%s, reason=%s) exceeded %ss; proceeding without blocking the event loop "
                "(the worker thread is left to finish on its own).", session_id, reason, self._FINALIZE_TIMEOUT_S,
            )
        except Exception:
            logger.debug("Session finalize hooks (%s, reason=%s) failed", session_id, reason, exc_info=True)
        return list(messages)
