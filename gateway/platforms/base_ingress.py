"""Adapter ingress before burst and active-session guards."""
import asyncio

from gateway.platforms.event import MessageEvent


async def handle_message(self, event: MessageEvent) -> None:
    """Process an incoming message; returns quickly by spawning a background
    task so new messages (and interrupts) can arrive while an agent runs."""
    from gateway.platforms.base import (Platform, coerce_plaintext_gateway_command,
                                        _thread_metadata_for_event, logger)
    event._gateway_accepted = False
    if not self._message_handler:
        # No handler = every inbound silently discarded on an adapter that still polls and sends;
        # say so once per adapter (#102260).
        if not getattr(self, "_no_message_handler_logged", False):
            self._no_message_handler_logged = True
            logger.error("[%s] Dropping inbound message: no gateway message handler is installed on this "
                         "adapter. The adapter is connected and can send, but every inbound message is "
                         "discarded.", self.name)
        return

    if event.allow_gateway_control:
        coerce_plaintext_gateway_command(event)
    # Identity FIRST: every key below (routing check, guard lookup, batch lane) derives from it.
    if self._drop_unresolved(event):
        return
    expected_session_key = str((event.metadata or {}).get("gateway_session_key") or "").strip()
    # Explicitly routed events already name their destination; recovering a
    # different topic would redirect them and yield before the session claim.
    if (not expected_session_key and getattr(self, "_topic_recovery_fn", None) is not None
            and event.source.platform == Platform.TELEGRAM and event.source.chat_type == "dm"):
        await asyncio.to_thread(self._apply_topic_recovery, event)
    session_key = self._event_session_key(event)
    if expected_session_key and session_key != expected_session_key:
        logger.warning("Dropping internally routed event: expected session=%s derived=%s",
                       expected_session_key, session_key)
        return
    from gateway.session_parallel import route_adapter_input
    if notice := await route_adapter_input(self, event):
        await self.send(chat_id=event.source.chat_id, content=notice,
                        reply_to=event.message_id, metadata=_thread_metadata_for_event(event))
        return
    session_key = self._event_session_key(event)
    from gateway.session_ingress_burst import burst_gate
    behind_burst = await burst_gate(self, event, session_key)  # a serial reader's command waits too
    # On-entry self-heal: clear a guard whose owner task already exited.
    if session_key in self._active_sessions:
        self._heal_stale_session_lock(session_key)
    if session_key in self._active_sessions or behind_burst:
        await self._handle_message_while_active(event, session_key)
        return
    # Guard installed synchronously BEFORE the task spawns so a second message can't race in.
    event._gateway_accepted = self._start_session_processing(event, session_key)
