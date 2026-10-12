"""CLI notification ownership, structured queueing and last-moment consumption."""

import queue

from agent.i18n import t


class CLIProcessNotificationsMixin:
    def _owns_process_notification(self, event: dict) -> bool:
        """Whether this session owns a delegation event (pre-compression keys resolve to their continuation; fail closed)."""
        event_key = str(event.get("session_key") or "")
        current_key = str(getattr(self, "session_id", "") or "")
        if not event_key or not current_key:
            return False
        if event_key == current_key:
            return True
        try:
            session_db = getattr(self, "_session_db", None)
            resolved_key = (
                session_db.resolve_resume_session_id(event_key) if session_db is not None else event_key
            ) or event_key
        except Exception:
            resolved_key = event_key
        return str(resolved_key) == current_key

    def _background_notifications_suppressed(self) -> bool:
        """Whether ``display.background_process_notifications`` is ``off`` for this CLI session.

        The key gates the gateway's completion injection (#9290) but the CLI drain never consulted
        it, so the documented ``off`` escape hatch silently did nothing here (#123114). Mirrors the
        gateway semantics: events are still drained, claimed and acknowledged — only the
        turn-starting injection is suppressed."""
        try:
            from cli import CLI_CONFIG
            mode = str((CLI_CONFIG.get("display") or {}).get("background_process_notifications") or "").strip().lower()
        except Exception:
            return False
        return mode == "off"

    def _drain_process_notifications(self, consumer: str) -> None:
        from hermes_cli.gateway_tools_notify import gateway_store, queue_admitted_turn
        store = gateway_store(self)
        if store is not None:
            # The process registry and its delivery claims live in the gateway, which admits the
            # notification turn itself; this viewer only paints it (IPS-086).
            queue_admitted_turn(self, store)
            return
        from tools.process_registry import process_registry
        from tools.async_delegation import claim_event_delivery, complete_event_delivery
        from tools.process_registry_notifications import (
            HEARTBEAT_DISPLAY_KIND, ProcessNotificationBatch, TimelineNotification, group_process_notifications,
            heartbeat_display_text)

        claimed = []
        for event, text in process_registry.drain_notifications(
            session_key=getattr(self, "session_id", "") or "", owns_event=self._owns_process_notification,
        ):
            claim = claim_event_delivery(event, consumer)
            if claim is None:
                continue
            claimed.append((event, text))
            complete_event_delivery(event, claim)
        if self._background_notifications_suppressed():
            # Subagent results are not process notifications: they still land.
            claimed = [(event, text) for event, text in claimed if event.get("type") == "async_delegation"]
        for notifications in group_process_notifications(claimed):
            event, text = notifications[0]
            evt_type = event.get("type", "completion")
            if evt_type == "completion":
                pending = ProcessNotificationBatch(notifications)
            elif evt_type == "heartbeat":
                pending = TimelineNotification(text, heartbeat_display_text(event), HEARTBEAT_DISPLAY_KIND)
            else:
                pending = TimelineNotification.for_delegation(text, event) if evt_type == "async_delegation" else text
                from agent.notification_presentation import diagnostic_process_event
                if diagnostic_process_event(event) and not isinstance(pending, TimelineNotification):
                    pending = TimelineNotification(text, text, "internal_notification", "diagnostic")
            self._pending_input.put(pending)

    def _delegation_busy_result_mode(self) -> str:
        """``delegation.busy_result_mode``: ``queue`` (default) holds a subagent result that finishes
        mid-turn for the next turn; ``steer`` carries it into the running turn's next tool result."""
        from cli import CLI_CONFIG, logger
        mode = str((CLI_CONFIG.get("delegation") or {}).get("busy_result_mode") or "queue").strip().lower()
        if mode not in ("queue", "steer"):
            logger.warning("delegation.busy_result_mode=%r is not 'queue' or 'steer'; using 'queue'", mode)
            return "queue"
        return mode

    def _offer_delegation_results_to_running_turn(self, agent) -> int:
        """Move this session's finished subagent results from the completion queue onto the running
        agent's tool-result carrier. Claims are acknowledged as the idle drain does when it enqueues;
        ``_requeue_undelivered_delegation_results`` turns any offer no tool batch carried into the
        ordinary next turn, so a result is never dropped."""
        from agent.background_results import offer
        from tools.async_delegation import claim_event_delivery, complete_event_delivery
        from tools.process_registry import process_registry
        from tools.process_registry_notifications import TimelineNotification, format_process_notification

        process_registry.restore_completions()
        queue_ = process_registry.completion_queue
        mine, others = [], []
        for _ in range(queue_.qsize()):
            try:
                event = queue_.get_nowait()
            except queue.Empty:
                break
            owned = event.get("type") == "async_delegation" and self._owns_process_notification(event)
            (mine if owned else others).append(event)
        for event in others:
            queue_.put(event)
        offered = 0
        for event in mine:
            text = format_process_notification(event)
            claim = claim_event_delivery(event, "cli-busy-steer") if text else None
            if claim is None:
                continue
            offer(agent, TimelineNotification.for_delegation(text, event), text)
            complete_event_delivery(event, claim)
            offered += 1
        return offered

    @staticmethod
    def _announce_steered_delegation_results(notifications) -> None:
        from cli import _ACCENT, _RST, _cprint
        for notification in notifications:
            preview = notification.display_text[:80] + ("..." if len(notification.display_text) > 80 else "")
            _cprint(f"  {_ACCENT}{t('cli.tui.steered', preview=preview)}{_RST}")

    def _announce_delegation_results_carried_so_far(self, agent) -> None:
        """Echo results as they land in the running turn, so the notice never trails the reply."""
        from agent.background_results import take_delivered
        self._announce_steered_delegation_results(take_delivered(agent))

    def _requeue_undelivered_delegation_results(self, agent) -> None:
        """After the run: results no tool batch carried (the turn ended without another tool call,
        or was stopped) become the ordinary next turn, exactly as ``queue`` mode would deliver them."""
        from agent.background_results import take_outcome
        delivered, undelivered = take_outcome(agent)
        self._announce_steered_delegation_results(delivered)
        for notification, _text in undelivered:
            self._pending_input.put(notification)

    def _tui_unwrap_input(self, user_input):
        """Unwrap ``_VoiceInputMessage`` / ``_SeededQueryMessage`` -> ``(text_or_tuple, is_voice_input, is_seeded_query)``."""
        from cli import _VoiceInputMessage, _SeededQueryMessage
        from tools.process_registry import process_registry
        from tools.process_registry_notifications import (
            PROCESS_COMPLETE_DISPLAY_KIND, ProcessNotificationBatch, TimelineNotification)
        if isinstance(user_input, ProcessNotificationBatch):
            rendered = user_input.render(process_registry)
            user_input = rendered and TimelineNotification(
                rendered, user_input.display_text(process_registry), PROCESS_COMPLETE_DISPLAY_KIND)
        # Voice-transcribed messages arrive wrapped in a sentinel so only genuine STT output gets the voice
        # prefix (#65827).
        is_voice_input = isinstance(user_input, _VoiceInputMessage)
        if is_voice_input:
            user_input = user_input.text
        is_seeded_query = isinstance(user_input, _SeededQueryMessage)
        if is_seeded_query:
            user_input = (user_input.text, user_input.images) if user_input.images else user_input.text
        return user_input, is_voice_input, is_seeded_query

