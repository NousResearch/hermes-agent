"""CLI notification ownership, structured queueing and last-moment consumption."""


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

    def _background_notifications_mode(self) -> str:
        """This CLI session's ``display.background_process_notifications`` (default ``concise``).

        The mode gates only the turn-starting injection, as in the gateway watcher: events are still
        drained, claimed and acknowledged (#123114). ``result``/``error`` are accepted by the config
        loader and preserved by migration 34→35, but this surface only compared against ``off`` and
        silently treated both as ``concise`` (#153198); the per-mode rules now live in
        ``tools.process_registry_notifications.notification_mode_surfaces``.
        """
        from tools.process_registry_notifications import normalize_notification_mode
        try:
            from cli import CLI_CONFIG
            raw = (CLI_CONFIG.get("display") or {}).get("background_process_notifications")
        except (ImportError, AttributeError, TypeError):
            return "concise"
        return "off" if raw is False else normalize_notification_mode(raw)

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
            heartbeat_display_text, notification_mode_surfaces)

        claimed = []
        for event, text in process_registry.drain_notifications(
            session_key=getattr(self, "session_id", "") or "", owns_event=self._owns_process_notification,
        ):
            claim = claim_event_delivery(event, consumer)
            if claim is None:
                continue
            claimed.append((event, text))
            complete_event_delivery(event, claim)
        mode = self._background_notifications_mode()
        # A muted notice is still drained, claimed and acknowledged above; only the turn-starting
        # injection is skipped. Subagent results are not process notifications: they still land.
        claimed = [(event, text) for event, text in claimed if notification_mode_surfaces(mode, event)]
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
            from hermes_cli.gateway_tools_notify import arm_adopted
            arm_adopted(self, user_input)
            user_input = (user_input.text, user_input.images) if user_input.images else user_input.text
        return user_input, is_voice_input, is_seeded_query

