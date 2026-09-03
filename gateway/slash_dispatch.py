"""Small slash-dispatch primitives shared by idle and busy gateway paths."""

import logging

from gateway.platforms.base import MessageEvent


logger = logging.getLogger("gateway.run")


class GatewaySlashDispatchMixin:
    """Shared handler lookup and native finite-choice delivery."""

    def _gateway_plain_command_handlers(self):
        """Return ordinary slash handlers shared by idle and busy dispatch."""
        return {
            "status": self._handle_status_command,
            "context": self._handle_context_command,
            "restart": self._handle_restart_command,
            "approve": self._handle_approve_command,
            "deny": self._handle_deny_command,
            "pause": self._handle_pause_command,
            "agents": self._handle_agents_command,
            "bg": self._handle_background_command,
            "btw": self._handle_btw_command,
            "kanban": self._handle_kanban_command,
            "group": self._handle_rooms_command,
            "subgoal": self._handle_subgoal_command,
            "heartbeat": self._handle_heartbeat_command,
            "busy": self._handle_busy_command,
            "yolo": self._handle_yolo_command,
            "verbose": self._handle_verbose_command,
            "footer": self._handle_footer_command,
            "help": self._handle_help_command,
            "commands": self._handle_commands_command,
            "profile": self._handle_profile_command,
            "update": self._handle_update_command,
            "version": self._handle_version_command,
        }

    async def _try_send_choice_picker(
        self,
        event: MessageEvent,
        session_key: str,
        title: str,
        choices: list,
        on_choice_selected,
    ) -> bool:
        """Send a native picker when supported, otherwise use text fallback."""
        adapter = self._adapter_for_source(event.source)
        has_picker = (
            adapter is not None
            and getattr(type(adapter), "send_choice_picker", None) is not None
        )
        if not has_picker:
            return False
        try:
            metadata = dict(
                self._thread_metadata_for_source(
                    event.source, self._reply_anchor_for_event(event)
                )
                or {}
            )
            requester_user_id = getattr(event.source, "user_id", None)
            if requester_user_id is not None:
                metadata["requester_user_id"] = str(requester_user_id)
            result = await adapter.send_choice_picker(
                chat_id=event.source.chat_id,
                title=title,
                choices=choices,
                session_key=session_key,
                on_choice_selected=on_choice_selected,
                metadata=metadata,
            )
            return bool(getattr(result, "success", False))
        except Exception as exc:
            logger.warning("send_choice_picker failed, falling back to text: %s", exc)
            return False
