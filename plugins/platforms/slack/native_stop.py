"""Slack Agent native stop-event bridge.

Slack's Agents & AI Apps surface emits ``agent_session_stopped`` when a user
clicks the native Stop control. The bridge turns that platform lifecycle event
into Hermes' existing ``/stop`` command instead of introducing a second
cancellation primitive.

The runtime adapter subclass is kept separate from ``adapter.py`` so the event
bridge remains small and independently testable while reusing the canonical
Slack adapter for message/session routing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from gateway.platforms.base import MessageEvent, MessageType

from . import adapter as _adapter

logger = logging.getLogger(__name__)

_NATIVE_STOP_REGISTRATION_ATTR = "_hermes_agent_session_stopped_registration"


class SlackAgentStopAdapter(_adapter.SlackAdapter):
    """Slack adapter with native Agent-session stop control wired to ``/stop``."""

    def _start_socket_mode_handler(self) -> None:
        # Register before the Socket Mode reader starts so there is no window
        # where Slack can deliver a stop event only to the catch-all listener.
        self._register_agent_session_stopped_listener()
        super()._start_socket_mode_handler()

    def _register_agent_session_stopped_listener(self) -> None:
        app = self._app
        if app is None:
            return

        # AsyncApp can outlive a concrete adapter across profile reloads and
        # error-path reinitialization. Registration authority therefore belongs
        # to the app, not the adapter instance. Keep one listener per app and
        # retarget its mutable owner to the newest adapter instead of stacking
        # callbacks or leaving a callback bound to a stale adapter.
        registration = getattr(app, _NATIVE_STOP_REGISTRATION_ATTR, None)
        if registration is not None:
            if not isinstance(registration, dict) or "adapter" not in registration:
                raise RuntimeError(
                    "Slack native-stop listener registration marker is invalid"
                )
            registration["adapter"] = self
            return

        registration = {"adapter": self}

        @app.event("agent_session_stopped")
        async def handle_agent_session_stopped(event, body):
            owner = registration.get("adapter")
            if owner is None:
                return
            await owner._handle_agent_session_stopped(event, body)

        setattr(app, _NATIVE_STOP_REGISTRATION_ATTR, registration)

    async def _handle_agent_session_stopped(
        self,
        event: dict,
        body: Optional[dict] = None,
    ) -> None:
        """Route Slack's native Stop button through the canonical ``/stop`` lane."""
        channel_id = str(event.get("channel") or "")
        thread_ts = str(event.get("thread_ts") or "")
        user_id = str(event.get("user") or "")
        team_id = str(
            event.get("team_id")
            or self._event_team_id(event, body)
            or ""
        )
        message_ts = str(event.get("message_ts") or "")

        # Stop is a destructive control-plane action. If Slack did not provide
        # enough identity to bind it to one exact Hermes lane, fail closed.
        if not channel_id or not thread_ts or not user_id:
            logger.warning(
                "[Slack] Ignoring malformed agent_session_stopped event "
                "(channel=%r thread_ts=%r user=%r)",
                channel_id,
                thread_ts,
                user_id,
            )
            return

        if team_id:
            self._remember_channel_team(channel_id, team_id)

        channel_type = str(event.get("channel_type") or "")
        if not channel_type and channel_id.startswith("D"):
            channel_type = "im"
        is_dm = channel_type in {"im", "mpim"}

        user_name = await self._resolve_user_name(
            user_id,
            chat_id=channel_id,
            team_id=team_id,
        )
        source = self.build_source(
            chat_id=channel_id,
            chat_name=self._channel_name_cache.get(
                (team_id, channel_id),
                channel_id,
            ),
            chat_type="dm" if is_dm else "group",
            user_id=user_id,
            user_name=user_name,
            thread_id=thread_ts,
            scope_id=team_id or None,
        )

        stop_event = MessageEvent(
            text="/stop",
            message_type=MessageType.COMMAND,
            source=source,
            raw_message=event,
            message_id=str(
                event.get("event_ts")
                or message_ts
                or f"agent_session_stopped:{thread_ts}"
            ),
            metadata={
                "slack_team_id": team_id,
                "slack_channel_id": channel_id,
                "thread_id": thread_ts,
                "thread_ts": thread_ts,
                "user_id": user_id,
                "native_agent_session_stop": True,
            },
        )

        # This is the only cancellation action. BasePlatformAdapter.handle_message
        # reaches the same gateway /stop path as a typed Slack command, preserving
        # authorization, session guards, agent interruption, cleanup, and the
        # normal user-visible "stopped" acknowledgement.
        await self.handle_message(stop_event)

        # UI cleanup is deliberately separate from cancellation authority. It
        # only settles Slack-native presentation state after the canonical stop
        # has been dispatched, and it is bound to Slack's exact channel/thread/
        # workspace/message identities.
        await self._settle_agent_session_stopped_ui(
            channel_id=channel_id,
            thread_ts=thread_ts,
            team_id=team_id,
            message_ts=message_ts,
        )

    async def _settle_agent_session_stopped_ui(
        self,
        *,
        channel_id: str,
        thread_ts: str,
        team_id: str,
        message_ts: str,
    ) -> None:
        """Best-effort clear of Slack-native stream/status presentation."""
        metadata = {
            "thread_id": thread_ts,
            "thread_ts": thread_ts,
            "slack_team_id": team_id,
        }

        try:
            await self.stop_typing(channel_id, metadata=metadata)
        except Exception:
            logger.debug(
                "[Slack] Failed to clear Assistant status after native stop",
                exc_info=True,
            )

        # ``message_ts`` identifies the concrete native streamed message Slack
        # put the Stop control on. Stop that exact stream even if Hermes lost
        # its in-memory stream bookkeeping across a restart. Never infer a
        # stream from channel alone: concurrent Slack threads share channel IDs.
        if not self._app or not message_ts:
            return

        try:
            await self._get_client(
                channel_id,
                team_id=team_id or None,
            ).chat_stopStream(
                channel=channel_id,
                ts=message_ts,
            )
        except Exception as exc:
            logger.debug(
                "[Slack] chat.stopStream failed after native stop for %s/%s: %s",
                channel_id,
                message_ts,
                exc,
            )
            return

        active_stream = self._active_streams.get(channel_id)
        if (
            active_stream is not None
            and str(active_stream.get("ts") or "") == message_ts
        ):
            self._active_streams.pop(channel_id, None)


def _build_adapter(config):
    return SlackAgentStopAdapter(config)


class _PlatformRegistrationProxy:
    """Preserve Slack registration metadata while replacing only its factory."""

    def __init__(self, ctx: Any):
        self._ctx = ctx

    def register_platform(self, *args, **kwargs):
        base_factory = kwargs.get("adapter_factory")
        if base_factory is not _adapter._build_adapter:
            raise RuntimeError(
                "Slack native-stop registration expected the canonical "
                "Slack adapter factory"
            )
        kwargs["adapter_factory"] = _build_adapter
        return self._ctx.register_platform(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._ctx, name)


def register(ctx) -> None:
    """Plugin entry point with the native-stop-capable runtime adapter."""
    _adapter.register(_PlatformRegistrationProxy(ctx))
