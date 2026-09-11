"""Wisdom integration with gateway command and post-delivery lifecycles."""

import asyncio
import logging
from typing import Any

from gateway.platforms.event import MessageEvent

logger = logging.getLogger("gateway.run")


class GatewayWisdomMixin:
    """Keep Wisdom's service and delivery hooks out of the runner facade."""

    async def _handle_wisdom_command(self, event: MessageEvent):
        """Dispatch `/wisdom` through the active profile's shared service."""
        source = event.source
        adapter = self._adapter_for_source(source)
        raw_args = event.get_command_args()
        if event.get_command() == "collective-wisdom-install":
            raw_args = f"install {raw_args}".strip()
        rich_handler = getattr(adapter, "send_wisdom_command", None)
        if callable(rich_handler):
            await rich_handler(raw_args, source=source)
            return ""

        from gateway.wisdom_command import (
            WisdomCommandContext,
            WisdomCommandController,
        )
        from hermes_wisdom.service import WisdomService

        from gateway.run import _profile_runtime_scope

        profile_home = self._resolve_profile_home_for_source(source)

        def command_action():
            with _profile_runtime_scope(profile_home):
                service = WisdomService()
                context = WisdomCommandContext(
                    user_id=str(source.user_id or ""),
                    chat_id=str(source.chat_id),
                    profile=getattr(source, "profile", None),
                    organization_id=service.store.active_org_id(),
                    is_group=str(source.chat_type or "").lower()
                    in {"group", "supergroup", "channel", "forum"},
                )
                return WisdomCommandController().execute(
                    raw_args, service, context
                )
        try:
            view = await asyncio.to_thread(command_action)
        except Exception as exc:
            logger.warning(
                "Collective Wisdom command failed (%s)", type(exc).__name__
            )
            from gateway.wisdom_command import command_error_text

            return f"Collective Wisdom could not continue: {command_error_text(exc)}"
        return view.to_text()

    async def _defer_wisdom_candidate_notice_after_delivery(
        self, source: Any, session_id: str, *, user_activity: bool = True
    ) -> None:
        """Surface local qualification after the originating client reply."""
        adapter = self._adapter_for_source(source)
        sender = getattr(adapter, "send_wisdom_candidate_notifications", None)
        if adapter is None or not callable(sender):
            return

        try:
            metadata = self._thread_metadata_for_source(source)
        except Exception:
            metadata = None

        async def _deliver() -> None:
            try:
                if user_activity:
                    from gateway.wisdom_mediation import schedule

                    if await schedule(self, adapter, source, session_id):
                        return
                await sender(
                    source.chat_id,
                    session_id,
                    metadata=metadata,
                )
            except Exception as exc:
                logger.warning(
                    "Wisdom candidate %s delivery failed: %s",
                    getattr(source, "platform", "platform"),
                    exc,
                    exc_info=True,
                )

        try:
            session_key = self._session_key_for_source(source)
        except Exception:
            session_key = None
        if session_key and hasattr(adapter, "register_post_delivery_callback"):
            try:
                generation = None
                active = getattr(adapter, "_active_sessions", {}).get(session_key)
                if active is not None:
                    generation = getattr(active, "_hermes_run_generation", None)
                adapter.register_post_delivery_callback(
                    session_key,
                    _deliver,
                    generation=generation,
                )
                return
            except Exception as exc:
                logger.debug(
                    "Wisdom candidate post-delivery callback registration failed: %s",
                    exc,
                )
        await _deliver()

    async def _continue_wisdom_start(self, event, source) -> str:
        denied = self._check_slash_access(source, "wisdom")
        if denied is not None:
            return denied
        adapter = self._adapter_for_source(source)
        continuation = getattr(adapter, "send_wisdom_continuation", None)
        if callable(continuation):
            try:
                token = event.get_command_args().strip().removeprefix("wisdom_")
                await continuation(token, source=source)
            except (PermissionError, ValueError) as exc:
                return str(exc)
            except Exception as exc:
                logger.warning("Collective Wisdom DM continuation failed (%s)", type(exc).__name__)
                return (
                    "Collective Wisdom could not continue that request. "
                    "Run /wisdom in this chat instead."
                )
        return ""


def enqueue_weekly_review() -> None:
    """Queue local evidence; only the active private session assesses and delivers."""
    from hermes_wisdom.service import WisdomService
    from hermes_wisdom.weekly_queue import enqueue_weekly_review as enqueue

    enqueue(WisdomService())


class WisdomCardRefresh:
    """Keep housekeeping from starting overlapping publication-card refreshes."""

    def __init__(self, adapters, loop):
        self.adapters = adapters
        self.loop = loop
        self.pending = None

    def tick(self) -> None:
        if not self.adapters or self.loop is None:
            return
        if self.pending is not None:
            if not self.pending.done():
                return
            try:
                self.pending.result()
            except Exception:
                logger.debug("Wisdom publication-card refresh failed", exc_info=True)
        from gateway.wisdom_publication_cards import refresh

        self.pending = asyncio.run_coroutine_threadsafe(refresh(self.adapters), self.loop)
