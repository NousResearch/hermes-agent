"""Session override notices and held-message lifecycle."""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Optional
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")
_STALE_OVERRIDE_LAST_COMPLETED_KEY = "stale_override_last_completed_at"
_STALE_OVERRIDE_PROMPT_TIMEOUT_SECONDS = 120.0

class GatewayStaleOverrideMixin:
    async def _mark_stale_override_turn_completed(
        self, session_key: str, *, is_internal: bool
    ) -> None:
        """Persist the completion clock used by stale-override notices."""
        if is_internal or not session_key:
            return
        try:
            await self.async_session_store.set_session_metadata(
                session_key, _STALE_OVERRIDE_LAST_COMPLETED_KEY, time.time()
            )
        except Exception:
            logger.debug(
                "Failed to persist stale-override completion clock for %s",
                session_key,
                exc_info=True,
            )

    async def _defer_stale_override_turn_completed(
        self,
        session_key: str,
        *,
        source: SessionSource,
        run_generation: int,
        is_internal: bool,
    ) -> None:
        """Start the idle clock after the delivery lifecycle, not generation."""
        if is_internal or not session_key:
            return

        notice_config = getattr(self.config, "stale_override_notice", None)
        if notice_config is None or notice_config.mode == "off":
            return
        try:
            home_channel = self.config.get_home_channel(source.platform)
        except Exception:
            home_channel = None
        from gateway.stale_override_notice import source_matches_channels

        if not source_matches_channels(
            source,
            notice_config.channels,
            home_channel=home_channel,
        ):
            return

        adapter = self._adapter_for_source(source)
        register = getattr(adapter, "register_post_delivery_callback", None)
        if not callable(register):
            await self._mark_stale_override_turn_completed(
                session_key, is_internal=False
            )
            return

        async def _after_delivery() -> None:
            await self._mark_stale_override_turn_completed(
                session_key, is_internal=False
            )

        register(
            session_key,
            _after_delivery,
            generation=run_generation,
        )

    def _stale_override_decision(
        self,
        *,
        source: SessionSource,
        session_key: str,
        notice_config,
    ):
        """Resolve which explicit session overrides differ from live defaults."""
        from gateway.stale_override_notice import (
            OverrideNoticeDecision,
            reasoning_effort,
            reasoning_matches_policy,
            route_label,
            routes_differ,
        )

        self._rehydrate_session_model_override(session_key)
        state = self._peek_session_state(session_key)
        if state is None:
            return OverrideNoticeDecision()

        model_override = state.conversation.model_override
        reasoning_override = state.conversation.reasoning_override

        default_model, default_runtime = self._resolve_session_agent_runtime(
            source=source,
            session_key=session_key,
            include_session_override=False,
        )
        current_model, current_runtime = self._resolve_session_agent_runtime(
            source=source,
            session_key=session_key,
        )
        default_provider = default_runtime.get("provider")
        current_provider = current_runtime.get("provider")
        model_stale = bool(
            notice_config.model != "off"
            and model_override is not None
            and routes_differ(
                current_model,
                current_provider,
                default_model,
                default_provider,
            )
        )

        default_reasoning = self._load_reasoning_config(current_model)
        reasoning_stale = reasoning_matches_policy(
            notice_config.reasoning,
            reasoning_override,
            default_reasoning,
        )

        return OverrideNoticeDecision(
            model_stale=model_stale,
            reasoning_stale=reasoning_stale,
            current_route=route_label(current_model, current_provider),
            default_route=route_label(default_model, default_provider),
            current_reasoning=reasoning_effort(reasoning_override),
            default_reasoning=reasoning_effort(default_reasoning),
        )

    async def _clear_stale_override_selection(
        self, session_key: str, *, model: bool, reasoning: bool
    ) -> None:
        """Clear selected session overrides through their canonical stores."""
        if model:
            conversation = self._session_state(session_key).conversation
            conversation.model_override = None
            conversation.one_turn_restore = None
            pending_notes = getattr(self, "_pending_model_notes", None)
            if isinstance(pending_notes, dict):
                pending_notes.pop(session_key, None)
            try:
                await self.async_session_store.set_model_override(session_key, None)
            except Exception:
                logger.debug(
                    "Failed to persist stale-override model reset for %s",
                    session_key,
                    exc_info=True,
                )
        if reasoning:
            self._set_session_reasoning_override(session_key, None)
        if model or reasoning:
            self._evict_cached_agent(session_key)

    def _schedule_stale_override_prompt_expiry(
        self, session_key: str, token: object
    ) -> None:
        """Expire a held message without ever submitting it implicitly."""

        async def _expire() -> None:
            await asyncio.sleep(_STALE_OVERRIDE_PROMPT_TIMEOUT_SECONDS)
            pending = getattr(self, "_stale_override_pending", {}).get(session_key)
            if pending is not None and pending.get("token") is token:
                self._stale_override_pending.pop(session_key, None)
                logger.info(
                    "Stale-override prompt expired; pending message was not sent "
                    "(session=%s)",
                    session_key,
                )

        task = asyncio.create_task(_expire())
        background = getattr(self, "_background_tasks", None)
        if background is not None:
            background.add(task)
            task.add_done_callback(background.discard)

    async def _maybe_handle_stale_override_notice(
        self,
        event: MessageEvent,
        session_key: str,
    ) -> tuple[bool, Optional[str]]:
        """Notify or hold the first ordinary message after an idle override."""
        metadata = getattr(event, "metadata", None)
        if isinstance(metadata, dict) and metadata.pop(
            "_stale_override_notice_bypass", False
        ):
            return False, None
        if getattr(event, "internal", False) or event.is_command():
            return False, None

        config = getattr(getattr(self, "config", None), "stale_override_notice", None)
        if config is None or config.mode == "off":
            return False, None

        from gateway.stale_override_notice import source_matches_channels

        home = None
        try:
            home = self.config.get_home_channel(event.source.platform)
        except Exception:
            pass
        if not source_matches_channels(
            event.source, config.channels, home_channel=home
        ):
            return False, None

        try:
            completed_at = await self.async_session_store.get_session_metadata(
                session_key, _STALE_OVERRIDE_LAST_COMPLETED_KEY, None
            )
            idle_seconds = max(0.0, time.time() - float(completed_at))
        except (TypeError, ValueError):
            return False, None
        except Exception:
            logger.debug("Failed to read stale-override completion clock", exc_info=True)
            return False, None
        if idle_seconds < config.idle_minutes * 60:
            return False, None

        try:
            session_entry = await self.async_session_store.get_or_create_session(
                event.source,
                touch_activity=False,
            )
        except Exception:
            logger.debug("Failed stale-override reset-policy preflight", exc_info=True)
            return False, None
        if getattr(session_entry, "was_auto_reset", False):
            logger.info(
                "Skipping stale-override prompt because session auto-reset is pending "
                "(session=%s)",
                session_key,
            )
            return False, None

        decision = self._stale_override_decision(
            source=event.source,
            session_key=session_key,
            notice_config=config,
        )
        if not decision.triggered:
            return False, None

        adapter = self._adapter_for_source(event.source)
        if adapter is None:
            return False, None
        notice_text = decision.message(idle_seconds / 60.0, held=config.mode == "confirm")
        send_metadata = self._thread_metadata_for_source(
            event.source, self._reply_anchor_for_event(event)
        )

        if config.mode == "info_only":
            try:
                await adapter.send(
                    event.source.chat_id,
                    f"ℹ️ {notice_text}",
                    metadata=send_metadata,
                )
            except Exception:
                logger.warning("Failed to send stale-override info notice", exc_info=True)
            return False, None

        has_picker = getattr(type(adapter), "send_choice_picker", None) is not None
        if not has_picker:
            logger.warning(
                "stale_override_notice mode=confirm requires send_choice_picker; "
                "allowing message on platform=%s",
                getattr(getattr(event.source, "platform", None), "value", "unknown"),
            )
            return False, None

        pending_map = getattr(self, "_stale_override_pending", None)
        if pending_map is None:
            pending_map = {}
            self._stale_override_pending = pending_map
        if session_key in pending_map:
            return True, (
                "⚠️ A previous message is still waiting on the override prompt. "
                "Resolve that prompt first; this newer message was not sent."
            )

        token = object()
        pending_map[session_key] = {"token": token, "event": event}
        send_metadata = dict(send_metadata or {})
        send_metadata["choice_layout"] = "vertical"
        send_metadata["choice_timeout_seconds"] = _STALE_OVERRIDE_PROMPT_TIMEOUT_SECONDS
        source = event.source
        is_thread = bool(source.thread_id or source.prospective_thread_id)
        owner_required = (
            source.chat_type == "dm"
            or (
                is_thread
                and getattr(self.config, "thread_sessions_per_user", False)
            )
            or (
                source.chat_type != "dm"
                and not is_thread
                and getattr(self.config, "group_sessions_per_user", True)
            )
        )
        if owner_required:
            send_metadata["requester_user_id"] = str(
                source.user_id_alt or source.user_id or ""
            )

        async def _on_choice_selected(_chat_id: str, value: str) -> str:
            pending = getattr(self, "_stale_override_pending", {}).get(session_key)
            if pending is None or pending.get("token") is not token:
                return "Selection expired — the original message was not sent."
            if value not in {
                "continue",
                "default_model",
                "default_reasoning",
                "defaults",
            }:
                return "Invalid selection — the original message was not sent."

            reset_model = value in {"default_model", "defaults"}
            reset_reasoning = value in {"default_reasoning", "defaults"}
            logger.info(
                "Stale-override selection accepted session=%s choice=%s "
                "reset_model=%s reset_reasoning=%s",
                session_key,
                value,
                reset_model,
                reset_reasoning,
            )
            self._stale_override_pending.pop(session_key, None)
            await self._clear_stale_override_selection(
                session_key,
                model=reset_model,
                reasoning=reset_reasoning,
            )
            held_event = pending["event"]
            if not isinstance(getattr(held_event, "metadata", None), dict):
                held_event.metadata = {}
            held_event.metadata["_stale_override_notice_bypass"] = True
            resume_adapter = self._adapter_for_source(held_event.source) or adapter
            await resume_adapter.handle_message(held_event)
            logger.info(
                "Stale-override held message re-dispatched session=%s choice=%s",
                session_key,
                value,
            )
            if value == "continue":
                return "Keeping the current overrides. Resuming your message..."
            return "Override updated. Resuming your message..."

        try:
            result = await adapter.send_choice_picker(
                chat_id=event.source.chat_id,
                title=f"⚠️ {notice_text}",
                choices=decision.choices(),
                session_key=session_key,
                on_choice_selected=_on_choice_selected,
                metadata=send_metadata,
            )
        except Exception:
            logger.warning("Failed to send stale-override confirmation", exc_info=True)
            result = None
        if not bool(getattr(result, "success", False)):
            pending_map.pop(session_key, None)
            return False, None

        logger.info(
            "Stale-override prompt shown session=%s idle_seconds=%.1f "
            "model_stale=%s reasoning_stale=%s",
            session_key,
            idle_seconds,
            decision.model_stale,
            decision.reasoning_stale,
        )
        self._schedule_stale_override_prompt_expiry(session_key, token)
        return True, None
