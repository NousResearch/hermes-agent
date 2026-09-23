"""Gateway integration for the authenticated Becky Actions/loops bridge.

The bridge itself owns the public JSON-RPC contract.  This mixin only supplies
the gateway-owned execution and lifecycle seams, keeping the profile/session
objects private to Hermes.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from typing import Any

from gateway.config import Platform
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource

logger = logging.getLogger(__name__)


class GatewayBeckyActionsMixin:
    """Start/stop Becky bridge and run its authenticated one-shot executor."""

    def _adapter_for_source(self, source: SessionSource):
        """Compatibility seam used by private-run tests and older adapters."""
        return self._delivery_adapter_for(source)

    async def _dispatch_becky_agent_reply(
        self, *, chat_id: str, thread_id: str, session_id: str, text: str,
        reply_to_message_id: str, auto_close_policy: str | None = None,
        new_topic: bool = False,
    ) -> None:
        adapter = self.adapters.get(Platform.TELEGRAM)
        handle_message = getattr(adapter, "handle_message", None)
        if not callable(handle_message):
            raise RuntimeError("Telegram agent adapter is unavailable")
        store = getattr(self, "session_store", None)
        entry = store.lookup_by_session_id(session_id) if callable(getattr(store, "lookup_by_session_id", None)) else None
        origin = getattr(entry, "origin", None)
        event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=SessionSource(
                platform=Platform.TELEGRAM,
                chat_id=str(chat_id),
                chat_type="group" if str(chat_id).startswith("-") else "dm",
                user_id=getattr(origin, "user_id", None),
                user_name=getattr(origin, "user_name", None),
                user_id_alt=getattr(origin, "user_id_alt", None),
                thread_id=str(thread_id),
            ),
            message_id=f"becky-dashboard-{__import__('uuid').uuid4().hex}",
            reply_to_message_id=str(reply_to_message_id),
            metadata={
                "becky_dashboard_reply": True,
                **({"becky_dashboard_new_topic": True} if new_topic else {}),
                **({"becky_auto_close_policy": auto_close_policy} if auto_close_policy else {}),
            },
            internal=True,
        )
        session_db = getattr(getattr(self, "_session_db", None), "_db", None)
        bind = getattr(session_db, "bind_telegram_topic", None)
        if callable(bind) and entry is not None:
            bind(
                chat_id=str(chat_id), thread_id=str(thread_id),
                user_id=getattr(origin, "user_id", None),
                session_key=getattr(entry, "session_key", ""), session_id=session_id,
            )
        await handle_message(event)

    def _register_becky_auto_close_after_delivery(
        self, *, event: MessageEvent, source: SessionSource, session_key: str,
        run_generation: int | None, agent_result: object,
    ) -> None:
        from gateway.becky_loops import (
            BECKY_AUTO_CLOSE_POLICY_SIMPLE_CALENDAR_TODOIST_SUCCESS,
            should_auto_close_becky_loop,
        )
        metadata = getattr(event, "metadata", None)
        if (
            not isinstance(metadata, dict)
            or metadata.get("becky_dashboard_new_topic") is not True
            or metadata.get("becky_auto_close_policy") != BECKY_AUTO_CLOSE_POLICY_SIMPLE_CALENDAR_TODOIST_SUCCESS
            or not isinstance(run_generation, int) or run_generation < 1
            or not should_auto_close_becky_loop(agent_result)
        ):
            return
        adapter = self.adapters.get(source.platform)
        register = getattr(adapter, "register_post_delivery_callback", None)
        if not callable(register):
            return

        async def _close_after_delivery() -> None:
            if getattr(event, "_hermes_delivery_succeeded", None) is not True:
                return
            current = getattr(self, "_is_session_run_current", None)
            if callable(current) and not current(session_key, run_generation):
                return
            pending = getattr(adapter, "_pending_messages", None)
            if isinstance(pending, dict) and session_key in pending:
                return
            await self._maybe_auto_close_becky_topic(source=source, agent_result=agent_result)

        try:
            register(session_key, _close_after_delivery, generation=run_generation)
        except Exception:
            logger.warning("Unable to register Becky auto-close callback", exc_info=True)

    async def _maybe_auto_close_becky_topic(self, *, source: SessionSource, agent_result: object) -> None:
        from gateway.becky_loops import should_auto_close_becky_loop
        if not should_auto_close_becky_loop(agent_result):
            return
        try:
            await self._handle_becky_close_command(
                chat_id=str(source.chat_id), thread_id=str(source.thread_id or ""),
                user_id=str(source.user_id or ""), message_id="",
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Becky automatic topic close failed", exc_info=True)

    async def _handle_becky_close_command(
        self, *, chat_id: str, thread_id: str, user_id: str, message_id: str,
    ) -> str:
        del user_id, message_id
        config = getattr(self, "_becky_loops_config", None)
        if config is None or str(chat_id) != str(getattr(config, "chat_id", "")):
            return "This topic is not managed by Becky."
        if not thread_id or thread_id == "1" or thread_id in getattr(config, "managed_topic_ids", frozenset()):
            return "This is a Hermes system topic and cannot be closed as a loop."
        controller = getattr(self, "_becky_loops_topic_controller", None)
        if controller is None or not getattr(controller, "is_connected", False) or not getattr(controller, "supports_close", False):
            return "Topic close is unavailable."
        try:
            closed_at = await controller.close_topic(chat_id=str(chat_id), thread_id=str(thread_id))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            if getattr(error, "code", None) != "topic_already_closed":
                return "Topic close is unavailable."
            closed_at = datetime.now(UTC)
        if closed_at.tzinfo is None or closed_at.utcoffset() is None:
            closed_at = closed_at.replace(tzinfo=UTC)
        session_ids: list[str] = []
        session_db = getattr(self, "_session_db", None)
        if session_db is not None:
            try:
                binding = await session_db.get_telegram_topic_binding(chat_id=str(chat_id), thread_id=str(thread_id))
                bound = str((binding or {}).get("session_id") or "").strip()
                if bound:
                    session_ids.append(bound)
                listing = getattr(session_db, "list_sessions_rich", None)
                if callable(listing):
                    offset = 0
                    while offset < 10_000:
                        page = await listing(
                            source="telegram", include_children=False, include_archived=False,
                            project_compression_tips=True, order_by_last_active=True,
                            limit=200, offset=offset,
                        )
                        page = page or []
                        for row in page:
                            if (
                                str(row.get("chat_id") or "") == str(chat_id)
                                and str(row.get("thread_id") or "") == str(thread_id)
                                and row.get("ended_at") is None
                            ):
                                ident = str(row.get("id") or "").strip()
                                if ident and ident not in session_ids:
                                    session_ids.append(ident)
                        if len(page) < 200:
                            break
                        offset += len(page)
                for ident in session_ids:
                    await session_db.end_session(ident, "telegram_topic_closed")
            except Exception:
                logger.warning("Telegram /close session finalization failed", exc_info=True)
        from gateway.becky_loops import source_ref_for
        state = await self._notify_becky_topic_closed(
            source_ref=source_ref_for(chat_id=str(chat_id), thread_id=str(thread_id)),
            closed_at=closed_at, control_method=str(getattr(controller, "method", "unavailable")),
        )
        return "Topic closed and archived." if state == "archived" else "Topic closed. Becky archive is pending; refresh the dashboard shortly."

    async def _notify_becky_topic_closed(self, *, source_ref: str, closed_at: datetime, control_method: str) -> str:
        config = getattr(self, "_becky_loops_config", None)
        token = str(getattr(config, "token", "") or "").strip()
        from gateway.run import _validated_becky_dashboard_url
        import os
        url = _validated_becky_dashboard_url(os.getenv("HERMES_BECKY_DASHBOARD_URL", "http://127.0.0.1:8787"))
        if not token or url is None:
            return "pending"
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
                response = await client.post(
                    url + "/api/internal/telegram-topic-close",
                    headers={"X-Hermes-Loops-Token": token},
                    json={"source_ref": source_ref, "closed_at": closed_at.isoformat(), "control_method": control_method},
                )
            payload = response.json() if response.status_code in {200, 202} else None
            return "archived" if isinstance(payload, dict) and payload.get("state") == "archived" else "pending"
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Becky Telegram close archive handoff failed", exc_info=True)
            return "pending"

    async def _execute_becky_one_shot(
        self, *, title: str, text: str, idempotency_key, note_default: str,
        policy_version: str,
    ):
        from agent.action_mutations import REVIEWED_MUTATIONS, reviewed_mutation_spec
        from gateway.becky_actions import OneShotResult
        from model_tools import becky_one_shot_dispatch_scope

        del title, policy_version
        normalized = " ".join(str(text).replace("\r\n", "\n").replace("\r", "\n").split())
        boilerplate = (
            "Convert this request into JSON with exactly two keys: title: a short topic name, 1–6 words "
            "message: the complete plain-text request"
        )
        if normalized == boilerplate or re.fullmatch(rf"[^:]{1,128} via [^:]{1,128}: {re.escape(boilerplate)}", normalized):
            return OneShotResult(schema_version="1", disposition="ignored", event=None)
        config = getattr(self, "_becky_loops_config", None)
        chat_id = str(getattr(config, "chat_id", "") or "")
        if not chat_id:
            return OneShotResult(schema_version="1", disposition="needs_loop", event=None)
        try:
            if self._get_proxy_url():
                return OneShotResult(schema_version="1", disposition="needs_loop", event=None)
        except Exception:
            return OneShotResult(schema_version="1", disposition="needs_loop", event=None)
        source = SessionSource(
            platform=Platform.TELEGRAM, chat_id=chat_id, chat_type="dm",
            user_id="becky-shortcut", user_name="Becky Shortcut",
            profile=getattr(self, "_becky_profile_name", None),
        )
        policy = (
            "This is a private Becky Shortcut one-shot. Execute immediately only when the request is exactly one "
            "unambiguous creation of one Google Calendar event, one Todoist task, or one note in the requested note "
            "destination. Use exactly one reviewed create tool call and no other state-changing call. Do not use "
            "terminal, shell, search, research, planning, tool search, or arbitrary commands. If the request is "
            "ambiguous, conversational, asks for more than one action, or is not one of those three creates, do not "
            f"call any tool. Return a short explanation instead. The profile's default note destination is {note_default}."
        )
        try:
            allowed = {name for name, spec in REVIEWED_MUTATIONS.items() if spec.one_shot}
            with becky_one_shot_dispatch_scope(allowed):
                result = await self._run_agent(
                    message=str(text), context_prompt=policy, history=[], source=source,
                    session_id=f"becky-one-shot-{idempotency_key}",
                    session_key=f"becky-one-shot:{idempotency_key}", message_type="text",
                    private_run=True,
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Becky one-shot agent execution failed", exc_info=True)
            return OneShotResult(schema_version="1", disposition="needs_loop", event=None)

        journal = getattr(self, "_becky_action_journal", None)
        if journal is None:
            from agent.action_mutations import get_action_journal
            journal = get_action_journal()
        mutation = journal.get(idempotency_key)
        if mutation is not None and mutation.action_type.value in {"calendar", "todoist", "note"} and mutation.operation in {"create_event", "create_task", "create_note"}:
            return OneShotResult(schema_version="1", disposition=mutation.status.value, event=mutation)
        events = result.get("turn_tool_events") if isinstance(result, dict) else None
        if not isinstance(events, list) or len(events) != 1 or not isinstance(events[0], dict):
            return OneShotResult(schema_version="1", disposition="needs_loop", event=None)
        event = events[0]
        requested_name, tool_name = event.get("requested_name"), event.get("name")
        if not isinstance(requested_name, str) or requested_name != tool_name or event.get("via_tool_search") is True:
            return OneShotResult(schema_version="1", disposition="needs_loop", event=None)
        spec = reviewed_mutation_spec(tool_name)
        mutation = journal.get(idempotency_key)
        if spec is None or not spec.one_shot or mutation is None:
            return OneShotResult(schema_version="1", disposition="needs_loop", event=None)
        return OneShotResult(schema_version="1", disposition=mutation.status.value, event=mutation)

    async def _start_becky_loops_bridge(self) -> None:
        try:
            from agent.action_mutations import get_action_journal
            from gateway.becky_loops import (
                TelegramTopicController, TelegramTopicSender, load_becky_loops_config,
                start_becky_loops_bridge,
            )
            from gateway.telegram_mtproto import MTProtoPrivateTopicController
            config = load_becky_loops_config()
            self._becky_loops_config = config
            self._becky_action_journal = get_action_journal()
            self._becky_profile_name = getattr(self, "_active_profile_name", lambda: None)()
            db = getattr(getattr(self, "_session_db", None), "_db", None)
            if config is None or db is None:
                return
            adapter = self.adapters.get(Platform.TELEGRAM)
            setter = getattr(adapter, "set_becky_close_command_handler", None)
            if callable(setter):
                setter(self._handle_becky_close_command)
            topic_sender = TelegramTopicSender(adapter) if config.topic_reply == "bot_api_private_topic" and adapter is not None else None
            topic_controller = TelegramTopicController(adapter) if config.topic_control == "bot_api_private_topic" and adapter is not None else None
            mtproto_controller = None
            if config.topic_control == "mtproto_private_topic":
                mtproto_controller = MTProtoPrivateTopicController.from_environment(chat_id=config.chat_id)
                if mtproto_controller is not None:
                    await mtproto_controller.start()
                    if getattr(mtproto_controller, "is_connected", False) and getattr(mtproto_controller, "supports_close", False):
                        topic_controller = mtproto_controller
                    else:
                        await mtproto_controller.stop()
                        mtproto_controller = None
            self._becky_loops_topic_controller = topic_controller
            bridge = await start_becky_loops_bridge(
                config=config, db=db, session_store=getattr(self, "session_store", None),
                topic_sender=topic_sender, topic_controller=topic_controller,
                action_journal=self._becky_action_journal,
                one_shot_executor=self._execute_becky_one_shot,
                agent_dispatcher=self._dispatch_becky_agent_reply if topic_sender is not None else None,
            )
            should_abort = getattr(self, "_startup_should_abort", None)
            try:
                aborted = bool(should_abort()) if callable(should_abort) else False
            except AttributeError:
                aborted = False
            if aborted:
                from gateway.becky_loops import stop_becky_loops_bridge
                await stop_becky_loops_bridge(bridge)
                if mtproto_controller is not None:
                    await mtproto_controller.stop()
                self._becky_loops_topic_controller = None
                return
            self._becky_loops_bridge = bridge
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.error("Becky loop bridge startup failed", exc_info=True)

    async def _stop_becky_loops_bridge(self) -> None:
        bridge, self._becky_loops_bridge = getattr(self, "_becky_loops_bridge", None), None
        controller, self._becky_loops_topic_controller = getattr(self, "_becky_loops_topic_controller", None), None
        try:
            from gateway.becky_loops import stop_becky_loops_bridge
            await stop_becky_loops_bridge(bridge)
        except asyncio.CancelledError:
            if controller is not None:
                try:
                    await asyncio.shield(controller.stop())
                except BaseException:
                    pass
            raise
        except Exception:
            logger.debug("Becky loop bridge shutdown failed", exc_info=True)
        if controller is not None:
            try:
                await controller.stop()
            except Exception:
                logger.debug("Becky topic controller shutdown failed", exc_info=True)
