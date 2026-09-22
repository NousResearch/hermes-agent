"""Exact-session typed plugin event admission and correlated terminal receipts."""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from collections import OrderedDict
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session_identity import replace_source

logger = logging.getLogger("gateway.run")

class GatewayPluginEventsMixin:
    def _schedule_plugin_system_event(self, *, content, system_event, receipt):
        """Schedule one typed event and return its terminal cross-thread receipt."""
        from gateway.run import safe_schedule_threadsafe
        from gateway.internal_events import (
            GatewaySystemEvent,
            resolve_gateway_event_receipt,
        )

        if not isinstance(system_event, GatewaySystemEvent):
            return None
        if not isinstance(receipt, concurrent.futures.Future):
            return None

        with self.__dict__.setdefault("_gateway_event_receipt_lock", threading.Lock()):
            receipt_cache = getattr(self, "_gateway_system_event_receipts", None)
            if receipt_cache is None:
                receipt_cache = OrderedDict()
                self._gateway_system_event_receipts = receipt_cache
            dedup_key = (system_event.plugin_id, system_event.event_id)
            existing = receipt_cache.get(dedup_key)
            if isinstance(existing, concurrent.futures.Future):
                retryable = existing.done() and not existing.cancelled() and existing.result().get("status") in {"busy", "stopping"}
                if not retryable:
                    receipt_cache.move_to_end(dedup_key)
                    return existing
                receipt_cache.pop(dedup_key)

            # Retain terminal receipts so a same-process retry cannot repeat writes.
            # Evict only completed oldest entries; pending receipts remain owned by
            # the live adapter turn.
            while len(receipt_cache) >= 1024:
                oldest_key, oldest = next(iter(receipt_cache.items()))
                if not oldest.done():
                    resolve_gateway_event_receipt(
                        receipt, "busy", event=system_event
                    )
                    return receipt
                receipt_cache.pop(oldest_key, None)
            receipt_cache[dedup_key] = receipt

        loop = getattr(self, "_gateway_loop", None)
        if (
            not getattr(self, "_running", False)
            or getattr(self, "_draining", False)
            or getattr(self, "_external_drain_active", False)
            or loop is None
            or loop.is_closed()
        ):
            resolve_gateway_event_receipt(receipt, "stopping", event=system_event)
            return receipt

        coro = self._dispatch_plugin_system_event(
            content=content,
            system_event=system_event,
            receipt=receipt,
        )
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if current_loop is loop:
            try:
                dispatch_future = loop.create_task(coro)
            except Exception:
                coro.close()
                resolve_gateway_event_receipt(
                    receipt, "agent_error", event=system_event
                )
                return receipt
            self._background_tasks.add(dispatch_future)
            dispatch_future.add_done_callback(self._background_tasks.discard)
        else:
            dispatch_future = safe_schedule_threadsafe(
                coro,
                loop,
                logger=logger,
                log_message="Plugin system event scheduling failed",
                log_level=logging.WARNING,
            )
            if dispatch_future is None:
                resolve_gateway_event_receipt(
                    receipt, "agent_error", event=system_event
                )
                return receipt

        def _finish_failed_dispatch(completed) -> None:
            try:
                completed.result()
            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                resolve_gateway_event_receipt(
                    receipt, "cancelled", event=system_event
                )
            except Exception:
                logger.warning(
                    "Plugin system event dispatch failed: plugin=%s event=%s",
                    system_event.plugin_id,
                    system_event.event_id,
                    exc_info=True,
                )
                resolve_gateway_event_receipt(
                    receipt, "agent_error", event=system_event
                )

        dispatch_future.add_done_callback(_finish_failed_dispatch)
        return receipt

    async def _gateway_system_event_target(
        self, system_event, *, check_eligibility: bool = True
    ):
        """Resolve and revalidate the exact open session for a typed event."""
        if (not getattr(self, "_running", False) or getattr(self, "_draining", False)
                or getattr(self, "_external_drain_active", False)):
            return "stopping", None, None

        from gateway.internal_events import (
            GatewaySystemEvent,
            GatewayExpectedRoute,
            gateway_system_event_is_eligible,
        )
        from hermes_cli.plugins import get_plugin_manager

        if not isinstance(system_event, GatewaySystemEvent):
            return "agent_error", None, None
        if check_eligibility and not gateway_system_event_is_eligible(system_event):
            return "unauthorized", None, None
        manager = get_plugin_manager()
        if not manager.gateway_injection_allowed(system_event.plugin_id):
            return "unauthorized", None, None

        entry = await self.async_session_store.lookup_by_session_key(
            system_event.session_key
        )
        if entry is None or entry.origin is None:
            return "route_mismatch", None, None
        if entry.session_id != system_event.expected_session_id:
            return "session_mismatch", None, None
        if entry.suspended:
            return "unauthorized", None, None
        if entry.resume_pending:
            return "busy", None, None

        recovery_state = (
            await self.async_session_store.typed_event_recovery_owner_state(
                system_event.session_key,
                system_event.expected_session_id,
            )
        )
        if recovery_state == "conflict":
            return "busy", None, None

        source = replace_source(self._restored_source(entry))
        if self._session_key_for_source(source) != system_event.session_key:
            return "route_mismatch", None, None
        source_profile = str(getattr(source, "profile", None) or "").strip()
        if not source_profile:
            profile_from_key = getattr(
                self.session_store, "_profile_from_session_key", None
            )
            if callable(profile_from_key):
                source_profile = str(
                    profile_from_key(system_event.session_key) or ""
                ).strip()
        if not source_profile:
            source_profile = "default"
        platform = (
            source.platform.value
            if hasattr(source.platform, "value")
            else str(source.platform)
        )
        actual_route = GatewayExpectedRoute(
            profile_name=source_profile,
            platform=platform,
            user_id=str(source.user_id or ""),
            chat_id=str(source.chat_id or ""),
            topic_id=str(getattr(source, "thread_id", None) or ""),
        )
        if actual_route != system_event.expected_route:
            return "route_mismatch", None, None

        raw_db = getattr(getattr(self, "_session_db", None), "_db", None)
        if raw_db is None:
            raw_db = getattr(self, "_session_db", None)
        get_session = getattr(raw_db, "get_session", None)
        if not callable(get_session):
            return "session_mismatch", None, None
        try:
            row = await asyncio.to_thread(
                get_session, system_event.expected_session_id
            )
        except Exception:
            return "session_mismatch", None, None
        if not isinstance(row, dict) or row.get("ended_at") is not None:
            return "session_mismatch", None, None

        try:
            authorized = self._is_user_authorized_for_source(
                source, allow_adapter_delegation=False
            )
        except Exception:
            authorized = False
        if not authorized:
            return "unauthorized", None, None

        adapter = self._delivery_adapter_for(source)
        if adapter is None:
            return "route_mismatch", None, None
        return None, entry, source

    async def _admit_gateway_system_event_turn(self, event, session_entry):
        """Revalidate a plugin-owned event at the model admission boundary."""
        from gateway.internal_events import (
            admit_message_event_receipt,
            resolve_message_event_receipt,
        )

        system_event = getattr(event, "gateway_system_event", None)
        status, current_entry, current_source = (
            await self._gateway_system_event_target(system_event)
        )
        if status is not None:
            resolve_message_event_receipt(event, status)
            return None
        if current_entry.session_id != session_entry.session_id:
            resolve_message_event_receipt(event, "session_mismatch")
            return None
        if not admit_message_event_receipt(event):
            return None
        return current_source

    async def _dispatch_plugin_system_event(
        self, *, content, system_event, receipt
    ) -> None:
        """Route a typed event through the existing adapter/session serializer."""
        from gateway.internal_events import resolve_gateway_event_receipt

        status, entry, source = await self._gateway_system_event_target(system_event)
        if status is not None:
            resolve_gateway_event_receipt(receipt, status, event=system_event)
            return
        if (not getattr(self, "_running", False) or getattr(self, "_draining", False)
                or getattr(self, "_external_drain_active", False)):
            resolve_gateway_event_receipt(
                receipt, "stopping", event=system_event
            )
            return

        event = MessageEvent(
            text=content,
            message_type=MessageType.TEXT,
            source=source,
            internal=True,
            allow_gateway_control=False,
            metadata={
                "hermes_plugin_id": system_event.plugin_id,
                "hermes_plugin_injection": True,
                "gateway_session_key": system_event.session_key,
                "gateway_session_id": system_event.expected_session_id,
                "gateway_session_strict": True,
            },
            gateway_system_event=system_event,
            gateway_event_receipt=receipt,
        )
        adapter = self._delivery_adapter_for(source)
        if adapter is None:
            resolve_gateway_event_receipt(
                receipt, "route_mismatch", event=system_event
            )
            return
        await adapter.handle_message(event)


    async def _prepare_gateway_system_event(self, event, source, entry, key, quick_key, generation):
        """Claim exact physical ownership before reading history; the plugin owns replay."""
        from gateway.internal_events import resolve_message_event_receipt
        from gateway.session import build_session_context
        registry = getattr(self, "_turn_leases", None)
        if registry is None:
            resolve_message_event_receipt(event, "agent_error")
            return None, None
        token = await registry.try_acquire(entry.session_id, owner_key=quick_key, generation=generation)
        if token is None:
            resolve_message_event_receipt(event, "busy")
            return None, None
        lease_state = self._session_state(quick_key).turn
        lease_state.lease_token, lease_state.lease_generation = token, generation
        if not await self._mark_durable_active_turn(
            event, key, entry.session_id, turn_lease_acquired=True,
        ):
            return None, None
        history = await self.async_session_store.load_transcript(entry.session_id)
        context = build_session_context(source, self.config, entry)
        tokens = self._set_session_env(context)
        try:
            prompt = self._pinned_session_context_prompt(context, False, key)
            self._bind_adapter_run_generation(self._delivery_adapter_for(source), key, generation)
            return self._PreparedTurn(
                history, prompt, event.text, None, None, "internal_notification",
                entry.session_id, event.gateway_system_event.receipt_id,
            ), tokens
        except BaseException:
            self._clear_session_env(tokens)
            raise
