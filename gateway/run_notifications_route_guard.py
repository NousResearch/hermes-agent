"""Async-delegation completion route guard for GatewayRunner.

Split out of ``gateway/run_notifications.py`` (itself split out of ``gateway/run.py``); bound
onto ``GatewayRunner`` via the MRO.

A pinned async-delegation completion may only re-point a chat's route when the pinned session
is genuinely that channel's own. A delegate child arms the watcher with its OWN session id, so
following that pin calls ``switch_session``: it ends the chat's real session and re-stamps the
child row as the channel's (2026-09-17 dev Discord hijack). Unknown ownership fails closed; a
pin that cannot own the route reaches the chat only after its delegate lineage names the
current owner. An unrelated pin is dropped before it can disclose output across chats.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional, cast

from gateway.session import SessionEntry, is_internal_subagent_row

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

class GatewayNotificationRouteGuardMixin:
    """Route ownership for pinned async-delegation completions."""

    async def _watch_owner_db_for_key(self, session_key: str):
        """Open the process event's owning profile DB, independent of ambient runtime scope."""
        from hermes_state import AsyncSessionDB

        owner_db = await asyncio.to_thread(self.session_store._db_for_key, session_key)
        return AsyncSessionDB(owner_db) if owner_db is not None else None

    async def _route_owner_session_db(self, session_key: str) -> Any:
        """Return the AsyncSessionDB for *session_key*, honoring test pins while keeping
        multiplexed profile routes bound to their owning profile store."""
        from gateway.run import _SESSION_DB_UNPINNED
        from gateway.session import SessionStore

        store = getattr(self, "session_store", None)
        pinned = getattr(self, "_session_db_pinned", _SESSION_DB_UNPINNED)
        if isinstance(store, SessionStore) and (
            pinned is _SESSION_DB_UNPINNED or store._named_profile_for_key(session_key) is not None
        ):
            return await self._watch_owner_db_for_key(session_key)
        return cast(Any, self._session_db)

    async def _classify_completion_target(self, parent_session_id: str, session_key: str = "") -> str:
        """Classify an async-completion target before adapter acceptance: ``"deliver"`` (spawning
        session live or compression-rotated with a live continuation; the resolver still retargets),
        ``"terminal"`` (parent gone for good — unknown / user boundary like /new; drop the durable row
        rather than falsely ack), ``"retry"`` (DB unavailable / rotation mid-flight; release the claim)."""
        try:
            session_db = (await self._route_owner_session_db(session_key)
                          if session_key else getattr(self, "_session_db", None))
            if session_db is None:
                return "retry"
            parent = await session_db.get_session(parent_session_id)
        except Exception:
            logger.debug("Async-completion pre-flight parent lookup failed for %s", parent_session_id, exc_info=True)
            return "retry"
        if parent is None:
            return "terminal"
        if not parent.get("ended_at"):
            return "deliver"
        end_reason = str(parent.get("end_reason") or "")
        if end_reason != "compression":
            # Only a USER-closed session (/new, user_exit, session_switch) is unreachable unless a
            # legacy delegate hijack caused the switch; idle/timeout ends stay routable.
            return await self._classify_non_compression_target(parent, parent_session_id, end_reason)
        try:
            tip_session_id = await session_db.get_compression_tip(parent_session_id)
            if not tip_session_id or tip_session_id == parent_session_id:
                # Rotation mid-flight: continuation not visible yet. Retry, don't drop.
                return "retry"
            tip = await session_db.get_session(tip_session_id)
        except Exception:
            logger.debug("Async-completion pre-flight tip lookup failed for %s", parent_session_id, exc_info=True)
            return "retry"
        if tip is None or tip.get("ended_at"):
            return "retry"
        return "deliver"

    async def _classify_non_compression_target(
        self, parent: Dict[str, Any], parent_session_id: str, end_reason: str,
    ) -> str:
        """Classify a non-compression ended completion target, healing legacy delegate hijacks."""
        from gateway.run import _USER_BOUNDARY_END_REASONS
        from gateway.session import SessionStore

        if end_reason == "session_switch":
            session_key = str(parent.get("session_key") or "").strip()
            store = getattr(self, "session_store", None)
            if session_key and isinstance(store, SessionStore):
                entry = await self.async_session_store.lookup_by_session_key(session_key)
                if entry is None and not getattr(store, "_routing_db_loaded", True):
                    return "retry"
                if entry is not None and entry.origin is not None:
                    try:
                        repaired = await asyncio.to_thread(
                            store._reconcile_poisoned_delegate_route,
                            session_key, entry, entry.origin,
                            quarantine_invalid=False,
                        )
                    except Exception:
                        logger.debug(
                            "Async-completion poisoned-route check failed for %s",
                            parent_session_id, exc_info=True,
                        )
                        return "retry"
                    if repaired is not None and repaired.session_id == parent_session_id:
                        return "deliver"
        return "terminal" if end_reason in _USER_BOUNDARY_END_REASONS else "deliver"

    async def _watch_event_route_verdict(self, evt: dict) -> str:
        """Return owned, drop, or retry for a queued process watch event's spawning route."""
        from gateway.run import _USER_BOUNDARY_END_REASONS
        from gateway.run_notifications import _raw_process_event_session_id

        if evt.get("type") not in {
            "watch_match", "watch_disabled", "watch_overflow_tripped", "watch_overflow_released", "heartbeat",
        } or _raw_process_event_session_id(evt):
            return "owned"
        key = str(evt.get("session_key") or "").strip()
        pin = str(evt.get("parent_session_id") or "").strip()
        if not key.startswith("agent:") or not pin:
            return "drop"  # A chat key alone cannot prove which conversation spawned the event.
        try:
            async with self._completion_event_scope(evt):
                generation = self._current_session_run_generation(key)
                entry = await self.async_session_store.lookup_by_session_key(key)
                if entry is None:
                    return "retry" if not getattr(self.session_store, "_routing_db_loaded", True) else "drop"
                session_db = await self._watch_owner_db_for_key(key)
                if session_db is None:
                    return "retry"
                row = await session_db.get_session(pin)
                if row is None:
                    return "drop"
                reason = str(row.get("end_reason") or "") if row.get("ended_at") else ""
                if reason in _USER_BOUNDARY_END_REASONS:
                    return "drop"
                if is_internal_subagent_row(row):
                    owns = await self._delegate_pin_belongs_to_route(
                        session_db, row, entry, raise_lookup_errors=True,
                    )
                elif reason == "compression":
                    owns = await self._resolve_compression_lineage_target(
                        session_db, entry, pin, raise_lookup_errors=True,
                    ) == entry.session_id
                else:
                    owns = pin == entry.session_id and row.get("session_key") == key
                if not owns:
                    return "drop"
                return "owned" if await self._unchanged_completion_route(entry, generation) else "drop"
        except Exception:
            logger.debug("Process watch route lookup failed for %s", pin, exc_info=True)
            return "retry"

    async def _watcher_message_route_owned(
        self, watcher: dict, process: Any, *, raise_lookup_errors: bool = False,
    ) -> bool:
        """Prove a direct process status still addresses its spawning conversation."""
        from gateway.run import _USER_BOUNDARY_END_REASONS
        from tools.process_registry_notifications import should_surface_notification

        if process is None:
            return False
        key = str(watcher.get("session_key") or "").strip()
        pin = str(watcher.get("parent_session_id") or getattr(process, "parent_session_id", "") or "").strip()
        if not key or not pin:
            return False
        async with self._completion_event_scope(watcher):
            generation = self._current_session_run_generation(key)
            entry = await self.async_session_store.lookup_by_session_key(key)
            if entry is None:
                if raise_lookup_errors and not getattr(self.session_store, "_routing_db_loaded", True):
                    raise RuntimeError("Process watcher routing database unavailable")
                return False
            try:
                session_db = await self._watch_owner_db_for_key(key)
                if session_db is None:
                    raise RuntimeError("Process watcher owner database unavailable")
                row = await session_db.get_session(pin)
            except Exception:
                if raise_lookup_errors:
                    raise
                logger.debug("Process watcher parent lookup failed for %s", pin, exc_info=True)
                return False
            if row is None:
                return False
            reason = str(row.get("end_reason") or "") if row.get("ended_at") else ""
            if reason in _USER_BOUNDARY_END_REASONS:
                return False
            if is_internal_subagent_row(row):
                owns = await self._delegate_pin_belongs_to_route(
                    session_db, row, entry, raise_lookup_errors=raise_lookup_errors)
            elif reason == "compression":
                owns = await self._resolve_compression_lineage_target(
                    session_db, entry, pin, raise_lookup_errors=raise_lookup_errors) == entry.session_id
            else:
                owns = pin == entry.session_id and row.get("session_key") == key
            if not owns or not await self._unchanged_completion_route(entry, generation):
                return False
            owner = {"owner_task_id": getattr(process, "owner_task_id", ""),
                     "task_id": getattr(process, "task_id", "")}
            return (str(getattr(process, "session_key", "") or "").strip() in {"", key, entry.session_id}
                    and should_surface_notification(owner))

    async def _send_watcher_message(self, platform_name: str, chat_id, thread_id, message_text: str, watcher: dict, session) -> Optional[bool]:
        """True sent, None terminally suppressed, False temporarily unavailable."""
        from gateway.run import _non_conversational_metadata
        from gateway.dead_targets import classify_dead_error
        try:
            source = await asyncio.to_thread(self._build_process_event_source, watcher)
            adapter = self._resolve_injection_adapter(platform_name, source)
            if not adapter:
                return False
            if not chat_id or not await self._watcher_message_route_owned(
                watcher, session, raise_lookup_errors=True,
            ):
                return None
            send_meta = {"thread_id": thread_id} if thread_id else None
            metadata = _non_conversational_metadata(send_meta, platform=platform_name)
            send_for_platform = getattr(adapter, "send_for_platform", None)
            if callable(send_for_platform):
                result = await send_for_platform(platform_name, chat_id, message_text, metadata=metadata)
            else:
                result = await adapter.send(chat_id, message_text, metadata=metadata)
            return self._watcher_send_result(result)
        except Exception as exc:
            logger.debug("Watcher delivery unavailable for %s", watcher.get("session_id"), exc_info=True)
            return None if classify_dead_error(str(exc)) else False

    @staticmethod
    def _watcher_send_result(result) -> Optional[bool]:
        """Retain transient failures without repeatedly sending to permanent failures."""
        from gateway.dead_targets import classify_dead_error
        from gateway.platforms.base import BasePlatformAdapter
        if getattr(result, "success", None) is not False:
            return True
        error = getattr(result, "error", "") or ""
        retryable = (getattr(result, "retryable", False)
                     or getattr(result, "retry_after", None) is not None
                     or BasePlatformAdapter._is_rate_limited_error(error)
                     or BasePlatformAdapter._is_retryable_error(error))
        return False if retryable and not classify_dead_error(error) else None

    async def _delegate_pin_belongs_to_route(
        self, session_db: Any, pinned_row: Dict[str, Any], entry: SessionEntry,
        *, raise_lookup_errors: bool = False,
    ) -> bool:
        """Prove a child pin descends from this exact chat owner before showing its output there."""
        try:
            owner = await session_db.get_session(entry.session_id)
        except Exception:
            if raise_lookup_errors:
                raise
            logger.debug("Delegate route owner lookup failed for %s", entry.session_id, exc_info=True)
            return False
        if (
            owner is None or owner.get("ended_at") or is_internal_subagent_row(owner)
            or owner.get("session_key") != entry.session_key
        ):
            return False
        seen: set[str] = set()
        row = pinned_row
        for _ in range(64):
            child_id = str(row.get("id") or "")
            if not child_id or child_id in seen:
                return False
            seen.add(child_id)
            if not is_internal_subagent_row(row):
                # A coordinator can rotate through compression after spawning this child.
                # Its old id is no longer the route, but a proven compression tip still is.
                if row.get("session_key") != entry.session_key or row.get("end_reason") != "compression":
                    return False
                try:
                    return await session_db.get_compression_tip(child_id) == entry.session_id
                except Exception:
                    if raise_lookup_errors:
                        raise
                    logger.debug("Delegate owner compression lookup failed for %s", child_id, exc_info=True)
                    return False
            parent_id = str(row.get("parent_session_id") or "")
            if not parent_id or parent_id in seen:
                return False
            if parent_id == entry.session_id:
                return True
            try:
                row = await session_db.get_session(parent_id)
            except Exception:
                if raise_lookup_errors:
                    raise
                logger.debug("Delegate lineage lookup failed for %s", parent_id, exc_info=True)
                return False
            if row is None:
                return False
        return False

    async def _resolve_compression_lineage_target(
        self, session_db: Any, session_entry: SessionEntry, pinned_session_id: str,
        *, raise_lookup_errors: bool = False,
    ) -> Optional[str]:
        """Return the live compression tip of ``pinned_session_id`` if the route owns that lineage, else None."""
        try:
            target_session_id = await session_db.get_compression_tip(pinned_session_id)
        except Exception:
            if raise_lookup_errors:
                raise
            logger.debug("Async-delegation compression-tip lookup failed for %s", pinned_session_id, exc_info=True)
            target_session_id = None
        if not target_session_id or target_session_id == pinned_session_id:
            logger.warning(
                "Async-delegation completion pinned to compressed session %s "
                "without a continuation; dropping injection.", pinned_session_id,
            )
            return None
        try:
            tip_row = await session_db.get_session(target_session_id)
        except Exception:
            if raise_lookup_errors:
                raise
            logger.debug("Compression continuation lookup failed for %s", target_session_id, exc_info=True)
            tip_row = None
        if tip_row is None or tip_row.get("ended_at") or is_internal_subagent_row(tip_row):
            logger.warning(
                "Async-delegation compression continuation %s is %s; dropping injection.",
                target_session_id, "unknown" if tip_row is None else "ended",
            )
            return None
        route_owns_lineage = session_entry.session_id in {pinned_session_id, target_session_id}
        if not route_owns_lineage:
            # Across several rotations, accept a stale route only when its own tip is the same live target.
            try:
                route_row = await session_db.get_session(session_entry.session_id)
                route_tip = (
                    await session_db.get_compression_tip(session_entry.session_id)
                    if route_row is not None
                    and route_row.get("ended_at")
                    and route_row.get("end_reason") == "compression"
                    else None
                )
            except Exception:
                if raise_lookup_errors:
                    raise
                logger.debug("Compression route lineage lookup failed for %s", session_entry.session_id, exc_info=True)
                route_tip = None
            route_owns_lineage = route_tip == target_session_id
        if not route_owns_lineage:
            logger.warning(
                "Async-delegation completion for compression lineage %s -> %s "
                "does not own current route %s; dropping injection.",
                pinned_session_id, target_session_id, session_entry.session_id,
            )
            return None
        return target_session_id

    async def _unchanged_completion_route(self, entry: SessionEntry, generation: int) -> Optional[SessionEntry]:
        """Return the same live route only while neither a command nor another writer replaced it."""
        if not self._is_session_run_current(entry.session_key, generation):
            return None
        current = await self.async_session_store.lookup_by_session_key(entry.session_key)
        if current is None or current.session_id != entry.session_id:
            return None
        return current if self._is_session_run_current(entry.session_key, generation) else None

    async def _resolve_async_delegation_session(
        self, session_entry: SessionEntry, pinned_session_id: str,
        *, raise_lookup_errors: bool = False,
    ) -> Optional[SessionEntry]:
        """Resolve only the current owner, its child, or a verified compression continuation.

        A matching chat key is historical peer identity, not evidence that no reset happened.
        Never use a completion to switch an ordinary previous conversation back into the route.
        """
        from gateway.run import _USER_BOUNDARY_END_REASONS

        generation = self._current_session_run_generation(session_entry.session_key)
        try:
            session_db = await self._route_owner_session_db(session_entry.session_key)
            if session_db is None:
                raise RuntimeError("Completion owner database unavailable")
            row = await session_db.get_session(pinned_session_id)
            if row is None:
                return None
            reason = str(row.get("end_reason") or "") if row.get("ended_at") else ""
            if reason in _USER_BOUNDARY_END_REASONS:
                return None
            if is_internal_subagent_row(row):
                owns = await self._delegate_pin_belongs_to_route(
                    session_db, row, session_entry, raise_lookup_errors=True,
                )
                return await self._unchanged_completion_route(session_entry, generation) if owns else None
            if reason != "compression":
                if pinned_session_id != session_entry.session_id:
                    return None
                row_key = row.get("session_key")
                if row_key is not None and row_key != session_entry.session_key:
                    return None
                return await self._unchanged_completion_route(session_entry, generation)
            target = await self._resolve_compression_lineage_target(
                session_db, session_entry, pinned_session_id, raise_lookup_errors=True,
            )
            if target is None:
                return None
            if not await self._unchanged_completion_route(session_entry, generation):
                return None
            if target == session_entry.session_id:
                return session_entry
            return await self.async_session_store.advance_compression_session(
                session_entry.session_key, session_entry.session_id, target,
            )
        except Exception:
            if raise_lookup_errors:
                raise
            logger.debug("Completion owner resolution unavailable for %s", pinned_session_id, exc_info=True)
            return None
