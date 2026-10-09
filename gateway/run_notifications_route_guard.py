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

import json
import logging
from typing import Any, Dict, Optional, cast

from gateway.session import SessionEntry, is_internal_subagent_row

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

# Platforms where a pinned async-delegation completion may re-point the channel route,
# so a pin must be a verified session OF that channel before switch_session runs.
# The extra peer-key checks are Discord-only by #131942's decision. Delegate provenance
# itself is guarded on every messaging platform.
_ROUTE_GUARD_PLATFORMS = ("discord",)


def _route_guard_platform(session_entry: SessionEntry) -> str:
    """Route-guard platform this session serves, lowercased; "" when unguarded/unknown."""
    platform = getattr(session_entry, "platform", None)
    name = getattr(platform, "value", platform)
    if not name:
        # gateway/session.py build_session_key layout is <ns>:<platform>:<chat_type>[...] with
        # ns = "agent:main" (default) or "agent:<profile>" — always two tokens, so the platform
        # is token 2 ("agent:dev:discord:thread:..."). Fall back to it when the entry has no
        # platform object (legacy/hand-built entries).
        parts = str(getattr(session_entry, "session_key", "") or "").split(":")
        name = parts[2] if len(parts) > 2 and parts[0] == "agent" else ""
    name = str(name or "").strip().lower()
    return name if name in _ROUTE_GUARD_PLATFORMS else ""


def _pin_route_rejection(pinned_row: Dict[str, Any], session_entry: SessionEntry) -> str:
    """Reason a pinned completion row cannot own *session_entry*'s route, else "".

    A delegate child (or any session that never carried this channel's key) must never
    become the route: switch_session would end the chat's real session and re-stamp the
    child row as the channel's (2026-09-17 dev Discord hijack).
    """
    if is_internal_subagent_row(pinned_row):
        return "pinned row has delegate execution provenance"
    raw_config = pinned_row.get("model_config")
    if isinstance(raw_config, str):
        try:
            raw_config = json.loads(raw_config)
        except (ValueError, TypeError):
            # A session row's model_config is JSON text when written by the store; a row that
            # carries something else simply has no delegation markers to inspect.
            raw_config = {}
    if isinstance(raw_config, dict) and raw_config.get("_delegate_from"):
        return f"pinned row is a delegate child of {raw_config['_delegate_from']}"
    parent = str(pinned_row.get("parent_session_id") or "")
    if parent and parent == str(getattr(session_entry, "session_id", "") or ""):
        return f"pinned row was spawned by the route's own session {parent}"
    row_key = pinned_row.get("session_key")
    if row_key is not None and str(row_key) != str(getattr(session_entry, "session_key", "") or ""):
        return f"pinned row session_key {row_key!r} is not this route"
    return ""


class GatewayNotificationRouteGuardMixin:
    """Route ownership for pinned async-delegation completions."""

    async def _delegate_pin_belongs_to_route(
        self, session_db: Any, pinned_row: Dict[str, Any], entry: SessionEntry,
    ) -> bool:
        """Prove a child pin descends from this exact chat owner before showing its output there."""
        try:
            owner = await session_db.get_session(entry.session_id)
        except Exception:
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
                logger.debug("Delegate lineage lookup failed for %s", parent_id, exc_info=True)
                return False
            if row is None:
                return False
        return False

    async def _resolve_compression_lineage_target(
        self, session_db: Any, session_entry: SessionEntry, pinned_session_id: str,
    ) -> Optional[str]:
        """Return the live compression tip of ``pinned_session_id`` if the route owns that lineage, else None."""
        try:
            target_session_id = await session_db.get_compression_tip(pinned_session_id)
        except Exception:
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
    ) -> Optional[SessionEntry]:
        """Resolve an async completion to its verified owning gateway session.

        Follow compression-rotation lineage (parent row ended, child continues), but never let a
        late completion override an unrelated /new or restored route. Unknown ownership fails
        closed; the result stays in the delegation records.
        """
        from gateway.run import _USER_BOUNDARY_END_REASONS
        session_db = cast(Any, self._session_db)
        if session_db is None:
            logger.warning(
                "Async-delegation completion has no session database; "
                "dropping injection (#55578 fail-closed)."
            )
            return None
        pinned_row = None
        # Snapshot the run generation before the row lookup awaits: a /stop or /new landing while
        # the lookup is pending must not let this completion re-point the route afterwards.
        run_generation = self._current_session_run_generation(session_entry.session_key)
        try:
            pinned_row = await session_db.get_session(pinned_session_id)
        except Exception:
            logger.debug("Async-delegation parent lookup failed for %s", pinned_session_id, exc_info=True)
        if pinned_row is None:
            logger.warning(
                "Async-delegation completion has unknown spawning session %s; "
                "dropping injection (#55578 fail-closed).", pinned_session_id,
            )
            return None
        target_session_id = pinned_session_id
        follows_compression = False
        if pinned_row.get("ended_at"):
            _end_reason = str(pinned_row.get("end_reason") or "")
            if _end_reason in _USER_BOUNDARY_END_REASONS:
                logger.warning(
                    "Async-delegation completion pinned to user-closed session %s "
                    "(end_reason=%r); dropping injection instead of resurrecting it "
                    "(#55578 fail-closed).", pinned_session_id, _end_reason,
                )
                return None
            if _end_reason != "compression":
                # Idle/timeout end (scale-to-zero norm): retarget only when this pin is
                # proven to belong to the current chat. Delegate children need ancestry;
                # ordinary gateway rows need the same durable key.
                owns_route = (
                    await self._delegate_pin_belongs_to_route(session_db, pinned_row, session_entry)
                    if is_internal_subagent_row(pinned_row)
                    else pinned_row.get("session_key") == session_entry.session_key
                )
                if not owns_route:
                    logger.warning("Async-delegation completion for ended session %s has no verified chat owner",
                                   pinned_session_id)
                    return None
                logger.info(
                    "Async-delegation completion pinned to %s-ended session %s; "
                    "retargeting to the chat's current session %s.",
                    _end_reason or "idle", pinned_session_id, session_entry.session_id,
                )
                return await self._unchanged_completion_route(session_entry, run_generation)
            follows_compression = True
            target_session_id = await self._resolve_compression_lineage_target(
                session_db, session_entry, pinned_session_id,
            )
            if target_session_id is None:
                return None
        if target_session_id == session_entry.session_id:
            if is_internal_subagent_row(pinned_row):
                logger.warning("Async-delegation completion route %s is itself a delegate child; dropping injection",
                               session_entry.session_key)
                return None
            return await self._unchanged_completion_route(session_entry, run_generation)
        if not follows_compression:
            guard_platform = _route_guard_platform(session_entry)
            # Delegate provenance is platform-independent. The additional peer-key checks
            # retain their narrower Discord compatibility boundary from #131942.
            rejection = (
                _pin_route_rejection(pinned_row, session_entry)
                if guard_platform or is_internal_subagent_row(pinned_row) else ""
            )
            if rejection:
                # A child may report to its verified coordinator. A foreign pin, branch, or
                # unknown lineage must never disclose its output to this chat.
                child_of_route = (
                    await self._delegate_pin_belongs_to_route(session_db, pinned_row, session_entry)
                    if is_internal_subagent_row(pinned_row) else False
                )
                logger.warning(
                    "Async-delegation completion pinned to %s rejected for %s route %s (%s); "
                    "%s (#57498 route guard).",
                    target_session_id, guard_platform, session_entry.session_key, rejection,
                    "delivering to verified coordinator " + session_entry.session_id if child_of_route
                    else "dropping injection without a verified owner",
                )
                return (
                    await self._unchanged_completion_route(session_entry, run_generation)
                    if child_of_route else None
                )
        prior_session_id = session_entry.session_id
        if not self._is_session_run_current(session_entry.session_key, run_generation):
            logger.warning(
                "Async-delegation completion for routing key %s was invalidated while resolving pinned "
                "session %s; leaving the route on %s and dropping injection.",
                session_entry.session_key, pinned_session_id, prior_session_id,
            )
            return None
        if follows_compression:
            switched = await self.async_session_store.advance_compression_session(
                session_entry.session_key, prior_session_id, target_session_id,
            )
        else:
            # CAS on the session this completion resolved against: a route replaced meanwhile
            # (/new, /resume) wins over the stale completion.
            switched = await self.async_session_store.switch_session(
                session_entry.session_key, target_session_id, expected_session_id=prior_session_id,
            )
        if switched is None:
            logger.warning(
                "Async-delegation completion could not bind routing key %s to "
                "owning session %s (route moved or unknown); dropping injection.",
                session_entry.session_key, target_session_id,
            )
            return None
        logger.info(
            "Pinned async-delegation completion to owning session %s (was %s) for routing key %s (#57498)",
            target_session_id, prior_session_id, session_entry.session_key,
        )
        return switched
