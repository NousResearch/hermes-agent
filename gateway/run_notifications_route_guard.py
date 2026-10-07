"""Async-delegation completion route guard for GatewayRunner.

Split out of ``gateway/run_notifications.py`` (itself split out of ``gateway/run.py``); bound
onto ``GatewayRunner`` via the MRO.

A pinned async-delegation completion may only re-point a chat's route when the pinned session
is genuinely that channel's own. A delegate child arms the watcher with its OWN session id, so
following that pin calls ``switch_session``: it ends the chat's real session and re-stamps the
child row as the channel's (2026-09-17 dev Discord hijack). Unknown ownership fails closed; a
pin that cannot own the route delivers into the chat's CURRENT session instead of dropping the
completion, because the durable claim is acked once this returns.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, cast

from gateway.session import SessionEntry

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

# Platforms where a pinned async-delegation completion may re-point the channel route,
# so a pin must be a verified session OF that channel before switch_session runs.
# Discord-only by decision (2026-09-18); widen per platform once each is proven.
_ROUTE_GUARD_PLATFORMS = ("discord",)
# Session-row sources that can never own a chat route.
_NON_ROUTE_SOURCES = ("subagent",)


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
    source = str(pinned_row.get("source") or "").strip().lower()
    if source in _NON_ROUTE_SOURCES:
        return f"pinned row source={source!r} is not a routable session"
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
                # Idle/timeout end (scale-to-zero norm): the chat route is still valid, so deliver to its
                # current session rather than drop (the row would be acked then silently lost).
                logger.info(
                    "Async-delegation completion pinned to %s-ended session %s; "
                    "retargeting to the chat's current session %s.",
                    _end_reason or "idle", pinned_session_id, session_entry.session_id,
                )
                return session_entry
            follows_compression = True
            target_session_id = await self._resolve_compression_lineage_target(
                session_db, session_entry, pinned_session_id,
            )
            if target_session_id is None:
                return None
        if target_session_id == session_entry.session_id:
            return session_entry
        if not follows_compression:
            guard_platform = _route_guard_platform(session_entry)
            rejection = _pin_route_rejection(pinned_row, session_entry) if guard_platform else ""
            if rejection:
                # Deliver into the chat's CURRENT session instead of re-pointing its route
                # onto a session that is not this channel's (the ended-child branch's
                # disposition). Guarded platforms only; 2026-09-17 dev Discord hijack.
                logger.warning(
                    "Async-delegation completion pinned to %s rejected for %s route %s (%s); "
                    "delivering to the chat's current session %s instead (#57498 route guard).",
                    target_session_id, guard_platform, session_entry.session_key, rejection,
                    session_entry.session_id,
                )
                return session_entry
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