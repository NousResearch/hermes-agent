"""Session housekeeping / stall / catalog-refresh watcher loops, bound onto ``GatewayRunner`` via the MRO.

``gateway.run`` internals are imported lazily inside method bodies (import cycle), so
``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from gateway.run import GatewayRunner

from gateway.session_stall import (
    format_session_stall_notification,
    resolve_session_idle_seconds_from_activity,
    should_clear_session_stall_notification,
    should_emit_session_stall_notification,
)

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

_SESSION_STORE_PRUNE_INTERVAL = 3600.0  # once per hour


async def _interruptible_sleep(runner, seconds: int) -> None:
    """Sleep in 1s increments so the watcher stops quickly when ``runner._running`` flips."""
    for _ in range(seconds):
        if not runner._running:
            break
        await asyncio.sleep(1)


class GatewaySessionWatchersMixin:
    """Session housekeeping / stall / catalog-refresh watcher loops for GatewayRunner."""

    async def _session_housekeeping_watcher(self, interval: int = 300):
        """Reclaim resources without ending durable conversations."""
        await asyncio.sleep(60)
        while self._running:
            try:
                await self._session_housekeeping()
            except Exception as e:
                logger.debug("Session housekeeping error: %s", e)
            await _interruptible_sleep(self, interval)

    async def _session_housekeeping(self) -> None:
        """Idle/pressure agent-cache sweeps plus the hourly SessionStore prune."""
        try:
            if evicted := self._sweep_idle_cached_agents():
                logger.info("Agent cache idle sweep: evicted %d agent(s)", evicted)
        except Exception as e:
            logger.debug("Idle agent sweep failed: %s", e)
        # Neither LRU cap nor idle TTL knows what a cached transcript costs in memory.
        try:
            # Neither the LRU cap nor the idle TTL is aware of how much memory a cached transcript costs, so
            # a busy gateway keeps every warm session's tool output resident until RSS hits the cgroup limit
            # (#80764). Shed LRU transcripts once the heap is over budget; they reload from the persisted
            # session on the next turn.
            self._sweep_agent_cache_under_pressure()
        except Exception as e:
            logger.debug("Agent cache pressure sweep failed: %s", e)
        # Prune stale SessionStore entries: the dict + sessions.json otherwise grow unbounded.
        prune_ts = getattr(self, "_last_session_store_prune_ts", 0.0)  # tests may omit
        if time.time() - prune_ts > _SESSION_STORE_PRUNE_INTERVAL:
            try:
                max_age = int(getattr(self.config, "session_store_max_age_days", 0) or 0)
                if max_age > 0 and (n := await self.async_session_store.prune_old_entries(max_age)):
                    logger.info("SessionStore prune: dropped %d stale entries", n)
            except Exception as e:
                logger.debug("SessionStore prune failed: %s", e)
            self._last_session_store_prune_ts = time.time()

    def _session_stall_timeout_seconds(self) -> float:
        """Return configured stall timeout (seconds); 0 disables the watchdog."""
        from gateway.run import _float_env
        return _float_env("HERMES_SESSION_STALL_TIMEOUT", 300)

    def _session_activity_for_stall(self, session_key: str) -> Optional[dict]:
        """Stall-progress snapshot from ``AIAgent.get_activity_summary()`` only; no other clocks.

        See #72039.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL
        agent = (getattr(self, "_running_agents", None) or {}).get(session_key)
        if agent is None or agent is _AGENT_PENDING_SENTINEL:
            return None
        try:
            summary = agent.get_activity_summary()
        except Exception:  # incl. AttributeError: agent without an activity summary
            return None
        return summary if isinstance(summary, dict) else None

    def _stall_candidates(self) -> Dict[str, tuple[Any, Any]]:
        """session_key -> (adapter, pending event) from every live adapter's pending slot (default
        + multiplex profiles, deduped by identity), then the overflow queues; first one wins."""
        candidates: Dict[str, tuple[Any, Any]] = {}
        maps = (getattr(self, "adapters", {}), *getattr(self, "_profile_adapters", {}).values())
        adapters = {id(a): a for m in maps for a in list(m.values()) if a is not None}
        for adapter in adapters.values():
            pending = getattr(adapter, "_pending_messages", None) or {}
            for session_key, event in list(pending.items()):
                if session_key and session_key not in candidates and event is not None:
                    candidates[session_key] = (adapter, event)
        for session_key, overflow in list((getattr(self, "_queued_events", None) or {}).items()):
            if not session_key or session_key in candidates or not overflow:
                continue
            source = getattr(overflow[0], "source", None)
            if source is not None and (adapter := self._adapter_for_source(source)) is not None:
                candidates[session_key] = (adapter, overflow[0])
        return candidates

    async def _check_session_stalls(self, timeout_seconds: float) -> int:
        """Notify once per stall episode for pending inbound sessions; returns notices sent."""
        if getattr(self, "_session_stall_notified", None) is None:  # tests may build bare runners
            self._session_stall_notified = {}
        notified_map = self._session_stall_notified
        sent, now, candidates = 0, time.time(), self._stall_candidates()
        # Every candidate carries a non-None pending event, so has_pending_inbound is always True.
        for session_key, (adapter, pending_event) in list(candidates.items()):
            activity = self._session_activity_for_stall(session_key)
            idle_seconds = resolve_session_idle_seconds_from_activity(activity, now=now)
            if should_clear_session_stall_notification(
                timeout_seconds=timeout_seconds, idle_seconds=idle_seconds, has_pending_inbound=True
            ):
                notified_map.pop(session_key, None)
            if idle_seconds is None or not should_emit_session_stall_notification(
                timeout_seconds=timeout_seconds, idle_seconds=idle_seconds,
                has_pending_inbound=True, already_notified=bool(notified_map.get(session_key)),
            ):
                continue
            if await self._notify_session_stall(
                session_key, adapter, pending_event, idle_seconds, activity or {},
                timeout_seconds, notified_map,
            ):
                sent += 1
        # Drop latches for sessions that no longer appear in any pending map.
        for key in [k for k in notified_map if k not in candidates]:
            notified_map.pop(key, None)
        return sent

    async def _notify_session_stall(self, session_key: str, adapter, pending_event,
                                    idle_seconds: float, activity: dict, timeout_seconds: float,
                                    notified_map: dict) -> bool:
        """Log one stall episode and deliver the notice. True only when sent (latched);
        undeliverable (no chat_id) latches without sending; send failures never latch."""
        from gateway.run import _STALL_NOTIFY_SEND_TIMEOUT_SECONDS
        logger.warning(
            "Session stall detected: session=%s idle=%.0fs (timeout=%.0fs, ~%d min); pending "
            "inbound present | last_activity=%s | provenance=%s (agent.session_stall_timeout)",
            session_key, idle_seconds, timeout_seconds, max(1, int(idle_seconds // 60)),
            activity.get("last_activity_desc") or activity.get("last_activity_description")
            or "unknown",
            activity.get("provenance") or activity.get("last_activity_provenance") or "unknown",
        )
        source = getattr(pending_event, "source", None)
        if not getattr(source, "chat_id", None):
            logger.warning("Session stall notify skipped (no chat_id): session=%s", session_key)
            notified_map[session_key] = True  # cannot deliver; latch to avoid log spam every tick
            return False
        # Re-read pending state + activity IMMEDIATELY before delivery: the snapshot ages while
        # earlier candidates await sends; an agent that progressed (or drained its queue) must not
        # get a false stall notice. Abort with the latch un-set so the next tick re-evaluates.
        # See #76354.
        still_pending = (
            (getattr(adapter, "_pending_messages", None) or {}).get(session_key) is not None
            or bool((getattr(self, "_queued_events", None) or {}).get(session_key))
        )
        fresh_idle = resolve_session_idle_seconds_from_activity(
            self._session_activity_for_stall(session_key), now=time.time()
        )
        if not still_pending or (fresh_idle is not None and fresh_idle < timeout_seconds):
            logger.info("Session stall notify aborted (no longer stale): session=%s pending=%s "
                        "fresh_idle=%s", session_key, still_pending, fresh_idle)
            notified_map.pop(session_key, None)  # re-arm so a FUTURE genuine stall notifies again
            return False
        try:
            metadata = self._thread_metadata_for_source(source)
            notice = format_session_stall_notification(idle_seconds)
            # Bound the send: a wedged adapter transport (network hang, dead websocket) must not
            # block the watcher pass — siblings would go unevaluated and the watcher stop.
            result = await asyncio.wait_for(
                adapter.send(str(source.chat_id), notice, metadata=metadata),
                timeout=_STALL_NOTIFY_SEND_TIMEOUT_SECONDS,
            )
            # Adapters often return SendResult(success=False) instead of raising.
            if result is not None and getattr(result, "success", True) is False:
                raise RuntimeError(getattr(result, "error", "send returned success=False"))
        except asyncio.TimeoutError:
            logger.warning(
                "Session stall notify send timed out after %.0fs for %s; will retry next tick",
                _STALL_NOTIFY_SEND_TIMEOUT_SECONDS, session_key,
            )
            return False
        except Exception as exc:
            logger.warning("Session stall notify failed for %s: %s", session_key, exc)
            return False
        notified_map[session_key] = True
        return True

    async def _model_catalog_refresh_watcher(self) -> None:
        """Refresh the /model picker's remote catalogs every TTL window. The picker itself only
        refreshes on a cold/stale open, so if nobody opens ``/model`` the cache never updates."""
        from hermes_cli.model_catalog import refresh_catalogs, refresh_interval_seconds
        await asyncio.sleep(30)  # let startup settle
        while self._running:
            try:
                await asyncio.to_thread(refresh_catalogs)
            except Exception as exc:
                logger.debug("Model catalog refresh failed: %s", exc)
            try:
                interval = refresh_interval_seconds()
            except Exception:
                interval = 1200.0
            deadline = time.monotonic() + interval
            while self._running and time.monotonic() < deadline:
                await asyncio.sleep(min(30.0, max(0.0, deadline - time.monotonic())))

    async def _session_stall_watcher(self, interval: float = 30.0):
        """Pending-inbound + stale-activity stall watchdog. Progress comes only from
        ``get_activity_summary()``; pending inbound is a notify policy gate, not a progress clock.
        Notify-only: never kills the turn (contrast ``gateway_timeout`` / ``shutdown_watchdog``).

        See #72016.
        See #72039.
        """
        # Short initial delay so startup reconnect noise does not false-fire.
        await asyncio.sleep(min(30.0, max(1.0, float(interval))))
        while self._running:
            try:
                if (timeout := self._session_stall_timeout_seconds()) > 0:
                    await self._check_session_stalls(timeout)
            except Exception as exc:
                logger.debug("Session stall watcher error: %s", exc)
            await _interruptible_sleep(self, max(1, int(float(interval))))

    async def _session_health_watcher(self: GatewayRunner, interval: float = 60.0) -> None:
        """Background task that detects and recovers wedged sessions (#58891).

        A session is **wedged** when its last persisted message is an
        ``assistant(tool_calls)`` with no matching ``tool`` result and no
        subsequent ``user`` message — the tool call was persisted but the
        result was never written, and the gateway process is still alive (so
        ``resume_pending`` was never set by the restart watchdog).  The session
        sits idle indefinitely because nothing triggers a new turn.

        This watcher runs every ``interval`` seconds (default 60s).  For each
        non-expired, non-suspended, non-``resume_pending`` session that is not
        currently running an agent, it checks
        ``SessionDB.has_dangling_tool_call_tail()``.  When a wedged session is
        found, it is marked ``resume_pending`` with reason
        ``"orphaned_tool_call"`` and a synthetic recovery turn is scheduled via
        ``_run_startup_resume_event`` — the same machinery used for
        restart-interrupted sessions.  The recovery turn rebuilds history
        (``strip_dangling_tool_call_tail`` removes the orphaned call), the
        ``_is_resume_pending`` branch injects a recovery note, and the session
        is back online without manual intervention.

        The watcher is deliberately conservative:
        - Sessions with an active agent (``_running_agents``) are skipped —
          the turn may still be in progress.
        - Sessions already marked ``resume_pending`` or ``suspended`` are
          skipped — they are handled by the existing restart-recovery or
          forced-wipe paths.
        - Sessions whose adapter is unavailable are skipped — they will be
          picked up when the platform reconnects (the reconnect watcher calls
          ``_schedule_resume_pending_sessions`` which honours the new reason).
        - At most one recovery is scheduled per session per watcher cycle;
          the ``resume_pending`` flag prevents re-detection in the next cycle.
        """
        await asyncio.sleep(90)  # initial delay — let startup restore finish
        while self._running:
            try:
                await self._session_health_probe()
            except Exception as e:
                logger.debug("Session health watcher error: %s", e)
            # Sleep in small increments so we can stop quickly.
            _slept = 0.0
            while _slept < interval and self._running:
                await asyncio.sleep(1)
                _slept += 1

    async def _session_health_probe(self: GatewayRunner) -> int:
        """Run one wedge-detection + recovery pass.  Returns the count of
        sessions scheduled for recovery.

        Separated from ``_session_health_watcher`` so it can be called directly
        in tests without the 90-second initial delay or the sleep loop.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL
        from gateway.platforms.base import MessageEvent, MessageType
        # Don't schedule recovery turns while the gateway is draining —
        # they would be immediately interrupted by the shutdown sequence.
        if getattr(self, "_draining", False):
            return 0
        session_items = await self.async_session_store.list_session_items()
        # A durable session may be reachable through multiple routing keys
        # after /resume or conversation-scope rebinding (#64934).  Treat an
        # agent running under any alias as active for the whole session.
        _running_session_ids = {
            entry.session_id
            for key, entry in session_items
            if key in self._running_agents
        }
        _pending_session_ids = {
            entry.session_id for _, entry in session_items if entry.resume_pending
        }
        _wedge_state: dict[str, bool] = {}
        _wedged: list = []
        for key, entry in session_items:
            # Skip sessions that are already being handled by another
            # path or are not candidates for health recovery.
            if entry.suspended or entry.resume_pending:
                continue
            if entry.expiry_finalized:
                continue
            if entry.session_id in _running_session_ids:
                continue
            if entry.session_id in _pending_session_ids:
                continue
            if entry.origin is None:
                continue
            # Skip sessions with a pending blocking approval — the agent
            # is waiting for a /approve or /deny, not a recovery turn.
            # A synthetic empty-text turn would spin up the agent only to
            # block again on the same approval gate.
            try:
                from tools.approval import has_blocking_approval
                if has_blocking_approval(key):
                    continue
            except Exception:
                pass  # approval module unavailable — proceed
            # DB check — is the last message an unanswered tool_call?
            # The async SessionStore facade keeps its SQLite lock and query
            # work off the gateway event loop.
            if entry.session_id not in _wedge_state:
                try:
                    _wedge_state[entry.session_id] = (
                        await self.async_session_store.has_dangling_tool_call_tail(
                            entry.session_id
                        )
                    )
                except Exception:
                    _wedge_state[entry.session_id] = False
            is_wedged = _wedge_state[entry.session_id]
            if not is_wedged:
                continue
            _wedged.append((key, entry))

        scheduled = 0
        _scheduled_session_ids: set[str] = set()
        for key, entry in _wedged:
            if entry.session_id in _scheduled_session_ids:
                continue
            source = entry.origin
            adapter = self._adapter_for_source(source)
            if adapter is None:
                logger.debug(
                    "Session health: wedged session %s has no adapter "
                    "(platform %s) — will retry on reconnect",
                    entry.session_id,
                    source.platform.value if source and source.platform else "?",
                )
                continue
            # Validate the session owner against the current allowlist before
            # auto-resuming — mirrors _schedule_resume_pending_sessions (#23778).
            # A session whose owner has been removed from the allowlist must not
            # silently receive an agent response from the health watcher.
            try:
                if not self._is_user_authorized(source):
                    logger.debug(
                        "Session health: skipping wedged session %s — "
                        "owner no longer authorized",
                        entry.session_id,
                    )
                    continue
            except Exception as exc:
                logger.debug(
                    "Session health: skipping wedged session %s — "
                    "authorization check failed: %s",
                    entry.session_id,
                    exc,
                )
                continue
            # Mark resume_pending so the _is_resume_pending branch in
            # _handle_message_with_agent injects the recovery note and
            # strip_dangling_tool_call_tail fires during history rebuild.
            if key in self._running_agents:
                continue
            if not await self.async_session_store.mark_resume_pending(
                key, reason="orphaned_tool_call"
            ):
                continue
            # The awaited persistence yields to inbound-message handling.
            # Never replace a runner that claimed this key in that window.
            if key in self._running_agents:
                continue
            # Pre-claim the runner slot so a real inbound message
            # arriving between mark and dispatch queues behind us
            # instead of spinning a duplicate agent (#45456).
            self._running_agents[key] = _AGENT_PENDING_SENTINEL
            self._running_agents_ts[key] = time.time()
            self._persist_active_agents()
            event = MessageEvent(
                text="",
                message_type=MessageType.TEXT,
                source=source,
                internal=True,
            )
            try:
                task = asyncio.create_task(
                    self._run_startup_resume_event(adapter, event, key)
                )
            except RuntimeError:
                # Event loop closed between probe start and task creation
                # (gateway shutting down).  Release the sentinel so the
                # session is not permanently stuck; resume_pending stays
                # set so the next boot picks it up.
                self._release_running_agent_state(key)
                continue
            _scheduled_session_ids.add(entry.session_id)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            logger.info(
                "Session health: detected wedged session %s "
                "(orphaned tool_call tail) — scheduled recovery turn",
                entry.session_id,
            )
            scheduled += 1
        return scheduled
