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
from typing import Any, Dict, Optional

from gateway.run_auto_resume import GatewayAutoResumeMixin
from gateway.session_stall import (
    format_session_stall_notification,
    resolve_session_idle_seconds_from_activity,
    should_clear_session_stall_notification,
    should_emit_session_stall_notification,
)

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")

_SESSION_STORE_PRUNE_INTERVAL = 3600.0  # once per hour
# Summaries running at once: each is a full-context LLM call, so a burst of due chats must not fan
# out unbounded, but one slow summary must not hold every other chat past its window either.
_CACHE_EXPIRY_CONCURRENCY = 3


async def _interruptible_sleep(runner, seconds: int) -> None:
    """Sleep in 1s increments so the watcher stops quickly when ``runner._running`` flips."""
    for _ in range(seconds):
        if not runner._running:
            break
        await asyncio.sleep(1)


class GatewaySessionWatchersMixin(GatewayAutoResumeMixin):
    """Session housekeeping / stall / catalog-refresh watcher loops for GatewayRunner (and, through
    ``GatewayAutoResumeMixin``, the ``delegation.auto_resume`` scheduler)."""

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
        await self._compact_idle_sessions_before_cache_expiry()
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

    def _cache_expiry_due_keys(self, skip=()) -> list[tuple[str, object]]:
        """Cached, idle, human-paced sessions at the end of a keep-alive budget (compact) or past their
        cache window (offload), as ``(key, agent)``; the claim tags ``agent._cache_edge_action``. *skip*
        (keys with a job in flight) is excluded before the claim, which latches the stretch."""
        from agent.prompt_cache_lifecycle import claim_compaction, claim_offload
        from gateway.run import _AGENT_PENDING_SENTINEL
        from gateway.run_agent_cache import _tuple_agent
        cache, lock = getattr(self, "_agent_cache", None), getattr(self, "_agent_cache_lock", None)
        if cache is None or lock is None:
            return []
        with lock:
            entries = [(key, _tuple_agent(entry)) for key, entry in cache.items() if key not in skip]
        due = []
        for key, agent in entries:
            if agent is None or agent is _AGENT_PENDING_SENTINEL or self._is_session_running(key):
                continue
            action = "compact" if claim_compaction(agent) else "offload" if claim_offload(agent) else None
            if action:
                agent._cache_edge_action = action
                due.append((key, agent))
        return due

    async def _compact_idle_sessions_before_cache_expiry(self) -> None:
        """Start one compaction job per due session and return without awaiting them: a summary can
        run for minutes, longer than a later session's fire range and the housekeeping interval, so
        serial summaries would let later chats go cold and stall the sweeps after this one. Jobs are
        background tasks (shutdown cancels them), one per session key, at most
        ``_CACHE_EXPIRY_CONCURRENCY`` summarising at once."""
        jobs = self.__dict__.setdefault("_cache_expiry_jobs", {})
        slots = self.__dict__.setdefault("_cache_expiry_slots", asyncio.Semaphore(_CACHE_EXPIRY_CONCURRENCY))
        for key, live_agent in self._cache_expiry_due_keys(skip=set(jobs)):
            task = self._retain_background_task(
                asyncio.create_task(self._compact_before_cache_expiry(key, live_agent, slots)))
            jobs[key] = task
            task.add_done_callback(lambda _done, key=key: jobs.pop(key, None))

    async def _compact_before_cache_expiry(self, key: str, live_agent, slots: asyncio.Semaphore) -> None:
        """Compact one idle chat through the manual ``/compress`` pipeline (temporary agent,
        compression lease, persist, evict) with an automatic trigger, in the chat's profile scope."""
        claimed_session = getattr(live_agent, "session_id", None)
        async with slots:
            try:
                authority, entry, owner = await asyncio.to_thread(self._cache_expiry_route, key)
                # /new (or a resume) while this job waited re-routed the key: the claim and the warm
                # window checked below belong to the old chat, not the one the route now serves.
                if entry is None or entry.origin is None or entry.session_id != claimed_session:
                    return
                with self._profile_scope_for_source(entry.origin):
                    live = authority.sessions.get(owner) if authority is not None else None
                    # The FIFO claim waits on ``claim_gate`` (as behind ``/compress``): a message
                    # arriving mid-summary starts its turn on the compacted history, not the old one.
                    async with live.claim_gate if live is not None else contextlib.nullcontext():
                        await self._compact_claimed_session(key, entry, live_agent)
            except Exception as e:
                logger.warning("Cache-expiry compaction failed for %s: %s", key, e, exc_info=True)

    def _cache_expiry_route(self, key: str):
        """``(authority, entry, owner session id)`` for *key*. Under an authority the entry may be a
        detached Desktop conversation's (cached agent, no routing entry of its own)."""
        from gateway.session_detached import _route_authority, route_entry
        authority = _route_authority(self, self.session_store, key)
        if authority is None:
            return None, self.session_store.lookup_by_session_key(key), None
        entry = route_entry(authority, key)
        return authority, entry, authority.logical_owner(entry.session_id) if entry is not None else None

    async def _compact_claimed_session(self, key: str, entry, live_agent) -> None:
        """The compaction runs on a temporary agent and evicts the cached one, so the next turn starts
        on the compacted history with a cold cache; nothing is sent until that next user message. An
        ``offload`` claim (the cache is over, nothing compacted it) instead runs the cold prune on the
        stored transcript through the live agent, which commits it to the session store."""
        from agent.prompt_cache_lifecycle import TRIGGER
        from agent.conversation_compression_manual import MIN_MESSAGES, CompressRequest
        if self._is_session_running(key):
            return
        history = await self.async_session_store.load_transcript(entry.session_id)
        if len(history or ()) < MIN_MESSAGES or self._is_session_running(key):
            return
        if live_agent.__dict__.pop("_cache_edge_action", "compact") == "offload":
            from agent.turn_proactive_prune import offload_cold_history
            after = await asyncio.to_thread(offload_cold_history, live_agent, history, entry.session_id)
            if after is not history:
                logger.info("Cache-expiry offload: session=%s", key)
            return
        await self._run_manual_compression(entry.origin, entry, history, CompressRequest(), trigger=TRIGGER)
        logger.info("Cache-expiry compaction: session=%s", key)

    def _compact_then_release(self, key: str, agent) -> bool:
        """An agent-cache eviction (idle TTL, LRU cap): a budget ends at its keep-alives or at the release,
        whichever comes first, so a chat whose step 2 is due at release (``claim_compaction_at_release``,
        which ends the budget: no keep-alive after it) is compacted through the cache-expiry job and its
        evicted agent released after that. True when the gateway loop now owns the release (the decision
        runs there: the LRU cap evicts from a turn thread under the cache lock, and the decision reads the
        session store); False leaves the caller to release as before."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        loop = getattr(self, "_gateway_loop", None)
        if (agent is None or agent is _AGENT_PENDING_SENTINEL or getattr(agent, "_prompt_cache_warmer", None) is None
                or loop is None or loop.is_closed() or not getattr(self, "_running", False)):
            return False
        loop.call_soon_threadsafe(self._decide_compact_then_release, key, agent)
        return True

    def _decide_compact_then_release(self, key: str, agent) -> None:
        from agent.prompt_cache_lifecycle import claim_compaction_at_release, claim_offload
        jobs = self.__dict__.setdefault("_cache_expiry_jobs", {})
        try:
            # Compact if step 2 is due at release; else, past the window, offload before the release.
            action = None if key in jobs else (
                "compact" if claim_compaction_at_release(agent) else "offload" if claim_offload(agent) else None)
            due = action is not None
            if due:
                agent._cache_edge_action = action
        except Exception:
            logger.debug("release-time compaction check failed for %s", key, exc_info=True)
            due = False
        if not due:
            self._spawn_release_thread(self._commit_then_release_soft, (agent, key), f"agent-cache-evict-{key[:24]}",
                                       inline_fallback=True, session_key=key)
            return
        slots = self.__dict__.setdefault("_cache_expiry_slots", asyncio.Semaphore(_CACHE_EXPIRY_CONCURRENCY))
        task = self._retain_background_task(asyncio.ensure_future(self._compact_then_release_job(key, agent, slots)))
        jobs[key] = task
        task.add_done_callback(lambda _done: jobs.pop(key, None))

    async def _compact_then_release_job(self, key: str, agent, slots: asyncio.Semaphore) -> None:
        try:
            await self._compact_before_cache_expiry(key, agent, slots)
        finally:
            self._spawn_release_thread(self._commit_then_release_soft, (agent, key), f"agent-cache-release-{key[:24]}",
                                       inline_fallback=True, session_key=key)

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

    def _stall_candidates(self) -> dict[str, tuple[Any, Any]]:
        """session_key -> (adapter, pending event) from every live adapter's pending slot (default
        + multiplex profiles, deduped by identity), then the overflow queues; first one wins."""
        candidates: dict[str, tuple[Any, Any]] = {}
        maps = (getattr(self, "adapters", {}), *getattr(self, "_profile_adapters", {}).values())
        adapters = {id(a): a for m in maps for a in list(m.values()) if a is not None}
        for adapter in adapters.values():
            pending = getattr(adapter, "_pending_messages", None) or {}
            for session_key, event in list(pending.items()):
                if session_key and session_key not in candidates and event is not None:
                    candidates[session_key] = (adapter, event)
        for session_key, state in list((self.__dict__.get("_sessions") or {}).items()):
            overflow = state.conversation.queued_events
            if not session_key or session_key in candidates or not overflow:
                continue
            source = getattr(overflow[0], "source", None)
            if source is not None and (adapter := self._delivery_adapter_for(source)) is not None:
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
            or bool((_queued_state := self._peek_session_state(session_key))
                    and _queued_state.conversation.queued_events)
        )
        fresh_idle = resolve_session_idle_seconds_from_activity(
            self._session_activity_for_stall(session_key), now=time.time()
        )
        if not still_pending or (fresh_idle is not None and fresh_idle < timeout_seconds):
            logger.info("Session stall notify aborted (no longer stale): session=%s pending=%s "
                        "fresh_idle=%s", session_key, still_pending, fresh_idle)
            notified_map.pop(session_key, None)  # re-arm so a FUTURE genuine stall notifies again
            return False
        from gateway.warning_notifications import present_notification
        from gateway.run import _async_profile_runtime_scope
        try:
            metadata = self._thread_metadata_for_source(source)
            notice = format_session_stall_notification(idle_seconds)
            result = None
            async def send_notice():
                nonlocal result
                # Bound the send: a wedged adapter transport (network hang, dead websocket) must not
                # block the watcher pass — siblings would go unevaluated and the watcher stop.
                result = await asyncio.wait_for(
                    adapter.send(str(source.chat_id), notice, metadata=metadata),
                    timeout=_STALL_NOTIFY_SEND_TIMEOUT_SECONDS,
                )
            async with _async_profile_runtime_scope(self._resolve_profile_home_for_source(source)):
                presented = await present_notification(send_notice, platform=source.platform)
            if not presented:
                notified_map[session_key] = True  # suppressed: latch so the stall is not re-evaluated every tick
                return False
            # Adapters often return SendResult(success=False) instead of raising.
            if result is not None and getattr(result, "success", True) is False:
                raise RuntimeError(getattr(result, "error", "send returned success=False"))
        except TimeoutError:
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
