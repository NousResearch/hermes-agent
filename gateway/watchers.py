"""Background monitor loops for GatewayRunner.

Extracted verbatim from ``gateway/run.py`` (god-file decomposition Phase 3c).
These are the long-running coroutines that redeliver pending obligations,
resume sessions, expire idle ones, reconnect adapters, run background tasks,
watch update progress, and supervise spawned processes.

They use only ``self`` state, so they live on a mixin that ``GatewayRunner``
inherits -- every ``self._`` call site resolves identically via the MRO,
making this a behavior-neutral move. Same shape as the earlier
``GatewayAuthorizationMixin`` / ``GatewayKanbanWatchersMixin`` /
``GatewaySlashCommandsMixin`` phases.

``_dispose_unused_adapter`` moves with them -- the reconnect watcher is its
only caller -- and is re-exported from ``gateway.run`` so the existing import
path keeps working.

Module-level names still owned by ``gateway/run.py`` are imported inside the
methods that use them: ``gateway.run`` imports this module, so a module-level
import would be circular. Resolving at call time also keeps
``monkeypatch.setattr("gateway.run.X", ...)`` working in existing tests.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import List, Optional

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType

# Match the logger run.py uses (logging.getLogger(__name__) where __name__ ==
# "gateway.run") so extracted log records keep their original logger name.
logger = logging.getLogger("gateway.run")


async def _dispose_unused_adapter(adapter: "BasePlatformAdapter | None") -> None:
    """Best-effort dispose for an adapter that never made it onto ``self.adapters``.

    The reconnect watcher in ``GatewayRunner._platform_reconnect_watcher``
    constructs a fresh adapter on every retry attempt. When the connect
    call fails — for any of the three reasons (non-retryable error,
    retryable error, exception during connect) — the adapter is dropped
    without ever being installed, so nothing else will call its
    ``disconnect()``. Any resources the adapter opened in ``__init__``
    (e.g. ``APIServerAdapter`` opens a SQLite ``ResponseStore`` that
    holds 2 fds — the db file and its WAL sidecar) stay open until
    garbage collection sweeps the unreachable object, which Python's
    cyclic GC does not do promptly for asyncio-bound objects with
    native handles. The cumulative leak is 2 fds × every retry at the
    300s backoff cap ≈ 12 fds/hour, and the default 2560-fd ulimit
    is exhausted in ~12h of continuous failure, after which every
    open() call on the gateway raises ``OSError: [Errno 24] Too many
    open files`` and the gateway becomes a zombie (#37011).

    This helper centralises the dispose-with-suppression so the three
    failure paths in the reconnect watcher can all call it without
    each one having to know that ``disconnect()`` may itself raise
    on a half-constructed adapter.

    ``adapter`` may be ``None``: the reconnect watcher initialises
    ``adapter = None`` before the ``try`` so the ``except Exception``
    arm can dispose a half-constructed object, and also early-returns
    here when ``_create_adapter()`` returned ``None``.
    """
    if adapter is None:
        return
    try:
        await adapter.disconnect()
    except Exception:
        # Half-constructed adapters (e.g. APIServerAdapter that
        # crashed during aiohttp app setup) can raise from
        # disconnect() on objects that never finished initializing.
        # We must not let that escape and abort the watcher loop.
        #
        # On Python 3.8+, ``asyncio.CancelledError`` inherits from
        # ``BaseException`` (not ``Exception``), so this ``except
        # Exception`` does not swallow task cancellation. We don't
        # re-raise explicitly because the watcher loop intentionally
        # treats dispose failures as best-effort: a failed ``disconnect``
        # call should not take down the reconnect watcher that
        # itself is what's keeping the gateway alive during a partial
        # outage.
        logger.debug(
            "Adapter dispose raised on unowned adapter %r",
            getattr(adapter, "name", type(adapter).__name__),
            exc_info=True,
        )


class GatewayWatchersMixin:
    """Background monitor loops mixed into :class:`GatewayRunner`."""

    async def _redeliver_pending_obligations(self) -> int:
        """Redeliver final responses recorded in the delivery ledger by a
        previous (now dead) gateway process.

        Runs at startup BEFORE ``_schedule_resume_pending_sessions``. A
        session with a recoverable obligation already produced its answer —
        the turn completed and only delivery is owed — so this method sends
        the stored text and clears ``resume_pending`` for that session,
        preventing the resume path from re-running (and re-paying for) a
        turn whose output we hold.

        Crash-ambiguity contract (see gateway/delivery_ledger.py):
        rows that were mid-send or previously rejected carry a visible
        recovered-reply marker so a possible duplicate is labeled, never
        silent. Returns the number of redeliveries attempted.
        """
        try:
            from gateway.delivery_ledger import (
                RECOVERED_MARKER,
                ledger_enabled,
                mark_delivered,
                mark_failed,
                sweep_recoverable,
            )

            if not await asyncio.to_thread(ledger_enabled):
                return 0
            # Only claim rows we can actually send this boot: self.adapters
            # holds a platform only after its connect() succeeded, and each
            # claim spends one of the row's three redelivery attempts.
            _deliverable = {
                getattr(p, "value", str(p)) for p in self.adapters
            }
            claimed = await asyncio.to_thread(
                sweep_recoverable, None, deliverable_platforms=_deliverable
            )
        except Exception:
            logger.debug("delivery ledger sweep failed", exc_info=True)
            return 0
        if not claimed:
            return 0

        redelivered = 0
        for row in claimed:
            try:
                platform = Platform(row["platform"])
            except Exception:
                logger.debug(
                    "obligation %s: unknown platform %r",
                    row["obligation_id"], row.get("platform"),
                )
                continue
            adapter = self.adapters.get(platform)
            if adapter is None:
                # Platform not connected this boot — leave the row claimed;
                # attempts cap + stale cutoff bound the retries on later boots.
                continue
            content = row["content"]
            if row.get("needs_marker"):
                content = RECOVERED_MARKER + content
            metadata = (
                {"thread_id": row["thread_id"]} if row.get("thread_id") else None
            )
            try:
                result = await adapter.send(
                    chat_id=row["chat_id"],
                    content=content,
                    metadata=metadata,
                )
            except Exception as send_err:
                logger.warning(
                    "obligation %s: redelivery send raised: %s",
                    row["obligation_id"], send_err,
                )
                result = None
            try:
                if result is not None and getattr(result, "success", False):
                    await asyncio.to_thread(mark_delivered, row["obligation_id"])
                    redelivered += 1
                    logger.info(
                        "Redelivered recovered final response to %s:%s "
                        "(obligation %s, attempt %d)",
                        row["platform"], row["chat_id"],
                        row["obligation_id"], row["attempts"],
                    )
                else:
                    await asyncio.to_thread(
                        mark_failed,
                        row["obligation_id"],
                        str(getattr(result, "error", "") or "send failed"),
                    )
            except Exception:
                logger.debug("delivery ledger update failed", exc_info=True)

            # The answer reached (or was owed to) this session — don't ALSO
            # re-run the turn via the resume path.
            session_key = row.get("session_key") or ""
            if session_key:
                try:
                    await self.async_session_store.clear_resume_pending(session_key)
                except Exception:
                    logger.debug(
                        "clear_resume_pending failed for %s", session_key,
                        exc_info=True,
                    )
        return redelivered

    def _schedule_resume_pending_sessions(self, platform=None) -> int:
        """Auto-continue fresh restart-interrupted sessions after startup.

        ``resume_pending`` already preserves the transcript AND the existing
        ``_is_resume_pending`` branch in ``_handle_message_with_agent``
        injects a reason-aware recovery system note on the next turn.  This
        method closes the UX gap by synthesizing that next turn once
        adapters are back online — the event text is empty so the existing
        injection path owns the wording and we never double up.

        Adapters that are not yet ready (adapter missing from
        ``self.adapters``) are skipped silently; their sessions stay
        ``resume_pending`` and will auto-resume on the next real user
        message, or when the platform reconnects — the reconnect watcher
        calls this again scoped to that ``platform``.

        ``platform`` (a ``Platform``) restricts the pass to sessions that
        originated on that platform.  The reconnect path passes it so a
        platform coming back online retries only its own sessions and never
        re-touches another platform's in-flight recoveries.  Sessions whose
        agent is already running are skipped regardless, so a session
        scheduled at startup is never resumed a second time.
        """
        # Resolved at call time: gateway.run imports this module, so a
        # module-level import here would be circular.
        from gateway.run import (
            _AGENT_PENDING_SENTINEL,
            _auto_continue_freshness_window,
        )

        window = _auto_continue_freshness_window()
        try:
            with self.session_store._lock:  # noqa: SLF001 — snapshot under lock
                self.session_store._ensure_loaded_locked()  # noqa: SLF001
                candidates = [
                    entry for entry in self.session_store._entries.values()  # noqa: SLF001
                    if entry.resume_pending
                    and not entry.suspended
                    and entry.origin is not None
                    and entry.resume_reason in self._AUTO_RESUME_REASONS
                    and (platform is None or entry.origin.platform == platform)
                ]
        except Exception as exc:
            logger.warning("Failed to enumerate resume-pending sessions: %s", exc)
            return 0

        # Defense-3 (#30719): break the SIGTERM-respawn loop. Only count this
        # boot when there are restart-interrupted sessions to resume — a clean
        # boot must not accrue toward the breaker. If too many such boots have
        # happened in the configured window, skip auto-resume for THIS boot:
        # the gateway still comes up and serves real inbound messages, it just
        # stops replaying the session that keeps killing it. The session stays
        # resume_pending, so a real user message can still continue it (a human
        # is now in the loop). Defenses 1-2 cover the cron/CLI/terminal paths;
        # this catches every other SIGTERM source (e.g. a raw `terminal(
        # "launchctl kickstart ai.hermes.gateway")`).
        if candidates:
            try:
                from gateway import restart_loop_guard as _rlg

                _max_restarts, _window = self._restart_loop_guard_config()
                if _rlg.check_and_record(_max_restarts, _window):
                    return 0
            except Exception as exc:  # noqa: BLE001 — breaker must fail OPEN
                logger.debug("Restart-loop guard check skipped: %s", exc)

        now = datetime.now()
        scheduled = 0
        for entry in candidates:
            marker = entry.last_resume_marked_at or entry.updated_at
            if marker is not None and (now - marker).total_seconds() > window:
                continue

            # Already being resumed (e.g. scheduled at startup and still
            # in-flight) — don't synthesize a second continuation turn.
            if self._is_session_running(entry.session_key):
                continue

            source = entry.origin
            adapter = self._adapter_for_source(source)
            if adapter is None:
                logger.debug(
                    "Skipping auto-resume for %s: adapter not ready for %s",
                    entry.session_key,
                    getattr(source.platform, "value", source.platform),
                )
                continue

            # Validate the session owner against the current allowlist
            # before auto-resuming. A session created before
            # TELEGRAM_ALLOWED_USERS (or equivalent) was configured, or
            # before the owner was removed from it, must not silently
            # receive a full agent response on gateway restart just
            # because it has a resume-pending marker (issue #23778).
            try:
                if not self._is_user_authorized(source):
                    logger.warning(
                        "Skipping auto-resume for %s: session owner is no "
                        "longer authorized under the current allowlist",
                        entry.session_key,
                    )
                    continue
            except Exception as exc:
                logger.warning(
                    "Skipping auto-resume for %s: authorization check failed: %s",
                    entry.session_key, exc,
                )
                continue

            # Claim the session slot *before* spawning the task so that an
            # inbound message arriving between task creation and the task's
            # first await (where _process_message_background sets the real
            # sentinel) sees the slot as occupied and queues behind it
            # instead of spinning up a duplicate AIAgent (#45456).
            _resume_state = self._session_state(entry.session_key)
            _resume_state.turn.agent = _AGENT_PENDING_SENTINEL
            _resume_state.turn.started_ts = time.time()
            self._persist_active_agents()

            # Empty-text internal event — the _is_resume_pending branch in
            # _handle_message_with_agent prepends the proper reason-aware
            # system note before the turn runs.
            event = MessageEvent(
                text="",
                message_type=MessageType.TEXT,
                source=source,
                internal=True,
            )
            task = asyncio.create_task(
                self._run_startup_resume_event(adapter, event, entry.session_key)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            if getattr(self, "_startup_restore_in_progress", False):
                tasks = getattr(self, "_startup_restore_tasks", None)
                if tasks is None:
                    tasks = []
                    self._startup_restore_tasks = tasks
                tasks.append(task)
            scheduled += 1
        if scheduled:
            logger.info(
                "Scheduled auto-resume for %d restart-interrupted session(s)",
                scheduled,
            )
        return scheduled

    async def _session_expiry_watcher(self, interval: int = 300):
        """Background task that finalizes expired sessions.

        Runs every ``interval`` seconds (default 5 min).  For each session
        whose reset policy has expired, invokes ``on_session_finalize``
        hooks, cleans up the cached AIAgent's tool resources, evicts the
        cache entry so it can be garbage-collected, and marks the session
        so it won't be finalized again.
        """
        # Resolved at call time: gateway.run imports this module, so a
        # module-level import here would be circular.
        from gateway.run import _AGENT_PENDING_SENTINEL

        await asyncio.sleep(60)  # initial delay — let the gateway fully start
        _finalize_failures: dict[str, int] = {}  # session_id -> consecutive failure count
        _MAX_FINALIZE_RETRIES = 3
        while self._running:
            try:
                await self.async_session_store._ensure_loaded()
                # Collect expired sessions first, then log a single summary.
                _expired_entries = []
                for key, entry in list(self.session_store._entries.items()):
                    if entry.expiry_finalized:
                        continue
                    if not await self.async_session_store._is_session_expired(entry):
                        continue
                    _expired_entries.append((key, entry))

                if _expired_entries:
                    # Extract platform names from session keys for a compact summary.
                    # Keys look like "agent:main:telegram:dm:12345" — platform is field [2].
                    _platforms: dict[str, int] = {}
                    for _k, _e in _expired_entries:
                        _parts = _k.split(":")
                        _plat = _parts[2] if len(_parts) > 2 else "unknown"
                        _platforms[_plat] = _platforms.get(_plat, 0) + 1
                    _plat_summary = ", ".join(
                        f"{p}:{c}" for p, c in sorted(_platforms.items())
                    )
                    logger.info(
                        "Session expiry: %d sessions to finalize (%s)",
                        len(_expired_entries), _plat_summary,
                    )

                for key, entry in _expired_entries:
                    try:
                        try:
                            from hermes_cli.lifecycle import finalize_session
                            _parts = key.split(":")
                            _platform = _parts[2] if len(_parts) > 2 else ""
                            finalize_session(
                                session_id=entry.session_id,
                                platform=_platform,
                                reason="session_expired",
                            )
                        except Exception:
                            pass
                        # Shut down memory provider and close tool resources
                        # on the cached agent.  Idle agents live in
                        # _agent_cache (not _running_agents), so look there.
                        _cached_agent = None
                        _cache_lock = getattr(self, "_agent_cache_lock", None)
                        if _cache_lock is not None:
                            with _cache_lock:
                                _cached = self._agent_cache.get(key)
                                _cached_agent = _cached[0] if isinstance(_cached, tuple) else _cached if _cached else None
                        # Fall back to _running_agents in case the agent is
                        # still mid-turn when the expiry fires.
                        if _cached_agent is None:
                            _exp_state = self._peek_session_state(key)
                            _cached_agent = _exp_state.turn.agent if _exp_state else None
                        if _cached_agent and _cached_agent is not _AGENT_PENDING_SENTINEL:
                            await self._cleanup_agent_resources_off_loop(
                                _cached_agent, context="session expiry"
                            )
                        # Drop the cache entry so the AIAgent (and its LLM
                        # clients, tool schemas, memory provider refs) can
                        # be garbage-collected.  Otherwise the cache grows
                        # unbounded across the gateway's lifetime.
                        self._evict_cached_agent(key)
                        # Permanently finalizing this session — one funnel
                        # call drops every conversation-scoped dict AND the
                        # boundary security state (approvals, update
                        # prompts, slash-confirm) so the dicts don't grow
                        # unbounded across the gateway's lifetime. (Idle
                        # agent-cache eviction must NOT do this: the
                        # session is still alive and a resumed turn rebuilds
                        # its agent from these overrides. Only true session
                        # finalization, /new, and /reset clear them.) See
                        # _CONVERSATION_SCOPED_STATE.
                        self._clear_conversation_scope(
                            key, reason="expiry_finalized"
                        )
                        # Persist the finalized flag to sessions.json AND
                        # state.db (single write-path, #9006) — also drops
                        # the persisted /model override, since finalization
                        # is a conversation boundary.
                        await self.async_session_store.set_expiry_finalized(entry)
                        logger.debug(
                            "Session expiry finalized for %s",
                            entry.session_id,
                        )
                        _finalize_failures.pop(entry.session_id, None)
                    except Exception as e:
                        failures = _finalize_failures.get(entry.session_id, 0) + 1
                        _finalize_failures[entry.session_id] = failures
                        if failures >= _MAX_FINALIZE_RETRIES:
                            logger.warning(
                                "Session finalize gave up after %d attempts for %s: %s. "
                                "Marking as finalized to prevent infinite retry loop.",
                                failures, entry.session_id, e,
                            )
                            await self.async_session_store.set_expiry_finalized(
                                entry, clear_model_override=False
                            )
                            _finalize_failures.pop(entry.session_id, None)
                        else:
                            logger.debug(
                                "Session finalize failed (%d/%d) for %s: %s",
                                failures, _MAX_FINALIZE_RETRIES, entry.session_id, e,
                            )

                if _expired_entries:
                    _done = sum(
                        1 for _, e in _expired_entries if e.expiry_finalized
                    )
                    _failed = len(_expired_entries) - _done
                    if _failed:
                        logger.info(
                            "Session expiry done: %d finalized, %d pending retry",
                            _done, _failed,
                        )
                    else:
                        logger.info(
                            "Session expiry done: %d finalized", _done,
                        )

                # Sweep agents that have been idle beyond the TTL regardless
                # of session reset policy.  This catches sessions with very
                # long / "never" reset windows, whose cached AIAgents would
                # otherwise pin memory for the gateway's entire lifetime.
                try:
                    _idle_evicted = self._sweep_idle_cached_agents()
                    if _idle_evicted:
                        logger.info(
                            "Agent cache idle sweep: evicted %d agent(s)",
                            _idle_evicted,
                        )
                except Exception as _e:
                    logger.debug("Idle agent sweep failed: %s", _e)

                # Periodically prune stale SessionStore entries.  The
                # in-memory dict (and sessions.json) would otherwise grow
                # unbounded in gateways serving many rotating chats /
                # threads / users over long time windows.  Pruning is
                # invisible to users — a resumed session just gets a
                # fresh session_id, exactly as if the reset policy fired.
                _last_prune_ts = getattr(self, "_last_session_store_prune_ts", 0.0)
                _prune_interval = 3600.0  # once per hour
                if time.time() - _last_prune_ts > _prune_interval:
                    try:
                        _max_age = int(
                            getattr(self.config, "session_store_max_age_days", 0) or 0
                        )
                        if _max_age > 0:
                            _pruned = await self.async_session_store.prune_old_entries(_max_age)
                            if _pruned:
                                logger.info(
                                    "SessionStore prune: dropped %d stale entries",
                                    _pruned,
                                )
                    except Exception as _e:
                        logger.debug("SessionStore prune failed: %s", _e)
                    self._last_session_store_prune_ts = time.time()
            except Exception as e:
                logger.debug("Session expiry watcher error: %s", e)
            # Sleep in small increments so we can stop quickly
            for _ in range(interval):
                if not self._running:
                    break
                await asyncio.sleep(1)

    async def _platform_reconnect_watcher(self) -> None:
        """Background task that periodically retries connecting failed platforms.

        Uses exponential backoff: 30s → 60s → 120s → 240s → 300s (cap).
        Retryable failures (network/DNS blips) keep retrying at the backoff
        cap indefinitely — they self-heal once connectivity returns, so a
        transient outage never requires manual intervention. Non-retryable
        failures (bad auth, etc.) drop out of the queue immediately. The
        circuit breaker (``_pause_failed_platform`` / ``/platform pause``)
        remains available for manual operator control via ``/platform list``
        and ``/platform resume <name>``, but is no longer triggered
        automatically — auto-pausing a recovered platform was the cause of
        bots silently staying dead after a transient DNS failure.
        """
        # Resolved at call time: gateway.run imports this module, so a
        # module-level import here would be circular.
        from gateway.run import (
            _platform_has_bot_credential,
            _reconnect_backoff,
        )

        await asyncio.sleep(10)  # initial delay — let startup finish
        while self._running:
            if not self._failed_platforms:
                # Nothing to reconnect — sleep and check again
                for _ in range(30):
                    if not self._running:
                        return
                    if self._failed_platforms:
                        break
                    await asyncio.sleep(1)
                continue

            now = time.monotonic()
            for platform in list(self._failed_platforms.keys()):
                if not self._running:
                    return
                info = self._failed_platforms.get(platform)
                if info is None:
                    # Removed concurrently (e.g. a manual /platform resume,
                    # or a reconnect that succeeded via a different path)
                    # between the snapshot above and this lookup. Not an
                    # error -- just nothing to do for it this pass.
                    continue
                # Skip paused platforms entirely — they need explicit
                # /platform resume to come back.
                if info.get("paused"):
                    continue
                if now < info["next_retry"]:
                    continue  # not time yet

                platform_config = info["config"]
                attempt = info["attempts"] + 1
                # Empty-token primary configs can never reconnect; drop them so
                # multiplex setups where a secondary profile owns the bot do
                # not spin forever (#64674).
                if not _platform_has_bot_credential(platform, platform_config):
                    logger.warning(
                        "Reconnect %s: no bot credential on queued config, "
                        "removing from retry queue",
                        platform.value,
                    )
                    del self._failed_platforms[platform]
                    continue
                logger.info(
                    "Reconnecting %s (attempt %d)...",
                    platform.value, attempt,
                )

                adapter = None
                try:
                    adapter = self._create_adapter(platform, platform_config)
                    if not adapter:
                        logger.warning(
                            "Reconnect %s: adapter creation returned None, removing from retry queue",
                            platform.value,
                        )
                        del self._failed_platforms[platform]
                        continue

                    adapter.set_message_handler(self._primary_message_handler())
                    adapter.set_fatal_error_handler(self._handle_adapter_fatal_error)
                    adapter.set_session_store(self.session_store)
                    adapter.set_busy_session_handler(self._handle_active_session_busy_message)
                    _set_reaction = getattr(adapter, "set_reaction_handler", None)
                    if callable(_set_reaction):
                        _set_reaction(self._handle_reaction_event)
                    adapter.set_topic_recovery_fn(self._recover_telegram_topic_thread_id)
                    adapter.set_authorization_check(self._make_adapter_auth_check(adapter.platform))
                    adapter._busy_text_mode = self._busy_text_mode

                    # Reconnect after an outage: preserve the platform's
                    # server-side update queue so messages sent while the bot
                    # was offline are delivered rather than dropped (#46621).
                    success = await self._connect_adapter_with_timeout(
                        adapter, platform, is_reconnect=True
                    )
                    if success:
                        self.adapters[platform] = adapter
                        self._sync_voice_mode_state_to_adapter(adapter)
                        # Wire voice input callback on reconnect as well (#60623).
                        if hasattr(adapter, "_voice_input_callback"):
                            adapter._voice_input_callback = self._handle_voice_channel_input
                        self.delivery_router.adapters = self.adapters
                        del self._failed_platforms[platform]
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="connected",
                            error_code=None,
                            error_message=None,
                        )
                        logger.info("✓ %s reconnected successfully", platform.value)

                        # Rebuild channel directory with the new adapter
                        try:
                            from gateway.channel_directory import build_channel_directory
                            await build_channel_directory(self.adapters)
                        except Exception:
                            pass

                        # A platform that was offline at gateway startup never
                        # got its restart-interrupted sessions auto-resumed —
                        # the startup pass skips sessions whose adapter isn't
                        # connected yet. Now that it's back, retry the
                        # auto-resume scoped to this platform so recovery
                        # doesn't silently wait for a manual user message.
                        try:
                            self._schedule_resume_pending_sessions(platform=platform)
                        except Exception:
                            logger.debug(
                                "resume-pending reschedule after %s reconnect failed",
                                platform.value,
                                exc_info=True,
                            )
                    # Check if the failure is non-retryable
                    elif adapter.has_fatal_error and not adapter.fatal_error_retryable:
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="fatal",
                            error_code=adapter.fatal_error_code,
                            error_message=adapter.fatal_error_message,
                        )
                        logger.warning(
                            "Reconnect %s: non-retryable error (%s), removing from retry queue",
                            platform.value, adapter.fatal_error_message,
                        )
                        # The adapter is about to be dropped from the queue
                        # without ever being installed on self.adapters, so
                        # nothing else will call disconnect() on it. We must
                        # dispose it here, otherwise the resource owners it
                        # constructed in __init__ (ResponseStore for
                        # APIServerAdapter, etc.) leak 2 fds each. The
                        # gateway hits the 2560-fd limit after ~12h of
                        # failed reconnects at the 300s backoff cap (#37011).
                        await _dispose_unused_adapter(adapter)
                        del self._failed_platforms[platform]
                    else:
                        self._update_platform_runtime_status(
                            platform.value,
                            platform_state="retrying",
                            error_code=adapter.fatal_error_code,
                            error_message=adapter.fatal_error_message or "failed to reconnect",
                        )
                        backoff = _reconnect_backoff(attempt)
                        info["attempts"] = attempt
                        info["next_retry"] = time.monotonic() + backoff
                        logger.info(
                            "Reconnect %s failed, next retry in %ds",
                            platform.value, backoff,
                        )
                        # Same fd-leak concern as the non-retryable branch
                        # above: the adapter failed to connect and is being
                        # thrown away. Without an explicit dispose call, the
                        # resources it opened in __init__ stay open until
                        # the next GC pass — and aiohttp/SQLite handles
                        # don't get GC'd promptly, so 2 fds/retry leak at
                        # 300s backoff cap = ~12 fds/hour (#37011).
                        await _dispose_unused_adapter(adapter)
                        # Retryable failures (network/DNS blips) keep retrying
                        # at the backoff cap indefinitely — they self-heal once
                        # connectivity returns. We do NOT auto-pause them: a
                        # transient outage must never require manual `/platform
                        # resume` to recover. Non-retryable failures (bad auth,
                        # etc.) already drop out of the queue via the
                        # `not fatal_error_retryable` branch above, so anything
                        # reaching here is by definition retryable.
                except Exception as e:
                    if adapter is not None:
                        # An exception escaping the connect call path
                        # (DNS timeout, aiohttp server.start() crash, etc.)
                        # leaves the adapter in the same unowned state as
                        # the two branches above. Dispose so __init__
                        # resources don't accumulate while the watcher
                        # keeps retrying.
                        await _dispose_unused_adapter(adapter)
                    self._update_platform_runtime_status(
                        platform.value,
                        platform_state="retrying",
                        error_code=None,
                        error_message=str(e),
                    )
                    backoff = _reconnect_backoff(attempt)
                    info["attempts"] = attempt
                    info["next_retry"] = time.monotonic() + backoff
                    logger.warning(
                        "Reconnect %s error: %s, next retry in %ds",
                        platform.value, e, backoff,
                    )
                    # A raised exception during reconnect (connect timeout, DNS
                    # resolution failure, etc.) is inherently transient — keep
                    # retrying at the backoff cap rather than auto-pausing.

            # Check every 10 seconds for platforms that need reconnection
            for _ in range(10):
                if not self._running:
                    return
                await asyncio.sleep(1)

    async def _run_background_task_inner(
        self,
        prompt: str,
        source: "SessionSource",
        task_id: str,
        event_message_id: Optional[str] = None,
        media_urls: Optional[List[str]] = None,
        media_types: Optional[List[str]] = None,
    ) -> None:
        """Execute a background agent task and deliver the result to the chat."""
        # Resolved at call time: gateway.run imports this module, so a
        # module-level import here would be circular.
        from gateway.run import (
            _checkpoint_agent_kwargs,
            _current_max_iterations,
            _load_gateway_config,
            _platform_config_key,
        )

        from run_agent import AIAgent

        media_urls = media_urls or []
        media_types = media_types or []

        adapter = self._adapter_for_source(source)
        if not adapter:
            logger.warning("No adapter for platform %s in background task %s", source.platform, task_id)
            return

        _thread_metadata = self._thread_metadata_for_source(source, event_message_id)

        try:
            user_config = _load_gateway_config()
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source,
                user_config=user_config,
            )
            if not runtime_kwargs.get("api_key"):
                await adapter.send(
                    source.chat_id,
                    f"❌ Background task {task_id} failed: no provider credentials configured.",
                    metadata=_thread_metadata,
                )
                return

            platform_key = _platform_config_key(source.platform)

            from hermes_cli.tools_config import _get_platform_tools
            enabled_toolsets = sorted(_get_platform_tools(user_config, platform_key))
            agent_cfg = user_config.get("agent") or {}
            disabled_toolsets = agent_cfg.get("disabled_toolsets") or None

            pr = self._provider_routing
            max_iterations = _current_max_iterations()
            reasoning_config = self._resolve_session_reasoning_config(
                source=source, model=model
            )
            self._reasoning_config = reasoning_config
            self._service_tier = self._resolve_session_service_tier(source=source)
            turn_route = self._resolve_turn_agent_config(prompt, model, runtime_kwargs)

            # Enrich the prompt with image descriptions so the background
            # agent can see user-attached images (same as the main flow).
            enriched_prompt = prompt
            if media_urls:
                image_paths = []
                for i, path in enumerate(media_urls):
                    mtype = media_types[i] if i < len(media_types) else ""
                    if mtype.startswith("image/"):
                        image_paths.append(path)
                if image_paths:
                    try:
                        enriched_prompt = await self._enrich_message_with_vision(
                            prompt, image_paths,
                        )
                    except Exception as e:
                        logger.warning("Background task vision enrichment failed: %s", e)

            def run_sync():
                agent = AIAgent(
                    model=turn_route["model"],
                    **turn_route["runtime"],
                    **_checkpoint_agent_kwargs(user_config),
                    max_iterations=max_iterations,
                    quiet_mode=True,
                    verbose_logging=False,
                    enabled_toolsets=enabled_toolsets,
                    disabled_toolsets=disabled_toolsets,
                    reasoning_config=reasoning_config,
                    service_tier=self._service_tier,
                    request_overrides=turn_route.get("request_overrides"),
                    providers_allowed=pr.get("only"),
                    providers_ignored=pr.get("ignore"),
                    providers_order=pr.get("order"),
                    provider_sort=pr.get("sort"),
                    provider_require_parameters=pr.get("require_parameters", False),
                    provider_data_collection=pr.get("data_collection"),
                    session_id=task_id,
                    platform=platform_key,
                    user_id=source.user_id,
                    user_id_alt=source.user_id_alt,
                    user_name=source.user_name,
                    chat_id=source.chat_id,
                    chat_name=source.chat_name,
                    chat_type=source.chat_type,
                    thread_id=source.thread_id,
                    session_db=getattr(self._session_db, "_db", self._session_db),
                    # Reload from disk — do not reuse the startup snapshot (#60955).
                    fallback_model=self._refresh_fallback_model(),
                )
                try:
                    return agent.run_conversation(
                        user_message=enriched_prompt,
                        task_id=task_id,
                    )
                finally:
                    self._cleanup_agent_resources(agent)

            result = await self._run_in_executor_with_context(run_sync)

            response = result.get("final_response", "") if result else ""
            if not response and result and result.get("error"):
                response = f"Error: {result['error']}"

            # Extract media files from the response
            if response:
                media_files, response = adapter.extract_media(response)
                from gateway.platforms.base import BasePlatformAdapter
                media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
                images, text_content = adapter.extract_images(response)

                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                header = f'✅ Background task complete\nPrompt: "{preview}"\n\n'

                if text_content:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + text_content,
                        metadata=_thread_metadata,
                    )
                elif not images and not media_files:
                    await adapter.send(
                        chat_id=source.chat_id,
                        content=header + "(No response generated)",
                        metadata=_thread_metadata,
                    )

                # Send extracted images
                for image_url, alt_text in (images or []):
                    try:
                        await adapter.send_image(
                            chat_id=source.chat_id,
                            image_url=image_url,
                            caption=alt_text,
                            metadata=_thread_metadata,
                        )
                    except Exception:
                        pass

                # Send media files, routing each by type so a TTS clip
                # arrives as a voice bubble / a clip as a video rather than
                # a generic document. Mirrors the streaming + kanban paths.
                from gateway.platforms.base import (
                    should_send_media_as_audio as _should_send_media_as_audio,
                )
                _IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
                _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
                for media_path, _is_voice in (media_files or []):
                    _ext = os.path.splitext(media_path)[1].lower()
                    try:
                        if _should_send_media_as_audio(source.platform, _ext, _is_voice):
                            await adapter.send_voice(
                                chat_id=source.chat_id,
                                audio_path=media_path,
                                metadata=_thread_metadata,
                            )
                        elif _ext in _VIDEO_EXTS:
                            await adapter.send_video(
                                chat_id=source.chat_id,
                                video_path=media_path,
                                metadata=_thread_metadata,
                            )
                        elif _ext in _IMAGE_EXTS:
                            await adapter.send_image_file(
                                chat_id=source.chat_id,
                                image_path=media_path,
                                metadata=_thread_metadata,
                            )
                        else:
                            await adapter.send_document(
                                chat_id=source.chat_id,
                                file_path=media_path,
                                metadata=_thread_metadata,
                            )
                    except Exception:
                        pass
            else:
                preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f'✅ Background task complete\nPrompt: "{preview}"\n\n(No response generated)',
                    metadata=_thread_metadata,
                )

        except Exception as e:
            logger.exception("Background task %s failed", task_id)
            try:
                await adapter.send(
                    chat_id=source.chat_id,
                    content=f"❌ Background task {task_id} failed: {e}",
                    metadata=_thread_metadata,
                )
            except Exception:
                pass

    async def _watch_update_progress(
        self,
        poll_interval: float = 2.0,
        stream_interval: float = 4.0,
        timeout: float = 1800.0,
    ) -> None:
        """Watch ``hermes update --gateway``, streaming output + forwarding prompts.

        Polls ``.update_output.txt`` for new content and sends chunks to the
        user periodically.  Detects ``.update_prompt.json`` (written by the
        update process when it needs user input) and forwards the prompt to
        the messenger.  The user's next message is intercepted by
        ``_handle_message`` and written to ``.update_response``.
        """
        # Resolved at call time: gateway.run imports this module, so a
        # module-level import here would be circular.
        from gateway.run import (
            _hermes_home,
            _non_conversational_metadata,
        )

        pending_path = _hermes_home / ".update_pending.json"
        claimed_path = _hermes_home / ".update_pending.claimed.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"
        prompt_path = _hermes_home / ".update_prompt.json"

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout

        # Resolve the adapter and chat_id for sending messages
        adapter = None
        chat_id = None
        session_key = None
        metadata = None
        for path in (claimed_path, pending_path):
            if path.exists():
                try:
                    pending = json.loads(path.read_text(encoding="utf-8"))
                    platform_str = pending.get("platform")
                    chat_id = pending.get("chat_id")
                    chat_type = pending.get("chat_type")
                    session_key = pending.get("session_key")
                    thread_id = pending.get("thread_id")
                    message_id = pending.get("message_id")
                    if platform_str and chat_id:
                        platform = Platform(platform_str)
                        adapter = self.adapters.get(platform)
                        metadata = self._thread_metadata_for_target(
                            platform,
                            chat_id,
                            thread_id,
                            chat_type=chat_type,
                            reply_to_message_id=message_id,
                            adapter=adapter,
                        )
                        # Fallback session key if not stored (old pending files)
                        if not session_key:
                            session_key = f"{platform_str}:{chat_id}"
                    break
                except Exception:
                    pass

        if not adapter or not chat_id:
            logger.warning("Update watcher: cannot resolve adapter/chat_id, falling back to completion-only")
            # Fall back to completion-only: wait for the exit code and send the
            # final notification. _send_update_notification re-resolves the
            # adapter on every call, so when the target platform is still
            # reconnecting it returns False and keeps the markers. Keep polling
            # until it actually delivers (returns True) instead of giving up
            # after the first completion check — otherwise a platform that
            # reconnects a few seconds after completion never gets notified.
            while (pending_path.exists() or claimed_path.exists()) and loop.time() < deadline:
                if exit_code_path.exists() and await self._send_update_notification():
                    return
                await asyncio.sleep(poll_interval)
            if (pending_path.exists() or claimed_path.exists()) and not exit_code_path.exists():
                exit_code_path.write_text("124", encoding="utf-8")
                await self._send_update_notification()
            return

        def _strip_ansi(text: str) -> str:
            from tools.ansi_strip import strip_ansi
            return strip_ansi(text)

        bytes_sent = 0
        last_stream_time = loop.time()
        buffer = ""

        async def _flush_buffer() -> None:
            """Send buffered output to the user."""
            nonlocal buffer, last_stream_time
            if not buffer.strip():
                buffer = ""
                return
            # Chunk to fit message limits (Telegram: 4096, others: generous)
            clean = _strip_ansi(buffer).strip()
            buffer = ""
            last_stream_time = loop.time()
            if not clean:
                return
            # Split into chunks if too long
            max_chunk = 3500
            chunks = [clean[i:i + max_chunk] for i in range(0, len(clean), max_chunk)]
            for chunk in chunks:
                try:
                    await adapter.send(
                        chat_id,
                        f"```\n{chunk}\n```",
                        metadata=_non_conversational_metadata(metadata, platform=platform),
                    )
                except Exception as e:
                    logger.debug("Update stream send failed: %s", e)

        while loop.time() < deadline:
            # Check for completion
            if exit_code_path.exists():
                # Read any remaining output
                if output_path.exists():
                    try:
                        content = output_path.read_text(encoding="utf-8")
                        if len(content) > bytes_sent:
                            buffer += content[bytes_sent:]
                            bytes_sent = len(content)
                    except OSError:
                        pass
                await _flush_buffer()

                # Send final status
                try:
                    exit_code_raw = exit_code_path.read_text(encoding="utf-8").strip() or "1"
                    exit_code = int(exit_code_raw)
                    if exit_code == 0:
                        await adapter.send(
                            chat_id,
                            "✅ Hermes update finished.",
                            metadata=_non_conversational_metadata(metadata, platform=platform),
                        )
                    else:
                        await adapter.send(
                            chat_id,
                            "❌ Hermes update failed (exit code {}).".format(exit_code),
                            metadata=_non_conversational_metadata(metadata, platform=platform),
                        )
                    logger.info("Update finished (exit=%s), notified %s", exit_code, session_key)
                except Exception as e:
                    logger.warning("Update final notification failed: %s", e)

                # Cleanup
                for p in (pending_path, claimed_path, output_path,
                          exit_code_path, prompt_path):
                    p.unlink(missing_ok=True)
                (_hermes_home / ".update_response").unlink(missing_ok=True)
                _up_done = self._peek_session_state(session_key)
                if _up_done is not None:
                    _up_done.persistent.update_prompt_pending = False
                return

            # Check for new output
            if output_path.exists():
                try:
                    content = output_path.read_text(encoding="utf-8")
                    if len(content) > bytes_sent:
                        buffer += content[bytes_sent:]
                        bytes_sent = len(content)
                except OSError:
                    pass

            # Flush buffer periodically
            if buffer.strip() and (loop.time() - last_stream_time) >= stream_interval:
                await _flush_buffer()

            # Check for prompts — only forward if we haven't already sent
            # one that's still awaiting a response.  Without this guard the
            # watcher would re-read the same .update_prompt.json every poll
            # cycle and spam the user with duplicate prompt messages.
            _up_pending_state = (
                self._peek_session_state(session_key) if session_key else None
            )
            if (prompt_path.exists() and session_key
                    and not (
                        _up_pending_state is not None
                        and _up_pending_state.persistent.update_prompt_pending
                    )):
                try:
                    prompt_data = json.loads(prompt_path.read_text(encoding="utf-8"))
                    prompt_text = prompt_data.get("prompt", "")
                    default = prompt_data.get("default", "")
                    if prompt_text:
                        # Flush any buffered output first so the user sees
                        # context before the prompt
                        await _flush_buffer()
                        # Try platform-native buttons first (Discord, Telegram)
                        sent_buttons = False
                        if getattr(type(adapter), "send_update_prompt", None) is not None:
                            try:
                                await adapter.send_update_prompt(
                                    chat_id=chat_id,
                                    prompt=prompt_text,
                                    default=default,
                                    session_key=session_key,
                                    metadata=_non_conversational_metadata(metadata, platform=platform),
                                )
                                sent_buttons = True
                            except Exception as btn_err:
                                logger.debug("Button-based update prompt failed: %s", btn_err)
                        if not sent_buttons:
                            default_hint = f" (default: {default})" if default else ""
                            _p = getattr(adapter, "typed_command_prefix", "/")
                            await adapter.send(
                                chat_id,
                                f"⚕ **Update needs your input:**\n\n"
                                f"{prompt_text}{default_hint}\n\n"
                                f"Reply `{_p}approve` (yes) or `{_p}deny` (no), "
                                f"or type your answer directly.",
                                metadata=_non_conversational_metadata(metadata, platform=platform),
                            )
                        # Keep the prompt marker on disk until the user
                        # answers. If the gateway restarts mid-prompt, the
                        # next watcher can recover by re-forwarding it from
                        # disk. Duplicate sends in the same process are
                        # still suppressed by _update_prompt_pending.
                        self._session_state(
                            session_key
                        ).persistent.update_prompt_pending = True
                        # .update_response to continue — it doesn't re-check
                        logger.info("Forwarded update prompt to %s: %s", session_key, prompt_text[:80])
                except (json.JSONDecodeError, OSError) as e:
                    logger.debug("Failed to read update prompt: %s", e)

            await asyncio.sleep(poll_interval)

        # Timeout
        if not exit_code_path.exists():
            logger.warning("Update watcher timed out after %.0fs", timeout)
            exit_code_path.write_text("124", encoding="utf-8")
            await _flush_buffer()
            try:
                await adapter.send(
                    chat_id,
                    "❌ Hermes update timed out after 30 minutes.",
                    metadata=_non_conversational_metadata(metadata, platform=platform),
                )
            except Exception:
                pass
            for p in (pending_path, claimed_path, output_path,
                      exit_code_path, prompt_path):
                p.unlink(missing_ok=True)
            (_hermes_home / ".update_response").unlink(missing_ok=True)
            _up_timeout_state = self._peek_session_state(session_key)
            if _up_timeout_state is not None:
                _up_timeout_state.persistent.update_prompt_pending = False

    async def _run_process_watcher(self, watcher: dict) -> None:
        """
        Periodically check a background process and push updates to the user.

        Runs as an asyncio task. Stays silent when nothing changed.
        Auto-removes when the process exits or is killed.

        Notification mode (from ``display.background_process_notifications``):
          - ``all``    — running-output updates + final message
          - ``result`` — final completion message only
          - ``error``  — final message only when exit code != 0
          - ``off``    — no messages at all
        """
        # Resolved at call time: gateway.run imports this module, so a
        # module-level import here would be circular.
        from gateway.run import (
            _non_conversational_metadata,
            _redact_gateway_user_facing_secrets,
        )

        from tools.process_registry import process_registry

        session_id = watcher["session_id"]
        interval = watcher["check_interval"]
        session_key = watcher.get("session_key", "")
        platform_name = watcher.get("platform", "")
        chat_id = watcher.get("chat_id", "")
        thread_id = watcher.get("thread_id", "")
        user_id = watcher.get("user_id", "")
        user_name = watcher.get("user_name", "")
        message_id = str(watcher.get("message_id") or "").strip() or None
        agent_notify = watcher.get("notify_on_complete", False)
        notify_mode = self._load_background_notifications_mode()

        logger.debug("Process watcher started: %s (every %ss, notify=%s, agent_notify=%s)",
                      session_id, interval, notify_mode, agent_notify)

        if notify_mode == "off" and not agent_notify:
            # Still wait for the process to exit so we can log it, but don't
            # push any messages to the user.
            while True:
                await asyncio.sleep(interval)
                session = process_registry.get(session_id)
                if session is None or session.exited:
                    break
            logger.debug("Process watcher ended (silent): %s", session_id)
            return

        last_output_len = 0
        while True:
            await asyncio.sleep(interval)

            session = process_registry.get(session_id)
            if session is None:
                break

            current_output_len = len(session.output_buffer)
            has_new_output = current_output_len > last_output_len
            last_output_len = current_output_len

            if session.exited:
                # --- Agent-triggered completion: inject synthetic message ---
                # Skip if the agent already consumed the result via wait/log.
                # poll() is read-only and intentionally does NOT mark consumed
                # (#10156) — a status check must not suppress this delivery turn.
                from tools.process_registry import format_process_notification, process_registry as _pr_check
                if agent_notify and not _pr_check.is_completion_consumed(session_id):
                    from agent.redact import redact_terminal_output
                    from tools.ansi_strip import strip_ansi
                    _command = getattr(session, "command", "") or ""
                    _raw = strip_ansi(session.output_buffer) if session.output_buffer else ""
                    _raw = redact_terminal_output(_raw, _command)
                    _command = _redact_gateway_user_facing_secrets(_command)
                    # Truncate at line boundaries so notifications never start
                    # mid-line (fixes #23284). Keep the last ~2000 chars but
                    # snap to the nearest preceding newline, then prepend a
                    # truncation marker when output was cut.
                    _LIMIT = 2000
                    if len(_raw) > _LIMIT:
                        _tail = _raw[-_LIMIT:]
                        _nl = _tail.find("\n")
                        _tail = _tail[_nl + 1:] if _nl != -1 else _tail
                        _out = f"[… output truncated — showing last {len(_tail)} chars]\n{_tail}"
                    else:
                        _out = _raw
                    completion_evt = {
                        "type": "completion",
                        "session_id": session_id,
                        "session_key": session_key,
                        "platform": platform_name,
                        "chat_type": watcher.get("chat_type", ""),
                        "chat_id": chat_id,
                        "thread_id": thread_id,
                        "user_id": user_id,
                        "user_name": user_name,
                        "message_id": message_id,
                        "started_at": getattr(session, "started_at", None),
                        "command": _command,
                        "exit_code": session.exit_code,
                        "completion_reason": getattr(session, "completion_reason", "exited"),
                        "termination_source": getattr(session, "termination_source", ""),
                        "output": _out,
                    }
                    synth_text = format_process_notification(completion_evt)
                    if not synth_text:
                        break
                    delivered = await self._deliver_completion_notification(
                        synth_text, completion_evt,
                    )
                    if delivered is False:
                        # The process remains terminal; retry after failed
                        # adapter injection instead of suppressing the result.
                        continue
                    break

                # --- Normal text-only notification ---
                # Skip when the agent already consumed this completion via
                # wait/log (#65379): process(wait) returned the exit code and
                # output inline, so the raw "[Background process ... finished
                # with exit code ...]" message would be a duplicate delivery
                # of the same completion. The agent_notify branch above
                # already honors _completion_consumed; without this check its
                # skip FALLS THROUGH to this block and re-delivers the output
                # the agent is actively summarizing. poll() is read-only and
                # intentionally does not mark consumed (#10156), so a status
                # check never suppresses this message.
                if _pr_check.is_completion_consumed(session_id):
                    logger.debug(
                        "Process watcher: completion for %s already consumed "
                        "via wait/log — skipping raw notification (#65379)",
                        session_id,
                    )
                    break
                # Decide whether to notify based on mode
                should_notify = (
                    notify_mode in {"all", "result"}
                    or (notify_mode == "error" and session.exit_code not in {0, None})
                )
                if should_notify:
                    new_output = session.output_buffer[-1000:] if session.output_buffer else ""
                    if new_output:
                        from agent.redact import redact_terminal_output
                        new_output = redact_terminal_output(
                            new_output, getattr(session, "command", "") or ""
                        )
                    message_text = (
                        f"[Background process {session_id} finished with exit code {session.exit_code}~ "
                        f"Here's the final output:\n{new_output}]"
                    )
                    adapter = None
                    for p, a in self.adapters.items():
                        if p.value == platform_name:
                            adapter = a
                            break
                    if adapter and chat_id:
                        try:
                            send_meta = {"thread_id": thread_id} if thread_id else None
                            await adapter.send(
                                chat_id,
                                message_text,
                                metadata=_non_conversational_metadata(send_meta, platform=platform_name),
                            )
                        except Exception as e:
                            logger.error("Watcher delivery error: %s", e)
                break

            elif has_new_output and notify_mode == "all" and not agent_notify:
                # New output available -- deliver status update (only in "all" mode)
                # Skip periodic updates for agent_notify watchers (they only care about completion)
                new_output = session.output_buffer[-500:] if session.output_buffer else ""
                if new_output:
                    from agent.redact import redact_terminal_output
                    new_output = redact_terminal_output(
                        new_output, getattr(session, "command", "") or ""
                    )
                message_text = (
                    f"[Background process {session_id} is still running~ "
                    f"New output:\n{new_output}]"
                )
                adapter = None
                for p, a in self.adapters.items():
                    if p.value == platform_name:
                        adapter = a
                        break
                if adapter and chat_id:
                    try:
                        send_meta = {"thread_id": thread_id} if thread_id else None
                        await adapter.send(
                            chat_id,
                            message_text,
                            metadata=_non_conversational_metadata(send_meta, platform=platform_name),
                        )
                    except Exception as e:
                        logger.error("Watcher delivery error: %s", e)

        logger.debug("Process watcher ended: %s", session_id)
