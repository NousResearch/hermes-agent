"""Telegram lifecycle methods; runtime dependencies remain on the adapter facade."""

import asyncio
from typing import Any, Dict, Optional
try:
    from telegram import Message, Update
    from telegram.ext import ContextTypes
except ImportError:
    Message = Update = Any
    class ContextTypes:
        DEFAULT_TYPE = Any


class TelegramLifecycleMixin:
    async def _drain_polling_connections(self) -> None:
        """Reset the httpx pool used for getUpdates polling before a reconnect.

        Half-closed connections (esp. via proxies) occupy pool slots until "Pool timeout: All connections in the connection pool
        are occupied". Only ``_request[0]`` (getUpdates) is reset; the general request stays untouched so concurrent sends are
        never interrupted. Relies on PTB 22.x's private ``(get_updates, general)`` tuple — review on PTB 23+."""
        from . import adapter as _adapter

        if not (self._app and self._app.bot):
            return
        try:
            polling_req = self._app.bot._request[0]  # noqa: SLF001
        except Exception:
            return
        # Bounded wall-clock deadline (not asyncio.wait_for): httpcore's pool close runs under
        # AsyncShieldCancellation and a wedged CLOSE-WAIT socket can hang it forever.
        if not await self._bounded_request_step(polling_req.shutdown(), "Polling request shutdown failed/timed out (non-fatal)"):
            # initialize() only rebuilds the client when ``client.is_closed``; an abandoned aclose()
            # leaves it false, so start_polling would reuse the CLOSE-WAIT socket (alive but deaf).
            # Swap in a fresh client before initialize(). See #87057.
            self._orphan_and_rebuild_polling_client(polling_req)
        if await self._bounded_request_step(polling_req.initialize(), "Polling request re-initialize failed/timed out (non-fatal)"):
            _adapter.logger.debug("[%s] Polling request pool drained before reconnect", self.name)
        else:
            self._orphan_and_rebuild_polling_client(polling_req)

    async def _bounded_request_step(self, awaitable, failure_msg: str) -> bool:
        """Await a request shutdown()/initialize() under ``_DRAIN_TIMEOUT``; False (debug-logged) on failure."""
        from . import adapter as _adapter

        try:
            await _adapter._await_with_thread_deadline(awaitable, timeout=_adapter._DRAIN_TIMEOUT)
            return True
        except Exception:
            _adapter.logger.debug("[%s] " + failure_msg, self.name, exc_info=True)
            return False

    def _orphan_and_rebuild_polling_client(self, polling_req) -> None:
        """Replace a wedged HTTPXRequest client after a hung aclose(): swap in a fresh client and close
        the old one in a detached, bounded task so it can't block the reconnect ladder.

        PTB's ``HTTPXRequest.initialize()`` only calls ``_build_client()`` when the current client reports
        ``is_closed``. If ``shutdown()`` was abandoned on a CLOSE-WAIT socket, that flag stays false and the
        next ``start_polling()`` reuses the dead getUpdates connection (#87057).
        """
        from . import adapter as _adapter

        old = getattr(polling_req, "_client", None)
        build = getattr(polling_req, "_build_client", None)
        if old is None or not callable(build) or getattr(old, "is_closed", True):
            return
        try:
            polling_req._client = build()  # noqa: SLF001
        except Exception:
            _adapter.logger.debug("[%s] Failed to rebuild polling HTTP client after hung drain", self.name, exc_info=True)
            return
        _adapter.logger.warning("[%s] Replaced wedged getUpdates HTTP client after drain timeout (likely CLOSE-WAIT socket)", self.name)

        async def _orphan_aclose() -> None:
            try:
                aclose = getattr(old, "aclose", None)
                if not callable(aclose):
                    return
                # Same cancellation-swallowing httpcore scope as shutdown(): wall-clock deadline.
                await _adapter._await_with_thread_deadline(aclose(), timeout=_adapter._DRAIN_TIMEOUT)
            except Exception:
                _adapter.logger.debug("[%s] Orphan polling client aclose failed (non-fatal)", self.name, exc_info=True)

        try:
            task = _adapter.asyncio.ensure_future(_orphan_aclose())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            task.add_done_callback(_adapter._consume_abandoned_task)
        except Exception:
            pass

    def _fence_polling(self) -> None:
        """Mark polling closed: no progress accepted, send path degraded."""
        self._polling_progress_accepting = False
        self._send_path_degraded = True

    def _begin_polling_generation(self) -> tuple[int, asyncio.Event]:
        """Start accepting progress for a new getUpdates polling generation."""
        from . import adapter as _adapter

        if self._teardown_started:
            self._fence_polling()
            progress = getattr(self, "_polling_progress_event", None)
            if progress is None:
                progress = self._polling_progress_event = _adapter.asyncio.Event()
            return getattr(self, "_polling_generation", 0), progress
        verifier = getattr(self, "_polling_progress_verifier_task", None)
        if verifier is not None and not verifier.done():
            verifier.cancel()
        self._polling_progress_verifier_task = None
        self._polling_generation = getattr(self, "_polling_generation", 0) + 1
        self._polling_progress_event = _adapter.asyncio.Event()
        self._polling_progress_accepting = True
        self._send_path_degraded = True
        # Reset stall-watchdog timestamps: no proven progress yet, age measured from here.
        # See #92991.
        self._polling_generation_started_monotonic = _adapter.time.monotonic()
        self._polling_last_progress_monotonic = None
        return self._polling_generation, self._polling_progress_event

    def _record_polling_progress(self, generation: int) -> None:
        """Record successful getUpdates I/O for the current generation only."""
        from . import adapter as _adapter

        if self._teardown_started or not self._polling_progress_accepting or generation != self._polling_generation:
            return
        if not self._polling_progress_event.is_set():
            # First confirmed round-trip resolves the "health pending" line both reconnect paths end on.
            _adapter.logger.info("[%s] Telegram polling confirmed healthy: getUpdates progressing (generation %d)", self.name, generation)
        self._polling_progress_event.set()
        self._polling_last_progress_monotonic = _adapter.time.monotonic()
        self._polling_network_error_count = 0
        if generation == self._polling_conflict_recovery_generation:
            self._polling_conflict_recovery_generation = None
        else:
            self._polling_conflict_count = 0
        # First proof getUpdates is flowing for this generation: flip a
        # published "retrying" (degraded connect, reconnect stamp, or the
        # mid-session recovery below) back to "connected" (#101391).
        if self._send_path_degraded and getattr(self, "_running", False) and not self.has_fatal_error:
            self._write_runtime_status_safe(
                "connected", platform_state="connected", error_code=None, error_message=None,
            )
        self._send_path_degraded = False

    def _observe_polling_request_result(self, request, generation, result):
        """Record getUpdates progress from an observed do_request result (purely observational: PTB still
        parses the untouched payload and owns any resulting exception)."""
        from . import adapter as _adapter

        status_code, payload = result
        if generation is None or not (200 <= status_code < 300):
            return
        try:
            # The request's own parser keeps health observation in agreement with PTB.
            envelope = request.parse_json_payload(payload)
        except Exception:
            return
        if isinstance(envelope, dict) and envelope.get("ok") is True and "result" in envelope:
            self._record_polling_progress(generation)

    def _instrument_polling_request(self, request):
        """Instrument one dedicated PTB getUpdates request with progress tracking.

        PTB request classes use ``__slots__`` (no ``__dict__`` on 3.13), so re-tag the instance to a thin ``__slots__ = ()``
        subclass overriding ``do_request`` — identical layout makes the ``__class__`` swap legal; works for test doubles too.

        On Python 3.13 their instances no longer carry a ``__dict__`` (the ``AbstractAsyncContextManager``
        MRO stopped yielding one), so ``request.do_request = wrapper`` raises ``AttributeError:
        'HTTPXRequest' object attribute 'do_request' is read-only`` and the whole Telegram connect fails
        (#64482). It only appeared to work on Python 3.12, where those instances still had a ``__dict__``.
        """
        from . import adapter as _adapter

        adapter = self

        class _InstrumentedPollingRequest(type(request)):
            __slots__ = ()

            async def do_request(self, *args, **kwargs):
                generation = _adapter._POLLING_GENERATION_CONTEXT.get()
                result = await super().do_request(*args, **kwargs)
                adapter._observe_polling_request_result(self, generation, result)
                return result

        request.__class__ = _InstrumentedPollingRequest
        return request

    async def _start_polling_once(
        self, app, *, drop_pending_updates: bool, error_callback, abandon_app_on_timeout: bool = False,
        schedule_verifier: bool = True) -> tuple[int, asyncio.Event]:
        """Start one generation and verify real getUpdates progress. Returns this generation's
        ``(generation, progress_event)`` so readiness-gating callers bind to exactly it."""
        from . import adapter as _adapter

        if self._teardown_started:
            raise _adapter._PollingLifecycleAbort("Telegram polling teardown started")
        generation, progress = self._begin_polling_generation()
        if not self._polling_progress_accepting:
            raise _adapter._PollingLifecycleAbort("Telegram polling teardown started")

        def _generation_error_callback(error: Exception) -> None:
            if self._teardown_started or generation != self._polling_generation or error_callback is None:
                return
            callback_context_token = _adapter._POLLING_GENERATION_CONTEXT.set(None)
            try:
                error_callback(error)
            finally:
                _adapter._POLLING_GENERATION_CONTEXT.reset(callback_context_token)

        context_token = _adapter._POLLING_GENERATION_CONTEXT.set(generation)
        try:
            # asyncio.wait_for can wait forever on httpcore/AnyIO shielded scopes; use the wall-deadline
            # helper and abandon the partial updater (caller rebuilds).
            await _adapter._await_with_thread_deadline(
                app.updater.start_polling(
                    allowed_updates=_adapter.Update.ALL_TYPES, drop_pending_updates=drop_pending_updates, error_callback=_generation_error_callback),
                timeout=_adapter._UPDATER_START_TIMEOUT,
                on_abandon=((lambda app=app: _adapter._shutdown_abandoned_app(app)) if abandon_app_on_timeout else None))
        finally:
            _adapter._POLLING_GENERATION_CONTEXT.reset(context_token)
        if self._teardown_started:
            self._fence_polling()
            raise _adapter._PollingLifecycleAbort("Telegram polling teardown started")
        if schedule_verifier:
            self._schedule_polling_progress_verifier(generation, progress)
        return generation, progress

    def _schedule_polling_progress_verifier(self, generation: int, progress: asyncio.Event) -> None:
        """Own exactly one tracked verifier for the current generation."""
        from . import adapter as _adapter

        if self._teardown_started:
            self._fence_polling()
            return
        previous = getattr(self, "_polling_progress_verifier_task", None)
        if previous is not None and not previous.done():
            previous.cancel()
        task = _adapter.asyncio.get_running_loop().create_task(self._verify_polling_after_reconnect(generation, progress))
        self._polling_progress_verifier_task = task
        self._background_tasks.add(task)

        def _clear_finished_verifier(finished: asyncio.Task) -> None:
            self._background_tasks.discard(finished)
            if self._polling_progress_verifier_task is finished:
                self._polling_progress_verifier_task = None

        task.add_done_callback(_clear_finished_verifier)

    def _get_general_request_drain_lock(self) -> asyncio.Lock:
        from . import adapter as _adapter

        lock = getattr(self, "_general_request_drain_lock", None)
        if lock is None:
            lock = self._general_request_drain_lock = _adapter.asyncio.Lock()
        return lock

    async def _drain_general_connections_after_pool_timeout(self) -> None:
        """Reset the general Bot API pool (``_request[1]``) after a confirmed send pool timeout — PTB
        guarantees the request was not sent, so resetting before retrying is safe."""
        from . import adapter as _adapter

        bot = getattr(getattr(self, "_app", None), "bot", None)
        if bot is None:
            bot = getattr(self, "_bot", None)
        if bot is None:
            return
        try:
            general_req = bot._request[1]  # noqa: SLF001
        except Exception:
            return
        async with self._get_general_request_drain_lock():
            await self._bounded_request_step(
                general_req.shutdown(), "General request shutdown failed/timed out after pool timeout (non-fatal)")
            if await self._bounded_request_step(
                general_req.initialize(), "General request re-initialize failed/timed out after pool timeout (non-fatal)"):
                _adapter.logger.warning("[%s] General request pool drained after Telegram pool timeout", self.name)

    def _spawn_polling_recovery(self, loop, coro) -> None:
        """Start ``coro`` as the tracked in-flight recovery task (reentrancy guard)."""
        self._polling_error_task = loop.create_task(coro)
        self._background_tasks.add(self._polling_error_task)
        self._polling_error_task.add_done_callback(self._background_tasks.discard)

    def _recovery_in_flight(self) -> bool:
        from . import adapter as _adapter

        return bool(self._polling_error_task and not self._polling_error_task.done())

    def _schedule_polling_recovery(self, error: Exception, *, reason: str) -> None:
        """Schedule background polling recovery without failing gateway startup: a transient bootstrap
        failure degrades only this adapter; the reconnect ladder recovers in the background."""
        from . import adapter as _adapter

        if self._teardown_started or self.has_fatal_error:
            return
        if self._recovery_in_flight():
            _adapter.logger.debug(
                "[%s] Telegram polling recovery already scheduled; ignoring %s: %s", self.name, reason, _adapter._redact_telegram_error_text(error))
            return
        self._send_path_degraded = True
        # Polling died mid-session on an adapter that published "connected"
        # at connect time. Without this, gateway_state.json keeps saying
        # connected for as long as the recovery ladder runs (#101391: 11 h).
        if getattr(self, "_running", False):
            self._mark_degraded()
        _adapter.logger.warning(
            "[%s] Telegram polling degraded (%s); gateway stays alive and will retry. Error: %s", self.name, reason,
            _adapter._redact_telegram_error_text(error))
        self._spawn_polling_recovery(_adapter.asyncio.get_running_loop(), self._handle_polling_network_error(error))

    async def _delete_webhook_best_effort(self, *, require_success: bool = False) -> bool:
        """Clear a stale webhook; ``require_success`` (cold start) raises so GatewayRunner disposes the
        partial adapter, while reconnects recover transient errors in background."""
        from . import adapter as _adapter

        if not self._bot:
            return False
        delete_webhook = getattr(self._bot, "delete_webhook", None)
        if not callable(delete_webhook):
            return True
        try:
            # Same shielded-cancellation class as initialize/start_polling: never let it pin connect.
            await _adapter._await_with_thread_deadline(delete_webhook(drop_pending_updates=False), timeout=_adapter._UPDATER_START_TIMEOUT)
            return True
        except Exception as err:
            if not self._looks_like_network_error(err):
                raise
            if require_success:
                raise OSError("Telegram deleteWebhook did not complete during initial connect") from err
            _adapter.logger.warning(
                "[%s] deleteWebhook failed with a recoverable network error; continuing to polling so getUpdates/retry can recover: %s",
                self.name, _adapter._redact_telegram_error_text(err))
            self._send_path_degraded = True
            return False

    async def _await_cold_start_readiness(self, progress: asyncio.Event, strict_error_event: asyncio.Event, strict_error: list) -> None:
        """Cold start: wait for THIS generation's first getUpdates success or the first polling error;
        raises OSError so GatewayRunner disposes the partial adapter and retries fresh."""
        from . import adapter as _adapter

        progress_wait = _adapter.asyncio.ensure_future(progress.wait())
        error_wait = _adapter.asyncio.ensure_future(strict_error_event.wait())
        try:
            # Losers are NOT cancelled here; the finally below does it.
            await _adapter._await_with_thread_deadline(
                _adapter.asyncio.wait({progress_wait, error_wait}, return_when=_adapter.asyncio.FIRST_COMPLETED), timeout=_adapter._INITIAL_POLLING_PROGRESS_TIMEOUT)
        except _adapter.asyncio.TimeoutError as exc:
            raise OSError(
                "Telegram getUpdates made no progress within "
                f"{_adapter._INITIAL_POLLING_PROGRESS_TIMEOUT:.0f}s during initial "
                "connect — failing startup so the gateway retries with a fresh adapter (#67498)"
           ) from exc
        finally:
            for fut in (progress_wait, error_wait):
                if not fut.done():
                    fut.cancel()
            await _adapter.asyncio.gather(progress_wait, error_wait, return_exceptions=True)
        if strict_error and not progress.is_set():
            raise OSError(
                "Telegram polling errored before first getUpdates success during initial connect: "
                f"{_adapter._redact_telegram_error_text(strict_error[0])}"
           ) from strict_error[0]
        if not progress.is_set():
            raise OSError("Telegram getUpdates did not become ready during initial connect")

    async def _start_polling_resilient(self, *, drop_pending_updates: bool, error_callback, require_progress: bool = False) -> bool:
        """Start PTB polling; ``require_progress`` (initial connect) demands real readiness. Reconnects
        may recover in background; on cold start a bootstrap failure raises (see _await_cold_start_readiness)."""
        from . import adapter as _adapter

        if self._teardown_started:
            return False
        if not (self._app and self._app.updater):
            raise RuntimeError("Telegram application/updater not initialized")
        # Strict cold start: background recovery must not run while the readiness gate waits, else a G1
        # error starts G2 on the same partial app and GatewayRunner never disposes it.
        strict_error: list[BaseException] = []
        strict_error_event = _adapter.asyncio.Event()
        strict_gate_open = True
        effective_callback = error_callback
        if require_progress:
            loop = _adapter.asyncio.get_running_loop()

            def _strict_error_callback(error: Exception) -> None:
                # Once the gate closes, delegate so later errors still reach background recovery.
                if not strict_gate_open:
                    if error_callback is not None:
                        error_callback(error)
                    return
                if not strict_error:
                    strict_error.append(error)
                # Called from the polling task; set on the loop to wake the strict waiter.
                loop.call_soon_threadsafe(strict_error_event.set)

            effective_callback = _strict_error_callback
        try:
            # Same watchdog bound as the reconnect ladders; the TimeoutError is an OSError subclass, so
            # the except below classifies it as a network error → background recovery.
            # Same watchdog bound as the reconnect ladders: a wedged httpx connection pool can hang
            # start_polling() forever at bootstrap too (#59614).
            generation, progress = await self._start_polling_once(
                self._app, drop_pending_updates=drop_pending_updates, error_callback=effective_callback,
                abandon_app_on_timeout=require_progress,
                # The strict gate IS the cold-start verifier; a background one would race it.
                schedule_verifier=not require_progress)
            if require_progress:
                await self._await_cold_start_readiness(progress, strict_error_event, strict_error)
                # Readiness proven — close the gate so later errors reach background recovery.
                strict_gate_open = False
                self._polling_error_callback_ref = error_callback
            return True
        except _adapter._PollingLifecycleAbort:
            return False
        except Exception as err:
            if self._teardown_started:
                return False
            if require_progress:
                raise
            if self._looks_like_polling_conflict(err):
                _adapter.logger.warning(
                    "[%s] Telegram polling bootstrap conflict; gateway stays alive while conflict retry runs: %s",
                    self.name, _adapter._redact_telegram_error_text(err))
                self._spawn_polling_recovery(_adapter.asyncio.get_running_loop(), self._handle_polling_conflict(err))
                return False
            if self._looks_like_network_error(err):
                self._schedule_polling_recovery(err, reason="polling bootstrap")
                return False
            raise

    async def _go_fatal_network(self, message: str, log_message: str, *log_args) -> None:
        """Retryable ``telegram_network_error`` fatal + runner handoff (supervisor rebuilds the adapter)."""
        from . import adapter as _adapter

        _adapter.logger.error(log_message, *log_args)
        self._set_fatal_error("telegram_network_error", message, retryable=True)
        await self._handoff_polling_fatal_error()

    async def _stop_updater_or_go_fatal(self, app, what: str) -> bool:
        """Bounded ``updater.stop()`` before a recovery restart; False = went fatal, caller returns.

        Wall-clock deadline, not asyncio.wait_for: a CLOSE-WAIT socket wedges stop() on epoll and PTB/AnyIO shielded cleanup
        hangs wait_for. On timeout the Updater's lifecycle lock may still be held, so rebuild the adapter instead."""
        from . import adapter as _adapter

        try:
            if app and app.updater and app.updater.running:
                try:
                    await _adapter._await_with_thread_deadline(app.updater.stop(), timeout=_adapter._UPDATER_STOP_TIMEOUT)
                except _adapter.asyncio.TimeoutError:
                    message = (
                        f"Telegram updater.stop() did not finish before the {what} deadline; "
                        "rebuilding the adapter instead of reusing an Updater whose lifecycle lock may still be held.")
                    await self._go_fatal_network(message, "[%s] %s (likely CLOSE-WAIT socket)", self.name, message)
                    return False
        except Exception:
            pass
        return True

    def _restart_polling_in_task(self, coro) -> None:
        """Run a recovery coroutine as the tracked in-flight ``_polling_error_task``."""
        from . import adapter as _adapter

        self._polling_error_task = _adapter.asyncio.get_running_loop().create_task(coro)

    async def _handle_polling_network_error(self, error: Exception) -> None:
        """Reconnect polling after a transient network interruption (NetworkError/TimedOut).

        Host connectivity loss (sleep, WiFi switch, VPN) kills the long-poll silently. Exponential back-off (5s→60s
        cap) up to MAX_NETWORK_RETRIES, then retryable-fatal so the supervisor restarts the gateway."""
        from . import adapter as _adapter

        if self._teardown_started or self.has_fatal_error:
            return
        MAX_NETWORK_RETRIES = 10
        BASE_DELAY = 5
        MAX_DELAY = 60
        self._polling_network_error_count += 1
        self._send_path_degraded = True
        attempt = self._polling_network_error_count
        if attempt > MAX_NETWORK_RETRIES:
            message = (
                "Telegram polling could not reconnect after %d network error retries. "
                "Escalating to gateway recovery." % MAX_NETWORK_RETRIES)
            await self._go_fatal_network(message, "[%s] %s Last error: %s", self.name, message, _adapter._redact_telegram_error_text(error))
            return
        delay = min(BASE_DELAY * (2 ** (attempt - 1)), MAX_DELAY)
        _adapter.logger.warning(
            "[%s] Telegram network error (attempt %d/%d), reconnecting in %ds. Error: %s", self.name, attempt,
            MAX_NETWORK_RETRIES, delay, _adapter._redact_telegram_error_text(error))
        await _adapter.asyncio.sleep(delay)
        if self._teardown_started:
            return
        # Stable local ref: a concurrent disconnect() may set self._app = None while we await.
        app = self._app
        # Unguarded stop() on a CLOSE-WAIT socket would leave _polling_error_task perpetually
        # "in-flight" so every probe skips reconnect for hours.
        if not await self._stop_updater_or_go_fatal(app, "network-recovery") or self._teardown_started:
            return
        # start_polling() bootstraps through the *general* pool before getUpdates; a confirmed pool timeout means the request
        # was never sent, so rebuilding that pool is safe. Generic network errors stay polling-only (sends untouched).
        if self._looks_like_pool_timeout(error):
            await self._drain_general_connections_after_pool_timeout()
        if self._teardown_started:
            return
        await self._drain_polling_connections()
        if self._teardown_started:
            return
        try:
            if not app:
                raise RuntimeError("Telegram application was torn down during reconnect")
            await self._start_polling_once(app, drop_pending_updates=False, error_callback=self._polling_error_callback_ref)
            _adapter.logger.info(
                "[%s] Telegram polling restarted after network error (attempt %d); health pending getUpdates progress", self.name, attempt)
        except _adapter._PollingLifecycleAbort:
            return
        except Exception as retry_err:
            if self._teardown_started:
                return
            _adapter.logger.warning("[%s] Telegram polling reconnect failed: %s", self.name, _adapter._redact_telegram_error_text(retry_err))
            # Polling is dead and no more error callbacks will fire — chain the retry ourselves.
            if not self.has_fatal_error and not self._teardown_started:
                task = _adapter.asyncio.ensure_future(self._handle_polling_network_error(retry_err))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                # The chained retry IS the in-flight recovery: it must replace the reentrancy guard.
                self._polling_error_task = task

    async def _polling_heartbeat_loop(self) -> None:
        """Detect dead Telegram TCP sockets (CLOSE-WAIT) by periodic probing.

        In CLOSE-WAIT epoll still reports the long-poll socket readable and nothing raises, so PTB's
        ``error_callback`` never fires. Probe ``get_me()`` on the *general* path (never the getUpdates pool);
        connect-level failures feed ``_handle_polling_network_error``. Runs for the connection's lifetime, catching
        steady-state wedges the one-shot verifier can't."""
        from . import adapter as _adapter

        HEARTBEAT_INTERVAL = 90   # seconds between probes
        PROBE_TIMEOUT = 15        # seconds before declaring the path dead
        # Wedged-recovery watchdog: note when a recovery task is first seen in-flight and force-escalate
        # if the *same* task object still runs past the stuck timeout.
        # Tracked locally so no _polling_error_task assignment site needs to stamp a timestamp: the
        # heartbeat notes when it first observes a given recovery task still in-flight, and force-escalates
        # if the *same* task object is still running after _POLLING_ERROR_TASK_STUCK_TIMEOUT. A healthy
        # ladder attempt completes (task done) or chains to a new task well before then, so a single
        # long-lived task is unambiguously wedged. See #66377.
        stuck_task_ref: _adapter.Optional[_adapter.asyncio.Task] = None
        stuck_task_since = 0.0
        while True:
            try:
                await _adapter.asyncio.sleep(HEARTBEAT_INTERVAL)
                if self._teardown_started or self.has_fatal_error:
                    return
                # A recovery task hung on an unbounded await gates every other recovery path forever
                # (alive but deaf): force retryable-fatal so the reconnector rebuilds the adapter.
                # Independent wedged-recovery watchdog (#66377): if the tracked recovery task has hung (any
                # await no local bound covers), every other recovery path is gated behind it and returns
                # early forever — the gateway stays alive but deaf.
                recovery_task = self._polling_error_task
                if recovery_task is not None and not recovery_task.done():
                    now = _adapter.time.monotonic()
                    if recovery_task is not stuck_task_ref:
                        stuck_task_ref = recovery_task
                        stuck_task_since = now
                    elif now - stuck_task_since > _adapter._POLLING_ERROR_TASK_STUCK_TIMEOUT:
                        stuck_for = now - stuck_task_since
                        _adapter.logger.error(
                            "[%s] Telegram reconnect task wedged for %.0fs with no ladder progress; forcing retryable-fatal so the gateway "
                            "reconnects instead of staying silently deaf.",
                            self.name, stuck_for)
                        with _adapter.contextlib.suppress(Exception):
                            recovery_task.cancel()
                        self._set_fatal_error(
                            "telegram_network_error",
                            "Telegram reconnect task wedged for %.0fs; forcing gateway reconnect." % stuck_for,
                            retryable=True)
                        await self._handoff_polling_fatal_error()
                        return
                else:
                    stuck_task_ref = None
                bot = self._app.bot if self._app else None
                if bot is None:
                    continue
                # No get_me() ⇒ not a live polling client (torn down / test double): exit, don't spin.
                if not callable(getattr(bot, "get_me", None)):
                    return
                await _adapter.asyncio.wait_for(bot.get_me(), PROBE_TIMEOUT)
                # get_me() refreshes PTB's cached bot user: adopt a BotFather rename before routing on it.
                self._bot_identity_checked_at = _adapter.time.monotonic()
                self._note_bot_username(getattr(bot, "username", None))
                # get_me() OK proves only the send path; a wedged long-poll shows as server-side queue.
                # get_me() succeeded — the general/send request path is healthy. That does NOT prove the
                # getUpdates consumer is alive: PTB can report updater.running=True while the long-poll task
                # is wedged, so DMs queue in the Bot API and never reach handlers (#42909). get_me() is
                # blind to this; get_webhook_info() exposes it via pending_update_count. Escalate only after
                # two consecutive probes see a non-zero queue while we believe we're polling, so a single
                # in-flight update (consumed before the next probe) never trips recovery.
                await self._probe_pending_updates(bot, PROBE_TIMEOUT)
                # An empty queue can't hide a wedge forever: no round-trip past the stall threshold ⇒ dead.
                # Even an empty queue cannot hide a wedged long-poll forever: Telegram answers within ~50s,
                # so a consumer with no successful round-trip past the stall threshold is dead (#92991).
                # Pure local-state check — no Bot API call needed.
                await self._check_polling_stall()
            except _adapter.asyncio.CancelledError:
                return
            except (_adapter.asyncio.TimeoutError, OSError) as probe_err:
                self._schedule_polling_recovery(probe_err, reason="heartbeat probe")
            except Exception as probe_err:
                # Non-connectivity errors (e.g. TelegramError 401) aren't CLOSE-WAIT symptoms.
                if self._looks_like_network_error(probe_err):
                    self._schedule_polling_recovery(probe_err, reason="heartbeat probe")

    async def _probe_pending_updates(self, bot, probe_timeout: float) -> None:
        """Detect a wedged or stopped getUpdates consumer via pending_update_count.

        PTB can report ``updater.running`` while the long-poll is stuck; get_me() stays healthy yet DMs queue in the
        Bot API. A stuck queue over two consecutive probes ⇒ dead consumer. Also covers the updater having stopped
        entirely (``running=False``, no reconnect in flight).

        PTB can report ``updater.running == True`` while its long-poll task is silently stuck (e.g. a socket
        that epoll keeps reporting readable on WSL2). ``get_me()`` stays healthy because it uses the general
        request path, so the CLOSE-WAIT heartbeat never fires — yet DMs queue in the Bot API and never reach
        handlers (#42909).
        We detect the stopped updater directly and feed the same ladder (#55769).
        """
        # Polling mode only: in webhook mode Telegram pushes and holds no server-side queue.
        from . import adapter as _adapter

        if self._teardown_started or self._webhook_mode:
            return
        # An in-flight reconnect owns recovery — don't double-trigger, and don't misread its brief
        # stop()->start_polling() window (updater.running transiently False) as dead.
        if self._recovery_in_flight():
            self._polling_not_running_count = 0
            return
        updater = getattr(self._app, "updater", None) if self._app else None
        if updater is None:
            self._polling_pending_stuck_count = 0
            return
        if not getattr(updater, "running", False):
            # Long-poll task gone, general-path calls still succeed, so no error_callback/probe ever
            # fires. Debounced over two probes so a just-starting updater never trips it.
            self._polling_pending_stuck_count = 0
            # We are in polling mode with no reconnect in flight, yet PTB's updater has stopped entirely.
            # This is distinct from the wedged-but-running consumer handled below: the long-poll task is
            # gone, get_me()/get_webhook_info() on the general request path still succeed, so no
            # error_callback or connectivity probe ever fires and the gateway silently stops receiving
            # messages while the process stays alive (#55769).
            self._polling_not_running_count += 1
            _adapter.logger.warning(
                "[%s] Telegram polling heartbeat: updater stopped while in polling mode (stuck probe %d/2)", self.name,
                self._polling_not_running_count)
            if self._polling_not_running_count >= 2:
                self._polling_not_running_count = 0
                self._escalate_stuck_consumer(
                    "[%s] Telegram updater is not running (long-poll task gone); triggering polling restart",
                    "Telegram updater stopped while in polling mode")
            return
        self._polling_not_running_count = 0
        get_webhook_info = getattr(bot, "get_webhook_info", None)
        if not callable(get_webhook_info):
            return
        try:
            info = await _adapter.asyncio.wait_for(get_webhook_info(), probe_timeout)  # type: ignore[arg-type]
        except (_adapter.asyncio.TimeoutError, OSError):
            return  # connectivity symptom for the get_me() path, not a stuck-queue signal
        pending = int(getattr(info, "pending_update_count", 0) or 0)
        if pending <= 0:
            self._polling_pending_stuck_count = 0
            return
        self._polling_pending_stuck_count += 1
        _adapter.logger.warning(
            "[%s] Telegram polling heartbeat: %d update(s) queued but not consumed (stuck probe %d/2)", self.name,
            pending, self._polling_pending_stuck_count)
        if self._polling_pending_stuck_count >= 2:
            self._polling_pending_stuck_count = 0
            self._escalate_stuck_consumer(
                "[%s] getUpdates consumer appears wedged (queue not draining); triggering polling restart",
                "getUpdates consumer wedged: pending updates not draining")

    def _escalate_stuck_consumer(self, log_message: str, reason: str) -> None:
        """Second consecutive stuck probe: restart polling via the network-error ladder (unless tearing down)."""
        from . import adapter as _adapter

        if self._teardown_started:
            return
        _adapter.logger.warning(log_message, self.name)
        self._polling_error_task = _adapter.asyncio.get_running_loop().create_task(self._handle_polling_network_error(RuntimeError(reason)))

    async def _check_polling_stall(self) -> None:
        """Watchdog the last successful getUpdates round-trip: a long-poll can wedge without raising
        (CLOSE-WAIT after a route flip) while every other probe stays blind; no round-trip for
        ``_POLLING_STALL_TIMEOUT`` ⇒ escalate through the bounded reconnect ladder.

        See #92991.
        """
        from . import adapter as _adapter

        if self._webhook_mode or self._teardown_started or self.has_fatal_error or self._recovery_in_flight():
            return
        now = _adapter.time.monotonic()
        last_progress = getattr(self, "_polling_last_progress_monotonic", None)
        generation_started = getattr(self, "_polling_generation_started_monotonic", None)
        if last_progress is not None:
            stalled_for = now - last_progress
        elif generation_started is not None:
            # No round-trip yet this generation: fallback for when the one-shot verifier could not run.
            stalled_for = now - generation_started
        else:
            return
        if stalled_for <= _adapter._POLLING_STALL_TIMEOUT:
            return
        _adapter.logger.error(
            "[%s] Telegram polling stalled: no getUpdates progress for %.0fs "
            "(generation %d). Rebuilding the long-poll consumer through the reconnect ladder instead of staying silently deaf.",
            self.name, stalled_for, getattr(self, "_polling_generation", 0))
        self._spawn_polling_recovery(
            _adapter.asyncio.get_running_loop(),
            self._handle_polling_network_error(
                RuntimeError("getUpdates made no progress for %.0fs (polling stall watchdog)" % stalled_for)))

    def _verifier_stale(self, generation: int, progress: asyncio.Event) -> bool:
        """True when a verifier's generation no longer matters (progressed, fatal, replaced, torn down)."""
        return (
            self._teardown_started or progress.is_set() or self.has_fatal_error
            or not self._polling_progress_accepting or generation != self._polling_generation
            or progress is not self._polling_progress_event)

    async def _verify_polling_after_reconnect(self, generation: Optional[int] = None, progress: Optional[asyncio.Event] = None) -> None:
        """Require getUpdates progress, using getMe only to classify failure: a general-path getMe
        success cannot heal polling health. Connectivity failures enter the guarded recovery ladder."""
        from . import adapter as _adapter

        PROBE_TIMEOUT = 10
        if self._teardown_started:
            return
        if generation is None:
            generation = self._polling_generation
        if progress is None:
            progress = self._polling_progress_event
        with _adapter.contextlib.suppress(_adapter.asyncio.TimeoutError):
            await _adapter.asyncio.wait_for(progress.wait(), timeout=_adapter._POLLING_PROGRESS_TIMEOUT)
        if self._verifier_stale(generation, progress):
            return
        app = self._app
        if not (app and app.updater and app.updater.running):
            _adapter.logger.warning("[%s] Updater made no getUpdates progress and is not running", self.name)
            self._schedule_polling_recovery(
                RuntimeError("Updater not running after polling progress deadline"),
                reason="polling progress verifier: updater not running")
            return
        try:
            await _adapter.asyncio.wait_for(app.bot.get_me(), PROBE_TIMEOUT)
        except Exception as probe_err:
            if self._verifier_stale(generation, progress):
                return
            if not self._looks_like_network_error(probe_err):
                _adapter.logger.warning(
                    "[%s] Polling progress verifier hit a non-connectivity error (not retrying): %s", self.name,
                    _adapter._redact_telegram_error_text(probe_err))
                return
            _adapter.logger.warning(
                "[%s] Polling progress verifier connectivity probe failed: %s", self.name, _adapter._redact_telegram_error_text(probe_err))
            self._schedule_polling_recovery(probe_err, reason="polling progress verifier connectivity failure")
            return
        if self._verifier_stale(generation, progress):
            return
        self._schedule_polling_recovery(
            RuntimeError("getUpdates made no progress before verifier deadline"),
            reason="polling progress verifier: general path healthy but getUpdates stalled")

    def _disarm_ptb_retry_loop(self) -> None:
        """Synchronously stop PTB's internal polling retry loop.

        PTB's ``network_retry_loop`` calls our ``error_callback`` *synchronously* on a 409 Conflict then polls again; our
        callback only schedules async recovery, so two sessions overlap and Telegram 409s on a ~31s cadence. Setting PTB's
        private ``stop_event`` makes its loop exit on the next tick; ``updater.stop()`` + drain + ``start_polling()`` then build
        a fresh one. Best-effort across PTB spellings. Deliberately NOT flipping ``updater._running``: stop() raises when
        already False, which would skip the real teardown and poison the next start."""
        from . import adapter as _adapter

        updater = getattr(self._app, "updater", None) if self._app else None
        if updater is None:
            return
        for attr in ("_Updater__polling_task_stop_event", "_polling_task_stop_event"):
            stop_event = getattr(updater, attr, None)
            if isinstance(stop_event, _adapter.asyncio.Event):
                if not stop_event.is_set():
                    stop_event.set()
                    _adapter.logger.debug("[%s] Disarmed PTB polling retry loop via %s", self.name, attr)
                return
        _adapter.logger.debug(
            "[%s] Could not disarm PTB polling retry loop (stop_event not found on this PTB version); falling back to async stop()",
            self.name)

    async def _handle_polling_conflict(self, error: Exception) -> None:
        """Recover a 409 Conflict: the previous gateway process was killed but Telegram holds its
        getUpdates session ~30s. Stop, wait (growing delay), drain, restart — MAX_CONFLICT_RETRIES
        times before going fatal; a failed retry must never return silently (limbo)."""
        from . import adapter as _adapter

        if self._teardown_started:
            return
        if self.has_fatal_error and self.fatal_error_code == "telegram_polling_conflict":
            return
        self._polling_conflict_count += 1
        MAX_CONFLICT_RETRIES = 5
        # 15s, 25s, 35s, 45s, 55s — clears Telegram's ~30s session window without hammering the API.
        RETRY_DELAY = 10 + (self._polling_conflict_count * 10)  # seconds
        if self._polling_conflict_count <= MAX_CONFLICT_RETRIES:
            _adapter.logger.warning(
                "[%s] Telegram polling conflict (%d/%d) — previous session still "
                "held open on Telegram's servers. Waiting %ds for it to expire. Error: %s",
                self.name, self._polling_conflict_count, MAX_CONFLICT_RETRIES,
                RETRY_DELAY, _adapter._redact_telegram_error_text(error))
            # Stop the updater before sleeping (no-op if PTB raised before running was set).
            if not await self._stop_updater_or_go_fatal(self._app, "conflict-retry"):
                return
            await _adapter.asyncio.sleep(RETRY_DELAY)
            if self._teardown_started:
                return
            await self._drain_polling_connections()
            if self._teardown_started:
                return
            # Stable local ref: a concurrent disconnect() may null self._app across the awaits above.
            app = self._app
            # Capture a stable local reference: self._app can be reassigned to None by a concurrent
            # disconnect() while we're suspended across the awaits above (same race #55992 fixed on the
            # network path). Re-reading self._app after that point would raise AttributeError deep inside
            # start_polling instead of failing fast here, where the except below reschedules or escalates to
            # fatal.
            expected_generation = self._polling_generation + 1
            if not app:
                raise RuntimeError("Telegram application was torn down during conflict reconnect")
            # drop_pending_updates=True makes Telegram terminate any other getUpdates session for this
            # token (zombie or our own prior retry); without it each retry is immediately 409'd.
            # The competing session is either a zombie from the previous gateway process (whose long-poll
            # hasn't expired server-side yet) or our own previous retry's still-expiring session. Without
            # this, each retry starts a new getUpdates session that immediately gets 409'd by the previous
            # one, creating the very conflict we are trying to recover from (#75017).
            self._polling_conflict_recovery_generation = expected_generation
            try:
                await self._start_polling_once(app, drop_pending_updates=True, error_callback=self._polling_error_callback_ref)
                _adapter.logger.info(
                    "[%s] Telegram polling restarted after conflict retry %d/%d; health pending getUpdates progress",
                    self.name, self._polling_conflict_count, MAX_CONFLICT_RETRIES)
                return
            except _adapter._PollingLifecycleAbort:
                return
            except Exception as retry_err:
                if self._teardown_started:
                    return
                _adapter.logger.warning(
                    "[%s] Telegram polling retry %d/%d failed: %s. Scheduling next attempt.", self.name,
                    self._polling_conflict_count, MAX_CONFLICT_RETRIES, _adapter._redact_telegram_error_text(retry_err))
                # Never return silently: alive-and-"connected" with no polling is limbo.
                if self._polling_conflict_count < MAX_CONFLICT_RETRIES and not self._teardown_started:
                    # get_running_loop(): get_event_loop() raises on 3.10+ from PTB's callback context.
                    self._restart_polling_in_task(self._handle_polling_conflict(retry_err))
                    return
                # Fall through to fatal on the last retry.
            finally:
                if self._polling_conflict_recovery_generation == expected_generation:
                    self._polling_conflict_recovery_generation = None
        if self._teardown_started:
            return
        # Retries exhausted — fatal so the runner surfaces it and the user knows to act.
        message = (
            "Telegram polling could not recover after %d retries (%ds total wait). "
            "The previous gateway session is still held open on Telegram's servers, "
            "or another process is using the same bot token. To recover: ensure no other Hermes or OpenClaw instance is running "
            "with this token, then restart the gateway with 'hermes gateway restart'."
            % (MAX_CONFLICT_RETRIES, sum(10 + i * 10 for i in range(1, MAX_CONFLICT_RETRIES + 1))))
        _adapter.logger.error("[%s] %s Original error: %s", self.name, message, _adapter._redact_telegram_error_text(error))
        # Snapshot whether WE transition to fatal: a concurrent retry task suspended past the entry
        # guard reaches this branch too. Only the first transition notifies.
        _already_fatal = self.has_fatal_error and self.fatal_error_code == "telegram_polling_conflict"
        self._set_fatal_error("telegram_polling_conflict", message, retryable=False)
        try:
            if self._app and self._app.updater:
                await _adapter._await_with_thread_deadline(self._app.updater.stop(), timeout=_adapter._UPDATER_STOP_TIMEOUT)
        except _adapter.asyncio.TimeoutError:
            _adapter.logger.warning("[%s] updater.stop() timed out after exhausting conflict retries (likely CLOSE-WAIT socket); proceeding to fatal notify", self.name)
        except Exception as stop_error:
            _adapter.logger.warning(
                "[%s] Failed stopping Telegram updater after exhausting conflict retries: %s", self.name, stop_error,
                exc_info=True,
            )
        if not _already_fatal:
            await self._handoff_polling_fatal_error()

    async def _handoff_polling_fatal_error(self) -> None:
        """Notify the runner without letting child teardown cancel this owner: ``disconnect()`` cancels
        the tracked recovery/heartbeat tasks, so release only the current owner from its field."""
        from . import adapter as _adapter

        current_task = _adapter.asyncio.current_task()
        if self._polling_error_task is current_task:
            self._polling_error_task = None
        if getattr(self, "_polling_heartbeat_task", None) is current_task:
            self._polling_heartbeat_task = None
        await self._notify_fatal_error()

    async def _bot_identity_refresh_loop(self) -> None:
        """Keep the cached @username fresh in webhook mode (no heartbeat calls ``get_me()`` there)."""
        from . import adapter as _adapter

        while True:
            try:
                await _adapter.asyncio.sleep(self._BOT_IDENTITY_TTL_SECONDS)
                if self._teardown_started or self.has_fatal_error:
                    return
                await self._refresh_bot_identity(force=True)
            except _adapter.asyncio.CancelledError:
                return
            except Exception:
                _adapter.logger.debug("[%s] Telegram identity refresh loop iteration failed", self.name, exc_info=True)

    def _start_post_connect_housekeeping(self) -> None:
        """Kick off deferred post-connect housekeeping; idempotent while a task is still running."""
        from . import adapter as _adapter

        task = self._post_connect_task
        if task and not task.done():
            return
        self._post_connect_task = _adapter.asyncio.ensure_future(self._run_post_connect_housekeeping())

    async def _register_command_menu(self) -> None:
        """Register the command menu (from COMMAND_REGISTRY) in every scope — Telegram picks the
        narrowest matching one per chat type; forum topics are handled lazily by _ensure_forum_commands."""
        from . import adapter as _adapter

        from telegram import BotCommand, BotCommandScopeAllPrivateChats, BotCommandScopeAllGroupChats, BotCommandScopeDefault
        from hermes_cli.commands_platforms import telegram_menu_commands, telegram_menu_max_commands
        if not self._bot:
            return
        # Telegram allows 100 commands but has an undocumented ~4KB payload limit; default cap 60.
        max_commands = telegram_menu_max_commands()
        menu_commands, hidden_count = telegram_menu_commands(max_commands=max_commands)
        bot_commands = [BotCommand(name, desc) for name, desc in menu_commands]
        for scope_cls in (BotCommandScopeDefault, BotCommandScopeAllPrivateChats, BotCommandScopeAllGroupChats):
            scope_name = getattr(scope_cls, "__name__", str(scope_cls))
            try:
                await self._bot.set_my_commands(bot_commands, scope=scope_cls())
                _adapter.logger.info("[%s] set_my_commands OK for scope %s (%d cmds)", self.name, scope_name, len(bot_commands))
            except Exception as scope_err:
                _adapter.logger.warning("[%s] set_my_commands FAILED for scope %s: %s", self.name, scope_name, scope_err)
        if hidden_count:
            _adapter.logger.info(
                "[%s] Telegram menu: %d commands registered, %d hidden (over %d limit). Use /commands for full list.",
                self.name, len(menu_commands), hidden_count, max_commands)

    async def _run_post_connect_housekeeping(self) -> None:
        """Command menu, status indicator and DM topics off the connect path; every step is non-fatal.

        DM topics — all off the connect path so a slow Bot API call cannot blow the gateway connect timeout
        (#46298).
        """
        from . import adapter as _adapter

        try:
            try:
                await self._register_command_menu()
            except Exception as e:
                _adapter.logger.warning(
                    "[%s] Could not register Telegram command menu: %s", self.name, _adapter._redact_telegram_error_text(e), exc_info=True)
            with _adapter.contextlib.suppress(Exception):
                await self._set_status_indicator(online=True)
            try:
                await self._setup_dm_topics()
            except Exception as topics_err:
                _adapter.logger.warning("[%s] DM topics setup failed (non-fatal): %s", self.name, topics_err, exc_info=True)
        except _adapter.asyncio.CancelledError:
            raise
        finally:
            if self._post_connect_task is _adapter.asyncio.current_task():
                self._post_connect_task = None

    async def _on_platform_update(self, update, context) -> None:
        """Catch-all PTB handler (group 99) firing ``gateway_platform_event`` per inbound update with a
        stable envelope (no raw SDK objects) and an internal auth source. Never raises into PTB."""
        from . import adapter as _adapter

        handler: _adapter.Optional[_adapter.Callable[[_adapter.Dict[str, _adapter.Any], _adapter.Any], _adapter.Awaitable[None]]] = getattr(self, "_platform_event_handler", None)
        if handler is None:
            return
        try:
            from hermes_cli.lifecycle import has_hook
            if not has_hook("gateway_platform_event"):
                return
            event = self._normalize_platform_event(update)
        except Exception:
            _adapter.logger.debug("[%s] gateway_platform_event normalize error", self.name, exc_info=True)
            return
        if event is None:
            return
        # The gateway-owned boundary runs the full profile-scoped auth chain before plugin dispatch.
        try:
            source = self._source_for_platform_event_auth(update)
            await handler(event, source)
        except Exception:
            _adapter.logger.debug("[%s] gateway_platform_event dispatch error", self.name, exc_info=True)

    def _source_for_platform_event_auth(self, update):
        """Route a supported update to its event-specific auth-source extractor (reactor / editor);
        raises ``ValueError`` for updates without one so the boundary fails closed."""
        if getattr(update, "message_reaction", None) is not None:
            return self._source_from_reaction_for_auth(update)
        edited = getattr(update, "edited_message", None)
        if edited is not None:
            source = self._source_from_message_for_auth(edited)
            # Tolerates missing identities for pairing-flow callers; this boundary must not.
            if not source.user_id or not source.chat_id:
                raise ValueError("gateway_platform_event message_edited requires editor and chat identities")
            return source
        raise ValueError("gateway_platform_event source extraction has no extractor for this update type")

    def _normalize_platform_event(self, update) -> Optional[Dict[str, Any]]:
        """Map a PTB update to a ``{platform, event_type, payload}`` envelope (hooks.md contracts), or
        ``None`` for types without one."""
        if getattr(update, "message_reaction", None) is not None:
            return self._normalize_reaction_event(update)
        if getattr(update, "edited_message", None) is not None:
            return self._normalize_message_edited_event(update)
        return None

    @staticmethod
    def _is_id_like(value: Any) -> bool:
        from . import adapter as _adapter

        return not isinstance(value, bool) and isinstance(value, (str, int))

    def _normalize_reaction_event(self, update) -> Optional[Dict[str, Any]]:
        """``message_reaction`` → ``reaction`` event: emojis (unicode), custom_emoji_ids, chat_id,
        message_id, thread_id (always None — reactions carry none)."""
        from . import adapter as _adapter

        mr = getattr(update, "message_reaction", None)
        if mr is None:
            return None
        chat = getattr(mr, "chat", None)
        new_reaction = getattr(mr, "new_reaction", None) or []
        if not isinstance(new_reaction, (list, tuple)):
            return None
        chat_id = getattr(chat, "id", None) if chat is not None else None
        message_id = getattr(mr, "message_id", None)
        if not self._is_id_like(chat_id) or not self._is_id_like(message_id):
            return None
        emojis: _adapter.List[str] = []
        custom_emoji_ids: _adapter.List[str] = []
        for r in new_reaction[:64]:
            emoji = getattr(r, "emoji", None)
            if isinstance(emoji, str) and emoji:
                emojis.append(emoji[:64])
            custom_id = getattr(r, "custom_emoji_id", None)
            if self._is_id_like(custom_id):
                custom_emoji_ids.append(str(custom_id)[:128])
        return {
            "platform": "telegram",
            "event_type": "reaction",
            "payload": {
                "emojis": emojis, "custom_emoji_ids": custom_emoji_ids, "chat_id": str(chat_id)[:128],
                "message_id": str(message_id)[:128], "thread_id": None},
        }

    def _normalize_message_edited_event(self, update) -> Optional[Dict[str, Any]]:
        """``edited_message`` → ``message_edited`` event (v1, additive): chat_id, message_id, thread_id
        (forum topic), text (edited text or caption, bounded), edited_at (ISO 8601 UTC or None)."""
        from . import adapter as _adapter

        message = getattr(update, "edited_message", None)
        if message is None:
            return None
        chat = getattr(message, "chat", None)
        chat_id = getattr(chat, "id", None) if chat is not None else None
        message_id = getattr(message, "message_id", None)
        if not self._is_id_like(chat_id) or not self._is_id_like(message_id):
            return None
        text = getattr(message, "text", None) or getattr(message, "caption", None)
        if not isinstance(text, str):
            text = None
        thread_id = None
        thread_id_raw = getattr(message, "message_thread_id", None)
        if self._is_id_like(thread_id_raw) and bool(getattr(message, "is_topic_message", False)):
            thread_id = str(thread_id_raw)[:128]
        edited_at = None
        edit_date = getattr(message, "edit_date", None)
        try:
            if edit_date is not None and hasattr(edit_date, "isoformat"):
                edited_at = str(edit_date.isoformat())[:64]
        except Exception:
            edited_at = None
        return {
            "platform": "telegram",
            "event_type": "message_edited",
            "payload": {
                "chat_id": str(chat_id)[:128], "message_id": str(message_id)[:128], "thread_id": thread_id,
                "text": text[:8192] if text is not None else None, "edited_at": edited_at},
        }

    def _register_handlers(self, app) -> None:
        """Register every PTB handler on ``app`` (initial connect and the transient-init rebuild)."""
        from . import adapter as _adapter

        app.add_handler(_adapter.TelegramMessageHandler(_adapter.filters.TEXT & ~_adapter.filters.COMMAND, self._handle_text_message))
        app.add_handler(_adapter.TelegramMessageHandler(_adapter.filters.COMMAND, self._handle_command))
        app.add_handler(_adapter.TelegramMessageHandler(
            _adapter.filters.LOCATION | getattr(_adapter.filters, "VENUE", _adapter.filters.LOCATION), self._handle_location_message))
        app.add_handler(_adapter.TelegramMessageHandler(
            _adapter.filters.PHOTO | _adapter.filters.VIDEO | _adapter.filters.AUDIO | _adapter.filters.VOICE | _adapter.filters.Document.ALL | _adapter.filters.Sticker.ALL,
            self._handle_media_message))
        app.add_handler(_adapter.CallbackQueryHandler(self._handle_callback_query))
        # Inline command picker; inert until the owner enables inline mode via BotFather /setinline.
        app.add_handler(_adapter.InlineQueryHandler(self._handle_inline_query))
        # gateway_platform_event observer: group 99 observes alongside, never displaces, core handlers.
        app.add_handler(_adapter.TypeHandler(_adapter.Update, self._on_platform_update), group=99)

    async def _build_ptb_requests(self) -> tuple:
        """Build the (general, getUpdates) HTTPXRequest pair: fallback-IP transport, explicit proxy, or
        direct DNS; the getUpdates request is instrumented for polling-progress tracking."""
        # PTB's pool_timeout=1s default trips "Pool timeout" on flaky networks; safer defaults + env overrides.
        from . import adapter as _adapter

        request_kwargs = {
            "connection_pool_size": _adapter.env_int("HERMES_TELEGRAM_HTTP_POOL_SIZE", 512),
            "pool_timeout": _adapter.env_float("HERMES_TELEGRAM_HTTP_POOL_TIMEOUT", 8.0),
            "connect_timeout": _adapter.env_float("HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT", 10.0),
            "read_timeout": _adapter.env_float("HERMES_TELEGRAM_HTTP_READ_TIMEOUT", 20.0),
            "write_timeout": _adapter.env_float("HERMES_TELEGRAM_HTTP_WRITE_TIMEOUT", 20.0),
            # PTB routes file requests to media_write_timeout; httpx budgets it per socket write (stall
            # tolerance, not bandwidth), so 60s rides out congested-link buffer stalls.
            "media_write_timeout": 60.0,
        }
        # CLOSE_WAIT fd leak: PTB's httpx.AsyncClient has no keepalive tuning; inject platform_httpx_limits()
        # while preserving PTB's max_connections (httpx_kwargs is spread last, so `limits` here wins).
        # CLOSE_WAIT fd leak (#31599, same class as #18451): PTB's HTTPXRequest builds the underlying
        # httpx.AsyncClient with `limits = httpx.Limits(max_connections=connection_pool_size)` and *no*
        # keepalive tuning, so httpx's default keepalive_expiry=5.0 applies. Behind an HTTP proxy
        # (Cloudflare Warp etc.) a peer-initiated FIN can sit in CLOSE_WAIT longer than that, leaking fds in
        # the general request pool (_request[1]) which _drain_polling_connections never resets.
        from gateway.platforms._http_client_limits import platform_httpx_limits
        _base_limits = platform_httpx_limits()
        if _base_limits is not None:
            import httpx as _httpx
            _pool_limits = _httpx.Limits(
                max_connections=request_kwargs["connection_pool_size"],
                max_keepalive_connections=_base_limits.max_keepalive_connections, keepalive_expiry=_base_limits.keepalive_expiry)
            # A long-poll is continuously active, so keepalive expiry can't protect it from a server-side
            # close: never hand getUpdates a pooled socket from a previous poll.
            _updates_limits = _httpx.Limits(
                max_connections=request_kwargs["connection_pool_size"], max_keepalive_connections=0,
                keepalive_expiry=_base_limits.keepalive_expiry)
        else:  # pragma: no cover — httpx always present alongside PTB
            _pool_limits = _updates_limits = None

        def _with_limits(httpx_kwargs: Optional[dict] = None) -> dict:
            """Merge tuned limits into httpx client kwargs (proxy/direct branches only; the fallback-IP
            branch must pass limits straight into the transport — httpx ignores client `limits` then)."""
            kwargs = dict(httpx_kwargs or {})
            if _pool_limits is not None and "limits" not in kwargs:
                kwargs["limits"] = _pool_limits
            return kwargs

        disable_fallback = _adapter.os.getenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "").strip().lower() in {"1", "true", "yes", "on"}
        fallback_ips = [] if disable_fallback else self._fallback_ips()
        if not fallback_ips and not disable_fallback:
            discovery_timeout = self._env_float_clamped("HERMES_TELEGRAM_FALLBACK_DISCOVERY_TIMEOUT", 5.0, min_value=0.0)
            _adapter.logger.warning("[%s] Discovering Telegram API fallback IPs via DNS-over-HTTPS…", self.name)
            try:
                fallback_ips = await _adapter._await_with_thread_deadline(_adapter.discover_fallback_ips(), timeout=discovery_timeout)
            except Exception as exc:
                _adapter.logger.warning(
                    "[%s] Telegram fallback-IP discovery failed after %.0fs; "
                    "using seed IPv4 Telegram API IPs so a blackholed IPv6 hostname path cannot hang initialize() (#87015): %s",
                    self.name, discovery_timeout, _adapter._redact_telegram_error_text(exc))
                fallback_ips = list(_adapter.SEED_FALLBACK_IPS)
            else:
                _adapter.logger.info("[%s] Auto-discovered Telegram fallback IPs: %s", self.name, ", ".join(fallback_ips))
        proxy_url = _adapter.resolve_proxy_url("TELEGRAM_PROXY", target_hosts=["api.telegram.org", *fallback_ips])

        def _pair(general_httpx: dict, updates_httpx: dict, **extra) -> tuple:
            return (_adapter.HTTPXRequest(**request_kwargs, **extra, httpx_kwargs=general_httpx),
                    _adapter.HTTPXRequest(**request_kwargs, **extra, httpx_kwargs=updates_httpx))

        if fallback_ips and not proxy_url and not disable_fallback:
            _adapter.logger.info("[%s] Telegram fallback IPs active: %s", self.name, ", ".join(fallback_ips))
            # Separate request/update pools reduce contention during polling reconnect + bootstrap calls.
            _transport_kwargs: dict = {"socket_options": _adapter.tcp_keepalive_socket_options()}
            # Keep request/update pools separate to reduce contention during polling reconnect + bot API
            # bootstrap/delete_webhook calls. httpx ignores the client-level `limits` kwarg when a custom
            # `transport` is supplied (#58790). Unlike the proxy/direct branches (which inject limits at the
            # client level via `_with_limits`), this branch MUST pass the tuned limits directly into
            # TelegramFallbackTransport so its inner AsyncHTTPTransport instances honour keepalive_expiry —
            # do not route this through `_with_limits`, httpx would discard it.
            if _pool_limits is not None:
                _transport_kwargs["limits"] = _pool_limits
            _updates_transport_kwargs = dict(_transport_kwargs)
            if _updates_limits is not None:
                _updates_transport_kwargs["limits"] = _updates_limits
            request, get_updates_request = _pair(
                {"transport": _adapter.TelegramFallbackTransport(fallback_ips, **_transport_kwargs)},
                {"transport": _adapter.TelegramFallbackTransport(fallback_ips, **_updates_transport_kwargs)})
        elif proxy_url:
            _adapter.logger.info("[%s] Proxy detected; passing explicitly to HTTPXRequest: %s", self.name, proxy_url)
            request, get_updates_request = _pair(_with_limits(), {"limits": _updates_limits}, proxy=proxy_url)
        else:
            if disable_fallback:
                _adapter.logger.info("[%s] Telegram fallback-IP transport disabled via env", self.name)
            request, get_updates_request = _pair(_with_limits(), {"limits": _updates_limits})
        return request, self._instrument_polling_request(get_updates_request)

    async def _initialize_app_with_retries(self, builder) -> None:
        """``app.initialize()`` with a bounded retry ladder; rebuilds ``self._app``/``self._bot`` from
        ``builder`` after each failed attempt; OSError when the per-attempt or total watchdog expires."""
        from . import adapter as _adapter

        _max_connect = 8
        _init_timeout = _adapter.env_float("HERMES_TELEGRAM_INIT_TIMEOUT", 30.0)  # per attempt
        # Total watchdog: bounds the whole connect loop even if the retry loop silently stalls.
        _total_deadline = _adapter.asyncio.get_running_loop().time() + _init_timeout * _max_connect + 120.0
        _timed_out = f"Telegram initialization timed out after {_max_connect} attempts ({_init_timeout:.0f}s each)"
        for _attempt in range(_max_connect):
            rebuild_app = False
            try:
                if _adapter.asyncio.get_running_loop().time() >= _total_deadline:
                    raise OSError(
                        f"{_timed_out} — total connect watchdog deadline ({_init_timeout * _max_connect + 120.0:.0f}s) exceeded. "
                        f"Check network connectivity to api.telegram.org or set HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT / "
                        f"HERMES_TELEGRAM_INIT_TIMEOUT to a lower value.")
                _adapter.logger.warning("[%s] Connecting to Telegram (attempt %d/%d)…", self.name, _attempt + 1, _max_connect)
                # On timeout the (possibly shielded) initialize() task is abandoned; release the half-built
                # app's httpx client so it isn't leaked across the ladder.
                await _adapter._await_with_thread_deadline(
                    self._app.initialize(), timeout=_init_timeout, on_abandon=lambda app=self._app: _adapter._shutdown_abandoned_app(app))
                break
            except _adapter.asyncio.TimeoutError:
                rebuild_app = True
                if _attempt >= _max_connect - 1:
                    raise OSError(
                        f"{_timed_out}. Check network connectivity to api.telegram.org "
                        f"or set HERMES_TELEGRAM_HTTP_CONNECT_TIMEOUT to a lower value.")
                wait = min(2 ** _attempt, 15)
                _adapter.logger.warning(
                    "[%s] Connect attempt %d/%d timed out after %.0fs — retrying in %ds", self.name, _attempt + 1,
                    _max_connect, _init_timeout, wait)
                await _adapter.asyncio.sleep(wait)
            except Exception as init_err:
                # OSError always retries; anything else only when it looks like a network error.
                rebuild_app = True
                if (not isinstance(init_err, OSError) and not self._looks_like_network_error(init_err)) or _attempt >= _max_connect - 1:
                    raise
                wait = min(2 ** _attempt, 15)
                _adapter.logger.warning(
                    "[%s] Connect attempt %d/%d failed: %s — retrying in %ds", self.name, _attempt + 1, _max_connect, init_err, wait)
                await _adapter.asyncio.sleep(wait)
            except BaseException:
                # CancelledError etc.: log for the operator, then reraise. LAST so the Exception handlers win.
                _adapter.logger.warning(
                    "[%s] Connect attempt %d/%d interrupted by %s — propagating", self.name, _attempt + 1, _max_connect,
                    "CancelledError" if isinstance(_adapter.sys.exc_info()[1], _adapter.asyncio.CancelledError) else type(_adapter.sys.exc_info()[1]).__name__)
                raise
            finally:
                # A failed attempt may leave the app half-initialized: rebuild a fresh Application from the
                # same builder for the next attempt and discard the old one.
                if rebuild_app and _attempt < _max_connect - 1:
                    old_app = self._app
                    self._app = builder.build()
                    self._bot = self._app.bot
                    self._register_handlers(self._app)  # keep core and observer handlers in lockstep
                    with _adapter.contextlib.suppress(Exception):
                        await _adapter._shutdown_abandoned_app(old_app)

    async def _start_webhook_mode(self, webhook_url: str, *, is_reconnect: bool) -> None:
        """Start PTB's webhook server (Telegram pushes updates; lets cloud platforms auto-wake suspended
        machines). SECURITY: TELEGRAM_WEBHOOK_SECRET is REQUIRED — without it the endpoint accepts forged
        updates (GHSA-3vpc-7q5r-276h); refuse to start rather than run fail-open."""
        from . import adapter as _adapter

        webhook_port = _adapter.env_int("TELEGRAM_WEBHOOK_PORT", 8443)
        # Default "" → tornado listens on IPv4 + IPv6; "0.0.0.0" is unreachable on IPv6-only networks.
        webhook_host = (_adapter.os.getenv("TELEGRAM_WEBHOOK_HOST", "").strip() or str((self.config.extra or {}).get("webhook_host") or "").strip())
        # Profile-scoped read; only an UNSCOPED read under multiplex falls back to process env.
        from agent.secret_scope import UnscopedSecretError, get_secret
        try:
            webhook_secret = (get_secret("TELEGRAM_WEBHOOK_SECRET") or "").strip()
        except UnscopedSecretError:
            webhook_secret = _adapter.os.getenv("TELEGRAM_WEBHOOK_SECRET", "").strip()
        if not webhook_secret:
            raise RuntimeError(
                "TELEGRAM_WEBHOOK_SECRET is required when TELEGRAM_WEBHOOK_URL is set. Without it, the "
                "webhook endpoint accepts forged updates from anyone who can reach it — see "
                "https://github.com/NousResearch/hermes-agent/security/advisories/GHSA-3vpc-7q5r-276h.\n\n"
                "Generate a secret and set it in your .env:\n  export TELEGRAM_WEBHOOK_SECRET=\"$(openssl rand -hex 32)\"\n\n"
                "Then register it with Telegram when setting the webhook via setWebhook's secret_token parameter.")
        from urllib.parse import urlparse
        webhook_path = urlparse(webhook_url).path or "/telegram"
        await self._app.updater.start_webhook(
            listen=webhook_host, port=webhook_port, url_path=webhook_path, webhook_url=webhook_url,
            secret_token=webhook_secret, allowed_updates=_adapter.Update.ALL_TYPES,
            drop_pending_updates=not is_reconnect,  # push-based ⇒ practically a no-op; mirrors polling
       )
        self._webhook_mode = True
        self._polling_progress_accepting = False
        self._send_path_degraded = False
        _adapter.logger.info(
            "[%s] Webhook server listening on %s:%d%s", self.name, webhook_host or "* (all interfaces, IPv4+IPv6)",
            webhook_port, webhook_path)

    async def _start_polling_mode(self, *, is_reconnect: bool) -> None:
        """Clear any stale webhook and start resilient long polling."""
        # Best-effort: a transient Bot API error must not fail gateway startup — degrade to recovery.
        from . import adapter as _adapter

        await self._delete_webhook_best_effort(require_success=not is_reconnect)
        loop = _adapter.asyncio.get_running_loop()

        def _polling_error_callback(error: Exception) -> None:
            if self._teardown_started or self._recovery_in_flight():
                return
            if self._looks_like_polling_conflict(error):
                # Stop PTB's network_retry_loop synchronously BEFORE scheduling async recovery, else PTB's
                # retry and our stop->restart overlap and produce a fresh 409.
                self._disarm_ptb_retry_loop()
                self._spawn_polling_recovery(loop, self._handle_polling_conflict(error))
            elif self._looks_like_network_error(error):
                _adapter.logger.warning("[%s] Telegram network _redact_telegram_error_text(error), scheduling reconnect: %s", self.name, error)
                self._spawn_polling_recovery(loop, self._handle_polling_network_error(error))
            else:
                _adapter.logger.error("[%s] Telegram polling _redact_telegram_error_text(error): %s", self.name, error, exc_info=True)

        self._polling_error_callback_ref = _polling_error_callback  # reused by _handle_polling_conflict
        polling_started = await self._start_polling_resilient(
            # Cold first boot drops the stale Bot API queue; a watcher reconnect preserves it.
            drop_pending_updates=not is_reconnect, error_callback=_polling_error_callback, require_progress=not is_reconnect)
        if not polling_started:
            _adapter.logger.warning(
                "[%s] Connected in degraded Telegram mode: gateway is alive, polling will be retried in the background", self.name)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Connect via long polling, or a webhook server if ``TELEGRAM_WEBHOOK_URL`` is set.

        ``is_reconnect``: False = cold boot (drop the stale Bot API queue); True = watcher reconnect (preserve queued
        updates, else every message sent during the outage is lost). Webhook env: TELEGRAM_WEBHOOK_URL,
        TELEGRAM_WEBHOOK_PORT (8443), TELEGRAM_WEBHOOK_HOST, TELEGRAM_WEBHOOK_SECRET."""
        # Explicit connect() is the only operation allowed to reopen polling after a completed teardown.
        from . import adapter as _adapter

        self._polling_teardown_started = False
        self._webhook_mode = False  # re-evaluated on every explicit connection
        if not _adapter.TELEGRAM_AVAILABLE:
            _adapter.logger.error("[%s] python-telegram-bot not installed. Run: pip install python-telegram-bot", self.name)
            self._set_fatal_error("missing_dependency", "python-telegram-bot not installed", retryable=False)
            return False
        if not self.config.token:
            _adapter.logger.error("[%s] No bot token configured", self.name)
            self._set_fatal_error("missing_credentials", "No bot token configured", retryable=False)
            return False
        try:
            if not self._acquire_platform_lock('telegram-bot-token', self.config.token, 'Telegram bot token'):
                return False
            builder = _adapter.Application.builder().token(self.config.token)
            custom_base_url = self.config.extra.get("base_url")
            if custom_base_url:
                builder = builder.base_url(custom_base_url)
                builder = builder.base_file_url(self.config.extra.get("base_file_url", custom_base_url))
                _adapter.logger.info("[%s] Using custom Telegram base_url: %s", self.name, custom_base_url)
            # Local-mode telegram-bot-api returns absolute server-side file paths; PTB needs local_mode=True
            # so download_*() reads from disk instead of a 404ing HTTP GET.
            if self.config.extra.get("local_mode"):
                builder = builder.local_mode(True)
                _adapter.logger.info("[%s] Using Telegram local_mode (read files from disk)", self.name)
            request, get_updates_request = await self._build_ptb_requests()
            builder = builder.request(request).get_updates_request(get_updates_request)
            self._app = builder.build()
            self._bot = self._app.bot
            # Plugin PTB handlers go BEFORE core: PTB dispatches the first matching handler per group.
            self._wire_plugin_handlers(self._app)
            self._register_handlers(self._app)
            await self._initialize_app_with_retries(builder)
            await self._app.start()
            webhook_url = _adapter.os.getenv("TELEGRAM_WEBHOOK_URL", "").strip()
            if webhook_url:
                await self._start_webhook_mode(webhook_url, is_reconnect=is_reconnect)
            else:
                await self._start_polling_mode(is_reconnect=is_reconnect)
            self._mark_connected()
            # WARNING, not INFO: "Connecting…" above is WARNING and reaches the terminal; an INFO success
            # line made healthy startups look stalled at "attempt 1/8".
            _adapter.logger.warning("[%s] Connected to Telegram (%s mode)", self.name, "webhook" if self._webhook_mode else "polling")
            # Heartbeat only in polling mode: webhook mode has no long-poll socket to wedge in CLOSE-WAIT.
            # WARNING, not INFO: the "Connecting to Telegram (attempt N/8)…" line above is emitted at
            # WARNING and reaches the terminal (the gateway's default stderr handler is WARNING-only), but
            # this success line was INFO and went to the log file only. A healthy startup therefore looked
            # permanently stalled at "attempt 1/8" on the console — the logging illusion in #90835. Both
            # sides of the connect transition must share a terminal-visible level so a real hang is the
            # *absence* of this line, not ambiguity.
            if not self._webhook_mode:
                self._restart_task_attr("_polling_heartbeat_task", self._polling_heartbeat_loop())
            # Seed the live identity from PTB's initialize() cache; polling rides the heartbeat's get_me(),
            # webhook mode gets a low-frequency refresh loop (else a BotFather rename breaks routing).
            self._note_bot_username(getattr(self._bot, "username", None))
            self._bot_identity_checked_at = _adapter.time.monotonic()
            if self._webhook_mode:
                self._restart_task_attr("_bot_identity_refresh_task", self._bot_identity_refresh_loop())
            # Command menu / DM topics / status indicator can stall for some tokens: defer to a cancellable
            # task so one slow call can't sink the (gateway-timed) connect while transport is live.
            # Command-menu registration, DM-topic setup, and the status indicator each make Bot API calls
            # that can stall for certain tokens. Running them here — inside the connect() coroutine that the
            # gateway wraps in a connect timeout — means one slow call blows the whole connect and the
            # adapter never comes up, even though polling/webhook is already live (#46298).
            self._start_post_connect_housekeeping()
            return True
        except Exception as e:
            self._release_platform_lock()
            safe_error = _adapter._redact_telegram_error_text(e)
            # Classify by exception TYPE (never message text): auth failures can never self-heal, so
            # marking them retryable put agents into a silent eternal reconnect loop.
            if self._looks_like_auth_error(e):
                message = (
                    f"Telegram bot token rejected: {safe_error}. "
                    "The token is invalid or was revoked — generate a new one "
                    "with @BotFather and update TELEGRAM_BOT_TOKEN.")
                self._set_fatal_error("telegram_auth_error", message, retryable=False)
            else:
                self._set_fatal_error("telegram_connect_error", f"Telegram startup failed: {safe_error}", retryable=True)
            _adapter.logger.error("[%s] Failed to connect to Telegram: %s", self.name, safe_error)
            return False

    async def _set_status_indicator(self, online: bool) -> None:
        """Set the bot's short description to the online/offline text (closest Bot API surface to
        presence). No-op unless ``extra.status_indicator``; failures are debug-logged."""
        from . import adapter as _adapter

        if not getattr(self, "_status_indicator_enabled", False):
            return
        bot = self._bot
        if bot is None:
            return
        text = (self._status_online_text if online else self._status_offline_text)[:120]  # Telegram cap
        try:
            await bot.set_my_short_description(short_description=text)
            _adapter.logger.info("[%s] Set bot status indicator to %r", self.name, text)
        except Exception as e:
            _adapter.logger.debug("[%s] Failed to set bot status indicator to %r: %s", self.name, text, _adapter._redact_telegram_error_text(e))

    @staticmethod
    def _collect_live_tasks(candidates, current_task) -> list:
        """Unique, unfinished tasks from ``candidates`` excluding ``current_task`` (so teardown never cancels itself)."""
        from . import adapter as _adapter

        seen: set[int] = set()
        out: list[_adapter.asyncio.Task] = []
        for task in candidates:
            if not task or task.done() or task is current_task or id(task) in seen:
                continue
            seen.add(id(task))
            out.append(task)
        return out

    def _clear_task_attrs_except(self, current_task, *attrs: str) -> None:
        for attr in attrs:
            if getattr(self, attr, None) is not current_task:
                setattr(self, attr, None)

    async def _cancel_pending_delivery_tasks(self) -> None:
        """Cancel every delayed-delivery task family before disconnect completes (media-group, photo-batch, text-batch flushes plus
        polling recovery all sit behind ``asyncio.sleep()`` and would dispatch ``handle_message`` into a torn-down session)."""
        from . import adapter as _adapter

        current_task = _adapter.asyncio.current_task()
        pending_tasks = self._collect_live_tasks(
            [
                *self._media_group_tasks.values(), *self._pending_photo_batch_tasks.values(), *self._pending_text_batch_tasks.values(),
                getattr(self, "_polling_error_task", None), getattr(self, "_polling_progress_verifier_task", None),
                # Hold-queue redispatch must be cancellable+awaitable on teardown too.
                getattr(self, "_held_inbound_redispatch_task", None),
           ],
            current_task)
        awaitable_tasks = [t for t in pending_tasks if _adapter.asyncio.isfuture(t) or _adapter.asyncio.iscoroutine(t)]
        # Hold-queue redispatch must be cancellable+awaitable on teardown so it cannot dispatch
        # handle_message into a torn-down session (same lifecycle rule teknium called out on #72037 for
        # shielded flush dispatch).
        for task in pending_tasks:
            task.cancel()
        if awaitable_tasks:
            await _adapter.asyncio.gather(*awaitable_tasks, return_exceptions=True)
        # Salvage buffered inbound events before clearing maps — unless permanent fatal, where no
        # reconnect can drain and hold would re-orphan them.
        if self._is_permanent_fatal():
            n_pending = len(self._pending_text_batches) + len(self._pending_photo_batches) + len(self._media_group_events)
            if n_pending:
                _adapter.logger.warning("[Telegram] Non-retryable fatal teardown; discarding %d pending inbound batch(es)", n_pending)
        else:
            for events, where in (
                (self._pending_text_batches, "text-batch-teardown"), (self._pending_photo_batches, "photo-batch-teardown"),
                (self._media_group_events, "media-group-teardown")):
                for event in list(events.values()):
                    self._hold_inbound_event(event, where=where)
        for d in (
            self._media_group_tasks, self._media_group_events, self._pending_photo_batch_tasks,
            self._pending_photo_batches, self._pending_text_batch_tasks, self._pending_text_batches):
            d.clear()
        self._clear_task_attrs_except(
            current_task, "_polling_error_task", "_polling_progress_verifier_task", "_held_inbound_redispatch_task")

    async def _await_disconnect_step(self, awaitable, timeout: float, step: str) -> bool:
        """Await one disconnect step; detach on timeout so teardown advances (``wait_for`` would wait for a
        PTB close that swallows ``CancelledError`` on a half-dead socket). Abandoned tasks are observed.

        ``asyncio.wait_for`` cancels an overdue child but then waits for it to exit. Detach at the deadline
        and continue — the abandoned task is observed via ``_consume_abandoned_task``. See #80598.
        """
        from . import adapter as _adapter

        task = _adapter.asyncio.ensure_future(awaitable)
        try:
            done, _pending = await _adapter.asyncio.wait({task}, timeout=timeout if timeout > 0 else None)
        except _adapter.asyncio.CancelledError:
            # asyncio.wait does NOT cancel its futures when itself cancelled; don't orphan the inner task.
            task.cancel()
            # Mirror the pattern used by GatewayRunner._await_adapter_cleanup_with_timeout. See #80598.
            task.add_done_callback(_adapter._consume_abandoned_task)
            raise
        if task in done:
            with _adapter.contextlib.suppress(_adapter.asyncio.CancelledError):
                await task
            return True
        task.cancel()
        task.add_done_callback(_adapter._consume_abandoned_task)
        _adapter.logger.warning("[%s] %s timed out after %.1fs during disconnect; continuing teardown", self.name, step, timeout)
        return False

    def _restart_task_attr(self, attr: str, coro) -> None:
        """Cancel any live task stored at ``self.<attr>`` and start ``coro`` in its place."""
        from . import adapter as _adapter

        prior = getattr(self, attr, None)
        if prior and not prior.done():
            prior.cancel()
        setattr(self, attr, _adapter.asyncio.ensure_future(coro))

    async def _cancel_task_attr(self, attr: str, label: str) -> None:
        """Cancel + bounded-await the task stored at ``self.<attr>`` (may be missing: object.__new__ tests), then clear it."""
        from . import adapter as _adapter

        task = getattr(self, attr, None)
        if task and not task.done():
            task.cancel()
            await self._await_disconnect_step(task, _adapter._DISCONNECT_STEP_TIMEOUT, label)
        setattr(self, attr, None)

    async def disconnect(self) -> None:
        """Stop polling/webhook, cancel pending delayed deliveries, and disconnect."""
        from . import adapter as _adapter
        from .choice_picker import cancel_choice_pages

        cancel_choice_pages(self)
        # Mark disconnected first so the drop guard short-circuits any flush that wins the race.
        self._mark_disconnected()
        self._polling_teardown_started = True
        self._polling_progress_accepting = False
        self._polling_generation = getattr(self, "_polling_generation", 0) + 1
        self._polling_progress_event = _adapter.asyncio.Event()
        self._send_path_degraded = True
        # Release the bot-token lock immediately so a wedged close cannot block the reconnect watcher.
        # The rest of teardown is best-effort against a half-dead transport. See #80598.
        self._release_platform_lock()
        # Cancel and await both polling lifecycle owners right after the fence, before any other teardown
        # await lets them start a new generation.
        current_task = _adapter.asyncio.current_task()
        lifecycle_tasks = self._collect_live_tasks(
            [getattr(self, "_polling_error_task", None), getattr(self, "_polling_progress_verifier_task", None)], current_task)
        for task in lifecycle_tasks:
            task.cancel()
        lifecycle_tasks = [t for t in lifecycle_tasks if _adapter.asyncio.isfuture(t) or _adapter.asyncio.iscoroutine(t)]
        if lifecycle_tasks:
            await self._await_disconnect_step(
                _adapter.asyncio.gather(*lifecycle_tasks, return_exceptions=True), _adapter._DISCONNECT_STEP_TIMEOUT, "lifecycle-task cancel")
        self._clear_task_attrs_except(current_task, "_polling_error_task", "_polling_progress_verifier_task")
        # Cancellation callbacks may have run while awaited; the fence stays authoritative.
        self._polling_progress_accepting = False
        self._send_path_degraded = True
        # Cancel deferred post-connect housekeeping so it cannot fire into a half-torn-down bot client.
        # Cancel deferred post-connect housekeeping (command-menu / DM-topic / status-indicator Bot API
        # calls) so it cannot fire into a half-torn-down bot client (#46298). getattr guards the
        # object.__new__ test pattern where __init__ (which sets this attr) is never called.
        post_connect_task = getattr(self, "_post_connect_task", None)
        if post_connect_task and not post_connect_task.done():
            post_connect_task.cancel()
            await self._await_disconnect_step(
                _adapter.asyncio.gather(post_connect_task, return_exceptions=True), _adapter._DISCONNECT_STEP_TIMEOUT, "post-connect cancel")
        self._post_connect_task = None
        # Cancel the heartbeat (and webhook-mode identity loop) before tearing down the app.
        await self._cancel_task_attr("_polling_heartbeat_task", "heartbeat cancel")
        await self._cancel_task_attr("_bot_identity_refresh_task", "identity-refresh cancel")
        # Mark the bot "Offline" while its HTTP client is still alive. Opt-in, non-fatal.
        with _adapter.contextlib.suppress(Exception):
            await self._await_disconnect_step(self._set_status_indicator(online=False), _adapter._DISCONNECT_STEP_TIMEOUT, "status-indicator update")
        await self._await_disconnect_step(self._cancel_pending_delivery_tasks(), _adapter._DISCONNECT_STEP_TIMEOUT, "pending-delivery cancel")
        if self._app:
            try:
                # Bounded: a CLOSE-WAIT socket can wedge updater.stop() forever; fall through on timeout.
                if self._app.updater and self._app.updater.running:
                    try:
                        await self._await_disconnect_step(self._app.updater.stop(), _adapter._UPDATER_STOP_TIMEOUT, "updater.stop()")
                    except Exception as stop_error:
                        _adapter.logger.warning(
                            "[%s] updater.stop() failed during disconnect: %s", self.name, _adapter._redact_telegram_error_text(stop_error))
                # app.stop()/shutdown() can also block on a half-dead httpx pool.
                # Detach-on-timeout so disconnect always returns (#80598).
                if self._app.running:
                    await self._await_disconnect_step(self._app.stop(), _adapter._DISCONNECT_STEP_TIMEOUT, "app.stop()")
                await self._await_disconnect_step(self._app.shutdown(), _adapter._DISCONNECT_STEP_TIMEOUT, "app.shutdown()")
            except Exception as e:
                _adapter.logger.warning("[%s] Error during Telegram disconnect: %s", self.name, _adapter._redact_telegram_error_text(e))
        self._app = None
        self._bot = None
        _adapter.logger.info("[%s] Disconnected from Telegram", self.name)
