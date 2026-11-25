"""Persistent Feishu message-id claims and safe release before gateway admission."""

import asyncio
import concurrent.futures
import contextvars
import json
import logging
import threading
import time

logger = logging.getLogger("plugins.platforms.feishu.adapter")
_DEDUP_TTL_SECONDS = 24 * 60 * 60
_APP_FINALIZERS = {}
_APP_FINALIZERS_LOCK = threading.Lock()


def _register_app_finalizer(app_id, task):
    if not app_id:
        return
    gate = concurrent.futures.Future()
    with _APP_FINALIZERS_LOCK:
        _APP_FINALIZERS.setdefault(app_id, set()).add(gate)

    def settle(done):
        error = RuntimeError("Feishu disconnect was cancelled") if done.cancelled() else done.exception()
        if error is None:
            with _APP_FINALIZERS_LOCK:
                remaining = _APP_FINALIZERS.get(app_id)
                if remaining is not None:
                    remaining.discard(gate)
                    if not remaining:
                        _APP_FINALIZERS.pop(app_id, None)
            gate.set_result(None)
        else:
            gate.set_exception(error)
    task.add_done_callback(settle)


async def _wait_app_finalizers(app_id):
    while app_id:
        with _APP_FINALIZERS_LOCK:
            pending = tuple(_APP_FINALIZERS.get(app_id, ()))
        if not pending:
            break
        outcomes = await asyncio.shield(asyncio.gather(
            *(asyncio.wrap_future(gate) for gate in pending), return_exceptions=True))
        if any(isinstance(outcome, BaseException) for outcome in outcomes):
            return False
    return True


async def _acquire_after_app_finalizers(app_id, acquire):
    while True:
        if not await _wait_app_finalizers(app_id):
            return None
        with _APP_FINALIZERS_LOCK:
            if not _APP_FINALIZERS.get(app_id):
                return acquire()


class _Claim:
    def __init__(self, adapter, message_id, token):
        self.adapter, self.message_id, self.token = adapter, message_id, token
        self.dispatched = False
        self.control_check = None

    async def release(self):
        await self.release_batch([self])

    async def release_batch(self, claims):
        if not all(isinstance(claim, _Claim) and claim.adapter._dedup_lock is self.adapter._dedup_lock
                   and claim.adapter._dedup_persist_lock_or_create() is self.adapter._dedup_persist_lock_or_create()
                   for claim in claims):
            return False
        try:
            await self.adapter._release_message_claims(claims)
        except Exception:
            logger.exception("[Feishu] Failed to release an unadmitted message claim")
        return True


class FeishuDedupMixin:
    def _capture_ingress_controls(self, event, session_key=None):
        if not event._ingress_claims:
            return
        if not hasattr(event, "_ingress_control_check"):
            event._ingress_control_check = self._ingress_control_snapshot(event, session_key)
        for claim in event._ingress_claims:
            if claim.control_check is None:
                claim.control_check = event._ingress_control_check

    def _ingress_control_snapshot(self, event, session_key=None):
        session_key = session_key or self._event_session_key(event)
        local_epochs = self.__dict__.setdefault("_feishu_ingress_control_epochs", {})
        local_epoch = local_epochs.get(session_key, 0)
        runner = getattr(self, "gateway_runner", None)
        if runner is None:
            return lambda: local_epochs.get(session_key, 0) == local_epoch
        from gateway.session_authorities import all_authorities
        from gateway.session_identity import identity_of
        try:
            route = runner._session_key_for_source(event.source)
            registry = getattr(runner, "session_authorities", None)
            identity = identity_of(event.source)
            owner_home = identity.runtime_home if identity is not None and callable(getattr(registry, "for_home", None)) else None
            if owner_home is not None:
                owner = registry.for_home(owner_home)
                if owner is None:
                    return lambda: False
                epochs = [(owner, owner.route_epoch(route))]
            else:
                epochs = [(authority, authority.route_epoch(route)) for authority in all_authorities(runner)]
        except Exception:
            logger.debug("[Feishu] Could not capture ingress control epochs", exc_info=True)
            return lambda: False

        def unchanged():
            try:
                current = [registry.for_home(owner_home)] if owner_home is not None else all_authorities(runner)
                return (local_epochs.get(session_key, 0) == local_epoch and len(current) == len(epochs)
                        and all(any(authority is other for other in current)
                                and authority.route_epoch(route) == epoch for authority, epoch in epochs))
            except Exception:
                logger.debug("[Feishu] Could not compare ingress control epochs", exc_info=True)
                return False
        return unchanged

    def _start_session_processing(self, event, session_key, *, interrupt_event=None):
        self._capture_ingress_controls(event, session_key)
        previous_entries = event._ingress_handler_entries
        # Startup restore queues the event before plugins or admission; its first replay is a new attempt.
        startup_replay = (getattr(event, "_hermes_startup_restore_replay", False)
                          and not getattr(event, "_feishu_startup_replay_started", False))
        started = super()._start_session_processing(event, session_key, interrupt_event=interrupt_event)
        if started and event._ingress_claims:
            if startup_replay:
                event._feishu_startup_replay_started = True
            event._mark_ingress_dispatched()
            task = self._session_tasks.get(session_key)
            if task is not None and hasattr(task, "add_done_callback"):
                task.add_done_callback(lambda done: self._settle_unstarted_ingress(
                    done, event, previous_entries, startup_replay))
        return started

    def _settle_unstarted_ingress(self, task, event, previous_entries, startup_replay):
        if ((previous_entries and not startup_replay) or event._ingress_handler_entries > previous_entries
                or not event._ingress_controls_unchanged()
                or getattr(task, "_feishu_control_cancel", False)):
            return
        cleanups = self.__dict__.setdefault("_feishu_ingress_cleanups", set())
        cleanup = asyncio.create_task(self._release_if_controls_unchanged(event))
        cleanups.add(cleanup)
        cleanup.add_done_callback(cleanups.discard)

    async def _release_if_controls_unchanged(self, event):
        if event._ingress_controls_unchanged():
            await event._release_unadmitted_claims()

    async def _handle_message_while_active(self, event, session_key):
        self._capture_ingress_controls(event, session_key)
        if getattr(event, "_hermes_startup_restore_replay", False):
            event._feishu_startup_replay_started = True
        event._mark_ingress_dispatched()
        await super()._handle_message_while_active(event, session_key)
        burst = self.__dict__.get("_shared_bursts", {}).get(session_key)
        if burst is not None and any(claim in burst.head._ingress_claims for claim in event._ingress_claims):
            tracked = self.__dict__.setdefault("_feishu_ingress_busy_bursts", {})
            if burst not in tracked:
                tracked[burst] = burst.head._ingress_handler_entries
                burst.done.add_done_callback(lambda done: not done.cancelled() and tracked.pop(burst, None))

    async def cancel_session_processing(self, session_key, *, release_guard=True, discard_pending=True):
        epochs = self.__dict__.setdefault("_feishu_ingress_control_epochs", {})
        epochs[session_key] = epochs.get(session_key, 0) + 1
        if (task := self._session_tasks.get(session_key)) is not None:
            task._feishu_control_cancel = True
        await super().cancel_session_processing(session_key, release_guard=release_guard, discard_pending=discard_pending)

    async def cancel_background_tasks(self):
        try:
            await super().cancel_background_tasks()
        finally:
            events = self._take_held_busy_bursts()
            tracked = self.__dict__.get("_feishu_ingress_busy_bursts", {})
            for burst, entries in list(tracked.items()):
                if (burst.commit is not None and burst.commit.cancelled() and not entries
                        and burst.head._ingress_handler_entries == entries and burst.head._ingress_controls_unchanged()):
                    events.append(burst.head)
                tracked.pop(burst, None)
            if cleanup := self._schedule_ingress_cleanup(events):
                await asyncio.shield(cleanup)
        while pending := tuple(self.__dict__.get("_feishu_ingress_cleanups", ())):
            await asyncio.shield(asyncio.gather(*pending, return_exceptions=True))

    def adopt_inbound_dedup(self, previous):
        """Share claims with a reconnecting instance while the old instance still has work."""
        if (getattr(previous, "platform", None) != self.platform
                or getattr(previous, "_dedup_state_path", None) != self._dedup_state_path):
            return
        with previous._dedup_lock:
            self._seen_message_ids = previous._seen_message_ids
            self._seen_message_order = previous._seen_message_order
            self._dedup_claim_tokens = previous.__dict__.setdefault("_dedup_claim_tokens", {})
            self._dedup_persist_lock = previous._dedup_persist_lock_or_create()
            self._dedup_lock = previous._dedup_lock
            self._feishu_disconnect_finalizers = previous.__dict__.setdefault("_feishu_disconnect_finalizers", set())

    def _load_seen_message_ids(self) -> None:
        try:
            payload = json.loads(self._dedup_state_path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError):
            logger.warning("[Feishu] Failed to load persisted dedup state from %s", self._dedup_state_path, exc_info=True)
            return
        seen_data = payload.get("message_ids", {}) if isinstance(payload, dict) else {}
        now = time.time()
        if isinstance(seen_data, list):  # legacy format: plain list of IDs, no timestamps
            entries = {str(item).strip(): 0.0 for item in seen_data if str(item).strip()}
        elif isinstance(seen_data, dict):
            entries = {}
            for key, value in seen_data.items():
                if isinstance(key, str) and key.strip() and isinstance(value, (int, float, str)):
                    try:
                        entries[key] = float(value)
                    except ValueError:
                        pass
        else:
            return
        valid = {m: ts for m, ts in entries.items() if ts == 0.0 or _DEDUP_TTL_SECONDS <= 0 or now - ts < _DEDUP_TTL_SECONDS}
        sorted_ids = sorted(valid, key=lambda k: valid[k], reverse=True)[:self._dedup_cache_size]
        self._seen_message_order = list(reversed(sorted_ids))
        self._seen_message_ids = {k: valid[k] for k in sorted_ids}

    async def _persist_seen_async(self):
        async with self._dedup_persist_lock_or_create():
            if not getattr(self, "_dedup_persistence_retired", False):
                await self._run_blocking(self._persist_seen_message_ids)

    async def _persist_seen_during_teardown(self):
        lock = self.__dict__.setdefault("_feishu_cleanup_executor_lock", threading.Lock())
        with lock:
            executor = self.__dict__.get("_feishu_cleanup_executor")
            if executor is None:
                executor = self._feishu_cleanup_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="hermes-feishu-dedup-cleanup")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(executor, contextvars.copy_context().run, self._persist_seen_message_ids)

    def _shutdown_dedup_cleanup_executor(self):
        lock = self.__dict__.setdefault("_feishu_cleanup_executor_lock", threading.Lock())
        with lock:
            executor = self.__dict__.pop("_feishu_cleanup_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    async def _claim_message_id(self, message_id):
        now, token = time.time(), object()
        with self._dedup_lock:
            seen_at = self._seen_message_ids.get(message_id)
            if seen_at is not None and (_DEDUP_TTL_SECONDS <= 0 or now - seen_at < _DEDUP_TTL_SECONDS):
                return None
            owners = self.__dict__.setdefault("_dedup_claim_tokens", {})
            if seen_at is not None:
                self._seen_message_order[:] = [item for item in self._seen_message_order if item != message_id]
            self._seen_message_ids[message_id] = now
            owners[message_id] = token
            self._seen_message_order.append(message_id)
            while len(self._seen_message_order) > self._dedup_cache_size:
                oldest = self._seen_message_order.pop(0)
                self._seen_message_ids.pop(oldest, None)
                owners.pop(oldest, None)
        claim = _Claim(self, message_id, token)
        # Keep the lock and the write alive after cancellation: a cancelled executor future
        # can still finish its write, so release must wait until that write has settled.
        writing = asyncio.create_task(self._persist_seen_async())
        try:
            await asyncio.shield(writing)
        except BaseException:
            async def release_after_write():
                try:
                    await writing
                except asyncio.CancelledError:
                    # The executor can cancel a queued write during adapter shutdown.
                    pass
                except Exception:
                    logger.exception("[Feishu] Persisting the inbound claim failed")
                await claim.release()

            cleanup = asyncio.create_task(release_after_write())
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                pass  # the shielded cleanup still owns the pending write
            raise
        return claim

    async def _is_duplicate(self, message_id: str) -> bool:
        return await self._claim_message_id(message_id) is None

    async def _release_message_claims(self, claims):
        async with self._dedup_persist_lock_or_create():
            # A /stop may have landed while release was waiting for the persistence lock.
            if any(claim.control_check is not None and not claim.control_check() for claim in claims):
                return
            with self._dedup_lock:
                owners = self.__dict__.setdefault("_dedup_claim_tokens", {})
                ids = {claim.message_id for claim in claims if owners.get(claim.message_id) is claim.token}
                if not ids:
                    return
                for message_id in ids:
                    owners.pop(message_id, None)
                    self._seen_message_ids.pop(message_id, None)
                self._seen_message_order[:] = [item for item in self._seen_message_order if item not in ids]
            if getattr(self, "_dedup_persistence_retired", False):
                return  # another process may own the on-disk cache after the app lock was released
            # A retired adapter cannot restart its SDK executor, but a late failure still
            # needs to persist while it still owns the app lock.
            if getattr(self, "_sdk_executor_closing", False):
                await self._persist_seen_during_teardown()
            else:
                try:
                    await self._run_blocking(self._persist_seen_message_ids)
                except asyncio.CancelledError:
                    # Executor shutdown can cancel a queued write without cancelling this cleanup task.
                    if asyncio.current_task().cancelling() or not self._sdk_executor_closing:
                        raise
                    await self._persist_seen_during_teardown()
                except RuntimeError:
                    if not self._sdk_executor_closing:
                        raise
                    await self._persist_seen_during_teardown()

    def _dedup_persist_lock_or_create(self) -> asyncio.Lock:
        lock = getattr(self, "_dedup_persist_lock", None)
        if lock is None:
            lock = self._dedup_persist_lock = asyncio.Lock()
        return lock

    def _schedule_ingress_cleanup(self, events):
        if not events:
            return None

        async def release_all():
            for event in events:
                await event._release_unadmitted_claims()

        cleanups = self.__dict__.setdefault("_feishu_ingress_cleanups", set())
        task = asyncio.create_task(release_all())
        cleanups.add(task)
        task.add_done_callback(cleanups.discard)
        return task

    def _begin_buffered_ingress_cleanup(self):
        # Detach before any await: a timed-out disconnect must not later dispatch these inputs.
        for task in (*self._pending_text_batch_tasks.values(), *self._pending_media_batch_tasks.values()):
            if task and not task.done():
                task.cancel()
        events = [*self._pending_text_batches.values(), *self._pending_media_batches.values()]
        self._reset_batch_buffers()
        events.extend(self._take_held_busy_bursts())
        return self._schedule_ingress_cleanup(events)

    def _take_held_busy_bursts(self):
        events = []
        bursts = self.__dict__.get("_shared_bursts", {})
        for key, burst in list(bursts.items()):
            if burst.commit is not None or bursts.get(key) is not burst:
                continue
            del bursts[key]
            self.__dict__.get("_feishu_ingress_busy_bursts", {}).pop(burst, None)
            if burst.timer is not None:
                burst.timer.cancel()
            if not burst.done.done():
                burst.done.cancel()
            if burst.head._ingress_controls_unchanged():
                events.append(burst.head)
        return events
