"""Matrix sync responses, checkpoints and dispatch lifecycle."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Coroutine
from typing import TYPE_CHECKING, Any, Dict, Optional

from gateway.platforms.base import BasePlatformAdapter
from plugins.platforms.matrix.unread import MatrixUnreadState, SYNC_FILTER
from plugins.platforms.matrix.sync_transport import (
    DurableSyncStore,
    SyncCheckpoints,
    SyncDispatch,
    is_invalid_sync_cursor,
)


class MatrixSyncMixin(BasePlatformAdapter):
    _client: Any
    _unread: MatrixUnreadState
    _closing: bool
    _sync_position: str | None
    _sync_checkpoints: SyncCheckpoints | None
    _resuming_sync: bool
    _last_sync_ts: float
    _joined_rooms: set[str]
    _refresh_dm_cache: Callable[[], Awaitable[None]]
    _schedule_pending_invite_joins: Callable[[dict[str, Any]], None]
    _rewind_failed_intake: Callable[[Any], Awaitable[bool]]
    _forget_processed_event: Callable[[str], None]
    _on_room_message: Callable[
        [Any], Coroutine[Any, Any, asyncio.Future[bool] | bool | None]
    ]
    _on_reaction: Callable[[Any], Awaitable[bool | None]]

    if TYPE_CHECKING:

        def _invalidate_room_identities(self, room_id: str | None = None) -> None: ...

    async def _connect_initial_sync(self, client: Any) -> None:
        """Full initial sync: seed joined rooms, DM cache, and dispatch queued to-device events."""
        from plugins.platforms.matrix.adapter import logger

        try:
            since = await client.sync_store.get_next_batch()
            self._resuming_sync = bool(since)
            try:
                sync_data = await client.sync(
                    since=since, timeout=10000, full_state=True, filter_id=SYNC_FILTER
                )
            except Exception as exc:
                if not since or not is_invalid_sync_cursor(exc):
                    raise
                logger.warning(
                    "Matrix: saved sync cursor was rejected; refreshing full state"
                )
                # A full sync returns recent history that was handled before the restart.
                self._resuming_sync = False
                sync_data = await client.sync(
                    timeout=10000, full_state=True, filter_id=SYNC_FILTER
                )
            if isinstance(sync_data, dict):
                self._joined_rooms.clear()
                await self._absorb_sync(client, sync_data, initial=True)
            else:
                raise TypeError(
                    f"Matrix: initial sync returned unexpected type {type(sync_data).__name__}"
                )
        except Exception as exc:
            logger.warning("Matrix: initial sync error: %s", exc)
            raise

    async def _sync_loop(self) -> None:
        from plugins.platforms.matrix.adapter import (
            _is_permanent_matrix_auth_error,
            logger,
        )

        client = self._client
        next_batch = self._sync_position or await client.sync_store.get_next_batch()
        while not self._closing:
            try:
                if await self._rewind_failed_intake(client):
                    next_batch = self._sync_position
                    await asyncio.sleep(5)
                    continue
                # 45s outer cap guards TCP-level hangs the 30s long-poll timeout cannot catch.
                # mautrix raises on every non-2xx, so a non-dict here is never an error object.
                sync_data = await asyncio.wait_for(
                    client.sync(since=next_batch, timeout=30000, filter_id=SYNC_FILTER),
                    timeout=45.0,
                )
                if isinstance(sync_data, dict):
                    next_batch = (
                        await self._absorb_sync(client, sync_data) or next_batch
                    )
                    await asyncio.sleep(
                        0
                    )  # let fresh invite joins start before the next sync
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._closing:
                    return
                # Detect permanent auth/permission failures. Transient 5xx outages must retry.
                if _is_permanent_matrix_auth_error(exc):
                    logger.error("Matrix: permanent auth error, stopping sync: %s", exc)
                    return
                if next_batch and is_invalid_sync_cursor(exc):
                    try:
                        await self._connect_initial_sync(client)
                        next_batch = self._sync_position
                        continue
                    except Exception:
                        if self._closing:
                            return
                logger.warning("Matrix: sync error: %s — retrying in 5s", exc)
                await asyncio.sleep(5)

    async def _absorb_sync(
        self, client: Any, sync_data: Dict[str, Any], *, initial: bool = False
    ) -> Optional[str]:
        """Apply one sync response: joined rooms, next_batch, event dispatch, pending invites. Returns next_batch.
        The initial (full-state) sync also seeds the DM cache and dispatches so the OlmMachine sees
        to-device key shares queued while offline."""
        from plugins.platforms.matrix.adapter import logger

        self._last_sync_ts = time.time()
        rooms_join = sync_data.get("rooms", {}).get("join", {})
        if rooms_join or initial:
            self._joined_rooms.update(rooms_join.keys())
            self._invalidate_room_identities()
        nb = sync_data.get("next_batch")  # incremental syncs resume from here
        if initial:
            await self._refresh_dm_cache()
        if client is self._client:
            self._unread.observe(client, sync_data, initial=initial)
            self._joined_rooms.difference_update(
                sync_data.get("rooms", {}).get("leave", {})
            )
        await self._dispatch_sync(sync_data)
        self._schedule_pending_invite_joins(sync_data)
        if nb:
            dispatch = getattr(client, "hermes_sync", None)
            store = client.sync_store
            if isinstance(dispatch, SyncDispatch) and isinstance(
                store, DurableSyncStore
            ):
                seen, buffered = dispatch.take_intakes()
                checkpoints = self._sync_checkpoints
                if checkpoints is None or checkpoints.store is not store:
                    checkpoints = self._sync_checkpoints = SyncCheckpoints(
                        store, dispatch.dispatching_intakes
                    )
                await checkpoints.commit(nb, seen, buffered)
            else:
                await store.put_next_batch(nb)
            if isinstance(dispatch, SyncDispatch):
                dispatch.acknowledge()
            self._sync_position = nb
            self._resuming_sync = True
        if initial:
            logger.info(
                "Matrix: initial dispatch checkpoint complete, joined %d rooms",
                len(self._joined_rooms),
            )
        return nb

    async def _dispatch_sync(self, sync_data: Dict[str, Any]) -> None:
        """Dispatch a sync response through the mautrix event machinery."""
        from plugins.platforms.matrix.adapter import logger

        client = self._client
        if not client or not hasattr(client, "handle_sync"):
            return
        dispatch = getattr(client, "hermes_sync", None)
        if isinstance(dispatch, SyncDispatch):
            try:
                await dispatch.dispatch_sync(sync_data)
            finally:
                for handler, event_id in dispatch.failed_sync_handlers:
                    if handler in {self._on_room_message, self._on_reaction}:
                        self._forget_processed_event(event_id)
            return
        tasks = client.handle_sync(sync_data)
        if inspect.isawaitable(tasks):
            tasks = await tasks
        if tasks:
            # return_exceptions=True: one failing handler must not drop its SIBLING events.
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(
                        "Matrix: event handler failed during sync dispatch: %s", result
                    )
            for result in results:
                if isinstance(result, BaseException):
                    raise result
