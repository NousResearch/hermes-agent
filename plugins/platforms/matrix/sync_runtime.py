"""Matrix sync absorption and event dispatch."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from plugins.platforms.matrix.adapter import MatrixAdapter

logger = logging.getLogger("plugins.platforms.matrix.adapter")


class MatrixSyncMixin:
    async def _connect_initial_sync(self: MatrixAdapter, client: Any) -> None:
        """Full initial sync: seed joined rooms, DM cache, and dispatch queued to-device events."""
        try:
            sync_data = await client.sync(timeout=10000, full_state=True)
            if isinstance(sync_data, dict):
                self._joined_rooms.clear()
                await self._absorb_sync(client, sync_data, initial=True)
            else:
                logger.warning("Matrix: initial sync returned unexpected type %s", type(sync_data).__name__)
        except Exception as exc:
            logger.warning("Matrix: initial sync error: %s", exc)


    async def _sync_loop(self: MatrixAdapter) -> None:
        from plugins.platforms.matrix.adapter import _is_permanent_matrix_auth_error
        client = self._client
        next_batch = await client.sync_store.get_next_batch()  # resume from the initial sync
        while not self._closing:
            try:
                # 45s outer cap guards TCP-level hangs the 30s long-poll timeout cannot catch.
                # mautrix raises on every non-2xx, so a non-dict here is never an error object.
                sync_data = await asyncio.wait_for(client.sync(since=next_batch, timeout=30000), timeout=45.0)
                if isinstance(sync_data, dict):
                    next_batch = await self._absorb_sync(client, sync_data) or next_batch
                    await asyncio.sleep(0)  # let fresh invite joins start before the next sync
            except asyncio.CancelledError:
                return
            except Exception as exc:
                if self._closing:
                    return
                # Detect permanent auth/permission failures. Transient 5xx outages must retry.
                if _is_permanent_matrix_auth_error(exc):
                    logger.error("Matrix: permanent auth error, stopping sync: %s", exc)
                    return
                logger.warning("Matrix: sync error: %s — retrying in 5s", exc)
                await asyncio.sleep(5)


    async def _absorb_sync(self: MatrixAdapter, client: Any, sync_data: Dict[str, Any], *, initial: bool = False) -> Optional[str]:
        """Apply one sync response: joined rooms, next_batch, event dispatch, pending invites. Returns next_batch.
        The initial (full-state) sync also seeds the DM cache and dispatches so the OlmMachine sees
        to-device key shares queued while offline."""
        self._permalink_routing.observe_sync(client, sync_data)
        self._last_sync_ts = time.time()
        rooms_join = sync_data.get("rooms", {}).get("join", {})
        if rooms_join or initial:
            self._joined_rooms.update(rooms_join.keys())
            self._invalidate_room_identities()
        nb = sync_data.get("next_batch")  # incremental syncs resume from here
        if nb:
            await client.sync_store.put_next_batch(nb)
        if initial:
            logger.info("Matrix: initial sync complete, joined %d rooms", len(self._joined_rooms))
            await self._refresh_dm_cache()
        try:
            await self._dispatch_sync(sync_data)
        except Exception as exc:
            logger.warning("Matrix: %s: %s", "initial sync event dispatch error" if initial else "sync event dispatch error", exc)
        self._schedule_pending_invite_joins(sync_data)
        return nb


    async def _dispatch_sync(self: MatrixAdapter, sync_data: Dict[str, Any]) -> None:
        """Dispatch a sync response through the mautrix event machinery."""
        client = self._client
        if not client or not hasattr(client, "handle_sync"):
            return
        tasks = client.handle_sync(sync_data)
        if inspect.isawaitable(tasks):
            tasks = await tasks
        if tasks:
            # return_exceptions=True: one failing handler must not drop its SIBLING events.
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Matrix: event handler failed during sync dispatch: %s", result)
