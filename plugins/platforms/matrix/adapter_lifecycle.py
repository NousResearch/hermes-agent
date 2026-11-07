"""Admission and quiescence for Matrix SDK callbacks and outbound operations."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from functools import wraps


logger = logging.getLogger(__name__)
# A failed ephemeral adapter must outlive its caller so native lock handles stay open.
_failed_owners = set()


@asynccontextmanager
async def matrix_activity(adapter):
    task = asyncio.current_task()
    active = getattr(adapter, "_matrix_active_tasks", None)
    if active is None:
        active = adapter._matrix_active_tasks = {}
    if adapter._closing and task not in active:
        raise RuntimeError("Matrix adapter is closing")
    active[task] = active.get(task, 0) + 1
    try:
        yield
    finally:
        active[task] -= 1
        if not active[task]:
            del active[task]


def track_matrix_activity(adapter, method):
    @wraps(method)
    async def tracked(*args, **kwargs):
        async with matrix_activity(adapter):
            return await method(*args, **kwargs)
    return tracked


def matrix_operation(method):
    @wraps(method)
    async def tracked(self, *args, **kwargs):
        async with matrix_activity(self):
            return await method(self, *args, **kwargs)
    return tracked


class MatrixLifecycleMixin:
    _matrix_activity = matrix_activity

    def _track_matrix_client(self, client):
        # mautrix schedules wait_sync=False handlers outside handle_sync's return list.
        client._catch_errors = track_matrix_activity(self, client._catch_errors)

    async def _drain_matrix_tasks(self):
        self._closing = True
        sync = self._sync_task
        if sync is not None and not sync.done():
            sync.cancel()
            await asyncio.gather(sync, return_exceptions=True)
        for tasks in (self._invite_join_tasks.values(), self._reaction_redaction_tasks):
            pending = list(tasks)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        self._invite_join_tasks.clear()
        self._reaction_redaction_tasks.clear()
        while active := set(getattr(self, "_matrix_active_tasks", {})) - {asyncio.current_task()}:
            _, pending = await asyncio.wait(active, timeout=30)
            if pending:
                raise RuntimeError("Matrix operations did not quiesce; retaining crypto ownership")

    async def _close_matrix_resources(self, api, crypto_db=None, *, close_http=True):
        _failed_owners.add(self)
        errors = []
        try:
            await self._drain_matrix_tasks()
        except Exception as exc:
            logger.error("Matrix cleanup failed", exc_info=True)
            errors.append(exc)
        if not errors and crypto_db is not None:
            try:
                await crypto_db.stop()
                self._crypto_db = None
            except Exception as exc:
                logger.error("Matrix resource close failed", exc_info=True)
                errors.append(exc)
        if close_http and api is not None:
            try:
                await api.session.close()
            except Exception as exc:
                logger.error("Matrix resource close failed", exc_info=True)
                errors.append(exc)
        if errors:
            raise ExceptionGroup("Matrix shutdown failed; retaining crypto ownership", errors)
        self._release_matrix_store_lock()
        _failed_owners.discard(self)

    async def disconnect(self):
        client = self._client
        await self._close_matrix_resources(client.api if client else None, self._crypto_db)
        self._client = None
