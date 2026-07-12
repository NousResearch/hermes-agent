"""Owner-isolated official A2A SDK SQLite task storage."""

from __future__ import annotations

import os
import asyncio
import logging
import stat
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home

from . import auth


class _SanitizingTaskStoreLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = "A2A task store operation"
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        return True


def _install_task_store_log_filter() -> None:
    logger = logging.getLogger("a2a.server.tasks.database_task_store")
    if not any(isinstance(item, _SanitizingTaskStoreLogFilter) for item in logger.filters):
        logger.addFilter(_SanitizingTaskStoreLogFilter())


def tasks_path() -> Path:
    return get_hermes_home() / "a2a" / "tasks.db"


def _authenticated_owner(context: Any) -> str:
    user = getattr(context, "user", None)
    if user is None or not getattr(user, "is_authenticated", False):
        raise PermissionError("authenticated A2A user required for task storage")
    owner = getattr(user, "user_name", "")
    if not isinstance(owner, str) or not owner:
        raise PermissionError("authenticated A2A user has no owner identity")
    return owner


def _prepare_database_path(path: Path) -> None:
    auth._secure_store_directory(auth.credentials_path())
    try:
        info = path.lstat()
    except FileNotFoundError:
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise PermissionError("A2A task database must be a regular file")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise PermissionError("A2A task database has an unexpected owner")


class OwnerAwareTaskStore:
    """Guard the SDK DatabaseTaskStore against cross-owner ID replacement."""

    def __init__(self, delegate: Any, path: Path):
        self._delegate = delegate
        self._path = path
        self._close_lock = asyncio.Lock()
        self._closed = False

    @property
    def task_model(self):
        return self._delegate.task_model

    @property
    def async_session_maker(self):
        return self._delegate.async_session_maker

    async def initialize(self) -> None:
        await self._delegate.initialize()
        try:
            self._path.chmod(0o600)
        except OSError as exc:
            raise PermissionError("A2A task database permissions are unsafe") from exc

    async def save(self, task, context) -> None:
        from sqlalchemy import select, text

        owner = _authenticated_owner(context)
        await self.initialize()
        async with self.async_session_maker() as session:
            await session.execute(text("BEGIN IMMEDIATE"))
            existing = (
                await session.execute(
                    select(self.task_model).where(self.task_model.id == task.id)
                )
            ).scalar_one_or_none()
            if existing is not None and existing.owner != owner:
                await session.rollback()
                raise PermissionError("task ID belongs to a different owner")
            await session.merge(self._delegate._to_orm(task, owner))
            await session.commit()

    async def get(self, task_id, context):
        _authenticated_owner(context)
        return await self._delegate.get(task_id, context)

    async def list(self, params, context):
        _authenticated_owner(context)
        return await self._delegate.list(params, context)

    async def delete(self, task_id, context) -> None:
        _authenticated_owner(context)
        await self._delegate.delete(task_id, context)

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            await self._delegate.engine.dispose()
            self._closed = True


def create_task_store() -> OwnerAwareTaskStore:
    """Create the official SDK DatabaseTaskStore for the active profile."""
    try:
        from sqlalchemy.ext.asyncio import create_async_engine

        from a2a.server.tasks.database_task_store import DatabaseTaskStore
    except ImportError as exc:
        raise RuntimeError("A2A SQLite task storage requires hermes-agent[a2a]") from exc

    path = tasks_path()
    _prepare_database_path(path)
    _install_task_store_log_filter()
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    delegate = DatabaseTaskStore(engine, owner_resolver=_authenticated_owner)
    return OwnerAwareTaskStore(delegate, path)


async def reconcile_orphaned_tasks(store: OwnerAwareTaskStore) -> int:
    """Mark tasks left nonterminal by a prior server process as failed."""
    from sqlalchemy import select

    nonterminal = {
        "TASK_STATE_UNSPECIFIED",
        "TASK_STATE_SUBMITTED",
        "TASK_STATE_WORKING",
        "TASK_STATE_INPUT_REQUIRED",
        "TASK_STATE_AUTH_REQUIRED",
    }
    await store.initialize()
    reconciled = 0
    async with store.async_session_maker.begin() as session:
        rows = (await session.execute(select(store.task_model))).scalars().all()
        for row in rows:
            status = row.status if isinstance(row.status, dict) else {}
            if status.get("state") not in nonterminal:
                continue
            row.status = {"state": "TASK_STATE_FAILED"}
            metadata = row.task_metadata if isinstance(row.task_metadata, dict) else {}
            row.task_metadata = {**metadata, "interrupted": "server_restart"}
            row.last_updated = datetime.now(UTC)
            reconciled += 1
    return reconciled
