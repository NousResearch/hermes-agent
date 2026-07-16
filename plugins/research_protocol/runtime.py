"""Explicit local runtime construction for the Research Protocol plugin."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from .approval import ApprovalService
from .planner_tools import PlannerToolHandlers
from .storage.artifacts import ArtifactStore
from .storage.postgres import PostgresApprovalStore, PostgresContextReader

READER_DATABASE_URL_ENV = "RESEARCH_PROTOCOL_READER_DATABASE_URL"
WRITER_DATABASE_URL_ENV = "RESEARCH_PROTOCOL_WRITER_DATABASE_URL"


class RuntimeConfigurationError(ValueError):
    """Raised when explicit plugin authority is missing or malformed."""


class _LazyAcquire:
    def __init__(self, owner: "LazyAsyncpgPool"):
        self._owner = owner
        self._context: Any = None

    async def __aenter__(self) -> Any:
        pool = await self._owner.get_pool()
        self._context = pool.acquire()
        return await self._context.__aenter__()

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        if self._context is None:
            return False
        return await self._context.__aexit__(exc_type, exc, traceback)


class LazyAsyncpgPool:
    """Create an asyncpg pool only inside the active Hermes event loop."""

    def __init__(
        self,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 4,
        command_timeout: float = 5.0,
    ):
        self._dsn = dsn
        self._min_size = min_size
        self._max_size = max_size
        self._command_timeout = command_timeout
        self._pool: Any = None
        self._lock: asyncio.Lock | None = None

    def __repr__(self) -> str:
        return "LazyAsyncpgPool(dsn=[REDACTED], created=%r)" % self.created

    @property
    def created(self) -> bool:
        return self._pool is not None

    async def get_pool(self) -> Any:
        if self._pool is not None:
            return self._pool
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._pool is None:
                try:
                    import asyncpg
                except ImportError as exc:
                    raise RuntimeConfigurationError(
                        "asyncpg is required for Research Protocol database tools"
                    ) from exc
                self._pool = await asyncpg.create_pool(
                    dsn=self._dsn,
                    min_size=self._min_size,
                    max_size=self._max_size,
                    command_timeout=self._command_timeout,
                    server_settings={
                        "application_name": "hermes-research-protocol",
                    },
                )
        return self._pool

    def acquire(self) -> _LazyAcquire:
        return _LazyAcquire(self)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


@dataclass(frozen=True)
class RuntimeBundle:
    handlers: PlannerToolHandlers
    artifact_available: bool
    context_available: bool
    approval_available: bool
    reader_pool: LazyAsyncpgPool | None
    writer_pool: LazyAsyncpgPool | None


def _bounded_int(
    config: Mapping[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeConfigurationError(f"database.{key} must be an integer")
    if not minimum <= value <= maximum:
        raise RuntimeConfigurationError(
            f"database.{key} must be between {minimum} and {maximum}"
        )
    return value


def _bounded_float(
    config: Mapping[str, Any],
    key: str,
    default: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeConfigurationError(f"database.{key} must be numeric")
    result = float(value)
    if not minimum <= result <= maximum:
        raise RuntimeConfigurationError(
            f"database.{key} must be between {minimum} and {maximum}"
        )
    return result


def build_runtime_bundle(
    plugin_config: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> RuntimeBundle:
    """Build bounded local dependencies without opening a database connection."""
    if not isinstance(plugin_config, Mapping):
        raise RuntimeConfigurationError("plugin configuration must be a mapping")
    root_value = plugin_config.get("artifact_root")
    if not isinstance(root_value, str) or not root_value.strip():
        raise RuntimeConfigurationError("an absolute artifact_root is required")
    root = Path(root_value).expanduser()
    if not root.is_absolute():
        raise RuntimeConfigurationError("an absolute artifact_root is required")

    artifact_store = ArtifactStore(root)
    environment = os.environ if environ is None else environ
    reader_dsn = environment.get(READER_DATABASE_URL_ENV, "").strip()
    writer_dsn = environment.get(WRITER_DATABASE_URL_ENV, "").strip()
    if reader_dsn and writer_dsn and reader_dsn == writer_dsn:
        raise RuntimeConfigurationError(
            "reader and writer database DSNs must be distinct"
        )
    if not reader_dsn and not writer_dsn:
        return RuntimeBundle(
            handlers=PlannerToolHandlers(
                artifact_store=artifact_store,
                context_reader=None,
                approval_service=None,
            ),
            artifact_available=True,
            context_available=False,
            approval_available=False,
            reader_pool=None,
            writer_pool=None,
        )

    raw_database = plugin_config.get("database", {})
    if not isinstance(raw_database, Mapping):
        raise RuntimeConfigurationError("database configuration must be a mapping")
    max_rows = _bounded_int(
        raw_database,
        "max_rows",
        100,
        minimum=1,
        maximum=1000,
    )
    max_bytes = _bounded_int(
        raw_database,
        "max_bytes",
        1_048_576,
        minimum=256,
        maximum=10_485_760,
    )
    timeout_seconds = _bounded_float(
        raw_database,
        "timeout_seconds",
        5.0,
        minimum=0.1,
        maximum=30.0,
    )
    pool_min_size = _bounded_int(
        raw_database,
        "pool_min_size",
        1,
        minimum=1,
        maximum=4,
    )
    pool_max_size = _bounded_int(
        raw_database,
        "pool_max_size",
        4,
        minimum=1,
        maximum=8,
    )
    if pool_min_size > pool_max_size:
        raise RuntimeConfigurationError(
            "database.pool_min_size must not exceed database.pool_max_size"
        )

    reader_pool = (
        LazyAsyncpgPool(
            reader_dsn,
            min_size=pool_min_size,
            max_size=pool_max_size,
            command_timeout=timeout_seconds,
        )
        if reader_dsn
        else None
    )
    writer_pool = (
        LazyAsyncpgPool(
            writer_dsn,
            min_size=pool_min_size,
            max_size=pool_max_size,
            command_timeout=timeout_seconds,
        )
        if writer_dsn
        else None
    )
    context_reader = (
        PostgresContextReader(
            reader_pool,
            max_rows=max_rows,
            max_bytes=max_bytes,
            timeout_seconds=timeout_seconds,
        )
        if reader_pool is not None
        else None
    )
    approval_service = (
        ApprovalService(
            PostgresApprovalStore(
                writer_pool,
                timeout_seconds=timeout_seconds,
            ),
            surface="research-protocol",
        )
        if writer_pool is not None
        else None
    )
    return RuntimeBundle(
        handlers=PlannerToolHandlers(
            artifact_store=artifact_store,
            context_reader=context_reader,
            approval_service=approval_service,
        ),
        artifact_available=True,
        context_available=reader_pool is not None,
        approval_available=writer_pool is not None,
        reader_pool=reader_pool,
        writer_pool=writer_pool,
    )


__all__ = [
    "LazyAsyncpgPool",
    "READER_DATABASE_URL_ENV",
    "RuntimeBundle",
    "RuntimeConfigurationError",
    "WRITER_DATABASE_URL_ENV",
    "build_runtime_bundle",
]
