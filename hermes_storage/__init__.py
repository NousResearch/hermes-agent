# hermes_storage — cloud storage adapters for SaaS-mode Hermes.
#
# Public API:
#   get_backend()       — async factory, returns the singleton StorageBackend.
#   reset_backend()     — reset singleton (used in tests).
#   StorageBackend      — Protocol (runtime_checkable) for type checks.
#   SQLiteBackend       — Local dev backend wrapping hermes_state.SessionDB.
#   NeonBackend         — Cloud backend using asyncpg + Neon PostgreSQL.
#
# Selection logic:
#   HERMES_MODE=saas  → NeonBackend  (requires NEON_DATABASE_URL or Secrets Manager)
#   anything else     → SQLiteBackend (local dev; no cloud creds required)

from __future__ import annotations

import logging
import os
from typing import Optional

from hermes_storage.backend import StorageBackend
from hermes_storage.sqlite_backend import SQLiteBackend

logger = logging.getLogger(__name__)

_backend: Optional[StorageBackend] = None


async def get_backend() -> StorageBackend:
    """
    Return the singleton StorageBackend, initialising it on first call.

    Thread-safety: this is NOT thread-safe across concurrent asyncio coroutines
    on first call.  The expected pattern is one call at startup (e.g. from an
    asyncio.on_startup hook) before the gateway begins serving requests.
    Subsequent calls return the cached singleton without I/O.

    Failure modes:
    - HERMES_MODE=saas but NEON_DATABASE_URL absent and Secrets Manager
      unreachable → RuntimeError from NeonBackend._resolve_dsn.
    - asyncpg pool creation fails (bad DSN, network unreachable) → asyncpg
      exception propagates to caller.  The gateway startup hook should treat
      this as fatal and refuse to start rather than serving with no persistence.
    """
    global _backend
    if _backend is None:
        mode = os.environ.get("HERMES_MODE", "local")
        if mode == "saas":
            logger.info("hermes_storage: HERMES_MODE=saas — selecting NeonBackend")
            from hermes_storage.neon_backend import NeonBackend  # lazy import
            backend = NeonBackend()
            await backend.initialize()
            _backend = backend
        else:
            logger.info(
                "hermes_storage: HERMES_MODE=%r (not 'saas') — selecting SQLiteBackend",
                mode,
            )
            _backend = SQLiteBackend()
    return _backend


async def reset_backend() -> None:
    """
    Close and discard the singleton backend.

    Used in tests to ensure each test gets a fresh backend.  Also useful for
    graceful shutdown before process exit.
    """
    global _backend
    if _backend is not None:
        await _backend.close()
        _backend = None


__all__ = [
    "get_backend",
    "reset_backend",
    "StorageBackend",
    "SQLiteBackend",
]
