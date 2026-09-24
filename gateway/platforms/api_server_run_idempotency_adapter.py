"""Backend-neutral run-idempotency store port, factories, and runtime selector.

``selected_run_idempotency_store_factory`` resolves the active state-store backend and
returns ``None`` for SQLite (callers keep the legacy module path) or a zero-arg factory
that builds a fresh configured PostgreSQL store otherwise.  A PostgreSQL selection with a
missing/unreachable DSN raises instead of silently degrading to SQLite.  Importing this
module never opens a database.
"""
from __future__ import annotations

from typing import Any


def open_run_idempotency_store(*, backend: str = "sqlite", dsn: str | None = None, **kwargs: Any):
    """Open an isolated store for conformance tests; never changes runtime routing."""
    if backend == "postgresql":
        if not dsn:
            raise ValueError("PostgreSQL run-idempotency store requires a DSN")
        from gateway.platforms.api_server_run_idempotency_postgresql import (
            PostgreSQLRunIdempotencyStore)
        return PostgreSQLRunIdempotencyStore(dsn, **kwargs)
    if backend == "sqlite":
        from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore
        return RunIdempotencyStore()
    raise ValueError("run-idempotency store backend must be sqlite or postgresql")


def open_configured_run_idempotency_store(config: dict[str, Any], *, secret_lookup=None, **kwargs: Any):
    """Explicit factory for an injected consumer, never a global runtime selector.

    Shares only the validated state-store DSN policy.  The caller controls lifetime and
    injection.  SQLite selection returns ``None`` (no PG store to open).
    """
    from state_store import resolve_state_store_config
    resolved = resolve_state_store_config(config, secret_lookup=secret_lookup)
    if resolved.backend == "sqlite":
        return None
    assert resolved.postgresql is not None
    lookup = secret_lookup
    if lookup is None:
        from state_store import _scoped_secret as lookup  # type: ignore[attr-defined]
    dsn = str(lookup(resolved.postgresql.dsn_env) or "").strip()
    if not dsn:
        raise ValueError("PostgreSQL run-idempotency store requires configured DSN")
    from gateway.platforms.api_server_run_idempotency_postgresql import (
        RunIdempotencyPostgreSQLConfig)
    return open_run_idempotency_store(
        backend="postgresql", dsn=dsn,
        settings=RunIdempotencyPostgreSQLConfig(
            connect_timeout_seconds=resolved.postgresql.connect_timeout_seconds,
            pool_max_size=resolved.postgresql.pool_max_size,
        ), **kwargs,
    )


def selected_run_idempotency_store_factory():
    """Resolve the runtime run-idempotency store factory from the selected state-store backend.

    ``None`` means SQLite is selected and the caller keeps its legacy module path.
    Otherwise a zero-arg factory building a fresh PostgreSQL store is returned (each caller
    owns its store/lifetime; no instance caching).  A PostgreSQL selection with a
    missing/unreachable DSN raises instead of degrading to SQLite.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except Exception:
        config = {}
    from state_store import resolve_state_store_config
    resolved = resolve_state_store_config(config)
    if resolved.backend == "sqlite":
        return None
    assert resolved.postgresql is not None
    from state_store import _scoped_secret
    dsn = str(_scoped_secret(resolved.postgresql.dsn_env) or "").strip()
    if not dsn:
        raise ValueError("PostgreSQL run-idempotency store requires configured DSN")
    connect_timeout_seconds = resolved.postgresql.connect_timeout_seconds
    pool_max_size = resolved.postgresql.pool_max_size
    from gateway.platforms.api_server_run_idempotency_postgresql import (
        PostgreSQLRunIdempotencyStore, RunIdempotencyPostgreSQLConfig)

    def factory():
        return PostgreSQLRunIdempotencyStore(
            dsn, settings=RunIdempotencyPostgreSQLConfig(
                connect_timeout_seconds=connect_timeout_seconds,
                pool_max_size=pool_max_size))

    return factory
