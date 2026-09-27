"""Backend-neutral gateway delivery-ledger port, factories, and runtime selector.

``selected_delivery_ledger`` resolves the active state-store backend and returns the
matching ledger: ``None`` for SQLite (the gateway keeps its legacy module path), or a
configured PostgreSQL ledger otherwise. A PostgreSQL selection with a missing/unreachable
DSN raises instead of silently degrading to SQLite. Every other entry point here is an
explicit factory whose caller controls lifetime and injection; importing this module
never opens a database.
"""
from __future__ import annotations
from typing import Any, Protocol, cast, runtime_checkable

from hermes_constants import get_hermes_home


@runtime_checkable
class DeliveryLedger(Protocol):
    """The complete final-delivery transition surface; no DB handles escape it."""
    def record_obligation(self, **kwargs: Any) -> Any: ...
    def mark_attempting(self, receipt: Any) -> Any: ...
    def mark_delivered(self, receipt: Any) -> bool: ...
    def mark_failed(self, receipt: Any, error: str = "") -> bool: ...
    def release_runtime_claim(self, receipt: Any, error: str = "") -> bool: ...


def open_delivery_ledger(*, backend: str = "sqlite", dsn: str | None = None, **kwargs: Any):
    """Open an isolated adapter for conformance tests; never changes gateway routing."""
    if backend == "postgresql":
        if not dsn:
            raise ValueError("PostgreSQL DeliveryLedger requires a DSN")
        from gateway.delivery_ledger_postgresql import PostgreSQLDeliveryLedger
        return PostgreSQLDeliveryLedger(dsn, **kwargs)
    if backend == "sqlite":
        from gateway import delivery_ledger
        return SqliteDeliveryLedger()
    raise ValueError("delivery ledger backend must be sqlite or postgresql")


def open_configured_delivery_ledger(config: dict[str, Any], *, secret_lookup=None, **kwargs: Any) -> DeliveryLedger:
    """Explicit factory for an injected consumer, never a global runtime selector.

    It intentionally shares only the validated state-store DSN policy.  The
    caller controls lifetime and injection, which prevents selected PostgreSQL
    configuration from opening legacy ``state.db`` or activating gateway runtime.
    """
    from state_store import resolve_state_store_config
    resolved = resolve_state_store_config(config, secret_lookup=secret_lookup)
    if resolved.backend == "sqlite":
        return SqliteDeliveryLedger()
    assert resolved.postgresql is not None
    lookup = secret_lookup
    if lookup is None:
        from state_store import _scoped_secret as lookup  # type: ignore[attr-defined]
    dsn = str(lookup(resolved.postgresql.dsn_env) or "").strip()
    if not dsn:
        raise ValueError("PostgreSQL DeliveryLedger requires configured DSN")
    from gateway.delivery_ledger_postgresql import DeliveryLedgerPostgreSQLConfig
    return cast(DeliveryLedger, open_delivery_ledger(
        backend="postgresql", dsn=dsn,
        settings=DeliveryLedgerPostgreSQLConfig(
            connect_timeout_seconds=resolved.postgresql.connect_timeout_seconds,
            pool_max_size=resolved.postgresql.pool_max_size,
        ), **kwargs,
    ))


_SELECTED_LEDGER_CACHE: dict[str, DeliveryLedger | None] = {}


def selected_delivery_ledger() -> DeliveryLedger | None:
    """Resolve the runtime delivery ledger from the selected state-store backend.

    ``None`` means SQLite is selected and the gateway keeps its legacy module path.
    Otherwise the configured PostgreSQL ledger is opened (and cached per Hermes home),
    raising on a missing or unreachable DSN rather than degrading to SQLite.
    """
    home = str(get_hermes_home())
    if home in _SELECTED_LEDGER_CACHE:
        return _SELECTED_LEDGER_CACHE[home]
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except Exception:
        config = {}
    from state_store import resolve_state_store_config
    if resolve_state_store_config(config).backend == "sqlite":
        result: DeliveryLedger | None = None
    else:
        result = open_configured_delivery_ledger(config)
    _SELECTED_LEDGER_CACHE[home] = result
    return result


class SqliteDeliveryLedger:
    """Compatibility object for direct tests over the unchanged module contract."""
    def record_obligation(self, **kwargs):
        from gateway import delivery_ledger
        return delivery_ledger.record_obligation(**kwargs)
    def mark_attempting(self, receipt):
        from gateway import delivery_ledger
        return delivery_ledger.mark_attempting(receipt)
    def mark_delivered(self, receipt):
        from gateway import delivery_ledger
        return delivery_ledger.mark_delivered(receipt)
    def mark_failed(self, receipt, error=""):
        from gateway import delivery_ledger
        return delivery_ledger.mark_failed(receipt, error)
    def release_runtime_claim(self, receipt, error=""):
        from gateway import delivery_ledger
        return delivery_ledger.release_runtime_claim(receipt, error)
    def close(self):
        return None
