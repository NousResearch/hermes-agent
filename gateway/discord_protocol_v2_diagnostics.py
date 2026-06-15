"""Read-only diagnostics for Discord Native Multi-Bot Protocol v2.

The helpers in this module are deliberately offline-safe: they inspect config and
SQLite state only, accept optional injected runtime/presence data, and never
connect Discord clients or resolve bot tokens.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from gateway.config import DiscordNativeMultibotConfig
from gateway.discord_protocol_v2_store import (
    SCHEMA_SQL,
    DiscordProtocolV2Store,
    default_db_path,
)
from gateway.secret_refs import redact_secret_ref, redact_sensitive_data


class SecretPresenceProvider(Protocol):
    """Optional redacted existence check for configured secret refs."""

    def has_secret(self, ref: str) -> bool:
        raise NotImplementedError


RuntimeIdentityState = Mapping[str, Mapping[str, Any]]


class _StoreQuerySurface(Protocol):
    db_path: Path
    conn: sqlite3.Connection

    def count_rows(
        self,
        table: str,
        where: str = "",
        params: tuple[Any, ...] = (),
    ) -> int:
        raise NotImplementedError


class _DiagnosticsStore:
    """Tiny read-only query surface used when callers do not inject a store."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()
        if self.db_path.exists():
            uri = f"file:{self.db_path}?mode=ro"
            self.conn = sqlite3.connect(uri, uri=True)
        else:
            self.conn = sqlite3.connect(":memory:")
            self.conn.executescript(SCHEMA_SQL)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        self.conn.close()

    def count_rows(
        self,
        table: str,
        where: str = "",
        params: tuple[Any, ...] = (),
    ) -> int:
        if not table.replace("_", "").isalnum():
            raise ValueError("invalid table name")
        sql = f"SELECT COUNT(*) AS count FROM {table}"
        if where:
            sql += f" WHERE {where}"
        row = self.conn.execute(sql, params).fetchone()
        return int(row["count"])


def build_health_snapshot(
    config: DiscordNativeMultibotConfig,
    *,
    store: DiscordProtocolV2Store | None = None,
    store_path: str | Path | None = None,
    presence_provider: SecretPresenceProvider | None = None,
    runtime_identity_state: RuntimeIdentityState | None = None,
) -> dict[str, Any]:
    """Return a redacted v2 health snapshot without network or token resolution."""

    created_store: _DiagnosticsStore | None = None
    effective_store: _StoreQuerySurface | DiscordProtocolV2Store | None = store
    if effective_store is None:
        created_store = _DiagnosticsStore(store_path)
        effective_store = created_store

    try:
        snapshot = {
            "schema_version": 1,
            "component": "discord_protocol_v2",
            "enabled": config.enabled,
            "mode": config.mode,
            "store_path": str(
                store_path
                if store_path is not None
                else getattr(effective_store, "db_path", default_db_path())
            ),
            "identities": _identity_health(config, runtime_identity_state or {}),
            "state_counts": _state_counts(effective_store),
            "expired_leases": _expired_leases(effective_store),
            "uncertain_sends": _uncertain_sends(effective_store),
            "last_reconciliation": _last_reconciliation(effective_store),
            "secret_refs": _secret_ref_health(config, presence_provider),
            "network": "not_used",
            "token_resolution": "not_used",
        }
        return sanitize_diagnostics(snapshot)
    finally:
        if created_store is not None:
            created_store.close()


def attach_health_snapshot(
    payload: dict[str, Any],
    config: DiscordNativeMultibotConfig,
    *,
    store: DiscordProtocolV2Store | None = None,
    store_path: str | Path | None = None,
    presence_provider: SecretPresenceProvider | None = None,
    runtime_identity_state: RuntimeIdentityState | None = None,
) -> dict[str, Any]:
    """Attach the shared diagnostics core to CLI command output."""

    result = dict(payload)
    result["diagnostics"] = build_health_snapshot(
        config,
        store=store,
        store_path=store_path,
        presence_provider=presence_provider,
        runtime_identity_state=runtime_identity_state,
    )
    return sanitize_diagnostics(result)


def sanitize_diagnostics(value: Any) -> Any:
    """Redact secret refs and token-like values from diagnostic structures."""

    return redact_sensitive_data(value)


def _identity_health(
    config: DiscordNativeMultibotConfig,
    runtime_identity_state: RuntimeIdentityState,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for identity in sorted(config.identities, key=lambda item: item.agent_id):
        runtime = dict(runtime_identity_state.get(identity.agent_id, {}))
        connected = bool(runtime.get("connected", False))
        unhealthy = bool(runtime.get("unhealthy", False))
        if runtime.get("status") in {"unhealthy", "error", "failed"}:
            unhealthy = True
        row = identity.redacted_snapshot()
        row.update(
            {
                "connected": connected,
                "unhealthy": unhealthy,
                "health_status": _identity_status(identity.enabled, connected, unhealthy),
            }
        )
        if runtime.get("status") is not None:
            row["runtime_status"] = runtime.get("status")
        if runtime.get("last_error") is not None:
            row["last_error"] = runtime.get("last_error")
        rows.append(row)

    return {
        "total": len(rows),
        "connected": sum(1 for row in rows if row["connected"]),
        "unhealthy": sum(1 for row in rows if row["unhealthy"]),
        "items": rows,
    }


def _identity_status(enabled: bool, connected: bool, unhealthy: bool) -> str:
    if not enabled:
        return "disabled"
    if unhealthy:
        return "unhealthy"
    if connected:
        return "connected"
    return "not_connected"


def _state_counts(store: _StoreQuerySurface) -> dict[str, Any]:
    return {
        "inbound": _counts_by_status(store, "inbound_deliveries"),
        "outbox": _counts_by_status(store, "outbox_deliveries"),
        "pending_inbound": store.count_rows(
            "inbound_deliveries", "status IN ('pending', 'retryable')"
        ),
        "pending_outbox": store.count_rows("outbox_deliveries", "status = 'pending'"),
    }


def _counts_by_status(store: _StoreQuerySurface, table: str) -> dict[str, int]:
    rows = store.conn.execute(
        f"SELECT status, COUNT(*) AS count FROM {table} GROUP BY status ORDER BY status"
    ).fetchall()
    return {str(row["status"]): int(row["count"]) for row in rows}


def _expired_leases(store: _StoreQuerySurface) -> dict[str, int]:
    now = _now()
    return {
        "inbound": store.count_rows(
            "inbound_deliveries",
            "status = 'leased' AND (lease_until IS NULL OR lease_until <= ?)",
            (now,),
        ),
        "outbox": store.count_rows(
            "outbox_deliveries",
            "status = 'leased' AND (lease_until IS NULL OR lease_until <= ?)",
            (now,),
        ),
    }


def _uncertain_sends(store: _StoreQuerySurface) -> dict[str, Any]:
    rows = store.conn.execute(
        """
        SELECT outbox_delivery_id, idempotency_key, target_agent_id, attempts, updated_at
        FROM outbox_deliveries
        WHERE status = 'uncertain'
        ORDER BY updated_at, outbox_delivery_id
        """
    ).fetchall()
    items = [dict(row) for row in rows]
    return {"count": len(items), "items": items[:20]}


def _last_reconciliation(store: _StoreQuerySurface) -> dict[str, Any] | None:
    row = store.conn.execute(
        """
        SELECT reconciliation_run_id, source_agent_event_id, outbox_delivery_id,
               status, payload_json, created_at, updated_at, version
        FROM reconciliation_runs
        ORDER BY updated_at DESC, created_at DESC, reconciliation_run_id DESC
        LIMIT 1
        """
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["payload"] = _loads_json(result.pop("payload_json", None), {})
    return result


def _secret_ref_health(
    config: DiscordNativeMultibotConfig,
    presence_provider: SecretPresenceProvider | None,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for identity in sorted(config.identities, key=lambda item: item.agent_id):
        provider_error: str | None = None
        if presence_provider is None:
            status = "unresolved"
        else:
            try:
                status = "present" if presence_provider.has_secret(identity.token_secret_ref) else "missing"
            except Exception as exc:  # pragma: no cover - defensive provider boundary
                status = "unresolved"
                provider_error = type(exc).__name__
        item = {
            "agent_id": identity.agent_id,
            "token_secret_ref": redact_secret_ref(identity.token_secret_ref),
            "status": status,
        }
        if status in {"missing", "unresolved"}:
            item["problem"] = status
        if provider_error:
            item["provider_error"] = provider_error
        items.append(item)

    return {
        "missing_or_unresolved": sum(
            1 for item in items if item["status"] in {"missing", "unresolved"}
        ),
        "items": items,
    }


def _loads_json(value: Any, default: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value or ""))
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
