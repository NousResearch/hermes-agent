"""Durable, inert Thousand Sunny route and bucket references."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RoutePolicy:
    """One immutable, versioned, credential-free route reference."""

    policy_ref: str
    policy_version: int
    provider_ref: str
    model_ref: str
    effort_ref: Optional[str]
    created_at: int


def normalize_task_route_references(
    bucket_key: Optional[str],
    route_policy_ref: Optional[str],
    route_policy_version: Optional[int],
) -> tuple[Optional[str], Optional[str], Optional[int]]:
    bucket = str(bucket_key).strip() or None if bucket_key is not None else None
    policy_ref = (
        str(route_policy_ref).strip() or None
        if route_policy_ref is not None
        else None
    )
    version = None
    if route_policy_version is not None:
        version = int(route_policy_version)
        if version < 1:
            raise ValueError("route_policy_version must be >= 1")
    if (policy_ref is None) != (version is None):
        raise ValueError("route_policy_ref and route_policy_version must be set together")
    return bucket, policy_ref, version


def put_route_policy(
    conn,
    *,
    policy_ref: str,
    policy_version: int,
    provider_ref: str,
    model_ref: str,
    effort_ref: Optional[str] = None,
) -> RoutePolicy:
    """Persist an immutable contract version without credentials or entitlement state."""
    values = {
        "policy_ref": str(policy_ref).strip(),
        "provider_ref": str(provider_ref).strip(),
        "model_ref": str(model_ref).strip(),
    }
    if not all(values.values()):
        raise ValueError("policy_ref, provider_ref, and model_ref are required")
    version = int(policy_version)
    if version < 1:
        raise ValueError("policy_version must be >= 1")
    effort = str(effort_ref).strip() if effort_ref is not None else None
    effort = effort or None
    with _kb.write_txn(conn):
        conn.execute(
            "INSERT OR IGNORE INTO sunny_route_policies "
            "(policy_ref, policy_version, provider_ref, model_ref, effort_ref, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                values["policy_ref"],
                version,
                values["provider_ref"],
                values["model_ref"],
                effort,
                int(time.time()),
            ),
        )
        row = conn.execute(
            "SELECT * FROM sunny_route_policies "
            "WHERE policy_ref = ? AND policy_version = ?",
            (values["policy_ref"], version),
        ).fetchone()
        if row is None:
            raise RuntimeError("route policy write could not be read back")
        stored = RoutePolicy(**dict(row))
        if (
            stored.provider_ref != values["provider_ref"]
            or stored.model_ref != values["model_ref"]
            or stored.effort_ref != effort
        ):
            raise ValueError(
                "route policy reference/version already identifies a different contract"
            )
    return stored


def get_route_policy(
    conn, policy_ref: str, policy_version: int
) -> Optional[RoutePolicy]:
    row = conn.execute(
        "SELECT * FROM sunny_route_policies "
        "WHERE policy_ref = ? AND policy_version = ?",
        (policy_ref, int(policy_version)),
    ).fetchone()
    return RoutePolicy(**dict(row)) if row else None


REFERENCE_UNSET = object()


def update_task_route_references(
    conn,
    task_id: str,
    *,
    bucket_key: Any = REFERENCE_UNSET,
    route_policy_ref: Any = REFERENCE_UNSET,
    route_policy_version: Any = REFERENCE_UNSET,
) -> bool:
    """Update inert references without changing task lifecycle state."""
    changed_fields: tuple[str, ...] = ()
    with _kb.write_txn(conn):
        existing = conn.execute(
            "SELECT bucket_key, route_policy_ref, route_policy_version "
            "FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        if existing is None:
            return False
        updates: list[str] = []
        params: list[Any] = []
        normalized: dict[str, Any] = {}
        for column, value in (
            ("bucket_key", bucket_key),
            ("route_policy_ref", route_policy_ref),
            ("route_policy_version", route_policy_version),
        ):
            if value is REFERENCE_UNSET:
                continue
            if column == "route_policy_version" and value is not None:
                value = int(value)
                if value < 1:
                    raise ValueError("route_policy_version must be >= 1")
            elif value is not None:
                value = str(value).strip() or None
            updates.append(f"{column} = ?")
            params.append(value)
            normalized[column] = value
        if not updates:
            return True
        final_ref = normalized.get("route_policy_ref", existing["route_policy_ref"])
        final_version = normalized.get(
            "route_policy_version", existing["route_policy_version"]
        )
        if (final_ref is None) != (final_version is None):
            raise ValueError(
                "route_policy_ref and route_policy_version must be set or cleared together"
            )
        cur = conn.execute(
            f"UPDATE tasks SET {', '.join(updates)} WHERE id = ?",
            (*params, task_id),
        )
        changed_fields = tuple(normalized)
    if cur.rowcount:
        _kb.notify_task_updated(conn, task_id, changed_fields)
    return bool(cur.rowcount)


# Imported last so the facade can import these public entry points at its tail.
from hermes_cli import kanban_db as _kb  # noqa: E402
