"""Shared/profile state coordination for RoomLink grants."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def grant_state_db_paths(profile_home: Path | str | None = None) -> tuple[Path, ...]:
    """Return the shared install DB and active profile DB without duplicates."""

    from gateway.hosted_rooms import default_db_path
    from hermes_constants import get_hermes_home

    shared = Path(default_db_path())
    home = Path(profile_home) if profile_home is not None else Path(get_hermes_home())
    profile = home / "state.db"
    return (shared,) if profile == shared else (shared, profile)


def reserve_grant_state(
    db_paths: Iterable[Path | str],
    *,
    claims: Mapping[str, Any],
    expires_at: float,
) -> None:
    """Reserve every enforcing store, compensating any partial write."""

    from gateway.hosted_rooms import (
        reserve_peer_room,
        restore_peer_room_reservations,
    )

    committed: list[tuple[Path | str, Mapping[str, Any]]] = []
    try:
        for db_path in db_paths:
            snapshot = reserve_peer_room(
                db_path,
                claims=claims,
                expires_at=expires_at,
            )
            committed.append((db_path, snapshot))
    except Exception:
        rollback_error: Exception | None = None
        for db_path, snapshot in reversed(committed):
            try:
                restore_peer_room_reservations(
                    db_path,
                    claims=claims,
                    snapshot=snapshot,
                )
            except Exception as exc:
                if rollback_error is None:
                    rollback_error = exc
        if rollback_error is not None:
            raise RuntimeError(
                "peer room reservation failed and could not be rolled back"
            ) from rollback_error
        raise


def revoke_grant_state(
    db_paths: Iterable[Path | str],
    *,
    claims: Mapping[str, Any],
    expires_at: float,
    exact: bool = False,
) -> None:
    """Best-effort every store, then report the first revocation failure."""

    from gateway.hosted_rooms import (
        revoke_room_grant_id,
        revoke_room_grant_scope,
    )

    revoke = revoke_room_grant_id if exact else revoke_room_grant_scope

    first_error: Exception | None = None
    for db_path in db_paths:
        try:
            revoke(
                db_path,
                claims=claims,
                expires_at=expires_at,
            )
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error
