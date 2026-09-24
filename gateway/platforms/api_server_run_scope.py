"""Exact participant identity and the existing Runs ownership-key encoding."""

import hashlib
from collections.abc import Mapping
from typing import Any

from gateway.hosted_room_peer import _identifier
from gateway.hosted_rooms_common import bounded_int

ROOM_RUN_SCOPE_FIELDS = (
    "room_id", "home_install_id", "authority_gateway_id", "authority_epoch",
    "member_id", "target_install_id", "target_profile",
)


def room_run_scope_key(identity: Mapping[str, Any]) -> str:
    """Preserve the legacy seven-field NUL-separated string hash, not grant JSON."""
    return hashlib.sha256("\0".join(str(identity[key]) for key in ROOM_RUN_SCOPE_FIELDS).encode()).hexdigest()


def validate_room_run_scope(identity: Any) -> dict[str, Any]:
    """Validate owner input without coercion, trimming, extra fields or NUL aliases."""
    if not isinstance(identity, Mapping) or set(identity) != set(ROOM_RUN_SCOPE_FIELDS):
        raise ValueError("room run scope must contain exactly the seven identity fields")
    result = {}
    for field in ROOM_RUN_SCOPE_FIELDS:
        value = identity[field]
        if field == "authority_epoch":
            result[field] = bounded_int(
                value, error=ValueError, message="authority_epoch must be a positive integer",
                low=1, high=2**63 - 1,
            )
        else:
            if type(value) is not str or "\0" in value or value != value.strip():
                raise ValueError(f"{field} must be an exact identifier string")
            result[field] = _identifier(value, field=field)
    return result
