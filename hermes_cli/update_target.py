"""Strict identity validation for pinned update intent."""

from __future__ import annotations

from dataclasses import dataclass
import re


_HEX = re.compile(r"^[0-9a-f]+$")


@dataclass(frozen=True)
class TargetRequest:
    """The exact checkout identity a pinned update request names."""

    revision: str
    install_id: str
    current_sha: str


def _validate_field(name: str, value: object, length: int) -> str:
    if not isinstance(value, str) or len(value) != length or _HEX.fullmatch(value) is None:
        raise ValueError(f"invalid-{name}")
    return value


def validate_target_request(
    revision: object, install_id: object, current_sha: object
) -> TargetRequest | None:
    """Validate a complete target identity without normalizing its fields."""
    supplied = (revision is not None, install_id is not None, current_sha is not None)
    if not any(supplied):
        return None
    if not all(supplied):
        raise ValueError("incomplete-target-intent")
    return TargetRequest(
        _validate_field("revision", revision, 40),
        _validate_field("install_id", install_id, 32),
        _validate_field("current_sha", current_sha, 40),
    )
