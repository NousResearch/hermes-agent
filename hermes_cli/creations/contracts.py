from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CreationsConfig:
    enabled: bool = False
    anvil_url: str = "http://127.0.0.1:3000"

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "CreationsConfig":
        value = value or {}
        return cls(bool(value.get("enabled", False)), str(value.get("anvil_url", cls.anvil_url)))


def _error(code: str, message: str) -> ValueError:
    return ValueError(f"{code}: {message}")


def validate_request_id(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 128 or not value.strip():
        raise _error("invalid_request", "requestId is required")
    return value.strip()


def validate_plan_body(body: Any) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise _error("invalid_request", "request body must be an object")
    space_id = body.get("spaceId")
    outputs = body.get("outputNodeIds")
    if not isinstance(space_id, str) or not space_id.strip():
        raise _error("invalid_request", "spaceId is required")
    if not isinstance(outputs, list) or not outputs or not all(isinstance(x, str) and x for x in outputs):
        raise _error("invalid_request", "outputNodeIds must be a non-empty list")
    for key in ("overrides", "providers"):
        if key in body and not isinstance(body[key], dict):
            raise _error("invalid_request", f"{key} must be an object")
    return body
