from __future__ import annotations

from dataclasses import dataclass
import os


def _env(name: str, *fallbacks: str) -> str | None:
    for key in (name, *fallbacks):
        value = os.getenv(key)
        if value is not None and value.strip():
            return value.strip()
    return None


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


@dataclass(slots=True)
class CoryWorkerConfig:
    control_plane_base_url: str
    internal_api_token: str
    model: str | None = None
    provider: str | None = None
    poll_interval_seconds: float = 5.0
    max_backoff_seconds: float = 60.0
    max_completion_attempts: int = 2
    request_timeout_seconds: float = 30.0
    once: bool = False

    @classmethod
    def from_env(cls) -> "CoryWorkerConfig":
        base_url = _env("CORY_CONTROL_PLANE_BASE_URL", "CONTROL_PLANE_BASE_URL")
        if not base_url:
            raise ValueError(
                "CORY_CONTROL_PLANE_BASE_URL or CONTROL_PLANE_BASE_URL is required",
            )

        token = _env(
            "CORY_CONTROL_PLANE_INTERNAL_API_TOKEN",
            "CONTROL_PLANE_INTERNAL_API_TOKEN",
        )
        if not token:
            raise ValueError(
                "CORY_CONTROL_PLANE_INTERNAL_API_TOKEN or CONTROL_PLANE_INTERNAL_API_TOKEN is required",
            )

        return cls(
            control_plane_base_url=base_url.rstrip("/"),
            internal_api_token=token,
            model=_env("HERMES_CORY_MODEL"),
            provider=_env("HERMES_CORY_PROVIDER"),
            poll_interval_seconds=_env_float("HERMES_CORY_POLL_INTERVAL_SECONDS", 5.0),
            max_backoff_seconds=_env_float("HERMES_CORY_MAX_BACKOFF_SECONDS", 60.0),
            max_completion_attempts=_env_int("HERMES_CORY_MAX_COMPLETION_ATTEMPTS", 2),
            request_timeout_seconds=_env_float("HERMES_CORY_REQUEST_TIMEOUT_SECONDS", 30.0),
        )
