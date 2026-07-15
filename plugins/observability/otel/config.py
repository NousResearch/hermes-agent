"""Environment configuration for the OpenTelemetry observer plugin."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}
_VALID_PROTOCOLS = {"http", "grpc"}


@dataclass(frozen=True)
class OTelConfig:
    enabled: bool = False
    endpoint: str = ""
    protocol: str = "http"
    service_name: str = "hermes-agent"
    headers: dict[str, str] | None = None


def _parse_enabled(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return False


def _parse_headers(value: str) -> dict[str, str]:
    if not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {
        str(key): str(header_value)
        for key, header_value in parsed.items()
        if isinstance(key, str) and isinstance(header_value, (str, int, float, bool))
    }


def load_config(environ: Mapping[str, str] | None = None) -> OTelConfig:
    """Load fail-closed activation settings from environment variables."""
    env = environ if environ is not None else os.environ
    protocol = env.get("HERMES_OTEL_PROTOCOL", "http").strip().lower()
    if protocol not in _VALID_PROTOCOLS:
        protocol = "http"

    service_name = env.get("HERMES_OTEL_SERVICE_NAME", "hermes-agent").strip()
    return OTelConfig(
        enabled=_parse_enabled(env.get("HERMES_OTEL_ENABLED", "0")),
        endpoint=env.get("HERMES_OTEL_ENDPOINT", "").strip(),
        protocol=protocol,
        service_name=service_name or "hermes-agent",
        headers=_parse_headers(env.get("HERMES_OTEL_HEADERS", "")),
    )
