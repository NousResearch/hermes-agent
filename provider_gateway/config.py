"""Configuration for the opt-in provider gateway."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_VALID_BACKENDS = {"native", "litellm"}
_VALID_ROUTING_STRATEGIES = {"round-robin", "lowest-cost", "lowest-latency"}


@dataclass(frozen=True)
class GatewayConfig:
    """Runtime configuration for provider gateway features."""

    enabled: bool = False
    backend: str = "native"
    track_usage: bool = True
    track_cost: bool = True
    routing_strategy: str = "round-robin"
    fallback_models: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "GatewayConfig":
        """Build a gateway config from a raw config mapping."""
        if not isinstance(data, Mapping):
            data = {}

        routing = data.get("routing", {})
        if not isinstance(routing, Mapping):
            routing = {}

        backend = _clean_choice(
            data.get("backend"),
            valid_values=_VALID_BACKENDS,
            default="native",
            field_name="provider_gateway.backend",
        )
        routing_strategy = _clean_choice(
            routing.get("strategy"),
            valid_values=_VALID_ROUTING_STRATEGIES,
            default="round-robin",
            field_name="provider_gateway.routing.strategy",
        )

        raw_fallbacks = routing.get("fallback_models", [])
        fallback_models = [
            item.strip()
            for item in raw_fallbacks
            if isinstance(item, str) and item.strip()
        ] if isinstance(raw_fallbacks, list) else []

        return cls(
            enabled=bool(data.get("enabled", False)),
            backend=backend,
            track_usage=bool(data.get("track_usage", True)),
            track_cost=bool(data.get("track_cost", True)),
            routing_strategy=routing_strategy,
            fallback_models=fallback_models,
        )


def load_gateway_config(root_config: Mapping[str, Any] | None = None) -> GatewayConfig:
    """Load provider gateway config from a full Hermes config mapping."""
    if root_config is None:
        try:
            from hermes_cli.config import load_config

            root_config = load_config()
        except Exception as exc:
            logger.debug("Could not load Hermes config for provider gateway: %s", exc)
            root_config = {}

    if not isinstance(root_config, Mapping):
        root_config = {}
    return GatewayConfig.from_dict(root_config.get("provider_gateway", {}))


def _clean_choice(
    value: Any,
    *,
    valid_values: set[str],
    default: str,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        return default
    cleaned = value.strip().lower()
    if cleaned not in valid_values:
        logger.debug("Invalid %s=%r, using %r", field_name, value, default)
        return default
    return cleaned
