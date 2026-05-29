"""Provider gateway foundation.

This package holds opt-in multi-provider routing helpers. Importing it must not
change Hermes runtime behavior; integration points should stay config-gated.
"""

from provider_gateway.config import GatewayConfig, load_gateway_config
from provider_gateway.policy import (
    ProviderGatewayPolicy,
    ProviderRouteCandidate,
    build_gateway_policy,
    should_consider_gateway_fallback,
)
from provider_gateway.status import build_gateway_status, format_gateway_status_lines
from provider_gateway.usage_tracker import (
    SCHEMA_VERSION,
    ProviderUsageRecord,
    ProviderUsageTracker,
)

__all__ = [
    "GatewayConfig",
    "ProviderGatewayPolicy",
    "ProviderRouteCandidate",
    "ProviderUsageRecord",
    "ProviderUsageTracker",
    "SCHEMA_VERSION",
    "build_gateway_policy",
    "build_gateway_status",
    "format_gateway_status_lines",
    "load_gateway_config",
    "should_consider_gateway_fallback",
]
