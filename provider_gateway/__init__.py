"""Provider gateway foundation.

This package holds opt-in multi-provider routing helpers. Importing it must not
change Hermes runtime behavior; integration points should stay config-gated.
"""

from provider_gateway.circuit_breaker import CircuitBreaker, CircuitState
from provider_gateway.config import GatewayConfig, load_gateway_config
from provider_gateway.discovery import OllamaDiscovery
from provider_gateway.guardrails import PIISanitizer, StreamingDeanonimizer
from provider_gateway.policy import (
    ProviderGatewayPolicy,
    ProviderRouteCandidate,
    build_gateway_policy,
    should_consider_gateway_fallback,
)
from provider_gateway.quota_manager import QuotaExceededError, QuotaManager
from provider_gateway.router import ProviderRouter
from provider_gateway.secure_store import DynamicCredentialStore
from provider_gateway.semantic_cache import SemanticCache
from provider_gateway.status import build_gateway_status, format_gateway_status_lines
from provider_gateway.usage_tracker import (
    SCHEMA_VERSION,
    ProviderUsageRecord,
    ProviderUsageTracker,
)

__all__ = [
    # Foundation (Fase 1)
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
    # Active Routing (Fase 2)
    "CircuitBreaker",
    "CircuitState",
    "ProviderRouter",
    "QuotaExceededError",
    "QuotaManager",
    "SemanticCache",
    # Ecosystem (Fase 4)
    "DynamicCredentialStore",
    "OllamaDiscovery",
    "PIISanitizer",
    "StreamingDeanonimizer",
]
