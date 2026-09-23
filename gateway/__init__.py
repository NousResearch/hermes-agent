"""Hermes Gateway - multi-platform messaging integration (sessions, context
injection, delivery routing, platform-specific toolsets)."""

# Source-swap canary (#96464): importing this module at package load captures
# the birth receipt (HEAD commit + on-disk identity of already-loaded repo
# modules) before run.py's body — and every lazy import it will ever do —
# runs. Purely passive: no behavior, no threads, just the earliest possible
# observation of the source generation this process starts against.
from . import import_sanity as _import_sanity  # noqa: F401

from .config import GatewayConfig, PlatformConfig, HomeChannel, load_gateway_config
from .session import (
    SessionContext,
    SessionStore,
    build_session_context_prompt,
)
from .delivery import DeliveryRouter, DeliveryTarget

__all__ = [
    "GatewayConfig", "PlatformConfig", "HomeChannel", "load_gateway_config",
    "SessionContext", "SessionStore", "build_session_context_prompt",
    "DeliveryRouter", "DeliveryTarget",
]
