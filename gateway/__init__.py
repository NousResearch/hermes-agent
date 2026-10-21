"""Hermes Gateway - multi-platform messaging integration (sessions, context
injection, delivery routing, platform-specific toolsets)."""

from importlib import import_module

# Control/status clients must not initialize config, plugins or user data.
# The existing package exports remain available when actually requested.
_EXPORTS = {
    "GatewayConfig": ".config",
    "PlatformConfig": ".config",
    "HomeChannel": ".config",
    "load_gateway_config": ".config",
    "SessionContext": ".session",
    "SessionStore": ".session",
    "build_session_context_prompt": ".session",
    "DeliveryRouter": ".delivery",
    "DeliveryTarget": ".delivery",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module, __name__), name)
