"""Hermes Gateway public API.

Public compatibility exports are resolved lazily.  Importing a narrow gateway
submodule (notably the privileged writer-only bootstrap) must not eagerly load
the general gateway configuration, session stack, delivery router, model
providers, or plugin discovery surface merely because Python imported this
package first.
"""

from __future__ import annotations

from typing import Any


_LAZY_EXPORTS = {
    "GatewayConfig": ("gateway.config", "GatewayConfig"),
    "PlatformConfig": ("gateway.config", "PlatformConfig"),
    "HomeChannel": ("gateway.config", "HomeChannel"),
    "load_gateway_config": ("gateway.config", "load_gateway_config"),
    "SessionContext": ("gateway.session", "SessionContext"),
    "SessionStore": ("gateway.session", "SessionStore"),
    "SessionResetPolicy": ("gateway.session", "SessionResetPolicy"),
    "build_session_context_prompt": (
        "gateway.session",
        "build_session_context_prompt",
    ),
    "DeliveryRouter": ("gateway.delivery", "DeliveryRouter"),
    "DeliveryTarget": ("gateway.delivery", "DeliveryTarget"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = [
    "GatewayConfig",
    "PlatformConfig",
    "HomeChannel",
    "load_gateway_config",
    "SessionContext",
    "SessionStore",
    "SessionResetPolicy",
    "build_session_context_prompt",
    "DeliveryRouter",
    "DeliveryTarget",
]
