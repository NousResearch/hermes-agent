"""Feishu multitenancy bundled plugin entry point."""

try:
    from .hermes_multitenancy import on_pre_gateway_dispatch, register
except ImportError:  # Allows direct pytest/importlib loading outside the plugin namespace.
    from hermes_multitenancy import on_pre_gateway_dispatch, register

__all__ = ["register", "on_pre_gateway_dispatch"]
