"""Transport layer types and registry for provider response normalization.

Usage:
    from agent.transports import get_transport
    transport = get_transport("anthropic_messages")
    result = transport.normalize_response(raw_response)
"""

from agent.transports.types import NormalizedResponse, ToolCall, Usage, build_tool_call, map_finish_reason  # noqa: F401

_REGISTRY: dict = {}


def register_transport(api_mode: str, transport_cls: type) -> None:
    """Register a transport class for an api_mode string."""
    _REGISTRY[api_mode] = transport_cls


def get_transport(api_mode: str):
    """Get a transport instance for the given api_mode.

    Returns None if no transport is registered for this api_mode.
    This allows gradual migration — call sites can check for None
    and fall back to the legacy code path.
    """
    if not _REGISTRY:
        _discover_transports()
    cls = _REGISTRY.get(api_mode)
    if cls is None:
        return None
    return cls()


def _discover_transports() -> None:
    """Import all transport modules to trigger auto-registration."""
    try:
        import agent.transports.anthropic  # noqa: F401
    except ImportError:
        pass
