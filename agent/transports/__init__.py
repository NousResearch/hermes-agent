"""Transport layer types and registry for provider response normalization.

Usage:
    from agent.transports import get_transport
    transport = get_transport("anthropic_messages")
    result = transport.normalize_response(raw_response)
"""

import threading

from agent.transports.types import (
    NormalizedResponse,
    ToolCall,
    Usage,
    build_tool_call,
    map_finish_reason,
)  # noqa: F401

_REGISTRY: dict = {}
_discovered: bool = False
_discovery_lock = threading.Lock()


def register_transport(api_mode: str, transport_cls: type) -> None:
    """Register a transport class for an api_mode string."""
    _REGISTRY[api_mode] = transport_cls


def get_transport(api_mode: str):
    """Get a transport instance for the given api_mode.

    Returns None if no transport is registered for this api_mode.
    This allows gradual migration — call sites can check for None
    and fall back to the legacy code path.
    """
    global _discovered
    cls = _REGISTRY.get(api_mode)
    if cls is None:
        with _discovery_lock:
            if not _discovered:
                _discover_transports()
                _discovered = True
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
    try:
        import agent.transports.codex  # noqa: F401
    except ImportError:
        pass
    try:
        import agent.transports.chat_completions  # noqa: F401
    except ImportError:
        pass
    try:
        import agent.transports.bedrock  # noqa: F401
    except ImportError:
        pass
    try:
        import agent.transports.tinfoil  # noqa: F401
    except ImportError:
        pass
