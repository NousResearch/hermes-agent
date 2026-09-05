"""Transport registry for provider response normalization.
    transport = get_transport("anthropic_messages")
    result = transport.normalize_response(raw_response)"""

import contextlib
import importlib

from agent.transports.types import (  # noqa: F401
    NormalizedResponse,
    ToolCall,
    Usage,
    build_tool_call,
    map_finish_reason,
)

_REGISTRY: dict = {}
_discovered: bool = False
_TRANSPORT_MODULES = ("anthropic", "codex", "chat_completions", "bedrock")


def register_transport(api_mode: str, transport_cls: type) -> None:
    """Register a transport class for an api_mode string."""
    _REGISTRY[api_mode] = transport_cls


def get_transport(api_mode: str):
    """Return a transport instance for ``api_mode``, or None so callers can fall back to the legacy path."""
    # A directly-imported transport leaves the registry partial; (re)discover on first use and on misses.
    if not _discovered or api_mode not in _REGISTRY:
        _discover_transports()
    cls = _REGISTRY.get(api_mode)
    return None if cls is None else cls()


def _discover_transports() -> None:
    """Import all transport modules to trigger auto-registration."""
    global _discovered
    _discovered = True
    for name in _TRANSPORT_MODULES:
        with contextlib.suppress(ImportError):
            importlib.import_module(f"agent.transports.{name}")
