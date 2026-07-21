"""Abstract base for provider transports.
A transport owns one api_mode's data path (convert_messages -> convert_tools -> build_kwargs
-> normalize_response), NOT client construction, streaming, credentials, caching, interrupts
or retries — those stay on AIAgent."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agent.transports.types import NormalizedResponse

_HERMES_SERVER_TOOL_KEY = "_hermes_server_tool"


def project_tools_for_transport(
    tools: Optional[List[Dict[str, Any]]], api_mode: str,
) -> Optional[List[Dict[str, Any]]]:
    """Project logical Hermes tools onto one provider transport.

    A function definition carrying ``_hermes_server_tool`` is server-only: it may be advertised solely to the
    api_mode named by that binding. The target transport consumes the binding and emits its provider-native
    tool definition; every other transport omits the tool entirely. This keeps Hermes-internal metadata off
    the wire and prevents fallbacks from exposing a client function whose handler cannot execute locally.
    Ordinary tools retain their original objects so the common path does not copy the tool list per request.
    """
    if tools is None:
        return None
    projected: List[Dict[str, Any]] = []
    changed = False
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or _HERMES_SERVER_TOOL_KEY not in function:
            projected.append(tool)
            continue
        binding = function.get(_HERMES_SERVER_TOOL_KEY)
        if isinstance(binding, dict) and binding.get("api_mode") == api_mode:
            projected.append(tool)
        else:
            # A malformed or foreign binding is deliberately omitted: forwarding it either leaks internal
            # metadata or exposes an unexecutable tool.
            changed = True
    return projected if changed else tools


class ProviderTransport(ABC):
    """Base class for provider-specific format conversion and normalization."""

    # Provider stop_reason -> OpenAI finish_reason. ``None`` means the provider
    # already speaks OpenAI vocabulary and map_finish_reason passes through.
    _STOP_REASON_MAP: Optional[Dict[str, str]] = None

    @property
    @abstractmethod
    def api_mode(self) -> str:
        """The api_mode string this transport handles (e.g. 'anthropic_messages')."""

    @abstractmethod
    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Convert OpenAI-format messages to the provider-native structure (e.g. (system, messages) for Anthropic)."""

    @abstractmethod
    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        """Convert OpenAI-format tool definitions to provider-native format."""

    @abstractmethod
    def build_kwargs(
        self, model: str, messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None, **params,
    ) -> Dict[str, Any]:
        """Primary entry point: convert messages/tools and return kwargs ready for the provider SDK."""

    @abstractmethod
    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize a raw provider response to NormalizedResponse (the only transport-layer return type)."""

    def validate_response(self, response: Any) -> bool:
        """Optional structural validity check; default accepts everything."""
        return True

    def extract_cache_stats(self, response: Any) -> Optional[Dict[str, int]]:
        """Optional: ``{'cached_tokens', 'creation_tokens'}`` or None (default)."""
        return None

    def map_finish_reason(self, raw_reason: str) -> str:
        """Map a provider stop reason via ``_STOP_REASON_MAP`` (unknown -> 'stop'); passthrough when no map."""
        return raw_reason if self._STOP_REASON_MAP is None else self._STOP_REASON_MAP.get(raw_reason, "stop")

    def project_tools(self, tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
        """Return only tool definitions executable through this transport."""
        return project_tools_for_transport(tools, self.api_mode)
