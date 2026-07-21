"""Abstract base for provider transports.

A transport owns the data path for one api_mode:
  convert_messages → convert_tools → build_kwargs → normalize_response

It does NOT own: client construction, streaming, credential refresh,
prompt caching, interrupt handling, or retry logic.  Those stay on AIAgent.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agent.transports.types import NormalizedResponse


_HERMES_SERVER_TOOL_KEY = "_hermes_server_tool"


def project_tools_for_transport(
    tools: Optional[List[Dict[str, Any]]], api_mode: str
) -> Optional[List[Dict[str, Any]]]:
    """Project logical Hermes tools onto one provider transport.

    A function definition carrying ``_hermes_server_tool`` is server-only: it
    may be advertised solely to the api_mode named by that binding.  The
    target transport consumes the binding and emits its provider-native tool
    definition; every other transport omits the tool entirely.  This keeps
    Hermes-internal metadata off the wire and prevents fallbacks from exposing
    a client function whose handler cannot execute locally.

    Ordinary tools retain their original objects so the common path does not
    copy the complete tool list on every request.
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
            changed = True
        # A malformed or foreign binding is deliberately omitted. Forwarding
        # it either leaks internal metadata or exposes an unexecutable tool.

    return projected if changed else tools


class ProviderTransport(ABC):
    """Base class for provider-specific format conversion and normalization."""

    @property
    @abstractmethod
    def api_mode(self) -> str:
        """The api_mode string this transport handles (e.g. 'anthropic_messages')."""
        ...

    @abstractmethod
    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Convert OpenAI-format messages to provider-native format.

        Returns provider-specific structure (e.g. (system, messages) for Anthropic,
        or the messages list unchanged for chat_completions).
        """
        ...

    @abstractmethod
    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        """Convert OpenAI-format tool definitions to provider-native format.

        Returns provider-specific tool list (e.g. Anthropic input_schema format).
        """
        ...

    @abstractmethod
    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Build the complete API call kwargs dict.

        This is the primary entry point — it typically calls convert_messages()
        and convert_tools() internally, then adds model-specific config.

        Returns a dict ready to be passed to the provider's SDK client.
        """
        ...

    @abstractmethod
    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize a raw provider response to the shared NormalizedResponse type.

        This is the only method that returns a transport-layer type.
        """
        ...

    def validate_response(self, response: Any) -> bool:
        """Optional: check if the raw response is structurally valid.

        Returns True if valid, False if the response should be treated as invalid.
        Default implementation always returns True.
        """
        return True

    def extract_cache_stats(self, response: Any) -> Optional[Dict[str, int]]:
        """Optional: extract provider-specific cache hit/creation stats.

        Returns dict with 'cached_tokens' and 'creation_tokens', or None.
        Default returns None.
        """
        return None

    def map_finish_reason(self, raw_reason: str) -> str:
        """Optional: map provider-specific stop reason to OpenAI equivalent.

        Default returns the raw reason unchanged.  Override for providers
        with different stop reason vocabularies.
        """
        return raw_reason

    def project_tools(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Return only tool definitions executable through this transport."""
        return project_tools_for_transport(tools, self.api_mode)
