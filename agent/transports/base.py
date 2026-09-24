"""Abstract base for provider transports.
A transport owns one api_mode's data path (convert_messages -> convert_tools -> build_kwargs
-> normalize_response), NOT client construction, streaming, credentials, caching, interrupts
or retries — those stay on AIAgent."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agent.transports.types import NormalizedResponse


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
