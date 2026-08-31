"""Abstract base for provider transports.

A transport owns the data path for one api_mode:
  convert_messages → convert_tools → build_kwargs → normalize_response

It does NOT own: client construction, streaming, credential refresh,
prompt caching, interrupt handling, or retry logic.  Those stay on AIAgent.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from agent.transports.types import NormalizedResponse
from utils import TOOL_REQUEST_KEYS, explicit_no_tools

logger = logging.getLogger(__name__)


class ProviderTransport(ABC):
    """Base class for provider-specific format conversion and normalization."""

    @staticmethod
    def sanitize_request_overrides(
        overrides: Optional[Dict[str, Any]],
        *,
        context: str = "request_overrides",
    ) -> Optional[Dict[str, Any]]:
        """Drop tool-offering keys from a user-config override dict.

        ``request_overrides`` comes from the user's ``config.yaml`` and is
        merged into the outgoing kwargs *after* the transport has decided which
        tools to expose — so ``request_overrides: {tools: [...]}`` would
        otherwise reinstate a tool surface that ``--no-tools`` removed. Under
        the explicit no-tools boundary those keys are stripped here, before any
        merge, on every transport that supports overrides.

        Strips rather than raises: a gateway serving live traffic must degrade
        to "no tools" — which is what the operator asked for — instead of
        aborting the turn. This mirrors how the rest of the tool pipeline
        handles bad input (``model_tools`` logs and continues when schema
        sanitization or tool-search assembly fails). Every dropped key is
        logged at warning level by name so the misconfiguration is visible.

        Returns the dict unchanged when the boundary is not in force, so this
        is a no-op on every normal request.
        """
        if not overrides or not explicit_no_tools():
            return overrides
        offending = [k for k in overrides if k in TOOL_REQUEST_KEYS]
        if not offending:
            return overrides
        for key in offending:
            logger.warning(
                "HERMES_NO_TOOLS=1: dropping %r from %s — an explicit no-tools "
                "run must not offer a tool surface. Remove it from config.yaml's "
                "request_overrides, or drop --no-tools.",
                key,
                context,
            )
        return {k: v for k, v in overrides.items() if k not in TOOL_REQUEST_KEYS}

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
