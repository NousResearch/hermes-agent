"""Cursor Agent API transport."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.chat_completions import ChatCompletionsTransport
from agent.transports.types import NormalizedResponse


class CursorAgentTransport(ProviderTransport):
    """Transport for api_mode='cursor_agent'.

    The streaming client already returns an OpenAI ChatCompletion-shaped
    SimpleNamespace; normalization reuses the chat_completions path.
    """

    _chat = ChatCompletionsTransport()

    @property
    def api_mode(self) -> str:
        return "cursor_agent"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        return messages

    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        return tools

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"model": model, "messages": messages}
        if tools is not None:
            kwargs["tools"] = tools
        return kwargs

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        if response is None or not getattr(response, "choices", None):
            return NormalizedResponse(
                content=None,
                tool_calls=None,
                finish_reason="stop",
            )
        return self._chat.normalize_response(response, **kwargs)

    def validate_response(self, response: Any) -> bool:
        return response is not None and hasattr(response, "choices") and bool(response.choices)


from agent.transports import register_transport  # noqa: E402

register_transport("cursor_agent", CursorAgentTransport)
