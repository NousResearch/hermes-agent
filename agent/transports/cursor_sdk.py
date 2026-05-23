"""Transport for the Cursor SDK provider."""

from __future__ import annotations

from typing import Any

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, Usage


class CursorSdkTransport(ProviderTransport):
    """Cursor SDK transport.

    The actual provider call is made in ``agent.cursor_sdk_adapter`` through a
    Node bridge to Cursor's official ``@cursor/sdk`` package. This transport
    owns the request/response shape exposed to the Hermes loop.
    """

    @property
    def api_mode(self) -> str:
        return "cursor_sdk"

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs) -> list[dict[str, Any]]:
        return messages

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Cursor SDK has its own agent tool harness. Hermes function schemas are
        # not forwarded as OpenAI tool calls on this transport.
        return []

    def build_kwargs(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params: Any,
    ) -> dict[str, Any]:
        return {
            "__cursor_sdk__": True,
            "model": model,
            "messages": self.convert_messages(messages),
            "api_key": params.get("api_key") or "",
            "cwd": params.get("cwd") or "",
            "runtime": params.get("runtime") or "",
            "timeout": params.get("timeout"),
        }

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        choice = response.choices[0]
        message = choice.message
        usage = None
        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            usage = Usage(
                prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(raw_usage, "total_tokens", 0) or 0,
            )
        return NormalizedResponse(
            content=getattr(message, "content", "") or "",
            tool_calls=None,
            finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
            usage=usage,
        )

    def validate_response(self, response: Any) -> bool:
        return bool(
            response is not None
            and getattr(response, "choices", None)
            and getattr(response.choices[0], "message", None) is not None
        )


from agent.transports import register_transport  # noqa: E402

register_transport("cursor_sdk", CursorSdkTransport)
