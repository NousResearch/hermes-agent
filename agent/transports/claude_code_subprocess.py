"""Claude Code CLI subprocess transport.

Routes inference through `claude -p` subprocess so Claude Max/Pro
subscribers use their subscription quota directly, bypassing OAuth
token rate limits on the external API endpoint.

This transport owns normalization of the SimpleNamespace response
returned by _call_claude_code_subprocess() into NormalizedResponse.
"""

from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, Usage


class ClaudeCodeSubprocessTransport(ProviderTransport):
    """Transport for api_mode='claude_code_subprocess'.

    The actual inference is done by _call_claude_code_subprocess() in
    chat_completion_helpers.py — this transport only handles format
    conversion and normalization so the conversation loop works
    unchanged.
    """

    @property
    def api_mode(self) -> str:
        return "claude_code_subprocess"

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
        system = params.get("system", "")
        return {"model": model, "messages": messages, "system": system}

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        if response is None:
            return NormalizedResponse(
                content="",
                finish_reason="stop",
                tool_calls=[],
                usage=Usage(input_tokens=0, output_tokens=0),
            )
        content_blocks = getattr(response, "content", []) or []
        text = " ".join(
            getattr(b, "text", "") for b in content_blocks
            if getattr(b, "type", "") == "text"
        ).strip()
        usage_obj = getattr(response, "usage", None)
        usage = Usage(
            prompt_tokens=getattr(usage_obj, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
        )
        stop_reason = getattr(response, "stop_reason", "end_turn") or "end_turn"
        finish_reason = "stop" if stop_reason == "end_turn" else stop_reason
        return NormalizedResponse(
            content=text,
            finish_reason=finish_reason,
            tool_calls=[],
            usage=usage,
        )

    def validate_response(self, response: Any) -> bool:
        if response is None:
            return False
        content_blocks = getattr(response, "content", None)
        if not isinstance(content_blocks, list):
            return False
        if not content_blocks:
            return getattr(response, "stop_reason", None) == "end_turn"
        return True

    def map_finish_reason(self, raw_reason: str) -> str:
        _MAP = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
        }
        return _MAP.get(raw_reason, "stop")


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("claude_code_subprocess", ClaudeCodeSubprocessTransport)
