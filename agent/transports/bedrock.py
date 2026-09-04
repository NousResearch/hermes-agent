"""AWS Bedrock Converse API transport.

Delegates format conversion to agent/bedrock_adapter.py. Bedrock uses its own
boto3 client, so client construction and calls stay on AIAgent.
"""

from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall, Usage


class BedrockTransport(ProviderTransport):
    """Transport for api_mode='bedrock_converse'."""

    # The adapter already maps inside normalize_converse_response; this serves raw-response access.
    _STOP_REASON_MAP = {
        "end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "stop_sequence": "stop",
        "guardrail_intervened": "content_filter", "content_filtered": "content_filter",
    }

    @property
    def api_mode(self) -> str:
        return "bedrock_converse"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Convert OpenAI messages to Bedrock Converse format."""
        from agent.bedrock_adapter import convert_messages_to_converse
        return convert_messages_to_converse(messages)

    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        """Convert OpenAI tool schemas to Bedrock Converse toolConfig."""
        from agent.bedrock_adapter import convert_tools_to_converse
        return convert_tools_to_converse(tools)

    def build_kwargs(
        self, model: str, messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None, **params,
    ) -> Dict[str, Any]:
        """Build converse() kwargs; params: max_tokens (4096), temperature, guardrail_config, region ('us-east-1')."""
        from agent.bedrock_adapter import build_converse_kwargs

        kwargs = build_converse_kwargs(
            model=model, messages=messages, tools=tools, max_tokens=params.get("max_tokens", 4096),
            temperature=params.get("temperature"), guardrail_config=params.get("guardrail_config"),
        )
        # Sentinel keys for dispatch — agent pops these before the boto3 call
        kwargs["__bedrock_converse__"] = True
        kwargs["__bedrock_region__"] = params.get("region", "us-east-1")
        return kwargs

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize either a raw boto3 dict or an already-normalized SimpleNamespace with .choices."""
        from agent.bedrock_adapter import normalize_converse_response

        ns = response if hasattr(response, "choices") and response.choices else normalize_converse_response(response)
        choice = ns.choices[0]
        msg = choice.message

        tool_calls = (
            [ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments) for tc in msg.tool_calls]
            if msg.tool_calls else None
        )
        provider_data = {
            key: getattr(msg, key) for key in ("reasoning_details", "bedrock_content_blocks") if getattr(msg, key, None)
        }
        return NormalizedResponse(
            content=msg.content, tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
            reasoning=getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None),
            usage=Usage.from_openai(ns.usage) if getattr(ns, "usage", None) else None, provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        """Raw Bedrock dict needs an 'output' key; a normalized namespace needs non-empty .choices."""
        if isinstance(response, dict):
            return "output" in response
        return bool(getattr(response, "choices", None)) if response is not None else False


from agent.transports import register_transport  # noqa: E402

register_transport("bedrock_converse", BedrockTransport)
