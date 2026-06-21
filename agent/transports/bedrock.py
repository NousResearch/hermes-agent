"""AWS Bedrock Converse API transport.

Delegates to the existing adapter functions in agent/bedrock_adapter.py.
Bedrock uses its own boto3 client (not the OpenAI SDK), so the transport
owns format conversion and normalization, while client construction and
boto3 calls stay on AIAgent.
"""

from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall, Usage

# ── Max output token limits for non-Anthropic Bedrock models ──────────────
# Source: AWS Bedrock model card docs (https://docs.aws.amazon.com/bedrock/
# latest/userguide/model-cards.html).  Substring matching — longest match
# wins.  Anthropic/Claude models are NOT listed here; they are resolved via
# _get_anthropic_max_output() from anthropic_adapter, which already has a
# complete and maintained Claude table.
_BEDROCK_NATIVE_OUTPUT_LIMITS: Dict[str, int] = {
    # Amazon Nova
    "amazon.nova-2-lite":      65_536,
    "amazon.nova-premier":     25_600,
    "amazon.nova-pro":          5_120,
    "amazon.nova-lite":         5_120,
    "amazon.nova-micro":        5_120,
    # Meta Llama 4
    "meta.llama4-maverick":     8_192,
    "meta.llama4-scout":        8_192,
    # Meta Llama 3.x
    "meta.llama3-3-70b":        4_096,
    "meta.llama3-2-90b":        4_096,
    "meta.llama3-2-11b":        4_096,
    "meta.llama3-2-3b":         4_096,
    "meta.llama3-2-1b":         4_096,
    "meta.llama3-1-405b":       4_096,
    "meta.llama3-1-70b":        4_096,
    "meta.llama3-1-8b":         4_096,
    "meta.llama3-70b":          8_192,
    "meta.llama3-8b":           8_192,
    # Mistral AI
    "mistral.magistral-small":  40_960,
    "mistral.mistral-large-3":  32_768,
    "mistral.devstral-2":       32_768,
    "mistral.pixtral-large":    16_384,
    "mistral.ministral-3":       8_192,
    "mistral.mistral-large-2402": 4_096,
    "mistral.mistral-small-2402": 4_096,
    "mistral.mixtral-8x7b":     4_096,
    "mistral.mistral-7b":       4_096,
    # Cohere
    "cohere.command-r-plus":    4_096,
    "cohere.command-r":         4_096,
    # DeepSeek
    "deepseek.r1":              8_192,
    "deepseek.deepseek-v3":     8_192,
    # AI21 Labs
    "ai21.jamba-1-5-large":     4_096,
    "ai21.jamba-1-5-mini":      4_096,
}

# Safe fallback for unknown future Bedrock-native models.  Bedrock will
# clamp to the model's actual ceiling if this exceeds it.
_BEDROCK_DEFAULT_OUTPUT_LIMIT = 8_192


def _get_bedrock_max_output(model: str) -> int:
    """Resolve the max output token limit for a Bedrock model ID.

    Resolution order:
    1. _BEDROCK_NATIVE_OUTPUT_LIMITS table — for non-Anthropic Bedrock models.
       Uses longest-substring match so date-stamped / versioned IDs resolve.
    2. _get_anthropic_max_output() — for Claude models (which carry provider
       prefixes like 'global.', 'us.', 'anthropic.' in Bedrock model IDs).
    3. _BEDROCK_DEFAULT_OUTPUT_LIMIT (8 192) — safe fallback; Bedrock clamps.
    """
    m = model.lower()
    # Check native table first (longest match wins)
    best_key = ""
    best_val: Optional[int] = None
    for key, val in _BEDROCK_NATIVE_OUTPUT_LIMITS.items():
        if key in m and len(key) > len(best_key):
            best_key = key
            best_val = val
    if best_val is not None:
        return best_val

    # Claude models delivered via Bedrock carry prefixes like
    # 'global.anthropic.claude-sonnet-4-6' — delegate to the Anthropic table.
    if "claude" in m or "anthropic" in m:
        from agent.anthropic_adapter import _get_anthropic_max_output
        return _get_anthropic_max_output(model)

    return _BEDROCK_DEFAULT_OUTPUT_LIMIT


class BedrockTransport(ProviderTransport):
    """Transport for api_mode='bedrock_converse'."""

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
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Build Bedrock converse() kwargs.

        Calls convert_messages and convert_tools internally.

        params:
            max_tokens: int — output token limit (default: model's native
                output ceiling via _get_bedrock_max_output — Claude models use
                the shared Anthropic table; all other Bedrock models use the
                _BEDROCK_NATIVE_OUTPUT_LIMITS table in this module)
            temperature: float | None
            guardrail_config: dict | None — Bedrock guardrails
            region: str — AWS region (default 'us-east-1')
        """
        from agent.bedrock_adapter import build_converse_kwargs

        region = params.get("region", "us-east-1")
        guardrail = params.get("guardrail_config")

        requested_max_tokens = params.get("max_tokens")
        if requested_max_tokens is None:
            # Resolve the model's declared output ceiling.  Claude models
            # (e.g. 'global.anthropic.claude-sonnet-4-6') delegate to the
            # shared Anthropic table; all other Bedrock-native models
            # (Amazon Nova, Meta Llama, Mistral, etc.) use the table in this
            # module.  Bedrock silently clamps overshoots to the actual limit.
            max_tokens = _get_bedrock_max_output(model)
        else:
            max_tokens = requested_max_tokens

        kwargs = build_converse_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=params.get("temperature"),
            guardrail_config=guardrail,
        )
        # Sentinel keys for dispatch — agent pops these before the boto3 call
        kwargs["__bedrock_converse__"] = True
        kwargs["__bedrock_region__"] = region
        return kwargs

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize Bedrock response to NormalizedResponse.

        Handles two shapes:
        1. Raw boto3 dict (from direct converse() calls)
        2. Already-normalized SimpleNamespace with .choices (from dispatch site)
        """
        from agent.bedrock_adapter import normalize_converse_response

        # Normalize to OpenAI-compatible SimpleNamespace
        if hasattr(response, "choices") and response.choices:
            # Already normalized at dispatch site
            ns = response
        else:
            # Raw boto3 dict
            ns = normalize_converse_response(response)

        choice = ns.choices[0]
        msg = choice.message
        finish_reason = choice.finish_reason or "stop"

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in msg.tool_calls
            ]

        usage = None
        if hasattr(ns, "usage") and ns.usage:
            u = ns.usage
            usage = Usage(
                prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(u, "completion_tokens", 0) or 0,
                total_tokens=getattr(u, "total_tokens", 0) or 0,
            )

        reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)

        return NormalizedResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            reasoning=reasoning,
            usage=usage,
        )

    def validate_response(self, response: Any) -> bool:
        """Check Bedrock response structure.

        After normalize_converse_response, the response has OpenAI-compatible
        .choices — same check as chat_completions.
        """
        if response is None:
            return False
        # Raw Bedrock dict response — check for 'output' key
        if isinstance(response, dict):
            return "output" in response
        # Already-normalized SimpleNamespace
        if hasattr(response, "choices"):
            return bool(response.choices)
        return False

    def map_finish_reason(self, raw_reason: str) -> str:
        """Map Bedrock stop reason to OpenAI finish_reason.

        The adapter already does this mapping inside normalize_converse_response,
        so this is only used for direct access to raw responses.
        """
        _MAP = {
            "end_turn": "stop",
            "tool_use": "tool_calls",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "guardrail_intervened": "content_filter",
            "content_filtered": "content_filter",
        }
        return _MAP.get(raw_reason, "stop")


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("bedrock_converse", BedrockTransport)
