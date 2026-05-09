"""Anthropic Messages API transport.

Delegates to the existing adapter functions in agent/anthropic_adapter.py.
This transport owns format conversion and normalization — NOT client lifecycle.
"""

from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse


class AnthropicTransport(ProviderTransport):
    """Transport for api_mode='anthropic_messages'.

    Wraps the existing functions in anthropic_adapter.py behind the
    ProviderTransport ABC.  Each method delegates — no logic is duplicated.
    """

    @property
    def api_mode(self) -> str:
        return "anthropic_messages"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Convert OpenAI messages to Anthropic (system, messages) tuple.

        kwargs:
            base_url: Optional[str] — affects thinking signature handling.
        """
        from agent.anthropic_adapter import convert_messages_to_anthropic

        base_url = kwargs.get("base_url")
        return convert_messages_to_anthropic(messages, base_url=base_url)

    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        """Convert OpenAI tool schemas to Anthropic input_schema format."""
        from agent.anthropic_adapter import convert_tools_to_anthropic

        return convert_tools_to_anthropic(tools)

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Build kwargs for ``client.beta.messages.{create,stream}``.

        Calls convert_messages and convert_tools internally. The output
        is shaped for the beta namespace specifically — typed kwargs for
        ``thinking``, ``output_config``, ``context_management``, ``betas``,
        ``speed``, ``metadata`` go through directly without the
        ``extra_body``/``extra_headers`` workarounds the plain
        ``messages.*`` namespace required.

        params (all optional):
            max_tokens: int
            reasoning_config: dict | None
            tool_choice: str | None
            is_oauth: bool
            preserve_dots: bool
            context_length: int | None
            base_url: str | None
            fast_mode: bool
            drop_context_1m_beta: bool
            tool_search_config: dict | None — see _apply_tool_search in
                anthropic_adapter.py for the schema. When None or
                disabled, no transformation is applied.
            session_id: str | None — included in metadata.user_id blob
                so Anthropic-side analytics can correlate per-session.
        """
        from agent.anthropic_adapter import build_anthropic_kwargs

        return build_anthropic_kwargs(
            model=model,
            messages=messages,
            tools=tools,
            max_tokens=params.get("max_tokens", 16384),
            reasoning_config=params.get("reasoning_config"),
            tool_choice=params.get("tool_choice"),
            is_oauth=params.get("is_oauth", False),
            preserve_dots=params.get("preserve_dots", False),
            context_length=params.get("context_length"),
            base_url=params.get("base_url"),
            fast_mode=params.get("fast_mode", False),
            drop_context_1m_beta=params.get("drop_context_1m_beta", False),
            tool_search_config=params.get("tool_search_config"),
            session_id=params.get("session_id"),
            cache_tools=params.get("cache_tools", False),
            cache_ttl=params.get("cache_ttl", "5m"),
        )

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize Anthropic response to NormalizedResponse.

        Parses content blocks (text, thinking, tool_use), maps stop_reason
        to OpenAI finish_reason, and collects reasoning_details in provider_data.
        """
        import json
        from agent.anthropic_adapter import _to_plain_data
        from agent.transports.types import ToolCall

        strip_tool_prefix = kwargs.get("strip_tool_prefix", False)
        _MCP_PREFIX = "mcp_"

        text_parts = []
        reasoning_parts = []
        reasoning_details = []
        tool_calls = []
        # Server-side tools (web_search_20250305, etc.) emit two distinct
        # block types in the same response: ``server_tool_use`` (Anthropic
        # logging the search Anthropic-side) and ``web_search_tool_result``
        # (the search results Anthropic fetched). We don't execute these
        # locally — Anthropic already did. Keep them in provider_data so
        # they survive into the next turn's history (Anthropic requires
        # the tool_result blocks to be present when re-submitting prior
        # assistant turns that reference them) and so the UI can show a
        # search citation panel.
        server_tool_blocks: list[dict] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "thinking":
                reasoning_parts.append(block.thinking)
                block_dict = _to_plain_data(block)
                if isinstance(block_dict, dict):
                    reasoning_details.append(block_dict)
            elif block.type == "tool_use":
                name = block.name
                if strip_tool_prefix and name.startswith(_MCP_PREFIX):
                    name = name[len(_MCP_PREFIX):]
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=name,
                        arguments=json.dumps(block.input),
                    )
                )
            elif (
                block.type
                in (
                    "server_tool_use",
                    "web_search_tool_result",
                )
                or block.type.startswith("tool_search_tool_")
            ):
                # tool_search_tool_<variant>_tool_result (e.g.
                # tool_search_tool_regex_tool_result) carries the discovered
                # tool_reference array. Anthropic auto-expands tool_reference
                # blocks across the conversation history so the model can
                # reuse discovered tools without re-searching — but only as
                # long as we round-trip the block back in messages on
                # subsequent turns. The block type is variant-specific, so
                # match by prefix rather than a fixed name.
                block_dict = _to_plain_data(block)
                if isinstance(block_dict, dict):
                    server_tool_blocks.append(block_dict)

        finish_reason = self._STOP_REASON_MAP.get(response.stop_reason, "stop")

        # Canonicalize tool_search_tool_*_tool_result block types to the
        # bare ``tool_search_tool_result`` form before persisting.
        # Anthropic's INPUT validator only accepts the bare canonical
        # type — variant-suffixed types (which appear on the wire
        # OUTPUT) are rejected with 400 "Input tag '<variant>_tool_result'
        # ... does not match any of the expected tags". Pydantic
        # already coerces fresh responses, but cached/streamed paths
        # can leak the wire variant; normalize here so persisted
        # sessions never contain a variant-suffixed type. See
        # _canonicalize_tool_search_result_types in
        # ``agent/anthropic_adapter.py`` for the full diagnosis.
        from agent.anthropic_adapter import _canonicalize_tool_search_result_types
        if server_tool_blocks:
            _canonicalize_tool_search_result_types(server_tool_blocks)

        provider_data = {}
        if reasoning_details:
            provider_data["reasoning_details"] = reasoning_details
        if server_tool_blocks:
            provider_data["server_tool_blocks"] = server_tool_blocks
        # Verbatim content array.  reasoning_details + tool_calls lose the
        # relative position of thinking blocks among text/tool_use blocks;
        # interleaved-thinking-2025-05-14 + clear_thinking_20251015 require
        # those positions to round-trip exactly or the API rejects the next
        # turn ("thinking blocks ... cannot be modified").  Captured here in
        # original order so convert_messages_to_anthropic can replay it
        # without recomposing.
        anthropic_content_blocks = _to_plain_data(response.content)
        if isinstance(anthropic_content_blocks, list) and anthropic_content_blocks:
            _canonicalize_tool_search_result_types(anthropic_content_blocks)
            provider_data["anthropic_content_blocks"] = anthropic_content_blocks
        # Structured stop_details (Anthropic SDK 0.88+, propagated through
        # streaming in 0.98+).  Today only refusal stops carry detail
        # (category=cyber|bio + human-readable explanation); future stop
        # types may add more.  Surface as-is so callers/UI can present
        # the refusal explanation rather than a bare "refusal" string.
        stop_details = _to_plain_data(getattr(response, "stop_details", None))
        if stop_details:
            provider_data["stop_details"] = stop_details

        return NormalizedResponse(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            reasoning="\n\n".join(reasoning_parts) if reasoning_parts else None,
            usage=None,
            provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        """Check Anthropic response structure is valid.

        An empty content list is legitimate when ``stop_reason == "end_turn"``
        — the model's canonical way of signalling "nothing more to add" after
        a tool turn that already delivered the user-facing text. Treating it
        as invalid falsely retries a completed response.
        """
        if response is None:
            return False
        content_blocks = getattr(response, "content", None)
        if not isinstance(content_blocks, list):
            return False
        if not content_blocks:
            return getattr(response, "stop_reason", None) == "end_turn"
        return True

    def extract_cache_stats(self, response: Any) -> Optional[Dict[str, int]]:
        """Extract Anthropic cache_read and cache_creation token counts."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        written = getattr(usage, "cache_creation_input_tokens", 0) or 0
        if cached or written:
            return {"cached_tokens": cached, "creation_tokens": written}
        return None

    # Promote the adapter's canonical mapping to module level so it's shared
    _STOP_REASON_MAP = {
        "end_turn": "stop",
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "stop_sequence": "stop",
        "refusal": "content_filter",
        "model_context_window_exceeded": "length",
    }

    def map_finish_reason(self, raw_reason: str) -> str:
        """Map Anthropic stop_reason to OpenAI finish_reason."""
        return self._STOP_REASON_MAP.get(raw_reason, "stop")


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("anthropic_messages", AnthropicTransport)
