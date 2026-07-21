"""Anthropic Messages API transport: conversion via agent/anthropic_adapter.py, normalization here."""

from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall

_MCP_PREFIX = "mcp__"
_THINKING_TYPES = ("thinking", "redacted_thinking")
# Blocks of an Anthropic-executed server tool (native web search / fetch).
_SERVER_TOOL_BLOCK_TYPES = ("server_tool_use", "web_search_tool_result", "web_fetch_tool_result")


def _unprefix_oauth_tool_name(name: str) -> str:
    """Reverse the OAuth-wire ``mcp__`` prefix back to the registered tool name.
    Two originals map onto one wire name (``read_file`` / ``mcp_linear_get_issue``), so
    resolve by registry lookup, never rewriting a name that already resolves natively.
    OAuth wire aliases are checked LAST so a real tool under the wire name still wins."""
    from agent.anthropic_adapter import _OAUTH_TOOL_NAME_REVERSE_ALIASES
    from tools.registry import registry as _tool_registry
    bare = name[len(_MCP_PREFIX):]
    for candidate in (name, "mcp_" + bare, bare):
        if _tool_registry.get_entry(candidate):
            return candidate
    return _OAUTH_TOOL_NAME_REVERSE_ALIASES.get(bare, name)


# build_kwargs params forwarded to build_anthropic_kwargs, with the defaults applied when absent.
_BUILD_KWARG_DEFAULTS = {
    "max_tokens": 16384, "reasoning_config": None, "tool_choice": None, "is_oauth": False, "preserve_dots": False,
    "context_length": None, "base_url": None, "fast_mode": False, "drop_context_1m_beta": False,
}


class AnthropicTransport(ProviderTransport):
    """Transport for api_mode='anthropic_messages'."""

    _STOP_REASON_MAP = {
        "end_turn": "stop", "tool_use": "tool_calls", "max_tokens": "length", "stop_sequence": "stop",
        "refusal": "content_filter", "model_context_window_exceeded": "length", "pause_turn": "pause_turn",
    }

    @property
    def api_mode(self) -> str:
        return "anthropic_messages"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Convert OpenAI messages to an Anthropic (system, messages) tuple; ``base_url`` affects thinking-signature handling."""
        from agent.anthropic_message_convert import convert_messages_to_anthropic
        return convert_messages_to_anthropic(messages, base_url=kwargs.get("base_url"))

    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        """Convert OpenAI tool schemas to Anthropic input_schema format."""
        from agent.anthropic_message_convert import convert_tools_to_anthropic
        return convert_tools_to_anthropic(tools)

    def build_kwargs(
        self, model: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **params,
    ) -> Dict[str, Any]:
        """Build Anthropic messages.create() kwargs (converts messages and tools internally)."""
        from agent.anthropic_adapter import build_anthropic_kwargs
        return build_anthropic_kwargs(
            model=model, messages=messages, tools=tools,
            **{key: params.get(key, default) for key, default in _BUILD_KWARG_DEFAULTS.items()},
        )

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Parse content blocks (text/thinking/tool_use), map stop_reason, collect reasoning_details."""
        import json
        from agent.anthropic_message_convert import _sanitize_replay_block, _to_plain_data
        strip_tool_prefix = kwargs.get("strip_tool_prefix", False)
        text_parts, reasoning_parts, reasoning_details, tool_calls = [], [], [], []
        citation_sources, seen_citation_urls = [], set()

        def _add_citation_source(url: Any, title: Any = None) -> None:
            if not isinstance(url, str) or not url or url in seen_citation_urls:
                return
            seen_citation_urls.add(url)
            citation_sources.append((" ".join(str(title or url).split()), url))

        # Anthropic signs each thinking block against the blocks PRECEDING it; when thinking
        # interleaves with tool_use the parallel lists lose that order and replay -> HTTP 400.
        ordered_blocks = []
        for block in response.content:
            block_dict = _to_plain_data(block)
            # Sanitize at capture so output-only SDK fields never persist and replay (400).
            clean_block = _sanitize_replay_block(block_dict) if isinstance(block_dict, dict) else None
            if clean_block is not None:
                ordered_blocks.append(clean_block)
            if block.type == "text":
                text_parts.append(block.text)
                # Citations arrive as structured metadata, not inline Markdown: ordered_blocks keeps them for
                # replay, and a compact source list is rendered into the neutral text channel below so
                # CLI/gateway users do not lose the URLs.
                for citation in getattr(block, "citations", None) or []:
                    citation_dict = _to_plain_data(citation)
                    if not isinstance(citation_dict, dict):
                        continue
                    url = citation_dict.get("url")
                    _add_citation_source(url, citation_dict.get("title") or citation_dict.get("cited_text") or url)
            elif block.type == "web_fetch_tool_result":
                # Fetch citations may use document-relative locations with no URL on the text block; the
                # server result stays the authoritative source for the fetched URL, so surface it as a fallback.
                result_content = (clean_block or {}).get("content")
                if isinstance(result_content, dict):
                    _add_citation_source(result_content.get("url"), result_content.get("title"))
            elif block.type in _THINKING_TYPES:
                if block.type == "thinking":
                    reasoning_parts.append(block.thinking)
                detail = clean_block if clean_block is not None else block_dict  # raw only if sanitize dropped it
                if isinstance(detail, dict):
                    reasoning_details.append(detail)
            elif block.type == "tool_use":
                name = block.name
                if strip_tool_prefix and name.startswith(_MCP_PREFIX):
                    name = _unprefix_oauth_tool_name(name)
                tool_calls.append(ToolCall(id=block.id, name=name, arguments=json.dumps(block.input)))
        provider_data = {"reasoning_details": reasoning_details} if reasoning_details else {}
        # Ordered channel only for the shapes the parallel lists cannot reconstruct: signed thinking
        # interleaved with tool_use, and server-tool blocks, which no parallel list carries at all.
        signed = any(b.get("type") in _THINKING_TYPES and (b.get("signature") or b.get("data")) for b in ordered_blocks)
        if (signed and any(b.get("type") == "tool_use" for b in ordered_blocks)) or any(
            b.get("type") in _SERVER_TOOL_BLOCK_TYPES for b in ordered_blocks
        ):
            provider_data["anthropic_content_blocks"] = ordered_blocks
        content = "\n".join(text_parts) if text_parts else None
        if citation_sources:
            sources = "Sources:\n" + "\n".join(
                f"- {url}" if title == url else f"- {title}: {url}" for title, url in citation_sources
            )
            content = f"{content}\n\n{sources}" if content else sources
        return NormalizedResponse(
            content=content, tool_calls=tool_calls or None,
            finish_reason=self.response_finish_reason(response),
            reasoning="\n\n".join(reasoning_parts) if reasoning_parts else None, usage=None,
            provider_data=provider_data or None,
        )

    def response_finish_reason(self, response: Any) -> str:
        """``stop_reason`` mapped to the OpenAI vocabulary. Bedrock InvokeModel guardrail blocks keep
        ``stop_reason=end_turn`` and hand back the guardrail's canned text as an ordinary reply; they
        must surface as ``content_filter`` so the loop treats them as a refusal, not model output."""
        from agent.bedrock_adapter import anthropic_response_guardrail_intervened
        if anthropic_response_guardrail_intervened(response):
            return "content_filter"
        return self.map_finish_reason(response.stop_reason)

    def validate_response(self, response: Any) -> bool:
        """Structural check; empty content is legitimate for ``end_turn``/``refusal`` (retrying
        either would loop forever)."""
        content_blocks = getattr(response, "content", None)
        return isinstance(content_blocks, list) and (
            bool(content_blocks) or getattr(response, "stop_reason", None) in {"end_turn", "refusal"}
        )

    def extract_cache_stats(self, response: Any) -> Optional[Dict[str, int]]:
        """Anthropic cache_read / cache_creation token counts."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        written = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return {"cached_tokens": cached, "creation_tokens": written} if cached or written else None


from agent.transports import register_transport  # noqa: E402

register_transport("anthropic_messages", AnthropicTransport)
