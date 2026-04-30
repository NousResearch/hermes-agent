"""OpenAI Responses API (Codex) transport.

Delegates to the existing adapter functions in agent/codex_responses_adapter.py.
This transport owns format conversion and normalization — NOT client lifecycle,
streaming, or the _run_codex_stream() call path.
"""

from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import NormalizedResponse, ToolCall


class ResponsesApiTransport(ProviderTransport):
    """Transport for api_mode='codex_responses'.

    Wraps the functions extracted into codex_responses_adapter.py (PR 1).
    """

    @property
    def api_mode(self) -> str:
        return "codex_responses"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Convert OpenAI chat messages to Responses API input items."""
        from agent.codex_responses_adapter import _chat_messages_to_responses_input
        return _chat_messages_to_responses_input(messages)

    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        """Convert OpenAI tool schemas to Responses API function definitions."""
        from agent.codex_responses_adapter import _responses_tools
        return _responses_tools(tools)

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Build Responses API kwargs.

        Calls convert_messages and convert_tools internally.

        params:
            instructions: str — system prompt (extracted from messages[0] if not given)
            reasoning_config: dict | None — {effort, enabled}
            session_id: str | None — used for prompt_cache_key + xAI conv header
            max_tokens: int | None — max_output_tokens
            request_overrides: dict | None — extra kwargs merged in
            provider: str | None — provider name for backend-specific logic
            base_url: str | None — endpoint URL
            base_url_hostname: str | None — hostname for backend detection
            is_github_responses: bool — Copilot/GitHub models backend
            is_codex_backend: bool — chatgpt.com/backend-api/codex
            is_xai_responses: bool — xAI/Grok backend
            github_reasoning_extra: dict | None — Copilot reasoning params
        """
        from agent.codex_responses_adapter import (
            _chat_messages_to_responses_input,
            _responses_tools,
        )

        from run_agent import DEFAULT_AGENT_IDENTITY

        instructions = params.get("instructions", "")
        payload_messages = messages
        if not instructions:
            if messages and messages[0].get("role") == "system":
                instructions = str(messages[0].get("content") or "").strip()
                payload_messages = messages[1:]
        if not instructions:
            instructions = DEFAULT_AGENT_IDENTITY

        is_github_responses = params.get("is_github_responses", False)
        is_codex_backend = params.get("is_codex_backend", False)
        is_xai_responses = params.get("is_xai_responses", False)

        # Resolve reasoning effort
        reasoning_effort = "medium"
        reasoning_enabled = True
        reasoning_config = params.get("reasoning_config")
        if reasoning_config and isinstance(reasoning_config, dict):
            if reasoning_config.get("enabled") is False:
                reasoning_enabled = False
            elif reasoning_config.get("effort"):
                reasoning_effort = reasoning_config["effort"]

        _effort_clamp = {"minimal": "low"}
        reasoning_effort = _effort_clamp.get(reasoning_effort, reasoning_effort)

        kwargs = {
            "model": model,
            "instructions": instructions,
            "input": _chat_messages_to_responses_input(payload_messages),
            "tools": _responses_tools(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "store": False,
        }

        session_id = params.get("session_id")
        if not is_github_responses and session_id:
            kwargs["prompt_cache_key"] = session_id

        if reasoning_enabled and is_xai_responses:
            kwargs["include"] = ["reasoning.encrypted_content"]
        elif reasoning_enabled:
            if is_github_responses:
                github_reasoning = params.get("github_reasoning_extra")
                if github_reasoning is not None:
                    kwargs["reasoning"] = github_reasoning
            else:
                kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
                kwargs["include"] = ["reasoning.encrypted_content"]
        elif not is_github_responses and not is_xai_responses:
            kwargs["include"] = []

        request_overrides = params.get("request_overrides")
        if request_overrides:
            kwargs.update(request_overrides)

        if is_codex_backend:
            prompt_cache_key = kwargs.get("prompt_cache_key")
            cache_scope_id = str(prompt_cache_key or session_id or "").strip()
            if cache_scope_id:
                existing_extra_headers = kwargs.get("extra_headers")
                merged_extra_headers: Dict[str, str] = {}
                if isinstance(existing_extra_headers, dict):
                    merged_extra_headers.update(
                        {
                            str(key): str(value)
                            for key, value in existing_extra_headers.items()
                            if key and value is not None
                        }
                    )
                merged_extra_headers["session_id"] = cache_scope_id
                merged_extra_headers["x-client-request-id"] = cache_scope_id
                kwargs["extra_headers"] = merged_extra_headers

        max_tokens = params.get("max_tokens")
        if max_tokens is not None and not is_codex_backend:
            kwargs["max_output_tokens"] = max_tokens

        if is_xai_responses and session_id:
            kwargs["extra_headers"] = {"x-grok-conv-id": session_id}

        return kwargs

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize Codex Responses API response to NormalizedResponse."""
        from agent.codex_responses_adapter import (
            _normalize_codex_response,
        )

        # _normalize_codex_response returns (SimpleNamespace, finish_reason_str)
        msg, finish_reason = _normalize_codex_response(response)

        tool_calls = None
        if msg and msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                provider_data = {}
                if hasattr(tc, "call_id") and tc.call_id:
                    provider_data["call_id"] = tc.call_id
                if hasattr(tc, "response_item_id") and tc.response_item_id:
                    provider_data["response_item_id"] = tc.response_item_id
                tool_calls.append(ToolCall(
                    id=tc.id if hasattr(tc, "id") else (tc.function.name if hasattr(tc, "function") else None),
                    name=tc.function.name if hasattr(tc, "function") else getattr(tc, "name", ""),
                    arguments=tc.function.arguments if hasattr(tc, "function") else getattr(tc, "arguments", "{}"),
                    provider_data=provider_data or None,
                ))

        # Extract reasoning items for provider_data
        provider_data = {}
        if msg and hasattr(msg, "codex_reasoning_items") and msg.codex_reasoning_items:
            provider_data["codex_reasoning_items"] = msg.codex_reasoning_items
        if msg and hasattr(msg, "codex_message_items") and msg.codex_message_items:
            provider_data["codex_message_items"] = msg.codex_message_items
        if msg and hasattr(msg, "reasoning_details") and msg.reasoning_details:
            provider_data["reasoning_details"] = msg.reasoning_details

        return NormalizedResponse(
            content=msg.content if msg else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            reasoning=msg.reasoning if msg and hasattr(msg, "reasoning") else None,
            usage=None,  # Codex usage is extracted separately in normalize_usage()
            provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        """Check Codex Responses API response has valid output structure.

        Returns True only if response.output is a non-empty list.
        Does NOT check output_text fallback — the caller handles that
        with diagnostic logging for stream backfill recovery.
        """
        if response is None:
            return False
        output = getattr(response, "output", None)
        if not isinstance(output, list) or not output:
            return False
        return True

    def preflight_kwargs(self, api_kwargs: Any, *, allow_stream: bool = False) -> dict:
        """Validate and sanitize Codex API kwargs before the call.

        Normalizes input items, strips unsupported fields, validates structure.
        """
        from agent.codex_responses_adapter import _preflight_codex_api_kwargs
        return _preflight_codex_api_kwargs(api_kwargs, allow_stream=allow_stream)

    def map_finish_reason(self, raw_reason: str) -> str:
        """Map Codex response.status to OpenAI finish_reason.

        Codex uses response.status ('completed', 'incomplete') +
        response.incomplete_details.reason for granular mapping.
        This method handles the simple status string; the caller
        should check incomplete_details separately for 'max_output_tokens'.
        """
        _MAP = {
            "completed": "stop",
            "incomplete": "length",
            "failed": "stop",
            "cancelled": "stop",
        }
        return _MAP.get(raw_reason, "stop")


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("codex_responses", ResponsesApiTransport)
