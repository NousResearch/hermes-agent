"""Claude Code CLI transport.

Converts between Hermes's OpenAI-format messages and Claude Code CLI's
stream-json event format.  Unlike other transports, Claude Code manages its
own tools (Read, Edit, Bash, etc.) — we don't pass Hermes tool definitions.

The build_kwargs output is consumed by a CLI runner, not an SDK client.
"""

import json
from typing import Any, Dict, List, Optional

from agent.transports.base import ProviderTransport
from agent.transports.types import (
    NormalizedResponse,
    ToolCall,
    Usage,
    build_tool_call,
)


class ClaudeCliTransport(ProviderTransport):
    """Transport for api_mode='claude_cli'.

    Bridges Hermes agent conversations to Claude Code CLI sessions.
    Claude Code executes tools natively — tool_calls in the normalized
    response are informational only (for logging/tracing).
    """

    @property
    def api_mode(self) -> str:
        return "claude_cli"

    def convert_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Separate system messages from conversation messages.

        Returns a dict with:
            system_prompt: str | None — concatenated system message content,
                suitable for --append-system-prompt.
            conversation: list — non-system messages in original order.
        """
        system_parts: list[str] = []
        conversation: list[Dict[str, Any]] = []

        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Handle structured content blocks
                    text = " ".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                    if text:
                        system_parts.append(text)
                elif content:
                    system_parts.append(content)
            else:
                conversation.append(msg)

        return {
            "system_prompt": "\n\n".join(system_parts) if system_parts else None,
            "conversation": conversation,
        }

    def convert_tools(self, tools: List[Dict[str, Any]]) -> Any:
        """Return None — Claude Code has its own native tools."""
        return None

    def build_kwargs(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **params,
    ) -> Dict[str, Any]:
        """Build kwargs dict for the Claude Code CLI runner.

        Returns a dict with:
            prompt: str — the last user message text.
            model: str — model identifier.
            system_prompt: str | None — from system messages.
            working_dir: str | None — working directory for the CLI session.
            session_id: str | None — resume an existing session.
            timeout: int | None — max execution time in seconds.

        params (all optional):
            working_dir: str
            session_id: str
            timeout: int
        """
        converted = self.convert_messages(messages)

        # Extract the last user message as the prompt string
        prompt = ""
        for msg in reversed(converted["conversation"]):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    prompt = " ".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                elif isinstance(content, str):
                    prompt = content
                break

        return {
            "prompt": prompt,
            "model": model,
            "system_prompt": converted["system_prompt"],
            "working_dir": params.get("working_dir"),
            "session_id": params.get("session_id"),
            "timeout": params.get("timeout"),
        }

    def normalize_response(self, response: Any, **kwargs) -> NormalizedResponse:
        """Normalize a list of StreamEvent objects to NormalizedResponse.

        Iterates over Claude Code CLI stream events and collects:
        - Text from non-partial ASSISTANT and RESULT events
        - Tool use events into ToolCall list (informational — already executed)
        - Thinking content from thinking blocks
        - Usage from the RESULT event
        - session_id into provider_data

        Args:
            response: list of StreamEvent objects from claude_code_core.
        """
        events: list = response if isinstance(response, list) else [response]

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage: Optional[Usage] = None
        session_id: Optional[str] = None
        has_error = False
        error_text: Optional[str] = None

        for event in events:
            msg_type = getattr(event, "message_type", None)
            # Normalize enum to string for comparison
            if msg_type is not None and not isinstance(msg_type, str):
                msg_type = msg_type.value if hasattr(msg_type, "value") else str(msg_type)

            # Capture session_id from any event that has it
            if getattr(event, "session_id", None):
                session_id = event.session_id

            # Check for errors
            if getattr(event, "error", None):
                has_error = True
                error_text = event.error
                continue

            # Skip partial events — only process complete chunks
            if getattr(event, "is_partial", False):
                continue

            # Collect thinking content
            if getattr(event, "thinking", None):
                thinking_parts.append(event.thinking)

            # Collect tool use events
            tool_use = getattr(event, "tool_use", None)
            if tool_use is not None:
                tool_calls.append(
                    build_tool_call(
                        id=getattr(tool_use, "tool_id", None),
                        name=getattr(tool_use, "tool_name", "unknown"),
                        arguments=getattr(tool_use, "tool_input", {}),
                    )
                )

            # Collect text from assistant and result events
            text = getattr(event, "text", None)
            if text and msg_type in ("assistant", "result"):
                text_parts.append(text)

            # Extract usage from result events
            if msg_type == "result":
                input_tok = getattr(event, "input_tokens", None) or 0
                output_tok = getattr(event, "output_tokens", None) or 0
                cache_read = getattr(event, "cache_read_tokens", None) or 0
                usage = Usage(
                    prompt_tokens=input_tok,
                    completion_tokens=output_tok,
                    total_tokens=input_tok + output_tok,
                    cached_tokens=cache_read,
                )

        # Determine finish reason
        finish_reason = "error" if has_error else "stop"

        # Build provider_data
        provider_data: Dict[str, Any] = {}
        if session_id:
            provider_data["session_id"] = session_id
        if error_text:
            provider_data["error"] = error_text

        return NormalizedResponse(
            content="\n".join(text_parts) if text_parts else None,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            reasoning="\n\n".join(thinking_parts) if thinking_parts else None,
            usage=usage,
            provider_data=provider_data or None,
        )

    def validate_response(self, response: Any) -> bool:
        """Check that event list is non-empty and contains useful content.

        Returns True if there is at least one RESULT event or an event
        with text content.
        """
        if not response:
            return False

        events = response if isinstance(response, list) else [response]
        for event in events:
            msg_type = getattr(event, "message_type", None)
            if msg_type is not None and not isinstance(msg_type, str):
                msg_type = msg_type.value if hasattr(msg_type, "value") else str(msg_type)

            if msg_type == "result":
                return True
            if getattr(event, "text", None):
                return True

        return False

    def map_finish_reason(self, raw_reason: str) -> str:
        """Map CLI-specific reasons to normalized set."""
        return self._FINISH_REASON_MAP.get(raw_reason, raw_reason)

    _FINISH_REASON_MAP: Dict[str, str] = {
        "stop": "stop",
        "error": "error",
        "timeout": "length",
    }


# Auto-register on import
from agent.transports import register_transport  # noqa: E402

register_transport("claude_cli", ClaudeCliTransport)
