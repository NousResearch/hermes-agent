"""
Hermes tool call parser.

Format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
Based on VLLM's Hermes2ProToolParser.extract_tool_calls()
"""

import json
import re
import uuid
from typing import List, Optional, Tuple

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from environments.tool_call_parsers import ParseResult, ToolCallParser, register_parser


@register_parser("hermes")
class HermesToolCallParser(ToolCallParser):
    """
    Parser for Hermes-format tool calls.

    Matches <tool_call>...</tool_call> tags containing JSON with "name" and "arguments".
    Also handles unclosed <tool_call> at end-of-string (truncated generation).
    """

    # Matches both closed and unclosed tool_call tags
    PATTERN = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>|<tool_call>\s*(.*)", re.DOTALL
    )

    def parse(self, text: str) -> ParseResult:
        if "<tool_call>" not in text:
            return text, None

        try:
            matches = self.PATTERN.findall(text)
            if not matches:
                return text, None

            tool_calls: List[ChatCompletionMessageToolCall] = []
            for match in matches:
                # match is a tuple: (closed_content, unclosed_content)
                raw_json = match[0] if match[0] else match[1]
                if not raw_json.strip():
                    continue

                tc_data = json.loads(raw_json)
                # Handle arguments: could be dict or already a JSON string
                raw_args = tc_data.get("arguments", {})
                if isinstance(raw_args, str):
                    # Already a string — pass through as-is.
                    # It may be a JSON string ("{...}") or a plain string ("ls").
                    args_str = raw_args
                else:
                    # Dict — serialize to JSON
                    args_str = json.dumps(raw_args, ensure_ascii=False)
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=Function(
                            name=tc_data["name"],
                            arguments=args_str,
                        ),
                    )
                )

            if not tool_calls:
                return text, None

            # Content is everything before the first <tool_call> tag
            content = text[: text.find("<tool_call>")].strip()
            return content if content else None, tool_calls

        except Exception:
            return text, None
