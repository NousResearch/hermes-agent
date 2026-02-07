"""
Llama 3.x / 4 tool call parser.

Format: The model outputs JSON objects with "name" and "arguments" (or "parameters") keys.
May be preceded by <|python_tag|> token. Supports multiple JSON objects separated
by content or semicolons.

Based on VLLM's Llama3JsonToolParser.extract_tool_calls()
"""

import json
import re
import uuid
from typing import List, Optional

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from environments.tool_call_parsers import ParseResult, ToolCallParser, register_parser


@register_parser("llama3_json")
@register_parser("llama4_json")
class LlamaToolCallParser(ToolCallParser):
    """
    Parser for Llama 3.x and 4 JSON-format tool calls.

    Finds JSON objects containing "name" + ("arguments" or "parameters") keys.
    Uses Python's json.JSONDecoder.raw_decode for robust extraction of
    JSON objects from mixed text.
    """

    BOT_TOKEN = "<|python_tag|>"

    # Regex to find the start of potential JSON objects
    JSON_START = re.compile(r"\{")

    def parse(self, text: str) -> ParseResult:
        # Quick check: need either the bot token or a JSON brace
        if self.BOT_TOKEN not in text and "{" not in text:
            return text, None

        try:
            decoder = json.JSONDecoder()
            tool_calls: List[ChatCompletionMessageToolCall] = []
            end_index = -1  # Track where the last parsed JSON ended

            for match in self.JSON_START.finditer(text):
                start = match.start()
                # Skip if this brace is inside a previously parsed JSON object
                if start <= end_index:
                    continue

                try:
                    obj, json_end = decoder.raw_decode(text[start:])
                    end_index = start + json_end

                    # Must have "name" and either "arguments" or "parameters"
                    name = obj.get("name")
                    args = obj.get("arguments", obj.get("parameters"))

                    if not name or args is None:
                        continue

                    # Normalize arguments to JSON string
                    if isinstance(args, dict):
                        args = json.dumps(args, ensure_ascii=False)
                    elif not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)

                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            type="function",
                            function=Function(name=name, arguments=args),
                        )
                    )
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue

            if not tool_calls:
                return text, None

            # Content is everything before the first tool call JSON
            # Find where the first tool call starts in the text
            first_tc_start = text.find("{")
            if self.BOT_TOKEN in text:
                first_tc_start = text.find(self.BOT_TOKEN)
            content = text[:first_tc_start].strip() if first_tc_start > 0 else None

            return content, tool_calls

        except Exception:
            return text, None
