"""
Hermes tool call parser.

Format: <tool_call>{"name": "func", "arguments": {...}}</tool_call>
Based on VLLM's Hermes2ProToolParser.extract_tool_calls()
"""

import json
from typing import List

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

from environments.tool_call_parsers import ParseResult, ToolCallParser, register_parser
from environments.tool_call_parsers.common import (
    extract_tagged_payloads,
    make_tool_call,
    split_content_from_first_marker,
)


@register_parser("hermes")
class HermesToolCallParser(ToolCallParser):
    """
    Parser for Hermes-format tool calls.

    Matches <tool_call>...</tool_call> tags containing JSON with "name" and
    "arguments". Also handles unclosed <tool_call> at end-of-string
    (truncated generation).
    """

    OPEN_TAG = "<tool_call>"
    CLOSE_TAG = "</tool_call>"

    def parse(self, text: str) -> ParseResult:
        if self.OPEN_TAG not in text:
            return text, None

        try:
            payloads = extract_tagged_payloads(text, self.OPEN_TAG, self.CLOSE_TAG)
            if not payloads:
                return text, None

            tool_calls: List[ChatCompletionMessageToolCall] = []
            for raw_json in payloads:
                tc_data = json.loads(raw_json)
                tool_calls.append(
                    make_tool_call(
                        name=tc_data["name"],
                        arguments=tc_data.get("arguments", {}),
                    )
                )

            if not tool_calls:
                return text, None

            content, _ = split_content_from_first_marker(text, self.OPEN_TAG)
            return content, tool_calls

        except Exception:
            return text, None
