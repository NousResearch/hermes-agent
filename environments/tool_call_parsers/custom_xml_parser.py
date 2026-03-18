"""
Custom XML tool call parser for models that output <tool_use> or <tool_call> format.

Handles formats like:
<tool_use>
<parameter=tool_name>web_search</parameter>
<parameter=arguments>{"query": "..."}</parameter>
</tool_use>

Or hybrid:
<tool_call>
<tool_use>
<parameter=tool_name>...</parameter>
<parameter=arguments>...</parameter>
</tool_use>
</tool_call>
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


@register_parser("custom_xml")
class CustomXmlToolCallParser(ToolCallParser):
    """
    Parser for custom XML-style tool calls with <tool_use> or <tool_call> tags
    containing <parameter> elements.
    """

    # Pattern to extract parameters
    PARAM_PATTERN = re.compile(
        r"<parameter=(\w+)>(.*?)</parameter>",
        re.DOTALL,
    )

    def _parse_block(self, block_content: str) -> Optional[Tuple[str, dict]]:
        """Parse a single tool block and return (tool_name, arguments)."""
        params = {}
        for match in self.PARAM_PATTERN.finditer(block_content):
            param_name = match.group(1).strip()
            param_value = match.group(2).strip()
            params[param_name] = param_value

        if "tool_name" in params:
            tool_name = params["tool_name"]
            args_str = params.get("arguments", "{}")
            try:
                if args_str.startswith("{"):
                    arguments = json.loads(args_str)
                else:
                    try:
                        arguments = json.loads(args_str)
                    except json.JSONDecodeError:
                        arguments = {"value": args_str}
            except json.JSONDecodeError:
                arguments = {}
            return tool_name, arguments

        # Alternative format: <parameter=name>
        if "name" in params:
            tool_name = params["name"]
            args_str = params.get("arguments", "{}")
            try:
                arguments = json.loads(args_str) if args_str.startswith("{") else {"value": args_str}
            except json.JSONDecodeError:
                arguments = {}
            return tool_name, arguments

        return None

    def parse(self, text: str) -> ParseResult:
        """Parse text for XML-style tool calls."""
        if "<tool_use>" not in text and "<tool_call>" not in text:
            return text, None

        tool_calls: List[ChatCompletionMessageToolCall] = []
        seen_tools = set()
        content_parts = []

        # Find all blocks that contain <parameter=tool_name>
        param_matches = list(self.PARAM_PATTERN.finditer(text))
        
        if not param_matches:
            return text, None

        # Find tool boundaries by looking for <parameter=tool_name> tags
        tool_starts = []
        for i, match in enumerate(param_matches):
            if match.group(1) == "tool_name":
                tool_starts.append(match.start())

        if not tool_starts:
            return text, None

        # Extract content before first tool
        first_tool_start = tool_starts[0]
        search_start = max(0, first_tool_start - 200)
        prefix_text = text[search_start:first_tool_start]
        tool_tag_start = -1
        for tag in ["<tool_call>", "<tool_use>"]:
            pos = prefix_text.rfind(tag)
            if pos != -1:
                tool_tag_start = search_start + pos
                break
        
        if tool_tag_start != -1:
            content_before = text[:tool_tag_start].strip()
            if content_before:
                content_parts.append(content_before)

        # Extract each tool
        for i, tool_start in enumerate(tool_starts):
            block_start = tool_start
            remaining = text[tool_start:]
            end_pos = len(text)
            for close_tag in ["</tool_use>", "</tool_call>"]:
                pos = remaining.find(close_tag)
                if pos != -1:
                    end_pos = tool_start + pos + len(close_tag)
                    break
            
            block_content = text[block_start:end_pos]
            parsed = self._parse_block(block_content)
            if parsed:
                tool_name, arguments = parsed
                tool_key = (tool_name, json.dumps(arguments, sort_keys=True))
                if tool_key not in seen_tools:
                    seen_tools.add(tool_key)
                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=f"call_{uuid.uuid4().hex[:8]}",
                            type="function",
                            function=Function(
                                name=tool_name,
                                arguments=json.dumps(arguments, ensure_ascii=False),
                            ),
                        )
                    )

        content = " ".join(content_parts).strip()
        if not tool_calls:
            return text, None

        return content if content else None, tool_calls
