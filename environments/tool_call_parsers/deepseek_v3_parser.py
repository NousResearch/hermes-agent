"""
DeepSeek V3 tool call parser.

Format uses special unicode tokens:
    <｜tool▁calls▁begin｜>
    <｜tool▁call▁begin｜>type<｜tool▁sep｜>function_name
    ```json
    {"arg": "value"}
    ```
    <｜tool▁call▁end｜>
    <｜tool▁calls▁end｜>

Based on VLLM's DeepSeekV3ToolParser.extract_tool_calls()
"""

import re
import uuid
from typing import List, Optional

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from environments.tool_call_parsers import ParseResult, ToolCallParser, register_parser


@register_parser("deepseek_v3")
class DeepSeekV3ToolCallParser(ToolCallParser):
    """
    Parser for DeepSeek V3 tool calls.

    Uses special unicode tokens with fullwidth angle brackets and block elements.
    Extracts type, function name, and JSON arguments from the structured format.
    """

    START_TOKEN = "<｜tool▁calls▁begin｜>"

    # Regex captures: type, function_name, function_arguments
    PATTERN = re.compile(
        r"<｜tool▁call▁begin｜>(?P<type>.*)<｜tool▁sep｜>(?P<function_name>.*)\n```json\n(?P<function_arguments>.*)\n```<｜tool▁call▁end｜>"
    )

    def parse(self, text: str) -> ParseResult:
        if self.START_TOKEN not in text:
            return text, None

        try:
            matches = self.PATTERN.findall(text)
            if not matches:
                return text, None

            tool_calls: List[ChatCompletionMessageToolCall] = []
            for match in matches:
                tc_type, func_name, func_args = match
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=Function(
                            name=func_name.strip(),
                            arguments=func_args.strip(),
                        ),
                    )
                )

            if not tool_calls:
                return text, None

            # Content is everything before the tool calls section
            content = text[: text.find(self.START_TOKEN)].strip()
            return content if content else None, tool_calls

        except Exception:
            return text, None
