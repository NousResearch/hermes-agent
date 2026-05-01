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

Fixes Issue #989: Support for multiple simultaneous tool calls.
"""

import re
import uuid
import logging
from typing import List, Optional, Tuple

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from environments.tool_call_parsers import ParseResult, ToolCallParser, register_parser

logger = logging.getLogger(__name__)

@register_parser("deepseek_v3")
class DeepSeekV3ToolCallParser(ToolCallParser):
    """
    Parser for DeepSeek V3 tool calls.

    Uses special unicode tokens with fullwidth angle brackets and block elements.
    Extracts type, function name, and JSON arguments from the structured format.
    Ensures all tool calls are captured when the model executes multiple actions.
    """

    START_TOKEN = "<｜tool▁calls▁begin｜>"

    # Updated PATTERN: Using \s* instead of literal \n for increased robustness
    # against variations in model formatting (Issue #989).
    PATTERN = re.compile(
        r"<｜tool▁call▁begin｜>(?P<type>.*?)<｜tool▁sep｜>(?P<function_name>.*?)\s*```json\s*(?P<function_arguments>.*?)\s*```\s*<｜tool▁call▁end｜>",
        re.DOTALL,
    )

    def parse(self, text: str) -> ParseResult:
        """
        Parses the input text and extracts all available tool calls.
        """
        if self.START_TOKEN not in text:
            return text, None

        try:
            # Using finditer to capture ALL tool calls in the sequence
            matches = list(self.PATTERN.finditer(text))
            if not matches:
                return text, None

            tool_calls: List[ChatCompletionMessageToolCall] = []
            
            for match in matches:
                func_name = match.group("function_name").strip()
                func_args = match.group("function_arguments").strip()
                
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=Function(
                            name=func_name,
                            arguments=func_args,
                        ),
                    )
                )

            if tool_calls:
                # Content is text before the first tool call block
                content_index = text.find(self.START_TOKEN)
                content = text[:content_index].strip()
                return content if content else None, tool_calls

            return text, None

        except Exception as e:
            logger.error(f"Error parsing DeepSeek V3 tool calls: {e}")
            return text, None
