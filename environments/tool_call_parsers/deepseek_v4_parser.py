"""
DeepSeek V4 (DSML) tool call parser.

DeepSeek-V4 uses DSML (DeepSeek Markup Language) format for tool calls.
When the API provider doesn't parse these into structured tool_calls,
they appear as raw text in the response content.

DSML tool call format (inferred from observed output):
    <｜DSML｜tool_calls_begin｜>
    <｜DSML｜tool_call_begin｜>function_name
    ```json
    {"arg": "value"}
    ```
    <｜DSML｜tool_call_end｜>
    <｜DSML｜tool_calls_end｜>

The parser also handles the ASCII pipe variant observed in issue #15453:
    < | DSML | tool_calls_begin >
    < | DSML | tool_call_begin > function_name
    ...
    < | DSML | tool_call_end >
    < | DSML | tool_calls_end >
"""

import re
import uuid
import logging
from typing import List, Optional

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)

from environments.tool_call_parsers import ParseResult, ToolCallParser, register_parser

logger = logging.getLogger(__name__)

# Unicode fullwidth pipe ＋ Unicode block element separator (▁ = U+2581)
# Matches: <｜DSML｜tool_calls_begin｜> and <｜DSML｜tool_calls_end｜>
_UNICODE_DSML_BEGIN = r"<｜DSML｜tool.calls.begin｜>"
_UNICODE_DSML_CALL_BEGIN = r"<｜DSML｜tool.call.begin｜>"
_UNICODE_DSML_CALL_END = r"<｜DSML｜tool.call.end｜>"

# ASCII pipe variant with optional spaces (as seen in issue #15453 raw output)
# Matches: < | DSML | tool_calls_begin > etc.
_ASCII_DSML_BEGIN = r"<\s*\|\s*DSML\s*\|\s*tool.calls.begin\s*>"
_ASCII_DSML_CALL_BEGIN = r"<\s*\|\s*DSML\s*\|\s*tool.call.begin\s*>"
_ASCII_DSML_CALL_END = r"<\s*\|\s*DSML\s*\|\s*tool.call.end\s*>"

# Combined pattern: either Unicode or ASCII variant
_DSML_CALL_BEGIN = f"(?:{_UNICODE_DSML_CALL_BEGIN}|{_ASCII_DSML_CALL_BEGIN})"
_DSML_CALL_END = f"(?:{_UNICODE_DSML_CALL_END}|{_ASCII_DSML_CALL_END})"

# Full tool-call block pattern:
#   <call_begin> function_name
#   ```json
#   {arguments}
#   ```
#   <call_end>
# Also handles the case where arguments are raw JSON without code fence.
TOOL_CALL_PATTERN = re.compile(
    _DSML_CALL_BEGIN
    + r"\s*(?P<function_name>[^\n<|]+?)\s*"
    + r"(?:```json\s*(?P<json_args>.*?)\s*```|(?P<raw_args>\{[^}]*\}))"
    + r"\s*"
    + _DSML_CALL_END,
    re.DOTALL,
)


@register_parser("deepseek_v4")
@register_parser("deepseek_dsml")
class DeepSeekV4ToolCallParser(ToolCallParser):
    """
    Parser for DeepSeek V4 DSML tool calls.

    Handles both Unicode fullwidth and ASCII pipe variants of the DSML
    tool call delimiters.  Extracts function name and JSON arguments
    from each tool call block.
    """

    # Fast-path sentinel: skip regex if neither variant is present.
    _SENTINELS = ("<｜DSML｜", "< | DSML |")

    def parse(self, text: str) -> ParseResult:
        if not any(s in text for s in self._SENTINELS):
            return text, None

        try:
            matches = list(TOOL_CALL_PATTERN.finditer(text))
            if not matches:
                return text, None

            tool_calls: List[ChatCompletionMessageToolCall] = []

            for match in matches:
                func_name = match.group("function_name").strip()
                func_args = (
                    match.group("json_args") or match.group("raw_args") or "{}"
                ).strip()

                if not func_name:
                    continue

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

            if not tool_calls:
                return text, None

            # Content is everything before the first DSML tag
            first_match = matches[0]
            content = text[: first_match.start()].strip()
            return content if content else None, tool_calls

        except Exception as e:
            logger.error(f"Error parsing DeepSeek V4 DSML tool calls: {e}")
            return text, None
