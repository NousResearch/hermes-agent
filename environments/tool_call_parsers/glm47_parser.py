"""
GLM 4.7 tool call parser.

Same as GLM 4.5 but with slightly different regex patterns.
The tool_call tags may wrap differently and arg parsing handles
newlines between key/value pairs.

Based on VLLM's Glm47MoeModelToolParser (extends Glm4MoeModelToolParser).
"""

import re

from environments.tool_call_parsers import ParseResult, register_parser
from environments.tool_call_parsers.glm45_parser import Glm45ToolCallParser


@register_parser("glm47")
class Glm47ToolCallParser(Glm45ToolCallParser):
    """
    Parser for GLM 4.7 tool calls.
    Extends GLM 4.5 with updated regex patterns.
    """

    def __init__(self):
        super().__init__()
        # GLM 4.7 uses a slightly different detail regex that includes
        # the <tool_call> wrapper and optional arg_key content
        self.FUNC_DETAIL_REGEX = re.compile(r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL)
        # GLM 4.7 handles newlines between arg_key and arg_value tags
        self.FUNC_ARG_REGEX = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )
