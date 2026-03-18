"""
Auto-detecting tool call parser for local/self-hosted models.

This parser attempts to detect and parse various tool call formats commonly
output by local models (llama.cpp, Ollama, etc.) that don't follow the
standard OpenAI tool_calls format.

Supported formats:
- Raw JSON: {"name": "tool", "arguments": "{...}"}
- XML wrapped: <tool_call>{...}</tool_call>
- XML with parameters: <tool_use><parameter=name>...</parameter></tool_use>
- Markdown code blocks: ```bash command```
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


@register_parser("auto")
class AutoToolCallParser(ToolCallParser):
    """
    Auto-detecting parser that tries multiple formats.
    
    Priority order:
    1. XML parameter format (<parameter=tool_name>)
    2. XML wrapped format (<tool_call>{...}</tool_call>)
    3. Raw JSON format ({"name": "...", "arguments": "..."})
    4. Markdown code blocks (```bash ...```)
    """

    # Pattern for XML parameter format
    PARAM_PATTERN = re.compile(
        r"<parameter=(\w+)>(.*?)</parameter>",
        re.DOTALL,
    )

    # Pattern for raw JSON tool calls
    JSON_TOOL_PATTERN = re.compile(
        r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*"((?:[^"\\]|\\.)*)"\s*\}'
    )

    # Pattern for markdown code blocks
    CODE_BLOCK_PATTERN = re.compile(
        r'```(?:bash|shell|sh)?\s*\n(.*?)\n```',
        re.DOTALL,
    )

    def _parse_xml_params(self, text: str) -> Optional[List[ChatCompletionMessageToolCall]]:
        """Parse XML parameter format."""
        if "<parameter=" not in text:
            return None

        param_matches = list(self.PARAM_PATTERN.finditer(text))
        if not param_matches:
            return None

        # Find tool boundaries
        tool_starts = []
        for match in param_matches:
            if match.group(1) == "tool_name":
                tool_starts.append(match.start())

        if not tool_starts:
            return None

        tool_calls = []
        seen = set()

        for tool_start in tool_starts:
            remaining = text[tool_start:]
            end_pos = len(text)
            for close_tag in ["</tool_use>", "</tool_call>"]:
                pos = remaining.find(close_tag)
                if pos != -1:
                    end_pos = tool_start + pos + len(close_tag)
                    break

            block_content = text[tool_start:end_pos]
            params = {}
            for match in self.PARAM_PATTERN.finditer(block_content):
                params[match.group(1).strip()] = match.group(2).strip()

            if "tool_name" in params:
                tool_name = params["tool_name"]
                args_str = params.get("arguments", "{}")
                try:
                    arguments = json.loads(args_str) if args_str.startswith("{") else {"value": args_str}
                except json.JSONDecodeError:
                    arguments = {}

                key = (tool_name, json.dumps(arguments, sort_keys=True))
                if key not in seen:
                    seen.add(key)
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

        return tool_calls if tool_calls else None

    def _parse_raw_json(self, text: str) -> Optional[List[ChatCompletionMessageToolCall]]:
        """Parse raw JSON format."""
        matches = self.JSON_TOOL_PATTERN.findall(text)
        
        if not matches:
            # Try with single quotes
            pattern = re.compile(r"\{\s*'name'\s*:\s*'([^']+)'\s*,\s*'arguments'\s*:\s*'((?:[^'\\]|\\.)*)'\s*\}")
            matches = pattern.findall(text)

        if not matches:
            return None

        tool_calls = []
        for tool_name, args_str in matches:
            try:
                args_str = args_str.replace('\\"', '"')
                json.loads(args_str)  # Validate
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=Function(
                            name=tool_name,
                            arguments=args_str,
                        ),
                    )
                )
            except json.JSONDecodeError:
                continue

        return tool_calls if tool_calls else None

    def _parse_code_blocks(self, text: str) -> Optional[List[ChatCompletionMessageToolCall]]:
        """Parse markdown code blocks as terminal commands."""
        matches = self.CODE_BLOCK_PATTERN.findall(text)
        if not matches:
            return None

        tool_calls = []
        for cmd in matches:
            cmd = cmd.strip()
            if cmd:
                tool_calls.append(
                    ChatCompletionMessageToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        type="function",
                        function=Function(
                            name="terminal",
                            arguments=json.dumps({"command": cmd}),
                        ),
                    )
                )

        return tool_calls if tool_calls else None

    def parse(self, text: str) -> ParseResult:
        """Parse text trying multiple formats."""
        if not text:
            return text, None

        # Try each format in priority order
        parsers = [
            ("xml_params", self._parse_xml_params),
            ("raw_json", self._parse_raw_json),
            ("code_blocks", self._parse_code_blocks),
        ]

        for name, parser_fn in parsers:
            try:
                tool_calls = parser_fn(text)
                if tool_calls:
                    # Strip parsed content from text
                    cleaned = text
                    if name == "code_blocks":
                        cleaned = self.CODE_BLOCK_PATTERN.sub('', cleaned).strip()
                    elif name == "raw_json":
                        cleaned = self.JSON_TOOL_PATTERN.sub('', cleaned).strip()
                    elif name == "xml_params":
                        # For XML, find and remove the tags
                        for tag in ["<tool_use>", "</tool_use>", "<tool_call>", "</tool_call>"]:
                            cleaned = cleaned.replace(tag, "")
                        # Remove parameter tags
                        cleaned = self.PARAM_PATTERN.sub('', cleaned).strip()
                    
                    return cleaned if cleaned else None, tool_calls
            except Exception:
                continue

        return text, None
