"""
Mistral tool call parser.

Supports two formats depending on tokenizer version:
- Pre-v11: content[TOOL_CALLS] [{"name": ..., "arguments": {...}}, ...]
- v11+:    content[TOOL_CALLS]tool_name1{"arg": "val"}[TOOL_CALLS]tool_name2{"arg": "val"}

Based on VLLM's MistralToolParser.extract_tool_calls()
The [TOOL_CALLS] token is the bot_token used by Mistral models.
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


def _generate_mistral_id() -> str:
    """Mistral tool call IDs are 9-char alphanumeric strings."""
    import random
    import string

    return "".join(random.choices(string.ascii_letters + string.digits, k=9))


@register_parser("mistral")
class MistralToolCallParser(ToolCallParser):
    """
    Parser for Mistral-format tool calls.

    Detects format by checking if the content after [TOOL_CALLS] starts with '['
    (pre-v11 JSON array) or with a tool name (v11+ format).
    """

    # The [TOOL_CALLS] token -- may appear as different strings depending on tokenizer
    BOT_TOKEN = "[TOOL_CALLS]"

    # Fallback regex for pre-v11 format when JSON parsing fails
    TOOL_CALL_REGEX = re.compile(r"\[?\s*(\{.*?\})\s*\]?", re.DOTALL)

    def parse(self, text: str) -> ParseResult:
        if self.BOT_TOKEN not in text:
            return text, None

        try:
            parts = text.split(self.BOT_TOKEN)
            content = parts[0].strip()
            raw_tool_calls = parts[1:]

            # Detect format: if the first raw part starts with '[', it's pre-v11
            first_raw = raw_tool_calls[0].strip() if raw_tool_calls else ""
            is_pre_v11 = first_raw.startswith("[") or first_raw.startswith("{")

            tool_calls: List[ChatCompletionMessageToolCall] = []

            if not is_pre_v11:
                # v11+ format: [TOOL_CALLS]tool_name{args}[TOOL_CALLS]tool_name2{args2}
                for raw in raw_tool_calls:
                    raw = raw.strip()
                    if not raw or "{" not in raw:
                        continue

                    brace_idx = raw.find("{")
                    tool_name = raw[:brace_idx].strip()
                    args_str = raw[brace_idx:]

                    tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id=_generate_mistral_id(),
                            type="function",
                            function=Function(name=tool_name, arguments=args_str),
                        )
                    )
            else:
                # Pre-v11 format: [TOOL_CALLS] [{"name": ..., "arguments": {...}}]
                try:
                    parsed = json.loads(first_raw)
                    if isinstance(parsed, dict):
                        parsed = [parsed]

                    for tc in parsed:
                        args = tc.get("arguments", {})
                        if isinstance(args, dict):
                            args = json.dumps(args, ensure_ascii=False)

                        tool_calls.append(
                            ChatCompletionMessageToolCall(
                                id=_generate_mistral_id(),
                                type="function",
                                function=Function(
                                    name=tc["name"], arguments=args
                                ),
                            )
                        )
                except json.JSONDecodeError:
                    # Fallback regex extraction
                    match = self.TOOL_CALL_REGEX.findall(first_raw)
                    if match:
                        for raw_json in match:
                            try:
                                tc = json.loads(raw_json)
                                args = tc.get("arguments", {})
                                if isinstance(args, dict):
                                    args = json.dumps(args, ensure_ascii=False)
                                tool_calls.append(
                                    ChatCompletionMessageToolCall(
                                        id=_generate_mistral_id(),
                                        type="function",
                                        function=Function(
                                            name=tc["name"], arguments=args
                                        ),
                                    )
                                )
                            except (json.JSONDecodeError, KeyError):
                                continue

            if not tool_calls:
                return text, None

            return content if content else None, tool_calls

        except Exception:
            return text, None
