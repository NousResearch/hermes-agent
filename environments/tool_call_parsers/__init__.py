"""
Tool Call Parser Registry

Client-side parsers that extract structured tool_calls from raw model output text.
Used in Phase 2 (VLLM server type) where ManagedServer's /generate endpoint returns
raw text without tool call parsing.

Each parser is a standalone reimplementation of the corresponding VLLM parser's
non-streaming extract_tool_calls() logic. No VLLM dependency -- only standard library
(re, json, uuid) and openai types.

Usage:
    from environments.tool_call_parsers import get_parser

    parser = get_parser("hermes")
    content, tool_calls = parser.parse(raw_model_output)
    # content = text with tool call markup stripped
    # tool_calls = list of ChatCompletionMessageToolCall objects, or None
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type

from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
)

logger = logging.getLogger(__name__)

# Type alias for parser return value
ParseResult = Tuple[Optional[str], Optional[List[ChatCompletionMessageToolCall]]]


class ToolCallParser(ABC):
    """
    Base class for tool call parsers.

    Each parser knows how to extract structured tool_calls from a specific
    model family's raw output text format.
    """

    @abstractmethod
    def parse(self, text: str) -> ParseResult:
        """
        Parse raw model output text for tool calls.

        Args:
            text: Raw decoded text from the model's completion

        Returns:
            Tuple of (content, tool_calls) where:
            - content: text with tool call markup stripped (the message 'content' field),
                       or None if the entire output was tool calls
            - tool_calls: list of ChatCompletionMessageToolCall objects,
                          or None if no tool calls were found
        """
        raise NotImplementedError


# Global parser registry: name -> parser class
PARSER_REGISTRY: Dict[str, Type[ToolCallParser]] = {}


def register_parser(name: str):
    """
    Decorator to register a parser class under a given name.

    Usage:
        @register_parser("hermes")
        class HermesToolCallParser(ToolCallParser):
            ...
    """

    def decorator(cls: Type[ToolCallParser]) -> Type[ToolCallParser]:
        PARSER_REGISTRY[name] = cls
        return cls

    return decorator


def get_parser(name: str) -> ToolCallParser:
    """
    Get a parser instance by name.

    Args:
        name: Parser name (e.g., "hermes", "mistral", "llama3_json")

    Returns:
        Instantiated parser

    Raises:
        KeyError: If parser name is not found in registry
    """
    if name not in PARSER_REGISTRY:
        available = sorted(PARSER_REGISTRY.keys())
        raise KeyError(
            f"Tool call parser '{name}' not found. Available parsers: {available}"
        )
    return PARSER_REGISTRY[name]()


def list_parsers() -> List[str]:
    """Return sorted list of registered parser names."""
    return sorted(PARSER_REGISTRY.keys())


# Import all parser modules to trigger registration via @register_parser decorators
# Each module registers itself when imported
from environments.tool_call_parsers.hermes_parser import HermesToolCallParser  # noqa: E402, F401
from environments.tool_call_parsers.longcat_parser import LongcatToolCallParser  # noqa: E402, F401
from environments.tool_call_parsers.mistral_parser import MistralToolCallParser  # noqa: E402, F401
from environments.tool_call_parsers.llama_parser import LlamaToolCallParser  # noqa: E402, F401
from environments.tool_call_parsers.qwen_parser import QwenToolCallParser  # noqa: E402, F401
from environments.tool_call_parsers.deepseek_v3_parser import DeepSeekV3ToolCallParser  # noqa: E402, F401
from environments.tool_call_parsers.deepseek_v3_1_parser import DeepSeekV31ToolCallParser  # noqa: E402, F401
from environments.tool_call_parsers.kimi_k2_parser import KimiK2ToolCallParser  # noqa: E402, F401
from environments.tool_call_parsers.glm45_parser import Glm45ToolCallParser  # noqa: E402, F401
from environments.tool_call_parsers.glm47_parser import Glm47ToolCallParser  # noqa: E402, F401
from environments.tool_call_parsers.qwen3_coder_parser import Qwen3CoderToolCallParser  # noqa: E402, F401
