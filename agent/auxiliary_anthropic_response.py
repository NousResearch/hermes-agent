"""Anthropic Messages -> chat.completions shape for auxiliary callers (``_AnthropicCompletionsAdapter``).

Aux consumers read ``choices[0].message``. The MoA aggregator also keeps that message in the agent's
chat-completions history, so a field dropped here is lost to every later hop.
"""

import logging
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)


def anthropic_chat_choice(response: Any, normalized: Any, model: str) -> SimpleNamespace:
    """``choices[0]`` for a normalized Anthropic response.

    A refusal's reason (``stop_details``: category, explanation) rides on the message, where the
    chat-completions normalizer reads it; without it MoA and aux refusals report no reason.
    """
    stop_details = (normalized.provider_data or {}).get("stop_details")
    if normalized.finish_reason == "content_filter":
        logger.warning("Anthropic declined an auxiliary request: model=%s stop_reason=%s stop_details=%s",
                       model, getattr(response, "stop_reason", None), stop_details)
    # ToolCall already duck-types as OpenAI shape via properties.
    message = SimpleNamespace(content=normalized.content, tool_calls=normalized.tool_calls,
                              reasoning=normalized.reasoning, stop_details=stop_details)
    return SimpleNamespace(index=0, message=message, finish_reason=normalized.finish_reason)
