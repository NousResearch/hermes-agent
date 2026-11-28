"""Whether a reasoning-only clean stop may be promoted to the visible answer.

A field name is not provenance: OpenRouter, DeepSeek, Bedrock and several adapters all put
private chain-of-thought in ``reasoning``. Promotion is therefore limited to trusted routes,
decided from the agent's LIVE route on every call so fallback activation, primary restore and
model switches never carry a decision to another route.
"""

from __future__ import annotations

from typing import Any

from agent.route_capabilities import configured_route_capability
from utils import base_url_host_matches

_ANSWER_IN_REASONING_CAPABILITY = "answer_in_reasoning"
_NEMOTRON_PARSER_MODEL_MARKER = "nemotron-3.5-lightning"


def answer_in_reasoning_capability(agent: Any) -> bool:
    """True when the live route may return a clean-stop reasoning payload as the answer.

    Trusted: a custom_providers per-model ``answer_in_reasoning`` opt-in or provider-level
    ``capabilities: {answer_in_reasoning: ...}`` block on the live route, or the local
    Nemotron-3.5-Lightning parser route from #109205. OpenRouter and non-chat-completions
    transports are never trusted.
    """
    base_url = str(getattr(agent, "base_url", "") or "")
    provider = str(getattr(agent, "provider", "") or "").strip().lower()
    if provider == "openrouter" or base_url_host_matches(base_url, "openrouter.ai"):
        return False

    api_mode = str(getattr(agent, "api_mode", "") or "").strip().lower()
    if api_mode and api_mode != "chat_completions":
        return False

    configured = configured_route_capability(agent, _ANSWER_IN_REASONING_CAPABILITY)
    if configured is not None:
        return configured

    model = str(getattr(agent, "model", "") or "")
    if _NEMOTRON_PARSER_MODEL_MARKER not in model.lower():
        return False
    from agent.model_metadata import is_local_endpoint

    return is_local_endpoint(base_url)
