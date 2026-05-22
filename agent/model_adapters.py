"""Model-specific adapters — extracted from run_agent.py.

Centralizes provider/model-specific message preparation, vision support,
reasoning configuration, and other model-family quirks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── URL / Provider detection ─────────────────────────────────────


def is_direct_openai_url(agent, base_url: str = None) -> bool:
    """True when base URL targets OpenAI's native API."""
    from utils import base_url_hostname
    if base_url is not None:
        hostname = base_url_hostname(base_url)
    else:
        hostname = getattr(agent, "_base_url_hostname", "") or base_url_hostname(
            getattr(agent, "_base_url_lower", "")
        )
    return hostname == "api.openai.com"


def is_azure_openai_url(agent, base_url: str = None) -> bool:
    """True when base URL targets Azure OpenAI."""
    if base_url is not None:
        url = str(base_url).lower()
    else:
        url = getattr(agent, "_base_url_lower", "") or ""
    return "openai.azure.com" in url


def is_github_copilot_url(agent, base_url: str = None) -> bool:
    """True when base URL targets GitHub Copilot."""
    from utils import base_url_hostname
    if base_url is not None:
        hostname = base_url_hostname(base_url)
    else:
        hostname = getattr(agent, "_base_url_hostname", "") or base_url_hostname(
            getattr(agent, "_base_url_lower", "")
        )
    return hostname == "api.githubcopilot.com"


def is_openrouter_url(agent) -> bool:
    """True when the base URL targets OpenRouter."""
    from utils import base_url_host_matches
    return base_url_host_matches(getattr(agent, "_base_url_lower", ""), "openrouter.ai")


def is_ollama_glm_backend(agent) -> bool:
    """True when provider is ollama or glm."""
    provider = (getattr(agent, "provider", "") or "").strip().lower()
    return provider in ("ollama", "glm")


def is_qwen_portal(agent) -> bool:
    """True when provider is qwen-oauth (Qwen portal)."""
    return getattr(agent, "provider", "") == "qwen-oauth"


def model_requires_responses_api(model: str) -> bool:
    """True for models requiring the Responses API path (GPT-5.x)."""
    m = model.lower()
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    return m.startswith("gpt-5")


def provider_model_requires_responses_api(model: str, *, provider: Optional[str] = None) -> bool:
    """True when this provider/model pair should use Responses API."""
    normalized_provider = (provider or "").strip().lower()
    if normalized_provider == "nous":
        return False
    if normalized_provider == "copilot":
        try:
            from hermes_cli.models import _should_use_copilot_responses_api
            return _should_use_copilot_responses_api(model)
        except Exception:
            pass
    return model_requires_responses_api(model)


# ── Vision support ───────────────────────────────────────────────


def content_has_image_parts(content: Any) -> bool:
    """Check if content contains image_url parts."""
    if isinstance(content, str):
        return False
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"image_url", "image", "input_image"}:
                return True
    return False


def api_kwargs_have_image_parts(api_kwargs: dict) -> bool:
    """Check if API kwargs contain image parts in messages."""
    def _contains_image(value: Any) -> bool:
        if isinstance(value, str):
            return False
        if isinstance(value, list):
            return any(_contains_image(v) for v in value)
        if isinstance(value, dict):
            if value.get("type") in {"image_url", "image", "input_image"}:
                return True
            return any(_contains_image(v) for v in value.values())
        return False
    messages = api_kwargs.get("messages", [])
    return any(_contains_image(m) for m in messages)


def model_supports_vision(agent) -> bool:
    """Check if the current model supports vision/image inputs."""
    model = (getattr(agent, "model", "") or "").lower()
    api_mode = getattr(agent, "api_mode", "") or ""
    if api_mode == "anthropic_messages":
        # Most Anthropic models support vision
        return "claude-3" in model or "claude-4" in model or "claude-sonnet-4" in model
    return True  # Assume vision support unless proven otherwise


# ── Model-specific message preparation ──────────────────────────


def get_transport(agent, api_mode: str = None) -> Any:
    """Get the message transport for the given api_mode."""
    mode = api_mode or getattr(agent, "api_mode", "") or "chat_completions"
    try:
        from agent.transports import get_transport as _get
        return _get(mode)
    except Exception:
        return None


# ── Reasoning helpers ────────────────────────────────────────────


def supports_reasoning_extra_body(agent) -> bool:
    """Check if the current provider supports reasoning via extra_body."""
    from utils import base_url_host_matches
    provider = (getattr(agent, "provider", "") or "").strip().lower()
    base_url = getattr(agent, "_base_url_lower", "") or ""
    # OpenRouter supports reasoning via extra_body
    if provider == "openrouter" or base_url_host_matches(base_url, "openrouter.ai"):
        return True
    # Direct OpenAI
    if provider == "openai" or base_url_host_matches(base_url, "api.openai.com"):
        return True
    return False


def needs_thinking_reasoning_pad(agent) -> bool:
    """Check if the model needs a thinking/reasoning content pad."""
    model = (getattr(agent, "model", "") or "").lower()
    return any(kw in model for kw in ("deepseek-r1", "deepseek-r1-distill", "qwq"))


def needs_kimi_tool_reasoning(agent) -> bool:
    """Check if Kimi/Moonshot needs tool reasoning workaround."""
    provider = (getattr(agent, "provider", "") or "").strip().lower()
    return provider in ("kimi", "moonshot")


def needs_deepseek_tool_reasoning(agent) -> bool:
    """Check if DeepSeek needs tool reasoning workaround."""
    provider = (getattr(agent, "provider", "") or "").strip().lower()
    return provider == "deepseek"


def needs_mimo_tool_reasoning(agent) -> bool:
    """Check if Xiaomi MiMo needs tool reasoning workaround."""
    provider = (getattr(agent, "provider", "") or "").strip().lower()
    return provider == "mimo"
