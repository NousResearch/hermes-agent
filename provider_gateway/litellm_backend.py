"""LiteLLM optional backend wrapper for the provider gateway.

Provides dynamic, opt-in integration with the LiteLLM SDK to support 100+ LLM backends.
Import-safe: does not crash if the 'litellm' package is not installed.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_LITELLM_AVAILABLE = False
try:
    import litellm

    # Configure LiteLLM defensively
    litellm.drop_params = True  # Safely ignore parameters unsupported by specific backends
    litellm.set_verbose = False
    _LITELLM_AVAILABLE = True
except ImportError:
    pass


def is_available() -> bool:
    """Return True if the 'litellm' package is successfully installed and available."""
    return _LITELLM_AVAILABLE


def complete(
    model: str,
    messages: list[dict[str, Any]],
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    stream: bool = False,
    **kwargs,
) -> Any:
    """Call a model using LiteLLM completion API.

    Requires 'litellm' to be installed.
    """
    if not _LITELLM_AVAILABLE:
        raise ImportError(
            "The 'litellm' package is not installed. "
            "Please install it using: pip install hermes-agent[gateway]"
        )

    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if api_key is not None:
        params["api_key"] = api_key
    if api_base is not None:
        params["api_base"] = api_base

    # Update with extra arguments
    params.update(kwargs)

    return litellm.completion(**params)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate the cost of a completion request in USD.

    Returns 0.0 if estimation is not possible or litellm is not available.
    """
    if not _LITELLM_AVAILABLE:
        return 0.0
    try:
        cost = litellm.completion_cost(
            model=model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
        if cost is None:
            return 0.0
        return float(cost)
    except Exception as exc:
        logger.debug("LiteLLM cost estimation failed: %s", exc)
        return 0.0


def list_models() -> list[str]:
    """Return a list of all model names supported natively by LiteLLM."""
    if not _LITELLM_AVAILABLE:
        return []
    try:
        return list(litellm.model_list)
    except Exception as exc:
        logger.debug("LiteLLM list models failed: %s", exc)
        return []
