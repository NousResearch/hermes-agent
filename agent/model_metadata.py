"""Model metadata, context lengths, and token estimation utilities.

Pure utility functions with no AIAgent dependency. Used by ContextCompressor
and run_agent.py for pre-flight context checks.
"""

import logging
import os
import time
from typing import Any, Dict, List

import requests

from hermes_constants import OPENROUTER_MODELS_URL

logger = logging.getLogger(__name__)

_model_metadata_cache: Dict[str, Dict[str, Any]] = {}
_model_metadata_cache_time: float = 0
_MODEL_CACHE_TTL = 3600

# Safe fallback context length for unknown models. Previous default of 128k
# caused failures when the actual model had a smaller context (e.g., 8k, 16k).
# 8192 is a conservative floor that works with most models.
# Override with MODEL_CONTEXT_LENGTH env var if needed.
SAFE_DEFAULT_CONTEXT_LENGTH = 8192

DEFAULT_CONTEXT_LENGTHS = {
    "anthropic/claude-opus-4": 200000,
    "anthropic/claude-opus-4.5": 200000,
    "anthropic/claude-opus-4.6": 200000,
    "anthropic/claude-sonnet-4": 200000,
    "anthropic/claude-sonnet-4-20250514": 200000,
    "anthropic/claude-haiku-4.5": 200000,
    "openai/gpt-4o": 128000,
    "openai/gpt-4-turbo": 128000,
    "openai/gpt-4o-mini": 128000,
    "google/gemini-2.0-flash": 1048576,
    "google/gemini-2.5-pro": 1048576,
    "meta-llama/llama-3.3-70b-instruct": 131072,
    "deepseek/deepseek-chat-v3": 65536,
    "qwen/qwen-2.5-72b-instruct": 32768,
}


def _get_fallback_context_length() -> int:
    """Get the fallback context length from env var or use safe default."""
    env_override = os.getenv("MODEL_CONTEXT_LENGTH")
    if env_override:
        try:
            return int(env_override)
        except ValueError:
            logger.warning(f"Invalid MODEL_CONTEXT_LENGTH value: {env_override}, using default")
    return SAFE_DEFAULT_CONTEXT_LENGTH


def fetch_model_metadata(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """Fetch model metadata from OpenRouter (cached for 1 hour)."""
    global _model_metadata_cache, _model_metadata_cache_time

    if not force_refresh and _model_metadata_cache and (time.time() - _model_metadata_cache_time) < _MODEL_CACHE_TTL:
        return _model_metadata_cache

    fallback_length = _get_fallback_context_length()
    
    try:
        response = requests.get(OPENROUTER_MODELS_URL, timeout=10)
        response.raise_for_status()
        data = response.json()

        cache = {}
        for model in data.get("data", []):
            model_id = model.get("id", "")
            cache[model_id] = {
                "context_length": model.get("context_length", fallback_length),
                "max_completion_tokens": model.get("top_provider", {}).get("max_completion_tokens", 4096),
                "name": model.get("name", model_id),
                "pricing": model.get("pricing", {}),
            }
            canonical = model.get("canonical_slug", "")
            if canonical and canonical != model_id:
                cache[canonical] = cache[model_id]

        _model_metadata_cache = cache
        _model_metadata_cache_time = time.time()
        logger.debug("Fetched metadata for %s models from OpenRouter", len(cache))
        return cache

    except Exception as e:
        logging.warning(f"Failed to fetch model metadata from OpenRouter: {e}")
        return _model_metadata_cache or {}


def get_model_context_length(model: str) -> int:
    """Get the context length for a model (API first, then fallback defaults).
    
    Resolution order:
    1. OpenRouter API metadata (live lookup)
    2. Built-in DEFAULT_CONTEXT_LENGTHS table (known models)
    3. MODEL_CONTEXT_LENGTH env var (user override)
    4. SAFE_DEFAULT_CONTEXT_LENGTH (8192 - conservative floor)
    
    When falling back to default, logs a warning so users know to set
    MODEL_CONTEXT_LENGTH if their model has a different context size.
    """
    fallback_length = _get_fallback_context_length()
    
    metadata = fetch_model_metadata()
    if model in metadata:
        return metadata[model].get("context_length", fallback_length)

    for default_model, length in DEFAULT_CONTEXT_LENGTHS.items():
        if default_model in model or model in default_model:
            return length

    # Unknown model - warn and use safe default
    logger.warning(
        f"Unknown model '{model}' - using conservative context length of {fallback_length:,} tokens. "
        f"Set MODEL_CONTEXT_LENGTH env var to override if your model supports more."
    )
    return fallback_length


def estimate_tokens_rough(text: str) -> int:
    """Rough token estimate (~4 chars/token) for pre-flight checks."""
    if not text:
        return 0
    return len(text) // 4


def estimate_messages_tokens_rough(messages: List[Dict[str, Any]]) -> int:
    """Rough token estimate for a message list (pre-flight only)."""
    total_chars = sum(len(str(msg)) for msg in messages)
    return total_chars // 4
