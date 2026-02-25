"""Shared auxiliary OpenAI client for cheap/fast side tasks.

Provides a single resolution chain so every consumer (context compression,
session search, web extraction, vision analysis, browser vision) picks up
the best available backend without duplicating fallback logic.

Resolution order for text tasks:
  1. OpenRouter  (OPENROUTER_API_KEY)
  2. Nous Portal (~/.hermes/auth.json active provider)
  3. Custom endpoint (OPENAI_BASE_URL + OPENAI_API_KEY)
  4. None

Resolution order for vision/multimodal tasks:
  1. OpenRouter
  2. Nous Portal
  3. None  (custom endpoints can't substitute for Gemini multimodal)
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

from openai import OpenAI

from hermes_constants import OPENROUTER_BASE_URL

logger = logging.getLogger(__name__)

# Default auxiliary models per provider
_OPENROUTER_MODEL = "google/gemini-3-flash-preview"
_NOUS_MODEL = "gemini-3-flash"
_NOUS_DEFAULT_BASE_URL = "https://inference-api.nousresearch.com/v1"
_AUTH_JSON_PATH = Path.home() / ".hermes" / "auth.json"


def _read_nous_auth() -> Optional[dict]:
    """Read and validate ~/.hermes/auth.json for an active Nous provider.

    Returns the provider state dict if Nous is active with tokens,
    otherwise None.
    """
    try:
        if not _AUTH_JSON_PATH.is_file():
            return None
        data = json.loads(_AUTH_JSON_PATH.read_text())
        if data.get("active_provider") != "nous":
            return None
        provider = data.get("providers", {}).get("nous", {})
        # Must have at least an access_token or agent_key
        if not provider.get("agent_key") and not provider.get("access_token"):
            return None
        return provider
    except Exception as exc:
        logger.debug("Could not read Nous auth: %s", exc)
        return None


def _nous_api_key(provider: dict) -> str:
    """Extract the best API key from a Nous provider state dict."""
    return provider.get("agent_key") or provider.get("access_token", "")


def _nous_base_url() -> str:
    """Resolve the Nous inference base URL from env or default."""
    return os.getenv("NOUS_INFERENCE_BASE_URL", _NOUS_DEFAULT_BASE_URL)


# ── Public API ──────────────────────────────────────────────────────────────


def get_text_auxiliary_client() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Return (client, model_slug) for text-only auxiliary tasks.

    Falls through OpenRouter -> Nous Portal -> custom endpoint -> (None, None).
    """
    # 1. OpenRouter
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        logger.debug("Auxiliary text client: OpenRouter")
        return OpenAI(api_key=or_key, base_url=OPENROUTER_BASE_URL), _OPENROUTER_MODEL

    # 2. Nous Portal
    nous = _read_nous_auth()
    if nous:
        logger.debug("Auxiliary text client: Nous Portal")
        return (
            OpenAI(api_key=_nous_api_key(nous), base_url=_nous_base_url()),
            _NOUS_MODEL,
        )

    # 3. Custom endpoint (both base URL and key must be set)
    custom_base = os.getenv("OPENAI_BASE_URL")
    custom_key = os.getenv("OPENAI_API_KEY")
    if custom_base and custom_key:
        model = os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        logger.debug("Auxiliary text client: custom endpoint (%s)", model)
        return OpenAI(api_key=custom_key, base_url=custom_base), model

    # 4. Nothing available
    logger.debug("Auxiliary text client: none available")
    return None, None


def get_vision_auxiliary_client() -> Tuple[Optional[OpenAI], Optional[str]]:
    """Return (client, model_slug) for vision/multimodal auxiliary tasks.

    Only OpenRouter and Nous Portal qualify — custom endpoints cannot
    substitute for Gemini multimodal.
    """
    # 1. OpenRouter
    or_key = os.getenv("OPENROUTER_API_KEY")
    if or_key:
        logger.debug("Auxiliary vision client: OpenRouter")
        return OpenAI(api_key=or_key, base_url=OPENROUTER_BASE_URL), _OPENROUTER_MODEL

    # 2. Nous Portal
    nous = _read_nous_auth()
    if nous:
        logger.debug("Auxiliary vision client: Nous Portal")
        return (
            OpenAI(api_key=_nous_api_key(nous), base_url=_nous_base_url()),
            _NOUS_MODEL,
        )

    # 3. Nothing suitable
    logger.debug("Auxiliary vision client: none available")
    return None, None
