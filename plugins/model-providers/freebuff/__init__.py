"""Freebuff model provider — local OpenAI-compatible proxy to Codebuff free tier."""

from __future__ import annotations

import logging
import os
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://127.0.0.1:8765/v1"

_FALLBACK_MODELS = (
    "deepseek/deepseek-v4-flash",
    "deepseek/deepseek-v4-pro",
    "moonshotai/kimi-k2.6",
    "minimax/minimax-m2.7",
    "minimax/minimax-m3",
    "mimo/mimo-v2.5",
    "mimo/mimo-v2.5-pro",
    "google/gemini-2.5-flash-lite",
    "google/gemini-3.1-flash-lite-preview",
)


def _resolve_base_url(base_url: str | None = None) -> str:
    for candidate in (
        (base_url or "").strip(),
        os.environ.get("FREEBUFF_BASE_URL", "").strip(),
    ):
        if candidate:
            return candidate.rstrip("/")
    return _DEFAULT_BASE_URL


class FreebuffProfile(ProviderProfile):
    """Freebuff via local freebuff2api proxy."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        resolved = _resolve_base_url(base_url)
        try:
            result = super().fetch_models(
                api_key=api_key,
                base_url=resolved,
                timeout=timeout,
            )
            if result:
                return list(result)
        except Exception as exc:
            logger.debug("fetch_models(freebuff): %s", exc)
        return list(_FALLBACK_MODELS)


freebuff = FreebuffProfile(
    name="freebuff",
    aliases=("freebuff-proxy", "codebuff-free"),
    display_name="Freebuff",
    description="Ad-supported free Codebuff models via local OpenAI-compatible proxy",
    signup_url="https://freebuff.com/cli",
    env_vars=("FREEBUFF_PROXY_API_KEY", "FREEBUFF_BASE_URL"),
    base_url=_DEFAULT_BASE_URL,
    hostname="127.0.0.1",
    supports_vision=False,
    default_aux_model="deepseek/deepseek-v4-flash",
    fallback_models=_FALLBACK_MODELS,
)

register_provider(freebuff)
