"""Concentrate AI provider profile.

Concentrate exposes an OpenAI Responses-compatible endpoint that routes to
many underlying model authors (Anthropic, Google, OpenAI, Meta, Mistral, …)
behind a single API key, with Zero Data Retention available for supported models.

API reference: https://concentrate.ai/docs/api-reference/endpoint/create-response
"""

from __future__ import annotations

import json
import logging
import urllib.request
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)


class ConcentrateProfile(ProviderProfile):
    """Concentrate AI — Responses API gateway with ZDR support."""

    auth_type = "api_key"

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch models from Concentrate catalog.

        The Concentrate /v1/models endpoint returns a raw list of objects with
        a ``slug`` field (not the OpenAI ``{"data": [{"id": ...}]}`` shape).
        No auth required for the catalog endpoint.
        """
        url = self.models_url
        req = urllib.request.Request(url)
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", _profile_user_agent())
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            if not isinstance(data, list):
                return None
            return [m["slug"] for m in data if isinstance(m, dict) and "slug" in m]
        except Exception as exc:
            logger.debug("fetch_models(concentrate): %s", exc)
            return None


concentrate = ConcentrateProfile(
    name="concentrate",
    aliases=("concentrateai", "concentrate-ai"),
    api_mode="codex_responses",
    display_name="Concentrate AI",
    description="Concentrate AI — unified gateway, Zero Data Retention for supported models",
    signup_url="https://concentrate.ai",
    env_vars=("CONCENTRATE_API_KEY",),
    base_url="https://api.concentrate.ai/v1",
    models_url="https://api.concentrate.ai/v1/models",
    fallback_models=(
        "claude-sonnet-4-6",
        "claude-opus-4-5",
        "gpt-4o",
        "gemini-2.5-flash",
        "meta-llama/llama-3.3-70b-instruct",
        "deepseek-r1",
        "mistral-large",
    ),
)

register_provider(concentrate)
