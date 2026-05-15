"""Cloudflare Workers AI provider profile."""

from __future__ import annotations

import os
from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class CloudflareProfile(ProviderProfile):
    """Cloudflare Workers AI — requires account_id in base_url."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Cloudflare's /models endpoint requires account_id in the URL."""
        from hermes_cli.config import get_env_value
        from hermes_cli.models import _fetch_cloudflare_models

        account_id = get_env_value("CLOUDFLARE_ACCOUNT_ID")
        if not account_id or not api_key:
            return None

        # Use the specific Cloudflare search API for dynamic discovery
        return _fetch_cloudflare_models(api_key, account_id, timeout=timeout)

    def build_extra_body(
        self,
        *,
        session_id: str | None = None,
        **context: Any,
    ) -> dict[str, Any]:
        """Attach Cloudflare AI Gateway attribution when configured."""
        extra: dict[str, Any] = {}
        gateway_id = os.getenv("CLOUDFLARE_GATEWAY_ID", "").strip()
        if gateway_id:
            extra["cf_gateway_id"] = gateway_id
        return extra


cloudflare = CloudflareProfile(
    name="cloudflare",
    aliases=("cf", "cloudflare-ai"),
    display_name="Cloudflare Workers AI",
    description="Cloudflare Workers AI — Llama 3, Mistral, Gemma, Hermes on serverless GPUs",
    signup_url="https://dash.cloudflare.com/?to=/:account/ai/workers-ai/models",
    env_vars=("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"),
    # base_url is dynamic; {account_id} is resolved at runtime.
    # When CLOUDFLARE_GATEWAY_ID is set, the URL is rewritten to the Gateway endpoint.
    base_url="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
    fallback_models=(
        # 2026 frontier models
        "@cf/meta/llama-4-scout-17b",
        "@cf/meta/llama-4-70b",
        "@cf/meta/llama-3.1-8b-instruct",
        "@cf/meta/llama-3.1-70b-instruct",
        "@cf/meta/llama-3-8b-instruct",
        "@cf/meta/llama-3-70b-instruct",
        "@cf/mistral/mistral-7b-instruct-v0.1",
        "@cf/mistral/mistral-small-3.1-24b-instruct-2503",
        # Google models on CF
        "@cf/google/gemma-4-26b-a4b-it",
        "@cf/google/gemma-4-2b-it",
        "@cf/google/gemma-3-27b-it",
        "@cf/google/gemma-27b-it",
        "@cf/google/gemma-7b-it",
        # Moonshot / Kimi on CF
        "@cf/moonshotai/kimi-k2.6",
        "@cf/moonshotai/kimi-k2.5",
        # NVIDIA on CF
        "@cf/nvidia/nemotron-mini-4b-instruct",
        # Hermes fine-tunes
        "@hf/nousresearch/hermes-2-pro-llama-3-8b",
        "@hf/nousresearch/hermes-3-llama-3.1-8b",
        # Open-weight community models
        "@cf/qwen/qwen1.5-7b-chat-awq",
        "@cf/qwen/qwen2.5-7b-instruct",
        "@cf/phi-2/phi-2",
    ),
    default_headers={
        "User-Agent": "HermesAgent/0.13.0",
    },
)

register_provider(cloudflare)