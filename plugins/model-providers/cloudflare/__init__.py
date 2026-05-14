"""Cloudflare Workers AI provider profile."""

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
        account_id = get_env_value("CLOUDFLARE_ACCOUNT_ID")
        if not account_id:
            return None

        # Cloudflare's OpenAI-compatible models list is at /v1/models
        # The base URL is constructed dynamically.
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
        return super().fetch_models(api_key=api_key, timeout=timeout)


cloudflare = CloudflareProfile(
    name="cloudflare",
    aliases=("cf", "cloudflare-ai"),
    display_name="Cloudflare Workers AI",
    description="Cloudflare Workers AI — serverless GPU inference",
    signup_url="https://dash.cloudflare.com/?to=/:account/ai/workers-ai/models",
    env_vars=("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"),
    # base_url is dynamic; placeholder for registry
    base_url="https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
    fallback_models=(
        "@cf/meta/llama-3.1-8b-instruct",
        "@cf/meta/llama-3.1-70b-instruct",
        "@cf/meta/llama-3-8b-instruct",
        "@cf/mistral/mistral-7b-instruct-v0.1",
        "@hf/nousresearch/hermes-2-pro-llama-3-8b",
    ),
)

register_provider(cloudflare)
