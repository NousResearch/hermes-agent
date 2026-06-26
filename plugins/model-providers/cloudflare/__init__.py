"""Cloudflare Workers AI provider profile.

Uses the OpenAI-compatible endpoint at:
  https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1

Set CLOUDFLARE_ACCOUNT_ID and the base URL will be constructed automatically.
Or set CLOUDFLARE_BASE_URL to the full URL including your account ID.
"""

import json
import logging
import os
import urllib.request

from providers import register_provider
from providers.base import ProviderProfile, _profile_user_agent

logger = logging.getLogger(__name__)

# Models the search API returns with @cf/meta-llama/ but that only work
# via inference as @cf/meta/.  The search endpoint is the authoritative
# catalog but its names don't always match what the inference endpoint
# accepts — this mapping normalises the discrepancy.
_MODEL_NAME_FIXUPS = {
    "@cf/meta-llama/": "@cf/meta/",
}


def _fix_model_name(name: str) -> str:
    """Map search-endpoint display names to working inference names."""
    for prefix, replacement in _MODEL_NAME_FIXUPS.items():
        if name.startswith(prefix):
            return replacement + name[len(prefix):]
    return name


def _fetch_cloudflare_models(
    *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
) -> list[str] | None:
    """Fetch the live model list from Cloudflare's /ai/models/search endpoint.

    Returns model ID strings with ``meta-llama`` normalised to ``meta`` so
    they work with the inference endpoint.
    """
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    if not account_id:
        return None

    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/models/search"
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", _profile_user_agent())

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        if not data.get("success") or not data.get("result"):
            return None
        models = []
        for m in data["result"]:
            task = m.get("task", {}).get("name", "")
            if task != "Text Generation":
                continue
            name = m.get("name", "")
            if name:
                models.append(_fix_model_name(name))
        return models or None
    except Exception as exc:
        logger.debug("fetch_cloudflare_models: %s", exc)
        return None


cloudflare = ProviderProfile(
    name="cloudflare",
    aliases=("workers-ai", "cf"),
    display_name="Cloudflare Workers AI",
    description="Cloudflare Workers AI — serverless GPU inference",
    signup_url="https://dash.cloudflare.com/sign-up",
    env_vars=("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID"),
    auth_type="api_key",
    supports_health_check=False,
    fallback_models=(
        "@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        "@cf/meta/llama-4-scout-17b-16e-instruct",
        "@cf/meta/llama-3.2-11b-vision-instruct",
        "@cf/meta/llama-3.2-3b-instruct",
        "@cf/meta/llama-3.2-1b-instruct",
        "@cf/meta/llama-3.1-8b-instruct-fp8",
        "@cf/openai/gpt-oss-120b",
        "@cf/openai/gpt-oss-20b",
        "@cf/moonshotai/kimi-k2.7-code",
        "@cf/moonshotai/kimi-k2.6",
        "@cf/google/gemma-4-26b-a4b-it",
        "@cf/qwen/qwen3-30b-a3b-fp8",
        "@cf/qwen/qwq-32b",
        "@cf/qwen/qwen2.5-coder-32b-instruct",
        "@cf/deepseek-ai/deepseek-r1-distill-qwen-32b",
        "@cf/mistralai/mistral-small-3.1-24b-instruct",
        "@cf/nvidia/nemotron-3-120b-a12b",
        "@cf/ibm-granite/granite-4.0-h-micro",
        "@cf/zai-org/glm-5.2",
        "@cf/zai-org/glm-4.7-flash",
    ),
)
cloudflare.fetch_models = _fetch_cloudflare_models

# Construct base_url from CLOUDFLARE_ACCOUNT_ID if not already set.
if not cloudflare.base_url:
    _account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
    if _account_id:
        cloudflare.base_url = (
            f"https://api.cloudflare.com/client/v4/accounts/{_account_id}/ai/v1"
        )

register_provider(cloudflare)
