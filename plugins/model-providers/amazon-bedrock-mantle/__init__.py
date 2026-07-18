"""Amazon Bedrock Mantle provider profile.

OpenAI-compatible chat for Mantle-hosted OSS models, plus Claude via the
Mantle Anthropic Messages route. Auth uses ``AWS_BEARER_TOKEN_BEDROCK`` or
IAM-minted short-lived bearers (see ``agent.bedrock_mantle``).

Runtime resolution is special-cased in ``hermes_cli.runtime_provider`` —
this profile supplies identity, defaults, and catalog fallbacks.
"""

from __future__ import annotations

from providers import register_provider
from providers.base import ProviderProfile


class AmazonBedrockMantleProfile(ProviderProfile):
    """Bedrock Mantle — region-scoped OpenAI-compatible /v1 surface."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        from agent.bedrock_mantle import (
            discover_mantle_models,
            resolve_mantle_bearer_token,
            resolve_mantle_region,
            mantle_openai_base_url,
        )

        region = resolve_mantle_region()
        # Prefer region encoded in base_url if caller passed one
        if base_url and "bedrock-mantle." in base_url:
            try:
                host = base_url.split("://", 1)[-1].split("/", 1)[0]
                # bedrock-mantle.<region>.api.aws
                parts = host.split(".")
                if len(parts) >= 4 and parts[0] == "bedrock-mantle":
                    region = parts[1]
            except Exception:
                pass

        token = (api_key or "").strip() or resolve_mantle_bearer_token(region)
        if not token:
            return list(self.fallback_models) or None
        return discover_mantle_models(region, token, timeout=timeout)


amazon_bedrock_mantle = AmazonBedrockMantleProfile(
    name="amazon-bedrock-mantle",
    aliases=(
        "bedrock-mantle",
        "mantle",
        "aws-bedrock-mantle",
        "amazon-mantle",
    ),
    display_name="Amazon Bedrock Mantle",
    description=(
        "Amazon Bedrock Mantle (OpenAI-compatible OSS models + Claude Messages route)"
    ),
    signup_url="https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html",
    # Explicit bearer; region via AWS_REGION / BEDROCK_MANTLE_REGION.
    # IAM mint is handled at runtime when the bearer env is empty.
    env_vars=("AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_MANTLE_BASE_URL"),
    base_url="https://bedrock-mantle.us-east-1.api.aws/v1",
    auth_type="api_key",
    supports_vision=True,
    supports_health_check=True,
    hostname="bedrock-mantle.us-east-1.api.aws",
    default_aux_model="gpt-oss-120b",
    fallback_models=(
        "gpt-oss-120b",
        "qwen3-coder-480b-a35b",
        "deepseek-v3.2",
        "kimi-k2.5",
        "glm-4.7",
        "anthropic.claude-sonnet-5",
        "anthropic.claude-opus-4-7",
    ),
)

register_provider(amazon_bedrock_mantle)
