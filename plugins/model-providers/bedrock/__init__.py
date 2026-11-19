"""AWS Bedrock provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


class BedrockProfile(ProviderProfile):
    """AWS Bedrock — no REST /v1/models endpoint; uses AWS SDK."""


bedrock = BedrockProfile(
    name="bedrock", aliases=("aws", "aws-bedrock", "amazon-bedrock", "amazon"), api_mode="bedrock_converse",
    env_vars=(),  # AWS SDK credentials — not env vars
    base_url="https://bedrock-runtime.us-east-1.amazonaws.com", auth_type="aws_sdk",
    supports_model_listing=False,  # listing goes through the AWS SDK, not a REST call
)

register_provider(bedrock)
