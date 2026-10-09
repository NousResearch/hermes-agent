"""AWS Bedrock provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


class BedrockProfile(ProviderProfile):
    """AWS Bedrock — no REST /v1/models endpoint; uses AWS SDK."""

    def resolve_route_policy(self, model: str, base_url: str = "", *, options=None) -> str | None:
        """Select the wire from the runtime builder's auth-aware Bedrock branch."""
        del model, base_url
        route = dict(options or {})
        if route.get("bedrock_openai"):
            return "codex_responses"
        if route.get("bedrock_anthropic"):
            return "anthropic_messages"
        return "bedrock_converse"


bedrock = BedrockProfile(
    name="bedrock", aliases=("aws", "aws-bedrock", "amazon-bedrock", "amazon"),
    display_name="AWS Bedrock", description="AWS Bedrock (Claude, Nova, Llama, DeepSeek; IAM or API key)",
    api_mode="bedrock_converse",
    env_vars=(),  # AWS SDK credentials — not env vars
    base_url="https://bedrock-runtime.us-east-1.amazonaws.com", base_url_env_var="BEDROCK_BASE_URL",
    auth_type="aws_sdk",
    supports_model_listing=False,  # listing goes through the AWS SDK, not a REST call
)

register_provider(bedrock)
