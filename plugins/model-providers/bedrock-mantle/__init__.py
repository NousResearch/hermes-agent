"""AWS Bedrock Mantle provider profile.

Mantle is Bedrock's OpenAI-compatible surface
(``https://bedrock-mantle.<region>.api.aws``), distinct from the native
``bedrock`` Converse provider. It serves two API families on one endpoint:

- GPT-5.5 / GPT-5.4 via the Responses API (``/openai/v1/responses``)
- ~40 open models (gpt-oss, qwen, mistral, deepseek, …) via Chat Completions
  (``/v1/chat/completions``)

Auth is AWS SigV4 (service ``bedrock``) over the standard botocore credential
chain — no API key. The SigV4 signing + Responses-route rewrite live in
``agent/bedrock_sigv4_adapter.py`` and are attached to the OpenAI client in
``agent/agent_runtime_helpers.create_openai_client``.
"""

from providers import register_provider
from providers.base import ProviderProfile


class BedrockMantleProfile(ProviderProfile):
    """Bedrock Mantle — OpenAI-compatible, SigV4-signed, no REST bearer key.

    Model listing is handled by the live-discovery branch in
    ``hermes_cli/models.py`` (SigV4 ``GET /v1/models``) with the
    ``fallback_models`` below as the offline fallback, so ``fetch_models``
    returns None (the generic profile fetch is api_key-only and never runs for
    aws_sdk providers anyway).
    """

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        return None


def _default_base_url() -> str:
    try:
        from agent.bedrock_sigv4_adapter import mantle_base_url

        return mantle_base_url()
    except Exception:
        return "https://bedrock-mantle.us-east-2.api.aws/v1"


bedrock_mantle = BedrockMantleProfile(
    name="bedrock-mantle",
    aliases=("mantle", "aws-mantle", "bedrock-openai", "amazon-bedrock-mantle"),
    api_mode="chat_completions",  # per-model upgrade to codex_responses for gpt-5.x
    env_vars=(),  # AWS SDK credentials — not env vars
    base_url=_default_base_url(),
    auth_type="aws_sdk",
    fallback_models=(
        # Claude (us-east-1)
        "anthropic.claude-opus-4-8",
        "anthropic.claude-fable-5",
        "anthropic.claude-haiku-4-5",
        "anthropic.claude-opus-4-7",
        # GPT (us-east-2)
        "openai.gpt-5.5",
        "openai.gpt-5.4",
        "openai.gpt-oss-120b",
        "openai.gpt-oss-20b",
    ),
)

register_provider(bedrock_mantle)
