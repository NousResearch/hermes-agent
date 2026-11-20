"""IO Intelligence (io.net) provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


ionet = ProviderProfile(
    name="io-net", aliases=("ionet", "io-intelligence", "io_net"), display_name="IO Intelligence",
    description="IO Intelligence — io.net's OpenAI-compatible API for open-weight models",
    signup_url="https://io.net/docs/guides/intelligence/api-keys-and-secrets",
    env_vars=("IONET_API_KEY", "IOINTELLIGENCE_API_KEY", "IONET_BASE_URL"),
    base_url="https://api.intelligence.io.solutions/api/v1", auth_type="api_key",
    default_aux_model="zai-org/GLM-5.3-Flash",
    # The live /models endpoint (keyless-readable) is the catalog source of truth: with a key the
    # picker fetches it live; these ids are the offline fallback. models.dev's io-net list is
    # stale, so the models.dev catalog merge is deliberately not used for this provider.
    fallback_models=(
        "deepseek-ai/DeepSeek-V4.1-Flash", "zai-org/GLM-5.3", "moonshotai/Kimi-K2.7-Code",
        "Qwen/Qwen3-Next-80B-A3B-Instruct", "deepseek-ai/DeepSeek-R1-0528", "meta-llama/Llama-3.3-70B-Instruct",
    ),
)

register_provider(ionet)
