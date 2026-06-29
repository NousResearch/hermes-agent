"""ClinePass provider profile.

ClinePass is a subscription that serves curated open-weight coding models
(GLM, Kimi, DeepSeek, MiniMax, MiMo, Qwen) behind an OpenAI-compatible Chat
Completions API at ``https://api.cline.bot/api/v1``, so the default
``openai_chat`` transport carries tool calling and streaming with no special
handling. Model IDs are namespaced (``cline-pass/<model>``) and pass through to
the endpoint unchanged.

Authentication is a bearer ``CLINE_API_KEY`` created from the ClinePass account
dashboard. The curated ``fallback_models`` list mirrors the published catalog;
the live set is fetched from the endpoint when a key is present.
"""

from providers import register_provider
from providers.base import ProviderProfile


clinepass = ProviderProfile(
    name="clinepass",
    aliases=("cline-pass", "cline"),
    env_vars=("CLINE_API_KEY", "CLINE_BASE_URL"),
    display_name="ClinePass",
    description="ClinePass: curated open-weight coding models (GLM, Kimi, DeepSeek, MiniMax, MiMo, Qwen)",
    signup_url="https://cline.bot/cline-pass",
    base_url="https://api.cline.bot/api/v1",
    auth_type="api_key",
    default_aux_model="cline-pass/deepseek-v4-flash",
    fallback_models=(
        "cline-pass/glm-5.2",
        "cline-pass/kimi-k2.7-code",
        "cline-pass/kimi-k2.6",
        "cline-pass/deepseek-v4-pro",
        "cline-pass/deepseek-v4-flash",
        "cline-pass/mimo-v2.5-pro",
        "cline-pass/mimo-v2.5",
        "cline-pass/minimax-m3",
        "cline-pass/qwen3.7-max",
        "cline-pass/qwen3.7-plus",
    ),
)

register_provider(clinepass)
