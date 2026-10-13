"""TokenRhythm provider profile (OpenAI-compatible Chat Completions; no Responses API)."""

from providers import register_provider
from providers.base import ProviderProfile

# Chat models online in TokenRhythm's public catalog (https://tokenrhythm.studio/api/models):
# id -> (context window, accepts images). Every one calls tools and reasons. The keyed
# /v1/models listing appends anything newer at pick time; this table is the offline picker and
# the capability source, since TokenRhythm is not in models.dev. Default model leads, flash next.
_MODELS: dict[str, tuple[int, bool]] = {
    "deepseek-v4-pro-0813": (1_000_000, False),
    "deepseek-v4-flash-0731": (1_000_000, False),
    "deepseek-flash": (1_000_000, True),
    "glm-5.3": (1_048_576, False),
    "glm-5.3-flash": (1_048_576, True),
    "glm-5.2": (1_000_000, False),
    "glm-5.1": (200_000, False),
    "kimi-k2.7-code": (256_000, True),
    "kimi-k2.6": (256_000, True),
    "qwen3.8-max": (1_000_000, True),
    "qwen3.8-flash": (1_000_000, True),
    "qwen3.8-27b": (1_000_000, True),
    "qwen3.7-max": (1_000_000, False),
    "qwen3.7-flash": (1_000_000, True),
    "minimax-m2.7": (200_000, False),
    "mimo-v2.5-pro": (256_000, False),
    "seed-2.1-pro": (262_144, True),
    "seed-2.1-turbo": (262_144, True),
    "longcat-2.0": (1_000_000, False),
}

tokenrhythm = ProviderProfile(
    name="tokenrhythm", aliases=("token-rhythm",), display_name="TokenRhythm",
    description="TokenRhythm — OpenAI-compatible gateway to DeepSeek, GLM, Kimi, Qwen, MiniMax and more",
    signup_url="https://tokenrhythm.studio/account/keys",
    env_vars=("TOKENRHYTHM_API_KEY", "TOKENRHYTHM_BASE_URL"),
    base_url="https://tokenrhythm.studio/v1", auth_type="api_key",
    default_aux_model="deepseek-v4-flash-0731",
    fallback_models=tuple(_MODELS),
    model_capabilities={
        model: {"context_window": context, "supports_tools": True, "supports_reasoning": True,
                "supports_vision": vision}
        for model, (context, vision) in _MODELS.items()
    },
)

register_provider(tokenrhythm)
