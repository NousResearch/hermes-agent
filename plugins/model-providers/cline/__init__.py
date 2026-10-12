"""Cline provider profile.

Cline API (https://api.cline.bot) — an OpenAI-compatible Chat Completions
gateway that fronts Anthropic, OpenAI, Google and the ClinePass open-model
catalog behind a single Bearer-authenticated endpoint.

Auth is a plain API key: ``Authorization: Bearer <CLINE_API_KEY>`` (the standard
OpenAI-compatible ``auth_type="api_key"`` transport), with the key read from the
``CLINE_API_KEY`` env var. Create the key at app.cline.bot → Settings → API Keys.

Model IDs use the OpenRouter ``provider/model`` convention. ClinePass subscription
models live in the ``cline-pass/`` namespace (``cline-pass/deepseek-v4-pro``,
``cline-pass/qwen3.8-max``, …); usage-billed models use the upstream prefix
(``anthropic/claude-sonnet-4-6``, ``openai/gpt-4o``, …).
"""

from providers import register_provider
from providers.base import ProviderProfile


cline = ProviderProfile(
    name="cline",
    aliases=("cline-pass", "cline.bot", "cline-api"),
    display_name="Cline",
    description="Cline — unified API for Anthropic, OpenAI, Google + ClinePass models",
    signup_url="https://app.cline.bot/settings/api-keys",
    env_vars=("CLINE_API_KEY",),
    base_url="https://api.cline.bot/api/v1",
    auth_type="api_key",
    # Cheap, universally available model for auxiliary calls (titles,
    # compression, vision) so they do not eat the user's paid usage. A
    # ``cline-pass/*`` aux model is deliberately avoided: those 500 without an
    # active ClinePass subscription, while the standard catalog works for any key.
    default_aux_model="anthropic/claude-haiku-5.5",
    fallback_models=(
        # Standard Cline API models (usage-billed, available to any key).
        "anthropic/claude-sonnet-5.5",
        "anthropic/claude-haiku-5.5",
        "openai/gpt-6.1-sol",
        "google/gemini-2.5-pro",
        # ClinePass subscription models (require an active ClinePass plan).
        "cline-pass/deepseek-v4-pro",
        "cline-pass/qwen3.8-max",
        "cline-pass/qwen3.7-max",
        "cline-pass/glm-5.3",
        "cline-pass/kimi-k3",
        "cline-pass/mimo-v2.5-pro",
        "cline-pass/deepseek-v4.1-flash",
    ),
)

register_provider(cline)
