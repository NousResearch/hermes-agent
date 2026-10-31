"""Venice.ai provider profile.

Registers Venice (https://api.venice.ai) as a first-class API-key model
provider so it appears in the CLI / TUI / dashboard / WebUI pickers with a
LIVE model catalog instead of degrading to a bare custom endpoint with no
model list.

Pipeline (all generic — no per-provider special-casing anywhere):

- ``providers.list_providers()`` discovers this bundled plugin under
  ``plugins/model-providers/`` and ``register_provider()`` makes the profile
  available to ``providers.get_provider_profile("venice")``.
- The auth registry in ``hermes_cli.auth`` auto-extends with any api-key
  profile that declares ``env_vars``, so ``resolve_api_key_provider_credentials()``
  resolves VENICE_API_KEY and the profile's base_url.
- ``hermes_cli.models.provider_model_ids("venice")`` then takes the generic
  profile-based live path: ``fetch_models()`` GETs
  ``https://api.venice.ai/api/v1/models`` and merges the result with these
  curated ``fallback_models`` (curated-first), so brand-new Venice models
  appear automatically and the picker never empties when the endpoint is
  unreachable.

``auth_type="api_key"`` keeps the profile out of the OAuth/special provider
paths; ``api_mode="chat_completions"`` matches Venice's OpenAI-compatible
API.
"""

from providers import register_provider
from providers.base import ProviderProfile

# Curated snapshot of https://api.venice.ai/api/v1/models (2026-08-13).
# Ordered to lead the picker with the agentic favorites; the live merge in
# provider_model_ids() appends any NEW model Venice adds after this list, so
# this snapshot only needs refreshing for reorderings, not for new models.
FALLBACK_MODELS: tuple[str, ...] = (
    "gemini-3-6-flash",
    "gemini-3-5-flash-lite",
    "zai-org-glm-5-2",
    "zai-org-glm-5-1",
    "zai-org-glm-5",
    "z-ai-glm-5-turbo",
    "z-ai-glm-5v-turbo",
    "olafangensan-glm-4.7-flash-heretic",
    "zai-org-glm-4.7-flash",
    "zai-org-glm-4.6",
    "zai-org-glm-4.7",
    "venice-uncensored-1-2",
    "venice-uncensored-role-play",
    "qwen-3-8-2-4t-a95b",
    "qwen-3-8-max",
    "qwen-3-7-max",
    "qwen-3-7-plus",
    "qwen-3-6-plus",
    "qwen3-6-27b",
    "qwen3-6-35b-a3b",
    "qwen3-5-9b",
    "qwen3-5-397b-a17b",
    "qwen3-5-35b-a3b",
    "qwen3-235b-a22b-thinking-2507",
    "qwen3-235b-a22b-instruct-2507",
    "qwen3-next-80b",
    "qwen3-vl-235b-a22b",
    "qwen3-coder-480b-a35b-instruct-turbo",
    "google-gemma-4-26b-a4b-it",
    "google-gemma-4-31b-it",
    "gemma-4-uncensored",
    "google-gemma-3-27b-it",
    "grok-4-3",
    "grok-4-5",
    "grok-4-6",
    "grok-4-20",
    "grok-4-20-multi-agent",
    "grok-build-0-1",
    "mistral-small-3-2-24b-instruct",
    "mistral-small-2603",
    "hermes-3-llama-3.1-405b",
    "gemini-3-1-pro-preview",
    "gemini-3-5-flash",
    "gemini-3-flash-preview",
    "claude-fable-5",
    "claude-opus-5",
    "claude-opus-5-fast",
    "claude-opus-4-8",
    "claude-opus-4-8-fast",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "openai-gpt-oss-120b",
    "kimi-k2-6",
    "kimi-k2-7-code",
    "kimi-k2-5",
    "kimi-k3",
    "inkling",
    "xiaomi-mimo-v2-5",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-v4-flash-0731",
    "deepseek-v3.2",
    "seed-2-1-turbo",
    "kimi-k3-fast-api",
    "deepseek-v4-flash-0731-fast",
    "aion-labs-aion-3-0",
    "aion-labs-aion-3-0-mini",
    "llama-3.2-3b",
    "llama-3.3-70b",
    "openai-gpt-52",
    "openai-gpt-52-codex",
    "openai-gpt-53-codex",
    "openai-gpt-54",
    "openai-gpt-54-pro",
    "openai-gpt-54-mini",
    "openai-gpt-55",
    "openai-gpt-55-pro",
    "openai-gpt-56-luna",
    "openai-gpt-56-luna-pro",
    "openai-gpt-56-terra",
    "openai-gpt-56-terra-pro",
    "openai-gpt-56-sol",
    "openai-gpt-56-sol-pro",
    "openai-gpt-4o-2024-11-20",
    "openai-gpt-4o-mini-2024-07-18",
    "minimax-m3-preview",
    "minimax-m25",
    "minimax-m27",
    "mercury-2",
    "nvidia-nemotron-3-nano-30b-a3b",
    "nvidia-nemotron-3-5-lightning-30b-a3b",
    "nvidia-nemotron-3-ultra-550b-a55b",
    "e2ee-gemma-3-27b-p",
    "e2ee-gemma-4-26b-a4b-uncensored-p",
    "e2ee-glm-5-2-p",
    "e2ee-gpt-oss-20b-p",
    "e2ee-gpt-oss-120b-p",
    "e2ee-qwen-2-5-7b-p",
    "e2ee-qwen3-6-35b-a3b-uncensored-p",
    "e2ee-qwen3-vl-30b-a3b-p",
    "e2ee-glm-5-1",
    "e2ee-qwen3-6-35b-a3b",
    "e2ee-qwen3-6-27b",
    "e2ee-gemma-4-31b",
    "e2ee-deepseek-v4-flash",
)


venice_profile = ProviderProfile(
    name="venice",
    display_name="Venice",
    description="Venice.ai — private, uncensored multi-model API",
    signup_url="https://venice.ai/",
    env_vars=("VENICE_API_KEY",),
    base_url="https://api.venice.ai/api/v1",
    auth_type="api_key",
    api_mode="chat_completions",
    fallback_models=FALLBACK_MODELS,
)

register_provider(venice_profile)
