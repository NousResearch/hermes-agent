"""CoreWeave Serverless Inference provider profile.

CoreWeave Serverless Inference (formerly W&B Inference) is a standard
OpenAI-compatible endpoint serving leading open models on CoreWeave. API-key
auth via ``COREWEAVE_API_KEY``; the ``/v1/models`` catalog is live so the picker
fetches models automatically.

The endpoint also accepts an optional ``openai-project`` header (``team/project``)
for usage attribution. It is only required for accounts whose default project
lacks Inference access. Because the value is per-user we do not ship a static
``default_headers`` for it; users that need it supply it via
``model.default_headers`` in ``config.yaml`` (see the provider docs), which
``_apply_user_default_headers`` merges onto both the main and auxiliary clients.
"""

from providers import register_provider
from providers.base import ProviderProfile

coreweave = ProviderProfile(
    name="coreweave",
    aliases=("coreweave-inference", "coreweave-serverless"),
    display_name="CoreWeave Serverless Inference",
    description="CoreWeave Serverless Inference — open models on CoreWeave",
    signup_url="https://wandb.ai/settings",  # where Inference API keys are issued
    # env_vars: 1st = API key, 2nd = optional base-url override
    env_vars=("COREWEAVE_API_KEY", "COREWEAVE_BASE_URL"),
    base_url="https://api.inference.wandb.ai/v1",
    auth_type="api_key",
    default_aux_model="meta-llama/Llama-3.1-8B-Instruct",
    fallback_models=(
        # Safety net only — the picker fetches the live /v1/models catalog.
        # Only tool-calling / agentic models belong here.
        "deepseek-ai/DeepSeek-V3.1",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "moonshotai/Kimi-K2-Instruct",
        "zai-org/GLM-4.5",
        "openai/gpt-oss-120b",
        "meta-llama/Llama-3.1-8B-Instruct",
    ),
)

register_provider(coreweave)
