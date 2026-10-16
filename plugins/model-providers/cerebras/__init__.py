"""Cerebras Inference provider profile.

Cerebras runs models on their Wafer-Scale Engine (WSE) — a single-chip
SRAM fabric that holds the entire model on-wafer. This eliminates the
off-chip HBM bottleneck that limits GPU autoregressive decoding, delivering
~10-30x faster inference than GPUs on supported models.

Cerebras is OpenAI-compatible at https://api.cerebras.ai/v1, so the
chat-completions transport works with no custom client.

Their catalog is small and curated (only models that compile cleanly to the
WSE dataflow architecture). Production models as of late 2026:

- gpt-oss-120b   (production flagship — best TTFT and output speed)
- llama-3.3-70b
- llama-4-scout
- kimi-k2
- qwen3-32b
- zai-glm-4.7

Use Cerebras for latency-sensitive and high-throughput workloads
(voice agents, interactive chat, batch generation) where supported models
fit. Pair with Fireworks / Together / Baseten for frontier reasoning
models (Kimi K3, DeepSeek V4 Pro) that exceed single-WSE capacity.
"""

from providers import register_provider
from providers.base import ProviderProfile


cerebras = ProviderProfile(
    name="cerebras",
    aliases=("cerebras-ai", "cs", "wse"),
    display_name="Cerebras Inference",
    description="Cerebras — wafer-scale silicon, ~10x faster than GPUs",
    signup_url="https://cloud.cerebras.ai/",
    env_vars=("CEREBRAS_API_KEY",),
    base_url="https://api.cerebras.ai/v1",
    auth_type="api_key",
    # Auxiliary model — fast/cheap chat for side tasks (compression,
    # session search, vision, title generation). gpt-oss-120b is the
    # flagship and has the highest sustained throughput on the WSE.
    default_aux_model="gpt-oss-120b",
    # Curated safety net shown in /model picker when the live catalog
    # fetch fails. Cerebras's catalog is small and stable — these are
    # the production models per inference-docs.cerebras.ai.
    fallback_models=(
        "gpt-oss-120b",
        "llama-3.3-70b",
        "llama-4-scout",
        "kimi-k2",
        "qwen3-32b",
        "zai-glm-4.7",
    ),
)

register_provider(cerebras)
