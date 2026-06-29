"""Cerebras provider profile.

Cerebras Inference is an OpenAI-compatible cloud endpoint backed by their
wafer-scale hardware (very high tokens/sec). Auth is a ``csk-…`` bearer key
on ``https://api.cerebras.ai/v1``; the standard ``/v1/models`` and
``/v1/chat/completions`` routes apply, so the default ``chat_completions``
transport handles it with no per-provider quirks.

Model entitlement is per-account — ``GET /v1/models`` returns only the ids the
key may use — so the live discovery path drives the picker; ``fallback_models``
below is just the offline catalog (mirrors models.dev's Cerebras entry).
"""

from __future__ import annotations

from providers import register_provider
from providers.base import ProviderProfile

cerebras = ProviderProfile(
    name="cerebras",
    env_vars=("CEREBRAS_API_KEY", "CEREBRAS_BASE_URL"),
    base_url="https://api.cerebras.ai/v1",
    display_name="Cerebras",
    description="Cerebras — wafer-scale OpenAI-compatible inference",
    signup_url="https://cloud.cerebras.ai/",
    fallback_models=(
        "gpt-oss-120b",
        "zai-glm-4.7",
    ),
    default_aux_model="gpt-oss-120b",
)

register_provider(cerebras)
