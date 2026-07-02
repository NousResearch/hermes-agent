"""Runware provider profile.

Runware (https://runware.ai) exposes an OpenAI-compatible chat completions
endpoint at https://api.runware.ai/v1, fronting both open-weight models run
on Runware's own Sonic Inference Engine (GLM, DeepSeek, MiniMax, Qwen,
Kimi, ...) and pass-through access to closed-source frontier models
(Anthropic, OpenAI, Google, xAI) under one account and bill. Model IDs are
plain slugs (e.g. "openai-gpt-5-4", "anthropic-claude-sonnet-4-6",
"minimax-m2-7") — no special parsing quirks elsewhere in the codebase.

GET /v1/models returns a bare JSON array (not the usual {"data": [...]}
wrapper) with rich per-model metadata — context_length, max_output_tokens,
and OpenRouter-shaped pricing — which the base ProviderProfile.fetch_models()
and agent.model_metadata.fetch_endpoint_model_metadata() both already parse,
so the model list and context lengths resolve dynamically from the live
catalog instead of a hardcoded table.

Docs: https://runware.ai/docs/platform/openai
"""

import os

from providers import register_provider
from providers.base import ProviderProfile


class RunwareProfile(ProviderProfile):
    """Runware — per-model max_tokens cap, resolved live where possible.

    When ``max_tokens`` is omitted from the request, Runware's OpenAI-compat
    endpoint does not fall back to the model's own completion-token limit —
    it 400s with "'settings.maxTokens' must be an integer between 1 and
    <model's real cap>" (e.g. 384000 for deepseek-v4-flash).

    ``get_max_tokens()`` resolves the cap from Runware's own live
    ``/v1/models`` response (``max_output_tokens``) via the same cached
    endpoint-metadata fetch already used for context-length resolution
    (``agent.model_metadata.fetch_endpoint_model_metadata`` — 5-minute
    in-memory cache, so this doesn't add a network round trip beyond what
    context-length resolution already performs on the same model/turn).

    ``_MODEL_MAX_TOKENS`` is a fallback only: used when no API key is set
    yet (picker/setup time), the live fetch fails, or a model isn't in the
    live response for some reason. Values below are a snapshot of
    Runware's real ``max_output_tokens`` (verified 2026-07-02) and may go
    stale as their catalog changes — the live path is authoritative.
    """

    _MODEL_MAX_TOKENS = {
        "openai-gpt-5-4": 128000,
        "openai-gpt-5-4-mini": 128000,
        "openai-gpt-5-4-nano": 128000,
        "openai-gpt-5-4-pro": 128000,
        "openai-gpt-5-mini": 128000,
        "openai-gpt-5-nano": 128000,
        "openai-gpt-5-5": 128000,
        "anthropic-claude-opus-4-7": 128000,
        "anthropic-claude-opus-4-8": 128000,
        "anthropic-claude-sonnet-4-6": 128000,
        "anthropic-claude-haiku-4-5": 64000,
        "anthropic-claude-fable-5": 128000,
        "google-gemini-3-1-pro": 65536,
        "google-gemini-3-1-flash-lite": 65536,
        "google-gemini-3-flash": 65536,
        "google-gemini-3-5-flash": 65536,
        "google-gemma-4-31b": 65536,
        "deepseek-v4-flash": 384000,
        "deepseek-v4-pro": 384000,
        "minimax-m2-5": 196608,
        "minimax-m2-7": 131072,
        "minimax-m2-7-highspeed": 131072,
        "minimax-m3": 512000,
        "zai-glm-4-7": 131072,
        "zai-glm-5-1": 131072,
        "moonshotai-kimi-k2-6": 49152,
        "xai-grok-4-3": 131072,
        "qwen35_397b_a17b_fp8": 128000,
        "qwen35_27b_fp8": 128000,
    }

    def get_max_tokens(self, model: str | None) -> int | None:
        if model:
            live = self._live_max_tokens(model)
            if live is not None:
                return live
        return self._MODEL_MAX_TOKENS.get(model or "", self.default_max_tokens)

    @staticmethod
    def _live_max_tokens(model: str) -> int | None:
        api_key = os.getenv("RUNWARE_API_KEY", "").strip()
        if not api_key:
            return None
        base_url = os.getenv("RUNWARE_BASE_URL", "").strip() or "https://api.runware.ai/v1"
        try:
            from agent.model_metadata import fetch_endpoint_model_metadata

            metadata = fetch_endpoint_model_metadata(base_url, api_key=api_key)
        except Exception:
            return None
        entry = metadata.get(model)
        if entry and entry.get("max_completion_tokens"):
            return entry["max_completion_tokens"]
        return None


runware = RunwareProfile(
    name="runware",
    aliases=("runware-ai", "runwareai"),
    display_name="Runware",
    description="Runware — open + closed source LLMs on one OpenAI-compatible endpoint",
    signup_url="https://runware.ai/signup",
    env_vars=("RUNWARE_API_KEY", "RUNWARE_BASE_URL"),
    base_url="https://api.runware.ai/v1",
    auth_type="api_key",
    default_aux_model="deepseek-v4-flash",
    fallback_models=(),
)

register_provider(runware)
