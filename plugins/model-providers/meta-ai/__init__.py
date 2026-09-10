"""Meta Model API (Muse Spark) provider profile — https://api.meta.ai/v1.

Bundled from albertodepaola/hermes-meta-provider; rides entirely on
ProviderProfile hooks (zero core edits). The reasoning dial is emitted as a
top-level ``reasoning_effort`` kwarg — not ``extra_body.reasoning``, whose
emission is gated by a core host allowlist a third-party plugin must not edit.
"""

import os
from typing import Any

from agent.reasoning_effort import META_AI_EFFORTS, clamp_effort
from providers import register_provider
from providers.base import ProviderProfile


class MetaAIProfile(ProviderProfile):
    """Meta Model API — top-level reasoning_effort, self-contained."""

    # Non-chat model prefixes excluded from the agent picker.  The live
    # /v1/models catalog includes image-generation and transcription models
    # that are not suitable for agentic chat.
    _NON_CHAT_PREFIXES = ("muse-image-", "muse-voice-")

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Fetch and filter the live catalog, excluding non-chat models."""
        live = super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
        if live is None:
            return None
        return [
            m for m in live
            if not any(m.startswith(p) for p in self._NON_CHAT_PREFIXES)
        ]

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, supports_reasoning: bool = False, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Ignores the core ``supports_reasoning`` gate (host-allowlist driven); Muse Spark always
        accepts ``reasoning_effort``. Muse 400s on ``none``: disabled/"none" -> ``minimal``
        (closest to off); unset/bespoke levels -> ``medium``."""
        rc = reasoning_config or {}
        effort = str(rc.get("effort") or "").strip().lower()
        if rc.get("enabled") is False or effort == "none":
            mapped = "minimal"
        else:
            clamped = clamp_effort(effort, META_AI_EFFORTS)
            mapped = clamped if clamped in META_AI_EFFORTS else "medium"
        return {}, {"reasoning_effort": mapped}


meta_ai = MetaAIProfile(
    name="meta-ai", aliases=("meta", "muse", "muse-spark", "model-api", "msl"), display_name="Meta Model API",
    description="Meta Muse Spark family (Meta Superintelligence Labs)",
    signup_url="https://developer.meta.com/ai/",
    # MODEL_API_KEY is Meta's documented env var; the aliases are conveniences.
    env_vars=("MODEL_API_KEY", "META_API_KEY", "META_MODEL_API_KEY", "META_BASE_URL"),
    base_url=os.getenv("META_BASE_URL", "").strip() or "https://api.meta.ai/v1", auth_type="api_key",
    # Responses API engages Muse prompt caching (0 cached tokens on chat/completions vs
    # 93-99% hits on /v1/responses); the hook above still covers custom non-api.meta.ai base URLs.
    api_mode="codex_responses",
    # Natively multimodal, but only on user turns: an image envelope inside a role:tool
    # message 400s "content did not match any supported type".
    supports_vision=True, supports_vision_tool_messages=False,
    # See #101668.
    default_aux_model="muse-spark-1.2-contributor",
    # Muse spends completion budget on hidden reasoning first; low caps can finish with empty content.
    default_max_tokens=16384,
    # Single safety-net entry, shown only when the live /v1/models fetch fails.
    fallback_models=("muse-spark-1.2",),
)

register_provider(meta_ai)
