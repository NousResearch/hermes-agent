"""mittwald AI Hosting provider profile — EU-hosted, OpenAI-compatible (vLLM)."""

from typing import Any

from agent.reasoning_effort import OPENAI_COMPAT_WIRE_EFFORTS, clamp_effort
from providers import register_provider
from providers.base import ProviderProfile

# ``/v1/models`` lists every hosted model in one array — chat models next to the
# speech, embedding, rerank and OCR endpoints. Only chat models may reach the
# ``/model`` picker, so the non-chat ids are filtered out by substring (the
# catalog grows; a deny-list of families survives new revisions better than an
# allow-list of exact ids).
_NON_CHAT_MARKERS = ("whisper", "-tts-", "embedding", "reranker", "-ocr")


def _is_chat_model(model_id: str) -> bool:
    name = (model_id or "").strip().lower()
    return bool(name) and not any(marker in name for marker in _NON_CHAT_MARKERS)


class MittwaldProfile(ProviderProfile):
    """mittwald AI Hosting — vLLM behind an OpenAI-compatible surface."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **ctx: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Top-level ``reasoning_effort``, clamped to the OpenAI-compat wire.

        Verified live: ``none`` turns Qwen's thinking off (no ``reasoning_content``
        in the reply), so the disabled case does not need vLLM's
        ``chat_template_kwargs.enable_thinking``. An unset effort is omitted so the
        server default applies.
        """
        if not isinstance(reasoning_config, dict):
            return {}, {}
        effort = str(reasoning_config.get("effort") or "").strip().lower()
        if effort == "none" or reasoning_config.get("enabled", True) is False:
            return {}, {"reasoning_effort": "none"}
        if effort:
            return {}, {"reasoning_effort": clamp_effort(effort, OPENAI_COMPAT_WIRE_EFFORTS)}
        return {}, {}

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 8.0
    ) -> list[str] | None:
        """Live catalog, narrowed to the chat models (see :data:`_NON_CHAT_MARKERS`)."""
        models = super().fetch_models(api_key=api_key, base_url=base_url, timeout=timeout)
        if models is None:
            return None
        return [m for m in models if _is_chat_model(m)]

    def default_vision_model(self) -> str | None:
        """Cheapest hosted model that accepts image input."""
        return "Qwen3.6-35B-A3B-FP8"


mittwald = MittwaldProfile(
    name="mittwald",
    aliases=("mittwald-ai", "mittwald-aihosting", "aihosting"),
    display_name="mittwald AI Hosting",
    description="mittwald AI Hosting — EU-hosted OpenAI-compatible inference",
    signup_url="https://developer.mittwald.de/docs/v2/platform/aihosting/access-and-usage/access/",
    # The key comes from mStudio → project → "AI-Hosting"; MITTWALD_LLM_API_KEY is the
    # name mittwald's own docs use. The trailing *_BASE_URL entry is the user override.
    env_vars=("MITTWALD_LLM_API_KEY", "MITTWALD_AI_API_KEY", "MITTWALD_BASE_URL"),
    base_url="https://llm.aihosting.mittwald.de/v1",
    auth_type="api_key",
    supports_vision=True,
    default_aux_model="Qwen3.5-0.8B",
    # Tool-calling chat models only, verified live against /v1/chat/completions.
    fallback_models=(
        "Qwen3.6-35B-A3B-FP8",
        "Qwen3.8-27B-NVFP4",
        "Qwen3.5-122B-A10B-FP8",
        "Ministral-3-14B-Instruct-2512",
        "gpt-oss-120b",
        "Qwen3.5-0.8B",
    ),
)

register_provider(mittwald)
