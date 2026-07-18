"""Cerebras Inference provider profile.

Cerebras' OpenAI-compatible API calls the replayable assistant reasoning field
``reasoning``. Hermes also stores provider-specific ``reasoning_content`` and
``reasoning_details`` fields, which Cerebras rejects as unknown input fields.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class CerebrasProfile(ProviderProfile):
    """Cerebras chat completions with strict reasoning-field replay."""

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prepared = list(messages)
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            cleaned = dict(message)
            if not cleaned.get("reasoning") and isinstance(
                cleaned.get("reasoning_content"), str
            ):
                cleaned["reasoning"] = cleaned["reasoning_content"]
            cleaned.pop("reasoning_content", None)
            cleaned.pop("reasoning_details", None)
            prepared[index] = cleaned
        return prepared

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        model: str | None = None,
        **context: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not isinstance(reasoning_config, dict):
            return {}, {}
        effort = str(reasoning_config.get("effort") or "").strip().lower()
        if effort == "none":
            # Cerebras gpt-oss cannot disable reasoning; omitting the field
            # leaves the documented server default intact. GLM supports none.
            if "glm-4.7" in (model or "").lower():
                return {}, {"reasoning_effort": "none"}
            return {}, {}
        if effort in {"low", "medium", "high"}:
            return {}, {"reasoning_effort": effort}
        if effort in {"xhigh", "max", "ultra"}:
            return {}, {"reasoning_effort": "high"}
        return {}, {}


cerebras = CerebrasProfile(
    name="cerebras",
    display_name="Cerebras",
    description="Cerebras Inference API",
    signup_url="https://cloud.cerebras.ai/",
    env_vars=("CEREBRAS_API_KEY", "CEREBRAS_BASE_URL"),
    base_url="https://api.cerebras.ai/v1",
    fallback_models=("gpt-oss-120b", "zai-glm-4.7", "gemma-4-31b"),
)

register_provider(cerebras)
