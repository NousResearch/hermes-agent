"""RainDragon provider profile: OpenAI-compatible Gemma 4 serving.

Thinking is a binary toggle read from ``chat_template_kwargs.enable_thinking``
(the Gemma 4 chat template; server default off). ``/reasoning none`` sends it
off, any enabled effort sends it on, and an unset reasoning_config leaves the
server default alone. There are no graded effort levels to map onto.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class RainDragonProfile(ProviderProfile):
    """RainDragon — thinking on/off via ``chat_template_kwargs`` in extra_body."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not reasoning_config or not isinstance(reasoning_config, dict):
            return {}, {}
        enabled = reasoning_config.get("enabled")
        if enabled is None:
            effort = (reasoning_config.get("effort") or "").strip().lower()
            if not effort:
                return {}, {}
            enabled = effort not in ("none", "minimal")
        return {"chat_template_kwargs": {"enable_thinking": bool(enabled)}}, {}


raindragon = RainDragonProfile(
    name="raindragon", aliases=("raindragon-ai",), display_name="RainDragon",
    description="RainDragon (Gemma 4, OpenAI-compatible)",
    signup_url="https://inference.raindragon.ai/auth", env_vars=("RAINDRAGON_API_KEY", "RAINDRAGON_BASE_URL"),
    base_url="https://inference.raindragon.ai/v1", auth_type="api_key",
    # No default_aux_model: auxiliary tasks use the main model. [0] is the setup default.
    fallback_models=("google/gemma-4-26B-A4B-it", "google/gemma-4-31B-it"),
)

register_provider(raindragon)
