"""NVIDIA NIM provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class NvidiaProfile(ProviderProfile):
    """NVIDIA hosted NIM.

    Nemotron 3 Ultra exposes thinking controls through request-body fields
    specific to its chat template, not through OpenRouter-style
    ``extra_body.reasoning`` or OpenAI-style ``reasoning_effort``.
    """

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        model_l = (model or "").strip().lower()
        if "nemotron-3-ultra" not in model_l:
            return {}, {}

        enabled = True
        effort = "medium"
        if reasoning_config and isinstance(reasoning_config, dict):
            if reasoning_config.get("enabled") is False:
                enabled = False
            raw_effort = str(reasoning_config.get("effort") or "").strip().lower()
            if raw_effort:
                effort = raw_effort

        if not enabled or effort == "none":
            return {"chat_template_kwargs": {"enable_thinking": False}}, {}

        chat_template_kwargs: dict[str, Any] = {"enable_thinking": True}
        if effort in {"minimal", "low", "medium"}:
            # NVIDIA's Build page exposes a binary high-vs-medium switch for
            # Ultra: medium uses this flag; high/default omit it.
            chat_template_kwargs["medium_effort"] = True

        return {"chat_template_kwargs": chat_template_kwargs}, {}


nvidia = NvidiaProfile(
    name="nvidia",
    aliases=("nvidia-nim",),
    env_vars=("NVIDIA_API_KEY",),
    display_name="NVIDIA NIM",
    description="NVIDIA NIM — accelerated inference",
    signup_url="https://build.nvidia.com/",
    fallback_models=(
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia/llama-3.3-70b-instruct",
    ),
    base_url="https://integrate.api.nvidia.com/v1",
    default_max_tokens=16384,
)

register_provider(nvidia)
