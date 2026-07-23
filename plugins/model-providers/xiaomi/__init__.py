"""Xiaomi MiMo provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class XiaomiProfile(ProviderProfile):
    """Xiaomi MiMo — disable thinking via ``extra_body.thinking`` when reasoning is off.

    MiMo (vLLM-served, OpenAI-compatible at ``api.xiaomimimo.com/v1``) reasons by
    default. Sending ``extra_body={"thinking": {"type": "disabled"}}`` makes it
    skip the reasoning pass (``reasoning_tokens`` -> 0); verified against
    ``api.xiaomimimo.com``. MiMo rejects a top-level ``reasoning_effort``
    (HTTP 400), so it has no effort granularity: any enabled level leaves the
    server default (thinking on) untouched. Mirrors the disable path used by the
    opencode-zen / kimi profiles.
    """

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        # reasoning_config is {"enabled": False} for "/reasoning none"; only then
        # do we override. No config / enabled → leave MiMo's server default.
        if (
            isinstance(reasoning_config, dict)
            and reasoning_config.get("enabled") is False
        ):
            extra_body["thinking"] = {"type": "disabled"}
        return extra_body, {}


xiaomi = XiaomiProfile(
    name="xiaomi",
    aliases=("mimo", "xiaomi-mimo"),
    env_vars=("XIAOMI_API_KEY",),
    base_url="https://api.xiaomimimo.com/v1",
    supports_health_check=False,  # /v1/models returns 401 even with valid key
    supports_vision=True,  # mimo-v2-omni is vision-capable
    supports_vision_tool_messages=False,  # rejects list-type tool content (400 "text is not set")
)

register_provider(xiaomi)
