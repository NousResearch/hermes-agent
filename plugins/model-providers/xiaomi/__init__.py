"""Xiaomi MiMo provider profile.

MiMo supports ``thinking`` via ``extra_body`` (``{"type": "enabled"}`` /
``{"type": "disabled"}``) in the OpenAI-compatible chat completions
endpoint.

When no ``reasoning_config`` is provided, thinking defaults to ``enabled``
(MiMo server default).  When the user explicitly disables reasoning
(e.g. ``reasoning_effort: none``), the ``thinking`` parameter is set to
``disabled`` to save tokens.

Refs #27325.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class XiaomiProfile(ProviderProfile):
    """Xiaomi MiMo — extra_body.thinking for token-aware thinking control."""

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """MiMo uses extra_body.thinking to control thinking mode.

        Enabled by default (server default); emits ``disabled`` when the
        user has explicitly turned reasoning off.
        """
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}

        if not reasoning_config or not isinstance(reasoning_config, dict):
            # No config → thinking enabled (default)
            extra_body["thinking"] = {"type": "enabled"}
            return extra_body, top_level

        enabled = reasoning_config.get("enabled", True)
        if enabled is False:
            extra_body["thinking"] = {"type": "disabled"}
            return extra_body, top_level

        extra_body["thinking"] = {"type": "enabled"}
        return extra_body, top_level


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
