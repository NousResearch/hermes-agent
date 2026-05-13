"""Xiaomi MiMo provider profile."""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class XiaomiProfile(ProviderProfile):
    """Xiaomi MiMo OpenAI-compatible provider quirks."""

    def _thinking_disabled(self, reasoning_config: dict | None) -> bool:
        return isinstance(reasoning_config, dict) and reasoning_config.get("enabled") is False

    def prepare_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize MiMo assistant tool-call turns.

        MiMo rejects assistant tool-call messages with an empty-string content
        field (``content: ""``) after tool execution with the misleading 400:
        "reasoning_content in the thinking mode must be passed back". OpenAI's
        tool-call shape permits omitting content entirely when tool_calls are
        present, and that is the safest representation for MiMo.
        """
        changed = False
        normalized: list[dict[str, Any]] = []
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and msg.get("tool_calls")
                and msg.get("content") == ""
            ):
                msg = dict(msg)
                msg.pop("content", None)
                changed = True
            normalized.append(msg)
        return normalized if changed else messages

    def build_extra_body(
        self, *, reasoning_config: dict | None = None, **context: Any
    ) -> dict[str, Any]:
        """Emit MiMo's explicit thinking switch.

        MiMo defaults several models (mimo-v2.5/pro, mimo-v2-pro/omni) to
        thinking mode. If we only omit reasoning controls when Hermes has
        `agent.reasoning_effort: none`, MiMo still thinks internally and then
        rejects multi-turn/tool-call replays unless every assistant message
        echoes `reasoning_content`.

        MiMo's OpenAI-compatible docs expose `thinking: enabled|disabled`, so
        send `disabled` explicitly when Hermes reasoning is disabled.
        """
        if self._thinking_disabled(reasoning_config):
            return {"thinking": "disabled"}
        return {}


xiaomi = XiaomiProfile(
    name="xiaomi",
    aliases=("mimo", "xiaomi-mimo"),
    env_vars=("XIAOMI_API_KEY",),
    base_url="https://api.xiaomimimo.com/v1",
    supports_health_check=False,  # /v1/models returns 401 even with valid key
)

register_provider(xiaomi)
